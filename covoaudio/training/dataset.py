"""
Streaming Dataset for Stage 1 Adapter Training
================================================
Sources (all streamed — nothing downloaded upfront):

  Primary:
    openai/librispeech_asr  clean  train.100   (~100h)
    openai/librispeech_asr  clean  train.360   (~360h)
    openai/librispeech_asr  other  train.500   (~500h)
    ──────────────────────────────────────────────────
    Total primary:                              ~960h

  Supplementary (enabled via --extra_data flag):
    mozilla-foundation/common_voice_13_0  en   (~2,400h validated)
    facebook/voxpopuli                    en   (~543h)

  Validation (downloaded fully — only ~5h, fast):
    openai/librispeech_asr  clean  validation

Audio format from HuggingFace LibriSpeech:
  array:        np.float32,  shape (N,)
  sampling_rate: 16000 Hz

Our model expects 24kHz input (audio_encoder resamples 24k→16k internally).
We resample 16k→24k here. The round-trip 16→24→16 is lossless for Whisper
because Whisper only sees up to 8kHz bandwidth anyway.
"""

import torch
import numpy as np
import torchaudio
import torchaudio.functional as AF
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets


# ── Token sequence length calculator (mirrors modeling_covo_audio.py) ─────────

def calc_seq_len(seq_len: int) -> int:
    for s in [2, 2, 2, 2]:
        seq_len = (seq_len + s - 1) // s
    return seq_len


# ── Waveform preprocessing ────────────────────────────────────────────────────

def preprocess_wav(array: np.ndarray, src_sr: int, target_sr: int = 24000,
                   max_duration: float = 30.0) -> torch.Tensor:
    """
    np.ndarray (float32) → torch.Tensor at target_sr, clipped and aligned.
    """
    wav = torch.from_numpy(array.copy()).float()

    # Resample to target (24kHz for model compatibility)
    if src_sr != target_sr:
        wav = AF.resample(wav, orig_freq=src_sr, new_freq=target_sr)

    # Clip to max duration
    max_samples = int(max_duration * target_sr)
    wav = wav[:max_samples]

    # Align to hop boundary
    hop = target_sr // 100           # 240 samples @ 24kHz
    wav = wav[: len(wav) // hop * hop]

    # Pad to multiple of 480 (model requirement)
    rem = wav.shape[0] % 480
    if rem:
        wav = F.pad(wav, (0, 480 - rem))

    return wav


# ── Sample builder (shared by streaming and map-style) ───────────────────────

def build_sample(wav: torch.Tensor, transcript: str, tokenizer,
                 audio_token_index: int) -> dict:
    """
    Given a preprocessed waveform and its transcript, build the tokenized
    training sample (input_ids, labels, attention_mask).

    Prompt format (Gemma 3 chat template):
        <bos><start_of_turn>user
        <|begofcAUDIO|><|cAUDIO|>×N<|endofcAUDIO|><end_of_turn>
        <start_of_turn>model
        {transcript}<eos>

    Labels: -100 for prompt tokens, transcript token ids for response.
    """
    # Number of audio placeholder tokens
    duration_frames  = len(wav) * 100 // 24000
    num_audio_tokens = calc_seq_len(duration_frames)

    audio_placeholder = (
        "<|begofcAUDIO|>"
        + "<|cAUDIO|>" * num_audio_tokens
        + "<|endofcAUDIO|>"
    )

    prompt   = f"<start_of_turn>user\n{audio_placeholder}<end_of_turn>\n<start_of_turn>model\n"
    response = transcript.strip() + tokenizer.eos_token

    prompt_ids   = tokenizer(prompt,   add_special_tokens=True,  return_tensors="pt").input_ids[0]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids[0]

    input_ids = torch.cat([prompt_ids, response_ids])
    labels    = input_ids.clone()
    labels[: len(prompt_ids)] = -100   # mask prompt — only supervise response

    return {
        "wav":            wav,
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }


# ── Streaming train dataset ───────────────────────────────────────────────────

class LibriSpeechStreamingDataset(IterableDataset):
    """
    Interleaves LibriSpeech train splits (and optionally extra corpora) using
    HuggingFace datasets streaming. No data is downloaded upfront — samples
    are fetched and preprocessed on-the-fly.

    Args:
        tokenizer:          Gemma3 tokenizer with audio special tokens added.
        audio_token_index:  Token id of <|cAUDIO|>.
        use_extra_data:     Also stream CommonVoice-en and VoxPopuli-en.
        shuffle_buffer:     Size of the in-memory shuffle buffer per dataset.
        seed:               Random seed for shuffling and interleaving.
        max_duration:       Skip samples longer than this (seconds).
        min_duration:       Skip samples shorter than this (seconds).
    """

    # (hf_dataset_id, config_name, split)
    LIBRISPEECH_SPLITS = [
        ("openslr/librispeech_asr", "clean", "train.100"),
        ("openslr/librispeech_asr", "clean", "train.360"),
        ("openslr/librispeech_asr", "other", "train.500"),
    ]
    EXTRA_SPLITS = [
        ("mozilla-foundation/common_voice_13_0", "en", "train"),
        ("facebook/voxpopuli",                   "en", "train"),
    ]

    def __init__(
        self,
        tokenizer,
        audio_token_index: int,
        use_extra_data:    bool  = False,
        shuffle_buffer:    int   = 1000,
        seed:              int   = 42,
        max_duration:      float = 20.0,   # skip very long utterances
        min_duration:      float = 1.0,    # skip very short utterances
    ):
        self.tokenizer         = tokenizer
        self.audio_token_index = audio_token_index
        self.use_extra_data    = use_extra_data
        self.shuffle_buffer    = shuffle_buffer
        self.seed              = seed
        self.max_duration      = max_duration
        self.min_duration      = min_duration

        self._dataset = self._build_interleaved()

    def _load_split(self, dataset_id: str, config: str, split: str):
        """Load one HuggingFace split in streaming mode with shuffle buffer."""
        ds = load_dataset(
            dataset_id,
            config,
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        return ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

    def _build_interleaved(self):
        """
        Build the interleaved streaming dataset.
        LibriSpeech splits are weighted proportionally to their size.
        Extra data (if enabled) is given lower weight to avoid domain shift.
        """
        splits      = list(self.LIBRISPEECH_SPLITS)
        # Weights proportional to hours: 100, 360, 500 → normalise
        ls_weights  = [100, 360, 500]

        if self.use_extra_data:
            splits     += list(self.EXTRA_SPLITS)
            # Extra datasets given 10% combined weight each
            ls_weights += [100, 60]   # rough CommonVoice-en and VoxPopuli-en sizes

        datasets_list = [
            self._load_split(did, cfg, sp)
            for did, cfg, sp in splits
        ]

        total = sum(ls_weights)
        probs = [w / total for w in ls_weights]

        return interleave_datasets(
            datasets_list,
            probabilities=probs,
            seed=self.seed,
            stopping_strategy="all_exhausted",
        )

    def _extract_audio_text(self, sample: dict):
        """
        Normalise audio/text field names across different HuggingFace datasets.
        LibriSpeech:  sample["audio"]["array"], sample["text"]
        CommonVoice:  sample["audio"]["array"], sample["sentence"]
        VoxPopuli:    sample["audio"]["array"], sample["normalized_text"]
        """
        audio_dict = sample.get("audio", {})
        array      = audio_dict.get("array")
        sr         = audio_dict.get("sampling_rate", 16000)
        transcript = (
            sample.get("text")
            or sample.get("sentence")
            or sample.get("normalized_text")
            or ""
        ).strip()
        return array, sr, transcript

    def __iter__(self):
        for sample in self._dataset:
            array, sr, transcript = self._extract_audio_text(sample)

            # ── Filter ────────────────────────────────────────────────────────
            if array is None or not transcript:
                continue
            duration = len(array) / sr
            if duration < self.min_duration or duration > self.max_duration:
                continue

            # ── Preprocess ────────────────────────────────────────────────────
            try:
                wav = preprocess_wav(array, src_sr=sr)
            except Exception:
                continue   # skip corrupt samples silently

            yield build_sample(wav, transcript, self.tokenizer, self.audio_token_index)


# ── Validation dataset (map-style, fully loaded) ─────────────────────────────

class LibriSpeechValDataset(torch.utils.data.Dataset):
    """
    Loads LibriSpeech validation.clean fully into memory.
    It's only ~5h / ~5,600 samples — no need to stream.

    This gives us a deterministic, fixed validation set we can iterate
    in a consistent order every eval step.
    """

    def __init__(self, tokenizer, audio_token_index: int,
                 max_duration: float = 20.0):
        self.tokenizer         = tokenizer
        self.audio_token_index = audio_token_index
        self.max_duration      = max_duration

        print("[ValDataset] Loading librispeech validation.clean (streaming=False)...")
        raw = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="validation",
            streaming=False,
            trust_remote_code=True,
        )
        # Pre-process all samples once so validation is fast
        self.samples = []
        skipped = 0
        for item in raw:
            array = item["audio"]["array"]
            sr    = item["audio"]["sampling_rate"]
            text  = item["text"].strip()

            duration = len(array) / sr
            if duration > max_duration or not text:
                skipped += 1
                continue
            try:
                wav = preprocess_wav(array, src_sr=sr)
            except Exception:
                skipped += 1
                continue
            self.samples.append((wav, text))

        print(f"[ValDataset] {len(self.samples)} val samples "
              f"({skipped} skipped). Ready.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav, transcript = self.samples[idx]
        return build_sample(wav, transcript, self.tokenizer, self.audio_token_index)
