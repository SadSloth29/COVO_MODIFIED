"""
Streaming Dataset for Stage 1 Adapter Training  (Qwen2 edition)
================================================================
Sources (all streamed — nothing downloaded upfront):

  Primary:
    openslr/librispeech_asr  clean  train.100   (~100 h)
    openslr/librispeech_asr  clean  train.360   (~360 h)
    openslr/librispeech_asr  other  train.500   (~500 h)
    ─────────────────────────────────────────────────────
    Total primary:                               ~960 h

  Supplementary (--extra_data flag):
    mozilla-foundation/common_voice_13_0  en    (~2 400 h validated)
    facebook/voxpopuli                    en    (~543 h)

  Validation (map-style, ~5 h / ~5 600 samples):
    openslr/librispeech_asr  clean  validation

Audio pipeline:
  HuggingFace delivers 16 kHz float32 waveforms.
  We resample to 24 kHz to match the model's expected sample rate.
  The audio_encoder internally resamples back to 16 kHz for Whisper;
  the 16→24→16 round-trip is lossless for Whisper (≤8 kHz bandwidth).

Prompt format — Qwen2 ChatML:
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  <|begofcAUDIO|><|cAUDIO|>×N<|endofcAUDIO|><|im_end|>
  <|im_start|>assistant
  {transcript}<|im_end|>

  Labels: -100 for all prompt tokens; transcript token ids for response.
"""

import os
os.environ["HF_DATASETS_CACHE"]    = "/workspace/hf_cache"
os.environ["HF_HOME"]              = "/workspace/hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/hf_hub"

import torch
import numpy as np
import torchaudio.functional as AF
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset, interleave_datasets


# ─────────────────────────────────────────────────────────────────────────────
# Token-sequence length (mirrors modeling_covo_audio.py conv-stack)
# ─────────────────────────────────────────────────────────────────────────────

def calc_seq_len(seq_len: int) -> int:
    """Compute number of audio tokens after the 4-layer 2× downsampling stack."""
    for _ in range(4):
        seq_len = (seq_len + 1) // 2   # ceiling division
    return seq_len


# ─────────────────────────────────────────────────────────────────────────────
# Waveform preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_wav(
    array:       np.ndarray,
    src_sr:      int,
    target_sr:   int   = 24_000,
    max_duration: float = 30.0,
) -> torch.Tensor:
    """
    np.ndarray (float32, mono) → torch.Tensor at target_sr.

    Steps:
      1. Resample src_sr → target_sr (usually 16 k → 24 k).
      2. Clip to max_duration.
      3. Trim to a multiple of the hop size (target_sr // 100 = 240 @ 24 k).
      4. Pad to multiple of 480 (model conv-stack requirement).
    """
    wav = torch.from_numpy(array.copy()).float()

    if src_sr != target_sr:
        wav = AF.resample(wav, orig_freq=src_sr, new_freq=target_sr)

    max_samples = int(max_duration * target_sr)
    wav = wav[:max_samples]

    hop = target_sr // 100           # 240 samples @ 24 kHz
    wav = wav[: len(wav) // hop * hop]

    rem = wav.shape[0] % 480
    if rem:
        wav = F.pad(wav, (0, 480 - rem))

    return wav


# ─────────────────────────────────────────────────────────────────────────────
# Sample builder  — Qwen2 ChatML format
# ─────────────────────────────────────────────────────────────────────────────

# Default system prompt — can be overridden via build_sample(system_prompt=…)
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def build_sample(
    wav:               torch.Tensor,
    transcript:        str,
    tokenizer,
    audio_token_index: int,
    system_prompt:     str = DEFAULT_SYSTEM_PROMPT,
) -> dict:
    """
    Build a tokenised training sample for Qwen2 (ChatML).

    Prompt (not supervised):
        <|im_start|>system\\n{system_prompt}<|im_end|>\\n
        <|im_start|>user\\n<|begofcAUDIO|><|cAUDIO|>×N<|endofcAUDIO|><|im_end|>\\n
        <|im_start|>assistant\\n

    Response (supervised):
        {transcript}<|im_end|>

    Labels: -100 for prompt positions; token ids for response positions.

    Returns dict with keys:
        wav            – preprocessed waveform tensor  (float32)
        input_ids      – token ids                     (long)
        labels         – supervised targets             (long, -100 masked)
        attention_mask – 1 everywhere                  (long)
    """
    # Number of <|cAUDIO|> placeholder tokens
    duration_frames  = len(wav) * 100 // 24_000
    num_audio_tokens = calc_seq_len(duration_frames)

    audio_placeholder = (
        "<|begofcAUDIO|>"
        + "<|cAUDIO|>" * num_audio_tokens
        + "<|endofcAUDIO|>"
    )

    # ── Qwen2 ChatML template ─────────────────────────────────────────────────
    # We build the string manually so it works regardless of whether the
    # tokenizer has apply_chat_template configured.
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    # Response ends with <|im_end|> (Qwen2 EOS in chat contexts)
    response = transcript.strip() + "<|im_end|>"

    prompt_ids   = tokenizer(prompt,   add_special_tokens=True,  return_tensors="pt").input_ids[0]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids[0]

    input_ids = torch.cat([prompt_ids, response_ids])
    labels    = input_ids.clone()
    labels[: len(prompt_ids)] = -100   # mask prompt — supervise response only

    return {
        "wav":            wav,
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streaming training dataset
# ─────────────────────────────────────────────────────────────────────────────

class LibriSpeechStreamingDataset(IterableDataset):
    """
    Interleaves LibriSpeech train splits (and optionally CommonVoice / VoxPopuli)
    using HuggingFace datasets streaming.  No data is downloaded upfront.

    Args:
        tokenizer:          Qwen2 tokenizer with audio special tokens added.
        audio_token_index:  Token id of <|cAUDIO|>.
        use_extra_data:     Also stream CommonVoice-en and VoxPopuli-en.
        shuffle_buffer:     In-memory shuffle buffer size per dataset shard.
        seed:               RNG seed for shuffling and interleaving.
        max_duration:       Skip samples longer than this (seconds).
        min_duration:       Skip samples shorter than this (seconds).
        system_prompt:      System message inserted before each user turn.
    """

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
        shuffle_buffer:    int   = 1_000,
        seed:              int   = 42,
        max_duration:      float = 20.0,
        min_duration:      float = 1.0,
        system_prompt:     str   = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer         = tokenizer
        self.audio_token_index = audio_token_index
        self.use_extra_data    = use_extra_data
        self.shuffle_buffer    = shuffle_buffer
        self.seed              = seed
        self.max_duration      = max_duration
        self.min_duration      = min_duration
        self.system_prompt     = system_prompt

        self._dataset = self._build_interleaved()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_split(self, dataset_id: str, config: str, split: str):
        ds = load_dataset(
            dataset_id,
            config,
            split=split,
            streaming=True,
            cache_dir="/workspace/",
            trust_remote_code=True,
        )
        return ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

    def _build_interleaved(self):
        splits     = list(self.LIBRISPEECH_SPLITS)
        ls_weights = [100, 360, 500]   # proportional to hours

        if self.use_extra_data:
            splits     += list(self.EXTRA_SPLITS)
            ls_weights += [100, 60]    # rough CommonVoice & VoxPopuli weights

        datasets_list = [self._load_split(d, c, s) for d, c, s in splits]

        total = sum(ls_weights)
        probs = [w / total for w in ls_weights]

        return interleave_datasets(
            datasets_list,
            probabilities=probs,
            seed=self.seed,
            stopping_strategy="all_exhausted",
        )

    @staticmethod
    def _extract_audio_text(sample: dict):
        """
        Normalise audio/text field names across HuggingFace datasets.
        Handles plain dicts, AudioDecoder objects, and VoxPopuli-style objects.
        """
        audio_obj = sample["audio"]

        if isinstance(audio_obj, dict):
            array, sr = audio_obj.get("array"), audio_obj.get("sampling_rate")
        elif hasattr(audio_obj, "array") and hasattr(audio_obj, "sampling_rate"):
            array, sr = audio_obj.array, audio_obj.sampling_rate
        elif hasattr(audio_obj, "get_all_samples"):
            decoded = audio_obj.get_all_samples()
            array   = decoded.data.squeeze(0).numpy()
            sr      = decoded.sample_rate
        else:
            array, sr = audio_obj["array"], audio_obj["sampling_rate"]

        transcript = (
            sample.get("text")
            or sample.get("sentence")
            or sample.get("normalized_text")
            or ""
        ).strip()
        return array, sr, transcript

    # ── IterableDataset protocol ──────────────────────────────────────────────

    def __iter__(self):
        for sample in self._dataset:
            array, sr, transcript = self._extract_audio_text(sample)

            if array is None or not transcript:
                continue
            duration = len(array) / sr
            if duration < self.min_duration or duration > self.max_duration:
                continue

            try:
                wav = preprocess_wav(array, src_sr=sr)
            except Exception:
                continue

            yield build_sample(
                wav, transcript,
                self.tokenizer, self.audio_token_index,
                system_prompt=self.system_prompt,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Map-style validation dataset
# ─────────────────────────────────────────────────────────────────────────────

class LibriSpeechValDataset(Dataset):
    """
    Loads LibriSpeech validation.clean fully into memory (~5 h / ~5 600 samples).
    Pre-processes all samples once so each validation pass is fast and
    perfectly deterministic.
    """

    def __init__(
        self,
        tokenizer,
        audio_token_index: int,
        max_duration:  float = 20.0,
        system_prompt: str   = DEFAULT_SYSTEM_PROMPT,
    ):
        self.tokenizer         = tokenizer
        self.audio_token_index = audio_token_index
        self.system_prompt     = system_prompt

        print("[ValDataset] Loading librispeech_asr validation.clean …")
        raw = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="validation",
            streaming=False,
            trust_remote_code=True,
        )

        self.samples: list[tuple[torch.Tensor, str]] = []
        skipped = 0
        for item in raw:
            array = item["audio"]["array"]
            sr    = item["audio"]["sampling_rate"]
            text  = item["text"].strip()

            if len(array) / sr > max_duration or not text:
                skipped += 1
                continue
            try:
                wav = preprocess_wav(array, src_sr=sr)
            except Exception:
                skipped += 1
                continue
            self.samples.append((wav, text))

        print(f"[ValDataset] {len(self.samples)} samples ready "
              f"({skipped} skipped).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        wav, transcript = self.samples[idx]
        return build_sample(
            wav, transcript,
            self.tokenizer, self.audio_token_index,
            system_prompt=self.system_prompt,
        )