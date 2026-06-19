"""
Streaming Dataset for Stage 1 Adapter Training  (Qwen2 edition)
================================================================
Sources (all streamed — nothing downloaded upfront):

  Primary:
    openslr/librispeech_asr  clean  train.100   (~100 h)
    openslr/librispeech_asr  clean  train.360   (~360 h)
    openslr/librispeech_asr  other  train.500   (~500 h)
    ─────────────────────────────────────────────────────
    Total primary                                ~960 h

  Supplementary (--extra_data flag):
    mozilla-foundation/common_voice_13_0  en    (~2 400 h validated)
    facebook/voxpopuli                    en    (~543 h)

  Validation (map-style, ~5 h / ~5 600 samples):
    openslr/librispeech_asr  clean  validation

Audio pipeline
──────────────
HuggingFace delivers 16 kHz float32 waveforms.
Model expects 24 kHz (resampled back to 16 kHz internally by audio_encoder).
16→24→16 is lossless for Whisper (≤8 kHz bandwidth).

Prompt format — Qwen2 ChatML  (must match get_dialog_prompt in modeling file)
─────────────────────────────────────────────────────────────────────────────
  <|im_start|>system
  {system_prompt}<|im_end|>
  <|im_start|>user
  <|begofcAUDIO|><|cAUDIO|>×N<|endofcAUDIO|><|im_end|>
  <|im_start|>assistant
  {transcript}<|im_end|>

Labels: -100 for all prompt tokens; transcript + <|im_end|> token ids supervised.
"""

import os
os.environ["HF_DATASETS_CACHE"]     = "/workspace/hf_cache"
os.environ["HF_HOME"]               = "/workspace/hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/hf_hub"

import torch
import numpy as np
import torchaudio.functional as AF
import torch.nn.functional as F
from torch.utils.data import IterableDataset, Dataset
from datasets import load_dataset, interleave_datasets


def calc_seq_len(seq_len: int, adapter_downsample: int = 8) -> int:
    """
    Frames after AudioAdapter's downsampling conv stack.

    num_layers = adapter_downsample.bit_length() - 1, matching AudioAdapter's
    own layer-count formula exactly (see modeling_covo_audio.py:AudioAdapter).
    For adapter_downsample=8 → 3 stride-2 stages → 8x downsampling.
    """
    num_layers = adapter_downsample.bit_length() - 1
    for _ in range(num_layers):
        seq_len = (seq_len + 1) // 2
    return seq_len


# ─────────────────────────────────────────────────────────────────────────────
# Waveform preprocessing  (mirrors get_dialog_prompt in modeling_covo_audio.py)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_wav(
    array:        np.ndarray,
    src_sr:       int,
    target_sr:    int   = 24_000,
    max_duration: float = 30.0,
) -> torch.Tensor:
    """
    np.ndarray (float32, mono) → torch.Tensor at target_sr.

    Exactly mirrors the wav pre-processing in get_dialog_prompt():
      1. Resample to 24 kHz
      2. Clip to max_duration
      3. Trim to hop-size multiple  (hop = target_sr // 100 = 240)
      4. Pad to multiple of 480     (model conv-stack requirement)
    """
    wav = torch.from_numpy(array.copy()).float()

    if src_sr != target_sr:
        wav = AF.resample(wav, orig_freq=src_sr, new_freq=target_sr)

    wav = wav[: int(max_duration * target_sr)]

    hop = target_sr // 100            # 240 @ 24 kHz
    wav = wav[: len(wav) // hop * hop]

    rem = wav.shape[0] % 480
    if rem:
        wav = F.pad(wav, (0, 480 - rem))

    return wav


# ─────────────────────────────────────────────────────────────────────────────
# Sample builder — Qwen2 ChatML
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant."
)


def build_sample(
    wav:                 torch.Tensor,
    transcript:          str,
    tokenizer,
    audio_token_index:   int,
    system_prompt:       str = DEFAULT_SYSTEM_PROMPT,
    adapter_downsample:  int = 8,
) -> dict:
    """
    Build a tokenised training sample.

    Prompt structure (Qwen2 ChatML, matches get_dialog_prompt):
        <|im_start|>system\\n{system_prompt}<|im_end|>\\n
        <|im_start|>user\\n<|begofcAUDIO|><|cAUDIO|>×N<|endofcAUDIO|><|im_end|>\\n
        <|im_start|>assistant\\n

    Supervised response:
        {transcript}<|im_end|>

    Labels: -100 on all prompt positions; token-ids on response positions.

    adapter_downsample MUST match config.adapter_downsample (default 8) so
    that the number of <|cAUDIO|> placeholders inserted here exactly equals
    the number of feature vectors AudioAdapter will actually produce. A
    mismatch here silently truncates real audio features in training (see
    calc_seq_len's docstring for the bug this previously caused).

    Returns
    -------
    dict with keys:
        wav            float32 tensor  (T_wav,)
        input_ids      long tensor     (T_seq,)
        labels         long tensor     (T_seq,)  — prompt masked with -100
        attention_mask long tensor     (T_seq,)  — all ones
    """
    # Number of <|cAUDIO|> placeholder tokens for this waveform.
    #
    # MUST be a FIXED constant, not duration-derived: audio_encoder() always
    # pads/trims every clip to a fixed 30s window (pad_or_trim -> N_SAMPLES)
    # before running Whisper, so the encoder always outputs 1500 frames and
    # the adapter always outputs calc_seq_len(1500, adapter_downsample)
    # vectors — regardless of how long the actual speech in `wav` is. Using
    # the real clip duration here (as a previous version of this function
    # did) produces a placeholder count that doesn't match what the adapter
    # actually emits for any clip shorter than 30s, which is the cause of
    # the "[build_inputs_embeds] placeholder/feature count mismatch"
    # warning seen at runtime.
    num_audio_tokens = calc_seq_len(1500, adapter_downsample=adapter_downsample)

    audio_placeholder = (
        "<|begofcAUDIO|>"
        + "<|cAUDIO|>" * num_audio_tokens
        + "<|endofcAUDIO|>"
    )

    # Build the full ChatML prompt string
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{audio_placeholder}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    # Response — ends with <|im_end|> which is Qwen2's chat EOS (id=151645)
    response = transcript.strip() + "<|im_end|>"

    prompt_ids   = tokenizer(prompt,   add_special_tokens=True,  return_tensors="pt").input_ids[0]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids[0]

    input_ids = torch.cat([prompt_ids, response_ids])
    labels    = input_ids.clone()
    labels[: len(prompt_ids)] = -100     # mask prompt; supervise response only

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
    Interleaves LibriSpeech train splits (+ optional CommonVoice / VoxPopuli)
    via HuggingFace datasets streaming.  Nothing is downloaded upfront.

    Args:
        tokenizer:          Qwen2 tokenizer (with audio special tokens added).
        audio_token_index:  Token id of <|cAUDIO|>.
        use_extra_data:     Also stream CommonVoice-en and VoxPopuli-en.
        shuffle_buffer:     In-memory shuffle buffer per shard.
        seed:               RNG seed.
        max_duration:       Drop samples longer than this (seconds).
        min_duration:       Drop samples shorter than this (seconds).
        system_prompt:      System message for each ChatML turn.
        adapter_downsample: MUST match config.adapter_downsample. Controls
                            how many <|cAUDIO|> placeholders are inserted per
                            clip — mismatching this silently truncates real
                            adapter output during training.
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
        audio_token_index:  int,
        use_extra_data:     bool  = False,
        shuffle_buffer:     int   = 1_000,
        seed:               int   = 42,
        max_duration:       float = 20.0,
        min_duration:       float = 1.0,
        system_prompt:      str   = DEFAULT_SYSTEM_PROMPT,
        adapter_downsample: int   = 8,
    ):
        self.tokenizer          = tokenizer
        self.audio_token_index  = audio_token_index
        self.use_extra_data     = use_extra_data
        self.shuffle_buffer     = shuffle_buffer
        self.seed               = seed
        self.max_duration       = max_duration
        self.min_duration       = min_duration
        self.system_prompt      = system_prompt
        self.adapter_downsample = adapter_downsample
        self._dataset           = self._build_interleaved()

    def _load_split(self, dataset_id: str, config: str, split: str):
        ds = load_dataset(
            dataset_id, config, split=split,
            streaming=True, cache_dir="/workspace/",
            trust_remote_code=True,
        )
        return ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

    def _build_interleaved(self):
        splits     = list(self.LIBRISPEECH_SPLITS)
        ls_weights = [100, 360, 500]


        if self.use_extra_data:
            splits     += list(self.EXTRA_SPLITS)
            ls_weights += [100, 60]

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
                adapter_downsample=self.adapter_downsample,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Map-style validation dataset
# ─────────────────────────────────────────────────────────────────────────────

class LibriSpeechValDataset(Dataset):
    """
    Loads LibriSpeech validation.clean fully into memory (~5 h / ~5 600 samples).
    All samples are pre-processed once; validation passes are fast and
    perfectly deterministic.
    """

    def __init__(
        self,
        tokenizer,
        audio_token_index:  int,
        max_duration:       float = 20.0,
        system_prompt:      str   = DEFAULT_SYSTEM_PROMPT,
        adapter_downsample: int   = 8,
    ):
        self.tokenizer          = tokenizer
        self.audio_token_index  = audio_token_index
        self.system_prompt      = system_prompt
        self.adapter_downsample = adapter_downsample

        print("[ValDataset] Loading librispeech_asr validation.clean …")
        raw = load_dataset(
            "openslr/librispeech_asr", "clean",
            split="validation", streaming=False,
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
            adapter_downsample=self.adapter_downsample,
        )
