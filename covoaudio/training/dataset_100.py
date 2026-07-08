"""
Streaming Dataset for Stage 1 Adapter Training – LibriSpeech 100h only
========================================================================
Source:
    openslr/librispeech_asr  clean  train.100   (~100 h)

All audio is streamed; nothing is downloaded upfront.

Validation (map-style, ~5 h / ~5 600 samples):
    openslr/librispeech_asr  clean  validation

Preprocessing, prompt format and label masking are exactly as in the
full version – see docstring of build_sample() for details.
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
from datasets import load_dataset


def calc_seq_len(seq_len: int, adapter_downsample: int = 8) -> int:
    """
    Frames after AudioAdapter's downsampling conv stack.
    (Same as full version.)
    """
    num_layers = adapter_downsample.bit_length() - 1
    for _ in range(num_layers):
        seq_len = (seq_len + 1) // 2
    return seq_len


def preprocess_wav(
    array:        np.ndarray,
    src_sr:       int,
    target_sr:    int   = 24_000,
    max_duration: float = 30.0,
) -> torch.Tensor:
    """
    np.ndarray (float32, mono) → torch.Tensor at target_sr.
    (Same as full version.)
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


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio accurately."
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
    (Same as full version.)
    """
    num_audio_tokens = calc_seq_len(1500, adapter_downsample=adapter_downsample)
    audio_placeholder = (
        "<|begofcAUDIO|>"
        + "<|cAUDIO|>" * num_audio_tokens
        + "<|endofcAUDIO|>"
    )
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nTranscribe the following audio precisely:\n{audio_placeholder}<|im_end|>\n\n"
        f"<|im_start|>assistant\n"
    )
    response = transcript.strip() + "<|endoftext|>"

    prompt_ids   = tokenizer(prompt,   add_special_tokens=True,  return_tensors="pt").input_ids[0]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids[0]

    input_ids = torch.cat([prompt_ids, response_ids])
    labels    = input_ids.clone()
    labels[: len(prompt_ids)] = -100

    return {
        "wav":            wav,
        "input_ids":      input_ids,
        "labels":         labels,
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streaming training dataset – LibriSpeech train.100 only
# ─────────────────────────────────────────────────────────────────────────────

class LibriSpeech100StreamingDataset(IterableDataset):
    """
    Streams only the clean 100‑hour subset of LibriSpeech.

    Args:
        tokenizer:          Qwen2 tokenizer (with audio special tokens added).
        audio_token_index:  Token id of <|cAUDIO|>.
        shuffle_buffer:     In‑memory shuffle buffer.
        seed:               RNG seed.
        max_duration:       Drop samples longer than this (seconds).
        min_duration:       Drop samples shorter than this (seconds).
        system_prompt:      System message for each ChatML turn.
        adapter_downsample: MUST match config.adapter_downsample.
    """

    def __init__(
        self,
        tokenizer,
        audio_token_index:  int,
        shuffle_buffer:     int   = 1_000,
        seed:               int   = 42,
        max_duration:       float = 20.0,
        min_duration:       float = 1.0,
        system_prompt:      str   = DEFAULT_SYSTEM_PROMPT,
        adapter_downsample: int   = 8,
    ):
        self.tokenizer          = tokenizer
        self.audio_token_index  = audio_token_index
        self.shuffle_buffer     = shuffle_buffer
        self.seed               = seed
        self.max_duration       = max_duration
        self.min_duration       = min_duration
        self.system_prompt      = system_prompt
        self.adapter_downsample = adapter_downsample

        # Load only train.100
        raw = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="train.100",
            streaming=True,
            cache_dir="/workspace/",
            trust_remote_code=True,
        )
        self._dataset = raw.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

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
# Map-style validation dataset – unchanged (uses LibriSpeech validation.clean)
# ─────────────────────────────────────────────────────────────────────────────

class LibriSpeechValDataset(Dataset):
    """
    Loads LibriSpeech validation.clean fully into memory (~5 h / ~5 600 samples).
    (Identical to the original version.)
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
