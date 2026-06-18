"""
Stage 1 Adapter Training — CovoAudio + Qwen2.5-3B-Instruct
===========================================================

Key design choices vs. the original pipeline:
  1. Epoch-based training   — no step budget; loops for `--num_epochs` full
                              passes over the (streamed) dataset.
  2. Epoch-level LR warmup  — LR ramps linearly over the first 50 % of epochs,
                              then cosine-decays to a small floor (1 % of peak).
  3. Dual loss              — cross-entropy (ASR) + cosine-embedding alignment.

Loss function detail
────────────────────
The adapter projects Whisper encoder outputs (768-d) into the Qwen2 hidden
space (2048-d).  To explicitly encourage the adapter to land in the LLM's
embedding manifold we add a cosine-embedding alignment term:

    L_align = 1 − cos(adapter_out, lm_embed_lookup(ground_truth_tokens))

This is averaged only over positions that carry a real audio token
(input_ids == audio_token_index) so it does not interfere with text positions.

Total loss:
    L = L_ce + λ · L_align          (λ = config.alignment_loss_weight, default 0.1)

CLI usage
─────────
    python train.py \\
        --output_dir /workspace/checkpoints \\
        --num_epochs 10 \\
        --batch_size 4 \\
        --grad_accum 4 \\
        --lr 2e-4 \\
        --extra_data          # optional: adds CommonVoice + VoxPopuli
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler

# ── Local imports ─────────────────────────────────────────────────────────────
# Adjust sys.path if running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from covoaudio.configuration_covo_audio import CovoAudioConfig
from covoaudio.modeling_covo_audio import CovoAudioForConditionalGeneration
from covoaudio.training.dataset import (
    LibriSpeechStreamingDataset,
    LibriSpeechValDataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

class AudioSpeechCollator:
    """
    Pad a batch of samples from build_sample() to uniform length.

    Padding strategy:
      • input_ids / attention_mask — right-padded with pad_token_id / 0
      • labels                     — right-padded with -100
      • wav                        — right-padded with zeros
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len  = max(b["input_ids"].shape[0] for b in batch)
        max_wav  = max(b["wav"].shape[0]        for b in batch)

        input_ids_list  = []
        labels_list     = []
        attn_mask_list  = []
        wav_list        = []

        for b in batch:
            seq_len  = b["input_ids"].shape[0]
            wav_len  = b["wav"].shape[0]
            pad_seq  = max_len - seq_len
            pad_wav  = max_wav - wav_len

            input_ids_list.append(
                F.pad(b["input_ids"],      (0, pad_seq), value=self.pad_token_id))
            labels_list.append(
                F.pad(b["labels"],         (0, pad_seq), value=-100))
            attn_mask_list.append(
                F.pad(b["attention_mask"], (0, pad_seq), value=0))
            wav_list.append(
                F.pad(b["wav"],            (0, pad_wav), value=0.0))

        return {
            "input_ids":      torch.stack(input_ids_list),
            "labels":         torch.stack(labels_list),
            "attention_mask": torch.stack(attn_mask_list),
            "wav":            torch.stack(wav_list),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Alignment loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_alignment_loss(
    adapter_outputs:    torch.Tensor,   # (B, T_total, D_llm)
    input_ids:          torch.Tensor,   # (B, T_total)
    lm_embedding_table: nn.Embedding,   # (vocab_size, D_llm)
    labels:             torch.Tensor,   # (B, T_total)
    audio_token_index:  int,
) -> torch.Tensor:
    """
    Cosine-embedding alignment loss.

    For positions where input_ids == audio_token_index (i.e. the audio
    placeholder slots), we compare:
      • adapter_out[pos]    — what the adapter actually produced
      • lm_embed(label[pos+1]) — the LLM embedding of the *next* ground-truth
                                 text token (the immediately following token
                                 the model is trying to predict)

    Intuition: the adapter's final audio token should be "close to" the
    embedding of the first word of the transcript, guiding the adapter to
    bridge the audio–text modality gap.

    Only positions that are both:
      (a) audio placeholder tokens  AND
      (b) followed by a valid (≥0) label
    are included in the average.

    Returns scalar loss (0 if no valid positions in batch).
    """
    B, T, D = adapter_outputs.shape

    # Audio positions: (B, T) boolean mask
    audio_mask = (input_ids == audio_token_index)   # True where <|cAUDIO|>

    # Shift labels left by 1 to get the "next token" for each position
    # Pad the last position with -100 so it's excluded.
    next_labels = torch.full_like(labels, -100)
    next_labels[:, :-1] = labels[:, 1:]

    # Valid: audio position AND the following label is a real token (not -100)
    valid_mask = audio_mask & (next_labels >= 0)    # (B, T)

    if not valid_mask.any():
        return adapter_outputs.new_tensor(0.0)

    # Gather adapter outputs and target embeddings for valid positions
    adapter_vecs = adapter_outputs[valid_mask]                  # (N, D)
    target_ids   = next_labels[valid_mask]                      # (N,)

    with torch.no_grad():
        target_vecs = lm_embedding_table(target_ids)            # (N, D)

    # cosine_embedding_loss with target=1 means "make them similar"
    targets = adapter_vecs.new_ones(adapter_vecs.shape[0])      # all +1
    loss = F.cosine_embedding_loss(
        adapter_vecs,
        target_vecs,
        targets,
        margin=0.0,
        reduction="mean",
    )
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Epoch-level cosine LR schedule with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

class EpochWarmupCosineScheduler:
    """
    Epoch-granularity scheduler:
      • epochs  0 … floor(warmup_frac × num_epochs) - 1 : linear ramp  0 → lr
      • epochs  warmup_end … num_epochs - 1              : cosine decay lr → lr_min

    Call scheduler.step(epoch) at the START of each epoch (before any
    optimizer.step calls inside that epoch).

    Args:
        optimizer:     PyTorch optimizer.
        num_epochs:    Total number of training epochs.
        warmup_frac:   Fraction of epochs used for warmup (default 0.5 → 50 %).
        lr_min_frac:   LR floor as a fraction of peak LR (default 0.01 → 1 %).
    """

    def __init__(
        self,
        optimizer:    torch.optim.Optimizer,
        num_epochs:   int,
        warmup_frac:  float = 0.5,
        lr_min_frac:  float = 0.01,
    ):
        self.optimizer   = optimizer
        self.num_epochs  = num_epochs
        self.warmup_end  = max(1, math.floor(warmup_frac * num_epochs))
        self.lr_min_frac = lr_min_frac
        # Capture base LRs from param groups
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def _compute_lr(self, epoch: int, base_lr: float) -> float:
        lr_min = base_lr * self.lr_min_frac
        if epoch < self.warmup_end:
            # Linear warmup: epoch 0 → tiny, epoch warmup_end-1 → base_lr
            return base_lr * (epoch + 1) / self.warmup_end
        else:
            # Cosine decay from base_lr to lr_min
            progress = (epoch - self.warmup_end) / max(
                1, self.num_epochs - self.warmup_end - 1
            )
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min + (base_lr - lr_min) * cosine

    def step(self, epoch: int):
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self._compute_lr(epoch, base_lr)

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:             CovoAudioForConditionalGeneration,
    loader:            DataLoader,
    optimizer:         torch.optim.Optimizer,
    device:            torch.device,
    epoch:             int,
    grad_accum:        int,
    alignment_weight:  float,
    audio_token_index: int,
    max_grad_norm:     float = 1.0,
    log_every:         int   = 50,
) -> dict:
    """Run one full pass over the (streaming) training data."""
    model.train()

    total_ce_loss    = 0.0
    total_align_loss = 0.0
    total_loss       = 0.0
    steps            = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        wav            = batch["wav"].to(device)

        # ── Forward pass ──────────────────────────────────────────────────────
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            wav=wav,
            output_adapter_hidden_states=True,  # needed for alignment loss
        )

        ce_loss    = outputs.loss
        align_loss = torch.tensor(0.0, device=device)

        # ── Alignment loss ────────────────────────────────────────────────────
        if alignment_weight > 0.0 and hasattr(outputs, "adapter_hidden_states"):
            # adapter_hidden_states: (B, T_audio, D_llm)
            # We need to scatter them back into the full-sequence positions.
            # model.get_input_embeddings() returns the LM embed table.
            lm_embed = model.language_model.get_input_embeddings()

            # Get the full-sequence hidden states at the audio positions.
            # The model's forward places adapter embeddings at audio positions
            # inside inputs_embeds before the LM forward.
            # We use the model's stored adapter_hidden_states (already projected
            # to D_llm) and the input_ids mask to align positions.
            align_loss = compute_alignment_loss(
                adapter_outputs=outputs.adapter_hidden_states,  # (B, T_full, D_llm)
                input_ids=input_ids,
                lm_embedding_table=lm_embed,
                labels=labels,
                audio_token_index=audio_token_index,
            )

        loss = ce_loss + alignment_weight * align_loss

        # ── Backward + accumulate ─────────────────────────────────────────────
        (loss / grad_accum).backward()

        total_ce_loss    += ce_loss.item()
        total_align_loss += align_loss.item()
        total_loss       += loss.item()
        steps            += 1

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % log_every == 0:
            avg_ce    = total_ce_loss    / steps
            avg_align = total_align_loss / steps
            avg_total = total_loss       / steps
            print(
                f"  [epoch {epoch+1}  step {step+1}] "
                f"ce={avg_ce:.4f}  align={avg_align:.4f}  total={avg_total:.4f}"
            )

    # Flush any remaining gradient accumulation buffer
    if steps % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return {
        "ce_loss":    total_ce_loss    / max(steps, 1),
        "align_loss": total_align_loss / max(steps, 1),
        "total_loss": total_loss       / max(steps, 1),
    }


@torch.no_grad()
def validate(
    model:             CovoAudioForConditionalGeneration,
    loader:            DataLoader,
    device:            torch.device,
    alignment_weight:  float,
    audio_token_index: int,
) -> dict:
    """Compute validation losses over the fixed val set."""
    model.eval()

    total_ce    = 0.0
    total_align = 0.0
    total       = 0.0
    n           = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        wav            = batch["wav"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            wav=wav,
            output_adapter_hidden_states=True,
        )

        ce_loss    = outputs.loss
        align_loss = torch.tensor(0.0, device=device)

        if alignment_weight > 0.0 and hasattr(outputs, "adapter_hidden_states"):
            lm_embed   = model.language_model.get_input_embeddings()
            align_loss = compute_alignment_loss(
                outputs.adapter_hidden_states,
                input_ids,
                lm_embed,
                labels,
                audio_token_index,
            )

        total_ce    += ce_loss.item()
        total_align += align_loss.item()
        total       += (ce_loss + alignment_weight * align_loss).item()
        n           += 1

    return {
        "val_ce":    total_ce    / max(n, 1),
        "val_align": total_align / max(n, 1),
        "val_loss":  total       / max(n, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-1 adapter training for CovoAudio (Qwen2 + Whisper)"
    )
    p.add_argument("--model_name_or_path", default=None,
                   help="Pretrained checkpoint dir; if None, init from config.")
    p.add_argument("--output_dir",         required=True)
    p.add_argument("--num_epochs",         type=int,   default=10)
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--grad_accum",         type=int,   default=4)
    p.add_argument("--lr",                 type=float, default=2e-4)
    p.add_argument("--weight_decay",       type=float, default=0.01)
    p.add_argument("--max_grad_norm",      type=float, default=1.0)
    p.add_argument("--warmup_frac",        type=float, default=0.5,
                   help="Fraction of epochs used for linear LR warmup (default 0.5 = 50%%).")
    p.add_argument("--lr_min_frac",        type=float, default=0.01,
                   help="LR floor as fraction of peak (default 0.01 = 1%%).")
    p.add_argument("--alignment_weight",   type=float, default=None,
                   help="Override config.alignment_loss_weight.")
    p.add_argument("--extra_data",         action="store_true",
                   help="Also stream CommonVoice-en and VoxPopuli-en.")
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--log_every",          type=int,   default=50)
    p.add_argument("--save_every_epoch",   action="store_true",
                   help="Save a checkpoint after every epoch (default: only best val).")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--dtype",              default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = {"float32": torch.float32,
               "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_name = (args.model_name_or_path
                      or "Qwen/Qwen2.5-3B-Instruct")
    print(f"[init] Loading tokenizer from {tokenizer_name} …")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        padding_side="right",
    )
    # Add audio special tokens if not already present
    audio_special_tokens = ["<|begofcAUDIO|>", "<|cAUDIO|>", "<|endofcAUDIO|>"]
    new_tokens = [t for t in audio_special_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        print(f"[init] Added {len(new_tokens)} audio special tokens.")

    audio_token_index = tokenizer.convert_tokens_to_ids("<|cAUDIO|>")

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model_name_or_path:
        print(f"[init] Loading model from {args.model_name_or_path} …")
        model = CovoAudioForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
        )
    else:
        print("[init] Initialising model from default CovoAudioConfig …")
        config = CovoAudioConfig()
        model  = CovoAudioForConditionalGeneration(config)

    # Resize embeddings if tokenizer vocab grew
    model.language_model.resize_token_embeddings(len(tokenizer))
    model = model.to(device=device, dtype=dtype)

    alignment_weight = (
        args.alignment_weight
        if args.alignment_weight is not None
        else model.config.alignment_loss_weight
    )
    print(f"[init] Alignment loss weight: {alignment_weight}")

    # ── Freeze strategy (Stage 1: train adapter only) ─────────────────────────
    # Freeze Whisper encoder and Qwen2 LM; only train the adapter projection.
    for name, param in model.named_parameters():
        if "audio_adapter" in name or "adapter" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[init] Trainable: {trainable:,} / {total:,} parameters "
          f"({100*trainable/total:.1f} %)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("[data] Building training dataset …")
    train_ds = LibriSpeechStreamingDataset(
        tokenizer=tokenizer,
        audio_token_index=audio_token_index,
        use_extra_data=args.extra_data,
        seed=args.seed,
    )
    print("[data] Building validation dataset …")
    val_ds = LibriSpeechValDataset(
        tokenizer=tokenizer,
        audio_token_index=audio_token_index,
    )

    collator   = AudioSpeechCollator(pad_token_id=tokenizer.pad_token_id or 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )

    # ── Optimizer + epoch-based scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    scheduler = EpochWarmupCosineScheduler(
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        warmup_frac=args.warmup_frac,
        lr_min_frac=args.lr_min_frac,
    )

    warmup_end = scheduler.warmup_end
    print(
        f"[schedule] {args.num_epochs} epochs total | "
        f"warmup epochs 0–{warmup_end-1} ({warmup_end}/{args.num_epochs} = "
        f"{100*warmup_end/args.num_epochs:.0f} %) | "
        f"cosine decay epochs {warmup_end}–{args.num_epochs-1}"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        # Update LR at the start of each epoch
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"\n{'='*70}\n"
            f"Epoch {epoch+1}/{args.num_epochs}  |  lr={current_lr:.2e}\n"
            f"{'='*70}"
        )

        t0 = time.time()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            grad_accum=args.grad_accum,
            alignment_weight=alignment_weight,
            audio_token_index=audio_token_index,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
        )
        train_time = time.time() - t0

        print(f"\n[epoch {epoch+1}] train  "
              f"ce={train_metrics['ce_loss']:.4f}  "
              f"align={train_metrics['align_loss']:.4f}  "
              f"total={train_metrics['total_loss']:.4f}  "
              f"({train_time/60:.1f} min)")

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            alignment_weight=alignment_weight,
            audio_token_index=audio_token_index,
        )
        print(f"[epoch {epoch+1}] val    "
              f"ce={val_metrics['val_ce']:.4f}  "
              f"align={val_metrics['val_align']:.4f}  "
              f"total={val_metrics['val_loss']:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if args.save_every_epoch:
            ckpt_dir = output_dir / f"epoch_{epoch+1:03d}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[ckpt] Saved epoch checkpoint → {ckpt_dir}")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"[ckpt] New best val_loss={best_val_loss:.4f} → {best_dir}")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[done] Final model saved → {final_dir}")
    print(f"[done] Best val_loss={best_val_loss:.4f} → {output_dir/'best'}")


if __name__ == "__main__":
    main()