"""
Stage 1 Adapter Training — CovoAudioForCausalLM (Qwen2.5-3B + Whisper-Small)
==============================================================================

What this script does
─────────────────────
• Epoch-based training loop — no step budget.
• 50 % epoch warmup  → cosine decay to 1 % of peak LR.
• Dual loss:
    L = L_ce  +  λ · L_align
  where L_align is a cosine-embedding loss that pulls the adapter's output
  embeddings (already in LLM hidden space) toward the LLM's own token
  embeddings for the ground-truth transcript tokens that immediately follow
  each audio placeholder.

How the forward pass works (from modeling_covo_audio.py)
─────────────────────────────────────────────────────────
  CovoAudioForCausalLM.forward() takes inputs_embeds (NOT input_ids + wavs).
  The caller is responsible for:
    1. Build token embeddings via  model.llm.get_input_embeddings()(input_ids)
    2. Run the audio encoder+adapter via  model.audio_encoder(wavs, device)
    3. Scatter adapter features into the embedding tensor at <|cAUDIO|> positions
  Then call model.forward(inputs_embeds=..., attention_mask=..., labels=...).

  This trainer does exactly the same thing as prepare_inputs_for_generation()
  but differentiably, so gradients flow through the adapter.

Alignment loss design
─────────────────────
After step (3) above, inputs_embeds already contains adapter outputs at audio
positions.  We extract those vectors and compare them with the LLM embedding
of the *next* ground-truth token (first token of the transcript).  This is
computed BEFORE the LLM forward so it adds no memory overhead from the LM.

    audio_vecs  = inputs_embeds[input_ids == cAUDIO_id]   # (N, D)
    next_ids    = labels shifted left by 1, masked to valid
    target_vecs = lm_embed(next_ids)                       # (N, D)  no grad
    L_align     = 1 − cosine_similarity(audio_vecs, target_vecs).mean()

Freeze policy (Stage 1)
───────────────────────
  Frozen  : model.encoder   (WhisperEncoder)
            model.llm       (Qwen2ForCausalLM)
  Trainable: model.audio_adapter  (AudioAdapter)

CLI usage
─────────
    python train.py \\
        --output_dir /workspace/checkpoints \\
        --num_epochs 10 \\
        --batch_size 4 \\
        --grad_accum 4 \\
        --lr 2e-4 \\
        [--extra_data]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from covoaudio.configuration_covo_audio import CovoAudioConfig
from covoaudio.modeling_covo_audio import CovoAudioForCausalLM, sequence_mask
from covoaudio.training.dataset import (
    LibriSpeechStreamingDataset,
    LibriSpeechValDataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────────────────────────

class AudioSpeechCollator:
    """
    Right-pads a batch of samples produced by build_sample() to equal length.

    Padding values:
        input_ids / attention_mask  → pad_token_id / 0
        labels                      → -100
        wav                         → 0.0
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_seq = max(b["input_ids"].shape[0] for b in batch)
        max_wav = max(b["wav"].shape[0]        for b in batch)

        input_ids_list, labels_list, attn_list, wav_list = [], [], [], []
        for b in batch:
            ps = max_seq - b["input_ids"].shape[0]
            pw = max_wav - b["wav"].shape[0]
            input_ids_list.append(F.pad(b["input_ids"],      (0, ps), value=self.pad_token_id))
            labels_list.append(   F.pad(b["labels"],         (0, ps), value=-100))
            attn_list.append(     F.pad(b["attention_mask"], (0, ps), value=0))
            wav_list.append(      F.pad(b["wav"],            (0, pw), value=0.0))

        return {
            "input_ids":      torch.stack(input_ids_list),
            "labels":         torch.stack(labels_list),
            "attention_mask": torch.stack(attn_list),
            "wav":            torch.stack(wav_list),
        }


# ─────────────────────────────────────────────────────────────────────────────
# inputs_embeds builder  (differentiable — gradients flow through adapter)
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs_embeds(
    model:             CovoAudioForCausalLM,
    input_ids:         torch.Tensor,   # (B, T)
    wav:               torch.Tensor,   # (B, T_wav)
    audio_token_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mirrors prepare_inputs_for_generation() but keeps the computation graph
    intact so that gradients flow back through audio_adapter.

    The key difference from the inference path:
      • audio_encoder() takes a *list* of 1-D wav tensors per segment.
        During training we have one segment per sample in the batch, so we
        process each sample individually and stack the results.
      • We need separate per-sample adapter outputs because each sample has a
        different number of <|cAUDIO|> tokens.

    Returns
    -------
    inputs_embeds  : (B, T, D_llm)  — ready for model.forward()
    audio_embeds   : (B, T, D_llm)  — same tensor; audio positions filled,
                                       text positions are the unmodified LM
                                       embeddings.  Alias returned for the
                                       alignment loss to index into without
                                       re-allocating.
    """
    device = input_ids.device
    B      = input_ids.shape[0]

    # ── Step 1: token embeddings for the whole sequence ───────────────────────
    lm_embed_fn = model.llm.get_input_embeddings()
    inputs_embeds = lm_embed_fn(input_ids)        # (B, T, D)

    # ── Step 2: audio features per sample ─────────────────────────────────────
    # audio_encoder() expects a list of 1-D wav tensors (one per segment).
    # For training we always have a single 30-s-or-less segment per sample.
    resampler = torch.nn.functional.interpolate    # not used; we call the method

    for b in range(B):
        single_wav = [wav[b]]                      # list of 1 tensor, as the model expects
        # audio_encoder resamples 24k→16k, computes mel, runs Whisper + adapter
        audio_feats = model.audio_encoder(single_wav, device)   # (1, T_audio, D)
        audio_feats = audio_feats.squeeze(0)       # (T_audio, D)

        # Positions in this sample's input_ids that are <|cAUDIO|>
        cAUDIO_mask = (input_ids[b] == audio_token_index)       # (T,)  bool
        n_audio_pos = cAUDIO_mask.sum().item()

        if n_audio_pos == 0:
            continue

        # Trim/pad features to match the number of placeholder tokens
        # (should always match; guard against off-by-one in streaming edge cases)
        n_feats = audio_feats.shape[0]
        if n_feats > n_audio_pos:
            audio_feats = audio_feats[:n_audio_pos]
        elif n_feats < n_audio_pos:
            pad = torch.zeros(
                n_audio_pos - n_feats, audio_feats.shape[-1],
                device=device, dtype=audio_feats.dtype,
            )
            audio_feats = torch.cat([audio_feats, pad], dim=0)

        # Scatter into inputs_embeds at <|cAUDIO|> positions
        inputs_embeds[b][cAUDIO_mask] = audio_feats.to(inputs_embeds.dtype)

    return inputs_embeds


# ─────────────────────────────────────────────────────────────────────────────
# Alignment loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_alignment_loss(
    inputs_embeds:      torch.Tensor,   # (B, T, D)  — already has audio feats
    input_ids:          torch.Tensor,   # (B, T)
    labels:             torch.Tensor,   # (B, T)     — -100 on prompt
    lm_embed_fn:        nn.Embedding,   # weight (V, D)
    audio_token_index:  int,
) -> torch.Tensor:
    """
    Cosine-embedding alignment loss.

    For each <|cAUDIO|> position p in the batch:
      • adapter_vec  = inputs_embeds[b, p, :]
      • next_label   = labels[b, p+1]   (the first real transcript token)
      • target_vec   = lm_embed(next_label)   [no grad]

    We only include positions where next_label ≥ 0 (not masked).

    Loss = mean(1 - cos(adapter_vec, target_vec))
         = cosine_embedding_loss(..., target=+1)

    Intuition: pushes the adapter's audio representation toward the LLM's own
    token embedding manifold, reducing the modality gap without any extra
    parameters.
    """
    # Audio positions: (B, T) bool
    audio_mask = (input_ids == audio_token_index)

    # Next-token labels: shift left by 1, fill last column with -100
    next_labels = torch.full_like(labels, -100)
    next_labels[:, :-1] = labels[:, 1:]

    # Only positions that are audio AND have a valid next label
    valid = audio_mask & (next_labels >= 0)     # (B, T)

    if not valid.any():
        return inputs_embeds.new_tensor(0.0)

    adapter_vecs = inputs_embeds[valid]          # (N, D)  — has grad
    target_ids   = next_labels[valid]            # (N,)

    with torch.no_grad():
        target_vecs = lm_embed_fn(target_ids)   # (N, D)  — no grad

    ones = adapter_vecs.new_ones(adapter_vecs.shape[0])
    return F.cosine_embedding_loss(adapter_vecs, target_vecs, ones,
                                   margin=0.0, reduction="mean")


# ─────────────────────────────────────────────────────────────────────────────
# Epoch-level cosine LR schedule with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

class EpochWarmupCosineScheduler:
    """
    Sets the learning rate at the START of each epoch:

      epochs 0 … warmup_end-1  :  linear ramp   0 → base_lr
      epochs warmup_end … N-1  :  cosine decay  base_lr → base_lr * lr_min_frac

    Call scheduler.step(epoch) before any optimizer.step() calls in that epoch.

    Args:
        optimizer   : PyTorch optimizer.
        num_epochs  : Total training epochs.
        warmup_frac : Fraction of epochs for warmup (default 0.5 → 50 %).
        lr_min_frac : LR floor as fraction of peak   (default 0.01 →  1 %).
    """

    def __init__(
        self,
        optimizer:   torch.optim.Optimizer,
        num_epochs:  int,
        warmup_frac: float = 0.5,
        lr_min_frac: float = 0.01,
    ):
        self.optimizer   = optimizer
        self.num_epochs  = num_epochs
        self.warmup_end  = max(1, math.floor(warmup_frac * num_epochs))
        self.lr_min_frac = lr_min_frac
        self.base_lrs    = [pg["lr"] for pg in optimizer.param_groups]

    def _lr(self, epoch: int, base_lr: float) -> float:
        lr_min = base_lr * self.lr_min_frac
        if epoch < self.warmup_end:
            # epoch=0 → base_lr/warmup_end  (never zero — avoids dead first step)
            return base_lr * (epoch + 1) / self.warmup_end
        progress = (epoch - self.warmup_end) / max(1, self.num_epochs - self.warmup_end - 1)
        return lr_min + (base_lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, epoch: int):
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self._lr(epoch, base)

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# One epoch of training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:             CovoAudioForCausalLM,
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
    model.train()

    total_ce = total_align = total = 0.0
    steps = 0
    optimizer.zero_grad()

    lm_embed_fn = model.llm.get_input_embeddings()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)       # (B, T)
        labels         = batch["labels"].to(device)          # (B, T)
        attention_mask = batch["attention_mask"].to(device)  # (B, T)
        wav            = batch["wav"].to(device)             # (B, T_wav)

        # ── Build inputs_embeds with audio features scattered in ──────────────
        # This is the differentiable path — gradients flow through audio_adapter.
        inputs_embeds = build_inputs_embeds(
            model, input_ids, wav, audio_token_index,
        )   # (B, T, D)

        # ── Alignment loss (before LM forward — no extra memory from LM) ──────
        align_loss = torch.tensor(0.0, device=device)
        if alignment_weight > 0.0:
            align_loss = compute_alignment_loss(
                inputs_embeds, input_ids, labels, lm_embed_fn, audio_token_index,
            )

        # ── LM forward (cross-entropy) ─────────────────────────────────────────
        # Pass inputs_embeds, NOT input_ids — the model's forward() uses
        # inputs_embeds directly and passes it straight to self.llm.
        lm_out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        ce_loss = lm_out.loss

        loss = ce_loss + alignment_weight * align_loss

        # ── Backward ──────────────────────────────────────────────────────────
        (loss / grad_accum).backward()

        total_ce    += ce_loss.item()
        total_align += align_loss.item()
        total       += loss.item()
        steps       += 1

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % log_every == 0:
            print(
                f"  [epoch {epoch+1}  step {step+1:>6}] "
                f"ce={total_ce/steps:.4f}  "
                f"align={total_align/steps:.4f}  "
                f"total={total/steps:.4f}"
            )

    # Flush any remaining accumulation buffer
    if steps % grad_accum != 0:
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad()

    n = max(steps, 1)
    return {"ce_loss": total_ce/n, "align_loss": total_align/n, "total_loss": total/n}


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:             CovoAudioForCausalLM,
    loader:            DataLoader,
    device:            torch.device,
    alignment_weight:  float,
    audio_token_index: int,
) -> dict:
    model.eval()
    total_ce = total_align = total = 0.0
    n = 0

    lm_embed_fn = model.llm.get_input_embeddings()

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        wav            = batch["wav"].to(device)

        # During validation no gradient is needed, but we still build embeds
        # the same way for consistency.
        inputs_embeds = build_inputs_embeds(
            model, input_ids, wav, audio_token_index,
        )

        align_loss = torch.tensor(0.0, device=device)
        if alignment_weight > 0.0:
            align_loss = compute_alignment_loss(
                inputs_embeds, input_ids, labels, lm_embed_fn, audio_token_index,
            )

        lm_out  = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        ce_loss = lm_out.loss

        total_ce    += ce_loss.item()
        total_align += align_loss.item()
        total       += (ce_loss + alignment_weight * align_loss).item()
        n           += 1

    n = max(n, 1)
    return {"val_ce": total_ce/n, "val_align": total_align/n, "val_loss": total/n}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-1 adapter training: CovoAudioForCausalLM"
    )
    p.add_argument("--model_name_or_path", default=None,
                   help="Pretrained checkpoint; if None, init from CovoAudioConfig defaults.")
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--num_epochs",       type=int,   default=10)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--grad_accum",       type=int,   default=4)
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--weight_decay",     type=float, default=0.01)
    p.add_argument("--max_grad_norm",    type=float, default=1.0)
    p.add_argument("--warmup_frac",      type=float, default=0.5,
                   help="Fraction of epochs for linear LR warmup (default=0.5 → 50%%).")
    p.add_argument("--lr_min_frac",      type=float, default=0.01,
                   help="LR floor as fraction of peak (default=0.01 → 1%%).")
    p.add_argument("--alignment_weight", type=float, default=None,
                   help="Override config.alignment_loss_weight.")
    p.add_argument("--extra_data",       action="store_true")
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--log_every",        type=int,   default=50)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--dtype",            default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_src = args.model_name_or_path or "Qwen/Qwen2.5-3B-Instruct"
    print(f"[init] tokenizer ← {tok_src}")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src, trust_remote_code=True, padding_side="right",
    )

    # Add audio special tokens if missing
    audio_specials = ["<|begofcAUDIO|>", "<|cAUDIO|>", "<|endofcAUDIO|>"]
    new_toks = [t for t in audio_specials if t not in tokenizer.get_vocab()]
    if new_toks:
        tokenizer.add_special_tokens({"additional_special_tokens": new_toks})
        print(f"[init] added {len(new_toks)} audio special tokens → "
              f"vocab size now {len(tokenizer)}")

    audio_token_index = tokenizer.convert_tokens_to_ids("<|cAUDIO|>")
    print(f"[init] <|cAUDIO|> id = {audio_token_index}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model_name_or_path:
        print(f"[init] model ← {args.model_name_or_path}")
        model = CovoAudioForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype,
        )
    else:
        print("[init] model ← CovoAudioConfig defaults")
        config = CovoAudioConfig()
        model  = CovoAudioForCausalLM(config)

    # Resize LM embeddings to match tokenizer (covers the audio special tokens)
    model.llm.resize_token_embeddings(len(tokenizer))
    model = model.to(device=device, dtype=dtype)

    alignment_weight = (
        args.alignment_weight
        if args.alignment_weight is not None
        else model.config.alignment_loss_weight
    )
    print(f"[init] alignment_loss_weight = {alignment_weight}")

    # ── Freeze: Stage 1 trains audio_adapter only ─────────────────────────────
    # Attribute names from modeling_covo_audio.py:
    #   self.llm           → Qwen2ForCausalLM   (frozen)
    #   self.encoder       → WhisperEncoder     (frozen)
    #   self.audio_adapter → AudioAdapter       (trainable)
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("audio_adapter.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable {trainable:,} / {total_p:,} "
          f"({100*trainable/total_p:.2f} %)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("[data] building train dataset …")
    train_ds = LibriSpeechStreamingDataset(
        tokenizer=tokenizer,
        audio_token_index=audio_token_index,
        use_extra_data=args.extra_data,
        seed=args.seed,
    )
    print("[data] building val dataset …")
    val_ds = LibriSpeechValDataset(
        tokenizer=tokenizer,
        audio_token_index=audio_token_index,
    )

    collator     = AudioSpeechCollator(pad_token_id=tokenizer.pad_token_id or 0)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        collate_fn=collator, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        collate_fn=collator, num_workers=2, pin_memory=True, shuffle=False,
    )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
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

    we = scheduler.warmup_end
    print(
        f"[schedule] {args.num_epochs} epochs | "
        f"warmup {we} epochs ({100*we/args.num_epochs:.0f} %) | "
        f"cosine decay epochs {we}–{args.num_epochs-1}"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val = float("inf")

    for epoch in range(args.num_epochs):
        scheduler.step(epoch)
        lr_now = scheduler.get_last_lr()[0]
        print(f"\n{'='*70}\nEpoch {epoch+1}/{args.num_epochs}  lr={lr_now:.3e}\n{'='*70}")

        t0 = time.time()
        tr = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            device=device, epoch=epoch, grad_accum=args.grad_accum,
            alignment_weight=alignment_weight, audio_token_index=audio_token_index,
            max_grad_norm=args.max_grad_norm, log_every=args.log_every,
        )
        elapsed = time.time() - t0
        print(f"\n[epoch {epoch+1}] train  "
              f"ce={tr['ce_loss']:.4f}  align={tr['align_loss']:.4f}  "
              f"total={tr['total_loss']:.4f}  ({elapsed/60:.1f} min)")

        vl = validate(
            model=model, loader=val_loader, device=device,
            alignment_weight=alignment_weight, audio_token_index=audio_token_index,
        )
        print(f"[epoch {epoch+1}] val    "
              f"ce={vl['val_ce']:.4f}  align={vl['val_align']:.4f}  "
              f"total={vl['val_loss']:.4f}")

        # Periodic checkpoint
        if args.save_every_epoch:
            ep_dir = out / f"epoch_{epoch+1:03d}"
            model.save_pretrained(ep_dir)
            tokenizer.save_pretrained(ep_dir)
            print(f"[ckpt] epoch checkpoint → {ep_dir}")

        # Best-val checkpoint
        if vl["val_loss"] < best_val:
            best_val = vl["val_loss"]
            best_dir = out / "best"
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"[ckpt] best val_loss={best_val:.4f} → {best_dir}")

    # Final checkpoint
    final_dir = out / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[done] final → {final_dir}  |  best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
