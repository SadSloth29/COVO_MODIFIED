"""
Stage 1 Adapter Training — CovoAudioForCausalLM (Qwen2.5-3B + Whisper-Small)
==============================================================================

Dual loss
─────────
    L = L_ce  +  λ · L_align

L_align: cosine-embedding loss between every adapter output vector at a
<|cAUDIO|> position and the LLM embedding of the first real transcript token
of that sample.

Why NOT "next_labels shift-left"
─────────────────────────────────
The audio placeholder sits entirely inside the prompt, which is label-masked
(-100).  Shifting labels left by 1 still gives -100 at every audio position,
so the valid mask is always empty → align_loss = 0 always.

Correct approach
─────────────────
For each sample b:
  1. Find first_resp_pos = first index where labels[b] >= 0.
  2. Collect all <|cAUDIO|> positions < first_resp_pos.
  3. Pair every audio vector with lm_embed(labels[b, first_resp_pos]).
     This is the first transcript token — the strongest cross-modal anchor.

Gradient note
─────────────
  inputs_embeds is built via:
    base = lm_embed(input_ids)          ← leaf, no grad through frozen embed
    embeds = base.clone()               ← clone breaks in-place aliasing
    embeds[b][audio_mask] = adapter_out ← index-assign on clone: grad flows
  Boolean-index assignment on a clone preserves autograd correctly.
  In-place assignment on the original leaf tensor (base[b][mask] = x) breaks
  autograd and was the second bug in the previous version.
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
    """Right-pads a batch from build_sample() to equal length."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_seq = max(b["input_ids"].shape[0] for b in batch)
        max_wav = max(b["wav"].shape[0]        for b in batch)

        ids_l, lbl_l, attn_l, wav_l = [], [], [], []
        for b in batch:
            ps = max_seq - b["input_ids"].shape[0]
            pw = max_wav - b["wav"].shape[0]
            ids_l.append( F.pad(b["input_ids"],      (0, ps), value=self.pad_token_id))
            lbl_l.append( F.pad(b["labels"],         (0, ps), value=-100))
            attn_l.append(F.pad(b["attention_mask"], (0, ps), value=0))
            wav_l.append( F.pad(b["wav"],            (0, pw), value=0.0))

        return {
            "input_ids":      torch.stack(ids_l),
            "labels":         torch.stack(lbl_l),
            "attention_mask": torch.stack(attn_l),
            "wav":            torch.stack(wav_l),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _diag_batch(batch: dict, tokenizer, audio_token_index: int, step: int):
    """Print a human-readable view of the first sample in the batch."""
    ids    = batch["input_ids"][0]
    labels = batch["labels"][0]
    wav    = batch["wav"][0]

    n_audio = (ids == audio_token_index).sum().item()
    # First unmasked label position
    resp_positions = (labels >= 0).nonzero(as_tuple=True)[0]
    first_resp_pos = resp_positions[0].item() if len(resp_positions) else -1
    first_resp_tok = labels[first_resp_pos].item() if first_resp_pos >= 0 else -1

    print(f"\n  ╔══ [diag step {step}] batch[0] ══")
    print(f"  ║  seq_len        : {ids.shape[0]}")
    print(f"  ║  wav_len        : {wav.shape[0]}  ({wav.shape[0]/24000:.2f}s @ 24kHz)")
    print(f"  ║  n_audio_tokens : {n_audio}  (id={audio_token_index})")
    print(f"  ║  first_resp_pos : {first_resp_pos}  (first unmasked label)")
    print(f"  ║  first_resp_tok : {first_resp_tok}  → '{tokenizer.decode([first_resp_tok]) if first_resp_tok>=0 else 'N/A'}'")
    # Show a window around the audio→response boundary
    if first_resp_pos >= 0:
        lo = max(0, first_resp_pos - 4)
        hi = min(ids.shape[0], first_resp_pos + 6)
        tok_slice = ids[lo:hi].tolist()
        lbl_slice = labels[lo:hi].tolist()
        decoded   = [tokenizer.decode([t]) for t in tok_slice]
        print(f"  ║  boundary window [{lo}:{hi}]:")
        print(f"  ║    input_ids : {tok_slice}")
        print(f"  ║    labels    : {lbl_slice}")
        print(f"  ║    decoded   : {decoded}")
    print(f"  ╚══")


def _diag_alignment(
    inputs_embeds:     torch.Tensor,
    input_ids:         torch.Tensor,
    labels:            torch.Tensor,
    lm_embed_fn:       nn.Embedding,
    audio_token_index: int,
    step:              int,
):
    """Print alignment loss diagnostics: how many valid pairs, cosine stats."""
    B = input_ids.shape[0]
    n_pairs = 0
    cos_vals = []
    for b in range(B):
        resp_pos = (labels[b] >= 0).nonzero(as_tuple=True)[0]
        if len(resp_pos) == 0:
            continue
        frp  = resp_pos[0].item()
        ftok = labels[b, frp].item()
        audio_pos = (input_ids[b, :frp] == audio_token_index).nonzero(as_tuple=True)[0]
        if len(audio_pos) == 0:
            continue
        n_pairs += len(audio_pos)
        with torch.no_grad():
            av = inputs_embeds[b, audio_pos]                     # (N, D)
            tv = lm_embed_fn(torch.tensor([ftok], device=input_ids.device))  # (1, D)
            tv = tv.expand(av.shape[0], -1)
            cos = F.cosine_similarity(av.float(), tv.float(), dim=-1)
            cos_vals.extend(cos.tolist())

    if cos_vals:
        print(f"\n  ╔══ [diag step {step}] alignment ══")
        print(f"  ║  valid audio-response pairs : {n_pairs}")
        print(f"  ║  cosine similarity  min={min(cos_vals):.4f}  "
              f"mean={sum(cos_vals)/len(cos_vals):.4f}  max={max(cos_vals):.4f}")
        print(f"  ╚══")
    else:
        print(f"\n  [diag step {step}] alignment: NO valid pairs found — check audio_token_index and label masking!")


# ─────────────────────────────────────────────────────────────────────────────
# inputs_embeds builder  (differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs_embeds(
    model:             CovoAudioForCausalLM,
    input_ids:         torch.Tensor,   # (B, T)
    wav:               torch.Tensor,   # (B, T_wav)
    audio_token_index: int,
) -> torch.Tensor:
    """
    Build inputs_embeds with audio adapter outputs scattered at <|cAUDIO|>
    positions.  Gradients flow through audio_adapter.

    Critical: we clone() the base embeddings before index-assigning so that
    autograd sees a non-leaf output tensor.  In-place assignment on the
    original leaf (lm_embed output) would silently break grad flow.
    """
    device = input_ids.device
    B      = input_ids.shape[0]

    lm_embed_fn   = model.llm.get_input_embeddings()
    base_embeds   = lm_embed_fn(input_ids)        # (B, T, D) — frozen, no grad
    inputs_embeds = base_embeds.clone()            # clone → grad can flow through

    for b in range(B):
        single_wav  = [wav[b]]
        audio_feats = model.audio_encoder(single_wav, device)  # (1, T_a, D)
        audio_feats = audio_feats.squeeze(0)                   # (T_a, D)

        cAUDIO_mask = (input_ids[b] == audio_token_index)      # (T,) bool
        n_pos       = cAUDIO_mask.sum().item()

        if n_pos == 0:
            continue

        # Guard: trim or zero-pad to match placeholder count
        n_feats = audio_feats.shape[0]
        if n_feats > n_pos:
            audio_feats = audio_feats[:n_pos]
        elif n_feats < n_pos:
            pad = inputs_embeds.new_zeros(n_pos - n_feats, audio_feats.shape[-1])
            audio_feats = torch.cat([audio_feats, pad], dim=0)

        # Index-assign on clone — autograd-safe
        inputs_embeds[b][cAUDIO_mask] = audio_feats.to(inputs_embeds.dtype)

    return inputs_embeds


# ─────────────────────────────────────────────────────────────────────────────
# Alignment loss  (fixed)
# ─────────────────────────────────────────────────────────────────────────────

def compute_alignment_loss(
    inputs_embeds:     torch.Tensor,   # (B, T, D)
    input_ids:         torch.Tensor,   # (B, T)
    labels:            torch.Tensor,   # (B, T)  — -100 on prompt
    lm_embed_fn:       nn.Embedding,
    audio_token_index: int,
) -> torch.Tensor:
    """
    Cosine-embedding alignment loss.

    For each sample b:
      • first_resp_pos  = first index where labels[b] >= 0
        (= first token of the transcript, right after <|im_start|>assistant\\n)
      • first_resp_tok  = labels[b, first_resp_pos]
      • audio_positions = all <|cAUDIO|> indices < first_resp_pos
      • For every audio position p:
            adapter_vec  = inputs_embeds[b, p]
            target_vec   = lm_embed(first_resp_tok)   [no grad]
            contribution = 1 - cos(adapter_vec, target_vec)

    Why the first response token?
      All audio tokens are in the prompt (labels=-100), so "next valid label"
      after any audio position is always the first transcript token.  This is
      the strongest cross-modal anchor we have at Stage 1.

    Returns scalar loss (0 if no valid pairs in batch).
    """
    B = input_ids.shape[0]
    adapter_vecs_list = []
    target_vecs_list  = []

    for b in range(B):
        # First supervised position in this sample
        resp_mask = (labels[b] >= 0)
        if not resp_mask.any():
            continue
        first_resp_pos = resp_mask.nonzero(as_tuple=True)[0][0].item()
        first_resp_tok = labels[b, first_resp_pos].item()

        # All <|cAUDIO|> positions before the response starts
        audio_positions = (input_ids[b, :first_resp_pos] == audio_token_index) \
                          .nonzero(as_tuple=True)[0]

        if len(audio_positions) == 0:
            continue

        av = inputs_embeds[b, audio_positions]                         # (N, D)
        with torch.no_grad():
            tv = lm_embed_fn(
                torch.tensor([first_resp_tok], device=input_ids.device)
            )                                                           # (1, D)
            tv = tv.expand(av.shape[0], -1)                            # (N, D)

        adapter_vecs_list.append(av)
        target_vecs_list.append(tv)

    if not adapter_vecs_list:
        return inputs_embeds.new_tensor(0.0)

    all_adapter = torch.cat(adapter_vecs_list, dim=0)   # (N_total, D)
    all_target  = torch.cat(target_vecs_list,  dim=0)   # (N_total, D)
    ones        = all_adapter.new_ones(all_adapter.shape[0])

    return F.cosine_embedding_loss(
        all_adapter.float(), all_target.float(), ones,
        margin=0.0, reduction="mean",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Epoch-level cosine LR schedule with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

class EpochWarmupCosineScheduler:
    """
    LR policy set at epoch granularity:
      epochs [0, warmup_end)    : linear ramp  0 → base_lr
      epochs [warmup_end, N)    : cosine decay base_lr → base_lr * lr_min_frac

    Call scheduler.step(epoch) at the START of each epoch.
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
            return base_lr * (epoch + 1) / self.warmup_end
        progress = (epoch - self.warmup_end) / max(1, self.num_epochs - self.warmup_end - 1)
        return lr_min + (base_lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, epoch: int):
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self._lr(epoch, base)

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
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
    tokenizer,
    max_grad_norm:     float = 1.0,
    log_every:         int   = 50,
    diag_every:        int   = 200,
) -> dict:
    model.train()

    total_ce = total_align = total = 0.0
    steps = 0
    optimizer.zero_grad()

    lm_embed_fn = model.llm.get_input_embeddings()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        wav            = batch["wav"].to(device)

        # ── Detailed batch diagnostics (first step + every diag_every steps) ──
        is_diag = (step == 0) or ((step + 1) % diag_every == 0)
        if is_diag:
            _diag_batch(batch, tokenizer, audio_token_index, step + 1)

        # ── Build inputs_embeds (grad flows through adapter) ──────────────────
        inputs_embeds = build_inputs_embeds(model, input_ids, wav, audio_token_index)

        # ── Alignment loss ────────────────────────────────────────────────────
        align_loss = inputs_embeds.new_tensor(0.0)
        if alignment_weight > 0.0:
            align_loss = compute_alignment_loss(
                inputs_embeds, input_ids, labels, lm_embed_fn, audio_token_index,
            )
            if is_diag:
                _diag_alignment(
                    inputs_embeds, input_ids, labels,
                    lm_embed_fn, audio_token_index, step + 1,
                )

        # ── LM forward ───────────────────────────────────────────────────────
        lm_out  = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        ce_loss = lm_out.loss

        loss = ce_loss + alignment_weight * align_loss

        # Sanity-print on first step
        if step == 0:
            print(f"\n  [diag step 1] ce_loss={ce_loss.item():.4f}  "
                  f"align_loss={align_loss.item():.4f}  "
                  f"align_weight={alignment_weight}  "
                  f"total={loss.item():.4f}")
            print(f"  [diag step 1] inputs_embeds.requires_grad={inputs_embeds.requires_grad}")

        # ── Backward ─────────────────────────────────────────────────────────
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

    # Flush remaining accum buffer
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

        inputs_embeds = build_inputs_embeds(model, input_ids, wav, audio_token_index)

        align_loss = inputs_embeds.new_tensor(0.0)
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
    p = argparse.ArgumentParser(description="Stage-1 adapter training: CovoAudioForCausalLM")
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--output_dir",         required=True)
    p.add_argument("--num_epochs",         type=int,   default=10)
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--grad_accum",         type=int,   default=4)
    p.add_argument("--lr",                 type=float, default=2e-4)
    p.add_argument("--weight_decay",       type=float, default=0.01)
    p.add_argument("--max_grad_norm",      type=float, default=1.0)
    p.add_argument("--warmup_frac",        type=float, default=0.1,
                   help="Fraction of epochs for linear warmup (default 0.5 = 50%%).")
    p.add_argument("--lr_min_frac",        type=float, default=0.01,
                   help="LR floor as fraction of peak (default 0.01 = 1%%).")
    p.add_argument("--alignment_weight",   type=float, default=0.0,
                   help="Override config.alignment_loss_weight.")
    p.add_argument("--extra_data",         action="store_true")
    p.add_argument("--num_workers",        type=int,   default=4)
    p.add_argument("--log_every",          type=int,   default=50)
    p.add_argument("--diag_every",         type=int,   default=200,
                   help="Print detailed diagnostics every N steps (default 200).")
    p.add_argument("--save_every_epoch",   action="store_true")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--dtype",              default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = {"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_src   = args.model_name_or_path or "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True, padding_side="right")

    audio_specials = ["<|begofcAUDIO|>", "<|cAUDIO|>", "<|endofcAUDIO|>"]
    new_toks = [t for t in audio_specials if t not in tokenizer.get_vocab()]
    if new_toks:
        tokenizer.add_special_tokens({"additional_special_tokens": new_toks})
        print(f"[init] added {len(new_toks)} audio special tokens → vocab {len(tokenizer)}")

    audio_token_index = tokenizer.convert_tokens_to_ids("<|cAUDIO|>")
    print(f"[init] <|cAUDIO|>      id = {audio_token_index}")
    print(f"[init] <|begofcAUDIO|> id = {tokenizer.convert_tokens_to_ids('<|begofcAUDIO|>')}")
    print(f"[init] <|endofcAUDIO|> id = {tokenizer.convert_tokens_to_ids('<|endofcAUDIO|>')}")
    print(f"[init] <|im_end|>      id = {tokenizer.convert_tokens_to_ids('<|im_end|>')}")

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model_name_or_path:
        model = CovoAudioForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
    else:
        print("[init] model ← CovoAudioConfig defaults")
        model = CovoAudioForCausalLM(CovoAudioConfig())

    model.llm.resize_token_embeddings(len(tokenizer))
    model = model.to(device=device, dtype=dtype)

    alignment_weight = (
        args.alignment_weight
        if args.alignment_weight is not None
        else model.config.alignment_loss_weight
    )
    print(f"[init] alignment_loss_weight = {alignment_weight}")

    # ── Freeze: only audio_adapter is trainable ───────────────────────────────
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("audio_adapter.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f} %)")

    # Quick sanity-check: make sure audio_adapter params are found
    adapter_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[init] trainable param groups: {adapter_params[:5]} {'...' if len(adapter_params)>5 else ''}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = LibriSpeechStreamingDataset(
        tokenizer=tokenizer, audio_token_index=audio_token_index,
        use_extra_data=args.extra_data, seed=args.seed,
    )
    val_ds = LibriSpeechValDataset(tokenizer=tokenizer, audio_token_index=audio_token_index)

    collator     = AudioSpeechCollator(pad_token_id=tokenizer.pad_token_id or 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              collate_fn=collator, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              collate_fn=collator, num_workers=2, pin_memory=True, shuffle=False)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8,
    )
    scheduler = EpochWarmupCosineScheduler(
        optimizer=optimizer, num_epochs=args.num_epochs,
        warmup_frac=args.warmup_frac, lr_min_frac=args.lr_min_frac,
    )
    we = scheduler.warmup_end
    print(f"[schedule] {args.num_epochs} epochs | warmup {we} ({100*we/args.num_epochs:.0f}%) | "
          f"cosine decay {we}–{args.num_epochs-1}")

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
            tokenizer=tokenizer,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every,
            diag_every=args.diag_every,
        )
        print(f"\n[epoch {epoch+1}] train  ce={tr['ce_loss']:.4f}  "
              f"align={tr['align_loss']:.4f}  total={tr['total_loss']:.4f}  "
              f"({(time.time()-t0)/60:.1f} min)")

        vl = validate(model=model, loader=val_loader, device=device,
                      alignment_weight=alignment_weight, audio_token_index=audio_token_index)
        print(f"[epoch {epoch+1}] val    ce={vl['val_ce']:.4f}  "
              f"align={vl['val_align']:.4f}  total={vl['val_loss']:.4f}")

        if args.save_every_epoch:
            ep_dir = out / f"epoch_{epoch+1:03d}"
            model.save_pretrained(ep_dir); tokenizer.save_pretrained(ep_dir)
            print(f"[ckpt] → {ep_dir}")

        if vl["val_loss"] < best_val:
            best_val = vl["val_loss"]
            best_dir = out / "best"
            model.save_pretrained(best_dir); tokenizer.save_pretrained(best_dir)
            print(f"[ckpt] best val_loss={best_val:.4f} → {best_dir}")

    final_dir = out / "final"
    model.save_pretrained(final_dir); tokenizer.save_pretrained(final_dir)
    print(f"\n[done] final → {final_dir}  |  best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
