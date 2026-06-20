"""
Stage 1 Adapter Training — CovoAudioForCausalLM (Qwen2.5-3B + Whisper-Small)
==============================================================================

Loss: standard ASR cross-entropy only.

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
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2ForCausalLM, WhisperForConditionalGeneration

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
    resp_positions = (labels >= 0).nonzero(as_tuple=True)[0]
    first_resp_pos = resp_positions[0].item() if len(resp_positions) else -1
    first_resp_tok = labels[first_resp_pos].item() if first_resp_pos >= 0 else -1

    print(f"\n  ╔══ [diag step {step}] batch[0] ══")
    print(f"  ║  seq_len        : {ids.shape[0]}")
    print(f"  ║  wav_len        : {wav.shape[0]}  ({wav.shape[0]/24000:.2f}s @ 24kHz)")
    print(f"  ║  n_audio_tokens : {n_audio}  (id={audio_token_index})")
    print(f"  ║  first_resp_pos : {first_resp_pos}  (first unmasked label)")
    print(f"  ║  first_resp_tok : {first_resp_tok}  → "
          f"'{tokenizer.decode([first_resp_tok]) if first_resp_tok>=0 else 'N/A'}'")
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
    # Decode the full transcript portion for a sanity read
    transcript_ids = [t for t in labels.tolist() if t >= 0]
    if transcript_ids:
        print(f"  ║  full target  : '{tokenizer.decode(transcript_ids)}'")
    print(f"  ╚══")


def _diag_predictions(lm_out, input_ids, labels, tokenizer, step: int):
    """Show what the model is currently predicting vs the ground truth."""
    with torch.no_grad():
        logits = lm_out.logits[0]            # (T, V) — first sample
        preds  = logits.argmax(dim=-1)       # (T,)

        resp_positions = (labels[0] >= 0).nonzero(as_tuple=True)[0]
        if len(resp_positions) == 0:
            return
        # Predictions are shifted: logits[t] predicts token[t+1]
        pred_window = []
        true_window = []
        for pos in resp_positions[:8]:
            p = pos.item()
            pred_tok = preds[p - 1].item() if p > 0 else -1
            true_tok = labels[0, p].item()
            pred_window.append(pred_tok)
            true_window.append(true_tok)

        pred_str = tokenizer.decode([t for t in pred_window if t >= 0])
        true_str = tokenizer.decode([t for t in true_window if t >= 0])
        print(f"\n  ╔══ [diag step {step}] prediction sample ══")
        print(f"  ║  ground truth : '{true_str}'")
        print(f"  ║  model predicts: '{pred_str}'")
        print(f"  ╚══")


# ─────────────────────────────────────────────────────────────────────────────
# inputs_embeds builder  (differentiable)
# ─────────────────────────────────────────────────────────────────────────────

def build_inputs_embeds(
    model:             CovoAudioForCausalLM,
    input_ids:         torch.Tensor,   # (B, T)
    wav:               torch.Tensor,   # (B, T_wav)
    audio_token_index: int,
    _warn_state:       dict = {"warned": False},   # mutable default = one-time latch
) -> torch.Tensor:
    """
    Build inputs_embeds with audio adapter outputs scattered at <|cAUDIO|>
    positions. Gradients flow through audio_adapter.

    We clone() the base embeddings before index-assigning so autograd sees a
    non-leaf output tensor. In-place assignment on the original leaf (the
    lm_embed output) would silently break grad flow into audio_adapter.

    Mismatch guard: if the adapter emits a different number of feature
    vectors than there are <|cAUDIO|> placeholders, we trim/pad to make
    shapes line up — but this previously happened SILENTLY every batch due
    to a stage-count bug in dataset.py's calc_seq_len (now fixed). A
    mismatch here should be rare (rounding only); if it's large or constant,
    dataset.py's adapter_downsample no longer matches
    model.config.adapter_downsample and needs to be re-synced.
    """
    device = input_ids.device
    B      = input_ids.shape[0]

    lm_embed_fn   = model.llm.get_input_embeddings()
    base_embeds   = lm_embed_fn(input_ids)        # (B, T, D) — frozen, no grad needed
    inputs_embeds = base_embeds.clone()            # clone → grad can flow through

    for b in range(B):
        single_wav  = [wav[b]]
        audio_feats = model.audio_encoder(single_wav, device)  # (1, T_a, D)
        audio_feats = audio_feats.squeeze(0)                   # (T_a, D)

        cAUDIO_mask = (input_ids[b] == audio_token_index)      # (T,) bool
        n_pos       = cAUDIO_mask.sum().item()

        if n_pos == 0:
            continue

        n_feats = audio_feats.shape[0]
        if n_feats != n_pos and not _warn_state["warned"]:
            gap = abs(n_feats - n_pos)
            print(
                f"\n  ⚠ [build_inputs_embeds] placeholder/feature count mismatch: "
                f"adapter emits {n_feats} vectors but input_ids has {n_pos} "
                f"<|cAUDIO|> tokens (gap={gap}). A gap of 0-1 is normal rounding; "
                f"a large or consistently nonzero gap means dataset.py's "
                f"adapter_downsample is out of sync with model.config.adapter_downsample "
                f"— check LibriSpeechStreamingDataset(adapter_downsample=...) matches "
                f"model.config.adapter_downsample. Truncating/padding to compensate; "
                f"this warning prints once."
            )
            _warn_state["warned"] = True

        if n_feats > n_pos:
            audio_feats = audio_feats[:n_pos]
        elif n_feats < n_pos:
            pad = inputs_embeds.new_zeros(n_pos - n_feats, audio_feats.shape[-1])
            audio_feats = torch.cat([audio_feats, pad], dim=0)

        inputs_embeds[b][cAUDIO_mask] = audio_feats.to(inputs_embeds.dtype)

    return inputs_embeds



# ─────────────────────────────────────────────────────────────────────────────
# Epoch-level warmup, continuously interpolated per step
# ─────────────────────────────────────────────────────────────────────────────

class EpochWarmupCosineScheduler:
    """
    LR envelope spans epochs exactly as before:
      virtual_epoch in [0, warmup_span)       : linear ramp  0 → base_lr
      virtual_epoch in [warmup_span, N)       : cosine decay base_lr → base_lr*lr_min_frac

    KEY FIX vs the previous version: LR is recomputed every optimizer step
    using a continuous "virtual epoch" coordinate

        virtual_epoch = epoch + (step_in_epoch / steps_per_epoch)

    instead of being frozen for the whole epoch. warmup_span = warmup_frac *
    num_epochs is now a float (not floor()'d), so the 50%-epoch contract is
    preserved exactly, but every step within that span gets its own LR.

    steps_per_epoch must be supplied (or estimated) since streamed datasets
    don't expose __len__. If unknown ahead of time, pass your best estimate;
    the schedule self-corrects across epoch boundaries since `epoch` is
    still the authoritative integer counter.
    """

    def __init__(
        self,
        optimizer:        torch.optim.Optimizer,
        num_epochs:        int,
        steps_per_epoch:    int,
        warmup_frac:       float = 0.5,
        lr_min_frac:       float = 0.01,
    ):
        self.optimizer       = optimizer
        self.num_epochs      = num_epochs
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.warmup_span     = warmup_frac * num_epochs   # float, NOT floored
        self.lr_min_frac     = lr_min_frac
        self.base_lrs        = [pg["lr"] for pg in optimizer.param_groups]

    def _lr(self, virtual_epoch: float, base_lr: float) -> float:
        lr_min = base_lr * self.lr_min_frac
        if virtual_epoch < self.warmup_span:
            if self.warmup_span <= 0:
                return base_lr
            return base_lr * (virtual_epoch / self.warmup_span)
        progress = (virtual_epoch - self.warmup_span) / max(
            1e-8, self.num_epochs - self.warmup_span
        )
        progress = min(progress, 1.0)
        return lr_min + (base_lr - lr_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, epoch: int, step_in_epoch: int):
        """Call this every optimizer step (or every micro-step), not just per-epoch."""
        virtual_epoch = epoch + (step_in_epoch / self.steps_per_epoch)
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self._lr(virtual_epoch, base)

    def get_last_lr(self) -> list[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:             CovoAudioForCausalLM,
    loader:            DataLoader,
    optimizer:         torch.optim.Optimizer,
    scheduler:         EpochWarmupCosineScheduler,
    device:            torch.device,
    epoch:             int,
    num_epochs:        int,
    grad_accum:        int,
    audio_token_index: int,
    tokenizer,
    max_grad_norm:     float = 1.0,
    log_every:         int   = 50,
    diag_every:        int   = 200,
    steps_per_epoch_hint: int = None,
) -> dict:
    model.train()

    total_ce = 0.0
    steps    = 0
    optimizer.zero_grad()

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{num_epochs}",
        unit="step",
        total=steps_per_epoch_hint,
        dynamic_ncols=True,
        leave=True,
    )

    for step, batch in enumerate(pbar):
        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        wav            = batch["wav"].to(device)

        # ── Step-level LR update — fixes the frozen-LR-per-epoch bug ──────────
        scheduler.step(epoch, step)

        is_diag = (step == 0) or ((step + 1) % diag_every == 0)
        if is_diag:
            _diag_batch(batch, tokenizer, audio_token_index, step + 1)

        inputs_embeds = build_inputs_embeds(model, input_ids, wav, audio_token_index)

        lm_out  = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        ce_loss = lm_out.loss

        if step == 0:
            print(f"\n  [diag step 1] ce_loss={ce_loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.3e}  "
                  f"inputs_embeds.requires_grad={inputs_embeds.requires_grad}")

        if is_diag:
            _diag_predictions(lm_out, input_ids, labels, tokenizer, step + 1)

        (ce_loss / grad_accum).backward()

        total_ce += ce_loss.item()
        steps    += 1

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix({
            "ce":  f"{total_ce/steps:.4f}",
            "lr":  f"{scheduler.get_last_lr()[0]:.2e}",
        })

        if (step + 1) % log_every == 0:
            print(
                f"  [epoch {epoch+1}  step {step+1:>6}] "
                f"ce={total_ce/steps:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.3e}"
            )

    if steps % grad_accum != 0:
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad()

    n = max(steps, 1)
    return {"ce_loss": total_ce / n, "steps": steps}


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:             CovoAudioForCausalLM,
    loader:            DataLoader,
    device:            torch.device,
    audio_token_index: int,
) -> dict:
    model.eval()
    total_ce = 0.0
    n = 0

    for batch in tqdm(loader, desc="Validating", unit="batch", dynamic_ncols=True, leave=False):
        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        wav            = batch["wav"].to(device)

        inputs_embeds = build_inputs_embeds(model, input_ids, wav, audio_token_index)
        lm_out  = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        ce_loss = lm_out.loss

        total_ce += ce_loss.item()
        n        += 1

    n = max(n, 1)
    return {"val_ce": total_ce / n}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 adapter training: CovoAudioForCausalLM")
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--output_dir",         required=True)
    p.add_argument("--num_epochs",         type=int,   default=10)
    p.add_argument("--steps_per_epoch",    type=int,   default=20000,
                   help="Estimated optimizer steps per epoch — REQUIRED for correct "
                        "continuous LR interpolation with a streamed dataset, since "
                        "streamed IterableDatasets don't expose __len__. Set this to "
                        "roughly (total_hours * 3600 / avg_clip_seconds / batch_size).")
    p.add_argument("--batch_size",         type=int,   default=4)
    p.add_argument("--grad_accum",         type=int,   default=4)
    p.add_argument("--lr",                 type=float, default=2e-4)
    p.add_argument("--weight_decay",       type=float, default=0.01)
    p.add_argument("--max_grad_norm",      type=float, default=1.0)
    p.add_argument("--warmup_frac",        type=float, default=0.5,
                   help="Fraction of epochs spanned by warmup (default 0.5 = 50%%).")
    p.add_argument("--lr_min_frac",        type=float, default=0.01,
                   help="LR floor as fraction of peak (default 0.01 = 1%%).")
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
    print(f"[init] <|cAUDIO|>      id = {audio_token_index}  (derived from tokenizer, not hardcoded)")
    print(f"[init] <|begofcAUDIO|> id = {tokenizer.convert_tokens_to_ids('<|begofcAUDIO|>')}")
    print(f"[init] <|endofcAUDIO|> id = {tokenizer.convert_tokens_to_ids('<|endofcAUDIO|>')}")
    print(f"[init] <|endoftext|>   id = {tokenizer.convert_tokens_to_ids('<|endoftext|>')}  "
          f"(a2t/ASR stop token)")
    print(f"[init] <|im_end|>      id = {tokenizer.convert_tokens_to_ids('<|im_end|>')}  "
          f"(a2ta stop token — NOT used to terminate ASR training labels)")

    
    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model_name_or_path:
        # Load pretrained base models
        base_llm = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
        base_whisper = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small", torch_dtype=dtype
        )
    
        
        covo_config = CovoAudioConfig(
            llm_config=base_llm.config,
            encoder_config=base_whisper.config,   
            audio_token_index=audio_token_index,
            adapter_downsample=CovoAudioConfig.adapter_downsample,  # default 8
        )
        model = CovoAudioForCausalLM(covo_config)
    
    
        model.llm.load_state_dict(base_llm.state_dict(), strict=False)
    
       
        model.audio_encoder.load_state_dict(base_whisper.state_dict(), strict=False)
    
        
        final_vocab_size = covo_config.vocab_size
        model.llm.resize_token_embeddings(final_vocab_size, mean_resizing=True)
    
    else:
        # No pretrained weights – use default config (already has expanded vocab)
        model = CovoAudioForCausalLM(CovoAudioConfig(audio_token_index=audio_token_index))
    
    
    model = model.to(device=device, dtype=dtype)

    # ── Freeze: only audio_adapter is trainable ───────────────────────────────
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("audio_adapter.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[init] trainable {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f} %)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    # adapter_downsample MUST come from the model's own config, never a
    # separate hardcoded default, so the number of <|cAUDIO|> placeholders
    # inserted here always matches what AudioAdapter actually produces.
    adapter_downsample = model.config.adapter_downsample
    print(f"[init] adapter_downsample = {adapter_downsample}  "
          f"(<|cAUDIO|> placeholder count derived from this)")

    train_ds = LibriSpeechStreamingDataset(
        tokenizer=tokenizer, audio_token_index=audio_token_index,
        use_extra_data=args.extra_data, seed=args.seed,
        adapter_downsample=adapter_downsample,
    )
    val_ds = LibriSpeechValDataset(
        tokenizer=tokenizer, audio_token_index=audio_token_index,
        adapter_downsample=adapter_downsample,
    )

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
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        warmup_frac=args.warmup_frac,
        lr_min_frac=args.lr_min_frac,
    )
    print(f"[schedule] {args.num_epochs} epochs | "
          f"warmup_span={args.warmup_frac*args.num_epochs:.1f} epochs "
          f"({args.warmup_frac*100:.0f}%) | "
          f"steps_per_epoch={args.steps_per_epoch} | "
          f"LR interpolated continuously every optimizer step")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val = float("inf")

    for epoch in range(args.num_epochs):
        print(f"\n{'='*70}\nEpoch {epoch+1}/{args.num_epochs}  "
              f"starting lr={scheduler.get_last_lr()[0]:.3e}\n{'='*70}")

        t0 = time.time()
        tr = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            scheduler=scheduler, device=device, epoch=epoch,
            num_epochs=args.num_epochs, grad_accum=args.grad_accum,
            audio_token_index=audio_token_index, tokenizer=tokenizer,
            max_grad_norm=args.max_grad_norm,
            log_every=args.log_every, diag_every=args.diag_every,
            steps_per_epoch_hint=args.steps_per_epoch,
        )
        print(f"\n[epoch {epoch+1}] train  ce={tr['ce_loss']:.4f}  "
              f"steps={tr['steps']}  ({(time.time()-t0)/60:.1f} min)")

        vl = validate(model=model, loader=val_loader, device=device,
                      audio_token_index=audio_token_index)
        print(f"[epoch {epoch+1}] val    ce={vl['val_ce']:.4f}")

        if args.save_every_epoch:
            ep_dir = out / f"epoch_{epoch+1:03d}"
            model.save_pretrained(ep_dir); tokenizer.save_pretrained(ep_dir)
            print(f"[ckpt] → {ep_dir}")

        if vl["val_ce"] < best_val:
            best_val = vl["val_ce"]
            best_dir = out / "best"
            model.save_pretrained(best_dir); tokenizer.save_pretrained(best_dir)
            print(f"[ckpt] best val_ce={best_val:.4f} → {best_dir}")

    final_dir = out / "final"
    model.save_pretrained(final_dir); tokenizer.save_pretrained(final_dir)
    print(f"\n[done] final → {final_dir}  |  best val_ce={best_val:.4f}")


if __name__ == "__main__":
    main()
