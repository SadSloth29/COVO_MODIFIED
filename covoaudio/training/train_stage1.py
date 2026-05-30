"""
Stage 1 Training — Adapter Alignment (LibriSpeech, Streaming)
==============================================================

What is frozen vs trained:
    FROZEN:  WhisperEncoder  (pretrained openai/whisper-medium)
    FROZEN:  Gemma 3 4B LLM  (pretrained google/gemma-3-4b-it)
    TRAINED: AudioAdapter    (~30M params — the only bridge between them)

Regularization applied:
    • Weight decay          (0.01 on all adapter params except biases/norms)
    • Gradient clipping     (max_norm=1.0)
    • Label smoothing       (0.1 — reduces overconfident transcript predictions)
    • Cosine LR w/ warmup   (OneCycleLR)
    • Dropout in adapter    (inject via wrapper, see AdapterWithDropout below)
    • Validation loss       (librispeech validation.clean, every eval_every steps)
    • Early stopping        (patience=5 eval rounds without improvement)

Step-based training (not epoch-based) because the dataset is streaming
and has no defined length. Use --max_steps to control total training.

Recommended for 960h LibriSpeech on 1×A100 80GB:
    --max_steps 200000 --batch_size 8 --grad_accum 4  → effective batch = 32
    Expected wall time: ~3-4 days

Usage:
    python training/train_stage1.py \\
        --gemma_ckpt     google/gemma-3-4b-it \\
        --whisper_ckpt   openai/whisper-medium \\
        --output_dir     checkpoints/stage1 \\
        --max_steps      200000 \\
        --batch_size     4 \\
        --grad_accum     8 \\
        --lr             3e-4 \\
        --warmup_steps   2000 \\
        --eval_every     2000 \\
        --save_every     5000

Quick smoke-test on first 500 steps:
    python training/train_stage1.py ... --max_steps 500 --eval_every 100
"""

import os
import sys
import math
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, WhisperModel
from transformers.models.gemma3 import Gemma3ForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling_covo_audio import CovoAudioForCausalLM, sequence_mask
from configuration_covo_audio import CovoAudioConfig
from training.dataset  import LibriSpeechStreamingDataset, LibriSpeechValDataset
from training.collator import AudioTextCollator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter dropout wrapper
# The original AudioAdapter/DownsampleLayer has no dropout.
# We inject it here without modifying the model file.
# ─────────────────────────────────────────────────────────────────────────────

class DropoutWrapper(nn.Module):
    """Wraps any module and applies dropout after its forward pass."""
    def __init__(self, module: nn.Module, p: float = 0.1):
        super().__init__()
        self.module  = module
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(self.module(x))

    # Delegate attribute lookup to the wrapped module so that
    # model.audio_adapter.downsample_layers[i].conv1d etc. still work
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def add_adapter_dropout(model: CovoAudioForCausalLM, p: float = 0.1):
    """Wraps each DownsampleLayer in the AudioAdapter with dropout."""
    layers = model.audio_adapter.downsample_layers
    for i in range(len(layers)):
        layers[i] = DropoutWrapper(layers[i], p=p)
    log.info(f"Added dropout p={p} to {len(layers)} adapter layers")


# ─────────────────────────────────────────────────────────────────────────────
# Audio → embedding injection  (training-time equivalent of
# prepare_inputs_for_generation's first-iteration path)
# ─────────────────────────────────────────────────────────────────────────────

def inject_audio(
    model:             CovoAudioForCausalLM,
    wavs:              list,
    input_ids:         torch.Tensor,
    audio_token_index: int,
) -> torch.Tensor:
    """
    Replaces <|cAUDIO|> placeholder embeddings with real audio features
    produced by the (frozen) WhisperEncoder + (trained) AudioAdapter.

    Returns inputs_embeds: (B, L, hidden_size) ready for model.forward().

    Gradient flow:
        WhisperEncoder  → no grad (frozen)
        AudioAdapter    → grad flows here ✓
        LLM embeddings  → no grad (frozen)
    """
    device = input_ids.device
    dtype  = next(model.llm.parameters()).dtype

    # Text-side: look up embeddings for all tokens (no grad needed for LLM)
    with torch.no_grad():
        inputs_embeds = model.llm.get_input_embeddings()(input_ids)   # (B, L, H)

    for b, wav in enumerate(wavs):
        n_placeholders = (input_ids[b] == audio_token_index).sum().item()
        if n_placeholders == 0:
            continue

        # audio_encoder calls WhisperEncoder (frozen) then AudioAdapter (trained)
        audio_feats = model.audio_encoder([wav.to(device)], device)   # (1, T, H)

        # Match length to placeholders (small off-by-one from padding)
        T = audio_feats.shape[1]
        if T > n_placeholders:
            audio_feats = audio_feats[:, :n_placeholders, :]
        elif T < n_placeholders:
            pad = torch.zeros(1, n_placeholders - T, audio_feats.shape[2],
                              device=device, dtype=audio_feats.dtype)
            audio_feats = torch.cat([audio_feats, pad], dim=1)

        mask = (input_ids[b] == audio_token_index)   # (L,) bool
        inputs_embeds[b][mask] = audio_feats[0].to(dtype)

    return inputs_embeds


# ─────────────────────────────────────────────────────────────────────────────
# Param group builder for optimizer
# Weight decay is applied only to weight matrices; biases and layer-norm
# params are excluded (standard practice, also used in BERT/GPT-style training)
# ─────────────────────────────────────────────────────────────────────────────

def get_adapter_param_groups(model: CovoAudioForCausalLM, weight_decay: float):
    """
    Returns two param groups:
      - decay:   weight matrices
      - no_decay: biases, LayerNorm/RMSNorm params
    """
    decay, no_decay = [], []
    for name, param in model.audio_adapter.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    # Also include dropout wrapper params if any
    for name, param in model.named_parameters():
        if "dropout" in name and param.requires_grad:
            no_decay.append(param)

    log.info(f"  Adapter decay params:    {len(decay)}")
    log.info(f"  Adapter no-decay params: {len(no_decay)}")

    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Loss with label smoothing
# We compute loss manually so we can add label smoothing, which
# model.forward() doesn't expose directly.
# ─────────────────────────────────────────────────────────────────────────────

_loss_fn = None   # created once after we know vocab_size

def compute_loss(logits: torch.Tensor, labels: torch.Tensor,
                 label_smoothing: float = 0.1) -> torch.Tensor:
    """
    Causal LM loss with label smoothing.
    logits: (B, L, V)
    labels: (B, L)   — -100 positions are ignored
    """
    global _loss_fn
    if _loss_fn is None:
        _loss_fn = nn.CrossEntropyLoss(ignore_index=-100,
                                       label_smoothing=label_smoothing)
    # Shift: predict token[i+1] from token[i]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:  ].contiguous()
    return _loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, audio_token_index, device,
             max_val_batches: int = 200) -> float:
    """
    Run validation on at most max_val_batches batches.
    Returns mean cross-entropy loss (no label smoothing for clean comparison).
    """
    model.eval()
    total_loss, n = 0.0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for batch in val_loader:
        if n >= max_val_batches:
            break
        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        inputs_embeds = inject_audio(model, batch["wavs"], input_ids, audio_token_index)
        outputs       = model(inputs_embeds=inputs_embeds,
                              attention_mask=attention_mask)
        shift_logits  = outputs.logits[:, :-1, :].contiguous()
        shift_labels  = labels[:, 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                       shift_labels.view(-1))

        total_loss += loss.item()
        n += 1

    model.train()
    return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, output_dir: str, tag: str,
                    step: int, val_loss: float):
    path = Path(output_dir) / tag
    path.mkdir(parents=True, exist_ok=True)

    # Only save the adapter weights (~120MB for 30M params in bfloat16)
    torch.save(
        {
            "adapter_state_dict": model.audio_adapter.state_dict(),
            "step":               step,
            "val_loss":           val_loss,
        },
        path / "audio_adapter.pt",
    )
    model.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    log.info(f"  ✓ Saved checkpoint → {path}  (val_loss={val_loss:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    log.info(f"\nLoading tokenizer: {args.gemma_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_ckpt)
    n_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|begofcAUDIO|>", "<|cAUDIO|>", "<|endofcAUDIO|>"
        ]
    })
    log.info(f"Added {n_added} audio special tokens")
    audio_token_index = tokenizer.convert_tokens_to_ids("<|cAUDIO|>")
    log.info(f"<|cAUDIO|> id = {audio_token_index}")

    # ── 2. Model ──────────────────────────────────────────────────────────────
    log.info("\nBuilding model...")
    config = CovoAudioConfig()
    config.audio_token_index = audio_token_index
    model = CovoAudioForCausalLM(config)

    # Load Gemma 3 4B pretrained weights
    log.info(f"Loading LLM weights: {args.gemma_ckpt}")
    llm = Gemma3ForCausalLM.from_pretrained(
        args.gemma_ckpt,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    llm.resize_token_embeddings(len(tokenizer))
    model.llm = llm
    del llm
    torch.cuda.empty_cache()

    # Load Whisper Medium pretrained weights
    log.info(f"Loading encoder weights: {args.whisper_ckpt}")
    whisper = WhisperModel.from_pretrained(
        args.whisper_ckpt,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.encoder = whisper.encoder
    del whisper
    torch.cuda.empty_cache()

    # ── 3. Freeze everything except AudioAdapter ──────────────────────────────
    log.info("\nParameter freeze plan:")
    for p in model.llm.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.audio_adapter.parameters():
        p.requires_grad = True

    log.info("  FROZEN:    Gemma 3 4B LLM")
    log.info("  FROZEN:    Whisper Medium encoder")
    log.info("  TRAINABLE: AudioAdapter")

    # Inject dropout into adapter layers (regularization)
    add_adapter_dropout(model, p=args.adapter_dropout)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)\n")

    model = model.to(device)
    model.train()

    # ── 4. Datasets ───────────────────────────────────────────────────────────
    log.info("Building streaming train dataset (LibriSpeech)...")
    train_ds = LibriSpeechStreamingDataset(
        tokenizer=tokenizer,
        audio_token_index=audio_token_index,
        use_extra_data=args.extra_data,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        max_duration=args.max_duration,
    )

    log.info("Loading validation set...")
    val_ds = LibriSpeechValDataset(
        tokenizer=tokenizer,
        audio_token_index=audio_token_index,
        max_duration=args.max_duration,
    )

    collator = AudioTextCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        padding_side="left",
        max_seq_len=args.max_seq_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    # ── 5. Optimizer & Scheduler ──────────────────────────────────────────────
    param_groups = get_adapter_param_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # OneCycleLR: warmup then cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.max_steps,
        pct_start=args.warmup_steps / args.max_steps,
        anneal_strategy="cos",
        div_factor=10.0,      # start lr = max_lr / 10
        final_div_factor=100, # end lr   = max_lr / 100
    )

    # GradScaler for mixed precision (fp16 on adapter; LLM is bfloat16)
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 6. Training loop ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss   = float("inf")
    patience_counter = 0
    global_step     = 0
    running_loss    = 0.0
    optimizer.zero_grad()

    log.info(f"\n{'='*60}")
    log.info(f"  Stage 1 Adapter Training")
    log.info(f"  LLM:           Gemma 3 4B IT")
    log.info(f"  Encoder:       Whisper Medium")
    log.info(f"  max_steps:     {args.max_steps:,}")
    log.info(f"  batch_size:    {args.batch_size}")
    log.info(f"  grad_accum:    {args.grad_accum}")
    log.info(f"  effective_bs:  {args.batch_size * args.grad_accum}")
    log.info(f"  lr:            {args.lr}")
    log.info(f"  warmup_steps:  {args.warmup_steps}")
    log.info(f"  weight_decay:  {args.weight_decay}")
    log.info(f"  label_smooth:  {args.label_smoothing}")
    log.info(f"  adapter_drop:  {args.adapter_dropout}")
    log.info(f"  eval_every:    {args.eval_every}")
    log.info(f"  early_stop_p:  {args.patience}")
    log.info(f"{'='*60}\n")

    for micro_step, batch in enumerate(train_loader):
        if global_step >= args.max_steps:
            break

        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # ── Forward ───────────────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            inputs_embeds = inject_audio(
                model, batch["wavs"], input_ids, audio_token_index
            )
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            loss = compute_loss(outputs.logits, labels,
                                label_smoothing=args.label_smoothing)
            loss = loss / args.grad_accum   # scale for gradient accumulation

        # ── Backward ──────────────────────────────────────────────────────
        scaler.scale(loss).backward()

        running_loss += loss.item() * args.grad_accum   # unscale for logging

        # ── Optimizer step (every grad_accum micro-steps) ─────────────────
        if (micro_step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for pg in param_groups for p in pg["params"]],
                max_norm=args.clip_grad,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # ── Logging ───────────────────────────────────────────────────
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                running_loss = 0.0
                lr_now = optimizer.param_groups[0]["lr"]
                log.info(
                    f"step {global_step:6d}/{args.max_steps}  "
                    f"loss={avg_loss:.4f}  "
                    f"grad_norm={grad_norm:.3f}  "
                    f"lr={lr_now:.2e}"
                )

            # ── Validation ────────────────────────────────────────────────
            if global_step % args.eval_every == 0:
                val_loss = evaluate(
                    model, val_loader, audio_token_index, device,
                    max_val_batches=args.max_val_batches,
                )
                log.info(
                    f"\n{'─'*50}\n"
                    f"  EVAL  step={global_step}  val_loss={val_loss:.4f}\n"
                    f"{'─'*50}"
                )

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    save_checkpoint(model, tokenizer, args.output_dir,
                                    "best", global_step, val_loss)
                else:
                    patience_counter += 1
                    log.info(f"  No improvement. Patience: {patience_counter}/{args.patience}")

                if patience_counter >= args.patience:
                    log.info(f"\nEarly stopping at step {global_step} "
                             f"(no improvement for {args.patience} eval rounds)")
                    break

            # ── Periodic save ─────────────────────────────────────────────
            if global_step % args.save_every == 0:
                save_checkpoint(model, tokenizer, args.output_dir,
                                f"step{global_step}", global_step, val_loss=0.0)

    # ── Final save ─────────────────────────────────────────────────────────
    save_checkpoint(model, tokenizer, args.output_dir,
                    "final", global_step, best_val_loss)
    log.info(f"\nDone. Best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints in: {args.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: Adapter alignment on LibriSpeech (streaming)"
    )

    # Model checkpoints
    p.add_argument("--gemma_ckpt",   default="google/gemma-3-4b-it")
    p.add_argument("--whisper_ckpt", default="openai/whisper-medium")
    p.add_argument("--output_dir",   default="checkpoints/stage1")

    # Data
    p.add_argument("--extra_data",     action="store_true",
                   help="Also stream CommonVoice-en + VoxPopuli-en")
    p.add_argument("--shuffle_buffer", type=int, default=1000,
                   help="Per-dataset shuffle buffer size")
    p.add_argument("--max_duration",   type=float, default=20.0,
                   help="Skip utterances longer than this (seconds)")
    p.add_argument("--max_seq_len",    type=int, default=2048)
    p.add_argument("--num_workers",    type=int, default=2)

    # Training
    p.add_argument("--max_steps",  type=int,   default=200000)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--grad_accum", type=int,   default=8,
                   help="Gradient accumulation steps (effective_bs = batch × accum)")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--adapter_dropout", type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)

    # Eval & saving
    p.add_argument("--eval_every",     type=int, default=2000)
    p.add_argument("--save_every",     type=int, default=5000)
    p.add_argument("--log_every",      type=int, default=100)
    p.add_argument("--max_val_batches",type=int, default=200,
                   help="Cap val batches per eval round (keeps eval fast)")
    p.add_argument("--patience",       type=int, default=5,
                   help="Early stop after N eval rounds without improvement")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
