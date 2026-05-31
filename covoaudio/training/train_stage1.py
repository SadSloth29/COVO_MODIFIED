"""
Stage 1 Training — Adapter Alignment (LibriSpeech, Streaming)
==============================================================
Multi-GPU training via torchrun (DistributedDataParallel).

Launch command (2 GPUs):
    torchrun --nproc_per_node=2 training/train_stage1.py [args]

Single GPU (no change to args):
    python training/train_stage1.py [args]

What is frozen vs trained:
    FROZEN:  WhisperEncoder  (pretrained openai/whisper-medium)
    FROZEN:  Gemma 3 4B LLM  (pretrained google/gemma-3-4b-it)
    TRAINED: AudioAdapter    (~30M params)
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, WhisperModel
from transformers.models.gemma3 import Gemma3ForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling_covo_audio import CovoAudioForCausalLM, sequence_mask
from configuration_covo_audio import CovoAudioConfig
from training.dataset  import LibriSpeechStreamingDataset, LibriSpeechValDataset
from training.collator import AudioTextCollator


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    """
    Initialise the process group when launched via torchrun.
    Falls back to single-GPU if env vars are not set.
    """
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return local_rank, world_size, torch.device(f"cuda:{local_rank}")


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def setup_logging(local_rank: int):
    """Only rank 0 logs — all other ranks are silent."""
    level = logging.INFO if local_rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter dropout wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DropoutWrapper(nn.Module):
    def __init__(self, module: nn.Module, p: float = 0.1):
        super().__init__()
        self.module  = module
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(self.module(x))


def add_adapter_dropout(model: CovoAudioForCausalLM, p: float = 0.1):
    layers = model.audio_adapter.downsample_layers
    for i in range(len(layers)):
        layers[i] = DropoutWrapper(layers[i], p=p)


# ─────────────────────────────────────────────────────────────────────────────
# Audio injection  (training-time forward pass)
# ─────────────────────────────────────────────────────────────────────────────

def inject_audio(
    model,                  # raw model (not DDP wrapper) for audio_encoder access
    ddp_model,              # DDP-wrapped model for embedding lookup
    wavs:              list,
    input_ids:         torch.Tensor,
    audio_token_index: int,
    log,
    debug_norms:       bool = False,
) -> torch.Tensor:
    """
    Replaces <|cAUDIO|> placeholder embeddings with audio features.

    Args:
        model:     The unwrapped CovoAudioForCausalLM (for audio_encoder).
        ddp_model: The DDP-wrapped model (for get_input_embeddings).
        debug_norms: When True, logs audio vs text embedding magnitudes.
                     Enable for the first few steps to diagnose loss=93.
    """
    device = input_ids.device
    # Get the underlying module whether DDP or not
    base = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
    dtype = next(base.llm.parameters()).dtype

    with torch.no_grad():
        inputs_embeds = base.llm.get_input_embeddings()(input_ids)  # (B, L, H)

    for b, wav in enumerate(wavs):
        n_placeholders = (input_ids[b] == audio_token_index).sum().item()

        if debug_norms:
            text_norm = inputs_embeds[b].norm(dim=-1).mean().item()
            log.info(
                f"[norm-debug] sample={b}  "
                f"placeholders={n_placeholders}  "
                f"text_embed_norm={text_norm:.6f}"
            )
            if n_placeholders == 0:
                log.warning(
                    f"[norm-debug] sample={b} has ZERO placeholders — "
                    f"audio_token_index={audio_token_index} not found in input_ids. "
                    f"Audio injection is silently failing."
                )
                continue

        if n_placeholders == 0:
            continue

        audio_feats = model.audio_encoder([wav.to(device)], device)  # (1, T, H)

        if debug_norms:
            audio_norm = audio_feats.norm(dim=-1).mean().item()
            ratio      = audio_norm / max(text_norm, 1e-9)
            log.info(
                f"[norm-debug] sample={b}  "
                f"audio_feat_norm={audio_norm:.6f}  "
                f"text_embed_norm={text_norm:.6f}  "
                f"ratio={ratio:.1f}x  "
                f"{'⚠ LARGE RATIO — adapter magnitude problem' if ratio > 10 else 'OK'}"
            )

        T = audio_feats.shape[1]
        if T > n_placeholders:
            audio_feats = audio_feats[:, :n_placeholders, :]
        elif T < n_placeholders:
            pad = torch.zeros(
                1, n_placeholders - T, audio_feats.shape[2],
                device=device, dtype=audio_feats.dtype,
            )
            audio_feats = torch.cat([audio_feats, pad], dim=1)

        mask = (input_ids[b] == audio_token_index)
        inputs_embeds[b][mask] = audio_feats[0].to(dtype)

    return inputs_embeds


# ─────────────────────────────────────────────────────────────────────────────
# Param groups  (weight decay only on weight matrices, not biases/norms)
# ─────────────────────────────────────────────────────────────────────────────

def get_adapter_param_groups(model: CovoAudioForCausalLM, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.audio_adapter.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Loss with label smoothing
# ─────────────────────────────────────────────────────────────────────────────

_loss_fn = None

def compute_loss(logits: torch.Tensor, labels: torch.Tensor,
                 label_smoothing: float = 0.1) -> torch.Tensor:
    global _loss_fn
    if _loss_fn is None:
        _loss_fn = nn.CrossEntropyLoss(ignore_index=-100,
                                       label_smoothing=label_smoothing)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:  ].contiguous()
    return _loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation  (rank 0 only)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, audio_token_index, device,
             max_val_batches: int, log) -> float:
    model.eval()
    base     = model.module if hasattr(model, "module") else model
    loss_fn  = nn.CrossEntropyLoss(ignore_index=-100)
    total, n = 0.0, 0

    pbar = tqdm(
        val_loader,
        desc="Validating",
        total=max_val_batches,
        leave=False,
        disable=not is_main_process(),
    )
    for batch in pbar:
        if n >= max_val_batches:
            break
        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        inputs_embeds = inject_audio(
            base, model, batch["wavs"], input_ids, audio_token_index,
            log, debug_norms=False,
        )
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        total += loss.item()
        n     += 1
        pbar.set_postfix(val_loss=f"{total/n:.4f}")

    model.train()
    return total / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint  (rank 0 only)
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, output_dir, tag, step, val_loss, log):
    if not is_main_process():
        return
    path = Path(output_dir) / tag
    path.mkdir(parents=True, exist_ok=True)

    base = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "adapter_state_dict": base.audio_adapter.state_dict(),
            "step":               step,
            "val_loss":           val_loss,
        },
        path / "audio_adapter.pt",
    )
    base.config.save_pretrained(path)
    tokenizer.save_pretrained(path)
    log.info(f"  ✓ Saved → {path}  (step={step}  val_loss={val_loss:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    local_rank, world_size, device = setup_distributed()
    log = setup_logging(local_rank)

    is_distributed = world_size > 1
    is_main        = is_main_process()

    log.info(f"Distributed: {is_distributed}  world_size={world_size}  "
             f"local_rank={local_rank}  device={device}")

    # ── 1. Tokenizer (all ranks load identically) ─────────────────────────────
    log.info(f"Loading tokenizer: {args.gemma_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(args.gemma_ckpt)
    n_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|begofcAUDIO|>", "<|cAUDIO|>", "<|endofcAUDIO|>"
        ]
    })
    log.info(f"Added {n_added} audio special tokens")

    audio_token_index = tokenizer.convert_tokens_to_ids("<|cAUDIO|>")
    assert audio_token_index != tokenizer.unk_token_id, \
        "<|cAUDIO|> resolved to UNK — add_special_tokens failed"
    log.info(f"<|cAUDIO|> id = {audio_token_index}")

    # ── 2. Build model ────────────────────────────────────────────────────────
    log.info("Building model...")
    config = CovoAudioConfig()
    config.audio_token_index = audio_token_index
    model  = CovoAudioForCausalLM(config)

    # Apply small init to adapter's final projection layer
    # CRITICAL: prevents loss=93 caused by large random adapter outputs
    last_layer = model.audio_adapter.downsample_layers[-1]
    # DropoutWrapper hasn't been added yet so access module directly
    target = last_layer.module if hasattr(last_layer, "module") else last_layer
    nn.init.normal_(target.linear2.weight, std=0.01)
    nn.init.zeros_(target.linear2.bias)
    log.info("Applied small init to adapter final projection layer")

    # Load pretrained LLM
    log.info(f"Loading LLM: {args.gemma_ckpt}")
    llm = Gemma3ForCausalLM.from_pretrained(
        args.gemma_ckpt,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    llm.resize_token_embeddings(len(tokenizer))
    model.llm = llm
    del llm
    torch.cuda.empty_cache()

    # Load pretrained Whisper encoder
    log.info(f"Loading encoder: {args.whisper_ckpt}")
    whisper = WhisperModel.from_pretrained(
        args.whisper_ckpt,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.encoder = whisper.encoder
    del whisper
    torch.cuda.empty_cache()

    # ── 3. Freeze ─────────────────────────────────────────────────────────────
    for p in model.llm.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.audio_adapter.parameters():
        p.requires_grad = True

    add_adapter_dropout(model, p=args.adapter_dropout)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── 4. Move to device & wrap in DDP ──────────────────────────────────────
    model = model.to(device)

    if is_distributed:
        # find_unused_parameters=True because LLM and encoder params are frozen
        # and never produce gradients — DDP needs to know this is intentional
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        log.info(f"Wrapped in DDP  (find_unused_parameters=True)")

    model.train()

    # Convenience: unwrapped model for audio_encoder calls
    base_model = model.module if is_distributed else model

    # ── 5. Datasets ───────────────────────────────────────────────────────────
    log.info("Building streaming train dataset...")
    train_ds = LibriSpeechStreamingDataset(
        tokenizer=tokenizer,
        audio_token_index=audio_token_index,
        use_extra_data=args.extra_data,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        max_duration=args.max_duration,
    )

    collator = AudioTextCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        padding_side="left",
        max_seq_len=args.max_seq_len,
    )

    # Streaming datasets don't support DistributedSampler.
    # Instead each rank gets a different shard of the interleaved stream
    # by using a rank-offset seed — each process sees different samples.
    if is_distributed:
        # Reseed the dataset per rank so each GPU sees different data
        train_ds_rank = LibriSpeechStreamingDataset(
            tokenizer=tokenizer,
            audio_token_index=audio_token_index,
            use_extra_data=args.extra_data,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed + local_rank,   # ← different seed per rank
            max_duration=args.max_duration,
        )
    else:
        train_ds_rank = train_ds

    train_loader = DataLoader(
        train_ds_rank,
        batch_size=args.batch_size,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Validation only on rank 0 to avoid redundant computation
    val_loader = None
    if is_main:
        log.info("Loading validation set (rank 0 only)...")
        val_ds = LibriSpeechValDataset(
            tokenizer=tokenizer,
            audio_token_index=audio_token_index,
            max_duration=args.max_duration,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )

    # ── 6. Optimizer & scheduler ──────────────────────────────────────────────
    param_groups = get_adapter_param_groups(base_model, weight_decay=args.weight_decay)
    optimizer    = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler    = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.max_steps,
        pct_start=args.warmup_steps / args.max_steps,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100,
    )

    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── 7. Training loop ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    best_val_loss    = float("inf")
    val_loss         = float("inf")
    patience_counter = 0
    global_step      = 0
    running_loss     = 0.0
    optimizer.zero_grad()

    # Norm debug: enable for first debug_norm_steps steps on rank 0 only
    debug_norm_steps = args.debug_norm_steps

    log.info(f"\n{'='*60}")
    log.info(f"  Stage 1 — Adapter Alignment")
    log.info(f"  GPUs:          {world_size}")
    log.info(f"  LLM:           Gemma 3 4B IT")
    log.info(f"  Encoder:       Whisper Medium")
    log.info(f"  max_steps:     {args.max_steps:,}")
    log.info(f"  batch/GPU:     {args.batch_size}")
    log.info(f"  grad_accum:    {args.grad_accum}")
    log.info(f"  effective_bs:  {args.batch_size * args.grad_accum * world_size}")
    log.info(f"  lr:            {args.lr}")
    log.info(f"  warmup_steps:  {args.warmup_steps}")
    log.info(f"  weight_decay:  {args.weight_decay}")
    log.info(f"  label_smooth:  {args.label_smoothing}")
    log.info(f"  adapter_drop:  {args.adapter_dropout}")
    log.info(f"  debug_norms:   first {debug_norm_steps} steps")
    log.info(f"{'='*60}\n")

    progress = tqdm(
        total=args.max_steps,
        desc="Stage 1",
        unit="step",
        disable=not is_main,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    for micro_step, batch in enumerate(train_loader):
        if global_step >= args.max_steps:
            break

        input_ids      = batch["input_ids"].to(device)
        labels         = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Norm debug active for the first N optimizer steps on rank 0 only
        do_debug = is_main and (global_step < debug_norm_steps)

        # ── Forward ───────────────────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
            inputs_embeds = inject_audio(
                base_model, model, batch["wavs"],
                input_ids, audio_token_index,
                log, debug_norms=do_debug,
            )
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
            loss = compute_loss(
                outputs.logits, labels,
                label_smoothing=args.label_smoothing,
            )
            loss = loss / args.grad_accum

        # ── Backward ──────────────────────────────────────────────────────────
        scaler.scale(loss).backward()
        running_loss += loss.item() * args.grad_accum

        # ── Optimizer step ────────────────────────────────────────────────────
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

            # ── Progress bar ──────────────────────────────────────────────────
            if is_main:
                avg_loss = running_loss / min(global_step, args.log_every)
                lr_now   = optimizer.param_groups[0]["lr"]
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss.item() * args.grad_accum:.4f}",
                    avg=f"{avg_loss:.4f}",
                    grad=f"{grad_norm:.2f}",
                    lr=f"{lr_now:.1e}",
                    val=f"{best_val_loss:.4f}" if best_val_loss < float("inf") else "n/a",
                )

            # ── Logging ───────────────────────────────────────────────────────
            if is_main and global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                running_loss = 0.0
                log.info(
                    f"step {global_step:6d}/{args.max_steps}  "
                    f"loss={loss.item() * args.grad_accum:.4f}  "
                    f"avg={avg_loss:.4f}  "
                    f"grad={grad_norm:.2f}  "
                    f"lr={lr_now:.2e}  "
                    f"gpu={local_rank}"
                )

            # ── Validation (rank 0 only) ───────────────────────────────────────
            if global_step % args.eval_every == 0 and is_main:
                progress.set_description("Evaluating")
                val_loss = evaluate(
                    model, val_loader, audio_token_index,
                    device, args.max_val_batches, log,
                )
                progress.set_description("Stage 1")
                log.info(
                    f"\n{'─'*50}\n"
                    f"  EVAL  step={global_step}  "
                    f"val_loss={val_loss:.4f}  "
                    f"best={best_val_loss:.4f}\n"
                    f"{'─'*50}"
                )
                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    save_checkpoint(model, tokenizer, args.output_dir,
                                    "best", global_step, val_loss, log)
                else:
                    patience_counter += 1
                    log.info(f"  No improvement. "
                             f"Patience: {patience_counter}/{args.patience}")
                    if patience_counter >= args.patience:
                        log.info(f"\nEarly stopping at step {global_step}")
                        break

            # ── Periodic save ─────────────────────────────────────────────────
            if global_step % args.save_every == 0:
                save_checkpoint(model, tokenizer, args.output_dir,
                                f"step{global_step}", global_step, val_loss, log)

        # Sync all ranks before next step
        barrier()

    # ── Final save ────────────────────────────────────────────────────────────
    progress.close()
    save_checkpoint(model, tokenizer, args.output_dir,
                    "final", global_step, best_val_loss, log)
    log.info(f"\nDone. Best val loss: {best_val_loss:.4f}")
    log.info(f"Checkpoints: {args.output_dir}")

    if is_distributed:
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1: Adapter alignment — torchrun multi-GPU"
    )
    p.add_argument("--gemma_ckpt",   default="google/gemma-3-4b-it")
    p.add_argument("--whisper_ckpt", default="openai/whisper-medium")
    p.add_argument("--output_dir",   default="checkpoints/stage1")

    p.add_argument("--extra_data",     action="store_true")
    p.add_argument("--shuffle_buffer", type=int,   default=1000)
    p.add_argument("--max_duration",   type=float, default=20.0)
    p.add_argument("--max_seq_len",    type=int,   default=2048)
    p.add_argument("--num_workers",    type=int,   default=2)

    p.add_argument("--max_steps",       type=int,   default=200000)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--grad_accum",      type=int,   default=8)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--warmup_steps",    type=int,   default=2000)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--clip_grad",       type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--adapter_dropout", type=float, default=0.1)
    p.add_argument("--seed",            type=int,   default=42)

    p.add_argument("--eval_every",      type=int, default=2000)
    p.add_argument("--save_every",      type=int, default=5000)
    p.add_argument("--log_every",       type=int, default=100)
    p.add_argument("--max_val_batches", type=int, default=200)
    p.add_argument("--patience",        type=int, default=5)

    # Norm debugging: logs audio vs text embedding magnitudes for first N steps
    # Set to 0 to disable. Use 5-10 to diagnose loss=93 issues.
    p.add_argument("--debug_norm_steps", type=int, default=5,
                   help="Log audio/text embedding norms for first N optimizer steps")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())