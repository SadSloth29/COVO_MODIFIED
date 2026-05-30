"""
Collator — pads variable-length sequences into fixed-size batches.

Left-pads token sequences (standard for decoder-only/causal LMs).
Right-pads waveforms (silence padding doesn't affect Whisper).
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AudioTextCollator:
    """
    Args:
        pad_token_id:  Token id for input_ids padding. Gemma3 uses 0 (<pad>).
        padding_side:  'left' for decoder-only models (default), 'right' otherwise.
        max_seq_len:   Hard cap on sequence length (prevents OOM on outliers).
                       Sequences longer than this are right-truncated at the
                       response boundary (never truncates inside audio tokens).
    """
    pad_token_id: int  = 0
    padding_side: str  = "left"
    max_seq_len:  int  = 2048

    def __call__(self, samples: List[Dict]) -> Dict:
        # ── 1. Wavs: right-pad to longest in batch ─────────────────────────
        wav_lengths = [s["wav"].shape[0] for s in samples]
        max_wav     = max(wav_lengths)
        wavs = []
        for s in samples:
            w = s["wav"]
            pad = max_wav - w.shape[0]
            wavs.append(torch.nn.functional.pad(w, (0, pad)) if pad > 0 else w)
        # wavs is a plain list; audio_encoder iterates it sample-by-sample

        # ── 2. Tokens: truncate then pad ───────────────────────────────────
        input_ids_list  = []
        labels_list     = []
        attn_mask_list  = []

        for s in samples:
            ids   = s["input_ids"]
            lbls  = s["labels"]
            amask = s["attention_mask"]

            # Truncate if over limit (keep the audio-token prefix intact)
            if len(ids) > self.max_seq_len:
                ids   = ids  [: self.max_seq_len]
                lbls  = lbls [: self.max_seq_len]
                amask = amask[: self.max_seq_len]

            input_ids_list.append(ids)
            labels_list.append(lbls)
            attn_mask_list.append(amask)

        max_len = max(x.shape[0] for x in input_ids_list)

        padded_ids    = []
        padded_labels = []
        padded_masks  = []

        for ids, lbls, amask in zip(input_ids_list, labels_list, attn_mask_list):
            pad = max_len - ids.shape[0]
            if pad == 0:
                padded_ids.append(ids)
                padded_labels.append(lbls)
                padded_masks.append(amask)
                continue

            pad_ids   = torch.full((pad,), self.pad_token_id, dtype=torch.long)
            pad_lbls  = torch.full((pad,), -100,              dtype=torch.long)
            pad_mask  = torch.zeros(pad,                       dtype=torch.long)

            if self.padding_side == "left":
                padded_ids.append(torch.cat([pad_ids,  ids  ]))
                padded_labels.append(torch.cat([pad_lbls, lbls ]))
                padded_masks.append(torch.cat([pad_mask, amask]))
            else:
                padded_ids.append(torch.cat([ids,   pad_ids  ]))
                padded_labels.append(torch.cat([lbls,  pad_lbls ]))
                padded_masks.append(torch.cat([amask, pad_mask]))

        return {
            "wavs":           wavs,                               # list[(T,)]
            "input_ids":      torch.stack(padded_ids),            # (B, L)
            "labels":         torch.stack(padded_labels),         # (B, L)
            "attention_mask": torch.stack(padded_masks),          # (B, L)
        }
