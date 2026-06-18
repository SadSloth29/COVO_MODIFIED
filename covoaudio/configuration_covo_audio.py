"""
CovoAudioConfig — Qwen2.5-3B-Instruct + Whisper-Small backbone.

Design notes
────────────
• LLM   : Qwen2 (transformers >= 4.37, Qwen2Config / Qwen2ForCausalLM)
• Encoder: Whisper Small  (80-mel, d_model=768)
• audio_token_index = 151936  — first slot past the standard Qwen2.5 vocab
• adapter_downsample = 8      — 4 × DownsampleLayer(stride=2) → 16× temporal
                                compression of 100-Hz Whisper frames
• alignment_loss_weight       — λ for the cosine-embedding alignment term
                                added on top of the cross-entropy ASR loss.
                                Set to 0.0 to disable; 0.1 is a safe default.

n_mels note
───────────
The WhisperConfig below uses num_mel_bins=80 (Whisper Small pretrained weights).
However modeling_covo_audio.py calls log_mel_spectrogram(..., n_mels=128).
These must agree or the conv1 weights will be mismatched when loading a
pretrained encoder.  The fix is applied in modeling_covo_audio.py (change to
n_mels=80) AND here num_mel_bins=80 is kept as the canonical value.
"""
from typing import Optional
from transformers import WhisperConfig, PretrainedConfig

try:
    from transformers import Qwen2Config as LLMConfig
    LLM_MODEL_TYPE = "qwen2"
except ImportError:
    raise ImportError("transformers >= 4.37 is required for Qwen2Config.")


class CovoAudioConfig(PretrainedConfig):
    model_type = "covo_audio"
    sub_configs = {
        "llm_config":     LLMConfig,
        "encoder_config": WhisperConfig,
    }
    has_no_defaults_at_init = True

    def __init__(
        self,
        llm_config:            Optional[LLMConfig]    = None,
        encoder_config:        Optional[WhisperConfig] = None,
        audio_token_index:     int   = 151936,
        adapter_downsample:    int   = 8,
        alignment_loss_weight: float = 0.1,
        **kwargs,
    ):
        # ── Qwen 2.5 3B Instruct ─────────────────────────────────────────────
        if llm_config is None:
            llm_config = LLMConfig(
                architectures=["Qwen2ForCausalLM"],
                model_type="qwen2",

                vocab_size=151936,

                hidden_size=2048,
                intermediate_size=11008,
                num_hidden_layers=36,
                num_attention_heads=16,
                num_key_value_heads=2,
                head_dim=128,

                hidden_act="silu",

                max_position_embeddings=32768,
                rope_theta=1000000.0,
                rope_scaling=None,

                attention_dropout=0.0,
                rms_norm_eps=1e-6,
                initializer_range=0.02,

                # Qwen2.5 special tokens:
                #   <|endoftext|>  = 151643
                #   <|im_start|>   = 151644
                #   <|im_end|>     = 151645  ← chat EOS
                bos_token_id=151643,
                eos_token_id=151645,
                pad_token_id=151643,

                use_cache=True,
                tie_word_embeddings=False,
                torch_dtype="bfloat16",
            )

        # ── Whisper Small ─────────────────────────────────────────────────────
        if encoder_config is None:
            encoder_config = WhisperConfig(
                _name_or_path="openai/whisper-small",
                architectures=["WhisperForConditionalGeneration"],
                model_type="whisper",

                vocab_size=51865,

                # 80 mel bins — Whisper Small pretrained weights.
                # modeling_covo_audio.py must also use n_mels=80 here.
                # DO NOT change to 128 (Large-V3 only).
                num_mel_bins=80,

                d_model=768,
                encoder_layers=12,
                encoder_attention_heads=12,
                encoder_ffn_dim=3072,

                decoder_layers=12,
                decoder_attention_heads=12,
                decoder_ffn_dim=3072,

                activation_function="gelu",
                dropout=0.0,
                attention_dropout=0.0,
                activation_dropout=0.0,
                encoder_layerdrop=0.0,
                decoder_layerdrop=0.0,

                init_std=0.02,

                max_source_positions=1500,
                max_target_positions=448,
                max_length=448,

                bos_token_id=50257,
                eos_token_id=50257,
                decoder_start_token_id=50258,
                begin_suppress_tokens=[220, 50257],

                use_cache=True,
                scale_embedding=False,
                classifier_proj_size=256,

                apply_spec_augment=False,
                mask_time_prob=0.05,
                mask_time_length=10,
                mask_time_min_masks=2,
                mask_feature_prob=0.0,
                mask_feature_length=10,
                mask_feature_min_masks=0,

                median_filter_width=7,
                torch_dtype="float16",
                use_weighted_layer_sum=False,
                num_hidden_layers=12,
            )

        self.audio_token_index     = audio_token_index
        self.adapter_downsample    = adapter_downsample
        self.alignment_loss_weight = alignment_loss_weight
        self.llm_config            = llm_config
        self.encoder_config        = encoder_config

        # Derived dimensions — used by the trainer and the adapter
        self.whisper_feats_dim = encoder_config.d_model      # 768
        self.llm_hidden_size   = llm_config.hidden_size      # 2048

        if "dtype" not in kwargs:
            kwargs["dtype"] = "bfloat16"
        self.dtype = kwargs["dtype"]

        super().__init__(**kwargs)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def num_hidden_layers(self) -> int:
        return self.llm_config.num_hidden_layers   # 36

    @property
    def hidden_size(self) -> int:
        return self.llm_config.hidden_size          # 2048

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self):
        output = super().to_dict()
        if hasattr(self, "llm_config") and isinstance(self.llm_config, PretrainedConfig):
            output["llm_config"]       = self.llm_config.to_dict()
            output["_llm_config_type"] = getattr(self.llm_config, "model_type", None)
        if hasattr(self, "encoder_config") and isinstance(self.encoder_config, PretrainedConfig):
            output["encoder_config"]       = self.encoder_config.to_dict()
            output["_encoder_config_type"] = getattr(self.encoder_config, "model_type", None)
        return output

    @classmethod
    def from_dict(cls, config_dict: dict, **kwargs):
        data = dict(config_dict)
        llm_conf = enc_conf = None

        if "llm_config" in data and data["llm_config"] is not None:
            llm_cls  = cls.sub_configs.get("llm_config")
            llm_conf = llm_cls.from_dict(data.pop("llm_config")) if llm_cls else data.pop("llm_config")

        if "encoder_config" in data and data["encoder_config"] is not None:
            enc_cls  = cls.sub_configs.get("encoder_config")
            enc_conf = enc_cls.from_dict(data.pop("encoder_config")) if enc_cls else data.pop("encoder_config")

        data.pop("_llm_config_type",     None)
        data.pop("_encoder_config_type", None)

        return cls(llm_config=llm_conf, encoder_config=enc_conf, **data)