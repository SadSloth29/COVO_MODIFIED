"""
CovoAudioConfig
================
Updated for:
  - Gemma 3 4B IT  (hidden_size=2560, 34 layers, 16 heads)
  - Whisper Medium  (d_model=1024, num_mel_bins=80)

IMPORTANT — Required patch in modeling_covo_audio.py:
  Change line in audio_encoder():
    FROM: mel_features = log_mel_spectrogram(audio, n_mels=128)
    TO:   mel_features = log_mel_spectrogram(audio, n_mels=self.config.encoder_config.num_mel_bins)

  Reason: Whisper Large-V3 uses 128 mel bins; Whisper Medium uses 80.
  The original hardcoded 128 will crash with a shape mismatch on
  WhisperEncoder.conv1 (in_channels=80 vs input channels=128).
"""

from typing import Optional
from transformers import Gemma3TextConfig, WhisperConfig
from transformers.configuration_utils import PretrainedConfig


class CovoAudioConfig(PretrainedConfig):
    model_type = "covo_audio"
    sub_configs = {
        "llm_config":     Gemma3TextConfig,
        "encoder_config": WhisperConfig,
    }
    has_no_defaults_at_init = True

    def __init__(
        self,
        llm_config:     Optional[Gemma3TextConfig] = None,
        encoder_config: Optional[WhisperConfig]    = None,
        audio_token_index: int = 262208,
        adapter_downsample: int = 8,
        **kwargs,
    ):
        # ── Gemma 3 4B IT ─────────────────────────────────────────────────────
        if llm_config is None:
            llm_config = Gemma3TextConfig(
                # Identity
                architectures=["Gemma3ForCausalLM"],
                model_type="gemma3_text",

                # Vocabulary
                vocab_size=262208,

                # Architecture — Gemma 3 4B
                hidden_size=2560,
                intermediate_size=10240,
                num_hidden_layers=34,
                num_attention_heads=16,
                num_key_value_heads=8,
                head_dim=256,

                # Activation
                hidden_activation="gelu_pytorch_tanh",

                # Position encoding
                max_position_embeddings=131072,
                rope_theta=1000000.0,
                rope_scaling=None,

                # Attention
                # Gemma 3 4B alternates local (sliding_window=1024) and
                # global attention layers. Keep the default here;
                # set sliding_window=None only if you encounter OOM on
                # very long audio sequences (> 30s stacked context).
                sliding_window=1024,
                attention_dropout=0.0,
                attention_bias=False,

                # Normalization
                rms_norm_eps=1e-6,
                initializer_range=0.02,

                # Tokens
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,

                use_cache=True,

                # IMPORTANT: tie_word_embeddings=True is Gemma's default but
                # CovoAudioForCausalLM.tie_weights() is a no-op, so in practice
                # lm_head and input embeddings are NOT shared.
                # Keep True for config compatibility; the no-op override handles it.
                tie_word_embeddings=True,

                torch_dtype="bfloat16",
            )

        # ── Whisper Medium ─────────────────────────────────────────────────────
        if encoder_config is None:
            encoder_config = WhisperConfig(
                _name_or_path="openai/whisper-medium",
                architectures=["WhisperForConditionalGeneration"],
                model_type="whisper",

                vocab_size=51865,

                # num_mel_bins=80 is correct for Whisper Medium.
                # Whisper Large-V3 uses 128. Do NOT change this to 128 here —
                # it would mismatch the pretrained conv1 weights.
                num_mel_bins=80,

                # Encoder architecture
                d_model=1024,
                encoder_layers=24,
                encoder_attention_heads=16,
                encoder_ffn_dim=4096,

                # Decoder architecture (not used during inference/training here
                # but required for WhisperConfig to be valid)
                decoder_layers=24,
                decoder_attention_heads=16,
                decoder_ffn_dim=4096,

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
                num_hidden_layers=24,
            )

        self.audio_token_index  = audio_token_index
        self.adapter_downsample = adapter_downsample
        self.llm_config         = llm_config
        self.encoder_config     = encoder_config
        # Derived: adapter input dim comes from Whisper encoder output
        self.whisper_feats_dim  = encoder_config.d_model   # 1024 for medium

        if "dtype" not in kwargs:
            kwargs["dtype"] = "bfloat16"
        self.dtype = kwargs["dtype"]

        super().__init__(**kwargs)

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def num_hidden_layers(self):
        return self.llm_config.num_hidden_layers   # 34 for 4B

    @property
    def hidden_size(self):
        return self.llm_config.hidden_size          # 2560 for 4B

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self):
        output = super().to_dict()
        if hasattr(self, "llm_config") and isinstance(self.llm_config, PretrainedConfig):
            output["llm_config"]      = self.llm_config.to_dict()
            output["_llm_config_type"] = getattr(self.llm_config, "model_type", None)
        if hasattr(self, "encoder_config") and isinstance(self.encoder_config, PretrainedConfig):
            output["encoder_config"]      = self.encoder_config.to_dict()
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
