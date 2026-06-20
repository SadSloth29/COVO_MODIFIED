"""
CovoAudioConfig — Qwen2.5-3B-Instruct + Whisper-Small backbone.

This is a deliberate DOWNSIZE of the original production config
(Qwen2-7B-Instruct + Whisper-Large-V3). The reference 7B/Large-V3 config was
provided as the baseline being scaled down from, not the target — this file
targets 3B/Small specifically.

What scales down vs what carries over unchanged
─────────────────────────────────────────────────
SCALES DOWN (architecture-size-specific dimensions):
  llm hidden_size          3584 → 2048
  llm intermediate_size    18944 → 11008
  llm num_hidden_layers    28 → 36     (3B is narrower but deeper than 7B)
  llm num_attention_heads  28 → 16
  llm num_key_value_heads  4 → 2
  whisper d_model          1280 → 768  (Large-V3 → Small)
  whisper num_mel_bins     128 → 80
  whisper encoder_layers   32 → 12
  whisper encoder_attention_heads  20 → 12
  whisper encoder_ffn_dim  5120 → 3072

CARRIES OVER UNCHANGED (protocol-level, not size-dependent):
  • The +16384 audio-codec vocab reservation. This range exists because the
    full model (a2ta mode) generates audio codec tokens interleaved with
    text, decoded by a separate vocoder (token2wav). The codec's token
    count (16384) is a property of that vocoder/speech-tokenizer, not of
    the LLM backbone size — it doesn't shrink just because the LLM did.
  • audio_token_index = base_vocab_size of whichever LLM you use, so that
    vocab_size = audio_token_index + 16384 still holds:
        Qwen2.5-3B-Instruct base vocab_size = 151936
        audio_token_index                  = 151936
        vocab_size                         = 151936 + 16384 = 168320
  • EOS-token-by-mode convention (a2t vs a2ta) — see note below. This is a
    decoding protocol choice, independent of model size.

eos_token_id note
───────────────────
bos_token_id == eos_token_id == 151643 (<|endoftext|>) for the base LLM
config, matching Qwen2.5's convention. The inference server selects the
ACTUAL stop token by generation mode at generate()-call time, not from this
config field:
    a2ta (interleaved audio+text chat): eos_token_id = <|im_end|>  (151645)
    a2t  (audio→text, i.e. ASR — what we train here): eos_token_id = <|endoftext|> (151643)
Training labels for ASR (a2t) must terminate with <|endoftext|>, matching
this config's eos_token_id — see dataset.py's build_sample().

n_mels note
───────────
Whisper Small uses num_mel_bins=80. modeling_covo_audio.py's audio_encoder()
must call log_mel_spectrogram(audio, n_mels=80) to match — keep these two in
sync for whichever Whisper size is actually used.
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
        llm_config:         Optional[LLMConfig]    = None,
        encoder_config:     Optional[WhisperConfig] = None,
        audio_token_index:  int   = 151936,   # = Qwen2.5-3B-Instruct base vocab_size
        adapter_downsample: int   = 8,
        **kwargs,
    ):
        # ── Qwen 2.5 3B Instruct ─────────────────────────────────────────────
        if llm_config is None:
            llm_config = LLMConfig(
                architectures=["Qwen2ForCausalLM"],
                model_type="qwen2",

                # base vocab (151936) + 16384 reserved audio-codec tokens.
                # See module docstring — this offset carries over from the
                # original 7B config's protocol, scaled to the 3B base vocab.
                vocab_size=151936 + 16384,   # = 168320

                hidden_size=2048,
                intermediate_size=11008,
                num_hidden_layers=36,
                num_attention_heads=16,
                num_key_value_heads=2,
                head_dim=128,            # hidden_size // num_attention_heads
                max_window_layers=36,

                hidden_act="silu",

                max_position_embeddings=32768,
                rope_theta=1000000.0,
                rope_scaling=None,
                sliding_window=32768,
                use_sliding_window=False,

                attention_dropout=0.0,
                rms_norm_eps=1e-6,
                initializer_range=0.02,

                # <|endoftext|> = 151643 for both bos and eos (Qwen2.5
                # convention). a2ta mode overrides eos at generate()-call
                # time with <|im_end|> (151645) — not baked in here, since
                # the same checkpoint serves both a2t and a2ta depending on
                # how it's invoked.
                bos_token_id=151643,
                eos_token_id=151643,

                use_cache=True,
                use_mrope=False,
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

                # 80 mel bins — Whisper Small. audio_encoder() in
                # modeling_covo_audio.py must call log_mel_spectrogram with
                # n_mels=80 to match. DO NOT use 128 here (Large-V3 only).
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

        self.audio_token_index  = audio_token_index
        self.adapter_downsample = adapter_downsample
        self.llm_config         = llm_config
        self.encoder_config     = encoder_config

        # Derived dimensions
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
