"""Hub-free tiny model factories for state control tests.

Provides a randomly initialized tiny Llama and a hand-built WordLevel tokenizer so
that hook-level behavioral tests can run without downloading models from the HF Hub.
The `local:<builder>` entries of `tests/utils/ci_models.yaml` are built by functions here.
"""
import torch
import torch.nn as nn
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import AddedToken, GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def tiny_llama(num_layers=4, hidden=32, heads=4, vocab=100):
    """Build a randomly initialized tiny Llama model in eval mode."""
    cfg = LlamaConfig(
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        vocab_size=vocab,
    )
    return LlamaForCausalLM(cfg).eval()


def tiny_gpt2(num_layers=4, hidden=32, heads=4, vocab=100):
    """Build a randomly initialized tiny GPT-2 model in eval mode.

    Exercises the `transformer.h` / `attn.c_proj` / `ln_1`,`ln_2` layout branch.
    """
    cfg = GPT2Config(
        n_embd=hidden,
        n_inner=2 * hidden,
        n_layer=num_layers,
        n_head=heads,
        n_positions=128,
        vocab_size=vocab,
    )
    return GPT2LMHeadModel(cfg).eval()


def tiny_gemma3_conditional(num_layers=4, hidden=32, heads=4, vocab=100):
    """Build a randomly initialized tiny Gemma 3 multimodal wrapper in eval mode.

    Exercises the `model.language_model.layers` root and the gemma-style norm conventions
    (`input_layernorm`, `pre_feedforward_layernorm`).
    """
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig

    text = Gemma3TextConfig(
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=hidden // heads,
        vocab_size=vocab,
        sliding_window=8,
    )
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    cfg = Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=4)
    return Gemma3ForConditionalGeneration(cfg).eval()


def tiny_gemma4_text_config(num_layers=6, hidden=32, heads=4, vocab=128, head_dim=8, global_head_dim=16):
    """Build a tiny Gemma 4 text config whose global attention layers use `global_head_dim`.

    Gemma 4 declares `head_dim` for its sliding layers and `global_head_dim` for its global
    (`full_attention`) layers, which are one in every six layers by default. Transformers >= 5.15
    registers the global layers' value through `per_layer_config`, which marks `head_dim` as a
    per-layer attribute of the config (`gemma4_declares_per_layer_head_dim()` reports whether the
    installed transformers does so).
    """
    from transformers import Gemma4TextConfig

    return Gemma4TextConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_attention_heads=heads,
        num_key_value_heads=heads // 2,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        vocab_size=vocab,
        vocab_size_per_layer_input=vocab,
        hidden_size_per_layer_input=8,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )


def gemma4_declares_per_layer_head_dim() -> bool:
    """Whether the installed transformers marks Gemma 4's `head_dim` as a per-layer attribute."""
    try:
        config = tiny_gemma4_text_config()
    except ImportError:
        return False
    return "head_dim" in (getattr(config, "per_layer_attributes", None) or ())


def tiny_gemma4_causal(num_layers=6, hidden=32, heads=4, vocab=128, head_dim=8, global_head_dim=16):
    """Build a randomly initialized tiny text-only Gemma 4 causal LM in eval mode.

    Its attention layers alternate head geometry, with `head_dim` on the sliding layers and
    `global_head_dim` on the global layers (and `o_proj` widths to match).
    """
    from transformers import Gemma4ForCausalLM

    config = tiny_gemma4_text_config(
        num_layers=num_layers, hidden=hidden, heads=heads, vocab=vocab,
        head_dim=head_dim, global_head_dim=global_head_dim,
    )
    return Gemma4ForCausalLM(config).eval()


def tiny_gemma4_conditional(num_layers=6, hidden=32, heads=4, vocab=128, head_dim=8, global_head_dim=16):
    """Build a randomly initialized tiny Gemma 4 multimodal wrapper in eval mode.

    Exercises the `model.language_model.layers` root with alternating head geometry (`head_dim` on
    the sliding layers and `global_head_dim` on the global layers).
    """
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    text = tiny_gemma4_text_config(
        num_layers=num_layers, hidden=hidden, heads=heads, vocab=vocab,
        head_dim=head_dim, global_head_dim=global_head_dim,
    )
    vision = {
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
        "image_size": 16, "patch_size": 8,
    }
    audio = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2}
    cfg = Gemma4Config(text_config=text, vision_config=vision, audio_config=audio, mm_tokens_per_image=4)
    return Gemma4ForConditionalGeneration(cfg).eval()


def tiny_lora(model=None, rank=2):
    """Wrap a tiny model in an unmerged LoRA adapter (default: `tiny_llama()`)."""
    from peft import LoraConfig, get_peft_model

    base = model if model is not None else tiny_llama()
    return get_peft_model(base, LoraConfig(r=rank, target_modules=["q_proj", "v_proj"])).eval()


def heterogeneous_head_stub(num_layers=4, hidden=32):
    """A llama-layout stub whose layers alternate head geometry, for head-geometry tests.

    Each layer carries a `self_attn` with a `head_dim` attribute and an `o_proj` `nn.Linear`,
    plus `input_layernorm`/`post_attention_layernorm` as `nn.Identity`, so the layout resolver
    matches `llama_style` (the layers lack `pre_feedforward_layernorm`). Layers alternate
    `head_dim` 4 and 8 with `o_proj.in_features` matched to `num_heads * head_dim`, mirroring
    Gemma 4's sliding/global alternation. The head geometry and PASTA's per-layer head map are
    specified to be read before any forward pass, so `forward` raises to prove that contract.
    """
    head_dims = [4 if i % 2 == 0 else 8 for i in range(num_layers)]
    heads = [hidden // head_dim for head_dim in head_dims]

    class _Attn(nn.Module):
        def __init__(self, head_dim, num_heads):
            super().__init__()
            self.head_dim = head_dim
            self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)

    class _Layer(nn.Module):
        def __init__(self, head_dim, num_heads):
            super().__init__()
            self.self_attn = _Attn(head_dim, num_heads)
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                _Layer(head_dims[i], heads[i]) for i in range(num_layers)
            )

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.config = LlamaConfig(
                hidden_size=hidden,
                num_hidden_layers=num_layers,
                num_attention_heads=heads[0],
            )
            self.config._attn_implementation = "eager"

        @property
        def device(self):
            return torch.device("cpu")

        @property
        def dtype(self):
            return torch.float32

        def parameters(self, recurse=True):
            return super().parameters(recurse=recurse)

        def forward(self, *args, **kwargs):
            raise AssertionError("forward must not be called")

    return _Stub().eval()


def hybrid_attention_stub(num_layers=4, hidden=32, heads=4, full_attention_interval=4):
    """A llama-layout stub whose stack interleaves linear-attention and full-attention layers.

    Mirrors the Qwen3.5 / Qwen3-Next 3:1 hybrid: layers where `(i + 1) % full_attention_interval`
    is nonzero carry a `linear_attn` module (with an `out_proj` `nn.Linear`) and no `self_attn`;
    the others carry a `self_attn` with a `head_dim` attribute and an `o_proj` `nn.Linear`. Every
    layer carries `input_layernorm` and `post_attention_layernorm` as `nn.Identity`, so the layout
    resolver matches `llama_style` and records the full-attention layers in `attention_layer_ids`.
    With the defaults, layer 0 is a linear-attention layer (the case under test) and layer 3 is a
    full-attention layer. Head geometry and hook module names are read before any forward pass, so
    `forward` raises to prove that contract.
    """
    head_dim = hidden // heads

    class _LinearAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_proj = nn.Linear(hidden, hidden, bias=False)

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = head_dim
            self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    class _Layer(nn.Module):
        def __init__(self, layer_idx):
            super().__init__()
            if (layer_idx + 1) % full_attention_interval != 0:
                self.layer_type = "linear_attention"
                self.linear_attn = _LinearAttn()
            else:
                self.layer_type = "full_attention"
                self.self_attn = _Attn()
            self.input_layernorm = nn.Identity()
            self.post_attention_layernorm = nn.Identity()

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(_Layer(i) for i in range(num_layers))

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.config = LlamaConfig(
                hidden_size=hidden,
                num_hidden_layers=num_layers,
                num_attention_heads=heads,
            )
            self.config._attn_implementation = "eager"

        @property
        def device(self):
            return torch.device("cpu")

        @property
        def dtype(self):
            return torch.float32

        def parameters(self, recurse=True):
            return super().parameters(recurse=recurse)

        def forward(self, *args, **kwargs):
            raise AssertionError("forward must not be called")

    return _Stub().eval()


def tiny_qwen3_next(num_layers=4, hidden=32, heads=2, vocab=100):
    """Build a randomly initialized hub-free tiny Qwen3-Next model in eval mode.

    A real `Qwen3NextForCausalLM` with a 3:1 hybrid stack (three Gated DeltaNet `linear_attn`
    layers before each `self_attn` layer). Qwen3.5's text stack subclasses Qwen3-Next's, so this
    exercises the same decoder-layer shape. Dense MLPs on every layer keep MoE out of the picture,
    and the Gated DeltaNet layers run on transformers' pure-torch kernels when `fla` and
    `causal_conv1d` are absent.
    """
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    head_dim = hidden // heads
    cfg = Qwen3NextConfig(
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=head_dim,
        vocab_size=vocab,
        max_position_embeddings=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=hidden,
        shared_expert_intermediate_size=hidden,
        mlp_only_layers=list(range(num_layers)),
        layer_types=[
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(num_layers)
        ],
    )
    return Qwen3NextForCausalLM(cfg).eval()


def wordlevel_tokenizer(
    words=("the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "attention", "span"),
    single="<s> $A",
):
    """BOS-prepending WordLevel tokenizer with offset mapping.

    Mimics Llama-style specials. The `single` template controls how many special
    tokens are prepended (e.g. "<s> <s> $A" for a multi-special-prefix test).
    """
    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single=single, pair="<s> $A $B", special_tokens=[("<s>", 0)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>"
    )


def reasoning_tag_tokenizer(
    special_tags: tuple[str, ...] = (),
    ordinary_tags: tuple[str, ...] = (),
    words: tuple[str, ...] = ("R", "A", "thought", "plan", "answer", "pre", "stop", "x"),
):
    """Tiny WordLevel tokenizer with reasoning delimiters as configurable tokens.

    The base vocabulary is a handful of whitespace-separated words plus `<pad>`/`<eos>`. Each tag in
    `special_tags` is added as an atomic special token (stripped by `skip_special_tokens=True`, as a
    special-token delimiter such as Gemma's `<|channel>`/`<channel|>` is). Each tag in
    `ordinary_tags` is added as an atomic ordinary token (round-trips through
    `skip_special_tokens=True`, as `<think>`/`</think>` do on Qwen and Granite). Tags are added
    atomically, so they encode to a single id regardless of the whitespace pre-tokenizer.

    Args:
        special_tags: Delimiter strings to register as special tokens.
        ordinary_tags: Delimiter strings to register as ordinary added tokens.
        words: The base vocabulary of ordinary whitespace-separated words.

    Returns:
        A `PreTrainedTokenizerFast` with `pad_token="<pad>"` and `eos_token="<eos>"`.
    """
    vocab = {"<pad>": 0, "<eos>": 1, **{word: index + 2 for index, word in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")
    if ordinary_tags:
        fast.add_tokens([AddedToken(tag, special=False, normalized=False) for tag in ordinary_tags])
    if special_tags:
        fast.add_special_tokens(
            {"additional_special_tokens": [AddedToken(tag, special=True, normalized=False) for tag in special_tags]}
        )
    return fast


def byte_level_tokenizer(chat_template: bool = True):
    """Build a byte-level tokenizer with no merges (one token per byte) plus `<pad>`, `<eos>`, and `<bos>`.

    Any string tokenizes without training or vocabulary files, and decoding returns the original
    text. The offset mapping is exact on ASCII text, which PASTA's substring mapping relies on.
    `<bos>` is prepended to each encoding. With `chat_template`, a minimal role-prefixed template
    is attached so `messages=` prompts render.
    """
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, **{char: index + 3 for index, char in enumerate(sorted(alphabet))}}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A", pair="<bos> $A $B", special_tokens=[("<bos>", 2)],
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>")
    if chat_template:
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
        )
    return tokenizer


def tiny_gemma4_grid():
    """Build the `(model, tokenizer)` pair of the `gemma4` entry in `tests/utils/ci_models.yaml`.

    The model is a six-layer dense Gemma 4 whose global attention layer uses twice the head
    dimension of the sliding layers, paired with `byte_level_tokenizer()`. It is built in process
    under a fixed seed, which adds a heterogeneous head geometry to the control grid without a Hub
    download.
    """
    from transformers import Gemma4ForCausalLM

    tokenizer = byte_level_tokenizer()
    config = tiny_gemma4_text_config(
        num_layers=6, hidden=32, heads=4, vocab=len(tokenizer), head_dim=8, global_head_dim=16,
    )
    # a `.spipe` records this reference as the model, as it records a hub id for downloaded models
    config.name_or_path = "local:tiny_gemma4_grid"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = Gemma4ForCausalLM(config).eval()
    return model, tokenizer
