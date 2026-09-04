"""Hub-free tiny model factories for state control tests.

Provides a randomly initialized tiny Llama and a hand-built WordLevel tokenizer so
that hook-level behavioral tests can run without downloading models from the HF Hub.
"""
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


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
