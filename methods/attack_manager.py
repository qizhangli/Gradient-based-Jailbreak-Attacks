import torch
import transformers
from transformers import LlamaForCausalLM


def get_embedding_matrix(model):
    if isinstance(model, (LlamaForCausalLM, transformers.models.mistral.modeling_mistral.MistralForCausalLM)):
        return model.model.embed_tokens.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_embeddings(model, input_ids):
    if isinstance(model, (LlamaForCausalLM, transformers.models.mistral.modeling_mistral.MistralForCausalLM)):
        return model.model.embed_tokens(input_ids)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_nonascii_toks(tokenizer, device='cpu'):
    def is_ascii(s):
        return s.isascii() and s.isprintable()
    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)
    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)
    return torch.tensor(ascii_toks, device=device)
