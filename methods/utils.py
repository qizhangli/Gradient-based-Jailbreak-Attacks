import os
import json
import torch
import csv
import time

import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple, Union
from .attack_manager import get_embedding_matrix, get_embeddings, get_nonascii_toks
import numpy as np
import gc
from .test_prefixes import _test_prefixes, _test_prefixes_short

def get_prompt(goal, control, target, tokenizer, model_path, sep=None):
    if "Llama-2" in model_path or "Mistral" in model_path:
        prompt = "[INST]"
        toks = tokenizer(prompt).input_ids
        _user_role_slice = slice(None, len(toks))
        prompt += f" {goal}"
        toks = tokenizer(prompt).input_ids
        _goal_slice = slice(_user_role_slice.stop, len(toks))
        seperator = " " if control else ""
        prompt += f"{seperator}{control}"
        toks = tokenizer(prompt).input_ids
        _control_slice = slice(_goal_slice.stop, len(toks))
        prompt += " [/INST]"
        toks = tokenizer(prompt).input_ids
        _assistant_role_slice = slice(_control_slice.stop, len(toks))
        if sep == None:
            seperator = " " if target else ""
        else:
            seperator = ""
        prompt += f"{seperator}{target}"
        toks = tokenizer(prompt).input_ids
        _target_slice = slice(_assistant_role_slice.stop, len(toks))
    else:
        raise RuntimeError("check model_path")
    return prompt, toks, _goal_slice, _control_slice, _target_slice

def get_pairs(data_file="./data/advbench/harmful_behaviors.csv", 
              control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
              here=False):
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        pairs = []
        next(reader)
        for i, line in enumerate(reader):
            goal = line[0]
            target = line[1].replace("Sure, here is", "Sure, here's")
            if here:
                target = target.replace("Sure, here's", "Here's")
            pairs.append([goal, control_init, target])
    return pairs

def prep_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_auth_token=True,
            use_fast=False,
        )

    model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_auth_token=True,
            low_cpu_mem_usage=True, 
            use_cache= False,
        ).eval()

    if args.model_name in ["llama2", "llama2-13b"]:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    return model, tokenizer

def token_gradients(model, input_ids, control_slice, target_slice):
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    embed_weights = model.model.embed_tokens.weight
        
    one_hot = torch.zeros(
        input_ids[control_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[control_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    embeds = model.model.embed_tokens(input_ids.unsqueeze(0)).detach()

    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    full_embeds = torch.cat(
        [
            embeds[:,:control_slice.start,:], 
            input_embeds, 
            embeds[:,control_slice.stop:,:]
        ], 
        dim=1)
    
    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss_none = F.cross_entropy(logits[0,loss_slice,:], targets, reduction="none")
    loss = loss_none.mean()
    grad = torch.autograd.grad(loss, [one_hot, ])[0].data
    return grad
    
def sample_control_autoprompt(tokenizer, control_toks, grad, batch_size, topk, allow_non_ascii=False, indices_nonascii=None):
    
    if not allow_non_ascii:
        grad[:, indices_nonascii] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)

    ### Note that this is different than GCG; in AutoPrompt, we choose a fix random pos instead of best ones
    new_token_pos = torch.randint(0, len(control_toks), (1,), device=grad.device)
    new_token_pos = new_token_pos.repeat(batch_size)

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def sample_control(tokenizer, control_toks, grad, batch_size, topk, allow_non_ascii=False, indices_nonascii=None):
    if not allow_non_ascii:
        grad[:, indices_nonascii] = np.infty
    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)
    original_control_toks = control_toks.repeat(batch_size, 1)
    assert batch_size % len(control_toks) == 0
    num_per_pos = batch_size // len(control_toks)
    new_token_pos = torch.arange(len(control_toks))[:, None].repeat(1, num_per_pos).view(-1).to(grad.device)

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.cat([torch.randperm(topk)[:num_per_pos] for _ in range(len(control_toks))], dim=0)[:, None].to(grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    return new_control_toks

def get_cand_losses(input_ids, control_slice, target_slice, cands, tokenizer, model):

    max_len = control_slice.stop - control_slice.start
    test_ids = [
        torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=input_ids.device)
        for control in cands
    ]
    pad_tok = 0
    while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
        pad_tok += 1
    nested_ids = torch.nested.nested_tensor(test_ids)
    test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(input_ids.device)
    ids = torch.scatter(
            input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1),
            1,
            locs,
            test_ids
        )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None
    
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=attn_mask)
    logits = outputs.logits
    logits_for_loss = logits[:, loss_slice, :]
    losses_none = F.cross_entropy(logits_for_loss.transpose(1,2), ids[:, target_slice], reduction="none")
    losses = losses_none.mean(1)
    return losses

def remove_backward_hooks(model):
    for name, module in model.named_modules():
        if len(module._backward_hooks) > 0:
            module._backward_hooks.clear()

def remove_forward_hooks(model):
    for name, module in model.named_modules():
        if len(module._forward_hooks) > 0:
            module._forward_hooks.clear()


def get_filtered_cands(tokenizer, control_cand, filter_cand, curr_control, goal, target, model_path, init_slices):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                _, _, goal_slice, control_slice, target_slice = get_prompt(goal, decoded_str, target, tokenizer, model_path)
                if goal_slice == init_slices[0] and control_slice == init_slices[1] and target_slice == init_slices[2]:
                    cands.append(decoded_str)
                else:
                    count += 1
            else:
                count += 1
        else:
            cands.append(decoded_str)
    
    if filter_cand:
        if len(cands) != 0:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        else:
            return None
    return cands

def log(log_file, save_dict):
    with open(log_file, "w") as f:
        json.dump(save_dict, f, indent=4)

def initial_record_dict(goal, target, control_init, loss_init):
    record_dict = {"params": {"goals": [goal] if not isinstance(goal, list) else goal, 
                              "targets": [target]} if not isinstance(target, list) else goal,
                   "controls": [],
                   "losses": [],
                   "responses": [],
                   "if_match": [],
                   "if_jb_long": [],
                   "if_jb_short": [],
                   "curr_controls": [],
                   "curr_losses": [],
                   "best_controls": [],
                   "best_losses": [],
                   "test_prefixes_long": _test_prefixes,
                   "test_prefixes_short": _test_prefixes_short,
                   }
    return record_dict

def update_record_dict_train(record_dict, curr_control, best_control, loss_cur, loss_best):
    record_dict["curr_controls"].append(curr_control)
    record_dict["curr_losses"].append(loss_cur)
    record_dict["best_controls"].append(best_control)
    record_dict["best_losses"].append(loss_best)

def update_record_dict_test(record_dict, response, if_match, if_jb_long, if_jb_short, best_control, loss_best):
    record_dict["responses"].append(response)
    record_dict["if_match"].append(if_match)
    record_dict["if_jb_long"].append(if_jb_long)
    record_dict["if_jb_short"].append(if_jb_short)
    record_dict["controls"].append(best_control)
    record_dict["losses"].append(loss_best)
    


def test_wb(record_dict, goal, control, target, tokenizer, model_path, model,max_new_tokens=512):
    prompt, _, _, _, _ = get_prompt(goal, control, "", tokenizer, model_path)
    inputs_test = tokenizer([prompt], return_tensors="pt").to(model.device)
    response = model.generate(**inputs_test,
                                do_sample=False,
                                max_new_tokens=max_new_tokens)[0]
    response = tokenizer.decode(
        response[len(inputs_test["input_ids"][0]) :], skip_special_tokens=True, spaces_between_special_tokens=False
    )

    if_match = (target in response)
    if_jb_long = not any([prefix in response for prefix in _test_prefixes])
    if_jb_short = not any([prefix in response for prefix in _test_prefixes_short])
    
    return response, if_match, if_jb_long, if_jb_short

