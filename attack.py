import os
import argparse
import torch
import random
import time

import numpy as np
from torch.backends import cudnn
from methods import utils
from methods import *

def main(args):
    pair_ls = utils.get_pairs()

    if "gcgens" in args.method:
        pair = pair_ls[:args.n_ens]
    else:
        pair = pair_ls[args.data_index]

    model, tokenizer = utils.prep_model(args)
    print("\n----------------------")
    print("method: ", args.method)
    print("----------------------\n")
    eval("{}(args, model, tokenizer, pair)".format(args.method))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llama2")
    parser.add_argument("--data_index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--savedir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="gcg")

    args = parser.parse_args()


    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    if "gcgens_combine" in args.method:
        args.gamma = float(args.method.split("_")[-3])
        args.lila_layer = int(args.method.split("_")[-2])
        args.n_ens = int(args.method.split("_")[-1])
        args.method = "gcgens_combine"
    elif "gcgens" in args.method:
        args.n_ens = int(args.method.split("_")[-1])
        args.method = "gcgens"
        

    elif "gcg_lila" in args.method:
        args.lila_layer = int(args.method.split("_")[-1])
        args.method = "gcg_lila"
    elif "gcg_lsgm" in args.method:
        args.gamma = float(args.method.split("_")[-1])
        args.method = "gcg_lsgm"
    elif "gcg_combine" in args.method:
        args.gamma   = float(args.method.split("_")[-2])
        args.lila_layer = int(args.method.split("_")[-1])
        args.method = "gcg_combine"
    elif "gcg" in args.method:
        args.method = "gcg"
    

    elif "autoprompt_lila" in args.method:
        args.lila_layer = int(args.method.split("_")[-1])
        args.method = "autoprompt_lila"
    elif "autoprompt_lsgm" in args.method:
        args.gamma   = float(args.method.split("_")[-1])
        args.method = "autoprompt_lsgm"
    elif "autoprompt_combine" in args.method:
        args.gamma   = float(args.method.split("_")[-2])
        args.lila_layer = int(args.method.split("_")[-1])
        args.method = "autoprompt_combine"
    elif "autoprompt" in args.method:
        args.method = "autoprompt"

    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    args.log_file = f"{args.savedir}_{timestamp}.json"

    if args.model_name == "llama2":
        args.model_path = "meta-llama/Llama-2-7b-chat-hf"
    elif args.model_name == "llama2-13b":
        args.model_path = "meta-llama/Llama-2-13b-chat-hf"
    elif args.model_name == "mistral":
        args.model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    
    
    main(args)