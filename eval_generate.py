
import os
import argparse
import math
import json
import torch

from methods import utils
from transformers import AutoTokenizer


@torch.inference_mode()
def get_output(args, pair_ls, model, tokenizer):
    
    prompt_ls = []
    for pair in pair_ls:
        goal, control, target = pair
        prompt, _, _, _, _ = utils.get_prompt(goal, control, "", tokenizer, args.model_path)
        prompt_ls.append(prompt)
        
    inputs = tokenizer(prompt_ls, return_tensors="pt", padding=True).to("cuda")

    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
    )
    
    outputs = []
    for i in range(output_ids.size(0)):
        if model.config.is_encoder_decoder:
            output_ids_one = output_ids[i]
        else:
            output_ids_one = output_ids[i][len(inputs["input_ids"][i]) :]
            
        output_one = tokenizer.decode(
            output_ids_one, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        outputs.append(output_one)
        
    return outputs

def get_test_cases(log_dir, steps, test_slice = slice(10, 520)):
    read_filename_ls = os.listdir(log_dir)
    filename_ls = []
    for filename in read_filename_ls:
        if "exp_" in filename:
            filename_ls.append(filename)
    assert len(filename_ls) == 1

    with open(os.path.join(log_dir, filename_ls[0]), "r") as f:
        curr_controls = json.load(f)["curr_controls"]
        control = curr_controls[steps-1]
    pair_ls = utils.get_pairs(control_init=control)
    return pair_ls[test_slice]

def generate_and_save(args,mode, model, tokenizer, pair_ls):

    save_dir = args.log_path + "/A_goal_response_max{}_steps{}.json".format(args.max_new_tokens, args.steps)
    
    def log():
        with open(save_dir, "w") as f:
            json.dump(json_data, f, indent=4)
    
    pair_batches = [pair_ls[i*args.chunk_size: (i+1)*args.chunk_size] for i in range(math.ceil(len(pair_ls) / args.chunk_size))]

    goal_ls, response_ls = [], []
    json_data = {}
    preview_sure = 0
    for i, pair_b in enumerate(pair_batches):
        print(i)
        output_b = get_output(args, pair_b, model, tokenizer)
        for (pair_one, output) in zip(pair_b, output_b):
            goal = pair_one[0]
            goal_ls.append(goal)
            response_ls.append(output)
    json_data = {"goals":goal_ls, "responses":response_ls}
    log()
        
    print(save_dir)
    

def main(args):
    pair_ls = get_test_cases(args.log_path, args.steps)
    model, tokenizer = utils.prep_model(args)
    generate_and_save(args,"all", model, tokenizer, pair_ls)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--chunk_size", type=int, default=10)
    
    args = parser.parse_args()


    if "llama2-13b" in args.log_path:
        args.model_name = "llama2-13b"
    elif "llama2" in args.log_path:
        args.model_name = "llama2"
    elif "mistral" in args.log_path:
        args.model_name = "mistral"


    if args.model_name == "llama2":
        args.model_path = "meta-llama/Llama-2-7b-chat-hf"
    elif args.model_name == "llama2-13b":
        args.model_path = "meta-llama/Llama-2-13b-chat-hf"
    elif args.model_name == "mistral":
        args.model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    
    main(args)