import os
import argparse
import math
import json
import torch
import random
import copy

import numpy as np

def read_filenames(logdir):
    read_filename_ls = os.listdir(logdir)
    filename_ls = []
    for filename in read_filename_ls:
        if "exp_" in filename:
            filename_ls.append(filename)
    print("n_filename: ", len(filename_ls))
    return filename_ls

def check_file(filename_ls, log_dir, steps=500):
    index_ls = []
    loss_ls = []
    loss_traj_ls = []
    if_match_traj_ls = []
    if_match_ls = []
    response_ls = []
    goal_ls = []
    for filename in filename_ls:
        with open(os.path.join(log_dir, filename), "r") as f:
            json_dict = json.load(f)
        if len(json_dict["controls"]) >= (1 + (steps//50)):
            index = int(filename.split("_ind")[1].split("_")[0])
            index_ls.append(index)
            loss_ls.append(json_dict["losses"][steps//50])
            if_match_ls.append(json_dict["if_match"][steps//50])
            response_ls.append(json_dict["responses"][steps//50])
            goal_ls.append(json_dict["params"]["goals"][0])
        else:
            print("WARNING ", filename)

    sort_arg = np.array(index_ls).argsort()
    print(set(np.arange(100).tolist()) ^ set(np.array(index_ls)[sort_arg].tolist()))
    goal_ls = np.array(goal_ls)[sort_arg].tolist()
    response_ls = np.array(response_ls)[sort_arg].tolist()
    n_match = sum(if_match_ls)
    loss = sum(loss_ls) / len(loss_ls)


    print("Loss", sum(loss_ls) / len(loss_ls), len(loss_ls))
    print("Match Rate", sum(if_match_ls) / len(if_match_ls), len(if_match_ls))
    return goal_ls, response_ls, n_match, loss


def main(args):
    log_dir = args.log_path
    filename_ls = read_filenames(log_dir)
    goal_ls, response_ls, n_match, loss = check_file(filename_ls, log_dir, args.steps)
    save_dict = {"n_match": n_match, "loss": loss, "goals": goal_ls, "responses": response_ls}
    with open(log_dir+"/A_goal_response_max512_steps{}.json".format(args.steps).format(args.steps), "w") as f:
        json.dump(save_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=500)
    
    args = parser.parse_args()

    
    main(args)