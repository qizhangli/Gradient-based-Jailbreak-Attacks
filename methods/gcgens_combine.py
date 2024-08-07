from .utils import *
import time
import copy

def get_hook(act):
    def _hook(m, i, o):
        act["act"] = i[0].data
    return _hook

def get_lsgm_hook(gamma):
    def lsgm_hook(m, i, o):
        i[0].data = gamma * i[0].data
    return lsgm_hook

def get_change_grad_hook(diff, tok):
    def change_grad_hook(m, i, o):
        i[0].data[:, tok, :] = (i[0].data[:, tok, :].norm(p=2, dim=(1,2), keepdim=True) * F.normalize(diff[:, tok, :].float(), dim=(1,2), p=2)).half()
    return change_grad_hook

def test_wb_ens(record_dict, goal_ls, best_control, target_ls, tokenizer, model_path, model):
    response_ls, if_match_ls, if_jb_long_ls, if_jb_short_ls = [], [], [], []
    for goal, target in zip(goal_ls, target_ls):
        response, if_match, if_jb_long, if_jb_short = test_wb(record_dict, goal, best_control, target, tokenizer, model_path, model,max_new_tokens=20)
        response_ls.append(response)
        if_match_ls.append(if_match)
        if_jb_long_ls.append(if_jb_long)
        if_jb_short_ls.append(if_jb_short)
    return response_ls, if_match_ls, if_jb_long_ls, if_jb_short_ls
        

def gcgens_combine(args, model, tokenizer, pair_ls):
    control = pair_ls[0][1]
    goal_ls, target_ls = [], []
    for pair in pair_ls:
        goal_ls.append(pair[0])
        target_ls.append(pair[2])

    curr_control = control
    best_control = control
    loss_best = 10000

    record_dict = initial_record_dict(goal_ls, target_ls, curr_control, loss_best)
    response_ls, if_match_ls, if_jb_long_ls, if_jb_short_ls = test_wb_ens(record_dict, goal_ls, best_control, target_ls, tokenizer, args.model_path, model)
    update_record_dict_test(record_dict, response_ls, if_match_ls, if_jb_long_ls, if_jb_short_ls, best_control, loss_best)
    log(args.log_file, record_dict)

    indices_nonascii = get_nonascii_toks(tokenizer, device=model.device)

    num_behaviors = len(goal_ls)
    steps_since_last_added_behavior = 0

    with torch.no_grad():
        act_init = dict()
        for j, (goal, target) in enumerate(zip(goal_ls[:num_behaviors], target_ls[:num_behaviors])):
            init_prompt, _, _, _, target_slice = get_prompt(goal, record_dict["controls"][0], target, tokenizer, args.model_path)
            loss_slice = slice(target_slice.start-1, target_slice.stop-1)
            init_inputs = tokenizer([init_prompt], return_tensors="pt").to(model.device)
            act_init[j] = dict()
            model.model.layers[args.lila_layer].register_forward_hook(get_hook(act_init[j]))

            model(**init_inputs)
            remove_forward_hooks(model)

    for i in range(args.steps):
        since = time.time()

        prompt_ls, goal_slice_ls, control_slice_ls, target_slice_ls, curr_ids_ls = [], [], [], [], []
        grad = 0
        for j, (goal, target) in enumerate(zip(goal_ls[:num_behaviors], target_ls[:num_behaviors])):
            prompt, _, goal_slice, control_slice, target_slice = get_prompt(goal, curr_control, target, tokenizer, args.model_path)
            curr_ids = tokenizer([prompt], return_tensors="pt").input_ids

            act_curr = dict()
            model.model.layers[args.lila_layer].register_forward_hook(get_hook(act_curr))
            with torch.no_grad():
                model(curr_ids.to(model.device), None)
            remove_forward_hooks(model)

            if i > 0:
                model.model.layers[args.lila_layer].register_full_backward_hook(get_change_grad_hook((act_init[j]["act"] - act_curr["act"]), 
                                                                                    tok = slice(target_slice.start-1, target_slice.start)))
            for name, module in model.named_modules():
                if "post_attention_layernorm" in name or "input_layernorm" in name:
                    module.register_full_backward_hook(get_lsgm_hook(args.gamma))
            grad += token_gradients(model, curr_ids[0].to(model.device), control_slice, target_slice)
            remove_backward_hooks(model)

            prompt_ls.append(prompt)
            goal_slice_ls.append(goal_slice)
            control_slice_ls.append(control_slice)
            target_slice_ls.append(target_slice)
            curr_ids_ls.append(curr_ids)
        
        grad = grad / num_behaviors
        grad = grad / grad.norm(dim=-1, keepdim=True)

        c_cands_sample = 0
        while True:
            cands_ids = sample_control(tokenizer, curr_ids[0, control_slice], grad, 
                                        batch_size=args.batch_size*(c_cands_sample+1), topk=args.topk*(c_cands_sample+1), indices_nonascii=indices_nonascii)
            _, _, init_goal_slice, init_control_slice, init_target_slice = get_prompt(goal, curr_control, target, tokenizer, args.model_path)
            init_slices = [init_goal_slice, init_control_slice, init_target_slice]
            cands = get_filtered_cands(tokenizer, cands_ids, True, curr_control, goal, target, args.model_path, init_slices)
            if cands != None:
                break
            else:
                c_cands_sample += 1
                print("cands == None")
        losses_ls = []
        for (curr_ids, control_slice, target_slice) in zip(curr_ids_ls, control_slice_ls, target_slice_ls):
            losses = get_cand_losses(curr_ids[0].to(model.device), control_slice, target_slice, cands, tokenizer, model)
            losses_ls.append(losses)
        losses_ls = torch.stack(losses_ls)
        avg_losses = losses_ls.mean(0)
        
        min_arg = avg_losses.argmin()
        curr_control = cands[min_arg]
        loss_cur = avg_losses[min_arg].item()
        if loss_cur < loss_best:
            best_control = curr_control
            loss_best = loss_cur
        print("Step {}, Current Loss {:.4f}, Best Loss {:.4f}, Time {:.1f}".format(i, loss_cur, loss_best, time.time() - since))
        print(curr_control)
        del cands_ids ; gc.collect()
        update_record_dict_train(record_dict, curr_control, best_control, loss_cur, loss_best)
        if (i+1) % 100 == 0:
            response_ls, if_match_ls, if_jb_long_ls, if_jb_short_ls = test_wb_ens(record_dict, goal_ls, best_control, target_ls, tokenizer, args.model_path, model)
            update_record_dict_test(record_dict, response_ls, if_match_ls, if_jb_long_ls, if_jb_short_ls, best_control, loss_best)
            log(args.log_file, record_dict)

