from .utils import *
import time
import copy

def get_hook(act):
    def _hook(m, i, o):
        act["act"] = i[0].data
    return _hook

def get_change_grad_hook(diff, tok):
    def change_grad_hook(m, i, o):
        i[0].data[:, tok, :] = (i[0].data[:, tok, :].norm(p=2, dim=(1,2), keepdim=True) * F.normalize(diff[:, tok, :].float(), dim=(1,2), p=2)).half()
    return change_grad_hook

def autoprompt_lila(args, model, tokenizer, pair):
    goal, control, target = pair
    curr_control = control
    best_control = control
    loss_best = 10000
    
    record_dict = initial_record_dict(goal, target, curr_control, loss_best)
    response, if_match, if_jb_long, if_jb_short = test_wb(record_dict, goal, best_control, target, tokenizer, args.model_path, model)
    update_record_dict_test(record_dict, response, if_match, if_jb_long, if_jb_short, best_control, loss_best)
    log(args.log_file, record_dict)

    indices_nonascii = get_nonascii_toks(tokenizer, device=model.device)
    _, _, init_goal_slice, init_control_slice, init_target_slice = get_prompt(goal, curr_control, target, tokenizer, args.model_path)
    init_slices = [init_goal_slice, init_control_slice, init_target_slice]

    with torch.no_grad():
        init_prompt, _, _, _, target_slice = get_prompt(goal, record_dict["controls"][0], target, tokenizer, args.model_path)
        loss_slice = slice(target_slice.start-1, target_slice.stop-1)
        init_inputs = tokenizer([init_prompt], return_tensors="pt").to(model.device)
        act_init = dict()
        model.model.layers[args.lila_layer].register_forward_hook(get_hook(act_init))
        model(**init_inputs)
        remove_forward_hooks(model)

    for i in range(args.steps):
        since = time.time()
        prompt, _, goal_slice, control_slice, target_slice = get_prompt(goal, curr_control, target, 
                                                                            tokenizer, args.model_path)
        curr_ids = tokenizer([prompt], return_tensors="pt").input_ids
        
        act_curr = dict()
        model.model.layers[args.lila_layer].register_forward_hook(get_hook(act_curr))
        with torch.no_grad():
            model(curr_ids.to(model.device), None)
        remove_forward_hooks(model)

        if i > 0:
            model.model.layers[args.lila_layer].register_full_backward_hook(get_change_grad_hook((act_init["act"] - act_curr["act"]), 
                                                                            tok=slice(target_slice.start, target_slice.start+1)))
        grad = token_gradients(model, curr_ids[0].to(model.device), control_slice, target_slice)
        remove_backward_hooks(model)
        c_cands_sample = 0
        while True:
            cands_ids = sample_control_autoprompt(tokenizer, curr_ids[0, control_slice], grad, 
                                        batch_size=args.batch_size*(c_cands_sample+1), topk=args.topk*(c_cands_sample+1), indices_nonascii=indices_nonascii)
            cands = get_filtered_cands(tokenizer, cands_ids, True, curr_control, goal, target, args.model_path, init_slices)
            if cands != None:
                break
            else:
                c_cands_sample += 1
                print("cands == None")
        losses = get_cand_losses(curr_ids[0].to(model.device), control_slice, target_slice, cands, tokenizer, model)
        curr_control = cands[losses.argmin()]
        loss_cur = losses.min().item()

        if loss_cur < loss_best:
            best_control = curr_control
            loss_best = loss_cur
        print("Step {}, Current Loss {:.4f}, Best Loss {:.4f}, Time {:.1f}".format(i, loss_cur, loss_best, time.time() - since))
        print(curr_control)
        del cands_ids ; gc.collect()
        update_record_dict_train(record_dict, curr_control, best_control, loss_cur, loss_best)
        if (i+1) % 50 == 0:
            response, if_match, if_jb_long, if_jb_short = test_wb(record_dict, goal, best_control, target, tokenizer, args.model_path, model)
            update_record_dict_test(record_dict, response, if_match, if_jb_long, if_jb_short, best_control, loss_best)
            log(args.log_file, record_dict)

