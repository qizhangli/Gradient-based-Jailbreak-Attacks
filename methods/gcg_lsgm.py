from .utils import *
import time

def get_lsgm_hook(gamma):
    def lsgm_hook(m, i, o):
        i[0].data = gamma * i[0].data
    return lsgm_hook

def gcg_lsgm(args, model, tokenizer, pair):
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
    for name, module in model.named_modules():
        if "post_attention_layernorm" in name or "input_layernorm" in name:
            print(name)
            module.register_full_backward_hook(get_lsgm_hook(args.gamma))

    for i in range(args.steps):
        since = time.time()
        prompt, _, goal_slice, control_slice, target_slice = get_prompt(goal, curr_control, target, tokenizer, args.model_path)
        curr_ids = tokenizer([prompt], return_tensors="pt").input_ids
        grad = token_gradients(model, curr_ids[0].to(model.device), control_slice, target_slice)

        c_cands_sample = 0
        while True:
            cands_ids = sample_control(tokenizer, curr_ids[0, control_slice], grad, 
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

