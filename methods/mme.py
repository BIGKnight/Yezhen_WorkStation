import torch
from utils.losses import *

def run_iter_mme(
    inputs,
    models, optimizers,
    meters,
    args,
    global_step,
    entire_steps
):
    net = models['net']
    classifier_optimizer = optimizers['classifier']
    encoder_optimizer = optimizers['encoder']
    main_optimizer = optimizers['main']
    
    src_inputs = inputs['src']['sample_1_q'][0].cuda()
    src_labels = inputs['src']['sample_1_q'][1].cuda()
    l_tgt_inputs = inputs['l_tgt']['sample_1_q'][0].cuda()
    l_tgt_labels = inputs['l_tgt']['sample_1_q'][1].cuda()
    ul_tgt_inputs = inputs['ul_tgt']['sample_1_q'][0].cuda()
    
    src_outputs = net(src_inputs, temp=args.temp, cosine=args.cosine)
    l_tgt_outputs = net(l_tgt_inputs, temp=args.temp, cosine=args.cosine)
    ul_tgt_outputs = net(ul_tgt_inputs, temp=args.temp, cosine=args.cosine, reverse=True)
    src_logits = src_outputs['output_logits']
    l_tgt_logits = l_tgt_outputs['output_logits']
    ul_tgt_features, ul_tgt_logits = ul_tgt_outputs['adapted_layer'], ul_tgt_outputs['output_logits']
    
    total_loss = 0
     # classification loss
    loss_cls_src = supervised_loss(
        src_logits, src_labels, 
        args.num_cls, args.batch_size, 
        global_step, entire_steps, 
        args.annealing,
        args.task_type
    ) / 2.
    loss_cls_tgt = supervised_loss(
        l_tgt_logits, l_tgt_labels, 
        args.num_cls, args.batch_size, 
        global_step, entire_steps, 
        args.annealing,
        args.task_type
    ) / 2.
    loss_cls = loss_cls_src + loss_cls_tgt
    
    ul_tgt_scores = torch.nn.functional.softmax(ul_tgt_logits, dim=1)
    loss_adent = -args.trade_off * torch.mean(torch.sum(ul_tgt_scores *(torch.log(ul_tgt_scores + 1e-5)), 1))
    
    total_loss = loss_cls - loss_adent
    main_optimizer.zero_grad()
    total_loss.backward()
    main_optimizer.step()
    
    # update meters
    meters['src_cls_loss'].update(loss_cls_src.item())
    meters['tgt_cls_loss'].update(loss_cls_tgt.item())
    meters['transfer_loss'].update(loss_adent.item())