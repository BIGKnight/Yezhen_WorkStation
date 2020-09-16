import torch
from utils.losses import *

def run_iter_irm(
    inputs,
    models, optimizers,
    meters,
    args,
    global_step,
    entire_steps
):
    net = models['net']
    optimizer = optimizers['main']
    
    src_inputs = inputs['src']['sample_1_q'][0].cuda()
    src_labels = inputs['src']['sample_1_q'][1].cuda()
    l_tgt_inputs = inputs['l_tgt']['sample_1_q'][0].cuda()
    l_tgt_labels = inputs['l_tgt']['sample_1_q'][1].cuda()
    ul_tgt_inputs = inputs['ul_tgt']['sample_1_q'][0].cuda()
    is_mask = len(inputs['l_tgt']['sample_1_q'])> 2
    src_masks = inputs['src']['sample_1_q'][2].cuda() if is_mask else 1
    l_tgt_masks = inputs['l_tgt']['sample_1_q'][2].cuda() if is_mask else 1
    src_inputs = src_inputs * src_masks
    l_tgt_inputs = l_tgt_inputs * l_tgt_masks
    
    src_outputs = net(src_inputs, temp=1, cosine=args.cosine)
    l_tgt_outputs = net(l_tgt_inputs, temp=1, cosine=args.cosine)
    src_logits = src_outputs['output_logits']
    l_tgt_logits = l_tgt_outputs['output_logits']
    src_logits = src_logits * src_masks
    l_tgt_logits = l_tgt_logits * l_tgt_masks

    
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
    loss_inv = loss_cls_src + loss_cls_tgt
    
    
    # irm penalty loss
    loss_irm = 0
    loss_irm += irm_penalty(
        src_logits, src_labels, 
        supervised_loss, 
        [args.num_cls, args.batch_size, global_step, entire_steps, args.annealing, args.task_type]
    ) / 2.
    loss_irm += irm_penalty(
        l_tgt_logits, l_tgt_labels, 
        supervised_loss, 
        [args.num_cls, args.batch_size, global_step, entire_steps, args.annealing, args.task_type]
    ) / 2.
    
    total_loss = loss_inv + args.trade_off * (loss_irm)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    # update meters
    meters['src_cls_loss'].update(loss_cls_src.item())
    meters['tgt_cls_loss'].update(loss_cls_tgt.item())
    meters['irm_loss'].update(loss_irm.item())