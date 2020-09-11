import torch
from utils.losses import *

def run_iter_dann(
    inputs,
    models, optimizers,
    meters,
    args,
    global_step,
    entire_steps
):
    net = models['net']
    ad_net = models['ad_net']
    main_optimizer = optimizers['main']
    dis_optimizer = optimizers['dis']
    
    src_inputs = inputs['src']['sample_1_q'][0].cuda()
    src_labels = inputs['src']['sample_1_q'][1].cuda()
    l_tgt_inputs = inputs['l_tgt']['sample_1_q'][0].cuda()
    l_tgt_labels = inputs['l_tgt']['sample_1_q'][1].cuda()
    ul_tgt_inputs = inputs['ul_tgt']['sample_1_q'][0].cuda()
    
    src_outputs = net(src_inputs, temp=1)
    l_tgt_outputs = net(l_tgt_inputs, temp=1)
    ul_tgt_outputs = net(ul_tgt_inputs, temp=1)
    src_features, src_logits = src_outputs['adapted_layer'], src_outputs['output_logits']
    l_tgt_logits = l_tgt_outputs['output_logits']
    ul_tgt_features, ul_tgt_logits = ul_tgt_outputs['adapted_layer'], ul_tgt_outputs['output_logits']
    
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
    
    features = torch.cat((src_features, ul_tgt_features), dim=0)
    loss_transfer = args.trade_off * DANN(features, ad_net, entire_steps)
    total_loss = loss_transfer + loss_cls
    main_optimizer.zero_grad()
    dis_optimizer.zero_grad()
    total_loss.backward()
    main_optimizer.step()
    dis_optimizer.step()
    # update meters
    meters['src_cls_loss'].update(loss_cls_src.item())
    meters['tgt_cls_loss'].update(loss_cls_tgt.item())
    meters['transfer_loss'].update(loss_transfer.item())