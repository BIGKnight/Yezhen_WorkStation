import torch
from utils.losses import *

def run_iter_cdan(
    inputs,
    models, optimizers,
    meters,
    args,
    global_step,
    entire_steps
):
    net = models['net']
    ad_net = models['ad_net']
    random_layer = models['random_layer']
    optimizer = optimizers['main']
    
    src_inputs = inputs['src']['sample_1_q'][0].cuda()
    src_labels = inputs['src']['sample_1_q'][1].cuda()
    l_tgt_inputs = inputs['l_tgt']['sample_1_q'][0].cuda()
    l_tgt_labels = inputs['l_tgt']['sample_1_q'][1].cuda()
    ul_tgt_inputs = inputs['ul_tgt']['sample_1_q'][0].cuda()    
    is_mask = len(inputs['l_tgt']['sample_1_q'])> 2
    src_masks = inputs['src']['sample_1_q'][2].cuda() if is_mask else 1
    l_tgt_masks = inputs['l_tgt']['sample_1_q'][2].cuda() if is_mask else 1
    ul_tgt_masks = inputs['ul_tgt']['sample_1_q'][2].cuda() if is_mask else 1
    src_inputs = src_inputs * src_masks
    l_tgt_inputs = l_tgt_inputs * l_tgt_masks
    ul_tgt_inputs = ul_tgt_inputs * ul_tgt_masks
    
    src_outputs = net(src_inputs, temp=1)
    l_tgt_outputs = net(l_tgt_inputs, temp=1)
    ul_tgt_outputs = net(ul_tgt_inputs, temp=1)
    
    src_features, src_logits = src_outputs['adapted_layer'], src_outputs['output_logits']
    l_tgt_logits = l_tgt_outputs['output_logits']
    ul_tgt_features, ul_tgt_logits = ul_tgt_outputs['adapted_layer'], ul_tgt_outputs['output_logits']
    src_logits = src_logits * src_masks
    l_tgt_logits = l_tgt_logits * l_tgt_masks
    ul_tgt_logits = ul_tgt_logits * ul_tgt_masks
    
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
    logits = torch.cat((src_logits, ul_tgt_logits), dim=0)
    softmax_out = torch.nn.functional.softmax(logits, dim=1)
    loss_transfer = CDAN(
        [features, softmax_out],
        ad_net,
        entire_steps,
        None, None,
        random_layer
    ) * args.trade_off
            
    total_loss = loss_transfer + loss_cls
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    # update meters
    meters['src_cls_loss'].update(loss_cls_src.item())
    meters['tgt_cls_loss'].update(loss_cls_tgt.item())
    meters['transfer_loss'].update(loss_transfer.item())