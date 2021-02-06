import torch
from utils.losses import *
import torch.nn.functional as F
from utils.utils import softmax_xent_two_logits

def run_iter_mimcontra(
    inputs,
    models, optimizers,
    meters,
    args,
    global_step,
    entire_steps
):
    assert args.K % args.batch_size == 0
    net = models['net']
#     queue = models['queue']
    classifier_optimizer = optimizers['classifier']
    encoder_optimizer = optimizers['encoder']
    main_optimizer = optimizers['main']
    src_inputs = inputs['src']['sample_1_q'][0].cuda()
    src_labels = inputs['src']['sample_1_q'][1].cuda()
    src_inputs_k = inputs['src']['sample_1_k'][0].cuda()
    src_labels_k = inputs['src']['sample_1_k'][1].cuda()
    src_inputs1 = inputs['src']['sample_2_q'][0].cuda()
    src_labels1 = inputs['src']['sample_2_q'][1].cuda()
    l_tgt_inputs = inputs['l_tgt']['sample_1_q'][0].cuda()
    l_tgt_labels = inputs['l_tgt']['sample_1_q'][1].cuda()
    ul_tgt_inputs = inputs['ul_tgt']['sample_1_q'][0].cuda()
    ul_tgt_inputs1 = inputs['ul_tgt']['sample_2_q'][0].cuda()
    ul_tgt_inputs_k = inputs['ul_tgt']['sample_1_k'][0].cuda()
    # run network
#     with torch.no_grad():
    src_outputs = net(src_inputs, temp=1)
    ul_tgt_outputs = net(ul_tgt_inputs, temp=1)
    l_tgt_outputs = net(l_tgt_inputs, temp=1)
    src_logits, src_features = src_outputs['output_logits'], src_outputs['adapted_layer']
    ul_tgt_logits = ul_tgt_outputs['output_logits']
    l_tgt_logits = l_tgt_outputs['output_logits']
    
    # mim + supervised
    total_loss, loss_cent, loss_infoent = 0, 0, 0
    # classifer loss
    loss_cls_src = supervised_loss(
        src_logits, src_labels, 
        args.num_cls, args.batch_size, 
        global_step, entire_steps, 
        args.annealing,
        args.task_type
    )
    loss_cls_tgt = supervised_loss(
        l_tgt_logits, l_tgt_labels, 
        args.num_cls, args.batch_size, 
        global_step, entire_steps, 
        args.annealing,
        args.task_type
    )
    loss_cls = (loss_cls_tgt + loss_cls_src) / 2.
    total_loss += loss_cls 
    meters['src_cls_loss'].update(loss_cls_src.item())
    meters['tgt_cls_loss'].update(loss_cls_tgt.item())
    
    # irm penalty loss
    if args.lambda_irm > 0:
        loss_irm = 0
        loss_irm += irm_penalty(
            src_logits, src_labels, 
            supervised_loss, 
            [args.num_cls, args.batch_size, global_step, entire_steps, args.annealing]
        ) * args.lambda_irm / 2.
        loss_irm += irm_penalty(
            l_tgt_logits, l_tgt_labels, 
            supervised_loss, 
            [args.num_cls, args.batch_size, global_step, entire_steps, args.annealing]
        ) * args.lambda_irm / 2.
        total_loss += loss_irm
        meters['irm_loss'].update(loss_irm.item())
    
    # mim loss
    tgt_preds = ul_tgt_logits.softmax(dim=1)
    p = tgt_preds.sum(dim=0) / (tgt_preds.sum() + 1e-12)
    log_p = torch.log(p + 1e-12)
    loss_cent += (softmax_xent_two_logits(ul_tgt_logits, ul_tgt_logits) * 0.1 * 0.5)
    loss_infoent += (torch.sum(p * log_p) * 0.1)
    total_loss += loss_cent + loss_infoent
    meters['mim_loss'].update(loss_cent.item() + loss_infoent.item())

    # consistant
    loss_consistant = 0
    ul_tgt_outputs_k = net(ul_tgt_inputs_k, temp=1)
    loss_consistant += kl_divergence_with_logits(
        ul_tgt_logits.detach(), 
        ul_tgt_outputs_k['output_logits'],
        confidence=0.95
    ) * args.trade_off
    total_loss += loss_consistant
    meters['consistant_loss'].update(loss_consistant.item())
    
    main_optimizer.zero_grad()
    total_loss.backward()
    main_optimizer.step()