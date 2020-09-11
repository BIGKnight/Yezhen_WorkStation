import torch
from utils.losses import *
from models.utils import *

def run_iter_adr(
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
    
    src_outputs = net(src_inputs, temp=1)
    l_tgt_outputs = net(l_tgt_inputs, temp=1)
    ul_tgt_outputs = net(ul_tgt_inputs, temp=1)
    
    src_logits = src_outputs['output_logits']
    l_tgt_logits = l_tgt_outputs['output_logits']
    ul_tgt_features = ul_tgt_outputs['features']
    
    ul_tgt_logits_1 = net.fc(
        F.dropout(
            torch.nn.functional.relu(
                net.bottleneck(
                    F.dropout(
                        grad_reverse(ul_tgt_features, lambd=1.0),
                        training=True, 
                        p=0.5)
                ),
                inplace=False
            ),
            training=True, 
            p=0.5
        )
    )
    ul_tgt_logits_2 = net.fc(
        F.dropout(
            torch.nn.functional.relu(
                net.bottleneck(
                    F.dropout(
                        grad_reverse(ul_tgt_features, lambd=1.0),
                        training=True, 
                        p=0.5)
                ),
                inplace=False
            ),
            training=True, 
            p=0.5
        )
    )
    div = (kl_divergence_with_logits(ul_tgt_logits_1.detach(), ul_tgt_logits_2) + \
           kl_divergence_with_logits(ul_tgt_logits_2.detach(), ul_tgt_logits_1)) / 2
    
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
    
    total_loss = loss_cls - args.trade_off * div
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    # update meters
    meters['src_cls_loss'].update(loss_cls_src.item())
    meters['tgt_cls_loss'].update(loss_cls_tgt.item())
    meters['transfer_loss'].update(args.trade_off * div.item())