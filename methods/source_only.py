import torch
from utils.losses import *
import matplotlib.pyplot as plt
import random

def run_iter_source_only(
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
    is_mask = len(inputs['l_tgt']['sample_1_q'])> 2
        
    l_tgt_inputs = inputs['l_tgt']['sample_1_q'][0].cuda()
    l_tgt_labels = inputs['l_tgt']['sample_1_q'][1].cuda()
    
    src_masks = inputs['src']['sample_1_q'][2].cuda() if is_mask else 1
    l_tgt_masks = inputs['l_tgt']['sample_1_q'][2].cuda() if is_mask else 1
    src_inputs = src_inputs * src_masks
    l_tgt_inputs = l_tgt_inputs * l_tgt_masks
    src_outputs = net(src_inputs, temp=1)
    l_tgt_outputs = net(l_tgt_inputs, temp=1)
    src_logits = src_outputs['output_logits']
    l_tgt_logits = l_tgt_outputs['output_logits']
    src_logits = src_logits * src_masks
    l_tgt_logits = l_tgt_logits * l_tgt_masks
    r = random.random()
#     if r < 0.005 and is_mask:
#         src_features = src_outputs['features'].mean(dim=1)
#         _, (inputs, gt, ft, pred) = plt.subplots(1, 4, figsize=(10, 10))
#         inputs.imshow(src_inputs.permute(0, 2, 3, 1).cpu().numpy()[0])
#         gt.imshow(src_labels.squeeze(1).cpu().numpy()[0], cmap=plt.cm.jet)
#         gt.set_title(str(src_labels[0].sum().cpu().numpy()))
#         ft.imshow(src_features.squeeze(1).detach().cpu().numpy()[0], cmap=plt.cm.jet)
#         pred.imshow(src_logits.squeeze(1).detach().cpu().numpy()[0], cmap=plt.cm.jet)
#         pred.set_title(str(src_logits[0].sum().detach().cpu().numpy()))
#         plt.show()
    # classification loss
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
    ) / 2.
#     loss_cls = loss_cls_src
    loss_cls = loss_cls_src + loss_cls_tgt
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    # update meters
    meters['src_cls_loss'].update(loss_cls_src.item())
    meters['tgt_cls_loss'].update(loss_cls_tgt.item())