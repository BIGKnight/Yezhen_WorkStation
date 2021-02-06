import torch
from utils.losses import *
import matplotlib.pyplot as plt
from models.utils import init_weights
import random
import numpy as np
# mi_data_iter
def run_iter_counting_mim(
    inputs,
    models, optimizers,
    meters,
    args,
    global_step,
    entire_steps
):
    net = models['net']
    mi_estimator = models['mi_estimator']
    mi_data_iter = models['mi_data_iter']
    
    optimizer = optimizers['main']
    mi_optimizer = optimizers['mi']
    src_inputs = inputs['src']['sample_1_q'][0].cuda()
    src_labels = inputs['src']['sample_1_q'][1].cuda()
    is_mask = len(inputs['l_tgt']['sample_1_q'])> 2
        
    l_tgt_inputs = inputs['l_tgt']['sample_1_q'][0].cuda()
    l_tgt_labels = inputs['l_tgt']['sample_1_q'][1].cuda()
    ul_tgt_inputs = inputs['ul_tgt']['sample_1_q'][0].cuda()
    ul_tgt_labels = inputs['ul_tgt']['sample_1_q'][1].cuda()
    
    src_masks = inputs['src']['sample_1_q'][2].cuda() if is_mask else 1
    l_tgt_masks = inputs['l_tgt']['sample_1_q'][2].cuda() if is_mask else 1
    ul_tgt_masks = inputs['ul_tgt']['sample_1_q'][2].cuda() if is_mask else 1
    
    src_inputs = src_inputs * src_masks
    l_tgt_inputs = l_tgt_inputs * l_tgt_masks
    ul_tgt_inputs = ul_tgt_inputs * ul_tgt_masks
    
#     src_outputs = net(src_inputs, temp=1)
#     l_tgt_outputs = net(l_tgt_inputs, temp=1)
#     ul_tgt_outputs = net(ul_tgt_inputs, temp=1)
    
#     src_logits = src_outputs['output_logits'] * src_masks
#     l_tgt_logits = l_tgt_outputs['output_logits'] * l_tgt_masks
#     ul_tgt_logits = ul_tgt_outputs['output_logits'] * ul_tgt_masks
    
#     # classification loss
#     loss_cls_src = supervised_loss(
#         src_logits, src_labels, 
#         args.num_cls, args.batch_size, 
#         global_step, entire_steps, 
#         args.annealing,
#         args.task_type
#     ) / 2.
#     loss_cls_tgt = supervised_loss(
#         l_tgt_logits, l_tgt_labels, 
#         args.num_cls, args.batch_size, 
#         global_step, entire_steps, 
#         args.annealing,
#         args.task_type
#     ) / 2.
    
    # target mutual information maximization
    mi_estimator.apply(init_weights)
    for i in range(args.K_iter):
        mi_tgt_inputs = next(mi_data_iter)
        mi_tgt_mask = mi_tgt_inputs['sample_1_q'][2].cuda() if is_mask else 1
        mi_tgt_im = mi_tgt_inputs['sample_1_q'][0].cuda() * mi_tgt_mask
        mi_tgt_labels = mi_tgt_inputs['sample_1_q'][1].cuda() * mi_tgt_mask
#         with torch.no_grad():
#             mi_tgt_outputs = net(mi_tgt_im, temp=1)
#         mi_tgt_x = mi_tgt_outputs['adapted_layer'].detach()
#         mi_tgt_y = torch.mean(mi_tgt_outputs['output_logits'] * mi_tgt_mask, dim=[2, 3]).detach()
        mi_tgt_x = mi_tgt_im.detach()
        mi_tgt_y = torch.mean(mi_tgt_labels, dim=[2, 3]).detach()
        
        permutation = np.random.permutation(args.batch_size)
        shuffled_mi_tgt_y = mi_tgt_y[permutation].detach()
        Exy = mi_estimator(mi_tgt_x, mi_tgt_y)
        ExEy = mi_estimator(mi_tgt_x, shuffled_mi_tgt_y)
        
        mi_loss = - (torch.mean(Exy) - torch.log(torch.mean(torch.exp(ExEy)) + 1e-8))  # maximize
#         mi_loss = - (torch.mean(Exy) - torch.mean(torch.exp(ExEy - 1)))  # maximize
        mi_optimizer.zero_grad()
        mi_loss.backward()
        mi_optimizer.step()
#         print((-mi_loss).item())
#         print(torch.mean(Exy).item(), torch.mean(torch.exp(ExEy)).item(), mi_loss.item())
    
#     ul_tgt_x = ul_tgt_outputs['adapted_layer']
#     ul_tgt_y = torch.mean(ul_tgt_logits, dim=[2, 3])
    ul_tgt_x = ul_tgt_inputs
    ul_tgt_y = torch.mean(ul_tgt_labels, dim=[2, 3])
    
    permutation = np.random.permutation(args.batch_size)
    shuffled_ul_tgt_y = ul_tgt_y[permutation].detach()
    
    pred_Exy = mi_estimator(ul_tgt_x, ul_tgt_y)
    pred_ExEy = mi_estimator(ul_tgt_x, shuffled_ul_tgt_y)
    
    estimated_mi_penalty = ((torch.mean(pred_Exy) - torch.log(torch.mean(torch.exp(pred_ExEy)) + 1e-8)) * args.trade_off)
#     loss_cls = loss_cls_src + loss_cls_tgt - estimated_mi_penalty
#     optimizer.zero_grad()
#     loss_cls.backward()
#     optimizer.step()
    # update meters
    print((estimated_mi_penalty).item())
#     meters['src_cls_loss'].update(loss_cls_src.item())
#     meters['tgt_cls_loss'].update(loss_cls_tgt.item())
#     meters['estimated_mi'].update(estimated_mi_penalty.item())
    