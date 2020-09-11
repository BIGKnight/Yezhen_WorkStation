import torch
from utils.losses import *
import torch.nn.functional as F
from utils.utils import softmax_xent_two_logits

def run_iter_instapbm(
    inputs,
    models, optimizer,
    meters,
    args,
    global_step,
    entire_steps
):
    assert args.K % args.batch_size == 0
    net = models['net']
    queue = models['queue']
    # obtain inputs
    src_inputs, src_labels = inputs['src_inputs'], inputs['src_labels']
    src_inputs1, src_labels1 = inputs['src_inputs1'], inputs['src_labels1']
    src_inputs_k, src_labels_k= inputs['src_inputs_k'], inputs['src_labels_k']
    ul_tgt_inputs, _ = inputs['ul_tgt_inputs'], inputs['ul_tgt_labels']
    ul_tgt_inputs1, _ = inputs['ul_tgt_inputs1'], inputs['ul_tgt_labels1']
    ul_tgt_inputs_k, _= inputs['ul_tgt_inputs_k'], inputs['ul_tgt_labels_k']
    
    # run network
    src_outputs = net(src_inputs, temp=1)
    ul_tgt_outputs = net(ul_tgt_inputs, temp=1)
    _, src_features =  src_outputs['output_logits'], src_outputs['adapted_layer']
    _, ul_tgt_features =  ul_tgt_outputs['output_logits'], ul_tgt_outputs['adapted_layer']
    src_logits = net.fc(src_features.detach())
    ul_tgt_logits = net.fc(ul_tgt_features.detach())
    # classification loss
    total_loss = 0
    loss_cls = supervised_loss(
        src_logits, src_labels, 
        args.num_cls, args.batch_size, 
        global_step, entire_steps, 
        args.annealing
    )
    total_loss += loss_cls
    meters['cls_loss'].update(loss_cls.item())
    # mim loss
    if args.mim:
        loss_cent = 0
        loss_infoent = 0
        if args.mim_tgt:
            ul_tgt_preds = ul_tgt_logits.softmax(dim=1)
            p = ul_tgt_preds.sum(dim=0) / (ul_tgt_preds.sum() + 1e-12)
            log_p = torch.log(p + 1e-12)
            loss_cent += softmax_xent_two_logits(ul_tgt_logits, ul_tgt_logits) * args.trade_off
            loss_infoent += torch.sum(p * log_p) * args.trade_off
            
        if args.mim_src:
            src_preds = src_logits.softmax(dim=1)
            p = src_preds.sum(dim=0) / (src_preds.sum() + 1e-12)
            log_p = torch.log(p + 1e-12)
            loss_cent += softmax_xent_two_logits(src_logits, src_logits) * args.trade_off
            loss_infoent += torch.sum(p * log_p) * args.trade_off
    
        total_loss += loss_cent
        total_loss += loss_infoent        
        meters['cent_loss'].update(loss_cent.item())
        meters['infoent_loss'].update(loss_infoent.item())
        meters['mim_loss'].update(loss_cent.item() + loss_infoent.item())
        
#         # enqueue and dequeue
#         ul_tgt_preds = ul_tgt_logits.softmax(dim=1)
#         scores = queue.sum(dim=0).detach() + ul_tgt_preds.sum(dim=0)
        
#         q = scores / (scores.sum() + 1e-12)
#         log_q = torch.log(q + 1e-12).detach()
        
#         p = ul_tgt_preds.sum(dim=0) / (ul_tgt_preds.sum() + 1e-12)
# #         log_p = torch.log(p + 1e-12)
        
#         loss_cent = softmax_xent_two_logits(ul_tgt_logits, ul_tgt_logits) * args.trade_off
#         loss_infoent = torch.sum(p * log_q) * args.trade_off
#         total_loss += loss_cent
#         total_loss += loss_infoent
        
#         queue[models['ptr']:models['ptr']+args.batch_size, :] = (ul_tgt_preds)
#         models['ptr'] = (models['ptr'] + args.batch_size) % args.K
        
#         meters['cent_loss'].update(loss_cent.item())
#         meters['infoent_loss'].update(loss_infoent.item())
#         meters['mim_loss'].update(loss_cent.item() + loss_infoent.item())
    
    # contrastive loss
    if args.contrastive:
        loss_contrastive = 0
        src_outputs_k = net(src_inputs_k, temp=1 )
        loss_contrastive += kl_divergence_with_logits(
            src_logits.detach(), 
            src_outputs_k['output_logits'],
            confidence=0.95
        )
            
        ul_tgt_outputs_k = net(ul_tgt_inputs_k, temp=1 )
        loss_contrastive += kl_divergence_with_logits(
            ul_tgt_logits.detach(), 
            ul_tgt_outputs_k['output_logits'],
            confidence=0.95
        ) 

        total_loss += loss_contrastive * args.trade_off
        meters['contrastive_loss'].update(loss_contrastive.item() * args.trade_off)
        
    if args.lirr:
        src_embedding = torch.zeros(args.batch_size, args.adapted_dim).cuda()
        ul_tgt_embedding = torch.ones(args.batch_size, args.adapted_dim).cuda()
        e_predictor = models['e_predictor']
        src_ez_features = F.normalize(src_features + src_embedding, p=2, dim=1)
        ul_tgt_ez_features = F.normalize(ul_tgt_features + ul_tgt_embedding, p=2, dim=1)
        
        # normalize the prototypes
        with torch.no_grad():
            w = e_predictor.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            e_predictor.weight.copy_(w)
            
        py_ez_src = F.softmax(e_predictor(src_ez_features), dim=1)
        py_ez_tgt = F.softmax(e_predictor(ul_tgt_ez_features), dim=1)
        py_z_src  = F.softmax(src_logits)
        py_z_tgt  = F.softmax(ul_tgt_logits)
        
    # mixup loss
    if args.mixup:
        alpha = torch.rand(args.batch_size, 1, 1, 1).cuda()
        src_mixed_inputs = alpha * src_inputs + (1 - alpha) * src_inputs1
        ul_tgt_mixed_inputs = alpha * ul_tgt_inputs + (1 - alpha) * ul_tgt_inputs1
        src_onehot_labels = torch.zeros(args.batch_size, args.num_cls, device=src_labels.device).scatter(
            dim=1, index=src_labels.long().reshape(args.batch_size, 1), value=1)
        src_onehot_labels1 = torch.zeros(args.batch_size, args.num_cls, device=src_labels1.device).scatter(
            dim=1, index=src_labels1.long().reshape(args.batch_size, 1), value=1)
        src_mixed_labels = alpha.reshape(args.batch_size, 1) * src_onehot_labels + (1 - alpha).reshape(args.batch_size, 1) * src_onehot_labels1
        loss_mixup = 0
        # ul_tgt_mixup_loss
        
        with torch.no_grad():
            ul_tgt_outputs1 = net(ul_tgt_inputs1, temp=1 )
        ul_tgt_mixed_outputs = net(ul_tgt_mixed_inputs, temp=1 )
        ul_tgt_logits1, ul_tgt_mixed_logits = ul_tgt_outputs1['output_logits'], ul_tgt_mixed_outputs['output_logits']
        ul_tgt_preds, ul_tgt_preds1 = ul_tgt_logits.softmax(dim=1), ul_tgt_logits1.softmax(dim=1)
        ul_tgt_mixed_q_score = (alpha.reshape(args.batch_size, 1) * ul_tgt_preds + 
                             (1 - alpha.reshape(args.batch_size, 1)) * ul_tgt_preds1
                            ).clamp(0, 1).detach()
        ul_tgt_mixed_p_score = ul_tgt_mixed_logits.softmax(dim=1)
        ul_tgt_mixup_loss = sim_KL_divergence(ul_tgt_mixed_q_score, ul_tgt_mixed_p_score, coef=args.trade_off, confidence=0.95) * 0.1

        # src_mixup_loss
        src_mixed_outputs = net(src_mixed_inputs, temp=1 )
        src_mixed_logits = src_mixed_outputs['output_logits']
        src_mixed_q_score = src_mixed_labels.clamp(0, 1).detach()
        src_mixed_p_score = src_mixed_logits.softmax(dim=1)
        src_mixup_loss = sim_KL_divergence(src_mixed_q_score, src_mixed_p_score, coef=args.trade_off) * 0.1
        
        total_loss += ul_tgt_mixup_loss
        total_loss += src_mixup_loss
        meters['ul_tgt_mixup_loss'].update(ul_tgt_mixup_loss.item())
        meters['src_mixup_loss'].update(src_mixup_loss.item())
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    
    