import torch
from utils.losses import *
from utils.utils import softmax_xent_two_logits

def momentum_update_key_encoder(net_q, net_k, head_q, head_k, m=0.998):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(
        list(net_q.parameters()) + list(head_q.parameters()),
        list(net_k.parameters()) + list(head_k.parameters())
    ):
        param_k.data = param_k.data * m + param_q.data * (1. - m)
        
def dequeue_and_enqueue(queue_buffer, queue_ptr, representations, K):
    z = torch.cat(representations, dim=0)
    size = z.shape[0]
    assert K % size == 0
    queue_buffer[:, queue_ptr:queue_ptr + size] = z.t()
    queue_ptr = (queue_ptr + size) % K
    return queue_ptr

def run_iter_moco(
    inputs,
    models, optimizer,
    meters,
    args,
    global_step,
    entire_steps
):
    encoder_q = models['net']
    encoder_k = models['ema_net']
    head_q = models['head_q']
    head_k = models['head_k']
    queue_buffer = models['queue']
    
    src_inputs_q, src_labels_q = inputs['src_inputs'], inputs['src_labels']
    src_inputs_k, src_labels_k = inputs['src_inputs_k'], inputs['src_labels_k']
    tgt_inputs_q = inputs['tgt_inputs']
    tgt_inputs_k = inputs['tgt_inputs_k']
    if args.moco_finetune == True:
        total_loss = 0
#         with torch.no_grad():
        src_outputs_q = encoder_q(src_inputs_q, temp=1, dropout=args.dropout)
        src_features_q = src_outputs_q['adapted_layer']
#             src_features_q = F.normalize(src_features_q, p=2, dim=1)
        if args.dropout:
            src_features_q = F.dropout(src_features_q, training=True, p=0.5)
        if args.cosine_classifer:
#                 src_features_q = F.normalize(src_features_q, p=2, dim=1)
            w = encoder_q.fc.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            encoder_q.fc.weight.copy_(w)
                
        src_logits = encoder_q.fc(src_features_q)
        loss_cls = supervised_loss(
            src_logits, src_labels_q, 
            args.num_cls, args.batch_size, 
            global_step, entire_steps, 
            args.annealing
        )
        total_loss += loss_cls
        meters['cls_loss'].update(loss_cls.item())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
            
    else:
#         src_outputs_q = encoder_q(src_inputs_q, temp=1, dropout=args.dropout)
#         with torch.no_grad():
#             src_outputs_k = encoder_k(src_inputs_k, temp=1, dropout=args.dropout)
#             src_ft_proj_k = head_k(src_outputs_k['adapted_layer'])
#         src_features_q = src_outputs_q['adapted_layer']
#         src_ft_proj_q = head_q(src_features_q)
#         src_z_q = nn.functional.normalize(src_ft_proj_q, dim=1)
#         src_z_k = nn.functional.normalize(src_ft_proj_k, dim=1)
        
        tgt_outputs_q = encoder_q(tgt_inputs_q, temp=1, dropout=args.dropout)
        with torch.no_grad():
            tgt_outputs_k = encoder_k(tgt_inputs_k, temp=1, dropout=args.dropout)
            tgt_ft_proj_k = head_q(tgt_outputs_k['adapted_layer'])
        tgt_features_q = tgt_outputs_q['adapted_layer']
        tgt_ft_proj_q = head_q(tgt_features_q)
        tgt_z_q = nn.functional.normalize(tgt_ft_proj_q, dim=1)
        tgt_z_k = nn.functional.normalize(tgt_ft_proj_k, dim=1)

        
             
        momentum_update_key_encoder(encoder_q, encoder_k, head_q, head_k, args.m)

#         src_l_pos = torch.einsum('nc,nc->n', [src_z_q, src_z_k]).unsqueeze(-1)
#         src_l_neg = torch.einsum('nc,ck->nk', [src_z_q, queue_buffer.clone().detach()])
#         src_logits = torch.cat([src_l_pos, src_l_neg], dim=1) / args.temp
#         src_labels = torch.zeros(src_logits.shape[0], dtype=torch.long).cuda()
        
        tgt_l_pos = torch.einsum('nc,nc->n', [tgt_z_q, tgt_z_k]).unsqueeze(-1)
        tgt_l_neg = torch.einsum('nc,ck->nk', [tgt_z_q, queue_buffer.clone().detach()])
        tgt_logits = torch.cat([tgt_l_pos, tgt_l_neg], dim=1) / args.temp
        tgt_labels = torch.zeros(tgt_logits.shape[0], dtype=torch.long).cuda()

#         models['queue_ptr'] = dequeue_and_enqueue(queue_buffer, models['queue_ptr'], [src_z_k, tgt_z_k], args.K)
        models['queue_ptr'] = dequeue_and_enqueue(queue_buffer, models['queue_ptr'], [tgt_z_k], args.K)

#         src_infonce_loss = supervised_loss(
#             src_logits, src_labels, 
#             src_logits.shape[1], src_logits.shape[0], 
#             global_step, entire_steps
#         )
        tgt_infonce_loss = supervised_loss(
            tgt_logits, tgt_labels, 
            tgt_logits.shape[1], tgt_logits.shape[0], 
            global_step, entire_steps
        )
#         infonce_loss = ((src_infonce_loss + tgt_infonce_loss) / 2.) * args.trade_off
        infonce_loss = tgt_infonce_loss * args.trade_off

        meters['infonce_loss'].update(infonce_loss.item())
        optimizer.zero_grad()
        infonce_loss.backward()
        optimizer.step()