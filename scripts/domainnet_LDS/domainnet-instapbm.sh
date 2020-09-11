#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env

cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/LDS/domainnet/instapbm/'

source_domain=${1}
target_domain=${2}

python3 main.py \
--dataset domainnet \
--domain_shift_type LDS \
--source ${source_domain} \
--target ${target_domain} \
--nepoch 50 \
--model_name resnet101 \
--image_size 224 \
--channels 3 \
--num_cls 345 \
--lr 0.0001 \
--milestone 45 \
--data_root /nfs/volume-92-5/wangyezhen_i/Datasets/visda2019/domainnet \
--outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_domainnet_instapbm \
--logf ${logf_root}${source_domain}_${target_domain}_instapbm.txt \
--batch_size 32 \
--nthreads 8 \
--method instapbm \
--trade_off 1 \
--K 512 \
--mim \
--contrastive \
--mixup \
--flip \
--quadrant \
--rotation \
--lw_rotation 1 \
--lw_quadrant 1 \
--lw_flip 1 \
--lr_rotation 0.0001 \
--lr_quadrant 0.0001 \
--lr_flip 0.0001 \
--logger_file_name domainnet_instapbm_LDS