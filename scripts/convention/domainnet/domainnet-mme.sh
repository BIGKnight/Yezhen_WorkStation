#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=(${3} ${4})
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/domainnet/mme/'

for((i = 0; i < 1; i++))
do
    for((j = 1; j < 2; j++))
    do
        source_domain=${datasets[${i}]}
        target_domain=${datasets[${j}]}
        echo "source: ${source_domain}; target: ${target_domain}."
        python3 main.py \
        --dataset domainnet \
        --domain_shift_type convention \
        --source ${source_domain} \
        --target ${target_domain} \
        --nepoch 60 \
        --model_name ${1} \
        --image_size 224 \
        --channels 3 \
        --num_cls 345 \
        --lr 1e-3 \
        --milestone 40 \
        --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/visda2019/domainnet \
        --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_domainnet_mme_ \
        --logf ${logf_root}${source_domain}_${target_domain}_domainnet_mme_.txt \
        --batch_size ${2} \
        --nthreads 8 \
        --method mme \
        --trade_off 0.1 \
        --temp 0.05 \
        --adj_lr_func none \
        --logger_file_name domainnet_mme_
    done
done