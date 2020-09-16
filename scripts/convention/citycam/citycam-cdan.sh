#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=(${1} ${2})
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/citycam/cdan/'

for((i = 0; i < 1; i++))
do
    for((j = 1; j < 2; j++))
    do
        source_domain=${datasets[${i}]}
        target_domain=${datasets[${j}]}
        echo "source: ${source_domain}; target: ${target_domain}."
        python3 main.py \
        --dataset citycam \
        --domain_shift_type convention \
        --source ${source_domain} \
        --target ${target_domain} \
        --nepoch 100 \
        --model_name CountingNet \
        --image_size 256 \
        --channels 3 \
        --num_cls 1 \
        --lr 1e-6 \
        --task_type reg \
        --milestone 100 \
        --optimizer_type adam \
        --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/CityCam \
        --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_citycam_cdan \
        --logf ${logf_root}${source_domain}_${target_domain}_citycam_cdan.txt \
        --batch_size ${3} \
        --nthreads 8 \
        --method cdan \
        --temp 1 \
        --adj_lr_func none \
        --trade_off 1 \
        --adapted_dim 512 \
        --logger_file_name citycam_cdan
    done
done