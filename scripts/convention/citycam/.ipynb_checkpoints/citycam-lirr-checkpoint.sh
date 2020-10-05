#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=(${1} ${2})
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/citycam/lirr/'

for((j = 0; j < 1; j++))
do
    source_domain=${datasets[0]}
    target_domain=${datasets[1]}
    echo "source: ${source_domain}; target: ${target_domain}."
    python3 main.py \
    --dataset citycam \
    --domain_shift_type convention \
    --source ${source_domain} \
    --target ${target_domain} \
    --nepoch 200 \
    --model_name CountingNet \
    --image_size 256 \
    --channels 3 \
    --num_cls 1 \
    --lr 1e-6 \
    --task_type reg \
    --milestone 200 \
    --optimizer_type adam \
    --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/CityCam \
    --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_citycam_lirr \
    --logf ${logf_root}${source_domain}_${target_domain}_citycam_lirr.txt \
    --batch_size ${3} \
    --nthreads 8 \
    --method lirrv1 \
    --lambda_adv 0.01 \
    --lambda_lirr 0.1 \
    --lambda_inv 0.5 \
    --lambda_env 0.5 \
    --distance_type sqr \
    --adapted_dim 512 \
    --target_labeled_portion $(expr $j \* 5) \
    --logger_file_name citycam_lirr
done