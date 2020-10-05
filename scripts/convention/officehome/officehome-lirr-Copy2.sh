#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=(${1} ${2})
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/officehome/lirr/'

for((j = 0; j < 1; j++))
do
    source_domain=${datasets[0]}
    target_domain=${datasets[1]}
    echo "source: ${source_domain}; target: ${target_domain}."
    python3 main.py \
    --dataset officehome \
    --domain_shift_type convention \
    --source ${source_domain} \
    --target ${target_domain} \
    --nepoch 80 \
    --model_name resnet34 \
    --image_size 224 \
    --channels 3 \
    --num_cls 65 \
    --lr 0.001 \
    --milestone 50 \
    --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/DDA_office/officehome \
    --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_officehome_lirr-v1 \
    --logf ${logf_root}${source_domain}_${target_domain}_officehome_lirr-v1.txt \
    --batch_size ${3} \
    --nthreads 8 \
    --method lirr-v1 \
    --lambda_adv 0.01 \
    --lambda_lirr 0.1 \
    --lambda_inv ${4} \
    --lambda_env ${5} \
    --distance_type sqr \
    --target_labeled_portion $(expr $j \* 5) \
    --logger_file_name officehome_lirr-v1
done