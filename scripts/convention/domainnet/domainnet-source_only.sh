#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=(${1} ${2})
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/domainnet/sourceonly/'

for((j = 1; j < 3; j++))
do
    source_domain=${datasets[${i}]}
    target_domain=${datasets[${j}]}
    echo "source: ${source_domain}; target: ${target_domain}."
    python3 main.py \
    --dataset domainnet \
    --domain_shift_type convention \
    --source ${source_domain} \
    --target ${target_domain} \
    --nepoch 40 \
    --model_name resnet34 \
    --image_size 224 \
    --channels 3 \
    --num_cls 345 \
    --lr 1e-3 \
    --milestone 30 \
    --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/visda2019/domainnet \
    --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_domainnet_source_only \
    --logf ${logf_root}${source_domain}_${target_domain}_domainnet_source_only.txt \
    --batch_size ${3} \
    --nthreads 8 \
    --method source_only \
    --temp 1 \
    --adj_lr_func none \
    --target_labeled_portion $(expr $j \* 5) \
    --logger_file_name domainnet_source_only
done