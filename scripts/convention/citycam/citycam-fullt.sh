#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=('398_fullt' 398)
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/citycam/sourceonly/'

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
    --nepoch 80 \
    --model_name CountingNet \
    --image_size 256 \
    --channels 3 \
    --num_cls 1 \
    --lr 1e-6 \
    --task_type reg \
    --milestone 80 \
    --optimizer_type adam \
    --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/CityCam \
    --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_citycam_fullt \
    --logf ${logf_root}${source_domain}_${target_domain}_citycam_fullt.txt \
    --batch_size ${1} \
    --nthreads 8 \
    --method fullt \
    --temp 1 \
    --adj_lr_func none \
    --target_labeled_portion $(expr $j \* 5) \
    --logger_file_name citycam_fullt
done