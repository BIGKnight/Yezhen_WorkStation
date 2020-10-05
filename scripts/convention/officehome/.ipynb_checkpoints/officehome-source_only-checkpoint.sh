#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=(${1} ${2})
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/officehome/sourceonly/'

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
    --nepoch 50 \
    --model_name resnet34 \
    --image_size 224 \
    --channels 3 \
    --num_cls 65 \
    --lr 0.001 \
    --milestone 35 \
    --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/DDA_office/officehome \
    --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_officehome_source_only \
    --logf ${logf_root}${source_domain}_${target_domain}_officehome_source_only.txt \
    --batch_size ${3} \
    --nthreads 8 \
    --method source_only \
    --target_labeled_portion 70 \
    --logger_file_name officehome_source_only
done
#     --target_labeled_portion $(expr $j \* 5) \