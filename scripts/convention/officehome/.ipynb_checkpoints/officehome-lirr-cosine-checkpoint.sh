#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=('Art' 'RealWorld' 'Clipart' 'Product')
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/officehome/lirr/'

for((i = 0; i < 1; i++))
do
    for((j = 1; j < 2; j++))
    do
        source_domain=${datasets[${i}]}
        target_domain=${datasets[${j}]}
        echo "source: ${source_domain}; target: ${target_domain}."
        python3 main.py \
        --dataset officehome \
        --domain_shift_type convention \
        --source ${source_domain} \
        --target ${target_domain} \
        --nepoch 50 \
        --model_name ${1} \
        --image_size 224 \
        --channels 3 \
        --num_cls 65 \
        --lr 0.0001 \
        --milestone 50 \
        --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/DDA_office/officehome \
        --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_officehome_lirr_cosine \
        --logf ${logf_root}${source_domain}_${target_domain}_officehome_lirr_cosine.txt \
        --batch_size ${2} \
        --nthreads 8 \
        --method lirr \
        --trade_off 1 \
        --temp 0.05 \
        --cosine \
        --logger_file_name officehome_lirr_cosine
    done
done