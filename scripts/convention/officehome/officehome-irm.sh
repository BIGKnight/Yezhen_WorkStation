#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=('Art' 'RealWorld' 'Clipart' 'Product')
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/officehome/irm/'

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
        --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_officehome_irm \
        --logf ${logf_root}${source_domain}_${target_domain}_officehome_irm.txt \
        --batch_size ${2} \
        --nthreads 8 \
        --method irm \
        --trade_off 1 \
        --logger_file_name officehome_irm
    done
done