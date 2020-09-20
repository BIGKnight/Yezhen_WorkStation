#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env

cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=('train' 'validation')
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/visda2017/instapbm/'

for((i = 0; i < 1; i++))
do
    for((j = 1; j < 2; j++))
    do
        source_domain=${datasets[${i}]}
        target_domain=${datasets[${j}]}
        if [ ${source_domain} = ${target_domain} ]
        then
            continue
        else
            echo "source: ${source_domain}; target: ${target_domain}."
            python3 main.py \
            --dataset visda2017 \
            --domain_shift_type convention \
            --source ${source_domain} \
            --target ${target_domain} \
            --nepoch 30 \
            --model_name ${1} \
            --image_size 224 \
            --channels 3 \
            --num_cls 12 \
            --lr 0.001 \
            --milestone 20 \
            --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/visda2017 \
            --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_visda17_mimcontra \
            --logf ${logf_root}${source_domain}_${target_domain}_mimcontra.txt \
            --batch_size ${2} \
            --nthreads 8 \
            --method mimcontra \
            --trade_off 0.1 \
            --lambda_irm 1 \
            --logger_file_name visda2017_mimcontra_convention
        fi
    done
done