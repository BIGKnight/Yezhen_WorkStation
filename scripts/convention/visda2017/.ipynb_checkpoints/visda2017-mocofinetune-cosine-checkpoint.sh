#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
cd /nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/
#CUDA_VISIBLE_DEVICES=0
datasets=(${1} 'validation')
logf_root='/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/output/convention/visda2017/sourceonly/'

for((i = 0; i < 1; i++))
do
    for((j = 0; j < 2; j++))
    do
        source_domain=${datasets[${i}]}
        target_domain=${datasets[${j}]}
#         if [ ${source_domain} = ${target_domain} ]
#         then
#             continue
#         else
            echo "source: ${source_domain}; target: ${target_domain}."
            python3 main.py \
            --dataset visda2017 \
            --domain_shift_type convention \
            --source ${source_domain} \
            --target ${target_domain} \
            --nepoch 100 \
            --model_name resnet101 \
            --image_size 224 \
            --channels 3 \
            --num_cls 12 \
            --lr 0.0001 \
            --milestone 100 \
            --data_root /nfs/volume-92-5/wangyezhen_i/Datasets/visda2017 \
            --outf /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/${source_domain}_${target_domain}_visda2017_mocofinetune \
            --logf ${logf_root}${source_domain}_${target_domain}_visda2017_mocofinetune.txt \
            --batch_size 64 \
            --nthreads 8 \
            --method moco \
            --trade_off 0.1 \
            --K 8192 \
            --temp 0.07 \
            --m 0.998 \
            --moco_finetune \
            --cosine_classifer \
            --load_path /nfs/volume-92-5/wangyezhen_i/CheckPoints/CLMS/train_validation_visda2017_moco/net_epoch_100.pth \
            --frozen conv1 bn1 layer1 layer2 layer3 layer4 bottleneck \
            --logger_file_name visda2017_mocofinetune
#         fi
    done
done