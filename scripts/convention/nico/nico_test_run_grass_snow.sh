#!/usr/bin/env zsh
cd /home/v-boli4/codebases/DA_Codebase
for method in 'source_only' 'dann' 'cdan' 'adr' 'irm' 'mme' 'lirr'
do
  python main.py \
  --dataset nico \
  --domain_shift_type convention \
  --source snow \
  --target grass \
  --nepoch 20 \
  --model_name resnet34 \
  --image_size 224 \
  --channels 3 \
  --num_cls 8 \
  --lr 0.001 \
  --milestone 10 \
  --data_root /home/v-boli4/codebases/external_datasets/NICO-ANIMAL \
  --outf  /home/v-boli4/codebases/DA_Codebase/output/convention/nico/snow_grass \
  --logf /home/v-boli4/codebases/DA_Codebase/output/convention/nico/grass_snow_nico_$method.txt \
  --batch_size 8 \
  --nthreads 8 \
  --method $method \
  --logger_file_name nico
done