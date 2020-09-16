#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
#CUDA_VISIBLE_DEVICES=0
methods=('source_only' 'dann' 'lirr' 'irm')
params=""
for((i=1;i<=$#;i++)); do 
    j=${!i}
    params="${params} $j "
done
for((i = 0; i < 4; i++))
do
    luban offline submit --username=wangyezhen_i --token=83d04072cb6e48e18511bf55e03713ad --projectId=92 --imageId=9329 --scriptPath=/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/scripts/convention/citycam/citycam-${methods[${i}]}.sh --gpus=1 --jobName=citycam-${methods[${i}]}-${1} --scriptParam="${params}" --backoffLimit=0 --vo volume-92-5
done