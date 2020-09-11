#!/bin/bash
source /nfs/project/wangyezhen/.Pytorch_Env
#CUDA_VISIBLE_DEVICES=0

luban offline submit \
--username=wangyezhen_i \
--token=83d04072cb6e48e18511bf55e03713ad \
--projectId=92 \
--imageId=7894 \
--scriptPath=/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/scripts/domainnet_LDS/domainnet-${1}.sh \
--gpus=1 \
--jobName="I-C-domainnet-LDS-${1}" \
--scriptParam="infograph clipart" \
--backoffLimit=0 \
--vo volume-92-5

luban offline submit \
--username=wangyezhen_i \
--token=83d04072cb6e48e18511bf55e03713ad \
--projectId=92 \
--imageId=7894 \
--scriptPath=/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/scripts/domainnet_LDS/domainnet-${1}.sh \
--gpus=1 \
--jobName="C-R-domainnet-LDS-${1}" \
--scriptParam="clipart real" \
--backoffLimit=0 \
--vo volume-92-5

luban offline submit \
--username=wangyezhen_i \
--token=83d04072cb6e48e18511bf55e03713ad \
--projectId=92 \
--imageId=7894 \
--scriptPath=/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/scripts/domainnet_LDS/domainnet-${1}.sh \
--gpus=1 \
--jobName="C-S-domainnet-LDS-${1}" \
--scriptParam="clipart sketch" \
--backoffLimit=0 \
--vo volume-92-5

luban offline submit \
--username=wangyezhen_i \
--token=83d04072cb6e48e18511bf55e03713ad \
--projectId=92 \
--imageId=7894 \
--scriptPath=/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/scripts/domainnet_LDS/domainnet-${1}.sh \
--gpus=1 \
--jobName="S-P-domainnet-LDS-${1}" \
--scriptParam="sketch painting" \
--backoffLimit=0 \
--vo volume-92-5

luban offline submit \
--username=wangyezhen_i \
--token=83d04072cb6e48e18511bf55e03713ad \
--projectId=92 \
--imageId=7894 \
--scriptPath=/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/scripts/domainnet_LDS/domainnet-${1}.sh \
--gpus=1 \
--jobName="P-R-domainnet-LDS-${1}" \
--scriptParam="painting real" \
--backoffLimit=0 \
--vo volume-92-5

luban offline submit \
--username=wangyezhen_i \
--token=83d04072cb6e48e18511bf55e03713ad \
--projectId=92 \
--imageId=7894 \
--scriptPath=/nfs/volume-92-5/wangyezhen_i/Projects/Theoretical_Projects/InstaPBM-V1/scripts/domainnet_LDS/domainnet-${1}.sh \
--gpus=1 \
--jobName="R-C-domainnet-LDS-${1}" \
--scriptParam="real clipart" \
--backoffLimit=0 \
--vo volume-92-5