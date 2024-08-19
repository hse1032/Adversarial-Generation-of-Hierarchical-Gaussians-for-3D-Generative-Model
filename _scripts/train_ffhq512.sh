export CUDA_LAUNCH_BLOCKING=0

# hyper-parameters
kimg=15000
gen_pose_cond=True
use_pe=True
metric=fid2k_full # fid2k-full for training
blur_fade_kimg=200
gamma=1
gpc_reg_prob=0.5
center_dists=1.0
prob_uniform=0.5 # only use for FFHQ
res_end=256
num_pts=256
nrr=512

expname=base
dataset=FFHQ512
dataset_path=../datasets/FFHQ_512.zip
outdir=../training-runs/${dataset}/${expname}
ngpus=8

# FFHQ512
python train.py --outdir=${outdir} --cfg=ffhq --data=${dataset_path} \
  --gpus=${ngpus} --batch=32 --gamma=${gamma} --gen_pose_cond=${gen_pose_cond} --neural_rendering_resolution_initial=${nrr} --metrics=${metric} --blur_fade_kimg=${blur_fade_kimg} \
  --kimg=${kimg} --gaussian_num_pts=${num_pts} --start_pe=${use_pe} --gpc_reg_prob=${gpc_reg_prob} \
  --center_dists=${center_dists} --prob_uniform=${prob_uniform} --res_end=${res_end}
