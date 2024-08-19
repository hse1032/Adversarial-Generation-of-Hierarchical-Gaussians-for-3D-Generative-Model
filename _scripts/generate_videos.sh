network_pkl=dummy
resolution=256

trunc=0.7
seeds=0-3
opacity_ones=False

network_pkl=training-runs/FFHQ256/base/00003-ffhq-FFHQ_256-gpus4-batch32-gamma1/network-snapshot-007800.pkl
network_pkl=ffhq256_final.pkl
# network_pkl=ffhq256.pkl

python gen_videos_gsparams.py --outdir=out/ --trunc=${trunc} --seeds=${seeds} --grid=2x2 \
    --network=${network_pkl} --image_mode image --g_type=G_ema --load_architecture=False \
    --nrr=${resolution} --postfix=_${seeds} --opacity_ones=${opacity_ones}
