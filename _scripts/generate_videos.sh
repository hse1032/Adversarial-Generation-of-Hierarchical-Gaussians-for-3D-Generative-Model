network_pkl=dummy
resolution=256

trunc=0.7
seeds=0-3
opacity_ones=False

python gen_videos_gsparams.py --outdir=out/ --trunc=${trunc} --seeds=${seeds} --grid=2x2 \
    --network=${network_pkl} --image_mode image --g_type=G_ema --load_architecture=False \
    --nrr=${resolution} --postfix=_${seeds} --opacity_ones=${opacity_ones}
