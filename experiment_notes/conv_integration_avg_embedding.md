# Setup

`python main.py --model CEDSR --data_train Coco --data_test Coco --use_classification --pre_train download --data_range all --scale 4 --save_models --test_every 100 --batch_size 16 --epochs 20 --save convavg --n_GPUs 4`

Epoch 20: PSNR: 25.233 (Best: 25.235 @epoch 19)
Baseline: PSNR: 24.034

A marginal improvement - we could probably have gotten an equivalent PSNR just by training the baseline for more epochs on Coco

Perceptual results may be worse

