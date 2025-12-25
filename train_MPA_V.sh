# 
CUDA_VISIBLE_DEVICES=2 python train.py --seq_type utr --prefix UTR_MRPA_V --label-class rl  --batchsize 128  --seq_max_len 120 --epochs 120 --train_file data/MPA/MPA_V_Random_train_val.csv --val_file data/MPA/MPA_V_Random_test.csv
