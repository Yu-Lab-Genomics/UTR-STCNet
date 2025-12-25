#  large  k=3  [0.5, 0.5, 0.5] padd_length = max_length
CUDA_VISIBLE_DEVICES=0 python main.py --seq_type utr --prefix UTR_MRPA_H --label-class rl  --batchsize 128  --seq_max_len 120 --epochs 300 --train_file data/MPA/MPA_H_train_val.csv --val_file data/MPA/MPA_H_test.csv
