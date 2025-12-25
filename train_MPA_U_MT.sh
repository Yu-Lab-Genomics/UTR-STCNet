#  large  k=3  [0.5, 0.5, 0.5] padd_length = 120
CUDA_VISIBLE_DEVICES=2  python train_mt.py --seq_type utr --prefix UTR_MRPA  --task MAP_U_H_V --seq_maxlen 120 --batchsize 256  
