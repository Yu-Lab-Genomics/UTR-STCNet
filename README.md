#  UTR-STCNet
This repository provides the code for the paper "UTR-STCNet: Decoding Translation-Related Functional Sequences in 5’UTRs Using Interpretable Deep Learning Models".

## Requirements
- torch == 2.4.1

## Data
Please download the dataset and place it under the `.data/` directory at the project root.These files can be downloaded at: https://drive.google.com/file/d/12td9ohx01xxaFnPP_dXzcHTJWH62TRVv/view?usp=drive_link

## Checkpoints
Please download the dataset and place it under the `.checkpoint/` directory at the project root.These files can be downloaded at: https://drive.google.com/file/d/1JPro_Dshr3MMYUX-j6ginB4O7teLZiIA/view?usp=drive_link

## Model Training
```
bash train_MPA_H.sh
```

## Multi Task
```
bash train_MPA_U_MT.sh
```

## Model Testing
```
python evaluation.py --val_file data/MPA/MPA_H_test.csv --modelfile xxx
```

## Requirements
If you find our work useful for your research, please cite:
```

```
