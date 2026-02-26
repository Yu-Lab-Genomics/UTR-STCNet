# UTR-STCNet

This repository provides the code for the paper **"Decoding Translation-Related Functional Sequences in 5’UTRs Using Interpretable Deep Learning Models"** (IEEE BIBM 2025).




## Requirements

- Python >= 3.8
- torch == 2.4.1
- scikit-learn
- scipy
- pandas
- numpy
- matplotlib
- tqdm
- pyyaml

## Data

Please download the dataset and place it under the `data/` directory at the project root. These files can be downloaded at: https://drive.google.com/file/d/12td9ohx01xxaFnPP_dXzcHTJWH62TRVv/view?usp=drive_link



## Checkpoints

Please download the checkpoints and place them under the `checkpoint/` directory at the project root. These files can be downloaded at: https://drive.google.com/file/d/1JPro_Dshr3MMYUX-j6ginB4O7teLZiIA/view?usp=drive_link

## Model Training

### Single-Task Training

```bash
# Train on MPA_H dataset
bash train_MPA_H.sh

# Train on MPA_U dataset
bash train_MPA_U.sh

# Train on MPA_V dataset
bash train_MPA_V.sh
```

### Multi-Task Training

Multi-task training is configured via `mt.yaml`. Available task configurations:
- `MAP_U_V`: Joint training on MPA_U + MPA_V
- `MAP_U_H_V`: Joint training on MPA_U + MPA_H + MPA_V

```bash
bash train_MPA_U_MT.sh
```

## Model Evaluation

```bash
python evaluation.py --val_file data/MPA/MPA_H_test.csv --modelfile checkpoint/your_model.pt
```



## Citation

If you find our work useful for your research, please cite:

```bibtex
@inproceedings{lin2025decoding,
  title={Decoding Translation-Related Functional Sequences in 5’UTRs Using Interpretable Deep Learning Models},
  author={Lin, Yuxi and Fang, Yaxue and Zhang, Zehong and Liu, Zhouwu and Zhong, Siyun and Wang, Zhongfang and Yu, Fulong},
  booktitle={2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={6873--6880},
  year={2025},
  organization={IEEE}
}
```
