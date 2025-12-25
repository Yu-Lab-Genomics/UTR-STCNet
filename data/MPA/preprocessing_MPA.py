import os
import sys
import torch

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler


# ========== 1. preprocessing MPA datasets ==========

# read csv and load in to a dictionary
Ex_data_dir = "data"
csv_path_ls = [os.path.join(Ex_data_dir,csv) for csv in ['GSM3130435_egfp_unmod_1.csv','GSM3130443_designed_library.csv','GSM4084997_varying_length_25to100.csv']]
for path in csv_path_ls:
    assert os.path.exists(path), f"The file {path} is not properly downloaded"

df_dict = {
  csv_path.split("_")[-1].replace(".csv","") : pd.read_csv(csv_path,low_memory=False)  for csv_path in csv_path_ls
           }
# align the columns name across datasets
df_dict['library'].rename({'total':'total_reads'},axis=1,inplace=True)
df_dict['25to100']['r13'] = np.zeros((df_dict['25to100'].shape[0],))

ttt = np.zeros((df_dict['25to100'].shape[0],))
columns_to_save = ['utr', 'rl']
# **************************************************  - MPA_U -   **************************************************
df = df_dict['1']
df.sort_values('total_reads', inplace=True, ascending=False)
df.reset_index(inplace=True, drop=True)
## train_val test spliting
# the most abundant 5'UTR is used as test set
test_df = df.iloc[:20000]
train_val_df = df.iloc[20000:280000]
test_df[columns_to_save].to_csv(os.path.join("data/MPA","MPA_U_test.csv"),index=False)
train_val_df[columns_to_save].to_csv(os.path.join("data/MPA","MPA_U_train_val.csv"),index=False)

# **************************************************  - MPA_H -   **************************************************
df = df_dict['library']
# taking the natrual utr and sort by reads count
human_lib = df[(df['library'] == 'human_utrs') | (df['library'] == 'snv')]
human_lib = human_lib.sort_values('total_reads', ascending=False).reset_index(drop=True)

# the top 25k abundant reads as test set
sub = human_lib.iloc[:7600]
remaining = human_lib.iloc[7600:]
sub[columns_to_save].to_csv(os.path.join("data/MPA","MPA_H_test.csv"),index=False)
remaining[columns_to_save].to_csv(os.path.join("data/MPA","MPA_H_train_val.csv"),index=False)

# **************************************************  - MPA_V -   **************************************************
df1 = df_dict['25to100']
# take the random seqs
df = df1[df1['set']=='random']
## Filter out UTRs with too few less reads
df=df[df['total_reads']>=10]
df['utr100'] = 75*'N' +df['utr']
df['utr100'] = df['utr100'].str[-100:]
df.sort_values('total_reads', inplace=True, ascending=False)
df.reset_index(inplace=True, drop=True)

## some natural sequence
human = df1[df1['set']=='human']
## Filter out UTRs with too few less reads
human=human[human['total_reads']>=10]
human['utr100'] = 75*'N' +human['utr']
human['utr100'] = human['utr100'].str[-100:]
human.sort_values('total_reads', inplace=True, ascending=False)
human.reset_index(inplace=True, drop=True)

e_test = pd.DataFrame(columns=df.columns)
e_train = pd.DataFrame(columns=df.columns)
for i in range(25,101):
    tmp = df[df['len']==i]
    tmp.sort_values('total_reads', inplace=True, ascending=False)
    tmp.reset_index(inplace=True, drop=True)
    e_test = pd.concat([e_test, tmp.iloc[:100]], ignore_index=True)
    e_train = pd.concat([e_train, tmp.iloc[100:]], ignore_index=True)

e_test[columns_to_save].to_csv(os.path.join("data/MPA","MPA_V_Random_test.csv"),index=False)
e_train[columns_to_save].to_csv(os.path.join("data/MPA","MPA_V_Random_train_val.csv"),index=False)

subhuman_test = pd.DataFrame(columns=human.columns)
subhuman_train = pd.DataFrame(columns=human.columns)
for i in range(25,101):
    tmp = human[human['len']==i]
    tmp.sort_values('total_reads', inplace=True, ascending=False)
    tmp.reset_index(inplace=True, drop=True)
    subhuman_test = pd.concat([subhuman_test, tmp.iloc[:30]], ignore_index=True)
    subhuman_train = pd.concat([subhuman_train, tmp.iloc[30:]], ignore_index=True)

subhuman_test[columns_to_save].to_csv(os.path.join("data/MPA","MPA_V_Human_test.csv"),index=False)
subhuman_train[columns_to_save].to_csv(os.path.join("data/MPA","MPA_V_Human_train_val.csv"),index=False)



print("The preprocssing for MPA tasks is Finished !!")