import os
import torch
from tqdm import tqdm
import numpy as np
import argparse

from UTR.utr_dataset_all import load_test_data, UTRDATA
from UTR.UTRFormer import utrformer_large as utrformer
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', type=str, default='0')
parser.add_argument('--prefix', type=str, default='UTR')
parser.add_argument('--seq_type', type=str, default='utr')
parser.add_argument('--label_class', type=str, default='rl')
parser.add_argument('--val_file', type=str, required=True)
parser.add_argument('--modelfile', type=str, required=True)
parser.add_argument('--batchsize', type=int, default=2048)
parser.add_argument('--seq_max_len', type=int, default=120)
args = parser.parse_args()


device_ids = list(map(int, args.device_ids.split(',')))
device = torch.device(f'cuda:{device_ids[0]}')


prefix = f'{args.prefix}_{args.label_class.upper()}'
print(f"Running inference")


val_data = load_test_data(args.val_file)
val_dataset = UTRDATA(val_data, args.seq_type, seqs_max_length=args.seq_max_len, label_class=args.label_class.lower())

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=args.batchsize,
                                         num_workers=4,
                                         shuffle=False)


model = utrformer(padding_idx=val_dataset.padding_idx,
                  token_cls=val_dataset.token_class,
                  pooling_size=args.seq_max_len).to(device)

model.load_state_dict({
    k.replace('module.', ''): v for k, v in torch.load(args.modelfile)["model"].items()
})
model.eval()


true_label = []
predict_label = []

with torch.no_grad():
    for _, batch in enumerate(tqdm(val_loader)):
        strs, tokens, labels = batch
        input = tokens.to(device)
        targets = torch.tensor(labels).to(device)

        out, _ = model(input)
        true_label.extend(targets.tolist())

        if len(out) == 1:
            predict_label.append(out.item())
        else:
            predict_label.extend(out.squeeze().cpu().detach().tolist())


y_true = true_label
y_pred = predict_label
r2 = r2_score(y_true, y_pred)
spearman_corr, _ = spearmanr(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Evaluation Results:")
print(f"  R-squared     : {round(r2, 4)}")
print(f"  Spearman R    : {round(spearman_corr, 4)}")
print(f"  RMSE          : {round(rmse, 4)}\n")
