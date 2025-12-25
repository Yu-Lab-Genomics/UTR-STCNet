from typing import List
import math
import random
from copy import deepcopy
from typing import Sequence, Tuple, List, Union
import itertools
from sympy import N
import torch
from typing import Any, Callable, Optional, Tuple, List
import torch.utils.data as data
import os
import numpy as np
import pandas as pd


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """
    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]): 
        batch_size = len(raw_batch)
        batch_labels, seq_str_list, masked_seq_str_list, masked_indices_list, seq_tok_type = zip(*raw_batch)
       
        masked_seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in masked_seq_str_list] 
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list] 
     
        max_len = max(len(seq_encoded) for seq_encoded in masked_seq_encoded_list)
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        masked_tokens = deepcopy(tokens)
        
        labels = []
        strs, masked_strs = [], []
        masked_indices = []
        for i, (label, seq_str, masked_seq_str, seq_encoded, masked_seq_encoded, indices_mask) in enumerate(
            zip(batch_labels, seq_str_list, masked_seq_str_list, seq_encoded_list, masked_seq_encoded_list, masked_indices_list) 
        ):
            labels.append(label)
            strs.append(seq_str)
            masked_strs.append(masked_seq_str)
            masked_indices.append(indices_mask)
            
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
                masked_tokens[i, 0] = self.alphabet.cls_idx
                
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            masked_seq = torch.tensor(masked_seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            
            masked_tokens[
                i,
                int(self.alphabet.prepend_bos) : len(masked_seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = masked_seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                masked_tokens[i, len(masked_seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        return labels, strs, masked_strs, tokens, masked_tokens, masked_indices, seq_tok_type


class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<pad>", "<unk>"), 
        append_toks: Sequence[str] = ("<cls>", "<mask>"), 
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
        mask_prob: float = 0.15, 
    ):
        self.mask_prob = mask_prob 
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.all_special_tokens = ['<pad>', '<mask>'] 
        self.unique_no_split_tokens = self.all_toks

        print(self.tok_to_idx)

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self):
        return BatchConverter(self)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


class FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs, secondary_structure, mfe, mask_prob = 0.15):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
        self.secondary_structure = list(secondary_structure)
        self.mfe = list(mfe)
        self.mask_prob = mask_prob
        
    @classmethod
    def from_file(cls, fasta_file, mask_prob = 0.15):
        sequence_labels, sequence_strs, secondary_structure,MFE = [], [], [], []
        cur_seq_label = None
        cur_ss = None
        cur_mfe = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, cur_ss, buf, cur_mfe
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            secondary_structure.append(cur_ss)
            MFE.append(cur_mfe)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            cur_ss = None
            cur_mfe = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"): 
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line.split("|")[-2]
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                    cur_ss = line.split("|")[1]
                    cur_mfe = line.split("|")[0]
                else:  
                    assert len(line.strip()) == len(cur_ss)
                    new_line = line.strip()
                    buf.append(new_line)

        _flush_current_seq()
        assert len(set(sequence_strs)) == len(sequence_strs)
        return cls(sequence_labels, sequence_strs, secondary_structure, MFE, mask_prob)

    def __len__(self):
        return len(self.sequence_labels)
    
    def mask_sequence(self, seq): 
        length = len(seq)
        max_length = math.ceil(length * self.mask_prob)
        rand = random.sample(range(0, length), max_length)
        # res = ''.join(['<mask>' if idx in rand else ele for idx, ele in enumerate(seq)])
        res = ''.join(['' if idx in rand else ele for idx, ele in enumerate(seq)])
        return rand, res
    
    def __getitem__(self, idx):
        sequence_str = self.sequence_strs[idx]
        sequence_label = self.sequence_labels[idx]
        sequence_token_type = self.sequence_token_type[idx]
        masked_indices, masked_sequence_str = self.mask_sequence(sequence_str)
        return sequence_label, sequence_str, masked_sequence_str, masked_indices, sequence_token_type
        # return sequence_label, sequence_str, masked_indices, masked_sequence_str

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0, train_data_index=None):
        if train_data_index is not None:
            sequence_strs = [self.sequence_strs[i] for i in train_data_index]
        else:
            sequence_strs = self.sequence_strs
        sizes = [(len(s), i) for i, s in enumerate(sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches
    
    def count_all_data_length(self):
        sizes = [len(s) for s in self.sequence_strs]
        sizes.sort()
        mean_of_length = round(sum(sizes) / len(sizes))
        max_value = max(sizes)
        median_value = sizes[len(sizes) // 2]
        min_value = min(sizes)
        if len(sizes) % 2 == 0:
            middle = len(sizes) // 2
            median_value = (sizes[middle - 1] + sizes[middle]) / 2
        return mean_of_length, median_value, min_value, max_value
    
    def count_atg(self):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        atg_num = 0
        without_atg_num = 0
        for _, s_idx in sizes:
            if "ATG" in self.sequence_strs[s_idx].upper():
                atg_num += 1
            else:
                without_atg_num += 1
        return atg_num, without_atg_num

    def build_train_dataset(self):
        train_idx = []
        test_idx = []
        for sp in set(self.sequence_labels):
            idx_of_sp = [index for index, value in enumerate(self.sequence_labels) if value == sp]
            sp_seqs_atg = []
            sp_seqs_not_atg = []
            for i in idx_of_sp:
                if "ATG" in self.sequence_strs[i].upper():
                    sp_seqs_atg.append(i)
                else:
                    sp_seqs_not_atg.append(i)
            sp_seqs_atg_sizes = [(len(self.sequence_strs[j]), j) for j in sp_seqs_atg]
            sp_seqs_atg_sizes.sort()
            sp_seqs_not_atg_sizes = [(len(self.sequence_strs[j]), j) for j in sp_seqs_not_atg]
            sp_seqs_not_atg_sizes.sort()
            ratio = round((len(sp_seqs_atg_sizes)/len(sp_seqs_not_atg_sizes))*10)
            train_ratio = 0.8
            for k in range(0, len(sp_seqs_atg_sizes), ratio):
                if k+ratio+1 > len(sp_seqs_atg_sizes):
                    k2 = len(sp_seqs_atg_sizes)
                else:
                    k2 = k+ratio
                t1 = sp_seqs_atg_sizes[k:k2]
                selected_numbers = random.sample(t1, int(len(t1)*train_ratio))
                remaining_numbers = [num for num in t1 if num not in selected_numbers]
                train_idx += selected_numbers
                test_idx += remaining_numbers

            for m in range(0, len(sp_seqs_not_atg_sizes), 10):
                if m+10+1 > len(sp_seqs_not_atg_sizes):
                    m2 = len(sp_seqs_not_atg_sizes)
                else:
                    m2 = m+10
                t2 = sp_seqs_not_atg_sizes[m:m2]
                selected_numbers = random.sample(t2, int(len(t2)*train_ratio))
                remaining_numbers = [num for num in t2 if num not in selected_numbers]
                train_idx += selected_numbers
                test_idx += remaining_numbers
        train_list = [item[1] for item in train_idx]
        test_list = [item[1] for item in test_idx]
        return train_list, test_list



# train_fasta = '/home/liuzhouwu/Dataset/FiveSpecies_Cao_allutr_with_energyNormalDist_structure.fasta'
# dataset = FastaBatchedDataset.from_file(train_fasta, mask_prob = 0.15)
# toks_per_batch = 4096 * 3
# batches = dataset.get_batch_indices(toks_per_batch = toks_per_batch, extra_toks_per_seq = 1)

# alphabet = Alphabet(mask_prob = 0.15, standard_toks = 'AGCT(.)')
# batches_loader = torch.utils.data.DataLoader(batches, 
#                                              batch_size = 1,
#                                              num_workers = 8,
#                                              shuffle= True)
# for i, batch in enumerate(batches_loader):
#     batch = torch.LongTensor(batch).numpy().tolist()
#     dataloader = torch.utils.data.DataLoader(dataset, 
#                                                 collate_fn=alphabet.get_batch_converter(), 
#                                                 batch_sampler=[batch], 
#                                                 shuffle = False)
#     for i, (labels, strs, masked_strs, toks, masked_toks, ttt, seq_tok_type) in enumerate(dataloader):
#         print(f"{i}>>>>>", "*"*50)
#         print(strs[0])
#         print(masked_strs[0])
#         print(toks[0])
#         print(masked_toks[0])
#         print(sorted(ttt[0]))


class StandardTransform:
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)

class BaseDataset(data.Dataset):
    def __init__(
        self,
        root: str = None,  # type: ignore[assignment]
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]


class RawData():
    def __init__(self, fname, seq="utr", label="rl"):
        self.df = pd.read_csv(fname)
        # self.df = self.df.sort_values(by=['total'], ascending=False).reset_index(drop=True)
        self.label = label
        self.input_seq = seq
        self.df_data = self.df.loc[:, [self.input_seq, self.label]]
        self.get_seq_map()
    
    def get_seq_map(self):
        if self.input_seq == "utr":
            self.seq_map = {
                "A":1,
                "C":2,
                "G":3,
                "T":4,
                "-":0
            }
        elif self.input_seq == "utr":
            self.seq_map = {
                "(":1,
                ".":2,
                ")":3,
                "-":0
            }
        else:
            self.seq_map = None

    def get_df(self):
        return self.df_data

    def get_seqs(self):
        return self.df_data[self.input_seq].to_list()

    def get_labels(self):
        return self.df_data[self.label]
    
    def encoder(self):
        encoder_seq = []
        for sub_seq in self.get_seqs():
            encoder_seq.append([self.seq_map[item] for item in sub_seq])
        return torch.Tensor(encoder_seq)

class RawData_SS():
    def __init__(self, fname):
        self.df = pd.read_csv(fname)
        # self.df = self.df.sort_values(by=['total'], ascending=False).reset_index(drop=True)
        self.df_data = self.df.loc[:, ['ss', 'rl']]

    def get_df(self):
        return self.df_data

    def get_seqs(self):
        return self.df_data['ss'].to_list()

    def get_labels(self, label):
        return self.df_data[label]
    
    def encoder(self):
        seq_map = {
            "(":1,
            ".":2,
            ")":3,
            "-":0
            }
        encoder_seq = []
        for sub_seq in self.get_seqs():
            encoder_seq.append([seq_map[item] for item in sub_seq])
        return torch.Tensor(encoder_seq)
    
class UTRDATA(BaseDataset):
    def __init__(
        self,
        root: str,
        utr: True,
        label_class: Optional[Callable] ="rl", # "te", "rpkm_rnaseq"
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        if utr:
            rawdata = RawData(root)
        else:
            rawdata = RawData_SS(root)
        self.data = rawdata.encoder()
        self.target = rawdata.get_labels()
        print(111)

    def __getitem__(self, index: int):
        input_data, input_target = self.data[index], self.target[index]
        assert input_target is not None
        return input_data, input_target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

# file = "/pool1/liuzhouwu/datasets/5UTR/MRL_Random50Nuc_SynthesisLibrary_Sample/4.1_train_data_GSM3130435_egfp_unmod_1.csv"
# data = UTRDATA(file, utr=True)
# for i in range(10):
#     ttt = data[i]