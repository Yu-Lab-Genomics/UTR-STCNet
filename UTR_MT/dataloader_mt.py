from csv import Error
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd

class DatasetWrapper(Dataset):
    """ 数据集包装类（添加来源标识）"""
    def __init__(self, base_dataset, dataset_id, seq_maxlength):
        if "MPA_" in base_dataset:
            self.dataset = pd.read_csv(base_dataset, usecols=["utr", "rl"])
        elif "RP_" in base_dataset:
            self.dataset = pd.read_csv(base_dataset, usecols=["utr", "log_te"])
            self.dataset = self.dataset[["utr", "log_te"]]
        else:
            assert Error
        self.seq_max_length = seq_maxlength
        self.dataset = self.dataset.values.tolist()
        self.dataset_id = dataset_id
        self.seq_map = {
                "A":[1, 0, 0, 0],
                "C":[0, 1, 0, 0],
                "G":[0, 0, 1, 0],
                "T":[0, 0, 0, 1],
                "N":[0, 0, 0, 0],
        }
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # data, label = self.dataset[idx]
        data_ori, data, label = self.encoder(idx)
        return data_ori, data, label, self.dataset_id
    
    def encoder(self, idx):
        seq_str, seq_label = self.dataset[idx]
        if len(seq_str) > self.seq_max_length:
            seq_str_ori = seq_str[-self.seq_max_length:]
        else:
            seq_str_ori = seq_str
        tokens = torch.zeros((self.seq_max_length, 4))
        encoder_seq = torch.tensor([self.seq_map[item] for item in seq_str_ori], dtype=torch.int64)
        tokens[-len(encoder_seq):]= encoder_seq
        return seq_str, tokens, seq_label

class DatasetSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_sizes, main_id, main_samples, other_samples, max_epochs=1):
        """
        :param dataset_sizes: 各数据集大小 [size1, size2, size3]
        :param main_id: 主数据集ID（从0开始）
        :param main_samples: 主数据集每批次采样数
        :param other_samples: 其他数据集每批次采样数（每个数据集）
        :param max_epochs: 最大训练轮次
        """
        self.dataset_sizes = dataset_sizes
        self.main_id = main_id
        self.main_samples = int(main_samples)
        self.other_samples = int(other_samples)

        # 计算全局索引偏移
        self.offsets = [0] * len(dataset_sizes)
        current = 0
        for i, size in enumerate(dataset_sizes):
            self.offsets[i] = current
            current += size

        # 主数据集参数
        self.main_size = dataset_sizes[main_id]
        self.batches_per_epoch = self.main_size // self.main_samples  # 向下取整
        self.total_batches = self.batches_per_epoch * max_epochs

    def __iter__(self):
        # 生成主数据集的不重复索引序列
        main_indices = torch.randperm(self.main_size)  # 每个epoch重新洗牌
        for batch_idx in range(self.batches_per_epoch):
            # 获取主数据集样本
            start = batch_idx * self.main_samples
            end = start + self.main_samples
            main_batch = main_indices[start:end]
            global_main = main_batch + self.offsets[self.main_id]

            # 获取其他数据集样本（允许重复）
            other_indices = []
            for i, size in enumerate(self.dataset_sizes):
                if i == self.main_id:
                    continue
                indices = torch.randint(0, size, (self.other_samples,))
                global_other = indices + self.offsets[i]
                other_indices.append(global_other)

            # 合并并打乱顺序
            all_indices = torch.cat([global_main] + other_indices)
            all_indices = all_indices[torch.randperm(len(all_indices))]
            yield all_indices.tolist()

    def __len__(self):
        return self.total_batches

# # 使用示例
# if __name__ == "__main__":
#     # 创建模拟数据集
#     data_dir = "/home/liuzhouwu/UTRInsight/UTRModel/dataset"
#     file_list = ["MPA_H_train_val.csv", "MPA_U_train_val.csv", "MPA_V_train_val.csv"]

#     batch_size = 256
#     # 包装数据集并合并
#     concat_dataset = ConcatDataset([DatasetWrapper(f"{data_dir}/{item}", idx) for idx, item in enumerate(file_list)])
#     dataset_sizes = [len(item) for item in concat_dataset.datasets]
#     # 初始化采样器
#     sampler = DatasetSampler(
#         dataset_sizes=dataset_sizes,
#         main_id=2,                                      # 主数据集为第三个
#         main_samples=batch_size,                        # 主数据集每批8个（不允许重复）
#         other_samples=batch_size*0.5,                    # 其他数据集每批各2个（允许重复）
#     )

#     # 创建DataLoader
#     dataloader = DataLoader(
#         concat_dataset,
#         batch_sampler=sampler,
#         num_workers=4,
#         pin_memory=True
#     )

#     for i in range(1):
#         # 验证第一个batch
#         for batch in dataloader:
#             data, labels, sources = batch
#             print(
#                 f"Batch大小: {len(data)}, 来源分布 - "
#                 f"数据集0: {(sources == 0).sum().item()}, "
#                 f"数据集1: {(sources == 1).sum().item()}, "
#                 f"主数据集2: {(sources == 2).sum().item()}"
#             )
#             break
        