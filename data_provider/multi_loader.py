import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from sklearn.preprocessing import StandardScaler
from typing import List

# ---------- Base dataset for one CSV ----------
class SingleElectricityDataset(Dataset):
    def __init__(self, csv_path, dataset_id, seq_len, label_len, pred_len, teacher_dim=42):
        self.dataset_id = dataset_id
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.teacher_dim = teacher_dim

        df = pd.read_csv(csv_path)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        data = df.values.astype(np.float32)

        # Pad or truncate to teacher_dim
        n_feat = data.shape[1]
        if n_feat < teacher_dim:
            pad = np.zeros((data.shape[0], teacher_dim - n_feat), dtype=np.float32)
            data = np.concatenate([data, pad], axis=1)
        elif n_feat > teacher_dim:
            data = data[:, :teacher_dim]

        self.data = data
        self.len = len(data)


    def __len__(self):
        return self.len - self.seq_len - self.pred_len + 1
    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        # Time encodings (you can leave as placeholders or add actual values)
        x_mark = np.zeros((self.seq_len, 4), dtype=np.float32)
        y_mark = np.zeros((self.label_len + self.pred_len, 4), dtype=np.float32)

        return (
            torch.tensor(seq_x, dtype=torch.float32),      # ✅ batch_x
            torch.tensor(seq_y, dtype=torch.float32),      # ✅ batch_y
            torch.tensor(x_mark, dtype=torch.float32),     # ✅ batch_x_mark
            torch.tensor(y_mark, dtype=torch.float32),     # ✅ batch_y_mark
            torch.tensor(self.dataset_id, dtype=torch.long)
        )



# ---------- Balanced sampler ----------
class BalancedSampler(Sampler):
    """Samples equally from each dataset in a ConcatDataset."""
    def __init__(self, datasets: List[Dataset], samples_per_dataset=None):
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.dataset_sizes = [len(d) for d in datasets]
        self.max_size = max(self.dataset_sizes)
        self.samples_per_dataset = samples_per_dataset or self.max_size

    def __iter__(self):
        indices = []
        offset = 0
        for i, size in enumerate(self.dataset_sizes):
            sample_count = self.samples_per_dataset
            rand_idx = np.random.randint(0, size, sample_count)
            indices.extend(list(rand_idx + offset))
            offset += size
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_dataset * self.num_datasets

# ---------- Loader builder ----------
def build_multi_loader(manifest_path, root_path, seq_len, label_len, pred_len, batch_size, teacher_dim=42):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    datasets = []
    for entry in manifest["datasets"]:
        csv_path = os.path.join(root_path, entry["name"])
        ds = SingleElectricityDataset(
            csv_path,
            entry["id"],
            seq_len,
            label_len,
            pred_len,
            teacher_dim=teacher_dim  # 👈 pass teacher_dim here
        )
        datasets.append(ds)

    concat_ds = ConcatDataset(datasets)
    sampler = BalancedSampler(datasets)
    loader = DataLoader(concat_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    print(f"✅ Built multi-dataset loader with {len(datasets)} datasets "
          f"and {len(concat_ds)} total samples (teacher_dim={teacher_dim}).")
    return loader

