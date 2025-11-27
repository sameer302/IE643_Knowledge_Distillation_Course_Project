import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
from typing import List


# ============================================================
# 1️⃣ Base dataset class (with padding to max feature dim)
# ============================================================
class SingleElectricityDataset(Dataset):
    def __init__(self, csv_path, dataset_id, seq_len, label_len, pred_len, pad_to_dim):
        """
        Each dataset is padded (not truncated) to pad_to_dim.
        Also stores its original feature count (orig_dim) for use in the model.
        """
        self.dataset_id = dataset_id
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.padded_to = pad_to_dim  # maximum width across all datasets

        # --- Load data ---
        df = pd.read_csv(csv_path)
        if "date" in df.columns:
            df = df.drop(columns=["date"])
        data = df.values.astype(np.float32)

        # --- Record original dimension ---
        n_feat = data.shape[1]
        self.orig_dim = n_feat

        # --- Pad up to pad_to_dim ---
        if n_feat < pad_to_dim:
            pad = np.zeros((data.shape[0], pad_to_dim - n_feat), dtype=np.float32)
            data = np.concatenate([data, pad], axis=1)
        elif n_feat > pad_to_dim:
            # Should never happen, since pad_to_dim is computed as global max
            raise ValueError(
                f"Dataset {os.path.basename(csv_path)} has {n_feat} features, "
                f"which exceeds pad_to_dim={pad_to_dim}"
            )

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

        # Dummy time features (zero placeholders)
        x_mark = np.zeros((self.seq_len, 4), dtype=np.float32)
        y_mark = np.zeros((self.label_len + self.pred_len, 4), dtype=np.float32)

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(x_mark, dtype=torch.float32),
            torch.tensor(y_mark, dtype=torch.float32),
            torch.tensor(self.dataset_id, dtype=torch.long)
        )


# ============================================================
# 2️⃣ Balanced sampler (same as before)
# ============================================================
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


# ============================================================
# 3️⃣ Loader builder (computes max feature dim across datasets)
# ============================================================
def build_multi_loader(manifest_path, root_path, seq_len, label_len, pred_len, batch_size):
    """
    Builds a multi-dataset loader where all datasets are padded to the same max feature dim.
    Returns:
        loader        : DataLoader ready for training
        dataset_dims  : list of original feature counts (in same order as manifest)
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    dataset_entries = manifest["datasets"]

    # --- Detect original feature dims for all datasets ---
    orig_dims = []
    for entry in dataset_entries:
        csv_path = os.path.join(root_path, entry["name"])
        df = pd.read_csv(csv_path, nrows=1)
        if "date" in df.columns:
            n_feat = len(df.columns) - 1
        else:
            n_feat = len(df.columns)
        orig_dims.append(n_feat)

    max_dim = max(orig_dims)
    print(f"🔍 Original feature dims per dataset: {orig_dims}")
    print(f"📦 Padding all datasets to max_dim={max_dim}")

    # --- Build dataset objects ---
    datasets = []
    for entry in dataset_entries:
        csv_path = os.path.join(root_path, entry["name"])
        ds = SingleElectricityDataset(
            csv_path,
            entry["id"],
            seq_len,
            label_len,
            pred_len,
            pad_to_dim=max_dim
        )
        datasets.append(ds)

    # --- Combine ---
    concat_ds = ConcatDataset(datasets)
    sampler = BalancedSampler(datasets)
    loader = DataLoader(concat_ds, batch_size=batch_size, sampler=sampler, drop_last=True)

    print(f"✅ Multi-dataset loader built with {len(datasets)} datasets | "
          f"Total samples: {len(concat_ds)} | Max dim: {max_dim}")

    return loader, orig_dims
