# utils/processing.py
import numpy as np
import pandas as pd
import torch

def zscore_normalize(df: pd.DataFrame):
    """Normalize each column using dataset's own mean/std."""
    mean = df.mean()
    std = df.std().replace(0, 1e-6)  # avoid divide-by-zero
    norm_df = (df - mean) / std
    return norm_df, mean, std


def denormalize_array(arr: np.ndarray, mean: pd.Series, std: pd.Series, col_name: str):
    """Denormalize one feature column for visualization."""
    return arr * std[col_name] + mean[col_name]


def create_sliding_windows(df: pd.DataFrame, seq_len: int, pred_len: int, step: int = 1):
    """Split normalized dataset into overlapping (X, Y) pairs for forecasting."""
    data = df.values
    T, D = data.shape
    samples, targets = [], []
    for start in range(0, T - (seq_len + pred_len) + 1, step):
        samples.append(data[start:start + seq_len])
        targets.append(data[start + seq_len:start + seq_len + pred_len])
    if len(samples) == 0:
        return np.zeros((0, seq_len, D)), np.zeros((0, pred_len, D))
    return np.stack(samples), np.stack(targets)


def detect_model_params(ckpt_path):
    """
    Try to auto-detect seq_len and pred_len from the checkpoint if stored there.
    Returns (seq_len, pred_len) with defaults if not found.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # case 1: if checkpoint saved args dict
        if isinstance(ckpt, dict):
            # common keys used in Autoformer configs
            for k in ["args", "config", "hparams"]:
                if k in ckpt:
                    cfg = ckpt[k]
                    seq_len = int(cfg.get("seq_len", 96))
                    pred_len = int(cfg.get("pred_len", 96))
                    return seq_len, pred_len
        # fallback to common defaults
        return 96, 96
    except Exception as e:
        print(f"[detect_model_params] Warning: {e}")
        return 96, 96
