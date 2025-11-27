#!/usr/bin/env python3
# plot_predictions_student.py
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from models import TSMixer  # student model

# ---------- Plot Helper ----------
def plot_sample(gt, pred, title, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(gt, label="Ground Truth (proj)", color="black", linewidth=2)
    plt.plot(pred, label="Student Prediction (proj)", color="tab:blue", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True,
                   help="CSV filenames (example: area1.csv area2.csv ...)")
    p.add_argument("--root_path", default="./dataset/electricity/")
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_dir", default="./plots/student_predictions/")
    p.add_argument("--teacher_dim", type=int, default=42,
                   help="Latent dimension used by teacher/student during KD")
    p.add_argument("--batch_eval", type=int, default=128,
                   help="Batch size used for evaluation (reduces memory usage)")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    print(f"Using device: {device}")

    # Load student checkpoint (state_dict or raw dict)
    print(f"📦 Loading student checkpoint from: {args.student_ckpt}")
    ckpt = torch.load(args.student_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # ckpt could be a plain state_dict or model.state_dict()
        state_dict = ckpt

    # -------------------------------------------
    # 🔧 Create per-dataset projection layers
    # -------------------------------------------
    teacher_dim = args.teacher_dim
    proj_layers = {}  # store one projector per dataset
    proj_info = {}    # store input dim for later use

    print(f"\n🔧 Target latent dimension (teacher_dim) = {teacher_dim}")
    for dataset_name in args.datasets:
        data_path_full = os.path.join(args.root_path, dataset_name)
        if not os.path.exists(data_path_full):
            raise FileNotFoundError(f"Dataset file not found: {data_path_full}")
        df = pd.read_csv(data_path_full)
        feature_cols = [c for c in df.columns if c != "date"]
        d_in = len(feature_cols)
        proj_info[dataset_name] = d_in

        if d_in != teacher_dim:
            # create a linear projector (random init). If you used a saved projector during KD, load it here instead.
            print(f"Will project {dataset_name} ({d_in} → {teacher_dim})")
            proj_layers[dataset_name] = nn.Linear(d_in, teacher_dim)
        else:
            print(f"{dataset_name} already in teacher space ({teacher_dim}) — using Identity")
            proj_layers[dataset_name] = nn.Identity()

    # Put projectors in eval mode (and on CPU for now). They will be moved to device later.
    for k in proj_layers:
        proj_layers[k].eval()

    # ----------------------------
    # Iterate datasets and evaluate
    # ----------------------------
    for dataset_name in args.datasets:
        print(f"\n🔍 Evaluating student on dataset: {dataset_name}")
        data_path_full = os.path.join(args.root_path, dataset_name)
        df = pd.read_csv(data_path_full)
        feature_cols = [c for c in df.columns if c != "date"]
        data = df[feature_cols].values.astype(np.float32)
        input_dim = data.shape[1]

        # Optional normalization: matches what you used during training. If you have saved means/stds, load them.
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0) + 1e-8
        data = (data - data_mean) / data_std

        print(f"✅ Loaded {dataset_name} | shape={data.shape}, input_dim={input_dim}")

        seq_len = args.seq_len
        pred_len = args.pred_len

        n_windows = len(data) - seq_len - pred_len + 1
        if n_windows <= 0:
            print(f"⚠️ Not enough timesteps for seq_len={seq_len} and pred_len={pred_len} in {dataset_name}. Skipping.")
            continue

        # Build windows (avoid building huge arrays if dataset is massive; this is still reasonable for moderate sizes)
        X_list = []
        Y_list = []
        for i in range(n_windows):
            X_list.append(data[i : i + seq_len])
            Y_list.append(data[i + seq_len : i + seq_len + pred_len])
        X = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32)  # (N, seq_len, d_in)
        Y = torch.tensor(np.stack(Y_list, axis=0), dtype=torch.float32)  # (N, pred_len, d_in)
        del X_list, Y_list

        # Instantiate student model in the teacher latent space
        # NOTE: student was trained in teacher_dim latent space, so student input_dim should be teacher_dim
        student = TSMixer(input_dim=teacher_dim, pred_len=pred_len, latent_dim=64)
        # safely load state dict (handles missing/mismatched keys)
        model_state = student.state_dict()
        matched, mismatched = [], []
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                matched.append(k)
            else:
                mismatched.append(k)
        student.load_state_dict(model_state, strict=False)
        student.to(device)
        student.eval()
        print(f"✅ Student model loaded. matched keys: {len(matched)}; skipped keys: {len(mismatched)}")

        # Move projector to device
        projector = proj_layers[dataset_name].to(device)
        projector.eval()

        # Batchwise evaluation (memory-safe)
        mse_total, mae_total, n_samples = 0.0, 0.0, 0
        batch_size = args.batch_eval
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                xb = X[start : start + batch_size].to(device)  # (B, seq_len, d_in)
                yb = Y[start : start + batch_size].to(device)  # (B, pred_len, d_in)

                # Project inputs and targets into teacher latent space before feeding student
                xb_proj = projector(xb)      # -> (B, seq_len, teacher_dim)
                yb_proj = projector(yb)      # -> (B, pred_len, teacher_dim)

                # Student inference
                y_pred = student(xb_proj)    # -> (B, pred_len, teacher_dim) expected

                # Move preds/targets to CPU for metric calc
                y_pred_cpu = y_pred.cpu()
                yb_proj_cpu = yb_proj.cpu()

                # batch metrics (mean per-sample)
                mse_batch = torch.mean((y_pred_cpu - yb_proj_cpu) ** 2).item()
                mae_batch = torch.mean(torch.abs(y_pred_cpu - yb_proj_cpu)).item()

                bsz = y_pred_cpu.shape[0]
                mse_total += mse_batch * bsz
                mae_total += mae_batch * bsz
                n_samples += bsz

        mse = mse_total / n_samples
        mae = mae_total / n_samples
        print(f"📊 {dataset_name} | MSE (proj space)={mse:.6f} | MAE (proj space)={mae:.6f}")

        # -------------------------
        # Plot a few random samples
        # -------------------------
        out_dir = os.path.join(args.save_dir, dataset_name.replace(".csv", ""))
        os.makedirs(out_dir, exist_ok=True)

        # draw up to 3 random unique indices
        n_plot = min(3, X.shape[0])
        rng = np.random.default_rng()
        sample_idxs = rng.choice(X.shape[0], size=n_plot, replace=False)

        with torch.no_grad():
            for idx in sample_idxs:
                xb = X[idx : idx + 1].to(device)
                yb = Y[idx].numpy()              # original (pred_len, d_in) on CPU (normalized)
                xb_proj = projector(xb)
                y_pred = student(xb_proj).cpu().numpy().squeeze()  # (pred_len, teacher_dim)
                yb_proj = projector(torch.tensor(yb, dtype=torch.float32).unsqueeze(0).to(device)).cpu().numpy().squeeze()

                # plot first latent feature (index 0) — these are in projected / latent space
                gt = yb_proj[:, 0]
                pred = y_pred[:, 0]
                save_path = os.path.join(out_dir, f"sample_{idx}.png")
                plot_sample(gt, pred, f"{dataset_name} | Sample {idx}", save_path)

        print(f"✅ Plots for {dataset_name} saved to {out_dir}")

    print("\n✅ All datasets processed. Done.")


if __name__ == "__main__":
    main()
