#!/usr/bin/env python3
# Experiment C: Learnable α parameter (trainable scalar)
import os, argparse, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models import TSMixer


# ---------- Dataset Loader ----------
def load_kd_dataset(cache_root, split="train"):
    y_teacher = np.load(os.path.join(cache_root, split, "y_teacher.npy"))
    y_true = np.load(os.path.join(cache_root, split, "y_true.npy"))
    seq_len = y_teacher.shape[1]
    if y_true.shape[1] > seq_len:
        x = y_true[:, -seq_len:, :]
    else:
        x = y_true
    x = torch.tensor(x, dtype=torch.float32)
    y_teacher = torch.tensor(y_teacher, dtype=torch.float32)
    y_true = torch.tensor(y_true[:, -seq_len:, :], dtype=torch.float32)
    return x, y_teacher, y_true


def load_multi_kd_datasets(base_dir, dataset_names, split="train", teacher_dim=42):
    xs, y_teachers, y_trues = [], [], []
    max_dim = teacher_dim
    print(f"🔧 Target latent dimension (teacher_dim) = {max_dim}")
    for dname in dataset_names:
        dpath = os.path.join(base_dir, dname)
        x, y_teacher, y_true = load_kd_dataset(dpath, split)
        in_dim = x.shape[-1]
        if in_dim != max_dim:
            proj = nn.Linear(in_dim, max_dim)
            with torch.no_grad():
                x = proj(x).detach(); y_teacher = proj(y_teacher).detach(); y_true = proj(y_true).detach()
            print(f"Projected {dname} ({in_dim} → {max_dim}) [detached]")
        else:
            print(f"{dname} already in teacher space ({in_dim})")
        xs.append(x); y_teachers.append(y_teacher); y_trues.append(y_true)
    X = torch.cat(xs, dim=0); Y_teacher = torch.cat(y_teachers, dim=0); Y_true = torch.cat(y_trues, dim=0)
    return X, Y_teacher, Y_true


# ---------- KD Training ----------
def train_kd(student, dataloader, optimizer, kd_criterion, gt_criterion, learnable_alpha, device):
    student.train()
    total_loss = 0.0
    for xb, y_teacher, y_true in dataloader:
        xb, y_teacher, y_true = xb.to(device), y_teacher.to(device), y_true.to(device)
        optimizer.zero_grad()
        y_pred = student(xb)
        L_teacher = kd_criterion(y_pred, y_teacher)
        L_true = gt_criterion(y_pred, y_true)
        alpha = torch.sigmoid(learnable_alpha)  # constrain α∈(0,1)
        loss = alpha * L_teacher + (1 - alpha) * L_true
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"↳ Avg KD loss = {avg_loss:.6f} | Learned α = {torch.sigmoid(learnable_alpha).item():.3f}")
    return avg_loss


# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--cache_root", default="./cached_predictions")
    p.add_argument("--teacher_dim", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_dir", default="./checkpoints/student_kd/")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"🚀 Loading cached data (latent_dim={args.teacher_dim})...")
    X, Y_teacher, Y_true = load_multi_kd_datasets(args.cache_root, args.datasets, "train", args.teacher_dim)
    X, Y_teacher, Y_true = X.to(device), Y_teacher.to(device), Y_true.to(device)
    print(f"✅ All tensors moved to {device}")

    dataset = TensorDataset(X, Y_teacher, Y_true)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim, pred_len = X.shape[-1], Y_teacher.shape[1]
    student = TSMixer(input_dim=input_dim, pred_len=pred_len, latent_dim=64).to(device)

    learnable_alpha = nn.Parameter(torch.tensor(0.0, device=device))  # start neutral
    optimizer = optim.Adam(list(student.parameters()) + [learnable_alpha], lr=args.lr)
    kd_criterion = nn.MSELoss()
    gt_criterion = nn.MSELoss()

    print(f"🚀 Starting KD training for {args.epochs} epochs on {len(X)} samples...")
    for epoch in range(args.epochs):
        loss = train_kd(student, loader, optimizer, kd_criterion, gt_criterion, learnable_alpha, device)
        print(f"Epoch {epoch+1}/{args.epochs} | KD Loss: {loss:.6f}")

    save_path = os.path.join(args.save_dir, f"{'_'.join(args.datasets)}_student_lr_alpha.pth")
    torch.save({"state_dict": student.state_dict(),
                "alpha": learnable_alpha.detach().cpu().item()}, save_path)
    print(f"✅ Student model (Exp C) saved to {save_path}")


if __name__ == "__main__":
    main()
