#!/usr/bin/env python3
# run_kd_projected.py
import os, argparse, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models import TSMixer  # student
from exp.exp_main import Exp_Main

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
    xs, y_teachers, y_trues, ds_labels = [], [], [], []
    max_dim = teacher_dim
    print(f"🔧 Target latent dimension (teacher_dim) = {max_dim}")

    for i, dname in enumerate(dataset_names):
        dpath = os.path.join(base_dir, dname)
        x, y_teacher, y_true = load_kd_dataset(dpath, split)

        # Projection logic remains the same …
        in_dim = x.shape[-1]
        if in_dim != max_dim:
            proj = nn.Linear(in_dim, max_dim)
            with torch.no_grad():
                x = proj(x).detach()
                y_teacher = proj(y_teacher).detach()
                y_true = proj(y_true).detach()
            print(f"Projected {dname} ({in_dim} → {max_dim}) [detached]")
        else:
            print(f"{dname} already in teacher space ({in_dim})")

        xs.append(x)
        y_teachers.append(y_teacher)
        y_trues.append(y_true)

        # dataset label tensor
        ds_labels.append(torch.full((x.shape[0],), i, dtype=torch.long))

    X = torch.cat(xs, dim=0)
    Y_teacher = torch.cat(y_teachers, dim=0)
    Y_true = torch.cat(y_trues, dim=0)
    DS_labels = torch.cat(ds_labels, dim=0)
    return X, Y_teacher, Y_true, DS_labels

# ---------- KD Training ----------
def train_kd(student, dataloader, optimizer, kd_criterion, gt_criterion, alpha_dict, device):
    student.train()
    total_loss = 0.0
    for xb, y_teacher, y_true, ds_labels in dataloader:
        xb, y_teacher, y_true, ds_labels = xb.to(device), y_teacher.to(device), y_true.to(device), ds_labels.to(device)
        optimizer.zero_grad()
        y_pred = student(xb)

        # compute loss per sample
        batch_loss = 0.0
        for i in range(xb.size(0)):
            dataset_id = int(ds_labels[i].item())
            alpha = alpha_dict[dataset_id]
            loss_i = alpha * kd_criterion(y_pred[i], y_teacher[i]) + (1 - alpha) * gt_criterion(y_pred[i], y_true[i])
            batch_loss += loss_i
        batch_loss = batch_loss / xb.size(0)

        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    return total_loss / len(dataloader)

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--cache_root", default="./cached_predictions")
    p.add_argument("--teacher_dim", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_dir", default="./checkpoints/student_kd/")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"🚀 Loading cached data (latent_dim={args.teacher_dim})...")
    X, Y_teacher, Y_true, DS_labels = load_multi_kd_datasets(args.cache_root, args.datasets, "train", args.teacher_dim)
    import gc
    import inspect
    import torch.nn as nn

    print("\n🔍 Searching for stray Linear layers in memory...")
    for obj in gc.get_objects():
        try:
            if isinstance(obj, nn.Linear):
                source = inspect.getsourcefile(type(obj))
                print(f"Found Linear layer: {obj} | from {source}")
                print(f"  → weight shape: {tuple(obj.weight.shape)} | params: {obj.weight.numel() + obj.bias.numel()}")
        except:
            pass

    dataset = TensorDataset(X, Y_teacher, Y_true, DS_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = X.shape[-1]
    pred_len = Y_teacher.shape[1]

    # build student
    student = TSMixer(input_dim=input_dim, pred_len=pred_len, latent_dim=64)
    student = student.to(device)

    print("\n🔍 Parameter breakdown (initial)")
    for name, param in student.named_parameters():
        print(f"{name:<50} {param.numel():>10}")
    print(f"Total student parameters: {sum(p.numel() for p in student.parameters()):,}")


    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    kd_criterion = nn.MSELoss()
    gt_criterion = nn.MSELoss()

    # 🔸 Define per-dataset alpha mapping (index = dataset order)
# Example: ["areawise", "countrywise", "household", "jerico"]
    alpha_dict = {
        0: 0.3,  # areawise
        1: 0.9,  # countrywise
        2: 0.4,  # household
        3: 0.1,  # jerico
    }


    print(f"🚀 Starting KD training for {args.epochs} epochs on {len(X)} samples...")
    for epoch in range(args.epochs):
        loss = train_kd(student, loader, optimizer, kd_criterion, gt_criterion, alpha_dict, device)
        print(f"Epoch {epoch+1}/{args.epochs} | KD Loss: {loss:.6f}")

    save_name = f"{'_'.join(args.datasets)}_student_projected_alpha_new2.pth"
    save_path = os.path.join(args.save_dir, save_name)
    torch.save(student.state_dict(), save_path)
    print(f"✅ Student (projected KD) saved to {save_path}")

if __name__ == "__main__":
    main()
