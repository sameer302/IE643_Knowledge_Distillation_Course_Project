import os, argparse, json, torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models import TSMixer  # ← you can later swap this with DLinear, etc.

# ---------- KD Dataset Loader ----------
def load_kd_dataset(cache_root, split="train"):
    # Load arrays
    y_teacher = np.load(os.path.join(cache_root, split, "y_teacher.npy"))
    y_true = np.load(os.path.join(cache_root, split, "y_true.npy"))

    # The input X should correspond to the last 'seq_len' portion of y_true,
    # since y_true includes both label_len + pred_len in Autoformer caching
    seq_len = y_teacher.shape[1]  # teacher was trained with pred_len=96
    if y_true.shape[1] > seq_len:
        x = y_true[:, -seq_len:, :]   # take last 96 timesteps as input
    else:
        x = y_true                    # already matches

    # Convert to float32 tensors
    x = torch.tensor(x, dtype=torch.float32)
    y_teacher = torch.tensor(y_teacher, dtype=torch.float32)
    y_true = torch.tensor(y_true[:, -seq_len:, :], dtype=torch.float32)  # match shape for loss
    return x, y_teacher, y_true


# ---------- KD Training Loop ----------
def train_kd(student, dataloader, optimizer, kd_criterion, gt_criterion, alpha, device):
    student.train()
    total_loss = 0.0
    for xb, y_teacher, y_true in dataloader:
        xb, y_teacher, y_true = xb.to(device), y_teacher.to(device), y_true.to(device)
        optimizer.zero_grad()
        y_pred = student(xb)
        loss = alpha * kd_criterion(y_pred, y_teacher) + (1 - alpha) * gt_criterion(y_pred, y_true) # loss function
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", required=True, help="Path inside cached_predictions/<dataset>")
    p.add_argument("--student_model", default="TSMixer")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_dir", default="./checkpoints/student_kd/")
    p.add_argument("--resume", default=None, help="Path to previous student checkpoint")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # load cached data
    X, Y_teacher, Y_true = load_kd_dataset(args.cache_root, "train")


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



    dataset = TensorDataset(X, Y_teacher, Y_true)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = X.shape[-1]
    print(f"Loaded dataset from {args.cache_root} | X: {X.shape}, Teacher: {Y_teacher.shape}")

    # build student
    student = TSMixer(input_dim=input_dim, pred_len=Y_teacher.shape[1], latent_dim=64 )  # adjust constructor if needed
    student = student.to(device)

    print("\n🔍 Parameter breakdown (initial):")
    for name, p in student.named_parameters():
        print(f"{name:<50} {p.numel():>10}")
    print(f"Total student parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # 🔁 Resume if checkpoint provided
    if args.resume and os.path.exists(args.resume):
        print(f"Loading previous student weights from {args.resume}")
        if args.resume and os.path.exists(args.resume):
            print(f"Loading previous student weights from {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu")
            state = ckpt if isinstance(ckpt, dict) else ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt

            model_state = student.state_dict()
            matched, mismatched = [], []

            for k, v in state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    matched.append(k)
                else:
                    mismatched.append(k)

            student.load_state_dict(model_state, strict=False)
            print(f"✅ Loaded {len(matched)} layers; skipped {len(mismatched)} mismatched ones:")
            for m in mismatched:
                print(f"   ⚠️ skipped → {m}")


    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    kd_criterion = nn.MSELoss()
    gt_criterion = nn.MSELoss()

    print(f"🚀 Starting KD training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_kd(student, loader, optimizer, kd_criterion, gt_criterion, args.alpha, device)
        print(f"Epoch {epoch+1}/{args.epochs} | KD Loss: {loss:.6f}")

    save_path = os.path.join(args.save_dir, os.path.basename(args.cache_root.rstrip('/')) + "_student.pth")
    torch.save(student.state_dict(), save_path)
    print(f"✅ KD student model saved to {save_path}")

if __name__ == "__main__":
    main()
