import os, torch, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from models import Autoformer, TSMixer
import torch.nn as nn

# ---------- Metrics ----------
def mse(a,b): return ((a-b)**2).mean().item()
def mae(a,b): return (a-b).abs().mean().item()

# ---------- Data loader ----------
def load_cached_dataset(cache_root, split="test"):
    y_true = np.load(os.path.join(cache_root, split, "y_true.npy"))
    y_teacher = np.load(os.path.join(cache_root, split, "y_teacher.npy"))
    X = torch.tensor(y_true[:, -y_teacher.shape[1]:, :], dtype=torch.float32)
    Y_true = torch.tensor(y_true[:, -y_teacher.shape[1]:, :], dtype=torch.float32)
    return X, Y_true, torch.tensor(y_teacher, dtype=torch.float32)

# ---------- Evaluation ----------
def evaluate(student, teacher_ckpt, cache_root, device="cpu", sample_plot=True):
    # load data
    X, Y_true, Y_teacher = load_cached_dataset(cache_root)
    loader = DataLoader(TensorDataset(X, Y_true, Y_teacher), batch_size=32, shuffle=False)

    student.eval()
    mse_student, mae_student = 0, 0
    mse_teacher, mae_teacher = 0, 0

    with torch.no_grad():
        for xb, yb, yteach in loader:
            xb, yb, yteach = xb.to(device), yb.to(device), yteach.to(device)
            y_pred = student(xb)
            mse_student += mse(y_pred, yb)
            mae_student += mae(y_pred, yb)
            mse_teacher += mse(yteach, yb)
            mae_teacher += mae(yteach, yb)

    n_batches = len(loader)
    print("\n📊 Evaluation Results")
    print("---------------------------")
    print(f"Student MSE : {mse_student/n_batches:.6f}")
    print(f"Student MAE : {mae_student/n_batches:.6f}")
    print(f"Teacher MSE : {mse_teacher/n_batches:.6f}")
    print(f"Teacher MAE : {mae_teacher/n_batches:.6f}")

    # ---- Optional visualization ----
    if sample_plot:
        idx = np.random.randint(0, X.shape[0])
        true = Y_true[idx].cpu().numpy()
        with torch.no_grad():
            stu = student(X[idx:idx+1].to(device)).detach().cpu().numpy().squeeze()
        teach = Y_teacher[idx].cpu().numpy()
        plt.figure(figsize=(10,4))
        plt.plot(true[:,0], label="Ground Truth", linewidth=2)
        plt.plot(teach[:,0], label="Teacher", linestyle="--")
        plt.plot(stu[:,0], label="Student", linestyle=":")
        plt.legend(); plt.title(f"Sample #{idx} prediction comparison")
        plt.tight_layout(); plt.show()

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", required=True)
    p.add_argument("--student_ckpt", required=True)
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # load student
    print(f"Loading student from {args.student_ckpt}")

    # --- Auto-detect input dimension from cached dataset ---
    sample_path = os.path.join(args.cache_root, "test", "y_true.npy")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(args.cache_root, "train", "y_true.npy")
    Y_true_sample = np.load(sample_path)
    input_dim = Y_true_sample.shape[-1]
    print(f"Detected input_dim = {input_dim} from cached data")

    # --- Initialize student model dynamically ---
    student = TSMixer(pred_len=96, input_dim=input_dim, latent_dim=64)
    ckpt = torch.load(args.student_ckpt, map_location="cpu")
    # ---- Safe state loading ----
    model_state = student.state_dict()
    matched, mismatched = [], []

    for k, v in ckpt.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            matched.append(k)
        else:
            mismatched.append(k)

    student.load_state_dict(model_state, strict=False)
    student.to(device)

    print(f"✅ Loaded {len(matched)} layers; skipped {len(mismatched)} mismatched ones:")
    for m in mismatched:
        print(f"   ⚠️ skipped → {m}")


    # evaluate
    evaluate(student, args.teacher_ckpt, args.cache_root, device)

    # ===============================================================
    # 🧩 PERFORMANCE PROFILING SECTION
    # ===============================================================
    import time
    import torch
    import os
    import argparse
    def safe_forward(model, name, sample_input, device):
        """Runs a safe forward pass for both Autoformer and Student models."""
        try:
            if "Autoformer" in name:
                B, L, D = sample_input.shape

                seq_len = 96
                label_len = 48
                pred_len = 96

                # Encoder input: first 96 timesteps
                x_enc = sample_input[:, :seq_len, :]

                # Decoder input: last label_len + pred_len timesteps = 144 total
                dec_total = label_len + pred_len
                if L >= dec_total:
                    x_dec = sample_input[:, :dec_total, :]
                else:
                    pad_len = dec_total - L
                    pad = torch.zeros((B, pad_len, D), device=device)
                    x_dec = torch.cat([pad, sample_input], dim=1)

                # Dummy time features
                x_mark_enc = torch.zeros((B, seq_len, 4), device=device)
                x_mark_dec = torch.zeros((B, dec_total, 4), device=device)

                _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            else:
                _ = model(sample_input)
            return True

        except Exception as e:
            print(f"⚠️ Forward failed for {name}: {e}")
            return False



    def profile_model(model, device, sample_input, name="Model"):
        """Profiles model parameters, size, latency, and memory (safe version)."""
        model.eval()
        model = model.to(device)

        # 1️⃣ Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 2️⃣ Model size (in MB)
        size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
        if hasattr(model, "checkpoint_path") and os.path.exists(model.checkpoint_path):
            size_mb = os.path.getsize(model.checkpoint_path) / (1024 * 1024)

        # 3️⃣ Measure inference speed & GPU memory
        n_warmup, n_iters = 3, 10
        avg_time, mem_mb = None, None

        with torch.no_grad():
            success = safe_forward(model, name, sample_input, device)
            if not success:
                print(f"❌ Skipping latency test for {name} due to forward failure.")
            else:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)

                # Warm-up
                for _ in range(n_warmup):
                    safe_forward(model, name, sample_input, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                # Timed runs
                start = time.perf_counter()
                for _ in range(n_iters):
                    safe_forward(model, name, sample_input, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()

                avg_time = (end - start) / n_iters * 1000  # ms/sample
                mem_mb = (
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    if device.type == "cuda"
                    else 0.0
                )

        print(f"\n🧩 {name} Resource Profile")
        print(f"---------------------------")
        print(f"Total Params        : {total_params:,}")
        print(f"Trainable Params    : {trainable_params:,}")
        print(f"Model Size (est.)   : {size_mb:.2f} MB")
        if avg_time is not None:
            print(f"Inference Latency   : {avg_time:.2f} ms/sample")
        if device.type == "cuda" and mem_mb is not None:
            print(f"Peak GPU Memory     : {mem_mb:.2f} MB")


    # ===============================================================
    # 📈 Run Profiling for Teacher & Student
    # ===============================================================
    print("\n📈 Profiling Teacher & Student")

    try:
        X_sample, Y_true, Y_teacher = load_cached_dataset(args.cache_root)
        sample_input = X_sample[:1, :96, :].to(device)  # 🔥 force match teacher seq_len=96


        from models import Autoformer
        teacher = Autoformer.Model(
            argparse.Namespace(
                seq_len=96, label_len=48, pred_len=96,
                enc_in=sample_input.shape[-1], dec_in=sample_input.shape[-1],
                c_out=sample_input.shape[-1], d_model=512, n_heads=8, e_layers=2,
                d_layers=1, d_ff=2048, dropout=0.05, factor=3,
                moving_avg=25, embed="timeF", freq="h", activation="gelu",
                output_attention=False, distil=True
            )
        )
        ckpt_t = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(ckpt_t.get("model", ckpt_t), strict=False)

        profile_model(teacher, device, sample_input, name="Teacher (Autoformer)")
    except Exception as e:
        print(f"⚠️ Teacher profiling failed: {e}")

    try:
        profile_model(student, device, sample_input, name="Student (TSMixer)")
    except Exception as e:
        print(f"⚠️ Student profiling failed: {e}")



