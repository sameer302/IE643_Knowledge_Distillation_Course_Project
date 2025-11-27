import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from exp.exp_main import Exp_Main
#  from run_finetune import ProjectedTeacher
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from models import Autoformer

# ✅ Override ProjectedTeacher here — independent of run_finetune.py
class ProjectedTeacher(nn.Module):
    def __init__(self, ckpt_path, input_dim, teacher_dim, configs):
        super(ProjectedTeacher, self).__init__()
        self.teacher_dim = teacher_dim
        self.proj_in = nn.Linear(input_dim, teacher_dim)
        self.proj_out = nn.Linear(teacher_dim, input_dim)

        # load the pretrained Autoformer
        self.teacher = Autoformer.Model(configs).float()
        # 🧩 For your case: checkpoint is a raw state_dict
        print(f"📂 Loading teacher checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.teacher.load_state_dict(ckpt, strict=False)
        print(f"✅ Loaded teacher. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        self.teacher.eval()

        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 🧠 No check before projection — inputs can have any dimension
        x_enc_proj = self.proj_in(x_enc)
        x_dec_proj = self.proj_in(x_dec)

        # ✅ Optional sanity checks
        if x_enc_proj.shape[-1] != self.teacher_dim:
            print(f"[Warning] x_enc_proj dim {x_enc_proj.shape[-1]} ≠ {self.teacher_dim}")
        if x_dec_proj.shape[-1] != self.teacher_dim:
            print(f"[Warning] x_dec_proj dim {x_dec_proj.shape[-1]} ≠ {self.teacher_dim}")

        # forward through teacher model
        y = self.teacher(x_enc_proj, x_mark_enc, x_dec_proj, x_mark_dec)
        if isinstance(y, tuple):
            y = y[0]
        return self.proj_out(y)


def plot_predictions(root_path, data_path, ckpt_path, teacher_dim, seq_len=128, label_len=64, pred_len=336, device='cuda:0'):
    # === Load test data ===
    print(f"\n📂 Evaluating dataset: {data_path}")
    args = argparse.Namespace(
        model="Autoformer", data="custom", features="M", target="OT", freq="h",
        seq_len=seq_len, label_len=label_len, pred_len=pred_len,
        enc_in=teacher_dim, dec_in=teacher_dim, c_out=teacher_dim,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
        moving_avg=25, factor=3, dropout=0.05, embed="timeF", activation="gelu",
        output_attention=False, distil=True, learning_rate=1e-4,
        use_gpu=True, gpu=0, train_epochs=1, batch_size=8, patience=3,
        root_path=root_path, data_path=data_path, checkpoints="./checkpoints",
        use_multi_gpu=False, devices="0", num_workers=4, use_amp=False,
    )

    exp = Exp_Main(args)
    _, test_loader = exp._get_data(flag='test')

    # 🧠 Detect dataset's actual feature count
    df = pd.read_csv(os.path.join(root_path, data_path))
    input_dim = len([c for c in df.columns if c != "date"])
    print(f"📊 {data_path}: detected {input_dim} feature columns")

    # 🧩 Build ProjectedTeacher with dynamic input projection
    model = ProjectedTeacher(
        ckpt_path,
        input_dim=input_dim,     # dataset-specific input dimension
        teacher_dim=teacher_dim, # global latent dimension (42)
        configs=args
    )
    model.eval()
    model.to(device)
    exp.model = model

    preds, trues = [], []
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs, batch_y_true = exp._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y_true.cpu().numpy())

            if i > 20:  # limit for quick plotting
                break

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # === Compute MSE and MAE ===
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    print(f"📊 {os.path.basename(data_path)} → MSE: {mse:.6f}, MAE: {mae:.6f}")

    # === Plot results ===
    plt.figure(figsize=(10, 5))
    plt.plot(trues[0, :, 0], label="Ground Truth", linewidth=2)
    plt.plot(preds[0, :, 0], label="Prediction", linewidth=2)
    plt.title(f"{os.path.basename(data_path)} | MSE={mse:.4f}, MAE={mae:.4f}")
    plt.xlabel("Time Step")
    plt.ylabel("Electricity (MW)")
    plt.legend()
    plt.grid(True)

    os.makedirs("./plots", exist_ok=True)
    save_path = f"./plots/{os.path.splitext(os.path.basename(data_path))[0]}_pred_vs_true.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot → {save_path}")

    return mse, mae


if __name__ == "__main__":
    root = "./dataset/electricity/"
    ckpt = "/home/sameer/Desktop/AutumnSem25/Courses/IE643/KD_Final/checkpoints/global_finetune/dataset_manifest/global_finetune/checkpoint.pth"
    teacher_dim = 42

    datasets = [
        # "electricity_areawise.csv",
        "electricity_jerico.csv",
        "electricity_countrywise.csv",
        "electricity_household.csv"
    ]

    results = []
    for ds in datasets:
        mse, mae = plot_predictions(root, ds, ckpt, teacher_dim)
        results.append({"Dataset": ds, "MSE": mse, "MAE": mae})

    # Save all results to a summary CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("./plots/results_summary.csv", index=False)
    print("\n📁 All results saved to ./plots/results_summary.csv")