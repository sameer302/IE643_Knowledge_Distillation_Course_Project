import os
import torch
import torch.nn as nn
import pandas as pd
import argparse
from exp.exp_main import Exp_Main
from models import Autoformer
from data_provider.multi_loader_new import build_multi_loader   # 🔹 new import


# ============================================================
# 1️⃣ ProjectedTeacherNew — supports per-dataset projections
# ============================================================
class ProjectedTeacherNew(nn.Module):
    """Wraps a pretrained Autoformer teacher with per-dataset projection adapters."""
    def __init__(self, teacher_ckpt, dataset_dims, teacher_dim, configs):
        """
        dataset_dims: list of original feature dims (in same order as manifest)
        teacher_dim : the latent dimension of the teacher (e.g., 42)
        """
        super().__init__()
        self.dataset_dims = dataset_dims
        self.teacher_dim = teacher_dim
        self.num_datasets = len(dataset_dims)

        # Per-dataset projection layers
        self.proj_in = nn.ModuleList([
            nn.Linear(d, teacher_dim) for d in dataset_dims
        ])
        self.proj_out = nn.ModuleList([
            nn.Linear(teacher_dim, d) for d in dataset_dims
        ])

        # Build teacher backbone
        tcfg = configs
        tcfg.enc_in = tcfg.dec_in = tcfg.c_out = teacher_dim
        self.teacher = Autoformer.Model(tcfg)

        # Load pretrained checkpoint
        print(f"📂 Loading teacher checkpoint from {teacher_ckpt}")
        ckpt = torch.load(teacher_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        res = self.teacher.load_state_dict(state, strict=False)
        print(f"✅ Loaded teacher. Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, dataset_id=None):
        """
        Forward pass that uses per-dataset projections.
        x_enc/x_dec : shape (B, seq_len, padded_dim)
        dataset_id  : (B,) tensor of dataset indices
        """
        if dataset_id is None:
            # fallback for backward compatibility
            pid = 0
        else:
            pid = int(dataset_id[0].item())  # assume same dataset in batch

        orig_d = self.dataset_dims[pid]

        # Crop to true dimension before projection
        x_enc = x_enc[:, :, :orig_d]
        x_dec = x_dec[:, :, :orig_d]

        # Project to teacher space
        x_enc_proj = self.proj_in[pid](x_enc)
        x_dec_proj = self.proj_in[pid](x_dec)

        # Teacher forward
        y = self.teacher(x_enc_proj, x_mark_enc, x_dec_proj, x_mark_dec)
        if isinstance(y, tuple):
            y = y[0]

        # Project back to original space
        out = self.proj_out[pid](y)
        return out


# ============================================================
# 2️⃣ Argument parser (same as original)
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="Path to manifest JSON or CSV file")
    p.add_argument("--root_path", default="./dataset/electricity/")
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--teacher_dim", type=int, default=42)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=336)
    p.add_argument("--train_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--adapter_lr", type=float, default=1e-3)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--freeze_teacher", type=int, default=1)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoints", type=str, default="./checkpoints/global_finetune_new/")
    p.add_argument("--max_train_steps", type=int, default=None,
                   help="Optional limit on total training steps (debug mode)")
    return p.parse_args()


# ============================================================
# 3️⃣ Main training logic
# ============================================================
def main():
    args = parse_args()

    # Detect feature count only for single CSV mode (manifest handled below)
    data_path_full = os.path.join(args.root_path, args.data_path)

    if args.data_path.endswith(".json"):
        print(f"📄 Manifest mode detected: {args.data_path}")
    else:
        df = pd.read_csv(data_path_full)
        feature_cols = [c for c in df.columns if c != "date"]
        input_dim = len(feature_cols)
        print(f"🔍 Single dataset mode detected with {input_dim} features")

    # --- Build base config (same as before) ---
    cfg = argparse.Namespace(
        model="Autoformer", data="custom", features="M", target="OT", freq="h",
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
        moving_avg=25, factor=3, dropout=0.05, embed="timeF", activation="gelu",
        output_attention=False, distil=True, learning_rate=args.adapter_lr,
        use_gpu=args.use_gpu, gpu=args.gpu, train_epochs=args.train_epochs,
        batch_size=args.batch_size, patience=5, loss="mse", lradj="type1",
        root_path=args.root_path, data_path=args.data_path,
        checkpoints=args.checkpoints, use_multi_gpu=False, devices="0",
        num_workers=4, use_amp=False, max_train_steps=args.max_train_steps,
    )

    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    cfg.checkpoints = os.path.join(args.checkpoints, dataset_name)
    os.makedirs(cfg.checkpoints, exist_ok=True)
    print(f"📁 Checkpoints → {cfg.checkpoints}")

    # ============================================================
    # Build loader (NEW) and get original dims
    # ============================================================
    if args.data_path.endswith(".json"):
        train_loader, dataset_dims = build_multi_loader(
            manifest_path=os.path.join(args.root_path, args.data_path),
            root_path=args.root_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size
        )
        cfg.data_loader = train_loader
    else:
        # single dataset fallback for backward compatibility
        dataset_dims = [input_dim]
        cfg.data_loader = None

    # ============================================================
    # Build ProjectedTeacherNew model
    # ============================================================
    model = ProjectedTeacherNew(args.teacher_ckpt, dataset_dims, args.teacher_dim, cfg)

    if args.freeze_teacher:
        for p in model.teacher.parameters():
            p.requires_grad = False
        print("🧊 Teacher backbone frozen (adapters only).")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = model.to(device)
    print(f"✅ Model placed on {device}")

    # ============================================================
    # Optimizer
    # ============================================================
    groups = [
        {"params": model.proj_in.parameters(), "lr": args.adapter_lr},
        {"params": model.proj_out.parameters(), "lr": args.adapter_lr},
    ]
    if not args.freeze_teacher:
        groups.append({"params": model.teacher.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.Adam(groups)

    # ============================================================
    # Train via Exp_Main (unchanged)
    # ============================================================
    exp = Exp_Main(cfg)
    exp.model = model
    exp.train(setting="global_finetune_new")

    print(f"\n✅ Finetuning complete. Checkpoints saved to: {cfg.checkpoints}")


if __name__ == "__main__":
    main()
