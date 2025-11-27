import os
import torch
import torch.nn as nn
import pandas as pd
import argparse
from exp.exp_main import Exp_Main
from models import Autoformer

# ========= PROJECTED TEACHER WRAPPER =========
class ProjectedTeacher(nn.Module):
    """Wraps pretrained Autoformer to adapt new feature dims via linear projections."""
    def __init__(self, teacher_ckpt, input_dim, teacher_dim, configs):
        super().__init__()
        self.input_dim = input_dim
        self.teacher_dim = teacher_dim

        # Linear adapters for dimension alignment
        self.proj_in = nn.Linear(input_dim, teacher_dim)
        self.proj_out = nn.Linear(teacher_dim, input_dim)

        # Build teacher model
        tcfg = configs
        tcfg.enc_in = tcfg.dec_in = tcfg.c_out = teacher_dim
        self.teacher = Autoformer.Model(tcfg)

        # Load pretrained checkpoint
        print(f"📂 Loading teacher checkpoint from {teacher_ckpt}")
        ckpt = torch.load(teacher_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        res = self.teacher.load_state_dict(state, strict=False)
        print(f"✅ Loaded teacher. Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")

    # inside run_finetune.py (replace existing forward)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Defensive forward: assert expected feature dims before projecting.
        Raises a clear error if shapes are unexpected and prints helpful diagnostics.
        """
        # Expected feature dim (teacher latent dim)
        T = self.teacher_dim

        # Basic checks (shapes should be [B, seq, feat])
        def check_tensor(name, t, expect_feat=None):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"[ProjectedTeacher] {name} is not a Tensor (got {type(t)})")
            if t.dim() != 3:
                raise ValueError(f"[ProjectedTeacher] {name} must be 3D [B,seq,feat], got shape {tuple(t.shape)}")
            if expect_feat is not None and t.shape[-1] != expect_feat:
                raise ValueError(f"[ProjectedTeacher] {name} expected feat_dim={expect_feat}, but got {t.shape[-1]} (shape {tuple(t.shape)})")

        # Check encoder inputs: expect teacher_dim features
        check_tensor("x_enc", x_enc, expect_feat=T)
        # x_mark_enc should be time features (commonly 4)
        check_tensor("x_mark_enc", x_mark_enc, expect_feat=x_mark_enc.shape[-1])
        # Check decoder input: should be label_len+pred_len x teacher_dim
        check_tensor("x_dec", x_dec, expect_feat=T)
        check_tensor("x_mark_dec", x_mark_dec, expect_feat=x_mark_dec.shape[-1])

        # Everything looks good — now project
        x_enc_proj = self.proj_in(x_enc)
        x_dec_proj = self.proj_in(x_dec)

        y = self.teacher(x_enc_proj, x_mark_enc, x_dec_proj, x_mark_dec)
        if isinstance(y, tuple):
            y = y[0]
        out = self.proj_out(y)
        return out



# ========= ARGUMENT PARSER =========
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="CSV file path for training dataset")
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
    p.add_argument("--checkpoints", type=str, default="./checkpoints/global_finetune/")
    p.add_argument("--max_train_steps", type=int, default=None,help="Optional limit on number of training iterations (for quick debugging)")
    return p.parse_args()


# ========= MAIN =========
def main():
    args = parse_args()

    # 1️⃣ detect feature count
    # --- Detect if data_path is a manifest (JSON) or single CSV ---
    data_path_full = os.path.join(args.root_path, args.data_path)

    if args.data_path.endswith(".json"):
        # Manifest mode: load one CSV temporarily to detect input dim
        import json
        with open(data_path_full, "r") as f:
            manifest = json.load(f)
        first_csv = manifest["datasets"][0]["name"]
        df = pd.read_csv(os.path.join(args.root_path, first_csv))
        feature_cols = [c for c in df.columns if c != "date"]
        input_dim = len(feature_cols)
        print(f"🔍 Detected {input_dim} feature columns from first dataset ({first_csv}) [Manifest mode]")
    else:
        df = pd.read_csv(data_path_full)
        feature_cols = [c for c in df.columns if c != "date"]
        input_dim = len(feature_cols)
        print(f"🔍 Detected {input_dim} feature columns in {args.data_path}")

    # 2️⃣ config
    cfg = argparse.Namespace(
        model="Autoformer", data="custom", features="M", target="OT", freq="h",
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
        moving_avg=25, factor=3, dropout=0.05, embed="timeF", activation="gelu",
        output_attention=False, distil=True, learning_rate=args.adapter_lr,
        use_gpu=args.use_gpu, gpu=args.gpu, train_epochs=args.train_epochs,
        batch_size=args.batch_size, patience=5, loss="mse", lradj="type1",
        root_path=args.root_path, data_path=args.data_path,
        checkpoints=args.checkpoints, use_multi_gpu=False, devices="0", num_workers=4, use_amp=False,
        max_train_steps=args.max_train_steps,
    )

    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    cfg.checkpoints = os.path.join(args.checkpoints, dataset_name)
    os.makedirs(cfg.checkpoints, exist_ok=True)
    print(f"📁 Saving checkpoints to: {cfg.checkpoints}")

    # 3️⃣ Build model
    model = ProjectedTeacher(args.teacher_ckpt, input_dim, args.teacher_dim, cfg)
    if args.freeze_teacher:
        for p in model.teacher.parameters():
            p.requires_grad = False
        print("🧊 Backbone frozen — training adapters only.")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = model.to(device)
    print(f"✅ Model on {device}")

    # 4️⃣ Optimizer
    groups = [
        {"params": model.proj_in.parameters(), "lr": args.adapter_lr},
        {"params": model.proj_out.parameters(), "lr": args.adapter_lr},
    ]
    if not args.freeze_teacher:
        groups.append({"params": model.teacher.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.Adam(groups)
    from data_provider.multi_loader import build_multi_loader # import multiloader here

    # If data_path points to the manifest, build the combined loader
    if args.data_path.endswith(".json"):
        train_loader = build_multi_loader(
            manifest_path=os.path.join(args.root_path, args.data_path),
            root_path=args.root_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size
        )
        cfg.data_loader = train_loader

    # 5️⃣ Train via Exp_Main
    exp = Exp_Main(cfg)
    exp.model = model
    exp.train(setting="global_finetune")

    print(f"\n✅ Finetuning complete. Checkpoints in {cfg.checkpoints}")


if __name__ == "__main__":
    main()
