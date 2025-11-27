import os
import torch
import torch.nn as nn
import pandas as pd
import argparse
from exp.exp_main import Exp_Main
from models import Autoformer


class ProjectedTeacher(nn.Module):
    """Wraps a pretrained Autoformer teacher with adapters for any feature count."""
    def __init__(self, teacher_ckpt, input_dim, teacher_dim, configs):
        super().__init__()
        self.input_dim = input_dim
        self.teacher_dim = teacher_dim

        # adapters
        self.proj_in = nn.Linear(input_dim, teacher_dim)
        self.proj_out = nn.Linear(teacher_dim, input_dim)

        # build teacher backbone
        tcfg = configs
        tcfg.enc_in = tcfg.dec_in = tcfg.c_out = teacher_dim
        self.teacher = Autoformer.Model(tcfg)

        # load pretrained teacher
        print(f"Loading teacher from {teacher_ckpt}")
        ckpt = torch.load(teacher_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        missing, unexpected = self.teacher.load_state_dict(state, strict=False)
        print(f"Loaded teacher. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.proj_in(x_enc)
        x_dec = self.proj_in(x_dec)
        y = self.teacher(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(y, tuple):
            y = y[0]
        return self.proj_out(y)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--root_path", default="./dataset/electricity/")
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--teacher_dim", type=int, default=42)
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--train_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--adapter_lr", type=float, default=1e-3) # learning rate for adapters
    p.add_argument("--backbone_lr", type=float, default=1e-5) # learning rate for backbone if unfrozen
    p.add_argument("--freeze_teacher", type=int, default=1)
    p.add_argument("--use_gpu", type=bool, default=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--checkpoints", type=str, default="./checkpoints/auto_finetune/")
    return p.parse_args()


def main():
    args = parse_args()

    # 1️⃣ detect feature count
    df = pd.read_csv(os.path.join(args.root_path, args.data_path))
    feature_cols = [c for c in df.columns if c != "date"]
    input_dim = len(feature_cols)
    print(f"Detected {input_dim} feature columns in {args.data_path}")

    # 2️⃣ config for backbone
    cfg = argparse.Namespace(
        model="Autoformer", data="custom", features="M", target="OT", freq="h",
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
        moving_avg=25, factor=3, dropout=0.05, embed="timeF", activation="gelu",
        output_attention=False, distil=True, learning_rate=args.adapter_lr,
        use_gpu=args.use_gpu, gpu=args.gpu, train_epochs=args.train_epochs,
        batch_size=args.batch_size, patience=3, loss="mse", lradj="type1",
        root_path=args.root_path, data_path=args.data_path,
        checkpoints=args.checkpoints, use_multi_gpu=False, devices="0", num_workers=4,use_amp=False,
    )

        # 🧩 Append dataset name to checkpoint path
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    cfg.checkpoints = os.path.join(args.checkpoints, dataset_name)

    # Make sure the folder exists
    os.makedirs(cfg.checkpoints, exist_ok=True)
    print(f"📁 Saving checkpoints to: {cfg.checkpoints}")


    # 3️⃣ build model
    model = ProjectedTeacher(args.teacher_ckpt, input_dim, args.teacher_dim, cfg)

    if args.freeze_teacher: # freezing backbone layers for adapter training
        for p in model.teacher.parameters():
            p.requires_grad = False
        print("Backbone frozen (training adapters only).")

    # 🚀 move model to GPU safely
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model = model.to(device)

    # double-check: move every param to GPU (handles adapters & teacher)
    for name, param in model.named_parameters():
        if param.device.type != device.type:
            print(f"⚠️ Moving {name} from {param.device} → {device}")
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad = param._grad.to(device)

    print(f"✅ Model is now on {device}")

    # 4️⃣ optimizer
    groups = [
        {"params": model.proj_in.parameters(), "lr": args.adapter_lr},
        {"params": model.proj_out.parameters(), "lr": args.adapter_lr},
    ]
    if not args.freeze_teacher:
        groups.append({"params": model.teacher.parameters(), "lr": args.backbone_lr})
    optimizer = torch.optim.Adam(groups)

    # make sure optimizer states are on GPU (some PyTorch versions need this)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


    # 5️⃣ train via existing Exp_Main
    exp = Exp_Main(cfg)
    exp.model = model
    exp.train(setting="finetune_auto")

    print(f"\n✅ Finetuning complete. Checkpoints in {args.checkpoints}")


if __name__ == "__main__":
    main()
