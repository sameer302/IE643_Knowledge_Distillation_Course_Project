#!/usr/bin/env python3
# cache_teacher_outputs.py
import os
import argparse
import torch
import numpy as np
import json
from exp.exp_main import Exp_Main
import argparse as _argparse

def parse_args():
    p = _argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="CSV file inside root_path")
    p.add_argument("--root_path", default="./dataset/electricity/")
    p.add_argument("--teacher_ckpt", required=True)
    p.add_argument("--checkpoints", default="./checkpoints/auto_finetune/")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--label_len", type=int, default=48)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--use_gpu", type=int, default=1)
    return p.parse_args()

def build_cfg(args, input_dim):
    # minimal config compatible with Exp_Main/data loader
    cfg = _argparse.Namespace(
        model="Autoformer", data="custom", features="M", target="OT", freq="h",
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
        moving_avg=25, factor=3, dropout=0.05, embed="timeF", activation="gelu",
        output_attention=False, distil=True, learning_rate=1e-3,
        use_gpu=bool(args.use_gpu), gpu=args.gpu, train_epochs=1,
        batch_size=32, patience=3, loss="mse", lradj="type1",
        root_path=args.root_path, data_path=args.data_path,
        checkpoints=args.checkpoints, use_multi_gpu=False, devices=str(args.gpu), num_workers=4, use_amp=False,
    )
    return cfg

def load_teacher_into_exp(exp, ckpt_path, device):
    print(f"Loading teacher checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    # try to load into exp.model
    missing, unexpected = exp.model.load_state_dict(state, strict=False)
    print(f"Teacher loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    exp.model.to(device)
    exp.model.eval()

def cache_split(exp, split_flag, out_dir, device, cfg):
    os.makedirs(out_dir, exist_ok=True)
    # get data loader for the split
    data_set, data_loader = exp._get_data(flag=split_flag)
    print(f"Caching split={split_flag} | samples (approx) loader len: {len(data_loader)}")
    teacher_preds = []
    targets = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            # move to device
            # Ensure all tensors are float32
            batch_x = batch_x.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()


            # prepare decoder input: zeros for prediction length
            # batch_y shape usually: (B, L, C) where L >= pred_len
            dec_inp = torch.zeros_like(batch_y[:, -cfg.pred_len:, :]).float().to(device)

            out = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # some models return (pred, attns)
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            out = out.detach().cpu().numpy()
            teacher_preds.append(out)
            targets.append(batch_y.detach().cpu().numpy())

    teacher_preds = np.concatenate(teacher_preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    np.save(os.path.join(out_dir, "y_teacher.npy"), teacher_preds)
    np.save(os.path.join(out_dir, "y_true.npy"), targets)
    meta = {
        "y_teacher_shape": list(teacher_preds.shape),
        "y_true_shape": list(targets.shape),
        "pred_len": cfg.pred_len
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {out_dir}/y_teacher.npy ({teacher_preds.shape}), y_true.npy ({targets.shape})")

def main():
    args = parse_args()

    # quick detect input_dim by reading dataset header via Exp_Main's data loader.
    # To do that, build a small cfg with enc_in=1 and let data loader discover columns,
    # but simpler: instantiate cfg after quick pass: we'll rely on Exp_Main data loader requiring cfg.enc_in to match dataset
    # so we need to detect number of numeric features
    import pandas as pd
    df = pd.read_csv(os.path.join(args.root_path, args.data_path))
    feature_cols = [c for c in df.columns if c != "date"]
    input_dim = len(feature_cols)
    print(f"Detected {input_dim} feature columns in {args.data_path}")

    cfg = build_cfg(args, input_dim)

    # set dataset-specific checkpoint dir under provided checkpoints (optional)
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    cache_root = os.path.join("cached_predictions_new_multiloader", dataset_name)
    os.makedirs(cache_root, exist_ok=True)

    # build exp and load teacher
    exp = Exp_Main(cfg)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and cfg.use_gpu else "cpu")
    load_teacher_into_exp(exp, args.teacher_ckpt, device)

    # cache for train, val, test splits
    for split in ["train", "val", "test"]:
        out_dir = os.path.join(cache_root, split)
        cache_split(exp, split, out_dir, device, cfg)

    print("✅ All done. Cached predictions saved to:", cache_root)

if __name__ == "__main__":
    main()
