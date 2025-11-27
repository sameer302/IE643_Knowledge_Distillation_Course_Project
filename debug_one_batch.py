# debug_one_batch.py
import os
import json
import torch
import torch.nn as nn
import pandas as pd
from data_provider.multi_loader import build_multi_loader
from run_finetune import ProjectedTeacher, parse_args  # adjust import path if needed

def main():
    # args (adjust paths as needed)
    args = parse_args()
    args.root_path = args.root_path or "./dataset/electricity/"
    manifest = os.path.join(args.root_path, args.data_path)

    args.output_attention = False
    args.distil = True
    args.embed = "timeF"
    args.activation = "gelu"
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 2048
    args.moving_avg = 25
    args.factor = 3
    args.dropout = 0.05
    args.freq = "h"


    print("Building loader...")
    loader = build_multi_loader(manifest, args.root_path, args.seq_len, args.label_len, args.pred_len, batch_size=4, teacher_dim=args.teacher_dim)

    print("Building model wrapper...")
    model = ProjectedTeacher(args.teacher_ckpt, input_dim=args.teacher_dim, teacher_dim=args.teacher_dim, configs=args)  # NOTE: input_dim param here is not used except for initialization; we pass teacher_dim to avoid mismatch
    model.eval()

    # move to CPU/GPU same as training
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    model.to(device)

    # grab one batch
    it = iter(loader)
    batch = next(it)
    print("Raw batch items count:", len(batch))
    for i, item in enumerate(batch):
        try:
            print(f" item[{i}] type={type(item)} shape={(item.shape if hasattr(item,'shape') else 'N/A')} dtype={getattr(item,'dtype',None)}")
        except Exception as e:
            print(" error printing item", i, e)

    # match Exp_Main convention: batch_x, batch_y, batch_x_mark, batch_y_mark, dataset_id
    if len(batch) == 5:
        batch_x, batch_x_mark, batch_y_full, batch_y_mark, ds_id = batch
    else:
        raise RuntimeError("Unexpected batch format")

    # Print shapes after moving to device
    batch_x = batch_x.float().to(device)
    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_full = batch_y_full.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    print("\nAfter .to(device):")
    print(" batch_x:", batch_x.shape)
    print(" batch_x_mark:", batch_x_mark.shape)
    print(" batch_y_full:", batch_y_full.shape)
    print(" batch_y_mark:", batch_y_mark.shape)
    print(" ds_id:", ds_id)

    # Build dec_inp same as Exp_Main does
    seq_len = args.seq_len
    label_len = args.label_len
    pred_len = args.pred_len

    zeros = torch.zeros((batch_x.size(0), pred_len, batch_x.size(2)), device=device)
    dec_inp = torch.cat([batch_y_full[:, :label_len, :], zeros], dim=1)
    print(" dec_inp:", dec_inp.shape)

    # Small wrapper to print inside model: monkeypatch proj_in to show input shape
    real_proj_in = model.proj_in
    def debug_proj_in(x):
        print(" >>> proj_in received tensor.shape:", x.shape, "dtype:", x.dtype)
        return real_proj_in(x)
    model.proj_in = debug_proj_in

    print("\nCalling model(...)")
    out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    print("Model output shape:", out.shape)

if __name__ == "__main__":
    main()
