import torch, numpy as np, time, os
from models import Autoformer, TSMixer
import argparse

def load_teacher(ckpt_path, input_dim):
    cfg = argparse.Namespace(
        seq_len=96, label_len=48, pred_len=96,
        enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
        dropout=0.05, factor=3, moving_avg=25, embed="timeF",
        freq="h", activation="gelu", output_attention=False, distil=True
    )
    model = Autoformer.Model(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("model", state), strict=False)
    model.eval()
    return model

def load_student(ckpt_path, input_dim):
    model = TSMixer(pred_len=96, input_dim=input_dim, latent_dim=64)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Safe load: skip mismatched layers
    model_state = model.state_dict()
    loaded_state = {}
    skipped = []

    for k, v in ckpt.items():
        if k in model_state and model_state[k].shape == v.shape:
            loaded_state[k] = v
        else:
            skipped.append(k)

    model_state.update(loaded_state)
    model.load_state_dict(model_state, strict=False)

    if skipped:
        print(f"⚠️ Skipped {len(skipped)} mismatched layers: {skipped[:3]} ...")

    model.eval()
    return model


def forecast(model, X, device):
    model = model.to(device)
    X = X.to(device)  # ✅ move input tensor to same device
    with torch.no_grad():
        if isinstance(model, Autoformer.Model):
            B, L, D = X.shape
            seq_len, label_len, pred_len = 96, 48, 96

            # ✅ ensure all tensors are on same device
            x_enc = X[:, :seq_len, :].to(device)
            x_mark_enc = torch.zeros((B, seq_len, 4), device=device)
            x_dec = torch.zeros((B, label_len + pred_len, D), device=device)
            x_mark_dec = torch.zeros((B, label_len + pred_len, 4), device=device)

            y_pred = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            y_pred = model(X.to(device))
    return y_pred.detach().cpu().numpy()

def profile_model(model, sample_input, device):
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(10):
        forecast(model, sample_input, device)
    torch.cuda.synchronize()
    end = time.perf_counter()
    latency = (end - start) / 10 * 1000
    mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return {"params": total_params, "size_mb": size_mb, "latency": latency, "memory": mem}
