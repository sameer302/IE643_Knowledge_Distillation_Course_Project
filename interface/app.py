import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# interface/app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
import os, time
from utils.processing import (
    zscore_normalize,
    denormalize_array,
    create_sliding_windows,
    detect_model_params,
)

st.set_page_config(page_title="KD Forecast Interface", layout="wide")

st.title("📈 Knowledge Distillation Forecast Interface")
st.markdown("Upload a CSV → Normalize → Predict with Teacher & Student → Visualize & Compare.")

# ===============================
# 1️⃣ File upload and inputs
# ===============================
uploaded = st.file_uploader("Upload a multivariate time-series CSV file", type=["csv"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
# Automatically detect columns
time_col = df.columns[0]
target_col = df.columns[-2]

# Try parsing time column (safe)
try:
    df[time_col] = pd.to_datetime(df[time_col])
    # st.success(f"✅ Time column detected: '{time_col}' (parsed as datetime)")
except Exception:
    st.warning(f"⚠️ Using '{time_col}' as time column (not parsed as datetime)")

# st.info(f"📌 Target column automatically selected: '{target_col}'")

# 🔹 Fixed model checkpoint paths (edit these to your actual files)
teacher_ckpt = "/home/sameer/Desktop/AutumnSem25/Courses/IE643/KD_Final/checkpoints/global_finetune_new/dataset_manifest/global_finetune_new/checkpoint.pth"
student_ckpt = "/home/sameer/Desktop/AutumnSem25/Courses/IE643/KD_Final/checkpoints/student_kd/electricity_areawise_normalized_clean_electricity_countrywise_normalized_electricity_household_normalized_electricity_jerico_normalized_student_projected_adaptive_smoothing.pth"

# st.write(f"Forecasting target: `{target_col}`")

# ===============================
# 2️⃣ Normalize dataset
# ===============================
# st.subheader("Step 1: Normalize Dataset")
# Drop non-numeric columns (e.g., time, strings)
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] == 0:
    st.error("No numeric columns found in the dataset.")
    st.stop()

# Normalize numeric columns only (used internally for modeling)
df_norm, mean, std = zscore_normalize(numeric_df)
# st.success("Numeric columns normalized using dataset's own z-score statistics (hidden from display).")

# --- Visualization: Plot raw uploaded data instead of normalized ---
st.subheader("📈 Uploaded Data Overview")

# Pick up to 5 numeric columns for visualization
cols_to_plot = st.multiselect(
    "Select columns to visualize (default: first 3 numeric columns)",
    options=list(numeric_df.columns),
    default=list(numeric_df.columns[:3])
)

from pandas.api.types import is_datetime64_any_dtype

# Determine x-axis (time or index)
x_axis = df[df.columns[0]]

if not is_datetime64_any_dtype(x_axis):
    st.warning(f"⚠️ Time column '{df.columns[0]}' is not datetime — using row index instead.")
    x_axis = np.arange(len(df))

# Build line plot
import plotly.express as px
plot_df = df[cols_to_plot]
plot_df["x_axis"] = x_axis
plot_df = plot_df.melt(id_vars="x_axis", var_name="Feature", value_name="Value")
fig = px.line(plot_df, x="x_axis", y="Value", color="Feature",
              title="Raw Uploaded Dataset",
              labels={"x_axis": "Time", "Value": "Feature Value"})
st.plotly_chart(fig, use_container_width=True)


# ===============================
# 3️⃣ Detect seq_len and pred_len
# ===============================
seq_len_t, pred_len_t = detect_model_params(teacher_ckpt)
seq_len_s, pred_len_s = detect_model_params(student_ckpt)

# Take common values (usually same)
SEQ_LEN = min(seq_len_t, seq_len_s)
PRED_LEN = min(pred_len_t, pred_len_s)

# st.info(f"Detected model parameters → seq_len={SEQ_LEN}, pred_len={PRED_LEN}")

# ===============================
# 4️⃣ Create sliding windows
# ===============================
X, Y = create_sliding_windows(df_norm, seq_len=SEQ_LEN, pred_len=PRED_LEN)
if X.shape[0] == 0:
    st.error("Dataset too short for detected seq_len/pred_len.")
    st.stop()
# st.success(f"Created {X.shape[0]} samples of shape {X.shape[1:]}")

# ===============================
# 5️⃣ Load teacher and student
# ===============================
from models import Autoformer, TSMixer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = X.shape[-1]

# ---- Teacher ----
teacher = Autoformer.Model(
    __import__("argparse").Namespace(
        seq_len=SEQ_LEN, label_len=48, pred_len=PRED_LEN,
        enc_in=input_dim, dec_in=input_dim, c_out=input_dim,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
        dropout=0.05, factor=3, moving_avg=25, embed="timeF", freq="h",
        activation="gelu", output_attention=False, distil=True
    )
)
ck_teacher = torch.load(teacher_ckpt, map_location="cpu")
teacher.load_state_dict(ck_teacher.get("model", ck_teacher), strict=False)
teacher.eval().to(device)

# ---- Student ----
student = TSMixer(pred_len=PRED_LEN, input_dim=input_dim, latent_dim=64)
ck_student = torch.load(student_ckpt, map_location="cpu")

# --- Safe state loading (ignore shape mismatches) ---
model_state = student.state_dict()
matched, mismatched = [], []

for k, v in ck_student.items():
    if k in model_state and model_state[k].shape == v.shape:
        model_state[k] = v
        matched.append(k)
    else:
        mismatched.append(k)

student.load_state_dict(model_state, strict=False)
student.eval().to(device)

# if mismatched:
#     st.warning(f"⚠️ Skipped {len(mismatched)} mismatched layers (dataset has different feature count): {mismatched[:3]}...")
# else:
#     st.success("✅ All student model layers matched successfully.")

# st.success("✅ Loaded both Teacher and Student models.")

# ===============================
# 6️⃣ Run predictions
# ===============================
run = st.button("Run Predictions")
if not run:
    st.stop()

# ===============================
# 6️⃣ Prepare data and run predictions
# ===============================
X_t = torch.tensor(X, dtype=torch.float32)
Y_t = torch.tensor(Y, dtype=torch.float32)
dl = DataLoader(TensorDataset(X_t, Y_t), batch_size=32, shuffle=False)

teacher_preds, student_preds, y_trues = [], [], []

# Determine target column index safely
col_idx = list(numeric_df.columns).index(target_col)

with torch.no_grad():
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)

        # ---------- Teacher Forward ----------
        seq_len, label_len = SEQ_LEN, 48
        x_enc = xb[:, :seq_len, :]
        x_dec = xb[:, :label_len + PRED_LEN, :]
        x_mark_enc = torch.zeros((xb.shape[0], seq_len, 4), device=device)
        x_mark_dec = torch.zeros((xb.shape[0], label_len + PRED_LEN, 4), device=device)

        t_out = teacher(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(t_out, tuple):
            t_out = t_out[0]
        teacher_preds.append(t_out.cpu().numpy())

        # ---------- Student Forward ----------
        s_out = student(xb)
        student_preds.append(s_out.cpu().numpy())

        # ---------- Ground Truth ----------
        y_trues.append(yb.cpu().numpy())

# Convert to arrays
teacher_preds = np.concatenate(teacher_preds)
student_preds = np.concatenate(student_preds)
y_trues = np.concatenate(y_trues)

# ===============================
# 7️⃣ Compute metrics + resource profiling
# ===============================

def mse(a, b): return np.mean((a - b) ** 2)
def mae(a, b): return np.mean(np.abs(a - b))

# Handle possible feature-count mismatch
output_dim = teacher_preds.shape[-1]
if col_idx >= output_dim:
    st.warning(f"⚠️ Target column index {col_idx} exceeds model output dim {output_dim}. "
               f"Using last available output column instead.")
    col_idx = output_dim - 1

# ---------- Accuracy metrics ----------
teacher_mse = mse(teacher_preds[:, :, col_idx], y_trues[:, :, col_idx])
teacher_mae = mae(teacher_preds[:, :, col_idx], y_trues[:, :, col_idx])
student_mse = mse(student_preds[:, :, col_idx], y_trues[:, :, col_idx])
student_mae = mae(student_preds[:, :, col_idx], y_trues[:, :, col_idx])

# ---------- Resource metrics ----------
def profile_model(model, device, name="Model", sample_input=None):
    """
    Return param count, size (MB), latency (ms/sample), and memory (MB).
    Auto-detects if model is Autoformer or TSMixer and runs accordingly.
    Clears CUDA cache before each run to isolate memory usage.
    """
    model.eval().to(device)

    with torch.no_grad():
        # 1️⃣ Parameter count and estimated model size
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes

        # 2️⃣ Prepare safe sample input
        if sample_input is None:
            sample_input = torch.randn(1, SEQ_LEN, input_dim).to(device)

        # 3️⃣ Define a helper for Autoformer forward
        def safe_forward(m, name, x):
            try:
                if "Autoformer" in name or hasattr(m, "encoder"):
                    B, L, D = x.shape
                    seq_len = SEQ_LEN
                    label_len = 48
                    pred_len = PRED_LEN
                    x_enc = x[:, :seq_len, :]
                    x_dec = x[:, :label_len + pred_len, :]
                    x_mark_enc = torch.zeros((B, seq_len, 4), device=device)
                    x_mark_dec = torch.zeros((B, label_len + pred_len, 4), device=device)
                    _ = m(x_enc, x_mark_enc, x_dec, x_mark_dec)
                else:
                    _ = m(x)
                return True
            except Exception as e:
                st.warning(f"⚠️ Profiling forward failed for {name}: {e}")
                return False

        # 4️⃣ Clear GPU cache before profiling (isolate per model)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # 5️⃣ Warm-up iterations
        n_warmup, n_iters = 3, 10
        for _ in range(n_warmup):
            safe_forward(model, name, sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # 6️⃣ Measure latency and peak memory
        start = time.perf_counter()
        for _ in range(n_iters):
            safe_forward(model, name, sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        latency_ms = (end - start) / n_iters * 1000
        mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        )

    return {
        "Params": total_params,
        "Model Size (MB)": round(size_mb, 2),
        "Latency (ms/sample)": round(latency_ms, 2),
        "GPU Memory (MB)": round(mem_mb, 2),
    }


# Profile both teacher & student
sample_in = torch.tensor(X_t[:1], dtype=torch.float32).to(device)
teacher_profile = profile_model(teacher, device, "Teacher", sample_in)
student_profile = profile_model(student, device, "Student", sample_in)

# ---------- Combine into single comparison table ----------
metrics_df = pd.DataFrame([
    {
        "Model": "Teacher",
        "MSE": teacher_mse,
        "MAE": teacher_mae,
        **teacher_profile
    },
    {
        "Model": "Student",
        "MSE": student_mse,
        "MAE": student_mae,
        **student_profile
    }
])

st.subheader("📊 Unified Model Performance & Resource Summary")

# Round numeric columns for neat display (no jinja2 required)
metrics_display = metrics_df.copy()
metrics_display["MSE"] = metrics_display["MSE"].round(6)
metrics_display["MAE"] = metrics_display["MAE"].round(6)
metrics_display["Model Size (MB)"] = metrics_display["Model Size (MB)"].round(2)
metrics_display["Latency (ms/sample)"] = metrics_display["Latency (ms/sample)"].round(2)
metrics_display["GPU Memory (MB)"] = metrics_display["GPU Memory (MB)"].round(2)

st.dataframe(metrics_display.set_index("Model"))



# ===============================
# 8️⃣ Sample visualization (auto-select best sample considering both Teacher & Student)
# ===============================
st.subheader("🎨 Sample Forecast Visualization")

# --- Compute per-sample MSE & MAE for both teacher and student ---
student_errors_mse = np.mean((student_preds[:, :, col_idx] - y_trues[:, :, col_idx]) ** 2, axis=1)
student_errors_mae = np.mean(np.abs(student_preds[:, :, col_idx] - y_trues[:, :, col_idx]), axis=1)
teacher_errors_mse = np.mean((teacher_preds[:, :, col_idx] - y_trues[:, :, col_idx]) ** 2, axis=1)
teacher_errors_mae = np.mean(np.abs(teacher_preds[:, :, col_idx] - y_trues[:, :, col_idx]), axis=1)

# Combine both models' errors
# Lower combined value => both teacher & student are performing well
combined_error = (
    student_errors_mse + student_errors_mae +
    teacher_errors_mse + teacher_errors_mae
)

# Find index with lowest joint error
best_idx = np.argmin(combined_error)

best_metrics = {
    "Student MSE": student_errors_mse[best_idx],
    "Student MAE": student_errors_mae[best_idx],
    "Teacher MSE": teacher_errors_mse[best_idx],
    "Teacher MAE": teacher_errors_mae[best_idx],
}

st.info(
    # f"✅ Best sample index = {best_idx}  "
    f"(Student MSE={best_metrics['Student MSE']:.6f}, MAE={best_metrics['Student MAE']:.6f};  "
    f"Teacher MSE={best_metrics['Teacher MSE']:.6f}, MAE={best_metrics['Teacher MAE']:.6f})"
)

# Allow manual override of sample index
idx = st.number_input(
    "Select sample index (auto-selected best shown by default)",
    min_value=0,
    max_value=X.shape[0] - 1,
    value=int(best_idx),
    step=1
)

# --- Extract the chosen sample ---
x_seq = X[idx, :, col_idx]
y_true = y_trues[idx, :, col_idx]
t_pred = teacher_preds[idx, :, col_idx]
s_pred = student_preds[idx, :, col_idx]

# --- Denormalize for plotting ---
x_seq_den = denormalize_array(x_seq, mean, std, target_col)
y_true_den = denormalize_array(y_true, mean, std, target_col)
t_pred_den = denormalize_array(t_pred, mean, std, target_col)
s_pred_den = denormalize_array(s_pred, mean, std, target_col)

# --- Plot history + forecasts ---
x_axis = np.arange(0, SEQ_LEN + PRED_LEN)
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(SEQ_LEN), y=x_seq_den,
                         name="Input (History)", line=dict(width=2)))
fig.add_trace(go.Scatter(x=np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN),
                         y=y_true_den, name="Ground Truth", line=dict(width=2)))
fig.add_trace(go.Scatter(x=np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN),
                         y=t_pred_den, name="Teacher", line=dict(dash="dash")))
fig.add_trace(go.Scatter(x=np.arange(SEQ_LEN, SEQ_LEN + PRED_LEN),
                         y=s_pred_den, name="Student", line=dict(dash="dot")))
fig.update_layout(
    title=f"Sample #{idx}: {target_col} Forecast ",
    xaxis_title="Time Steps",
    yaxis_title=target_col,
)
st.plotly_chart(fig, use_container_width=True)
