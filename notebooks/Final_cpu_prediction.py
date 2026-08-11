# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
try:
    import torch
    print("torch already present:", torch.__version__)
except ImportError:
    print("torch not found — installing it first")
    %pip install -q torch
    dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -q --no-build-isolation torch-geometric torch-geometric-temporal
# MAGIC %pip install -q scikit-learn -U
# MAGIC

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import torch
TORCH_VERSION = torch.__version__.split("+")[0]
CUDA_TAG = ("cu" + torch.version.cuda.replace(".", "")) if (torch.cuda.is_available() and torch.version.cuda) else "cpu"
wheel_url = f"https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_TAG}.html"
%pip install -q torch-sparse torch-scatter -f $wheel_url
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip uninstall -y torch-sparse torch-scatter torch-cluster torch-spline-conv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import torch
import os

# Get PyTorch and CUDA versions
torch_version = torch.__version__.split('+')[0]
cuda_version = torch.version.cuda
if cuda_version is None:
    cuda_version = 'cpu'
else:
    cuda_version = f"cu{cuda_version.replace('.', '')}"

print(f"PyTorch version: {torch_version} | CUDA version: {cuda_version}")

# Install torch-scatter and torch-sparse matching your environment
os.environ['TORCH'] = torch_version
os.environ['CUDA'] = cuda_version

!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import TGCN, GConvGRU, GCLSTM, GConvLSTM
print("torch_geometric_temporal imports cleanly.")

# COMMAND ----------

import torch
print('torch:', torch.__version__, ' CUDA available:', torch.cuda.is_available())

# COMMAND ----------

# spark.range(10).count()

# COMMAND ----------

# from pyspark.sql.functions import udf
# from pyspark.sql.types import IntegerType

# @udf(returnType=IntegerType())
# def test_udf(x):
#     return x + 1

# spark.range(10).select(test_udf("id")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confirm S3 access via the existning external connrction

# COMMAND ----------


FE_OUTPUT_ROOT ="s3://research-s20426/tgnn_dataset_v2"   
# Quick reachability check — lists the top level of the FE output root
try:
    display(dbutils.fs.ls(FE_OUTPUT_ROOT))
    print("S3 path is reachable.")
except Exception as e:
    print("Could not list FE_OUTPUT_ROOT — check the external connection / path.")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ### Match the Fe_output_root path

# COMMAND ----------

SPLITS = ["train", "val", "test"]
NODE_PATHS = {s: f"{FE_OUTPUT_ROOT}/snapshots/{s}/nodes" for s in SPLITS}
EDGE_PATHS = {s: f"{FE_OUTPUT_ROOT}/snapshots/{s}/edges" for s in SPLITS}
DATASET_STATS_PATH = f"{FE_OUTPUT_ROOT}/dataset_stats.json"

print(NODE_PATHS)
print(EDGE_PATHS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load all t_idx snapshots per split via Spark (nodes + edges)

# COMMAND ----------

from pyspark.sql import functions as F

QUICK_TEST = False   # <-- full run: loads every t_idx partition per split
N_PARTITIONS = 4       # only used when QUICK_TEST = True

def load_split_spark(path, quick_test=False, n_partitions=4):
    sdf = spark.read.parquet(path)   # auto-detects the t_idx= partition column

    if quick_test:
        keep_t = [
            r["t_idx"] for r in
            sdf.select("t_idx").distinct().orderBy("t_idx").limit(n_partitions).collect()
        ]
        print(f"  [{path}] quick_test -> keeping t_idx {keep_t}")
        sdf = sdf.filter(F.col("t_idx").isin(keep_t))
    else:
        n_parts = sdf.select("t_idx").distinct().count()
        print(f"  [{path}] loading all {n_parts} t_idx partitions")

    return sdf

nodes_sdf, edges_sdf = {}, {}

for split in SPLITS:

    print(f"\nLoading {split} nodes...")
    nodes_sdf[split] = load_split_spark(NODE_PATHS[split], QUICK_TEST, N_PARTITIONS)

    print(f"Loading {split} edges...")
    edges_sdf[split] = load_split_spark(EDGE_PATHS[split], QUICK_TEST, N_PARTITIONS)

    # Keep timestamps common to nodes and edges — small distinct-value collects, not full rows
    node_t = {r["t_idx"] for r in nodes_sdf[split].select("t_idx").distinct().collect()}
    edge_t = {r["t_idx"] for r in edges_sdf[split].select("t_idx").distinct().collect()}
    common_t = sorted(node_t & edge_t)

    # NOTE: .cache() is intentionally NOT used here — on serverless compute it's implemented via
    # PERSIST TABLE, which serverless doesn't support (NOT_SUPPORTED_WITH_SERVERLESS). Serverless
    # Spark/Photon already reuses/optimizes query plans on its own, so this just means the filtered
    # DataFrame gets recomputed from the parquet read on each downstream action instead of being
    # materialized once — a minor cost, not a correctness issue.
    nodes_sdf[split] = nodes_sdf[split].filter(F.col("t_idx").isin(common_t))
    edges_sdf[split] = edges_sdf[split].filter(F.col("t_idx").isin(common_t))

    n_rows = nodes_sdf[split].count()
    e_rows = edges_sdf[split].count()

    print(f"{split}: {len(common_t)} usable snapshots")
    print(" Node rows:", n_rows)
    print(" Edge rows:", e_rows)

print("\nTrain node columns:", nodes_sdf["train"].columns)
print("Train edge columns:", edges_sdf["train"].columns)

# COMMAND ----------

# ------------------------------------------------------------------
# Distributed null/NaN check — runs as one Spark job per split instead of
# collecting everything first and checking in pandas/torch afterward.
# ------------------------------------------------------------------
for split in SPLITS:
    null_row = nodes_sdf[split].select([
        F.sum(F.col(c).isNull().cast("int")).alias(c) for c in nodes_sdf[split].columns
    ]).collect()[0].asDict()
    nonzero = {k: v for k, v in null_row.items() if v}
    print(f"{split:5s} node nulls (non-zero columns only): {nonzero or 'none'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Identify id / target /edge endpoint columns

# COMMAND ----------

# Column names only — Spark DataFrame.columns is metadata, no data is scanned/collected here.
node_cols = nodes_sdf["train"].columns
edge_cols = edges_sdf["train"].columns

# --- Node side ---
NODE_ID_COL_OVERRIDE = None     # e.g. "node_id"
TARGET_COL_OVERRIDE = None      # e.g. "cpu_util_rs"

node_id_candidates = [c for c in ["node_id", "msname"] if c in node_cols]
NODE_ID_COL = NODE_ID_COL_OVERRIDE or (node_id_candidates[0] if node_id_candidates else None)
assert NODE_ID_COL, "Couldn't find a node id column — set NODE_ID_COL_OVERRIDE manually."

target_candidates = [c for c in node_cols if "cpu" in c.lower()]
TARGET_COL = TARGET_COL_OVERRIDE or (target_candidates[0] if target_candidates else None)
assert TARGET_COL, "Couldn't find a cpu-related column — set TARGET_COL_OVERRIDE manually."

ID_COLS = list({NODE_ID_COL, "t_idx"} & set(node_cols))

print("NODE_ID_COL:", NODE_ID_COL)
print("TARGET_COL :", TARGET_COL)

# --- Edge side ---
EDGE_SRC_COL_OVERRIDE = None     # e.g. "src_node_id"
EDGE_DST_COL_OVERRIDE = None     # e.g. "dst_node_id"

def _find_endpoint_col(cols, keyword):
    matches = [c for c in cols if keyword in c.lower()]
    return matches[0] if matches else None

EDGE_SRC_COL = EDGE_SRC_COL_OVERRIDE or _find_endpoint_col(edge_cols, "src")
EDGE_DST_COL = EDGE_DST_COL_OVERRIDE or _find_endpoint_col(edge_cols, "dst")
assert EDGE_SRC_COL and EDGE_DST_COL, (
    "Couldn't auto-detect edge endpoint columns — set EDGE_SRC_COL_OVERRIDE / "
    "EDGE_DST_COL_OVERRIDE manually. Available edge columns: " + str(edge_cols)
)
print("EDGE_SRC_COL:", EDGE_SRC_COL)
print("EDGE_DST_COL:", EDGE_DST_COL)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build a temporal graph signal per split

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore", message=".*torch-sparse.*")

import numpy as np
import pandas as pd
from pyspark.sql.types import NumericType
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def get_common_nodes(node_sdf, node_id_col, t_col="t_idx"):
    total_t = node_sdf.select(t_col).distinct().count()
    node_t_counts = node_sdf.groupBy(node_id_col).agg(F.countDistinct(t_col).alias("t_count"))
    common = node_t_counts.filter(F.col("t_count") == total_t).select(node_id_col)
    common_nodes = sorted(r[node_id_col] for r in common.collect())
    return common_nodes, total_t

def build_signal(node_sdf, edge_sdf):
    t_idxs = sorted(r["t_idx"] for r in node_sdf.select("t_idx").distinct().collect())

    common_nodes, total_t = get_common_nodes(node_sdf, NODE_ID_COL)
    node_pos = {nid: i for i, nid in enumerate(common_nodes)}
    N = len(common_nodes)

    # Numeric feature columns straight from the Spark schema — no collect needed
    feature_cols = [
        f.name for f in node_sdf.schema.fields
        if f.name not in set(ID_COLS + [TARGET_COL]) and isinstance(f.dataType, NumericType)
    ]

    # Filter + prune columns in Spark before collecting node features
    node_sdf_common = (
        node_sdf
        .filter(F.col(NODE_ID_COL).isin(common_nodes))
        .select(NODE_ID_COL, "t_idx", *feature_cols, TARGET_COL)
    )
    node_pdf = node_sdf_common.toPandas()

    # Filter to common-node edges + dedup in Spark, then collect only src/dst
    edge_sdf_common = (
        edge_sdf
        .filter(F.col(EDGE_SRC_COL).isin(common_nodes) & F.col(EDGE_DST_COL).isin(common_nodes))
        .dropDuplicates([EDGE_SRC_COL, EDGE_DST_COL])
        .select(EDGE_SRC_COL, EDGE_DST_COL)
    )
    edge_pdf = edge_sdf_common.toPandas()

    src_idx = edge_pdf[EDGE_SRC_COL].map(node_pos).to_numpy()
    dst_idx = edge_pdf[EDGE_DST_COL].map(node_pos).to_numpy()
    edge_index = np.vstack([src_idx, dst_idx]).astype(np.int64)
    edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)

    features, targets = [], []
    for t in t_idxs:
        sub = node_pdf[node_pdf.t_idx == t].drop_duplicates(subset=[NODE_ID_COL]).set_index(NODE_ID_COL)
        sub = sub.loc[common_nodes]
        features.append(sub[feature_cols].to_numpy(dtype=np.float32))
        targets.append(sub[TARGET_COL].to_numpy(dtype=np.float32))

    print(f"  steps={len(t_idxs)}  nodes={N}  edges={edge_index.shape[1]}  feature_dim={len(feature_cols)}")
    signal = StaticGraphTemporalSignal(edge_index=edge_index, edge_weight=edge_weight,
                                        features=features, targets=targets)
    return signal, feature_cols, N

print("Building train signal..."); train_signal, feature_cols, N_train = build_signal(nodes_sdf["train"], edges_sdf["train"])
print("Building val signal...");   val_signal,   _,           N_val   = build_signal(nodes_sdf["val"],   edges_sdf["val"])
print("Building test signal...");  test_signal,  _,           N_test  = build_signal(nodes_sdf["test"],  edges_sdf["test"])

IN_CHANNELS = len(feature_cols)
print("\nfeature columns used as model input:", feature_cols)
print("IN_CHANNELS:", IN_CHANNELS)

# NOTE: .unpersist() left in place but these are now no-ops since .cache() isn't used on
# serverless compute — harmless to call on a non-cached DataFrame.
# for split in SPLITS:
#     nodes_sdf[split].unpersist()
#     edges_sdf[split].unpersist()

# COMMAND ----------

# Final sanity check — NaNs anywhere in x / edge_attr / y of the built tensors
for name, signal in [("train", train_signal), ("val", val_signal), ("test", test_signal)]:
    x_nan = any(torch.isnan(s.x).any().item() for s in signal)
    e_nan = any(torch.isnan(s.edge_attr).any().item() for s in signal)
    y_nan = any(torch.isnan(s.y).any().item() for s in signal)
    print(f"{name:5s}  steps={signal.snapshot_count:3d}  x_nan={x_nan}  edge_attr_nan={e_nan}  y_nan={y_nan}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define 4 TGNN models

# COMMAND ----------

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN, GConvGRU, GCLSTM, GConvLSTM

HIDDEN_CHANNELS = 16

class TGCNPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.recurrent = TGCN(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, 1)
    def forward(self, x, edge_index, edge_weight, h):
        h = self.recurrent(x, edge_index, edge_weight, h)
        return self.linear(h).squeeze(-1), h

class GConvGRUPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=2):
        super().__init__()
        self.recurrent = GConvGRU(in_channels, hidden_channels, K)
        self.linear = nn.Linear(hidden_channels, 1)
    def forward(self, x, edge_index, edge_weight, h):
        h = self.recurrent(x, edge_index, edge_weight, h)
        return self.linear(h).squeeze(-1), h

class GCLSTMPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=2):
        super().__init__()
        self.recurrent = GCLSTM(in_channels=in_channels, out_channels=hidden_channels, K=K)
        self.linear = nn.Linear(hidden_channels, 1)
    def forward(self, x, edge_index, edge_weight, state):
        h, c = state if state is not None else (None, None)
        h, c = self.recurrent(x, edge_index, edge_weight, h, c)
        return self.linear(h).squeeze(-1), (h, c)

class GConvLSTMPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, K=2):
        super().__init__()
        self.recurrent = GConvLSTM(in_channels=in_channels, out_channels=hidden_channels, K=K)
        self.linear = nn.Linear(hidden_channels, 1)
    def forward(self, x, edge_index, edge_weight, state):
        h, c = state if state is not None else (None, None)
        h, c = self.recurrent(x, edge_index, edge_weight, h, c)
        return self.linear(h).squeeze(-1), (h, c)

model_builders = {
    "TGCN":      lambda: TGCNPredictor(IN_CHANNELS, HIDDEN_CHANNELS),
    "GConvGRU":  lambda: GConvGRUPredictor(IN_CHANNELS, HIDDEN_CHANNELS, K=2),
    "GCLSTM":    lambda: GCLSTMPredictor(IN_CHANNELS, HIDDEN_CHANNELS, K=2),
    "GConvLSTM": lambda: GConvLSTMPredictor(IN_CHANNELS, HIDDEN_CHANNELS, K=2),
}
print("Models ready:", list(model_builders.keys()))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and Evaluation Function

# COMMAND ----------

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# Hyperparameters
# ============================================================
EPOCHS = 30
LR = 0.01

# ============================================================
# Detach hidden state (supports Tensor or Tuple, e.g. GCLSTM's (h, c))
# ============================================================
def detach_state(state):
    if state is None:
        return None
    if isinstance(state, tuple):
        return tuple(s.detach() for s in state)
    return state.detach()

# ============================================================
# Train — returns the fitted model AND its per-epoch loss history
# ============================================================
# def train_model(model, signal, epochs=EPOCHS, lr=LR, clip_grad_norm=1.0):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     model.train()

#     loss_history = []

#     for epoch in range(epochs):
#         state = None
#         total_loss = 0.0
#         steps = 0

#         optimizer.zero_grad()

#         for snapshot in signal:
#             out, state = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, state)
#             loss = F.mse_loss(out, snapshot.y)

#             total_loss += loss
#             steps += 1
#             state = detach_state(state)

#         total_loss = total_loss / steps
#         total_loss.backward()

#         if clip_grad_norm is not None:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

#         optimizer.step()

#         loss_history.append(total_loss.item())

#         if epoch == 0 or (epoch + 1) % 10 == 0:
#             print(f"    epoch {epoch+1:>3}/{epochs}  loss={total_loss.item():.4f}")

#     return model, loss_history


# ============================================================
# Train — returns the fitted model AND its per-epoch loss history
# ============================================================
def train_model(model, signal, epochs=EPOCHS, lr=LR, clip_grad_norm=1.0, eval_signal=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    eval_loss_history = []

    for epoch in range(epochs):
        model.train()
        state = None
        total_loss = 0.0
        steps = 0

        optimizer.zero_grad()

        for snapshot in signal:
            out, state = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, state)
            loss = F.mse_loss(out, snapshot.y)

            total_loss += loss
            steps += 1
            state = detach_state(state)

        total_loss = total_loss / steps
        total_loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()

        loss_history.append(total_loss.item())

        if eval_signal is not None:
            model.eval()
            eval_state = None
            eval_total_loss = 0.0
            eval_steps = 0
            with torch.no_grad():
                for snapshot in eval_signal:
                    out, eval_state = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, eval_state)
                    eval_total_loss += F.mse_loss(out, snapshot.y).item()
                    eval_steps += 1
            eval_loss_history.append(eval_total_loss / eval_steps)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            msg = f"    epoch {epoch+1:>3}/{epochs}  train_loss={total_loss.item():.4f}"
            if eval_signal is not None:
                msg += f"  test_loss={eval_loss_history[-1]:.4f}"
            print(msg)

    return model, loss_history, eval_loss_history

# ============================================================
# Evaluate — RMSE computed via np.sqrt (works on every sklearn version;
# the old `squared=False` kwarg was removed in recent sklearn releases)
# ============================================================
def evaluate_model(model, signal):
    model.eval()
    preds, trues = [], []
    state = None

    with torch.no_grad():
        for snapshot in signal:
            out, state = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, state)
            preds.append(out.detach().cpu().numpy())
            trues.append(snapshot.y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)

    return rmse, mae, r2, preds, trues

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and Evaluate all 4 models

# COMMAND ----------

# fitted = {}
# histories = {}
# predictions = {}   # name -> {"train": (preds, trues), "val": ..., "test": ...}
# rows = []

# for name, build in model_builders.items():

#     print(f"\n{'='*60}")
#     print(f"Training {name}")
#     print("="*60)

#     model = build()
#     model, loss_history = train_model(model, train_signal)

#     fitted[name] = model
#     histories[name] = loss_history

#     print("Evaluating on train set...")
#     train_rmse, train_mae, train_r2, train_preds, train_trues = evaluate_model(model, train_signal)

#     print("Evaluating on validation set...")
#     val_rmse, val_mae, val_r2, val_preds, val_trues = evaluate_model(model, val_signal)

#     print("Evaluating on test set...")
#     test_rmse, test_mae, test_r2, test_preds, test_trues = evaluate_model(model, test_signal)

#     predictions[name] = {
#         "train": (train_preds, train_trues),
#         "val":   (val_preds, val_trues),
#         "test":  (test_preds, test_trues),
#     }

#     rows.append({
#         "Model": name,
#         "Train RMSE": train_rmse, "Train MAE": train_mae, "Train R²": train_r2,
#         "Validation RMSE": val_rmse, "Validation MAE": val_mae, "Validation R²": val_r2,
#         "Test RMSE": test_rmse, "Test MAE": test_mae, "Test R²": test_r2,
#     })

# results_df = pd.DataFrame(rows).sort_values(by="Test RMSE", ascending=True).reset_index(drop=True)

# print("\nModel Performance")
# print(results_df)
# results_df
fitted = {}
histories = {}
test_histories = {}
predictions = {}   # name -> {"train": (preds, trues), "val": ..., "test": ...}
rows = []

for name, build in model_builders.items():

    print(f"\n{'='*60}")
    print(f"Training {name}")
    print("="*60)

    model = build()
    model, loss_history, eval_loss_history = train_model(model, train_signal, eval_signal=test_signal)

    fitted[name] = model
    histories[name] = loss_history
    test_histories[name] = eval_loss_history

    print("Evaluating on train set...")
    train_rmse, train_mae, train_r2, train_preds, train_trues = evaluate_model(model, train_signal)

    print("Evaluating on validation set...")
    val_rmse, val_mae, val_r2, val_preds, val_trues = evaluate_model(model, val_signal)

    print("Evaluating on test set...")
    test_rmse, test_mae, test_r2, test_preds, test_trues = evaluate_model(model, test_signal)

    predictions[name] = {
        "train": (train_preds, train_trues),
        "val":   (val_preds, val_trues),
        "test":  (test_preds, test_trues),
    }

    rows.append({
        "Model": name,
        "Train RMSE": train_rmse, "Train MAE": train_mae, "Train R²": train_r2,
        "Validation RMSE": val_rmse, "Validation MAE": val_mae, "Validation R²": val_r2,
        "Test RMSE": test_rmse, "Test MAE": test_mae, "Test R²": test_r2,
    })

results_df = pd.DataFrame(rows).sort_values(by="Test RMSE", ascending=True).reset_index(drop=True)

print("\nModel Performance")
print(results_df)
results_df

# COMMAND ----------

# ------------------------------------------------------------------
# Where Section 10's evaluation outputs get saved. matplotlib/pandas can only
# write to local disk directly, so figures/tables are written to LOCAL_RESULTS_DIR
# first, then copied up to RESULTS_ROOT on S3 via dbutils.fs.cp.
# (Restored here — the cells below call save_artifact(...) but the notebook this
# was exported from no longer had this definition anywhere, which would have
# raised NameError the moment 10.2 ran.)
# ------------------------------------------------------------------
# import os

# RESULTS_ROOT = f"{FE_OUTPUT_ROOT}/results"           # persisted results land here in S3
# LOCAL_RESULTS_DIR = "/local_disk0/tmp/results"        # local scratch dir

# os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
# dbutils.fs.mkdirs(f"file:{LOCAL_RESULTS_DIR}")

# def save_artifact(filename):
#     # Copy a file already written to LOCAL_RESULTS_DIR up to RESULTS_ROOT on S3.
#     local_path = f"{LOCAL_RESULTS_DIR}/{filename}"
#     dest_path = f"{RESULTS_ROOT}/{filename}"
#     dbutils.fs.cp(f"file:{local_path}", dest_path)
#     print(f"  saved -> {dest_path}")
#     return local_path

# print("Evaluation artifacts will be written locally to:", LOCAL_RESULTS_DIR)
# print("and persisted to:", RESULTS_ROOT)


# COMMAND ----------

# ------------------------------------------------------------------
# 10.1 Overfitting gap table
# ------------------------------------------------------------------
eval_df = results_df.copy()
eval_df["RMSE Gap (Test - Train)"] = eval_df["Test RMSE"] - eval_df["Train RMSE"]
eval_df["RMSE Gap (Val - Train)"] = eval_df["Validation RMSE"] - eval_df["Train RMSE"]
eval_df["Overfit Ratio (Test/Train RMSE)"] = eval_df["Test RMSE"] / eval_df["Train RMSE"]

display_cols = [
    "Model",
    "Train RMSE", "Validation RMSE", "Test RMSE",
    "RMSE Gap (Test - Train)", "Overfit Ratio (Test/Train RMSE)",
    "Train R²", "Validation R²", "Test R²",
]
print("Train / Val / Test trade-off")
print(eval_df[display_cols].to_string(index=False))
eval_df[display_cols]

# COMMAND ----------

# ------------------------------------------------------------------
# 10.2 Train vs Val vs Test RMSE, grouped bar chart per model
# ------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results_df))
width = 0.25

ax.bar(x - width, results_df["Train RMSE"], width, label="Train")
ax.bar(x,          results_df["Validation RMSE"], width, label="Validation")
ax.bar(x + width,  results_df["Test RMSE"], width, label="Test")

ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"])
ax.set_ylabel("RMSE")
ax.set_title("Train vs Validation vs Test RMSE by model")
ax.legend()
plt.tight_layout()
fig.savefig("outputs/cpu_results/train_test_val_by_RMSE.png", dpi=150, bbox_inches="tight")
# save_artifact("train_test_val_by_RMSE.png")
plt.show()

# COMMAND ----------

# ------------------------------------------------------------------
# 10.3 Learning curves — training vs test loss per epoch, one subplot per model
# ------------------------------------------------------------------
model_names = list(histories.keys())
n_models = len(model_names)
n_cols = 2
n_rows = int(np.ceil(n_models / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
axes = axes.flatten()

for i, name in enumerate(model_names):
    ax = axes[i]
    ax.plot(range(1, len(histories[name]) + 1), histories[name],
             label="train", color="tab:blue")
    if name in test_histories and test_histories[name]:
        ax.plot(range(1, len(test_histories[name]) + 1), test_histories[name],
                 label="test", color="tab:orange", linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(name)
    ax.legend()

# Hide any unused subplot axes (when n_models is odd)
for j in range(n_models, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Training vs Test loss curves per model", y=1.02)
plt.tight_layout()
fig.savefig("outputs/cpu_results/learning_curves.png", dpi=150, bbox_inches="tight")
# save_artifact("learning_curves.png")
plt.show()

# COMMAND ----------

# ------------------------------------------------------------------
# 10.4 Predicted vs actual + residuals for the best model (lowest Test RMSE)
# ------------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_preds, best_trues = predictions[best_model_name]["test"]
residuals = best_preds - best_trues

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(best_trues, best_preds, alpha=0.4, s=10)
lims = [min(best_trues.min(), best_preds.min()), max(best_trues.max(), best_preds.max())]
axes[0].plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
axes[0].set_xlabel("Actual (scaled CPU target)")
axes[0].set_ylabel("Predicted")
axes[0].set_title(f"{best_model_name}: Predicted vs Actual (test)")
axes[0].legend()

axes[1].scatter(best_preds, residuals, alpha=0.4, s=10)
axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Residual (Predicted - Actual)")
axes[1].set_title(f"{best_model_name}: Residuals (test)")

plt.tight_layout()
plt.savefig("outputs/cpu_results/predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Best model by Test RMSE: {best_model_name}")
print(f"  Test RMSE: {results_df.iloc[0]['Test RMSE']:.4f}")
print(f"  Test MAE:  {results_df.iloc[0]['Test MAE']:.4f}")
print(f"  Test R²:   {results_df.iloc[0]['Test R²']:.4f}")
print(f"  Residual mean: {residuals.mean():.4f}  (near 0 = unbiased)")
print(f"  Residual std:  {residuals.std():.4f}")