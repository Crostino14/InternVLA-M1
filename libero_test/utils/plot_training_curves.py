#!/usr/bin/env python3
"""
Plot training loss + MSE curves for InternVLA L3 fine-tuning runs.
Reads from WandB offline .wandb binary files.

Usage:
    python plot_training_curves.py
    python plot_training_curves.py --base_dir /custom/path --output_dir ./plots
"""

import os, sys, json, glob, re, argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_BASE_DIR  = "/mnt/beegfs/a.cardamone7/checkpoints/InternVLA_L3_finetune_libero_goal"
DEFAULT_OUT_DIR   = "./output/training_plots"

RUNS = {
    "eps10": "internvla_l3_eps10_l3_spatial_finetune",
    "eps25": "internvla_l3_eps25_l3_spatial_finetune",
    "eps50": "internvla_l3_eps50_l3_spatial_finetune",
}

LOSS_CANDIDATES = ["action_dit_loss", "train/loss", "loss", "train_loss", "total_loss", "action_loss"]
MSE_CANDIDATES  = ["mse_score", "train/mse_loss", "mse_loss", "action_mse", "train/action_mse", "mse"]
STEP_CANDIDATES = ["global_step", "step", "trainer/global_step", "_step"]

COLORS = {"eps10": "#E74C3C", "eps25": "#3498DB", "eps50": "#2ECC71"}
SMOOTH_WEIGHT = 0.9

# ─── WandB binary reader ──────────────────────────────────────────────────────

def read_wandb_file(wandb_file: str) -> list:
    records = []
    try:
        from wandb.proto import wandb_internal_pb2 as pb
        from wandb.sdk.internal.datastore import DataStore

        ds = DataStore()
        ds.open_for_scan(wandb_file)

        while True:
            raw = ds.scan_data()
            if raw is None:
                break
            try:
                record = pb.Record()
                record.ParseFromString(raw)
                if record.HasField("history"):
                    row = {}
                    for item in record.history.item:
                        # Gestisce sia key flat che nested_key
                        if item.nested_key:
                            key = "/".join(item.nested_key)  # ← FIX
                        elif item.key:
                            key = item.key
                        else:
                            continue

                        if item.value_json:
                            try:
                                row[key] = json.loads(item.value_json)
                            except Exception:
                                pass
                    if row:
                        records.append(row)
            except Exception:
                continue
        try:
            ds.close()
        except Exception:
            pass

    except ImportError:
        print("  [ERROR] wandb library not found.")
        sys.exit(1)
    except Exception as e:
        print(f"  [WARN] Could not read {wandb_file}: {e}")

    return records

def load_run(base_dir: str, run_name: str) -> pd.DataFrame:
    """Load and merge all WandB history files for a given run."""
    run_path   = Path(base_dir) / run_name
    wandb_base = run_path / "wandb" / "wandb"

    if not wandb_base.exists():
        print(f"  [WARN] WandB dir not found: {wandb_base}")
        return pd.DataFrame()

    all_records = []
    for sub in sorted(wandb_base.iterdir()):
        if not sub.is_dir():
            continue
        for wf in sub.glob("*.wandb"):
            recs = read_wandb_file(str(wf))
            all_records.extend(recs)
            print(f"    {wf.name}: {len(recs)} steps")

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    step_col = next((c for c in STEP_CANDIDATES if c in df.columns), None)
    if step_col:
        df = df.sort_values(step_col).drop_duplicates(step_col).reset_index(drop=True)

    return df


def find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return first matching column name from candidates list."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ema_smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


# ─── Individual plots ─────────────────────────────────────────────────────────

def plot_individual(label: str, df: pd.DataFrame, out_dir: str):
    step_col = find_col(df, STEP_CANDIDATES)
    loss_col = find_col(df, LOSS_CANDIDATES)
    mse_col  = find_col(df, MSE_CANDIDATES)

    if loss_col is None and mse_col is None:
        print(f"  [SKIP] No loss/mse columns found for {label}. Cols: {list(df.columns)}")
        return

    x     = df[step_col].values if step_col else np.arange(len(df))
    ncols = int(loss_col is not None) + int(mse_col is not None)

    titles = []
    if loss_col: titles.append("Training Loss")
    if mse_col:  titles.append("MSE Loss")

    color = COLORS.get(label, "#8E44AD")
    fig   = make_subplots(rows=1, cols=ncols, subplot_titles=titles,
                          horizontal_spacing=0.12)

    col_idx = 1
    if loss_col:
        mask  = df[loss_col].notna()
        y_raw = df.loc[mask, loss_col].values
        x_l   = x[mask.values]
        y_sm  = ema_smooth(y_raw, SMOOTH_WEIGHT)
        fig.add_trace(go.Scatter(x=x_l, y=y_raw, mode="lines", name="raw",
                                 opacity=0.25, line=dict(color=color, width=1),
                                 showlegend=True), row=1, col=col_idx)
        fig.add_trace(go.Scatter(x=x_l, y=y_sm, mode="lines", name="smoothed",
                                 line=dict(color=color, width=2.5),
                                 showlegend=True), row=1, col=col_idx)
        fig.update_xaxes(title_text="Steps", row=1, col=col_idx)
        fig.update_yaxes(title_text="Loss",  row=1, col=col_idx)
        col_idx += 1

    if mse_col:
        mask   = df[mse_col].notna() 
        y_raw2 = df.loc[mask, mse_col].values
        x_m    = x[mask.values]
        y_sm2  = ema_smooth(y_raw2, SMOOTH_WEIGHT)
        fig.add_trace(go.Scatter(x=x_m, y=y_raw2, mode="lines", name="raw (MSE)",
                                opacity=0.25, line=dict(color=color, width=1),
                                showlegend=True), row=1, col=col_idx)
        fig.add_trace(go.Scatter(x=x_m, y=y_sm2, mode="lines", name="smoothed (MSE)",
                                line=dict(color=color, width=2.5),
                                showlegend=True), row=1, col=col_idx)
        fig.update_xaxes(title_text="Steps", row=1, col=col_idx)
        fig.update_yaxes(title_text="MSE",   row=1, col=col_idx)

    fig.update_layout(
        title_text=f"InternVLA L3 Fine-tuning — {label}",
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                    xanchor="center", x=0.5),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"loss_{label}.png")
    fig.write_image(out_path, scale=2)
    print(f"  Saved: {out_path}")


# ─── Combined plot ─────────────────────────────────────────────────────────────

def plot_combined(run_data: dict, out_dir: str):
    has_loss = any(find_col(df, LOSS_CANDIDATES) is not None
                   for df in run_data.values() if not df.empty)
    has_mse  = any(find_col(df, MSE_CANDIDATES)  is not None
                   for df in run_data.values() if not df.empty)

    ncols = int(has_loss) + int(has_mse)
    if ncols == 0:
        print("  [SKIP] No data for combined plot")
        return

    titles = []
    if has_loss: titles.append("Training Loss — All Runs")
    if has_mse:  titles.append("MSE Loss — All Runs")

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.12)

    for label, df in run_data.items():
        if df.empty:
            continue

        color    = COLORS.get(label, "#8E44AD")
        step_col = find_col(df, STEP_CANDIDATES)
        x        = df[step_col].values if step_col else np.arange(len(df))
        loss_col = find_col(df, LOSS_CANDIDATES)
        mse_col  = find_col(df, MSE_CANDIDATES)

        col_idx = 1
        if has_loss:
            if loss_col:
                y_raw = df[loss_col].dropna().values
                x_l   = x[:len(y_raw)]
                y_sm  = ema_smooth(y_raw, SMOOTH_WEIGHT)
                fig.add_trace(go.Scatter(x=x_l, y=y_raw, mode="lines",
                                         opacity=0.15, showlegend=False,
                                         line=dict(color=color, width=0.8)),
                              row=1, col=col_idx)
                fig.add_trace(go.Scatter(x=x_l, y=y_sm, mode="lines",
                                         name=label, legendgroup=label,
                                         line=dict(color=color, width=2.5),
                                         showlegend=True),
                              row=1, col=col_idx)
            fig.update_xaxes(title_text="Steps", row=1, col=col_idx)
            fig.update_yaxes(title_text="Loss",  row=1, col=col_idx)
            col_idx += 1

        if has_mse:
            if mse_col:
                mask   = df[mse_col].notna()
                y_raw2 = df.loc[mask, mse_col].values
                x_m    = x[mask.values]

                y_sm2  = ema_smooth(y_raw2, SMOOTH_WEIGHT)
                fig.add_trace(go.Scatter(x=x_m, y=y_raw2, mode="lines",
                                        opacity=0.15, showlegend=False,
                                        line=dict(color=color, width=0.8)),
                            row=1, col=col_idx)
                fig.add_trace(go.Scatter(x=x_m, y=y_sm2, mode="lines",
                                        name=label, legendgroup=label,
                                        line=dict(color=color, width=2.5),
                                        showlegend=False),
                            row=1, col=col_idx)
            fig.update_xaxes(title_text="Steps", row=1, col=col_idx)
            fig.update_yaxes(title_text="MSE",   row=1, col=col_idx)

    fig.update_layout(
        title_text="InternVLA L3 Fine-tuning — Comparison (eps10 vs eps25 vs eps50)",
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                    xanchor="center", x=0.5),
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "loss_combined.png")
    fig.write_image(out_path, scale=2)
    print(f"  Saved: {out_path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot InternVLA training curves")
    parser.add_argument("--base_dir",   default=DEFAULT_BASE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("InternVLA Training Curves Plotter")
    print("=" * 60)

    run_data = {}
    for label, run_name in RUNS.items():
        print(f"\nLoading [{label}] — {run_name}")
        df = load_run(args.base_dir, run_name)
        if df.empty:
            print(f"  [WARN] No data found")
        else:
            step_col = find_col(df, STEP_CANDIDATES)
            loss_col = find_col(df, LOSS_CANDIDATES)
            mse_col  = find_col(df, MSE_CANDIDATES)
            max_step = int(df[step_col].max()) if step_col else len(df)
            print(f"  OK: {len(df)} steps | max_step={max_step}")
            print(f"  loss_col={loss_col} | mse_col={mse_col}")
            print(f"  All columns: {[c for c in df.columns if not c.startswith('_')]}")
        run_data[label] = df

    print("\n" + "=" * 60)
    print("Generating individual plots...")
    for label, df in run_data.items():
        if not df.empty:
            plot_individual(label, df, args.output_dir)

    print("\nGenerating combined plot...")
    plot_combined(run_data, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()