#!/usr/bin/env python3
"""
Compute L3 Syntactic Generalisation Results with Fine-Tuning.
Output: formatted Excel file (syntactic_results.xlsx)

Usage:
    python compute_syntactic_results.py
    python compute_syntactic_results.py --base_dir /path/to/results
"""

import os
import re
import glob
import argparse
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_BASE_DIR = (
    "/home/A.CARDAMONE7/outputs/results/libero_goal/syntactic_command_variation/internvla-m1_l3_finetuned"
)

VARIANTS = {
    "10eps": "10eps_50k_steps",
    "25eps": "25eps_50k_steps",
    "50eps": "50eps_50k_steps",
}

LEVELS = ["l1", "l2", "l3", "default"]

TASK_SR_PATTERN = re.compile(r"Task SR:\s*([0-9.]+)")
FILENAME_PATTERN = re.compile(r".*seed(\d+)_task(\d+)-l\d+\.txt$", re.IGNORECASE)

# ──────────────────────────────────────────────────────────────────────────────
# Styles (defined once, reused everywhere)
# ──────────────────────────────────────────────────────────────────────────────
TITLE_FONT    = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
HDR_FONT      = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
DATA_FONT     = Font(name="Calibri", size=11)
NA_FONT       = Font(name="Calibri", size=11, italic=True, color="888888")
FOOTER_FONT   = Font(name="Calibri", size=9, italic=True, color="666666")

TITLE_FILL    = PatternFill("solid", fgColor="1F3864")   # dark navy
HDR_FILL      = PatternFill("solid", fgColor="2E75B6")   # mid blue
ROW_FILL_A    = PatternFill("solid", fgColor="EBF3FB")   # light blue stripe
ROW_FILL_B    = PatternFill("solid", fgColor="FFFFFF")   # white

CENTER  = Alignment(horizontal="center", vertical="center")
LEFT    = Alignment(horizontal="left",   vertical="center", indent=1)

THIN   = Side(style="thin",   color="B0C4DE")
THICK  = Side(style="medium", color="2E75B6")

def inner_border():
    return Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def extract_task_sr(filepath):
    with open(filepath, "r", errors="replace") as f:
        content = f.read()
    m = TASK_SR_PATTERN.search(content)
    return float(m.group(1)) if m else None


def get_seed_task_key(filename):
    m = FILENAME_PATTERN.match(filename)
    return (int(m.group(1)), int(m.group(2))) if m else None


def collect_sr_values(level_dir):
    files = sorted(glob.glob(os.path.join(level_dir, "*.txt")))
    seen = {}
    for fp in files:
        key = get_seed_task_key(os.path.basename(fp)) or os.path.basename(fp)
        seen[key] = fp

    sr_values, missing = [], []
    for key, fp in seen.items():
        sr = extract_task_sr(fp)
        if sr is not None:
            sr_values.append(sr * 100.0)
        else:
            missing.append(fp)

    if missing:
        print(f"  [WARNING] No Task SR in {len(missing)} file(s):")
        for f in missing[:5]:
            print(f"    {f}")
    return sr_values


# ──────────────────────────────────────────────────────────────────────────────
# Excel builder
# ──────────────────────────────────────────────────────────────────────────────
def build_excel(results, output_path):
    wb = Workbook()

    # Calcola baseline globale: media di tutte le medie SR
    all_means = []
    for variant in results.values():
        for mean_sr, std_sr, n in variant.values():
            if not np.isnan(mean_sr):
                all_means.append(mean_sr)
    
    if all_means:
        baseline_mean = np.mean(all_means)
        baseline_std = np.std(all_means, ddof=1) if len(all_means) > 1 else 0.0
        baseline_display = f"{baseline_mean:.2f}% ± {baseline_std:.2f}%"
    else:
        baseline_display = "N/A"

    # ── Sheet 1: Summary table ──────────────────────────────────────────────
    ws = wb.active
    ws.title = "Results"

    ws.column_dimensions["A"].width = 3
    ws.column_dimensions["B"].width = 28
    ws.column_dimensions["C"].width = 20
    ws.column_dimensions["D"].width = 20
    ws.column_dimensions["E"].width = 20
    ws.column_dimensions["F"].width = 20
    ws.row_dimensions[1].height = 8

    # Title row (row 2)
    ws.merge_cells("B2:F2")
    t = ws["B2"]
    t.value     = "L3 Syntactic Generalisation — Fine-Tuning Results"
    t.font      = TITLE_FONT
    t.fill      = TITLE_FILL
    t.alignment = CENTER
    ws.row_dimensions[2].height = 30

    # Header row (row 3)
    headers = ["Fine-tuned Model", "L1  Mean SR% ± Std%", "L2  Mean SR% ± Std%", "L3  Mean SR% ± Std%", "Baseline Mean SR% ± Std%"]
    for col, h in enumerate(headers, start=2):
        c = ws.cell(row=3, column=col, value=h)
        c.font      = HDR_FONT
        c.fill      = HDR_FILL
        c.alignment = CENTER
        c.border    = inner_border()
    ws.row_dimensions[3].height = 22

    # Data rows (rows 4-6)
    model_labels = {
        "10eps": "InternVLA-M1 — 10 Episodes",
        "25eps": "InternVLA-M1 — 25 Episodes",
        "50eps": "InternVLA-M1 — 50 Episodes",
    }
    fills = [ROW_FILL_A, ROW_FILL_B, ROW_FILL_A]

    for i, (variant, label) in enumerate(model_labels.items()):
        row = 4 + i
        fill = fills[i]
        ws.row_dimensions[row].height = 20

        lc = ws.cell(row=row, column=2, value=label)
        lc.font = Font(name="Calibri", size=11, bold=True)
        lc.fill = fill
        lc.alignment = LEFT
        lc.border = inner_border()

        for j, level in enumerate(LEVELS):
            mean_sr, std_sr, n = results[variant][level]
            col = 3 + j
            c = ws.cell(row=row, column=col)
            c.fill   = fill
            c.border = inner_border()
            if np.isnan(mean_sr):
                c.value     = "N/A"
                c.font      = NA_FONT
                c.alignment = CENTER
            else:
                c.value     = f"{mean_sr:.2f}% ± {std_sr:.2f}%"
                c.font      = DATA_FONT
                c.alignment = CENTER

        # Aggiungi colonna baseline
        c_baseline = ws.cell(row=row, column=6)
        c_baseline.fill   = fill
        c_baseline.border = inner_border()
        c_baseline.value     = baseline_display
        c_baseline.font      = DATA_FONT
        c_baseline.alignment = CENTER

    # ── Sheet 2: Raw data per (variant, level) ─────────────────────────────
    ws2 = wb.create_sheet("Raw Data")
    ws2.column_dimensions["A"].width = 3
    ws2.column_dimensions["B"].width = 28
    ws2.column_dimensions["C"].width = 12
    ws2.column_dimensions["D"].width = 14
    ws2.column_dimensions["E"].width = 14
    ws2.column_dimensions["F"].width = 14
    ws2.column_dimensions["G"].width = 18

    ws2.row_dimensions[1].height = 8
    ws2.merge_cells("B2:G2")
    t2 = ws2["B2"]
    t2.value     = "Raw Statistics per Variant × Level"
    t2.font      = TITLE_FONT
    t2.fill      = TITLE_FILL
    t2.alignment = CENTER
    ws2.row_dimensions[2].height = 28

    raw_headers = ["Variant", "Level", "N files", "Mean SR (%)", "Std SR (%)", "Baseline Mean SR% ± Std%"]
    for col, h in enumerate(raw_headers, start=2):
        c = ws2.cell(row=3, column=col, value=h)
        c.font      = HDR_FONT
        c.fill      = HDR_FILL
        c.alignment = CENTER
        c.border    = inner_border()
    ws2.row_dimensions[3].height = 22

    raw_row = 4
    for variant, label in model_labels.items():
        for level in LEVELS:
            mean_sr, std_sr, n = results[variant][level]
            fill = ROW_FILL_A if raw_row % 2 == 0 else ROW_FILL_B
            ws2.row_dimensions[raw_row].height = 18
            row_vals = [
                label, level.upper(), n,
                round(mean_sr, 4) if not np.isnan(mean_sr) else "N/A",
                round(std_sr,  4) if not np.isnan(std_sr)  else "N/A",
                baseline_display,
            ]
            for col, val in enumerate(row_vals, start=2):
                c = ws2.cell(row=raw_row, column=col, value=val)
                c.font      = DATA_FONT
                c.fill      = fill
                c.alignment = CENTER if col > 2 else LEFT
                c.border    = inner_border()
                if col in (5, 6) and isinstance(val, float):
                    c.number_format = "0.00"
            raw_row += 1

    # Footer
    for ws_obj in [ws, ws2]:
        last = ws_obj.max_row + 2
        fc = ws_obj.cell(
            row=last, column=2,
            value="Source: InternVLA-M1 syntactic variation evaluation — libero_goal benchmark"
        )
        fc.font = FOOTER_FONT

    wb.save(output_path)
    print(f"\nExcel saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",     default=DEFAULT_BASE_DIR)
    parser.add_argument("--output_xlsx",  default="syntactic_results.xlsx")
    args = parser.parse_args()

    print(f"Base directory: {args.base_dir}\n")

    results = {}
    for variant_label, variant_folder in VARIANTS.items():
        variant_dir = os.path.join(args.base_dir, variant_folder)
        results[variant_label] = {}
        print(f"{'='*55}")
        print(f"Variant: {variant_label}  ({variant_folder})")
        print(f"{'='*55}")

        for level in LEVELS:
            level_dir = os.path.join(variant_dir, level)
            if not os.path.isdir(level_dir):
                print(f"  [{level.upper()}] Directory not found: {level_dir}")
                results[variant_label][level] = (float("nan"), float("nan"), 0)
                continue

            sr_values = collect_sr_values(level_dir)
            n = len(sr_values)
            if n == 0:
                print(f"  [{level.upper()}] No valid files found.")
                results[variant_label][level] = (float("nan"), float("nan"), 0)
                continue

            mean_sr = np.mean(sr_values)
            std_sr  = np.std(sr_values, ddof=1) if n > 1 else 0.0
            results[variant_label][level] = (mean_sr, std_sr, n)
            print(f"  [{level.upper()}] n={n:3d} | Mean={mean_sr:6.2f}% ± {std_sr:5.2f}%")
        print()

    build_excel(results, args.output_xlsx)


if __name__ == "__main__":
    main()