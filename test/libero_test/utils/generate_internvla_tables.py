#!/usr/bin/env python3
"""
Generate InternVLA-M1 syntactic variation tables from evaluation logs.

This script groups logs by task and variation, computes mean and standard
deviation across seeds, and writes Excel tables for L1 or L2 evaluation levels.
"""

import re
import math
import argparse
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side


# Task mapping: command text → task number
TASK_MAPPING = {
    "Pull the middle layer of the drawer": 1,
    "Draw out the middle drawer of the cabinet": 1,
    "Slide out the middle drawer of the cabinet": 1,
    "Ease out the middle drawer of the cabinet": 1,
    
    "Set the bowl on the stove": 2,
    "Position the bowl on the stove": 2,
    "Rest the bowl on the stove": 2,
    "Lay the bowl on the stove": 2,
    
    "Place the wine bottle on the top of the cabinet": 3,
    "Position the wine bottle on the top of the cabinet": 3,
    "Rest the wine bottle on the top of the cabinet": 3,
    "Lay the wine bottle on the top of the cabinet": 3,
    
    "Pull the top layer of the drawer and place the bowl inside": 4,
    "Draw out the top drawer of the cabinet and position the bowl inside": 4,
    "Slide out the top drawer of the cabinet and rest the bowl inside": 4,
    "Ease out the top drawer of the cabinet and lay the bowl inside": 4,
    
    "Place the bowl on the top of the cabinet": 5,
    "Position the bowl on the top of the cabinet": 5,
    "Rest the bowl on the top of the cabinet": 5,
    "Lay the bowl on the top of the cabinet": 5,
    
    "Move the plate to the front of the stove": 6,
    "Slide the plate to the front of the stove": 6,
    "Shift the plate to the front of the stove": 6,
    "Guide the plate to the front of the stove": 6,
    
    "Place the cream cheese on the bowl": 7,
    "Put the cream cheese on the bowl": 7,
    "Position the cream cheese in the bowl": 7,
    "Position the cream cheese on the bowl": 7,
    "Rest the cream cheese in the bowl": 7,
    "Rest the cream cheese on the bowl": 7,
    "Lay the cream cheese in the bowl": 7,
    "Lay the cream cheese on the bowl": 7,
    
    "Switch on the stove": 8,
    "Ignite the stove": 8,
    "Start the stove": 8,
    "Power on the stove": 8,
    
    "Place the bowl on the plate": 9,
    "Position the bowl on the plate": 9,
    "Rest the bowl on the plate": 9,
    "Lay the bowl on the plate": 9,
    
    "Place the wine bottle on the rack": 10,
    "Position the wine bottle on the rack": 10,
    "Rest the wine bottle on the rack": 10,
    "Lay the wine bottle on the rack": 10,
    
    # L2 variations
    "The middle layer of the drawer needs to be opened": 1,
    "The middle layer of the drawer should be opened": 1,
    "Let the middle layer of the drawer be opened": 1,
    "The middle drawer layer must be opened": 1,
    "The middle drawer of the cabinet needs to be opened": 1,
    "The middle drawer of the cabinet should be opened": 1,
    "Let the middle drawer of the cabinet be opened": 1,
    "The middle drawer of the cabinet must be opened": 1,
    
    "The stove needs to have the bowl on it": 2,
    "The bowl should be put on the stove": 2,
    "Let the bowl be put on the stove": 2,
    "The bowl must be put on the stove": 2,
    
    "Top of the cabinet needs to have the wine bottle on it": 3,
    "The wine bottle should be put on the top of the cabinet": 3,
    "Let the wine bottle be put on top of the cabinet": 3,
    "Let the wine bottle be put on the top of the cabinet": 3,
    "The wine bottle must be put on the cabinet's top": 3,
    "The wine bottle must be put on the top of the cabinet": 3,
    
    "The top layer of the drawer needs to be opened and the bowl needs to be put inside": 4,
    "The top layer of the drawer should be opened and the bowl should be put inside": 4,
    "Let the top drawer be opened and the bowl be put inside": 4,
    "The top layer of the drawer must be opened and the bowl must be put inside": 4,
    "The top drawer of the cabinet needs to be opened and the bowl needs to be put inside": 4,
    "The top drawer of the cabinet should be opened and the bowl should be put inside": 4,
    "Let the top drawer of the cabinet be opened and the bowl be put inside": 4,
    "The top drawer of the cabinet must be opened and the bowl must be put inside": 4,
    
    "The top of the cabinet needs to have the bowl on it": 5,
    "The bowl should be put on the top of the cabinet": 5,
    "Let the bowl be put on top of the cabinet": 5,
    "Let the bowl be put on the top of the cabinet": 5,
    "The bowl must be put on the cabinet's top": 5,
    "The bowl must be put on the top of the cabinet": 5,
    
    "The space in front of the stove needs to have the plate in it": 6,
    "The plate should be pushed to the front of the stove": 6,
    "Let the plate be pushed to the front of the stove": 6,
    "The plate must be pushed to the front of the stove": 6,
    "The plate must be pushed to the stove's front": 6,
    
    "The cream cheese needs to be put on the bowl": 7,
    "The cream cheese should be put on the bowl": 7,
    "Let the cream cheese be put on the bowl": 7,
    "The cream cheese must be put on the bowl": 7,
    
    "The stove needs to be turned on": 8,
    "The stove should be turned on": 8,
    "Let the stove be turned on": 8,
    "The stove must be turned on": 8,
    
    "The plate needs to have the bowl on it": 9,
    "The bowl should be put on the plate": 9,
    "Let the bowl be put on the plate": 9,
    "The bowl must be put on the plate": 9,
    
    "The rack needs to be filled with the wine bottle in it": 10,
    "The wine bottle should be put on the rack": 10,
    "Let the wine bottle be put on the rack": 10,
    "The wine bottle must be put on the rack": 10,
}

# Variation type mapping
VARIATION_TYPE = {
    1: {
        "Pull the middle layer of the drawer": "Original",
        "Draw out the middle drawer of the cabinet": "V1",
        "Slide out the middle drawer of the cabinet": "V2",
        "Ease out the middle drawer of the cabinet": "V3",
        "The middle layer of the drawer needs to be opened": "Original",
        "The middle layer of the drawer should be opened": "V1",
        "Let the middle layer of the drawer be opened": "V2",
        "The middle drawer layer must be opened": "V3",
        "The middle drawer of the cabinet needs to be opened": "Original",
        "The middle drawer of the cabinet should be opened": "V1",
        "Let the middle drawer of the cabinet be opened": "V2",
        "The middle drawer of the cabinet must be opened": "V3",
    },
    2: {
        "Set the bowl on the stove": "Original",
        "Position the bowl on the stove": "V1",
        "Rest the bowl on the stove": "V2",
        "Lay the bowl on the stove": "V3",
        "The stove needs to have the bowl on it": "Original",
        "Let the bowl be put on the stove": "V1",
        "The bowl should be put on the stove": "V1",
        "The bowl must be put on the stove": "V2",
    },
    3: {
        "Place the wine bottle on the top of the cabinet": "Original",
        "Position the wine bottle on the top of the cabinet": "V1",
        "Rest the wine bottle on the top of the cabinet": "V2",
        "Lay the wine bottle on the top of the cabinet": "V3",
        "Top of the cabinet needs to have the wine bottle on it": "Original",
        "The wine bottle should be put on the top of the cabinet": "V1",
        "Let the wine bottle be put on top of the cabinet": "V2",
        "Let the wine bottle be put on the top of the cabinet": "V2",
        "The wine bottle must be put on the cabinet's top": "V3",
        "The wine bottle must be put on the top of the cabinet": "V3",
    },
    4: {
        "Pull the top layer of the drawer and place the bowl inside": "Original",
        "Draw out the top drawer of the cabinet and position the bowl inside": "V1",
        "Slide out the top drawer of the cabinet and rest the bowl inside": "V2",
        "Ease out the top drawer of the cabinet and lay the bowl inside": "V3",
        "The top layer of the drawer needs to be opened and the bowl needs to be put inside": "Original",
        "The top layer of the drawer should be opened and the bowl should be put inside": "V1",
        "Let the top drawer be opened and the bowl be put inside": "V2",
        "The top layer of the drawer must be opened and the bowl must be put inside": "V3",
        "The top drawer of the cabinet needs to be opened and the bowl needs to be put inside": "Original",
        "The top drawer of the cabinet should be opened and the bowl should be put inside": "V1",
        "Let the top drawer of the cabinet be opened and the bowl be put inside": "V2",
        "The top drawer of the cabinet must be opened and the bowl must be put inside": "V3",
    },
    5: {
        "Place the bowl on the top of the cabinet": "Original",
        "Position the bowl on the top of the cabinet": "V1",
        "Rest the bowl on the top of the cabinet": "V2",
        "Lay the bowl on the top of the cabinet": "V3",
        "The top of the cabinet needs to have the bowl on it": "Original",
        "The bowl should be put on the top of the cabinet": "V1",
        "Let the bowl be put on top of the cabinet": "V2",
        "Let the bowl be put on the top of the cabinet": "V2",
        "The bowl must be put on the cabinet's top": "V3",
        "The bowl must be put on the top of the cabinet": "V3",
    },
    6: {
        "Move the plate to the front of the stove": "Original",
        "Slide the plate to the front of the stove": "V1",
        "Shift the plate to the front of the stove": "V2",
        "Guide the plate to the front of the stove": "V3",
        "The space in front of the stove needs to have the plate in it": "Original",
        "The plate should be pushed to the front of the stove": "V1",
        "Let the plate be pushed to the front of the stove": "V2",
        "The plate must be pushed to the front of the stove": "V3",
        "The plate must be pushed to the stove's front": "V3",
    },
    7: {
        "Place the cream cheese on the bowl": "Original",
        "Put the cream cheese on the bowl": "Original",
        "Position the cream cheese in the bowl": "V1",
        "Position the cream cheese on the bowl": "V1",
        "Rest the cream cheese in the bowl": "V2",
        "Rest the cream cheese on the bowl": "V2",
        "Lay the cream cheese in the bowl": "V3",
        "Lay the cream cheese on the bowl": "V3",
        "The cream cheese needs to be put on the bowl": "Original",
        "The cream cheese should be put on the bowl": "V1",
        "Let the cream cheese be put on the bowl": "V2",
        "The cream cheese must be put on the bowl": "V3",
    },
    8: {
        "Switch on the stove": "Original",
        "Ignite the stove": "V1",
        "Start the stove": "V2",
        "Power on the stove": "V3",
        "The stove needs to be turned on": "Original",
        "The stove should be turned on": "V1",
        "Let the stove be turned on": "V2",
        "The stove must be turned on": "V3",
    },
    9: {
        "Place the bowl on the plate": "Original",
        "Position the bowl on the plate": "V1",
        "Rest the bowl on the plate": "V2",
        "Lay the bowl on the plate": "V3",
        "The plate needs to have the bowl on it": "Original",
        "The bowl should be put on the plate": "V1",
        "Let the bowl be put on the plate": "V2",
        "The bowl must be put on the plate": "V3",
    },
    10: {
        "Place the wine bottle on the rack": "Original",
        "Position the wine bottle on the rack": "V1",
        "Rest the wine bottle on the rack": "V2",
        "Lay the wine bottle on the rack": "V3",
        "The rack needs to be filled with the wine bottle in it": "Original",
        "The wine bottle should be put on the rack": "V1",
        "Let the wine bottle be put on the rack": "V2",
        "The wine bottle must be put on the rack": "V3",
    },
}

# Baseline original values for InternVLA with std (from user evaluation data)
ORIGINAL_VALUES = {
    "L1": {
        1: (92.7, 1.2),     # "Pull the middle layer of the drawer" - Mean SR%: 92,7% ± 1,2%
        2: (98.0, 2.0),     # "Set the bowl on the stove" - Mean SR%: 98,0% ± 2,0%
        3: (92.0, 2.0),     # "Place the wine bottle on top of cabinet" - Mean SR%: 92,0% ± 2,0%
        4: (77.3, 6.1),     # "Pull top layer and place bowl" - Mean SR%: 77,3% ± 6,1%
        5: (99.3, 1.2),     # "Place bowl on top of cabinet" - Mean SR%: 99,3% ± 1,2%
        6: (75.3, 1.2),     # "Move plate to front of stove" - Mean SR%: 75,3% ± 1,2%
        7: (85.3, 4.6),     # "Place cream cheese on bowl" - Mean SR%: 85,3% ± 4,6%
        8: (100.0, 0.0),    # "Switch on stove" - Mean SR%: 100% ± 0,0%
        9: (97.3, 2.3),     # "Place bowl on plate" - Mean SR%: 97,3% ± 2,3%
        10: (96.7, 1.2),    # "Place wine bottle on rack" - Mean SR%: 96,7% ± 1,2%
    },
    "L2": {
        1: (92.7, 1.2),     # "Middle layer needs to be opened" - Mean SR%: 92,7% ± 1,2%
        2: (100.0, 0.0),    # "Stove needs to have bowl" - Mean SR%: 100% ± 0,0%
        3: (92.7, 1.2),     # "Top of cabinet needs wine bottle" - Mean SR%: 92,7% ± 1,2%
        4: (65.3, 5.0),     # "Top layer needs opened, bowl inside" - Mean SR%: 65,3% ± 5,0%
        5: (80.0, 2.0),     # "Top of cabinet needs bowl" - Mean SR%: 80,0% ± 2,0%
        6: (11.3, 1.2),     # "Space in front of stove needs plate" - Mean SR%: 11,3% ± 1,2%
        7: (70.7, 2.3),     # "Cream cheese needs to be put on bowl" - Mean SR%: 70,7% ± 2,3%
        8: (100.0, 0.0),    # "Stove needs to be turned on" - Mean SR%: 100% ± 0,0%
        9: (74.7, 1.2),     # "Plate needs to have bowl" - Mean SR%: 74,7% ± 1,2%
        10: (96.0, 0.0),    # "Rack needs wine bottle" - Mean SR%: 96,0% ± 0,0%
    },
}

# Task descriptions for titles
TASK_DESCRIPTIONS = {
    1: "Open the middle layer of the drawer",
    2: "Put the bowl on the stove",
    3: "Put the wine bottle on the top of the cabinet",
    4: "Open the top layer of the drawer and put the bowl inside",
    5: "Put the bowl on the top of the cabinet",
    6: "Push the plate to the front of the stove",
    7: "Put the cream cheese on the bowl",
    8: "Turn on the stove",
    9: "Put the bowl on the plate",
    10: "Put the wine bottle on the rack",
}


def extract_task_index_from_filename(filepath):
    """Extract zero-based task index from a log filename.

    Args:
        filepath (str): Path to one log file.

    Returns:
        int | None: Parsed task index, or `None` when the pattern is missing.
    """
    filename = Path(filepath).name
    match = re.search(r'_task(\d+)_seed', filename)
    if match:
        return int(match.group(1))  # 0-based index
    return None


def is_single_task_file(filepath):
    """Check whether a log filename targets a single task.

    Args:
        filepath (str): Path to one log file.

    Returns:
        bool: True when the filename follows single-task naming.
    """
    filename = Path(filepath).name
    return "task" in filename and "_seed" in filename and filename.count("task") >= 2


def parse_txt_evaluation(filepath):
    """Parse one evaluation log and extract variation-level task success rates.

    Args:
        filepath (str): Path to one evaluation log file.

    Returns:
        dict[str, dict]: Command text to parsed rate metadata.
    """
    results = {}
    task_index_offset = None
    
    filename = Path(filepath).name
    if is_single_task_file(filepath):
        task_index_offset = extract_task_index_from_filename(filepath)
    
    # Extract seed number
    seed_match = re.search(r'_seed(\d+)', filename)
    seed_num = int(seed_match.group(1)) if seed_match else None
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by each "Testing VERSION:" section
    version_blocks = re.split(r'(?=Testing VERSION:)', content)
    
    for block in version_blocks:
        if 'Testing VERSION:' not in block:
            continue
        
        # Extract variation command
        cmd_match = re.search(r'Variation Command: (.+?)\n', block)
        if not cmd_match:
            continue
        command_text = cmd_match.group(1).strip()
        
        rate = None
        
        # First try: Extract from VERSION vX RESULTS section
        rate_match = re.search(r'Success Rate: ([\d.]+)%', block)
        if rate_match:
            rate = float(rate_match.group(1))
        else:
            # Fallback: Extract last success rate from episode data
            success_matches = re.findall(r'Total successes: \d+ \(([\d.]+)%\)', block)
            if success_matches:
                rate = float(success_matches[-1])
        
        if rate is not None:
            results[command_text] = {
                "rate": rate,
                "task_offset": task_index_offset,
                "seed": seed_num,
                "filename": filename,
            }
    
    return results


def classify_variation(task_text, task_offset=None):
    """Map command text to task id and variation label.

    Args:
        task_text (str): Command text from the log.
        task_offset (int | None): Optional task index inferred from filename.

    Returns:
        tuple[int | None, str | None]: `(task_num, variant_type)` pair.
    """
    task_num = None
    
    if task_offset is not None:
        task_num = task_offset + 1  # Convert 0-based to 1-based
    else:
        task_num = TASK_MAPPING.get(task_text)
    
    if task_num is None:
        return None, None
    
    if task_num in VARIATION_TYPE:
        variant_type = VARIATION_TYPE[task_num].get(task_text)
        if variant_type:
            return task_num, variant_type
    
    return task_num, "Unknown"


def compute_stats(rates):
    """Compute sample mean and standard deviation for rate values.

    Args:
        rates (list[float]): Task success rate values in percentage.

    Returns:
        tuple[float, float]: `(mean, std)` in percentage points.
    """
    if not rates:
        return 0.0, 0.0
    
    mean = sum(rates) / len(rates)
    
    if len(rates) == 1:
        return mean, 0.0
    
    variance = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
    std = math.sqrt(variance)
    
    return mean, std


def format_value(mean, std):
    """Format one mean ± std value using percentage style.

    Args:
        mean (float): Mean task success rate.
        std (float): Standard deviation.

    Returns:
        str: Formatted value, or `N/A` when both inputs are zero.
    """
    if mean == 0 and std == 0:
        return "N/A"
    formatted = f"{mean:.1f}% ± {std:.1f}%"
    return formatted.replace(".", ",")


def aggregate_results(file_list, level="L1"):
    """Aggregate parsed log results across files and seeds.

    Args:
        file_list (list[str]): Input log files.
        level (str): Level label kept for compatibility with caller options.

    Returns:
        dict: Nested task/variant/seed structure with task success rate values.
    """
    # Structure: {task: {variant: {seed: rate}}}
    aggregated = {}
    
    for filepath in file_list:
        results = parse_txt_evaluation(filepath)
        task_offset = extract_task_index_from_filename(filepath) if is_single_task_file(filepath) else None
        
        for task_text, data in results.items():
            rate = data["rate"]
            file_task_offset = data.get("task_offset", task_offset)
            seed_num = data.get("seed")
            
            task_num, variant = classify_variation(task_text, file_task_offset)
            
            if task_num is None or variant is None:
                continue
            
            if task_num not in aggregated:
                aggregated[task_num] = {}
            if variant not in aggregated[task_num]:
                aggregated[task_num][variant] = {}
            
            # Only add if we don't already have this seed for this variant
            if seed_num not in aggregated[task_num][variant]:
                aggregated[task_num][variant][seed_num] = rate
    
    return aggregated


def create_task_table(task_num, aggregated_data, level="L1", model_name="InternVLA-M1"):
    """Create one Excel workbook for a single task table.

    Args:
        task_num (int): Task id in `[1, 10]`.
        aggregated_data (dict): Aggregated output from `aggregate_results`.
        level (str): Syntactic level label (`L1` or `L2`).
        model_name (str): Model label printed in the table.

    Returns:
        openpyxl.Workbook: Workbook containing one task sheet.
    """
    wb = Workbook()
    ws = wb.active
    
    task_desc = TASK_DESCRIPTIONS.get(task_num, f"Task {task_num}")
    title = f"{level} Verb Substitution Evaluation on Task {task_num} \"{task_desc}\""
    ws.title = f"Task {task_num}"
    
    # Title (merged cells)
    ws.merge_cells('A1:F1')
    title_cell = ws['A1']
    title_cell.value = title
    title_cell.font = Font(bold=True, size=11)
    title_cell.alignment = Alignment(horizontal='left', wrap_text=True)
    
    # Header row
    header = [
        "Model",
        f"Original {level}\nMean SR% ± std%",
        "V1 Mean\nMean SR% ± std%",
        "V2 Mean\nMean SR% ± std%",
        "V3 Mean\nMean SR% ± std%",
        "Mean SR% ± std%\n(Original+V1+V2+V3)",
    ]
    ws.append([])  # Empty row
    ws.append(header)
    
    # Format header
    for cell in ws[3]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    
    # Data row
    task_data = aggregated_data.get(task_num, {})
    original_mean, original_std = ORIGINAL_VALUES[level].get(task_num, (0.0, 0.0))
    
    row = [model_name]
    task_means = []
    
    for variant in ["Original", "V1", "V2", "V3"]:
        if variant == "Original":
            # Original value with its std
            row.append(format_value(original_mean, original_std))
            task_means.append(original_mean)
        else:
            variant_data = task_data.get(variant, {})
            if variant_data:
                rates = list(variant_data.values())
                mean, std = compute_stats(rates)
                row.append(format_value(mean, std))
                task_means.append(mean)
            else:
                row.append("N/A")

    if len(task_means) == 4:
        mean4, std4 = compute_stats(task_means)
        row.append(format_value(mean4, std4))
    else:
        row.append("N/A")
    
    ws.append(row)
    
    # Format data row
    for cell in ws[4]:
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Adjust column widths
    ws.column_dimensions['A'].width = 15
    for col in ['B', 'C', 'D', 'E', 'F']:
        ws.column_dimensions[col].width = 18
    
    return wb


def create_summary_table(aggregated_data, level="L1", model_name="InternVLA-M1"):
    """Create one summary workbook with all tasks in one sheet.

    Args:
        aggregated_data (dict): Aggregated output from `aggregate_results`.
        level (str): Syntactic level label (`L1` or `L2`).
        model_name (str): Model label printed in the table title.

    Returns:
        openpyxl.Workbook: Workbook with one summary sheet.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    
    title = f"{level} Verb Substitution Evaluation - {model_name}"
    ws.merge_cells('A1:F1')
    title_cell = ws['A1']
    title_cell.value = title
    title_cell.font = Font(bold=True, size=12)
    title_cell.alignment = Alignment(horizontal='center', wrap_text=True)
    
    # Header row
    header = ["Task", "Original", "V1", "V2", "V3", "Mean (4 valori)"]
    ws.append([])
    ws.append(header)
    
    # Format header
    for cell in ws[3]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        cell.alignment = Alignment(horizontal='center', vertical='center')
    
    # Data rows
    for task_num in sorted(range(1, 11)):
        task_data = aggregated_data.get(task_num, {})
        original_mean, original_std = ORIGINAL_VALUES[level].get(task_num, (0.0, 0.0))
        
        row = [f"Task {task_num}"]
        task_means = []
        
        for variant in ["Original", "V1", "V2", "V3"]:
            if variant == "Original":
                row.append(format_value(original_mean, original_std))
                task_means.append(original_mean)
            else:
                variant_data = task_data.get(variant, {})
                if variant_data:
                    rates = list(variant_data.values())
                    mean, std = compute_stats(rates)
                    row.append(format_value(mean, std))
                    task_means.append(mean)
                else:
                    row.append("N/A")

        if len(task_means) == 4:
            mean4, std4 = compute_stats(task_means)
            row.append(format_value(mean4, std4))
        else:
            row.append("N/A")
        
        ws.append(row)
    
    # Adjust column widths
    ws.column_dimensions['A'].width = 12
    for col in ['B', 'C', 'D', 'E', 'F']:
        ws.column_dimensions[col].width = 16
    
    return wb


def main():
    """Parse CLI arguments and generate task or summary Excel tables.

    Returns:
        None: Writes output files and prints progress.
    """
    parser = argparse.ArgumentParser(description="Generate InternVLA evaluation tables")
    parser.add_argument("--files", nargs="+", required=True, help="Log files to process")
    parser.add_argument("--level", default="L1", choices=["L1", "L2"], help="Evaluation level")
    parser.add_argument("--model", default="InternVLA-M1", help="Model name")
    parser.add_argument("--output", default="internvla_tables.xlsx", help="Output Excel file")
    parser.add_argument("--task", type=int, help="Specific task to generate table for (1-10)")
    parser.add_argument("--summary", action="store_true", help="Generate summary table only")
    
    args = parser.parse_args()
    
    # Aggregate results
    print(f"Processing {len(args.files)} files...")
    aggregated = aggregate_results(args.files, level=args.level)
    
    if args.task:
        # Generate table for specific task
        wb = create_task_table(args.task, aggregated, level=args.level, model_name=args.model)
        wb.save(args.output)
        print(f"✓ Task {args.task} table saved to {args.output}")
    elif args.summary:
        # Generate summary table
        wb = create_summary_table(aggregated, level=args.level, model_name=args.model)
        wb.save(args.output)
        print(f"✓ Summary table saved to {args.output}")
    else:
        # Generate all task tables in separate sheets
        wb = Workbook()
        wb.remove(wb.active)
        
        for task_num in range(1, 11):
            task_wb = create_task_table(task_num, aggregated, level=args.level, model_name=args.model)
            ws = task_wb.active
            
            # Add to main workbook
            new_ws = wb.create_sheet(f"Task {task_num}")
            for row in ws.iter_rows(values_only=True):
                new_ws.append(row)
            
            # Copy formatting
            for i, row in enumerate(ws.iter_rows(), start=1):
                for j, cell in enumerate(row, start=1):
                    new_cell = new_ws.cell(row=i, column=j)
                    if cell.font:
                        new_cell.font = Font(bold=cell.font.bold, size=cell.font.size,
                                           color=cell.font.color)
                    if cell.fill:
                        new_cell.fill = PatternFill(start_color=cell.fill.start_color,
                                                   end_color=cell.fill.end_color,
                                                   fill_type=cell.fill.fill_type)
                    if cell.alignment:
                        new_cell.alignment = Alignment(horizontal=cell.alignment.horizontal,
                                                      vertical=cell.alignment.vertical,
                                                      wrap_text=cell.alignment.wrap_text)
        
        # Add summary sheet
        summary_wb = create_summary_table(aggregated, level=args.level, model_name=args.model)
        ws = summary_wb.active
        new_ws = wb.create_sheet("Summary", 0)
        for row in ws.iter_rows(values_only=True):
            new_ws.append(row)
        
        # Copy formatting for summary
        for i, row in enumerate(ws.iter_rows(), start=1):
            for j, cell in enumerate(row, start=1):
                new_cell = new_ws.cell(row=i, column=j)
                if cell.font:
                    new_cell.font = Font(bold=cell.font.bold, size=cell.font.size,
                                       color=cell.font.color)
                if cell.fill:
                    new_cell.fill = PatternFill(start_color=cell.fill.start_color,
                                               end_color=cell.fill.end_color,
                                               fill_type=cell.fill.fill_type)
                if cell.alignment:
                    new_cell.alignment = Alignment(horizontal=cell.alignment.horizontal,
                                                  vertical=cell.alignment.vertical,
                                                  wrap_text=cell.alignment.wrap_text)
        
        wb.save(args.output)
        print(f"✓ All task tables saved to {args.output}")
    
    print(f"✓ Aggregated data from {len(aggregated)} tasks")


if __name__ == "__main__":
    main()
