#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:32:32 2025

@author: xinyuhu
"""
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Paths
ribo_path = "/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_006/Image/RB"
ribo_mask_path = "/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_006/Image/RB_output"
cd140a_mask_path = "/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_006/Image/CD140a_output"
cd24_mask_path = "/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_006/Image/CD24_output"
cd140a_path = "/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_006/Image/CD140a"
cd24_path = "/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_006/Image/CD24"

output_dir = os.path.abspath("/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_006/Image/HighRB_Assignment_Outputs")
os.makedirs(output_dir, exist_ok=True)

# Parameters
ribo_threshold = 4000
min_area = 12500
overlap_threshold = 0.0000000
cd140a_intensity_threshold = 100
cd24_intensity_threshold = 100
min_high_rb_pixels_per_cell = 200

def get_valid_tif_files(folder):
    return sorted([f for f in os.listdir(folder) if f.endswith(".tif") and not f.startswith("._")])

def has_neighbor(label_id, labeled_mask):
    cell_mask = (labeled_mask == label_id)
    dilated = cv2.dilate(cell_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    border = np.logical_and(dilated, labeled_mask != label_id)
    neighbor_labels = np.unique(labeled_mask[border])
    neighbor_labels = neighbor_labels[(neighbor_labels != 0) & (neighbor_labels != label_id)]
    return len(neighbor_labels) > 0

ribo_files = get_valid_tif_files(ribo_path)

all_cell_counts = {"CD140a": 0, "CD24": 0, "DoubleNeg": 0}
high_rb_cell_counts = {"CD140a": 0, "CD24": 0, "DoubleNeg": 0}

for file in ribo_files:
    base_name = file.replace("_RB.tif", "")

    # Load images
    ribo_img = tiff.imread(os.path.join(ribo_path, file))
    ribo_mask = tiff.imread(os.path.join(ribo_mask_path, base_name + "_RB_mask.tif"))
    cd140a_mask = tiff.imread(os.path.join(cd140a_mask_path, base_name + "_CD140a_mask.tif"))
    cd24_mask = tiff.imread(os.path.join(cd24_mask_path, base_name + "_CD24_mask.tif"))
    cd140a_img = tiff.imread(os.path.join(cd140a_path, base_name + "_CD140a.tif"))
    cd24_img = tiff.imread(os.path.join(cd24_path, base_name + "_CD24.tif"))

    # Thresholded intensity masks
    cd140a_int_thresh_mask = (cd140a_img > cd140a_intensity_threshold).astype(np.uint8)
    cd24_int_thresh_mask = (cd24_img > cd24_intensity_threshold).astype(np.uint8)

    # Label RB cells
    num_rb_labels, labeled_rb_mask = cv2.connectedComponents((ribo_mask > 0).astype(np.uint8))
    rb_cell_labels = list(range(1, num_rb_labels))
    cell_type_map = {}

    for label in rb_cell_labels:
        cell_mask = (labeled_rb_mask == label).astype(np.uint8)
        cell_area = np.count_nonzero(cell_mask)
        if cell_area < min_area:
            continue

        overlap_cd140a = np.count_nonzero(np.logical_and(cell_mask, np.logical_and(cd140a_mask > 0, cd140a_int_thresh_mask > 0)))
        overlap_cd24 = np.count_nonzero(np.logical_and(cell_mask, np.logical_and(cd24_mask > 0, cd24_int_thresh_mask > 0)))

        overlap_cd140a_pct = (overlap_cd140a / cell_area) * 100
        overlap_cd24_pct = (overlap_cd24 / cell_area) * 100

        if overlap_cd140a_pct > overlap_threshold and np.any(cd140a_mask[cell_mask.astype(bool)] > 0):
            cell_type_map[label] = "CD140a"
        elif overlap_cd24_pct > overlap_threshold:
            cell_type_map[label] = "CD24"
        else:
            cell_type_map[label] = "DoubleNeg"

        all_cell_counts[cell_type_map[label]] += 1

    # Identify high RB+ pixels and assign to cells
    ribo_high_exp = (ribo_img > ribo_threshold).astype(np.uint8)
    ribo_high_exp_masked = cv2.bitwise_and(ribo_high_exp, (ribo_mask > 0).astype(np.uint8))
    high_rb_labels = labeled_rb_mask[ribo_high_exp_masked > 0]
    unique, counts = np.unique(high_rb_labels[high_rb_labels > 0], return_counts=True)
    for label_id, count in zip(unique, counts):
        if count >= min_high_rb_pixels_per_cell and label_id in cell_type_map:
            ctype = cell_type_map[label_id]
            high_rb_cell_counts[ctype] += 1

    # Visualization
    cell_type_overlay = np.zeros((ribo_img.shape[0], ribo_img.shape[1], 3), dtype=np.uint8)
    color_map = {
        "CD140a": (31, 119, 180),    # blue
        "CD24": (255, 127, 14),      # orange
        "DoubleNeg": (160, 160, 160) # gray
    }
    for label, ctype in cell_type_map.items():
        mask = (labeled_rb_mask == label)
        cell_type_overlay[mask] = color_map[ctype]

    high_rb_pixels = (ribo_img > ribo_threshold) & (ribo_mask > 0)
    overlay_combined = cell_type_overlay.copy()
    overlay_combined[high_rb_pixels] = [255, 0, 0]  # Red for high RB+

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(ribo_img, cmap='hot')
    axes[0].set_title("RB+ Intensity")
    axes[0].axis('off')

    axes[1].imshow(cell_type_overlay)
    axes[1].set_title("Cell Types")
    axes[1].axis('off')

    rb_only_rgb = np.zeros_like(cell_type_overlay)
    rb_only_rgb[high_rb_pixels] = [255, 0, 0]
    axes[2].imshow(rb_only_rgb)
    axes[2].set_title("High RB+ Pixels")
    axes[2].axis('off')

    axes[3].imshow(overlay_combined)
    axes[3].set_title("Cell Types + High RB+")
    axes[3].axis('off')

    fig.suptitle(f"Visualization for {base_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

# Save summary results
summary_df = pd.DataFrame({
    "Cell Type": ["CD140a", "CD24", "DoubleNeg"],
    "Total Cells": [all_cell_counts["CD140a"], all_cell_counts["CD24"], all_cell_counts["DoubleNeg"]],
    "High RB+ Cells": [high_rb_cell_counts["CD140a"], high_rb_cell_counts["CD24"], high_rb_cell_counts["DoubleNeg"]]
})

summary_df["% High RB+ Cells"] = summary_df["High RB+ Cells"] / summary_df["Total Cells"] * 100

# Reorder rows for plotting: DoubleNeg, CD24, CD140a
summary_df = summary_df.set_index("Cell Type").loc[["DoubleNeg", "CD24", "CD140a"]].reset_index()

# Save to Excel
excel_path = os.path.join(output_dir, "HighRB_CellType_Assignment.xlsx")
summary_df.to_excel(excel_path, index=False)

# Plot summary bar chart
plt.figure(figsize=(8, 6))
bars = plt.bar(summary_df["Cell Type"], summary_df["% High RB+ Cells"], color=["#a0a0a0", "#ff7f0e", "#1f77b4"])
plt.ylabel("% High RB+ Cells")
plt.title("Proportion of High RB+ Cells per Population")
plt.ylim(0, 40)
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords="offset points", ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "HighRB_PerPopulation_BarPlot.svg"))
plt.show()

# --- Plot distribution of all RB pixels ---
plt.figure(figsize=(10, 5))
sns.kdeplot(all_rb_pixel_values, fill=True)
plt.title("RB Pixel Intensity Distribution (All RB pixels)")
plt.xlabel("RB Intensity")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "RB_Intensity_AllPixels_KDE.svg"))
plt.show()
