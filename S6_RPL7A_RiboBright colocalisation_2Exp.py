#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 15:37:55 2025

@author: xinyuhu
"""

import numpy as np
import pandas as pd
import os
from tifffile import imread
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import threshold_otsu
from scipy.stats import mannwhitneyu

# -------- Manders' Coefficient Function --------
def manders_coefficient(ch1, ch2, threshold1=None, threshold2=None):
    if threshold1 is None:
        threshold1 = threshold_otsu(ch1)
    if threshold2 is None:
        threshold2 = threshold_otsu(ch2)

    mask1 = ch1 > threshold1
    mask2 = ch2 > threshold2
    overlap = mask1 & mask2

    M1 = np.sum(ch1[overlap]) / np.sum(ch1[mask1]) if np.sum(ch1[mask1]) > 0 else np.nan
    M2 = np.sum(ch2[overlap]) / np.sum(ch2[mask2]) if np.sum(ch2[mask2]) > 0 else np.nan
    return M1, M2

# -------- Path Configuration --------
root_dir = "/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_015/"
experiment_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f)) and f.startswith("Exp")]

# -------- Main Processing --------
results = []

for exp in experiment_folders:
    exp_path = os.path.join(root_dir, exp)
    subfolders = [f for f in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, f))]
    
    print(f"\nProcessing {exp} with {len(subfolders)} subfolders...")
    
    for folder in subfolders:
        folder_path = os.path.join(exp_path, folder)
        ome_files = [f for f in os.listdir(folder_path) if f.endswith('.ome.tif')]
        if len(ome_files) != 1:
            print(f"⚠️ Warning: {folder_path} has {len(ome_files)} tif files, expected 1. Skipping.")
            continue
        file_path = os.path.join(folder_path, ome_files[0])
        print(f"Processing file: {file_path}")

        image = imread(file_path)
        if image.shape[0] == 3:
            ch1, ch2, ch3 = image[0], image[1], image[2]
        elif image.shape[1] == 3:
            ch1, ch2, ch3 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # -------- Manders Coefficient per z-stack and averaging --------
        z_slices = ch1.shape[0]
        M1_12_list, M2_12_list, M1_13_list, M2_13_list = [], [], [], []

        for z in range(z_slices):
            M1_12, M2_12 = manders_coefficient(ch1[z], ch2[z])
            M1_13, M2_13 = manders_coefficient(ch1[z], ch3[z])
            M1_12_list.append(M1_12)
            M2_12_list.append(M2_12)
            M1_13_list.append(M1_13)
            M2_13_list.append(M2_13)

        M1_12_avg = np.nanmean(M1_12_list)
        M2_12_avg = np.nanmean(M2_12_list)
        M1_13_avg = np.nanmean(M1_13_list)
        M2_13_avg = np.nanmean(M2_13_list)

        results.append({
            "Experiment": exp,
            "Folder": folder,
            "File": ome_files[0],
            "M1_Ch1_Ch2": M1_12_avg,
            "M2_Ch1_Ch2": M2_12_avg,
            "M1_Ch1_Ch3": M1_13_avg,
            "M2_Ch1_Ch3": M2_13_avg
        })

# -------- Save Raw Results --------
df_results = pd.DataFrame(results)
raw_output_csv = os.path.join(root_dir, "Manders_Coefficients_Summary.csv")
df_results.to_csv(raw_output_csv, index=False)
print(f"\n✅ Manders analysis completed. Results saved to: {raw_output_csv}")

# -------- Mann-Whitney U Test and Fold Change --------
stat_summary = []

for metric in ["M1_Ch1_Ch2", "M1_Ch1_Ch3", "M2_Ch1_Ch2", "M2_Ch1_Ch3"]:
    pass  # keep for fold change calculation

# Perform pairwise tests: Ch2 vs Ch3 for M1 and M2
for m_label, col1, col2 in [("M1", "M1_Ch1_Ch2", "M1_Ch1_Ch3"),
                             ("M2", "M2_Ch1_Ch2", "M2_Ch1_Ch3")]:
    x = df_results[col1].dropna()
    y = df_results[col2].dropna()
    stat, p = mannwhitneyu(x, y, alternative='two-sided')
    
    fold_median = np.median(y) / np.median(x) if np.median(x) != 0 else np.nan
    fold_mean = np.mean(y) / np.mean(x) if np.mean(x) != 0 else np.nan
    
    stat_summary.append({
        "Metric": m_label,
        "Group1": col1,
        "Group2": col2,
        "p_value": p,
        "FoldChange_Median": fold_median,
        "FoldChange_Mean": fold_mean
    })

df_stats = pd.DataFrame(stat_summary)
stat_output_csv = os.path.join(root_dir, "Manders_Statistics_Summary.csv")
df_stats.to_csv(stat_output_csv, index=False)
print(f"✅ Statistical summary saved to: {stat_output_csv}")

# -------- Violin Plot with Box and Colored Points --------
plot_df = pd.DataFrame({
    'Value': np.concatenate([df_results["M1_Ch1_Ch2"].dropna(), df_results["M1_Ch1_Ch3"].dropna(),
                             df_results["M2_Ch1_Ch2"].dropna(), df_results["M2_Ch1_Ch3"].dropna()]),
    'Group': (['M1_Ch1_Ch2'] * df_results["M1_Ch1_Ch2"].dropna().shape[0] +
              ['M1_Ch1_Ch3'] * df_results["M1_Ch1_Ch3"].dropna().shape[0] +
              ['M2_Ch1_Ch2'] * df_results["M2_Ch1_Ch2"].dropna().shape[0] +
              ['M2_Ch1_Ch3'] * df_results["M2_Ch1_Ch3"].dropna().shape[0]),
    'Experiment': (list(df_results.loc[~df_results["M1_Ch1_Ch2"].isna(), "Experiment"]) +
                   list(df_results.loc[~df_results["M1_Ch1_Ch3"].isna(), "Experiment"]) +
                   list(df_results.loc[~df_results["M2_Ch1_Ch2"].isna(), "Experiment"]) +
                   list(df_results.loc[~df_results["M2_Ch1_Ch3"].isna(), "Experiment"]))
})

plt.figure(figsize=(10, 6))
sns.violinplot(data=plot_df, x="Group", y="Value", inner="box", palette='Set2')
sns.stripplot(data=plot_df, x="Group", y="Value", hue="Experiment",
              dodge=True, size=5, jitter=True, palette='tab10', alpha=0.8)

# Annotate p-values
plt.text(0.5, 1.02, f"p={df_stats.loc[df_stats['Metric']=='M1','p_value'].values[0]:.3e}", ha='center', va='bottom')
plt.text(2.5, 1.02, f"p={df_stats.loc[df_stats['Metric']=='M2','p_value'].values[0]:.3e}", ha='center', va='bottom')

plt.ylim(0.4, 1.05)
plt.ylabel("Manders Coefficient")
plt.title("Manders Coefficient (Averaged Z-stacks) with p-values (Mann-Whitney U)", fontsize=14)
plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

violin_output = os.path.join(root_dir, "ViolinPlot_Manders_Ch2_vs_Ch3_colored.svg")
plt.savefig(violin_output, dpi=300)
print(f"📸 Violin plot saved: {violin_output}")
plt.show()
