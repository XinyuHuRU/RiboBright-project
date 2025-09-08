#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:27:15 2025

@author: xinyuhu
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.exposure import rescale_intensity
from tqdm import tqdm
import tifffile

# --- Load image with shape handling ---
def load_image_stack(path):
    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()  # could be (3, Z, Y, X) or (Z, 3, Y, X)
        if image.ndim != 4:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        if image.shape[0] == 3:
            ch1, ch2, ch3 = image[0], image[1], image[2]  # shape: [Z, Y, X]
        elif image.shape[1] == 3:
            ch1, ch2, ch3 = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]  # shape: [Z, Y, X]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        return ch1, ch2, ch3  # each: [Z, Y, X]

# --- Compute Manders Coefficients ---
def compute_manders(ch1, ch2):
    ch1 = rescale_intensity(ch1.astype(float))
    ch2 = rescale_intensity(ch2.astype(float))
    mask = (ch1 > 0) | (ch2 > 0)
    ch1 = ch1[mask]
    ch2 = ch2[mask]
    M1 = np.sum(ch1[ch2 > 0]) / np.sum(ch1) if np.sum(ch1) > 0 else 0
    M2 = np.sum(ch2[ch1 > 0]) / np.sum(ch2) if np.sum(ch2) > 0 else 0
    return M1, M2

# --- Analyze folder per experiment ---
def analyze_folder(experiment_path, experiment_label):
    results = []
    for group in ['RB_ER', 'RB_Mito']:
        group_path = os.path.join(experiment_path, group)
        for subdir, _, files in os.walk(group_path):
            for file in files:
                if not file.endswith('.ome.tif') or file.startswith("._"):
                    continue
                filepath = os.path.join(subdir, file)
                try:
                    ch1, ch2, ch3 = load_image_stack(filepath)  # each shape: [Z, Y, X]
                    num_z = ch1.shape[0]
                    m1s, m2s = [], []

                    for z in range(num_z):
                        rb = ch1[z]
                        if group == 'RB_ER':
                            other = ch2[z]  # ER
                        elif group == 'RB_Mito':
                            other = ch3[z]  # Mito
                        else:
                            continue

                        m1, m2 = compute_manders(rb, other)
                        m1s.append(m1)
                        m2s.append(m2)

                    results.append({
                        "Experiment": experiment_label,
                        "Group": group,
                        "Image": file,
                        "M1": np.mean(m1s),
                        "M2": np.mean(m2s)
                    })

                except Exception as e:
                    print(f"Failed: {filepath}\nError: {e}")
    return results

# --- Experiment paths ---
experiment_roots = {
    "RB_XH_013": "/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_013",
    "RB_XH_017": "/Volumes/bifchem/Projects Hansen/Lab Members folders/XH/RiboBright/RB_XH_017"
}

# --- Run analysis ---
all_results = []
for label, path in experiment_roots.items():
    print(f"Analyzing {label}...")
    all_results.extend(analyze_folder(path, label))

df = pd.DataFrame(all_results)
df.to_csv("manders_results.csv", index=False)

# --- Plotting ---
sns.set(style="white", font_scale=1.2)
palette = {"RB_XH_013": "#1f77b4", "RB_XH_017": "#ff7f0e"}

for metric in ["M1", "M2"]:
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Group", y=metric, data=df,
        inner=None, palette="pastel", cut=0
    )
    sns.stripplot(
        x="Group", y=metric, data=df,
        hue="Experiment", palette=palette, dodge=True,
        jitter=True, size=6, edgecolor="gray", linewidth=0.7
    )
    plt.title(f"Manders {metric} Co-localization (Z-averaged per image)")
    plt.ylabel(f"{metric} coefficient")
    plt.xlabel("Group (RB co-localized with)")
    plt.legend(title="Experiment", loc="upper right")
    plt.tight_layout()
    plt.savefig(f"manders_{metric}_violin.png", dpi=300)
    plt.show()


