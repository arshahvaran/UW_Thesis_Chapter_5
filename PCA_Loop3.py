# Open Python Command Prompt
# Navigate to the following directory
# cd C:\Users\PHYS3009\Desktop\Chapter4\Data_Preparation\PCA
# Run the code
# python PCA_Loop3.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import numpy as np

# Function to convert wavelength to RGB color
def wavelength_to_rgb(wavelength):
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R = G = B = 0

    if (wavelength >= 380) and (wavelength < 440):
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif (wavelength >= 440) and (wavelength < 490):
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif (wavelength >= 490) and (wavelength < 510):
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif (wavelength >= 510) and (wavelength < 580):
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif (wavelength >= 580) and (wavelength < 645):
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif (wavelength >= 645) and (wavelength <= 750):
        R = 1.0
        G = 0.0
        B = 0.0

    if (wavelength >= 380) and (wavelength < 420):
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif (wavelength >= 420) and (wavelength < 645):
        factor = 1.0
    elif (wavelength >= 645) and (wavelength <= 750):
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)

    R = round(intensity_max * (R * factor) ** gamma)
    G = round(intensity_max * (G * factor) ** gamma)
    B = round(intensity_max * (B * factor) ** gamma)

    return (R, G, B)

# Define unique colors based on wavelengths
unique_colors = {wavelength: wavelength_to_rgb(wavelength) for wavelength in [443, 475, 482, 560, 561, 655, 668, 717, 840, 865]}
unique_colors[717] = (140, 0, 0)  # Dark red
unique_colors[840] = (100, 0, 0)  # Dark red
unique_colors[865] = (60, 0, 0)  # Very dark red

# Load Loop.xlsx and TSS_Indexes_Calculated_2.xlsx
loop_file_path = "C:\\Users\\PHYS3009\\Desktop\\Chapter4\\Data_Preparation\\PCA\\Loop_TSS.xlsx"
loop_df = pd.read_excel(loop_file_path)
input_file_path = "C:\\Users\\PHYS3009\\Desktop\\Chapter4\\Data_Preparation\\PCA\\TSS_Indexes_Calculated_2.xlsx"
df = pd.read_excel(input_file_path)

# Initialize figure
coef_size = 3
nrows, ncols = 9, 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * coef_size, nrows * coef_size))

chla_fill = '#f6fef8'

# Define dependent variable and output directory
dependent_var = 'TSS_mgL'
output_dir = "C:\\Users\\PHYS3009\\Desktop\\Chapter4\\Data_Preparation\\PCA\\TSS_Indexes"

# Loop through Loop.xlsx
for index, row in loop_df.iterrows():
    product = row['Product']
    X1 = row['X1']
    X2 = row['X2']
    independent_vars = df.columns[X1:X2]
    df_filtered = df.dropna(subset=[dependent_var] + list(independent_vars))
    X = df_filtered[independent_vars]
    y = df_filtered[dependent_var]
    pca = PCA(n_components=len(independent_vars))
    pca.fit(X)
    
    # Explained Variance Plot
    explained_variance_ratio = pca.explained_variance_ratio_ * 100
    axs[index, 0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, width=0.25, color='black')
    axs[index, 0].set_ylim(0, 100)
    axs[index, 0].set_title(f"{product} - Explained Variance", fontname="Times New Roman", fontsize="10")
    axs[index, 0].set_xlabel("Principal Components", fontname="Times New Roman")
    axs[index, 0].set_ylabel("Explained Variance (%)", fontname="Times New Roman")
    axs[index, 0].set_facecolor(tss_fill)
    for tick in axs[index, 0].get_xticklabels() + axs[index, 0].get_yticklabels():
        tick.set_fontname("Times New Roman")
    
    # Loadings and Contributions
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(independent_vars))], index=independent_vars)
    for j, pc in enumerate(['PC1', 'PC2']):
        contributions = (loadings[pc] ** 2) / (loadings[pc] ** 2).sum() * 100
        contributions_sorted = contributions.sort_values(ascending=False)
        bar_colors = []
        for label in contributions_sorted.index:
            try:
                wavelength = int(label.split('_')[-1][:-2])
                color = unique_colors.get(wavelength, (0, 0, 0))
            except ValueError:
                color = (0, 0, 0)
            bar_colors.append(color)
        axs[index, j + 1].bar(contributions_sorted.index, contributions_sorted.values, color=[tuple(np.array(c)/255) for c in bar_colors], edgecolor='black', linewidth=0.5, width=0.25)
        axs[index, j + 1].set_ylim(0, 100)
        axs[index, j + 1].set_title(f"{product} - Contributions to {pc}", fontname="Times New Roman", fontsize="10")
        new_labels = [label[-5:] for label in contributions_sorted.index]
        axs[index, j + 1].set_xticks(range(len(new_labels)))
        axs[index, j + 1].set_xticklabels(new_labels, fontname="Times New Roman", rotation=45)
        for tick in axs[index, j + 1].get_yticklabels():
            tick.set_fontname("Times New Roman")
        axs[index, j + 1].set_xlabel("Original Variables", fontname="Times New Roman")
        axs[index, j + 1].set_ylabel("Contribution (%)", fontname="Times New Roman")
        axs[index, j + 1].set_facecolor(tss_fill)
        

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "TSS_PCA_Plots.png"))
plt.savefig(os.path.join(output_dir, "TSS_PCA_Plots.svg"))