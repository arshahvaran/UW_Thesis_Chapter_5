# Open Python Command Prompt
# Navigate to the following directory
# cd C:\Users\PHYS3009\Desktop\Chapter4\Data_Preparation\PCA
# Run the code
# python PCA_Loop.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Load Loop.xlsx to get X1, X2, and Product names
loop_file_path = "C:\\Users\\PHYS3009\\Desktop\\Chapter4\\Data_Preparation\\PCA\\Loop.xlsx"
loop_df = pd.read_excel(loop_file_path)

# Initialize the coefficient for figure size
coef_size = 3

# Initialize the final figure to contain 9 rows and 3 columns of subplots
nrows, ncols = 9, 3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * coef_size, nrows * coef_size))


# Load the main data file
input_file_path = "C:\\Users\\PHYS3009\\Desktop\\Chapter4\\Data_Preparation\\PCA\\TSS_Indexes_Calculated_2.xlsx"
df = pd.read_excel(input_file_path)

# Define the dependent variable
dependent_var = 'TSS_mgL'

# Output directory
output_dir = "C:\\Users\\PHYS3009\\Desktop\\Chapter4\\Data_Preparation\\PCA\\TSS_Indexes"

# Loop through each row in Loop.xlsx
for index, row in loop_df.iterrows():
    product = row['Product']
    X1 = row['X1']
    X2 = row['X2']

    # Define independent variables based on X1, X2
    independent_vars = df.columns[X1:X2]

    # Remove rows with missing values
    df_filtered = df.dropna(subset=[dependent_var] + list(independent_vars))

    # Perform PCA
    X = df_filtered[independent_vars]
    y = df_filtered[dependent_var]
    pca = PCA(n_components=len(independent_vars))
    pca.fit(X)

    # Plot Explained Variance with bar width set to 0.25
    explained_variance_ratio = pca.explained_variance_ratio_ * 100
    axs[index, 0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, width=0.25)
    axs[index, 0].set_ylim(0, 100)
    axs[index, 0].set_title(f"{product} - Explained Variance", fontname="Times New Roman", fontsize="10")
    axs[index, 0].set_xlabel("Principal Components", fontname="Times New Roman")
    axs[index, 0].set_ylabel("Explained Variance (%)", fontname="Times New Roman")
    for tick in axs[index, 0].get_xticklabels() + axs[index, 0].get_yticklabels():
        tick.set_fontname("Times New Roman")

    # Loadings and Contributions
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(independent_vars))], index=independent_vars)

    for j, pc in enumerate(['PC1', 'PC2']):
        contributions = (loadings[pc] ** 2) / (loadings[pc] ** 2).sum() * 100
        contributions_sorted = contributions.sort_values(ascending=False)
        axs[index, j + 1].bar(contributions_sorted.index, contributions_sorted.values, width=0.25)
        axs[index, j + 1].set_ylim(0, 100)
        axs[index, j + 1].set_title(f"{product} - Contributions to {pc}", fontname="Times New Roman", fontsize="10")
        
        # Modify x-axis tick labels to display only the last 5 characters
        new_labels = [label[-5:] for label in contributions_sorted.index]
        axs[index, j + 1].set_xticks(range(len(new_labels)))  # Explicitly set tick positions
        axs[index, j + 1].set_xticklabels(new_labels, fontname="Times New Roman", rotation=45)

        # Set the font for y-axis
        for tick in axs[index, j + 1].get_yticklabels():
            tick.set_fontname("Times New Roman")
            
        axs[index, j + 1].set_xlabel("Original Variables", fontname="Times New Roman")
        axs[index, j + 1].set_ylabel("Contribution (%)", fontname="Times New Roman")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Combined_PCA_Plots.png"))
