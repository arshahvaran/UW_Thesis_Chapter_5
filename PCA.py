# Open Python Command Prompt
# Navigate to the following directory
# cd C:\Users\PHYS3009\Desktop\Chapter4\Data_Preparation\PCA
# Run the code
# python PCA.py

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Load the Excel file
input_file_path = "C:\\Users\\PHYS3009\\Desktop\\Chapter4\\Data_Preparation\\PCA\\TSS_Indexes_Calculated_2.xlsx"
df = pd.read_excel(input_file_path)

# Define the dependent and independent variables
dependent_var = 'TSS_mgL'
independent_vars = df.columns[40:45]  # Change these indices if you want to use different columns for independent variables

# Remove rows with missing values in the specified columns
df_filtered = df.dropna(subset=[dependent_var] + list(independent_vars))

# Perform PCA
X = df_filtered[independent_vars]
y = df_filtered[dependent_var]

pca = PCA(n_components=len(independent_vars))
principal_components = pca.fit_transform(X)

# Save the explained variance ratio plot
output_dir = "C:\\Users\\PHYS3009\\Desktop\\Chapter4\\Data_Preparation\\PCA\\TSS_Indexes"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

explained_variance_ratio = pca.explained_variance_ratio_ * 100
plt.figure(figsize=(10, 6))  # Set the size similar to the second set of plots
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5)
plt.ylim(0, 100)
plt.ylabel('Explained Variance (%)', fontname="Times New Roman", fontsize=12)
plt.xlabel('Principal Components', fontname="Times New Roman", fontsize=12)
plt.title('Explained Variance by Each Principal Component', fontname="Times New Roman")
plt.xticks(rotation=0, ha='center', fontsize=10, fontname="Times New Roman")  # Set x-tick label font
plt.yticks(fontsize=10, fontname="Times New Roman")  # Set y-tick label font
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Explained_Variance.png'))

# Save variable contributions to each principal component
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(independent_vars))], index=independent_vars)
loadings['Abs_Max'] = loadings.abs().max(axis=1)

# Generate plots only for the first two principal components
for i, pc in enumerate(loadings.columns[:2]):  # Limit to PC1 and PC2
    contributions = (loadings[pc]**2) / (loadings[pc]**2).sum() * 100
    contributions_sorted = contributions.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    contributions_sorted.plot(kind='bar')
    plt.xticks(rotation=45, ha='right', fontsize=10, fontname="Times New Roman")
    plt.yticks(fontsize=10, fontname="Times New Roman")
    plt.ylim(0, 100)
    plt.xlabel('Original Variables', fontname="Times New Roman", fontsize=12)
    plt.ylabel('Contribution (%)', fontname="Times New Roman", fontsize=12)
    plt.title(f'Variable Contributions to {pc}', fontname="Times New Roman")
    plt.tight_layout()  # Adjust layout to make sure labels are visible
    plt.savefig(os.path.join(output_dir, f'Variable_Contributions_to_{pc}.png'))


# Save the PCA results and loadings to Excel
output_file_path = os.path.join(output_dir, 'PCA_Results_and_Loadings.xlsx')

# Create a DataFrame to hold the PCA results
df_pca = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(len(independent_vars))])
df_pca[dependent_var] = y.reset_index(drop=True)

# Save to Excel
with pd.ExcelWriter(output_file_path) as writer:
    df_pca.to_excel(writer, sheet_name='PCA_Results', index=False)
    loadings.to_excel(writer, sheet_name='Loadings')

print(f"PCA results, plots, and loadings saved to {output_dir}")
