# Open Python Command Prompt
# Navigate to the following directory
# cd C:\Users\PHYS3009\Desktop\Chapter4\Data_Preparation\Heatmaps
# Run the code
# python Heatmaps.py




import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to create a heatmap with specific customizations and lines
def create_custom_heatmap(df, row_labels, title, color_palette, output_file, plot_type, fig_height=8):
    plt.figure(figsize=(8, fig_height))
    sns.set(font="Times New Roman", font_scale=1.1)

    # Customize the label based on plot_type
    label_text = r'$R^2$' + f" ({plot_type})"

    ax = sns.heatmap(df, annot=True, cmap=color_palette, vmin=0, vmax=1, fmt=".2f", linewidths=.5,
                     annot_kws={"size": 10}, cbar_kws={'label': label_text, 'shrink': 1.0})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontname="Times New Roman")
    ax.set_xticklabels(['Level 1' if label == 'Level 1.1' else label for label in df.columns], fontname="Times New Roman")

    ax.set_yticklabels(row_labels, rotation=0, fontname="Times New Roman")

    ax.axvline(x=1, color='k', linewidth=2)
    ax.axvline(x=len(df.columns) - 1, color='k', linewidth=2)
    ax.axhline(y=len(df.index) - 1, color='k', linewidth=2)

    plt.xlabel('', fontname="Times New Roman")
    plt.ylabel('', fontname="Times New Roman")

    plt.tight_layout()
    plt.savefig(output_file, format='svg')
    plt.close()

# Output directory
output_directory = "C:\\Users\\PHYS3009\\Desktop\\Chapter4\\Data_Preparation\\Heatmaps"

# Creating directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Reading the Excel files and "R2" sheets
chla_file_path = f"{output_directory}\\Chla_Indexes_Calculated_2_Output_2.xlsx"
tss_file_path = f"{output_directory}\\TSS_Indexes_Calculated_2_Output_2.xlsx"

chla_df = pd.read_excel(chla_file_path, sheet_name='R2')
tss_df = pd.read_excel(tss_file_path, sheet_name='R2')

# Preparing the data
tss_df_clean = tss_df.drop(columns=["Unnamed: 0"]).apply(pd.to_numeric, errors='coerce')
chla_df_clean = chla_df.drop(columns=["Unnamed: 0"]).apply(pd.to_numeric, errors='coerce')

# Renaming second "Level 1" column
tss_df_clean = tss_df_clean.rename(columns={"Level 1.1": "Level 1"})
chla_df_clean = chla_df_clean.rename(columns={"Level 1.1": "Level 1"})

# Extracting row labels
tss_row_labels = tss_df['Unnamed: 0']
chla_row_labels = chla_df['Unnamed: 0']

# Custom figure height
tss_fig_height = 5  # You can adjust this as needed

# Creating the heatmaps
create_custom_heatmap(tss_df_clean, tss_row_labels, 'TSS R2 Heatmap', 'YlOrBr', f"{output_directory}\\TSS_R2_Heatmap_Custom.svg", 'TSS', fig_height=tss_fig_height)
create_custom_heatmap(chla_df_clean, chla_row_labels, 'Chl-a R2 Heatmap', 'YlGn', f"{output_directory}\\Chla_R2_Heatmap_Custom.svg", 'Chl-a')
