import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# File path
file_path = "C:\\Users\\PHYS3009\\Desktop\\Boxplot\\Boxplot.xlsx"

# Read the data using openpyxl as the engine
data = pd.read_excel(file_path, engine='openpyxl')

# Drop rows with NaN values
data = data.dropna()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(9,5))

from matplotlib.font_manager import FontProperties
font_properties = FontProperties()
font_properties.set_family('Times New Roman')


from matplotlib.patches import Patch



# Define positions for each column
# positions = list(range(len(data.columns)))
positions = [0, 1, 1.22, 3.59, 3.75, 6.63, 7.03, 8.56, 12.41, 13.19]  # Example custom gaps


# Define colors for each column
color_map = {
    "443": "black",
    "482": "black",
    "561": "black",
    "655": "black",
    "865": "black",
    "475": "white",
    "560": "white",
    "668": "white",
    "717": "white",
    "840": "white"
}

# Set the y-axis range slightly below the minimum and slightly above the maximum
#global_min = data.min().min()
#global_max = data.max().max()
#y_padding = (global_max - global_min) * 0.05
#ax.set_ylim(global_min - y_padding, global_max + y_padding)
ax.set_ylim(0,0.15)
# Create a continuous spectrum background
ymin, ymax = ax.get_ylim()
x = np.linspace(min(positions)-3.25, max(positions)-4.55, 2000)
y = np.linspace(ymin, ymax, 10)
X, Y = np.meshgrid(x, y)
Z = X
cax = ax.imshow(Z, cmap='nipy_spectral', aspect='auto', extent=[min(positions)-3.25, max(positions)-4.55, ymin, ymax], alpha=0.35)

# Loop through each column to create a boxplot with specified colors
for i, col in enumerate(data.columns):
    col_color = color_map.get(str(col), "b")
    ax.boxplot(data[col].dropna().values, positions=[positions[i]], patch_artist=True, boxprops=dict(facecolor=col_color), medianprops=dict(color="grey"))

# Calculate the average of each column and plot a scatter plot
averages = data.mean()
ax.scatter(positions, averages, color='black', zorder=2)
ax.plot(positions, averages, color='black', linestyle='--', zorder=1)

# Set the x-axis labels
ax.set_xticks(positions)
ax.set_xticklabels(data.columns, fontname = 'Times New Roman', fontsize=10, rotation=90)
ax.set_xlim(min(positions)-0.5, max(positions)+0.5)

# Set labels
ax.set_xlabel('Wavelength (nm)', fontname = 'Times New Roman', fontsize=12)
ax.set_ylabel(r'Rrs (sr$^{-1}$)', fontname = 'Times New Roman', fontsize=12)

for label in ax.get_yticklabels():
    label.set_fontproperties(font_properties)
    label.set_fontsize(10)

# Create custom legend handles
legend_elements = [Patch(facecolor='black', edgecolor='black', label='Satellite (Landsat 9 - OLI)'),
                   Patch(facecolor='white', edgecolor='black', label='RPAS (MicaSense RedEdge-MX)'),
                   Line2D([0], [0], marker='o', color='black', label='Outliers', markersize=5, markerfacecolor='black', linestyle='None')]

# Add the legend to the plot
ax.legend(handles=legend_elements, loc='upper right', prop=font_properties)



plt.show()
