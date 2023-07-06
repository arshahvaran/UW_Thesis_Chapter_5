# Open Python Command Prompt
# In Command Prompt navigate to the following directory (if the BoxPlot_InSitu_Data fodler is located in Desktop):
# cd C:\Users\PHYS3009\Desktop\BoxPlot_InSitu_Data
# In Command Prompt install the following libraries:
# pip install pandas
# pip install matplotlib
# pip install openpyxl
# pip install seaborn
# Or if you want to run the code on VScode Terminal, uncomment the above installations, then press Ctrl+Shift+P in VSCode, type "Select Default Profile", then select "Command Prompt" and then proceses with the following line
# In Command Prompt run the following code in the terminal:
# python BoxPlot_InSitu_Data.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Load the dataset
df = pd.read_excel("Data.xlsx", engine='openpyxl')

# Create matplotlib figure
fig = plt.figure(figsize=(10,6))

# Create matplotlib axes
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# Create seaborn boxplots
box1 = sns.boxplot(x="Site_Number", y="Chl_A_ugL", data=df, color='green', ax=ax1, width=0.35, fliersize=6, linewidth=1,  flierprops = dict(markerfacecolor = 'g', marker='d', markersize=8))
box2 = sns.boxplot(x="Site_Number", y="TSS_Corrected_mg_L", data=df, color='peru', ax=ax2, width=0.35, fliersize=6, linewidth=1, flierprops = dict(markerfacecolor = 'peru', marker='d', markersize=8))

# Set gridlines
ax1.grid(False)
ax2.grid(False)

# Set axis labels
ax1.set_xlabel("Site Number", fontname='Times New Roman', fontsize=12)
ax1.set_ylabel("Chl-a (ug/L)", fontname='Times New Roman', fontsize=12, color='green')
ax2.set_ylabel("TSS (mg/L)", fontname='Times New Roman', fontsize=12, color='peru')

# Set tick labels to Times New Roman
for label in ax1.get_xticklabels():
    label.set_fontname('Times New Roman')
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')
for label in ax2.get_yticklabels():
    label.set_fontname('Times New Roman')

# Add legend
green_patch = mpatches.Patch(color='green', label='Chl-a', alpha=0.7)
brown_patch = mpatches.Patch(color='peru', label='TSS', alpha=0.7)
green_diamond = Line2D([0], [0], marker='D', color='w', label='Chl-a Outliers', markerfacecolor='darkgreen', markersize=8)
brown_diamond = Line2D([0], [0], marker='D', color='w', label='TSS Outliers', markerfacecolor='saddlebrown', markersize=8)
green_line = Line2D([0], [0], color='#006400', linewidth=1.5, linestyle='-', label='Chl-a Medians')
brown_line = Line2D([0], [0], color='#8B4513', linewidth=1.5, linestyle='-', label='TSS Medians')

plt.legend(handles=[green_patch, brown_patch, green_diamond, brown_diamond, green_line, brown_line], loc='upper left', prop={'size': 8, 'family': 'Times New Roman'})

# Set line style and color and also patch transparency for Chl-a
for i, patch in enumerate(ax1.artists):
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
    for j in range(i*6,i*6+6):
        line = ax1.lines[j]
        line.set_color('darkgreen')
        line.set_mfc('darkgreen')
        line.set_mec('darkgreen')
        line.set_linewidth(1.5)
        line.set_linestyle('-')

# Set line style and color and also patch transparency for Chl-a
for i, patch in enumerate(ax2.artists):
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
    for j in range(i*6,i*6+6):
        line = ax2.lines[j]
        line.set_color('saddlebrown')
        line.set_mfc('saddlebrown')
        line.set_mec('saddlebrown')
        line.set_linewidth(1.5)
        line.set_linestyle('-')

plt.show()
