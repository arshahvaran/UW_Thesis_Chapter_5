# Open Python Command Prompt
# Navigate to the following directory
# cd C:\Users\PHYS3009\Desktop\Final_Plots\
# Run the code
# python Final_Plots.py




import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = 'Times New Roman'

# Function to calculate 95% confidence interval for a given set of x and y data points
def confidence_interval(x, y, y_pred):
    residuals = y - y_pred
    s_err = np.sum(np.power(residuals, 2))
    s_err = np.sqrt(s_err / (len(y) - 2))
    t_score = stats.t.ppf(0.975, len(y) - 2)
    ci = t_score * s_err * np.sqrt(1/len(y) + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    return ci

# Define the input and output directories
input_dir = "C:\\Users\\PHYS3009\\Desktop\\Final_Plots"
output_dir = "C:\\Users\\PHYS3009\\Desktop\\Final_Plots\\Output"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the equations for Y2 for each dataset
equations = {
    'Chla_L9': lambda x: 4.719 * (x ** 12.33) + 0.6885,
    'Chla_REMX': lambda x: 70.28 * x + 0.3177,
    'TSS_L9': lambda x: 510.3 * (x ** 2) - 43.94 * x + 4.041,
    'TSS_REMX': lambda x: 5139 * (x ** 2) - 163.4 * x + 1.526
}

upper_equations = {
    'Chla_L9': lambda x: 7.142 * (x ** 15.99) + 2.677,
    'Chla_REMX': lambda x: 77.7 * x + 0.4601,
    'TSS_L9': lambda x: 708.3 * (x ** 2) + 33.57 * x + 6.74,
    'TSS_REMX': lambda x: 7323 * (x ** 2) - 23.59 * x + 3.55
}

lower_equations = {
    'Chla_L9': lambda x: 2.295 * (x ** 8.667) - 1.3,
    'Chla_REMX': lambda x: 62.86 * x + 0.1752,
    'TSS_L9': lambda x: 312.3 * (x ** 2) - 121.5 * x + 1.342,
    'TSS_REMX': lambda x: 2956 * (x ** 2) - 303.2 * x - 0.4973
}

# Define the given R2 and RMSE values for the right plots
given_metrics = {
    'Chla_REMX': {'R2': 0.97, 'RMSE': 0.17, 'MAE':0.22 , 'n':16},
    'Chla_L9': {'R2': 0.75, 'RMSE': 4.34, 'MAE':2.62 , 'n':52},
    'TSS_REMX': {'R2': 0.86, 'RMSE': 1.47, 'MAE':1.16 , 'n':17},
    'TSS_L9': {'R2': 0.90, 'RMSE': 4.18, 'MAE':3.08 , 'n':56}
}

# Background colors
bg_colors = {'Chla': '#f6fef8', 'TSS': '#f9e8d8'}

# Ordered keys for plotting
ordered_keys = ['Chla_REMX', 'Chla_L9', 'TSS_REMX', 'TSS_L9']

# Read data and calculate Y2
data_dict = {}
for filename in os.listdir(input_dir):
    if filename.endswith('.xlsx'):
        key = filename.split('.')[0]
        df = pd.read_excel(os.path.join(input_dir, filename))
        x_values = df.iloc[:, 1].values
        df['Y2'] = equations[key](x_values)
        data_dict[key] = df

# Initialize the 8-in-one figure
fig, axs = plt.subplots(4, 2, figsize=(15, 24))

# Loop through each dataset to plot
for i, key in enumerate(ordered_keys):
    df = data_dict[key]
    y1 = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    y2 = df['Y2'].values

    # Generate smoother set of X points for plotting the Y2 curve
    x_smooth = np.linspace(min(x), max(x), 500)
    y2_smooth = equations[key](x_smooth)

    # Background color
    bg_color = bg_colors['Chla'] if 'Chla' in key else bg_colors['TSS']

      # Scatter plot of Y1 vs X along with Y2 equation line
    axs[i, 1].scatter(x, y1, label='Data Points', s=30 ,c='black', marker='x')
    axs[i, 1].plot(x_smooth, y2_smooth, label='Regression Model', color='black', linewidth=2)
    
    
    # Calculate R^2 and RMSE for the plots on the right (Y1 vs X)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y1)
    r_squared_right = r_value ** 2
    rmse_right = np.sqrt(np.mean((y1 - (intercept + slope * x)) ** 2))

    # Add upper and lower 95% confidence bounds for the plots on the right
    upper_y2_smooth = upper_equations[key](x_smooth)
    lower_y2_smooth = lower_equations[key](x_smooth)
    axs[i, 1].fill_between(x_smooth, lower_y2_smooth, upper_y2_smooth, color='gray', alpha=0.2, label='95% CI')


    if 'Chla' in key:
        axs[i, 1].set_ylabel('Response: Chl-a (μg/L)', fontsize=12, font='Times New Roman')
        if 'REMX' in key:
            axs[i, 1].set_title('Regression Model for Chl-a', fontsize=12, font='Times New Roman', fontweight='bold')
            axs[i, 1].text(0.03, 0.66, f'Sensor: RedEdge-MX', fontsize=12, font='Times New Roman', transform=axs[i, 1].transAxes)
            axs[i, 1].set_xlabel('Predictor: R', fontsize=12, font='Times New Roman')
        else:
            axs[i, 1].set_title('Regression Model for Chl-a', fontsize=12, font='Times New Roman', fontweight='bold')
            axs[i, 1].text(0.03, 0.66, f'Sensor: OLI-2', fontsize=12, font='Times New Roman', transform=axs[i, 1].transAxes)
            axs[i, 1].set_xlabel('Predictor: R/G', fontsize=12, font='Times New Roman')
    if 'TSS' in key:
        axs[i, 1].set_ylabel('Response: TSS (mg/L)', fontsize=12, font='Times New Roman')
        if 'REMX' in key:
            axs[i, 1].set_title('Regression Model for TSS', fontsize=12, font='Times New Roman', fontweight='bold')
            axs[i, 1].text(0.03, 0.66, f'Sensor: RedEdge-MX', fontsize=12, font='Times New Roman', transform=axs[i, 1].transAxes)
            axs[i, 1].set_xlabel('Predictor: G', fontsize=12, font='Times New Roman')
        else:
            axs[i, 1].set_title('Regression Model for TSS', fontsize=12, font='Times New Roman', fontweight='bold')
            axs[i, 1].text(0.03, 0.66, f'Sensor: OLI-2', fontsize=12, font='Times New Roman', transform=axs[i, 1].transAxes)
            axs[i, 1].set_xlabel('Predictor: NIR', fontsize=12, font='Times New Roman')
            


    axs[i, 1].legend()
    axs[i, 1].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

       
    # Add additional legend box for the plots on the right
    unit_right = "μg/L" if "Chla" in key else "mg/L"
    legend2_text_right = f'R² = {given_metrics[key]["R2"]:.2f}\nRMSE ({unit_right}) = {given_metrics[key]["RMSE"]}\nMAE ({unit_right}) = {given_metrics[key]["MAE"]}\nn = {given_metrics[key]["n"]}'
    axs[i, 1].text(0.03, 0.70, legend2_text_right, transform=axs[i, 1].transAxes, fontsize=12, font='Times New Roman')

    axs[i, 1].set_facecolor(bg_color)

    # Set identical axis limits for all plots on the right
    y_max_right = max(max(y1), max(y2))
    x_max_right = max(x)
    axs[i, 1].set_ylim([-0.02 * y_max_right, y_max_right*1.02])
    #axs[i, 1].set_xlim([-0.001 * x_max_right, x_max_right*1.01])

    # Individual axis limits for each plot on the left (a little beyond the max)
    max_limit_left = max(max(y1), max(y2))
    extended_limit_high = max_limit_left * 1.05
    extended_limit_low = -0.02 *max_limit_left
    axs[i, 0].set_ylim([extended_limit_low, extended_limit_high])
    axs[i, 0].set_xlim([extended_limit_low, extended_limit_high])

    # Draw the 1:1 line that covers the whole graph (in black color)
    axs[i, 0].plot([extended_limit_low, extended_limit_high], [extended_limit_low, extended_limit_high], label='1:1 line', linewidth=1, color='black', zorder=1, linestyle='-')
    
    # Scatter plot of Y2 vs Y1 with 1:1 line
    axs[i, 0].scatter(y1, y2, label='Data Points', s=30 ,c='black', marker='x', zorder=2)

    # Calculate R^2 and RMSE for the plots on the left (Y2 vs Y1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y1, y2)
    r_squared_left = r_value ** 2
    rmse_left = np.sqrt(np.mean((y2 - (intercept + slope * y1)) ** 2))
    mae_left = np.mean(np.abs(y2 - (intercept + slope * y1)))

     # Add trendline
    slope, intercept, r_value, p_value, std_err = stats.linregress(y1, y2)
    y_pred = intercept + slope * y1
    axs[i, 0].plot(y1, y_pred, label='Trendline', color='black', linewidth=2)
    
    # 95% confidence interval for the plots on the left (around the trendline)
    ci_left = confidence_interval(y1, y2, y_pred)
    axs[i, 0].fill_between(sorted(y1), sorted(y_pred - ci_left), sorted(y_pred + ci_left), color='gray', alpha=0.2, label='95% CI')

    slope_intercept_text = f'Slope = {slope:.2f}\nIntercept = {intercept:.2f}'
    axs[i, 0].text(0.76, 0.03, slope_intercept_text, transform=axs[i, 0].transAxes, fontsize=12, font='Times New Roman')

    if 'Chla' in key:
        axs[i, 0].set_xlabel('Observed Chl-aₒ (μg/L)', fontsize=12, font='Times New Roman')
        axs[i, 0].set_ylabel('Predicted Chl-aₚ (μg/L)', fontsize=12, font='Times New Roman')
        if 'REMX' in key:
            axs[i, 0].set_title('Observed vs Predicted Chl-a', fontsize=12, font='Times New Roman', fontweight='bold')
            axs[i, 0].text(0.03, 0.58, f'Sensor: RedEdge-MX\nPredictor: R', fontsize=12, font='Times New Roman', transform=axs[i, 0].transAxes)
        else:
            axs[i, 0].set_title('Observed vs Predicted Chl-a', fontsize=12, font='Times New Roman', fontweight='bold')
            axs[i, 0].text(0.03, 0.58, f'Sensor: OLI-2\nPredictor: R/G', fontsize=12, font='Times New Roman', transform=axs[i, 0].transAxes)

    if 'TSS' in key:
        axs[i, 0].set_xlabel('Observed TSSₒ (mg/L)', fontsize=12, font='Times New Roman')
        axs[i, 0].set_ylabel('Predicted TSSₚ (mg/L)', fontsize=12, font='Times New Roman')
        if 'REMX' in key:
            axs[i, 0].set_title('Observed vs Predicted TSS', fontsize=12, font='Times New Roman', fontweight='bold')
            axs[i, 0].text(0.03, 0.58, f'Sensor: RedEdge-MX\nPredictor: G', fontsize=12, font='Times New Roman', transform=axs[i, 0].transAxes)
        else:
            axs[i, 0].set_title('Observed vs Predicted TSS', fontsize=12, font='Times New Roman', fontweight='bold')
            axs[i, 0].text(0.03, 0.58, f'Sensor: OLI-2\nPredictor: NIR', fontsize=12, font='Times New Roman', transform=axs[i, 0].transAxes)
            


    axs[i, 0].legend()
    axs[i, 0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # Add additional legend box for the plots on the left
    unit_left = "μg/L" if "Chla" in key else "mg/L"
    legend2_text_left = f'R² = {r_squared_left:.2f}\nRMSE ({unit_left}) = {rmse_left:.2f}\nMAE ({unit_left}) = {mae_left:.2f}\nn = {given_metrics[key]["n"]}'
    axs[i, 0].text(0.03, 0.66, legend2_text_left, transform=axs[i, 0].transAxes, fontsize=12, font='Times New Roman')

    axs[i, 0].set_facecolor(bg_color)
    
    # Make the subplot square in shape
    axs[i, 0].set_aspect('equal', 'box')

    # Set font properties
    for ax in [axs[i, 0], axs[i, 1]]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)
            item.set_fontname('Times New Roman')

# Save the plots
plt.tight_layout()
plt.subplots_adjust(left=-0.02)  # Adjusts the left space to 10% of the figure width

plt.savefig(os.path.join(output_dir, 'final_plots_with_95CI_and_trendline_corrected_v2.png'), dpi=1500)
plt.savefig(os.path.join(output_dir, 'final_plots_with_95CI_and_trendline_corrected_v2.svg'), dpi=1500)

# Show the plots
#plt.show()


