# Instal the required libraries using the code below:
# pip install pandas numpy scikit-learn openpyxl
# Open Python Command Prompt
# Navigate to the following directory
# cd C:\Users\PHYS3009\Desktop\Metrics
# Run the code
# python Metrics.py


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Define the fit equations
fit_equations = {
    'TSS_REMX_R': lambda x: 702.3 * (x ** 1.288),
    'TSS_L9_I5': lambda x: 0.9468 * (x ** 2) - 3.391 * x + 5.592,
    'TSS_REMX_G': lambda x: 5139 * (x ** 2) - 163.4 * x + 1.526,
    'Chla_REMX_G': lambda x: 587.1 * (x ** 2) - 0.02983 * x + 0.4194,
    'TSS_L9_NIR': lambda x: 510.3 * (x ** 2) - 43.94 * x + 4.041,
    'Chla_REMX_R': lambda x: 70.28 * x + 0.3177,
    'TSS_REMX_RE': lambda x: -7933 * (x ** 2) + 601.5 * x - 0.8251
}

# Initialize an empty DataFrame to store the metrics
metrics_df = pd.DataFrame(columns=['File', 'RMSE', 'R2', 'MAE', 'BIAS'])

# Loop through all the files and fit equations
input_directory = "C:\\Users\\PHYS3009\\Desktop\\Metrics"  # Replace with your directory
output_file_path = os.path.join(input_directory, 'output_metrics.xlsx')

for file_name, equation in fit_equations.items():
    # Read the Excel file
    file_path = os.path.join(input_directory, f"{file_name}.xlsx")
    df = pd.read_excel(file_path)
    
    # Extract Y and X columns (assuming they are the first and the second columns)
    y_true = df.iloc[:, 0].values
    x_values = df.iloc[:, 1].values
    
    # Calculate the predicted y values using the fit equation
    y_pred = equation(x_values)
    
    # Calculate RMSE, R2, MAE, and BIAS
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    
    # Append the metrics to the DataFrame
    new_row = pd.DataFrame({
        'File': [file_name],
        'RMSE': [rmse],
        'R2': [r2],
        'MAE': [mae],
        'BIAS': [bias]
    })
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

# Save the metrics DataFrame to an Excel file
metrics_df.to_excel(output_file_path, index=False)
