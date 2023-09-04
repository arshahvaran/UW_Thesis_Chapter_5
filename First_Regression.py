# Open Python Command Prompt
# Navigate to the following directory
# cd C:\Users\PHYS3009\Desktop\Chapter4\Data_Preparation
# Run the code
# python First_Regression.py


import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to calculate custom metrics based on provided formulas
def custom_metrics(y_true, y_pred):
    n = len(y_true)
    
    # Calculate RMSLE (Root Mean Square Log Error)
    rmsle = np.sqrt(np.sum((y_true - y_pred)**2) / n)
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate Bias
    bias = np.mean(y_true - y_pred)
    
    # Calculate R^2 (Coefficient of Determination)
    mean_y_true = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y_true)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return rmsle, mae, bias, r2

# Define file paths
base_directory = r"C:\Users\PHYS3009\Desktop\Chapter4\Data_Preparation"
input_files = ["Chla_Indexes_Calculated_2.xlsx", "TSS_Indexes_Calculated_2.xlsx"]
output_directory = r"C:\Users\PHYS3009\Desktop\Chapter4\Data_Preparation\Output1"

# Loop through each input file
for file_name in input_files:
    file_path = os.path.join(base_directory, file_name)
    df = pd.read_excel(file_path)
    
    dependent_var = df.columns[0]
    results_list = []
    
    # Loop through each independent variable
    for independent_var in df.columns[1:]:
        filtered_df = df[[independent_var, dependent_var]].dropna()
        
        if len(filtered_df) > 1:
            X = filtered_df[[independent_var]]
            y = filtered_df[dependent_var]
            
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            
            rmsle, mae, bias, r2 = custom_metrics(y, y_pred)
            
            results_list.append({
                'Headers': independent_var,
                'RMSLE': rmsle,
                'MAE': mae,
                'BIAS': bias,
                'R2': r2
            })
            
    results_df = pd.DataFrame(results_list)
    
    # Save the results to an Excel file
    output_file_name = file_name.replace(".xlsx", "_Output.xlsx")
    output_file_path = os.path.join(output_directory, output_file_name)
    results_df.to_excel(output_file_path, index=False)
