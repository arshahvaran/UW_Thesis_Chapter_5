# Instal the required libraries using the code below:
# pip install pandas numpy scikit-learn openpyxl
# Open Python Command Prompt
# Navigate to the following directory
# cd C:\Users\PHYS3009\Desktop\CV_Polynomial
# Run the code
# python CV_Polynomial.py


from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import os

# Function to perform regression
def perform_regression(X, y, degree):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Perform the regression
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Make predictions
    y_pred = model.predict(X_poly)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    bias = np.mean(y_pred - y)
    r2 = r2_score(y, y_pred)
    
    # Extract coefficients
    coef = model.coef_
    coef[0] = model.intercept_  # Replace the first coefficient with the intercept
    
    return coef, rmse, mae, bias, r2

# Initialize lists to collect output
output_data = []

# List of file paths
file_paths = [
    "C:/Users/PHYS3009/Desktop/Regression_Analysis/Chla_L9.xlsx",
    "C:/Users/PHYS3009/Desktop/Regression_Analysis/Chla_REMX.xlsx",
    "C:/Users/PHYS3009/Desktop/Regression_Analysis/TSS_L9.xlsx",
    "C:/Users/PHYS3009/Desktop/Regression_Analysis/TSS_REMX.xlsx",
]

# Output directory
output_dir = "C:/Users/PHYS3009/Desktop/Regression_Analysis/Output"

# Loop through each file
for file_path in file_paths:
    # Read Excel file into DataFrame
    df = pd.read_excel(file_path)
    
    # Separate y and x columns
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    
    # Determine whether to perform k-fold cross-validation
    n = len(df)
    k_fold = False
    if n == 52 or n == 56:
        k_fold = True
    
    # Loop through each x column
    for x_col in X.columns:
        X_single = X[[x_col]]
        
        # If k-fold cross-validation is needed
        if k_fold:
            kf = KFold(n_splits=5, shuffle=True, random_state=1)
            metrics = {'rmse': [], 'mae': [], 'bias': [], 'r2': []}
            
            for train_index, test_index in kf.split(df):
                X_train, X_test = X_single.iloc[train_index], X_single.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                for degree in [1, 2, 3]:
                    coef, rmse, mae, bias, r2 = perform_regression(X_train, y_train, degree)
                    
                    # Append metrics
                    metrics['rmse'].append(rmse)
                    metrics['mae'].append(mae)
                    metrics['bias'].append(bias)
                    metrics['r2'].append(r2)
                    
                    # Append output data
                    output_data.append({
                        'File': os.path.basename(file_path),
                        'X_Column': x_col,
                        'Degree': degree,
                        'a': coef[0],
                        'b': coef[1],
                        'c': coef[2] if degree >= 2 else None,
                        'd': coef[3] if degree == 3 else None,
                        'RMSE': np.mean(metrics['rmse']),
                        'MAE': np.mean(metrics['mae']),
                        'Bias': np.mean(metrics['bias']),
                        'R2': np.mean(metrics['r2'])
                    })
        else:
            # Simple regression
            for degree in [1, 2, 3]:
                coef, rmse, mae, bias, r2 = perform_regression(X_single, y, degree)
                
                # Append output data
                output_data.append({
                    'File': os.path.basename(file_path),
                    'X_Column': x_col,
                    'Degree': degree,
                    'a': coef[0],
                    'b': coef[1],
                    'c': coef[2] if degree >= 2 else None,
                    'd': coef[3] if degree == 3 else None,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Bias': bias,
                    'R2': r2
                })

# Convert output data to DataFrame and save as Excel
output_df = pd.DataFrame(output_data)
output_file_path = os.path.join(output_dir, "CV_Output.xlsx")
output_df.to_excel(output_file_path, index=False, engine='openpyxl')
