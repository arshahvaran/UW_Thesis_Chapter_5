# Open Python Command Prompt
# Navigate to the following directory
# cd C:\Users\PHYS3009\Desktop\Final_Map\
# Run the code
# python Raster_Calculations.py

import arcpy
from arcpy import env
from arcpy.sa import *

# Set environment settings
env.workspace = "C:/Users/PHYS3009/Desktop/Final_Map/Outputs"
env.overwriteOutput = True

# Check if Spatial Analyst extension is available
if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.CheckOutExtension("Spatial")
else:
    raise Exception("Spatial Analyst Extension is not available")

# Perform calculations for HHLS, NRO, SMCO
datasets = ['HHLS', 'NRO', 'SMCO']

for dataset in datasets:
    try:
        if dataset == 'HHLS':
            projectnumber = '1'
        elif dataset == 'NRO':
            projectnumber = '2'
        elif dataset == 'SMCO':
            projectnumber = '3'
        else:
            projectnumber = 'Unknown'
        
        red_input = f"C:/Users/PHYS3009/Desktop/Final_Map/{dataset}/Project{projectnumber}_transparent_reflectance_red.tif"
        green_input = f"C:/Users/PHYS3009/Desktop/Final_Map/{dataset}/Project{projectnumber}_transparent_reflectance_green.tif"

        red_raster = Raster(red_input)
        green_raster = Raster(green_input)

        chla_output = 70.28 * red_raster + 0.3177
        tss_output = 5139 * (green_raster ** 2) - 163.4 * green_raster + 1.526

        chla_output.save(f"C:/Users/PHYS3009/Desktop/Final_Map/Outputs/C_R_{dataset}.tif")
        tss_output.save(f"C:/Users/PHYS3009/Desktop/Final_Map/Outputs/T_R_{dataset}.tif")
        
        print(f"Successfully processed dataset {dataset}")
        
    except Exception as e:
        print(f"An error occurred while processing dataset {dataset}: {e}")


# Special calculations for L9
green_raster_L9 = Raster("C:/Users/PHYS3009/Desktop/Final_Map/L9/RT_LC09_L1TP_018030_20220614_20220615_02_T1_B3.TIF")
red_raster_L9 = Raster("C:/Users/PHYS3009/Desktop/Final_Map/L9/RT_LC09_L1TP_018030_20220614_20220615_02_T1_B4.TIF")
Chla_L9 = 4.719 * ((green_raster_L9 / red_raster_L9) ** 12.33) + 0.6885
Chla_L9.save("C:/Users/PHYS3009/Desktop/Final_Map/Outputs/C_L9.tif")

nir_raster_L9 = Raster("C:/Users/PHYS3009/Desktop/Final_Map/L9/RT_LC09_L1TP_018030_20220614_20220615_02_T1_B5.TIF")
TSS_L9 = 510.3 * (nir_raster_L9 ** 2) - 43.94 * nir_raster_L9 + 4.041
TSS_L9.save("C:/Users/PHYS3009/Desktop/Final_Map/Outputs/T_L9.tif")

# Don't forget to check the extension back in when you are done
arcpy.CheckInExtension("Spatial")
