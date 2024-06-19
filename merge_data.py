import xarray as xr
import pandas as pd
import os

# Specify the file paths
file1 = '/content/DM COMP4040/hadisd.3.4.0.2023f_19310101-20240101_400010-99999_heat_stress.nc'
file2 = '/content/DM COMP4040/hadisd.3.4.0.2023f_19310101-20240101_400010-99999_humidity.nc'
# file3 = '/content/DM COMP4040/hadisd.3.4.0.2023f_19310101-20240101_466890-99999.nc'

# Function to load a dataset and handle errors
def load_dataset(file_path):
    if os.path.exists(file_path):
        return xr.open_dataset(file_path)
    else:
        print(f"Error: File {file_path} does not exist.")
        return None

# Load the datasets
ds1 = load_dataset(file1)
ds2 = load_dataset(file2)
# ds3 = load_dataset(file3)

# Check if both datasets are loaded
if ds1 is not None and ds2 is not None:
    # Merge the datasets with override option
    combined_ds = xr.merge([ds1, ds2], compat='override')

    # Reduce the time range to the first 100 time steps
    combined_ds_reduced = combined_ds.isel(time=slice(0, 100))

    # Select only numeric variables for resampling
    numeric_vars = {var: combined_ds_reduced[var] for var in combined_ds_reduced.data_vars if combined_ds_reduced[var].dtype.kind in 'if'}
    numeric_ds = xr.Dataset(numeric_vars)

    # Downsample data to monthly averages (if time dimension exists)
    if 'time' in numeric_ds.dims:
        numeric_ds = numeric_ds.resample(time='1M').mean()

    # Convert the reduced numeric dataset to a DataFrame for display
    combined_df = numeric_ds.to_dataframe()

    # Save the reduced dataset to a CSV file
    output_csv_file = 'reduced_combined_dataset.csv'
    combined_df.to_csv(output_csv_file)

    # Print the first few rows of the reduced dataset to verify the merge
    print(combined_df.head())
else:
    print("One or both datasets could not be loaded. Please check the file paths and ensure the files exist.")
