import pandas as pd
import xarray as xr

# Load the provinces data
provinces_df = pd.read_csv('vietnam_provinces.csv')

# Open the NetCDF file
decompressed_file_path = '/home/kyv/Desktop/COMP4040-DataMining/Project/data/decompress/decompressed.nc'
xr_dataset = xr.open_dataset(decompressed_file_path)

# Extract the entire time series data
temperatures = xr_dataset['temperatures'].values
dewpoints = xr_dataset['dewpoints'].values
precipitations = xr_dataset['precip1_depth'].values

# Assuming uniform data for all provinces, you can aggregate these time series
# For simplicity, let's take the mean value over the entire time range
mean_temperature = temperatures.mean()
mean_dewpoint = dewpoints.mean()
mean_precipitation = precipitations.mean()

# Create lists to store the aggregated data for each province
mean_temperatures = [mean_temperature] * len(provinces_df)
mean_dewpoints = [mean_dewpoint] * len(provinces_df)
mean_precipitations = [mean_precipitation] * len(provinces_df)

# Add the aggregated data to the DataFrame
provinces_df['mean_temperature'] = mean_temperatures
provinces_df['mean_dewpoint'] = mean_dewpoints
provinces_df['mean_precipitation'] = mean_precipitations

# Save the combined data to a new CSV file
provinces_df.to_csv('vietnam_climate_data.csv', index=False)

print("Data extraction complete. The dataset is saved as 'vietnam_climate_data.csv'.")
