from .time_utils import check_datarange,days_in_year
import os
import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import datetime
from datetime import timedelta 




### Load the model data and calculate model mass balance for each basin and total mass balance for whole region
## Interpolate the data to each IMBIE time and calculate the time varying mass change
def process_model_data(mod_ds,time_var, IMBIE_total_mass_change_sum, \
                       start_date_cftime, end_date_cftime, start_date_fract, end_date_fract, \
                       rho_ice, projection, shape_filename, icesheet):
    
    # Model data
    lithk = mod_ds['lithk']
    
    lithk['time'] = [date.year + (date.dayofyr-1) / days_in_year(date) for date in time_var.data]

    # Load basin shapefile 
    basins_gdf = gpd.read_file(shape_filename)
    
    # Check the selcted dates are within the range of model data
    check_datarange(time_var,start_date_cftime, end_date_cftime)
        
    # Set start_date as the first date in 'Year' and filtered_time_var as all subsequent dates
    start_date_imbie = start_date_fract
    # filtered_time_var = IMBIE_total_mass_change_sum['Year'].iloc[0:]
    filtered_time_var = IMBIE_total_mass_change_sum['Year'].values
    
    #calculate area = x_resolution*y_resolution
    x_coords = mod_ds['x'].values
    y_coords = mod_ds['y'].values
    x_resolution = abs(x_coords[1] - x_coords[0])
    y_resolution = abs(y_coords[1] - y_coords[0])
    
    # Create a list of Point geometries from coordinate grids
    points = [Point(x, y) for x in x_coords for y in y_coords]
    

    # Initialize an empty list to store rows
    model_mass_change_rows = []
    
    # Interpolate limnsw at the start date (initial reference)
    lithk_start = lithk.interp(time=start_date_fract).data.transpose().flatten()

    
    
    # Loop through each filtered time step to calculate the residual
    for i, time_step in enumerate(filtered_time_var):
     
        # Interpolate limnsw at the current time_step
        lithk_current = lithk.interp(time=time_step).data.transpose().flatten()
        
        # Calculate the residual (difference from start)
        lithk_delta = lithk_current - lithk_start
    
        lithk_delta[np.isnan(lithk_delta)] = 0
    
        #calculate area = x_resolution*y_resolution
        lithk_delta = (lithk_delta * x_resolution*y_resolution)*rho_ice * 1e-12

        ## TOTAL AREA
        # Sum all of the area mass change
        model_total_mass_balance_unmasked= np.nansum(lithk_delta)
               
        ## BASIN AREA
        # Sum all of the basin mass change
        lithk_delta_flat = lithk_delta.flatten()
        
        # Create a lithk_df DataFrame with Geometry and Values
        lithk_df = pd.DataFrame({
            'geometry': points,
            'lithk_delta': lithk_delta_flat
        })
    
        # Convert lithk_df DataFrame to lithk_gdf GeoDataFrame
        lithk_gdf = gpd.GeoDataFrame(lithk_df, geometry='geometry', crs=projection)    
        
    
        # Perform the spatial join only once in the first iteration
        if i == 0:
            joined_gdf = gpd.sjoin(lithk_gdf, basins_gdf, how="inner", predicate='intersects')
    
    
        # Update the lithk_delta in joined_gdf
        joined_gdf['lithk_delta'] = lithk_gdf['lithk_delta']
           
        # Sum lithk_delta values by basin
        if icesheet == "GIS":
             # Sum lithk_delta values by subregion column
            basin_mass_change_sums = joined_gdf.groupby('SUBREGION1')['lithk_delta'].sum()
            # Sum lithk_delta values by the 'Regions' column
            region_mass_change_sums = None  # No regions for Greenland
        elif icesheet == "AIS":
            # Sum lithk_delta values by subregion column
            basin_mass_change_sums = joined_gdf.groupby('Subregion')['lithk_delta'].sum()
            # Sum lithk_delta values by the 'Regions' column
            region_mass_change_sums = joined_gdf.groupby('Regions')['lithk_delta'].sum()
        else:
            raise ValueError("Invalid iceshee value. Must be 'GIS' or 'AIS'.")
        
        # Sum all of the basin mass change
        model_total_mass_balance_masked = basin_mass_change_sums.sum()
          

        # Store the residual for the current time step
        row = {
            'Time_Step': str(time_step),  # Convert time_step to string for consistency
            'model_total_mass_balance_unmasked': model_total_mass_balance_unmasked,
            'model_total_mass_balance_masked': model_total_mass_balance_masked,
            'basin_mass_change_sums': basin_mass_change_sums,
            'region_mass_change_sums': region_mass_change_sums
        }
        model_mass_change_rows.append(row)
    
    # Convert the list of rows into a DataFrame
    model_mass_change_df = pd.DataFrame(model_mass_change_rows)  
    
    # Return all results as a dictionary
    return model_mass_change_df




### Extract time varying IMBIE mass balance data and calculate the time varying mass difference 
def process_imbie_data(obs_filename,start_date_fract,end_date_fract,mass_balance_column):

    # Load the CSV file
    mass_balance_data = pd.read_csv(obs_filename)
    
    # Column names
    date_column = 'Year'
    
    # Ensure the 'Year' column is treated as float to capture the fractional year part
    mass_balance_data['Year'] = mass_balance_data['Year'].astype(float)
    

    # Sort the data by 'Date' column to ensure it’s in increasing order of both year and fraction
    mass_balance_data = mass_balance_data.sort_values(by='Year')
        

    
    # Check if the column exists in the DataFrame
    if mass_balance_column not in mass_balance_data.columns:
        raise ValueError(f"Error: The column '{mass_balance_column}' does not exist in the CSV file.")
  
    # Get the initial mass balance value for the start date
    data_start_date = mass_balance_data[mass_balance_data['Year'] == start_date_fract]
   
    if data_start_date.empty:
        raise ValueError(f'Error: No data available for the start date {start_date_fract}.')
    mass_balance_start_value = data_start_date[mass_balance_column].iloc[0]  # value of start date
    
    # Filter data between start_date_converted and end_date_converted (inclusive)
    filtered_data = mass_balance_data[
        (mass_balance_data['Year'] >= start_date_fract) & (mass_balance_data['Year'] <= end_date_fract)
    ].copy()
    
    
    # Initialize the previous date's mass balance value to the starting mass balance
    previous_mass_balance = mass_balance_start_value
    
    # Calculate monthly mass change from the initial date for each time step
    mass_changes = []  # To store the daily mass changes
    
    for index, row in filtered_data.iterrows():
        current_mass_balance = row[mass_balance_column]
        # Mass change = 0 for initial time
        if index == filtered_data.index[0]:  # First iteration
            mass_change = 0  # Set first change to 0
        else:
            # Calculate the change from the previous date's balance
            mass_change = current_mass_balance - previous_mass_balance

        mass_changes.append(mass_change)
      
    # Assign the calculated mass changes to a new column in the DataFrame
    imbie_mass_balance_data = pd.DataFrame({'Year': filtered_data['Year']})
    imbie_mass_balance_data[mass_balance_column] = mass_changes

    return  imbie_mass_balance_data
   
    
 

### Calculate mass balance difference of IMBIE and model data
def calculate_model_imbie_residuals(start_date_fract, end_date_fract, \
                  icesheet, basin_result, IMBIE_total_mass_change_sum, mass_balance_column, \
                  obs_east_filename=None, obs_west_filename=None, obs_peninsula_filename=None):

    print_regionalresult_check = 'NO'  # Default status
    
    # Ensure 'Year' is the same type for merging
    IMBIE_total_mass_change_sum['Year'] = IMBIE_total_mass_change_sum['Year'].astype(str)
    basin_result['Year'] = basin_result['Time_Step'].astype(str)  # Assuming 'Time_Step' corresponds to 'Year'
    
    # Merge the IMBIE and model data on 'Year'
    merged_df = IMBIE_total_mass_change_sum.merge(
        basin_result, on='Year', how='inner'
    )
    
    # Compute the delta mass change
    merged_df['delta_masschange_masked'] = merged_df[mass_balance_column] - merged_df['model_total_mass_balance_masked']
    merged_df['delta_masschange_unmasked'] = merged_df[mass_balance_column] - merged_df['model_total_mass_balance_unmasked']

    # Store results in a DataFrame
  
    model_imbie_results_df = merged_df[['Year', mass_balance_column, 'delta_masschange_masked', 'delta_masschange_unmasked']]
    # Create an explicit copy before renaming columns
    model_imbie_results_df = model_imbie_results_df.copy()  
    # Rename the column safely
    model_imbie_results_df.rename(columns={mass_balance_column: 'IMBIE_total_mass_change_sum'}, inplace=True)

    # Initialize an empty DataFrame for regional results
    regional_results_df = pd.DataFrame()

    if icesheet == "AIS":  
        # Check if observation files exist
        if (obs_east_filename and os.path.exists(obs_east_filename)) and \
           (obs_west_filename and os.path.exists(obs_west_filename)) and \
           (obs_peninsula_filename and os.path.exists(obs_peninsula_filename)):

            print_regionalresult_check = 'YES'
            
            # Process IMBIE mass change data for each region
            imbie_east = process_imbie_data(obs_east_filename, start_date_fract, end_date_fract, mass_balance_column)
            imbie_west = process_imbie_data(obs_west_filename, start_date_fract, end_date_fract, mass_balance_column)
            imbie_peninsula = process_imbie_data(obs_peninsula_filename, start_date_fract, end_date_fract, mass_balance_column)
    
            # Merge the regional data on 'Year'
            regional_df = imbie_east.merge(imbie_west, on='Year', suffixes=('_East', '_West')) \
                                    .merge(imbie_peninsula, on='Year')
            
            # Rename columns to differentiate mass changes
            regional_df.rename(columns={
                mass_balance_column + '_East': 'IMBIE_Mass_Change_East',
                mass_balance_column + '_West': 'IMBIE_Mass_Change_West',
                mass_balance_column: 'IMBIE_Mass_Change_Peninsula'
            }, inplace=True)
            
            # Merge with the model's basin_result DataFrame
            regional_results_df = regional_df.merge(basin_result, on='Year', how='inner')

            # Compute delta mass changes
            regional_results_df['Delta_MassChange_East'] = regional_results_df['IMBIE_Mass_Change_East'] - regional_results_df['region_mass_change_sums'].apply(lambda x: x.get('East', 0))
            regional_results_df['Delta_MassChange_West'] = regional_results_df['IMBIE_Mass_Change_West'] - regional_results_df['region_mass_change_sums'].apply(lambda x: x.get('West', 0))
            regional_results_df['Delta_MassChange_Peninsula'] = regional_results_df['IMBIE_Mass_Change_Peninsula'] - regional_results_df['region_mass_change_sums'].apply(lambda x: x.get('Peninsula', 0))

    # Create a results dictionary with DataFrames
    results = {
        "Model_IMBIE_Comparison": model_imbie_results_df,
        "Regional_Mass_Change_Summary": regional_results_df,
        "print_regionalresult_check": print_regionalresult_check
    }

    return results




def write_and_display_mass_change_comparison_all_dates(icesheet, basin_result, results, mass_balance_type, start_date_fract, end_date_fract, csv_filename):


    # Initialize list to store rows of data for CSV
    data_rows = []
    
    print_regionalresult_check = results.get('print_regionalresult_check')
    
    # Add mass change comparison header
    data_rows.append([f"Mass change comparison ({mass_balance_type})", f"{start_date_fract} - {end_date_fract}"])
    data_rows.append(['Date', 'Basin/Region', 'Model mass change (Gt)', 'IMBIE mass change (Gt)', 'Residual (Gt)'])
    
    # Extract 'Model_IMBIE_Comparison' DataFrame
    model_imbie_df = results['Model_IMBIE_Comparison']
    
    # Convert 'Year' in model_imbie_df and 'Time_Step' in basin_result to float
    model_imbie_df['Year'] = pd.to_numeric(model_imbie_df['Year'], errors='coerce')
    basin_result['Time_Step'] = pd.to_numeric(basin_result['Time_Step'], errors='coerce')
    
    # Filter to get the first available date after start_date_fract
    filtered_basin = basin_result[basin_result['Time_Step'] > start_date_fract].sort_values(by='Time_Step')
    
    # Extract basin and region names from the first available entry
    basins, regions = [], []
    if not filtered_basin.empty:
        first_entry = filtered_basin.iloc[0]
    
        # Extract basin names from the index of the Series
        if isinstance(first_entry['basin_mass_change_sums'], pd.Series):
            basins = list(first_entry['basin_mass_change_sums'].index)
    
        # Extract region names for AIS if applicable
        if icesheet == "AIS" and print_regionalresult_check == 'YES' and isinstance(first_entry['region_mass_change_sums'], pd.Series):
            regions = list(first_entry['region_mass_change_sums'].index)
    
    # Add rows for each basin with zero values for the start_date
    for basin in basins:
        data_rows.append([start_date_fract, basin, "0.00", "--", "--"])
    
    # Add rows for each region with zero values for the start_date 
    if icesheet == "AIS" and print_regionalresult_check == 'YES':
        for region in regions:
            data_rows.append([start_date_fract, region, "0.00", "0.00", "0.00"])
    
    # Add totals (masked and unmasked) with zero values for the start_date
    data_rows.append([start_date_fract, 'Masked_Total', "0.00", "0.00", "0.00"])
    data_rows.append([start_date_fract, 'Unmasked_Total', "0.00", "0.00", "0.00"])
    
    # Print the table header
    print(f"\n Time-varying Mass change comparison ({mass_balance_type}): {start_date_fract} - {end_date_fract}")
    print(f"{'Date':<15} {'Basin/Region':<20} {'Model mass change (Gt)':<25} {'IMBIE mass change (Gt)':<25} {'Residual (Gt)':<20}")
    
    # Process only valid dates in results
    for date in model_imbie_df['Year']:
        try:
            # Ensure date exists in basin_result DataFrame
            basin_row = basin_result[np.isclose(basin_result['Time_Step'], date, atol=1e-4)]
    
            if not basin_row.empty:
                # Extract basin mass change sums
                basin_mass_change_sums = basin_row.iloc[0]['basin_mass_change_sums']
    
                if isinstance(basin_mass_change_sums, pd.Series):
                    for basin, model_mass_change in basin_mass_change_sums.items():
                        data_rows.append([date, basin, f"{model_mass_change:.2f}", "--", "--"])
    
                # Regional mass changes for AIS
                if icesheet == "AIS" and print_regionalresult_check == 'YES':
                    region_mass_change_sums = basin_row.iloc[0]['region_mass_change_sums']
    
                    if isinstance(region_mass_change_sums, pd.Series):
                        for region, model_mass_change in region_mass_change_sums.items():
                            # Retrieve IMBIE and residual mass changes 
                            imbie_row = model_imbie_df[model_imbie_df['Year'] == date]
    
                            imbie_mass_change = imbie_row[f'IMBIE_total_mass_change_sum'].values[0] if not imbie_row.empty else "--"
                            residual_mass_change = imbie_row[f'delta_masschange_masked'].values[0] if not imbie_row.empty else "--"
    
                            # Format numbers if they are numeric
                            imbie_mass_change = f"{imbie_mass_change:.2f}" if isinstance(imbie_mass_change, (float, int)) else "--"
                            residual_mass_change = f"{residual_mass_change:.2f}" if isinstance(residual_mass_change, (float, int)) else "--"
    
                            data_rows.append([date, region, f"{model_mass_change:.2f}", imbie_mass_change, residual_mass_change])
    
                # Total mass balance masked
                model_total_mass_balance_masked = basin_row.iloc[0].get('model_total_mass_balance_masked', '--')
    
                imbie_row = model_imbie_df[model_imbie_df['Year'] == date]
                imbie_total_mass_change_sum = imbie_row['IMBIE_total_mass_change_sum'].values[0] if not imbie_row.empty else "--"
                delta_masschange_masked = imbie_row['delta_masschange_masked'].values[0] if not imbie_row.empty else "--"
    
                model_total_mass_balance_masked = f"{model_total_mass_balance_masked:.2f}" if isinstance(model_total_mass_balance_masked, (float, int)) else "--"
                imbie_total_mass_change_sum = f"{imbie_total_mass_change_sum:.2f}" if isinstance(imbie_total_mass_change_sum, (float, int)) else "--"
                delta_masschange_masked = f"{delta_masschange_masked:.2f}" if isinstance(delta_masschange_masked, (float, int)) else "--"
    
                data_rows.append([date, 'Masked_Total', model_total_mass_balance_masked, imbie_total_mass_change_sum, delta_masschange_masked])
    

                print(f"{date:.4f} {'Masked_Total':<20} {model_total_mass_balance_masked:<25} {imbie_total_mass_change_sum:<25} {delta_masschange_masked:<20}")

    
                # Total mass balance unmasked
                model_total_mass_balance_unmasked = basin_row.iloc[0].get('model_total_mass_balance_unmasked', '--')
                delta_masschange_unmasked = imbie_row['delta_masschange_unmasked'].values[0] if not imbie_row.empty else "--"
    
                model_total_mass_balance_unmasked = f"{model_total_mass_balance_unmasked:.2f}" if isinstance(model_total_mass_balance_unmasked, (float, int)) else "--"
                delta_masschange_unmasked = f"{delta_masschange_unmasked:.2f}" if isinstance(delta_masschange_unmasked, (float, int)) else "--"
    
                data_rows.append([date, 'Unmasked_Total', model_total_mass_balance_unmasked, imbie_total_mass_change_sum, delta_masschange_unmasked])
    
        except Exception as e:
            print(f"Error processing date {date}: {e}")
    
    
    # Convert the data rows into a pandas DataFrame
    df = pd.DataFrame(data_rows)
    # Write the DataFrame to a CSV file
    print(f"\nWriting data to CSV file: {csv_filename}")
    df.to_csv(csv_filename, index=False, header=False)




def  write_mass_change_comparison_all_dates(icesheet, basin_result, results, mass_balance_type, start_date_fract, end_date_fract, csv_filename):
    
    
    # Initialize list to store rows of data for CSV
    data_rows = []
    
    print_regionalresult_check = results.get('print_regionalresult_check')
    
    # Add mass change comparison header
    data_rows.append([f"Mass change comparison ({mass_balance_type})", f"{start_date_fract} - {end_date_fract}"])
    data_rows.append(['Date', 'Basin/Region', 'Model mass change (Gt)', 'IMBIE mass change (Gt)', 'Residual (Gt)'])
    
    # Extract 'Model_IMBIE_Comparison' DataFrame
    model_imbie_df = results['Model_IMBIE_Comparison']
    
    # Convert 'Year' in model_imbie_df and 'Time_Step' in basin_result to float
    model_imbie_df['Year'] = pd.to_numeric(model_imbie_df['Year'], errors='coerce')
    basin_result['Time_Step'] = pd.to_numeric(basin_result['Time_Step'], errors='coerce')
    
    # Filter to get the first available date after start_date_fract
    filtered_basin = basin_result[basin_result['Time_Step'] > start_date_fract].sort_values(by='Time_Step')
    
    # Extract basin and region names from the first available entry
    basins, regions = [], []
    if not filtered_basin.empty:
        first_entry = filtered_basin.iloc[0]
    
        # Extract basin names from the index of the Series
        if isinstance(first_entry['basin_mass_change_sums'], pd.Series):
            basins = list(first_entry['basin_mass_change_sums'].index)
    
        # Extract region names for AIS if applicable
        if icesheet == "AIS" and print_regionalresult_check == 'YES' and isinstance(first_entry['region_mass_change_sums'], pd.Series):
            regions = list(first_entry['region_mass_change_sums'].index)
    
  
    # Process only valid dates in results
    for date in model_imbie_df['Year']:
        try:
            # Ensure date exists in basin_result DataFrame
            basin_row = basin_result[np.isclose(basin_result['Time_Step'], date, atol=1e-4)]
    
            if not basin_row.empty:
                # Extract basin mass change sums
                basin_mass_change_sums = basin_row.iloc[0]['basin_mass_change_sums']
    
                if isinstance(basin_mass_change_sums, pd.Series):
                    for basin, model_mass_change in basin_mass_change_sums.items():
                        data_rows.append([date, basin, f"{model_mass_change:.2f}", "--", "--"])
    
                # Regional mass changes for AIS
                if icesheet == "AIS" and print_regionalresult_check == 'YES':
                    region_mass_change_sums = basin_row.iloc[0]['region_mass_change_sums']
    
                    if isinstance(region_mass_change_sums, pd.Series):
                        for region, model_mass_change in region_mass_change_sums.items():
                            # Retrieve IMBIE and residual mass changes 
                            imbie_row = model_imbie_df[model_imbie_df['Year'] == date]
    
                            imbie_mass_change = imbie_row[f'IMBIE_total_mass_change_sum'].values[0] if not imbie_row.empty else "--"
                            residual_mass_change = imbie_row[f'delta_masschange_masked'].values[0] if not imbie_row.empty else "--"
    
                            # Format numbers if they are numeric
                            imbie_mass_change = f"{imbie_mass_change:.2f}" if isinstance(imbie_mass_change, (float, int)) else "--"
                            residual_mass_change = f"{residual_mass_change:.2f}" if isinstance(residual_mass_change, (float, int)) else "--"
    
                            data_rows.append([date, region, f"{model_mass_change:.2f}", imbie_mass_change, residual_mass_change])
    
                # Total mass balance masked
                model_total_mass_balance_masked = basin_row.iloc[0].get('model_total_mass_balance_masked', '--')
    
                imbie_row = model_imbie_df[model_imbie_df['Year'] == date]
                imbie_total_mass_change_sum = imbie_row['IMBIE_total_mass_change_sum'].values[0] if not imbie_row.empty else "--"
                delta_masschange_masked = imbie_row['delta_masschange_masked'].values[0] if not imbie_row.empty else "--"
    
                model_total_mass_balance_masked = f"{model_total_mass_balance_masked:.2f}" if isinstance(model_total_mass_balance_masked, (float, int)) else "--"
                imbie_total_mass_change_sum = f"{imbie_total_mass_change_sum:.2f}" if isinstance(imbie_total_mass_change_sum, (float, int)) else "--"
                delta_masschange_masked = f"{delta_masschange_masked:.2f}" if isinstance(delta_masschange_masked, (float, int)) else "--"
    
                data_rows.append([date, 'Masked_Total', model_total_mass_balance_masked, imbie_total_mass_change_sum, delta_masschange_masked])
    
                # print(f"{date:<15} {'Masked_Total':<20} {model_total_mass_balance_masked:<25} {imbie_total_mass_change_sum:<25} {delta_masschange_masked:<20}")
    
                # Total mass balance unmasked
                model_total_mass_balance_unmasked = basin_row.iloc[0].get('model_total_mass_balance_unmasked', '--')
                delta_masschange_unmasked = imbie_row['delta_masschange_unmasked'].values[0] if not imbie_row.empty else "--"
    
                model_total_mass_balance_unmasked = f"{model_total_mass_balance_unmasked:.2f}" if isinstance(model_total_mass_balance_unmasked, (float, int)) else "--"
                delta_masschange_unmasked = f"{delta_masschange_unmasked:.2f}" if isinstance(delta_masschange_unmasked, (float, int)) else "--"
    
                data_rows.append([date, 'Unmasked_Total', model_total_mass_balance_unmasked, imbie_total_mass_change_sum, delta_masschange_unmasked])
    
        except Exception as e:
            print(f"Error processing date {date}: {e}")
    
    
    # Convert the data rows into a pandas DataFrame
    df = pd.DataFrame(data_rows)
    # Write the DataFrame to a CSV file
    print(f"\nWriting data to CSV file: {csv_filename}")
    df.to_csv(csv_filename, index=False, header=False)




