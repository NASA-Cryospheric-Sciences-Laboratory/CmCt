# import xarray as xr
# import pandas as pd
# import numpy as np
# import os
# import glob
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from bs4 import BeautifulSoup
# import tempfile
# import urllib.request
# import netCDF4 
# import requests
# import logging
# from typing import List, Dict, Set
# import json

#################################################################################################################
########################## Dowload and process OGGM data for selected region, model, scenario ###################
#################################################################################################################


reference_glaciers = ['RGI60-01.00570', 'RGI60-01.01104', 'RGI60-01.09162', 
                      'RGI60-02.00377', 'RGI60-02.01104', 'RGI60-02.05098', 'RGI60-02.17733', 
                      'RGI60-02.17739', 'RGI60-02.18415', 'RGI60-02.18778', 'RGI60-03.04539', 
                      'RGI60-03.04552', 'RGI60-06.00234', 'RGI60-06.00236', 'RGI60-06.00238', 
                      'RGI60-06.00377', 'RGI60-06.00443', 'RGI60-06.00472', 'RGI60-07.00493', 
                      'RGI60-07.00504', 'RGI60-08.00006', 'RGI60-08.00188', 'RGI60-08.00199', 
                      'RGI60-08.00213', 'RGI60-08.00312', 'RGI60-08.00449', 'RGI60-08.00987', 
                      'RGI60-08.01126', 'RGI60-08.01258', 'RGI60-08.01657', 'RGI60-08.01779', 
                      'RGI60-08.02666', 'RGI60-10.01737', 'RGI60-11.00106', 'RGI60-11.00289', 
                      'RGI60-11.00719', 'RGI60-11.00781', 'RGI60-11.00787', 'RGI60-11.00804', 
                      'RGI60-11.00897', 'RGI60-11.01876', 'RGI60-11.01987', 'RGI60-11.02704', 
                      'RGI60-11.02774', 'RGI60-11.03135', 'RGI60-11.03209', 'RGI60-11.03638', 
                      'RGI60-11.03674', 'RGI60-11.03756', 'RGI60-12.00161', 'RGI60-12.01132', 
                      'RGI60-13.06361', 'RGI60-13.08624', 'RGI60-13.11609', 'RGI60-13.18096', 
                      'RGI60-16.00543', 'RGI60-17.13715']
    
summary_rows = []
all_rgi_ids_by_region = {}
reference_rows = []

# for testing
# selected_regions = ["06", "07"]
# selected_model = "ACCESS-CM2"
# selected_scenario = "ssp126"

for region_code in selected_regions:
    # Extract files for the selected region
    base_url = f"https://cluster.klima.uni-bremen.de/~oggm/oggm-standard-projections/oggm_v16/2023.3/CMIP6/2100/RGI{region_code}/"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    file_links = [
        a['href']
        for a in soup.find_all('a', href=True)
        if a['href'].endswith('.nc') and 'w5e5' in a['href']
    ]
    # Filter files for selected model and scenario
    selected_files = [
    base_url + fname
    for fname in file_links
    if (
        (parts := fname.split("_")) and
        len(parts) >= 7 and
        parts[5] == selected_model and
        parts[6] == selected_scenario
    )
]

    mass_change_kg_list = []
    elevation_change_list = []
    all_rgi_ids = set()

    # Loop over selected files and gather glacier-level data
    for i, file_url in enumerate(selected_files):
        file_url_bytes = file_url + "#mode=bytes"
        print(f"File {i + 1} of {len(selected_files)} for region {region_code}: Processing {file_url_bytes}")

        with xr.open_dataset(file_url_bytes) as ds:
            rgi_ids = ds["rgi_id"].values
            all_rgi_ids.update(rgi_ids)

            volume = ds["volume"]                # (time, rgi_id)
            area = ds["area"]                    # (rgi_id,) 
            # Calculate mass change
            mass = volume * 900                  # kg
            mass_change = mass.diff(dim="time") / 1e12  # Gt/year

            # Calculate elevation change
            volume_change = volume.diff(dim="time")
            elevation_change = volume_change / area  # (time, rgi_id)

            # check for model errors
            area_zero_vol_nonzero = ((area == 0) & (volume_change != 0)) | ((area != 0) & (volume_change == 0))
            elevation_change = elevation_change.where(~area_zero_vol_nonzero, np.nan)
            # Set elevation_change = 0 where both area and volume_change are 0
            both_zero = (area == 0) & (volume_change == 0)
            elevation_change = elevation_change.where(~both_zero, 0)
            
            hydro_years = volume_change["hydro_year"].values
            calendar_years = volume_change["calendar_year"].values

            # Save reference glacier data
            for idx, rgi_id in enumerate(rgi_ids):
                if rgi_id in reference_glaciers:
                    # For each time step
                    for t in range(len(calendar_years)):
                        reference_rows.append({
                            "rgi_id": rgi_id,
                            "region": region_code,
                            "hydro_year": hydro_years[t],
                            "calendar_year": calendar_years[t],
                            "dmdt": mass_change[t, idx].item(),
                            "dhdt": elevation_change[t, idx].item()
                        })

            mass_change_kg_list.append(mass_change)
            elevation_change_list.append(elevation_change)

    # Stack all glacier elevation changes into one array (dim: time, rgi_id)
    region_elevation_change = xr.concat(elevation_change_list, dim="rgi_id")  # shape: (time, all_rgi_id)
    # Simple mean across glaciers for each year
    dhdt_simple_avg = region_elevation_change.mean(dim="rgi_id")  # shape: (time,)

    # Mass change aggregation
    region_mass_change_kg = xr.concat(mass_change_kg_list, dim="rgi_id")
    total_mass_change_per_year = region_mass_change_kg.sum(dim="rgi_id")   # (time,)

    # Collect RGI ids
    all_rgi_ids_by_region[region_code] = all_rgi_ids

    # Create a DataFrame for the annual results
    annual_df = pd.DataFrame({
        "region": region_code,
        "model": selected_model,
        "scenario": selected_scenario,
        "hydro_year": hydro_years,
        "calendar_year": calendar_years,
        #"five_year_period": five_year_periods,
        "dmdt": total_mass_change_per_year.values,
        "dhdt": dhdt_simple_avg.values
    })
    

    # add five year period averages
    block_size = 5
    # For calendar year
    calendar_blocks = (annual_df['calendar_year'] - annual_df['calendar_year'].min()) // block_size
    last_years = annual_df.groupby(calendar_blocks)['calendar_year'].transform('max')
    annual_df['calendar_year_5yr'] = last_years
   
    annual_df['dmdt_5yr_avg_calendar'] = annual_df.groupby(calendar_blocks)['dmdt'].transform('mean')
    annual_df['dhdt_5yr_avg_calendar'] = annual_df.groupby(calendar_blocks)['dhdt'].transform('mean')

    annual_df.loc[annual_df['calendar_year'] != annual_df['calendar_year_5yr'], 'dmdt_5yr_avg_calendar'] = np.nan
    annual_df.loc[annual_df['calendar_year'] != annual_df['calendar_year_5yr'], 'dhdt_5yr_avg_calendar'] = np.nan
    annual_df.loc[annual_df['calendar_year'] != annual_df['calendar_year_5yr'], 'calendar_year_5yr'] = np.nan
    # Standard error for dmdt and dhdt
    # dmdt_5yr_se_calendar = annual_df.groupby(calendar_blocks)['dmdt'].transform(lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    # dhdt_5yr_se_calendar = annual_df.groupby(calendar_blocks)['dhdt'].transform(lambda x: x.std(ddof=1) / np.sqrt(len(x)))

    # 95% CI
    # annual_df['dmdt_5yr_ci_upper_calendar'] = annual_df['dmdt_5yr_avg_calendar'] + 1.96 * dmdt_5yr_se_calendar
    # annual_df['dmdt_5yr_ci_lower_calendar'] = annual_df['dmdt_5yr_avg_calendar'] - 1.96 * dmdt_5yr_se_calendar
    # annual_df['dhdt_5yr_ci_upper_calendar'] = annual_df['dhdt_5yr_avg_calendar'] + 1.96 * dhdt_5yr_se_calendar
    # annual_df['dhdt_5yr_ci_lower_calendar'] = annual_df['dhdt_5yr_avg_calendar'] - 1.96 * dhdt_5yr_se_calendar

    # For hydro year

    # # Sort by hydro_year for correct rolling calculation
    # annual_df = annual_df.sort_values("hydro_year")

    # # 5-year rolling averages (centered)
    # annual_df['dmdt_5yr_avg_hydro'] = annual_df['dmdt'].rolling(window=5, min_periods=1, center=True).mean()
    # annual_df['dhdt_5yr_avg_hydro'] = annual_df['dhdt'].rolling(window=5, min_periods=1, center=True).mean()

    # hydro_blocks = (annual_df['hydro_year'] - annual_df['hydro_year'].min()) // block_size
    # annual_df['hydro_year_5yr'] = annual_df.groupby(hydro_blocks)['hydro_year'].transform('max')
    # annual_df['dmdt_5yr_avg_hydro'] = annual_df.groupby(hydro_blocks)['dmdt'].transform('mean')
    # annual_df['dhdt_5yr_avg_hydro'] = annual_df.groupby(hydro_blocks)['dhdt'].transform('mean')

    # dmdt_5yr_se_hydro = annual_df.groupby(hydro_blocks)['dmdt'].transform(lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    # dhdt_5yr_se_hydro = annual_df.groupby(hydro_blocks)['dhdt'].transform(lambda x: x.std(ddof=1) / np.sqrt(len(x)))

    # annual_df['dmdt_5yr_ci_upper_hydro'] = annual_df['dmdt_5yr_avg_hydro'] + 1.96 * dmdt_5yr_se_hydro
    # annual_df['dmdt_5yr_ci_lower_hydro'] = annual_df['dmdt_5yr_avg_hydro'] - 1.96 * dmdt_5yr_se_hydro
    # annual_df['dhdt_5yr_ci_upper_hydro'] = annual_df['dhdt_5yr_avg_hydro'] + 1.96 * dhdt_5yr_se_hydro
    # annual_df['dhdt_5yr_ci_lower_hydro'] = annual_df['dhdt_5yr_avg_hydro'] - 1.96 * dhdt_5yr_se_hydro

    summary_rows.append(annual_df)

reference_df = pd.DataFrame(reference_rows)
os.makedirs(os.path.dirname(reference_glaciers_output_path), exist_ok=True)
reference_df.to_csv(reference_glaciers_output_path, index=False)
print(f"\nReference glacier data saved to {reference_glaciers_output_path}")


# Export as csv
summary_df = pd.concat(summary_rows, ignore_index=True)
os.makedirs(os.path.dirname(output_summary_csv), exist_ok=True)
summary_df.to_csv(output_summary_csv, index=False)
print(f"\n Summary statistics saved to {output_summary_csv}")


# Convert sets to lists for JSON serialization
rgiids_export = {region: list(rgi_ids) for region, rgi_ids in all_rgi_ids_by_region.items()}
with open("../output_csv/all_rgi_ids_by_region.json", "w") as f:
    json.dump(rgiids_export, f)
print("Exported RGI IDs by region to ../output_csv/all_rgi_ids_by_region.json")


#################################################################################################################
#################################### plots for mass change in selected region  ##################################
#################################################################################################################


if observation_dataset == 'Mass Change (Hugonnet, 2021)':
    plt.figure(figsize=(12, 6))
    for region in selected_regions:
        region_data = summary_df[summary_df['region'] == region]
        plt.plot(region_data['calendar_year'], region_data['dmdt'], label=f"RGI{region}")

    plt.title("Mass Change by Region")
    plt.xlabel("Calendar Year")
    plt.ylabel("Mass Change (Gt)")
    plt.legend()
    plt.grid()
    plt.show()

elif observation_dataset == 'Mass Change (Dussaillant, 2025)':
    plt.figure(figsize=(12, 6))
    for region in selected_regions:
        region_data = summary_df[summary_df['region'] == region]
        plt.plot(region_data['hydro_year'], region_data['dmdt'], label=f"RGI{region}")

    plt.title("Mass Change by Region")
    plt.xlabel("Hydrological Year")
    plt.ylabel("Mass Change (Gt)")
    plt.legend()
    plt.grid()
    plt.show()

elif observation_dataset == 'Elevation Change (Hugonnet, 2021)':
    plt.figure(figsize=(12, 6))
    for region in selected_regions:
        region_data = summary_df[summary_df['region'] == region]
        plt.plot(region_data['calendar_year'], region_data['dhdt'], label=f"RGI{region}")

    plt.title("Average Elevation Change by Region")
    plt.xlabel("Calendar Year")
    plt.ylabel("Average Elevation Change (m/year)")
    plt.legend()
    plt.grid()
    plt.show()