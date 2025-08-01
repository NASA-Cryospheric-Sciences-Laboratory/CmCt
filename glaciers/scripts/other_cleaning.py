import os
import pandas as pd
import xarray as xr
import numpy as np
import io
import requests
from bs4 import BeautifulSoup
import tempfile


# for testing
link_path = "https://cluster.klima.uni-bremen.de/~oggm/oggm-standard-projections/oggm_v16/2023.3/CMIP6/2100"
source_type = "Link"
selected_model_text = "ACCESS-CM2"
selected_scenario_text = "ssp126"
selected_regions = ["06"]
source_type = "Link" 

# ======================================================
# ASSUMPTIONS & DATA REQUIREMENTS FOR OTHER_CLEANING
# ======================================================
#
# 1. GENERAL
# - The user must define:
#     - source_type: Either "Upload" (CSV) or "Link" (.nc files).
#     - selected_model_text: The model name to process (e.g., "custom_model").
#     - selected_scenario_text: The scenario name to process (e.g., "historical").
#     - selected_regions: A list of region codes (e.g., ["01", "02"]).
# - The script processes only data matching (selected_model_text, selected_scenario_text).
# - Output: A DataFrame (summary_df) with columns:
#       ['calendar_year', 'hydro_year', 'dmdt', 'dhdt',
#        'dmdt_5yr_avg', 'dhdt_5yr_avg', 'calendar_year_5yr',
#        'model', 'scenario', 'region'].
#
# ------------------------------------------------------
# 2. CSV UPLOADS (source_type == "Upload")
# - File type: A single .csv file.
# - Data format: Annual time series data only (sub-annual data not supported).
# - Required columns (minimum):
#       'region'       -> Region code for each glacier entry (will be zero-padded to 2 digits).
#       'rgi_id'       -> Unique glacier identifiers.
#       'volume'       -> Glacier volume values (for dmdt calculation).
#       'area'         -> Glacier area values (for dhdt calculation).
#       'calendar_year' OR 'hydro_year' -> One must be present (annual temporal resolution).
# - Optional columns:
#       'model'        -> Model name. If missing, defaults to 'custom'.
#       'scenario'     -> Scenario name. If missing, defaults to 'custom'.
# - Wide format support:
#       Years can be provided as column headers (e.g., 2000, 2001, 2002...).
#       Data will be automatically converted from wide to long format.
# - Auto-conversion:
#       If only 'hydro_year' exists, 'calendar_year' is generated (hydro_year - 1).
#       If only 'calendar_year' exists, 'hydro_year' is generated (calendar_year + 1).
#       Region codes are standardized to 2-digit format (e.g., '1' -> '01').
# - Filtering:
#       If multiple models/scenarios exist, data is filtered by selected_model_text & selected_scenario_text.
#       If only one unique model/scenario exists, all data is used regardless of user selection.
#       Data is then processed for each region in selected_regions.
#
# ------------------------------------------------------
# 3. NETCDF FILES (source_type == "Link")
# - File type: A directory containing multiple .nc files.
# - Data format: Annual time series data only.
# - File naming format:
#       Filenames must contain the model and scenario as parts [5] and [6] when split by "_".
#       Example: some_output_run_v1_modelA_historical_RGI01.nc
#                parts[5] = "modelA", parts[6] = "historical".
#       Region identification: Region string (e.g., "01", "02") must be in the filename.
# - Variables inside .nc files:
#       'volume'  -> Glacier volume (time × rgi_id) - annual values.
#       'area'    -> Glacier area (time × rgi_id) - annual values.
#       'time'    -> Represents calendar_year (annual temporal resolution).
#       'rgi_id'  -> Glacier identifiers.
# - Filtering:
#       Only files matching (selected_model_text, selected_scenario_text) are processed.
#       Regions are identified by checking if zero-padded region string is in the filename.
#
# ------------------------------------------------------
# 4. CALCULATIONS
# - Mass change (dmdt):
#       dmdt = Δvolume * 900 (kg/m³) / 1e12  -> Gt/year (annual differences).
# - Elevation change (dhdt):
#       dhdt = Δvolume / area                -> m/year (annual differences).
# - 5-Year Averages:
#       Data is grouped into 5-year blocks based on calendar_year.
#       The average dmdt and dhdt are assigned to the last year of each block.
#       All other years within the block are set to NaN for these averages.
#       A 'calendar_year_5yr' column marks the years with 5-year averages.
# - Reference Glacier Tracking:
#       Individual glacier data for reference glaciers is saved separately.
#       RGI IDs by region are tracked and exported for comparison workflows.
#
# ------------------------------------------------------
# 5. OUTPUT
# - The final summary_df contains:
#       'calendar_year', 'hydro_year', 'dmdt', 'dhdt',
#       'dmdt_5yr_avg', 'dhdt_5yr_avg', 'calendar_year_5yr',
#       'model', 'scenario', 'region'.
# - Additional outputs:
#       Reference glacier data saved to reference_glaciers_output_path.
#       RGI IDs by region exported to JSON for downstream compatibility.
# - This summary_df is used downstream for plotting and comparisons with observation datasets.
# - All region codes are standardized to 2-digit zero-padded format for consistency.
#
# ======================================================


# ======================================================
# Helper Functions
# ======================================================

def validate_required_columns(df, cols):
    """Ensure DataFrame has required columns."""
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def ensure_calendar_and_hydro_year(df):
    """Ensure both calendar_year and hydro_year are present."""
    if 'calendar_year' not in df.columns and 'hydro_year' not in df.columns:
        raise ValueError("Data must include either 'calendar_year' or 'hydro_year'.")
    if 'calendar_year' not in df.columns:
        df['calendar_year'] = df['hydro_year'] - 1
    if 'hydro_year' not in df.columns:
        df['hydro_year'] = df['calendar_year'] + 1
    return df

def standardize_region_format(df):
    """Ensure region codes are zero-padded to 2 digits (e.g., '1' -> '01')."""
    if 'region' in df.columns:
        # Convert to string and pad with zeros
        df['region'] = df['region'].astype(str).str.zfill(2)
        print(f"Standardized region format. Unique regions: {sorted(df['region'].unique())}")
    return df

def load_csv_from_upload(uploaded_file):
    """Load CSV data, validate, and filter by selected model/scenario."""
    content = io.BytesIO(uploaded_file["content"])
    df = pd.read_csv(content)

    # STEP 1: Handle wide format FIRST (before checking for year columns)
    df = reshape_wide_to_long(df)
    
    # STEP 2: Standardize region format
    df = standardize_region_format(df)
    
    # STEP 3: Now check for required columns (excluding model/scenario for now)
    base_required_cols = {"region", "rgi_id", "volume", "area"}
    validate_required_columns(df, base_required_cols)
    
    # STEP 4: Ensure calendar_year and hydro_year exist
    df = ensure_calendar_and_hydro_year(df)

    # STEP 5: Handle missing model/scenario columns
    if 'model' not in df.columns:
        print("'model' column not found. Setting all values to 'custom'.")
        df['model'] = 'custom'
    
    if 'scenario' not in df.columns:
        print("'scenario' column not found. Setting all values to 'custom'.")
        df['scenario'] = 'custom'
    
    # Check if there's only one unique model/scenario value
    unique_models = df['model'].unique()
    unique_scenarios = df['scenario'].unique()
    
    if len(unique_models) == 1 and len(unique_scenarios) == 1:
        print(f"Found single model: {unique_models[0]}, scenario: {unique_scenarios[0]}")
        # Use the actual values from the file
        actual_model = unique_models[0]
        actual_scenario = unique_scenarios[0]
    else:
        print(f"Multiple models ({len(unique_models)}) or scenarios ({len(unique_scenarios)}) found.")
        print(f"Models: {unique_models}")
        print(f"Scenarios: {unique_scenarios}")
        # Filter by user-selected values
        df = df[(df["model"] == selected_model_text) & (df["scenario"] == selected_scenario_text)]
        if df.empty:
            raise ValueError(f"No data found for model={selected_model_text}, scenario={selected_scenario_text} in CSV.")
        return df
    
    # If there's only one model/scenario, we can use all the data
    print(f"Using all data from the file (model={actual_model}, scenario={actual_scenario})")
    return df

def collect_model_scenario_files(link_path):
    """
    Collect files by (model, scenario).
    Returns a dict: { (model, scenario): [file1, file2, ...] }
    """
    all_files = [f for f in os.listdir(link_path) if f.endswith(".nc")]
    if not all_files:
        raise ValueError(f"No .nc files found in {link_path}")

    model_scenario_map = {}
    for fname in all_files:
        parts = fname.split("_")
        if len(parts) < 7:
            print(f"Skipping {fname}: does not follow OGGM naming convention.")
            continue
        model = parts[5]
        scenario = parts[6]
        model_scenario_map.setdefault((model, scenario), []).append(fname)
    return model_scenario_map

def compute_mass_elevation_changes(ds):
    """
    Compute dmdt and dhdt from xarray Dataset.
    """
    volume = ds["volume"]
    area = ds["area"]

    # Mass change (Gt/year)
    mass = volume * 900  # kg
    mass_change = mass.diff(dim="time") / 1e12  # Gt/year

    # Elevation change (m/year)
    volume_change = volume.diff(dim="time")
    elevation_change = volume_change / area

    # Handle invalid cases
    invalid = ((area == 0) & (volume_change != 0)) | ((area != 0) & (volume_change == 0))
    elevation_change = elevation_change.where(~invalid, np.nan)
    both_zero = (area == 0) & (volume_change == 0)
    elevation_change = elevation_change.where(~both_zero, 0)

    return elevation_change, mass_change

def compute_5yr_averages(df):
    """Add 5-year averaged columns for dmdt and dhdt."""
    block_size = 5
    calendar_blocks = (df['calendar_year'] - df['calendar_year'].min()) // block_size
    df['dmdt_5yr_avg'] = df.groupby(calendar_blocks)['dmdt'].transform('mean')
    df['dhdt_5yr_avg'] = df.groupby(calendar_blocks)['dhdt'].transform('mean')
    
    # Add calendar_year_5yr column (same as OGGM output)
    df['calendar_year_5yr'] = np.nan
    for block_id, block_indices in df.groupby(calendar_blocks).groups.items():
        last_year_idx = block_indices[-1]
        df.loc[last_year_idx, 'calendar_year_5yr'] = df.loc[last_year_idx, 'calendar_year']

    # Set all but last year of block to NaN
    for block in df.groupby(calendar_blocks).groups.values():
        df.loc[block[:-1], ['dmdt_5yr_avg', 'dhdt_5yr_avg']] = np.nan

    return df

def reshape_wide_to_long(df):
    """Convert wide format (years as columns) to long format."""
    # Identify year columns (assuming they're numeric and likely years)
    year_cols = []
    for col in df.columns:
        try:
            year = int(col)
            if 1900 <= year <= 2200:  # reasonable year range
                year_cols.append(col)
        except (ValueError, TypeError):
            continue
    
    if not year_cols:
        print("No year columns detected. Assuming data is already in long format.")
        return df
    
    print(f"Detected wide format with year columns: {sorted(year_cols)}")
    
    # Keep non-year columns as identifiers
    id_cols = [col for col in df.columns if col not in year_cols]
    
    # Melt the DataFrame
    df_long = pd.melt(df, 
                      id_vars=id_cols, 
                      value_vars=year_cols,
                      var_name='calendar_year', 
                      value_name='volume')
    
    df_long['calendar_year'] = pd.to_numeric(df_long['calendar_year'])
    
    # Remove rows with NaN volume values
    df_long = df_long.dropna(subset=['volume'])
    
    print(f"Converted to long format: {len(df_long)} rows")
    return df_long

# Add these variables at the top after the imports section:

# Add these variables after the existing global variables
all_rgi_ids_by_region = {}
reference_rows = []

# Add reference glaciers list (same as in oggm_output_cleaning_hugonnet.py)
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

# Modify the process_nc_combination function to track RGI IDs and save reference glaciers:
def process_nc_combination(link_path, files, model, scenario, regions):
    """
    Load and process all .nc files for a single (model, scenario) combination.
    Loops through regions, producing summary_df for each.
    """
    summary_frames = []
    for region in regions:
        # Ensure region is zero-padded
        region = str(region).zfill(2)
        
        ds_list = [xr.open_dataset(os.path.join(link_path, f)) for f in files if region in f]
        if not ds_list:
            print(f"No data found for region={region} in model={model}, scenario={scenario}")
            continue

        combined_ds = xr.concat(ds_list, dim="rgi_id")
        
        # Track RGI IDs for this region
        rgi_ids = combined_ds["rgi_id"].values
        all_rgi_ids_by_region[region] = set(rgi_ids)

        # Compute changes
        elevation_change, mass_change = compute_mass_elevation_changes(combined_ds)
        
        # Save reference glacier data
        calendar_years = combined_ds["time"].values[1:]  # after diff
        hydro_years = calendar_years + 1  # Convert calendar to hydro year
        
        for idx, rgi_id in enumerate(rgi_ids):
            if rgi_id in reference_glaciers:
                for t in range(len(calendar_years)):
                    reference_rows.append({
                        "rgi_id": rgi_id,
                        "region": region,
                        "hydro_year": hydro_years[t],
                        "calendar_year": calendar_years[t],
                        "dmdt": mass_change[t, idx].item(),
                        "dhdt": elevation_change[t, idx].item()
                    })

        # Aggregate by year
        dhdt_simple_avg = elevation_change.mean(dim="rgi_id")
        dmdt_sum = mass_change.sum(dim="rgi_id")

        # Build summary DataFrame
        summary_df = pd.DataFrame({
            "calendar_year": calendar_years,
            "hydro_year": hydro_years,  # Add this line
            "dmdt": dmdt_sum.values,
            "dhdt": dhdt_simple_avg.values,
            "model": model,
            "scenario": scenario,
            "region": region
        })
        summary_df = compute_5yr_averages(summary_df)
        summary_frames.append(summary_df)

    return pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

def load_multiple_csv_files(uploaded_files):
    """Load and combine multiple CSV files."""
    all_dfs = []
    
    for uploaded_file in uploaded_files:
        print(f"Processing file: {uploaded_file.name}")
        content = io.BytesIO(uploaded_file.content)
        df = pd.read_csv(content)
        
        # Apply same processing steps
        df = reshape_wide_to_long(df)
        df = standardize_region_format(df)
        base_required_cols = {"region", "rgi_id", "volume", "area"}
        validate_required_columns(df, base_required_cols)
        df = ensure_calendar_and_hydro_year(df)
        
        # Handle missing model/scenario columns
        if 'model' not in df.columns:
            df['model'] = 'custom'
        if 'scenario' not in df.columns:
            df['scenario'] = 'custom'
            
        all_dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Handle model/scenario filtering logic
    unique_models = combined_df['model'].unique()
    unique_scenarios = combined_df['scenario'].unique()
    
    if len(unique_models) == 1 and len(unique_scenarios) == 1:
        print(f"Found single model: {unique_models[0]}, scenario: {unique_scenarios[0]}")
        return combined_df
    else:
        print(f"Multiple models ({len(unique_models)}) or scenarios ({len(unique_scenarios)}) found.")
        filtered_df = combined_df[(combined_df["model"] == selected_model_text) & 
                                 (combined_df["scenario"] == selected_scenario_text)]
        if filtered_df.empty:
            raise ValueError(f"No data found for model={selected_model_text}, scenario={selected_scenario_text}")
        return filtered_df

def process_link_data_by_region(link_path, selected_regions):
    """Process remote NetCDF files organized by region folders (OGGM-style)."""
    summary_frames = []
    for region in selected_regions:
        region = str(region).zfill(2)
        region_url = f"{link_path}/RGI{region}/"
        print(f"Processing region {region} from {region_url}")

        # List files in the remote region folder
        response = requests.get(region_url)
        if response.status_code != 200:
            print(f"Could not access {region_url}. Skipping region {region}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        file_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.nc')]

        # Filter files for selected model and scenario
        selected_files = [
            region_url + fname
            for fname in file_links
            if (
                (parts := fname.split("_")) and
                len(parts) >= 7 and
                parts[5] == selected_model_text and
                parts[6] == selected_scenario_text
            )
        ]
        if not selected_files:
            print(f"No files found for model={selected_model_text}, scenario={selected_scenario_text} in region {region}")
            continue

        ds_list = []
        for file_url in selected_files:
            print(f"Downloading and opening {file_url}")
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
                    for chunk in r.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    tmp.flush()
                    ds = xr.open_dataset(tmp.name)
                    ds_list.append(ds)

        if not ds_list:
            print(f"No datasets loaded for region {region}")
            continue

        combined_ds = xr.concat(ds_list, dim="rgi_id")
        rgi_ids = combined_ds["rgi_id"].values
        all_rgi_ids_by_region[region] = set(rgi_ids)
        elevation_change, mass_change = compute_mass_elevation_changes(combined_ds)
        calendar_years = combined_ds["time"].values[1:]
        hydro_years = calendar_years + 1
        for idx, rgi_id in enumerate(rgi_ids):
            if rgi_id in reference_glaciers:
                for t in range(len(calendar_years)):
                    reference_rows.append({
                        "rgi_id": rgi_id,
                        "region": region,
                        "hydro_year": hydro_years[t],
                        "calendar_year": calendar_years[t],
                        "dmdt": mass_change[t, idx].item(),
                        "dhdt": elevation_change[t, idx].item()
                    })
        dhdt_simple_avg = elevation_change.mean(dim="rgi_id")
        dmdt_sum = mass_change.sum(dim="rgi_id")
        summary_df = pd.DataFrame({
            "calendar_year": calendar_years,
            "hydro_year": hydro_years,
            "dmdt": dmdt_sum.values,
            "dhdt": dhdt_simple_avg.values,
            "model": selected_model_text,
            "scenario": selected_scenario_text,
            "region": region
        })
        summary_df = compute_5yr_averages(summary_df)
        summary_frames.append(summary_df)
    return pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()



# Modify the CSV processing section to track RGI IDs and save reference glaciers:
if source_type == "Upload":
    # Handle multiple file uploads
    if isinstance(uploaded_file, (list, tuple)):
        custom_model_data = load_multiple_csv_files(uploaded_file)
    else:
        custom_model_data = load_multiple_csv_files([uploaded_file])
    
    # Process all regions at once
    summary_frames = []
    for region in selected_regions:
        region = str(region).zfill(2)
        df_region = custom_model_data[custom_model_data["region"] == region]
        if df_region.empty:
            print(f"No data for region {region} in CSV files.")
            continue

        # Track RGI IDs for this region
        all_rgi_ids_by_region[region] = set(df_region["rgi_id"].unique())

        # Calculate mass and elevation change
        df_region = df_region.sort_values(["region", "calendar_year", "rgi_id"])
        df_region["dmdt"] = df_region.groupby("rgi_id")["volume"].diff() * 900 / 1e12
        df_region["dhdt"] = df_region.groupby("rgi_id")["volume"].diff() / df_region["area"]

        # Save reference glacier data
        for _, row in df_region.iterrows():
            if row["rgi_id"] in reference_glaciers and pd.notna(row["dmdt"]):
                reference_rows.append({
                    "rgi_id": row["rgi_id"],
                    "region": region,
                    "hydro_year": row["hydro_year"],
                    "calendar_year": row["calendar_year"],
                    "dmdt": row["dmdt"],
                    "dhdt": row["dhdt"]
                })

        # Aggregate by year for region summary
        region_summary = df_region.groupby(["calendar_year", "hydro_year"]).agg({
            "dmdt": "sum",
            "dhdt": "mean"
        }).reset_index()
        region_summary["model"] = selected_model_text
        region_summary["scenario"] = selected_scenario_text
        region_summary["region"] = region

        region_summary = compute_5yr_averages(region_summary)
        summary_frames.append(region_summary)

    summary_df = pd.concat(summary_frames, ignore_index=True)

elif source_type == "Link":
    # Use the new region-based processing
    summary_df = process_link_data_by_region(link_path, selected_regions)

else:
    raise ValueError("Invalid source_type. Must be 'Upload' or 'Link'.")

print("Custom model data loaded and cleaned successfully.")
print(f"Processed model={selected_model_text}, scenario={selected_scenario_text}")

# Remove the duplicate save section and keep only:
os.makedirs(os.path.dirname(output_summary_csv), exist_ok=True)
summary_df.to_csv(output_summary_csv, index=False)
print(f"Summary statistics saved to {output_summary_csv}")

