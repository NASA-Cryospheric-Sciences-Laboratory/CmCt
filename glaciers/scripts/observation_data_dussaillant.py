import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import urllib.request
import zipfile
import requests
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#################################################################################################################
#################################### Download data if not already downloaded ####################################
#################################################################################################################
# for testing
selected_regions = ["06"]


# Define paths
data_url = "https://wgms.ch/downloads/wgms-amce-2025-02b.zip"
local_zip = "/Users/erika/Documents/GitHub/CmCt/glaciers/data/dussailant_data/wgms-amce-2025-02b.zip"
extract_dir = "/Users/erika/Documents/GitHub/CmCt/glaciers/data/dussailant_data"

unzipped_folder = os.path.join(extract_dir, "wgms-amce-2025-02b")



# Download if not already present
if not os.path.isfile(local_zip):
    os.makedirs(os.path.dirname(local_zip), exist_ok=True)
    print("Downloading WGMS AMCE dataset...")
    urllib.request.urlretrieve(data_url, local_zip)

    # Unzip
    print("Extracting files...")
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Data downloaded and extracted to {extract_dir}")
else:
    print("Data already present, skipping download.")


#################################################################################################################
############################## Combine subregions for region 17 if in selected region ###########################
#################################################################################################################

# no overlap of RGIIDs between SA1 and SA2  
if "17" in selected_regions:
    base_dir = os.path.join(unzipped_folder, "individual-glacier")

    output_file_mass = os.path.join(base_dir, "SUA_gla_MEAN-CAL-mass-change-series_obs_unobs.csv")
    output_file_error = os.path.join(base_dir, "SUA_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv")

    # Only process if the combined files do not already exist
    if not (os.path.exists(output_file_mass) and os.path.exists(output_file_error)):
        # Match both SA1 and SA2 files
        region_files_mass = glob.glob(os.path.join(base_dir, "SA*_gla_MEAN-CAL-mass-change-series_obs_unobs.csv"))
        region_files_error = glob.glob(os.path.join(base_dir, "SA*_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv"))

        # Read and concatenate
        dfs_mass = []
        for file in region_files_mass:
            df_mass = pd.read_csv(file)
            dfs_mass.append(df_mass)

        dfs_error = []
        for file in region_files_error:
            df_error = pd.read_csv(file)
            dfs_error.append(df_error)

        # Combine into one DataFrame
        df_combined_mass = pd.concat(dfs_mass, ignore_index=True)
        df_combined_error = pd.concat(dfs_error, ignore_index=True)

        df_combined_mass.to_csv(output_file_mass, index=False)
        df_combined_error.to_csv(output_file_error, index=False)

        print(f"Combined {len(region_files_mass)} files for region 17")
    else:
        print("Region 17 combined files already exist, skipping.")


#################################################################################################################
########################## Convert Indexing for region 12 from GLIMS to RGI if selected #########################
#################################################################################################################

if "12" in selected_regions:
     
    # File paths
    series_file = os.path.join(unzipped_folder, "individual-glacier/CAU_gla_MEAN-CAL-mass-change-series_obs_unobs.csv")
    error_file = os.path.join(unzipped_folder, "individual-glacier/CAU_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv")
    script_dir = os.getcwd()
    conversion_file = "../data/12_GLIMSId_RGIId_dict.csv"
    # Load DataFrames
    wgms_series_df = pd.read_csv(series_file)
    wgms_error_df = pd.read_csv(error_file)
    conversion_df = pd.read_csv(conversion_file)

    # Process if wgms_series_df does not contain RGIID column
    if 'RGIId' not in wgms_series_df.columns and 'RGIId' not in wgms_error_df.columns:
        print("Processing WGMS files region 12 to add RGIId...")
       
        # Clean up: strip spaces from column names and values
        for df in [wgms_series_df, wgms_error_df, conversion_df]:
            df.columns = df.columns.str.strip()
            for col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Map GLIMS_ID → RGIId6
        glims_to_rgi = dict(zip(conversion_df.iloc[:,0], conversion_df.iloc[:,1]))
        wgms_series_df['RGIId'] = wgms_series_df['GLIMS_ID'].map(glims_to_rgi)
        wgms_error_df['RGIId'] = wgms_error_df['GLIMS_ID'].map(glims_to_rgi)

        # Save the updated DataFrames
        wgms_series_df.to_csv(series_file, index=False)
        wgms_error_df.to_csv(error_file, index=False)


#################################################################################################################
###################### Definition for cleaning up mass change data for selected region ##########################
#################################################################################################################

# Shape check: all series and error files have the same shape
def aggregate_from_raw(region_id, base_dir, all_rgi_ids=None):
    """
    Loads raw series + error files, aggregates total mass change and CI.
    Returns a DataFrame with region, year, mass_change, std, ci_lower, ci_upper.
    """
    # Define mapping from region ID to RGI code
    RGI_ID_TO_NAME = {
        "01": "ALA", "02": "WNA", "03": "ACN", "04": "ACS", "05": "GRL",
        "06": "ISL", "07": "SJM", "08": "SCA", "09": "RUA", "10": "ASN",
        "11": "CEU", "12": "CAU", "13": "ASC", "14": "ASW", "15": "ASE",
        "16": "TRP", "17": "SUA", "18": "NZL", "19": "ANT"
    }
    
    region_code = RGI_ID_TO_NAME.get(region_id)
    if not region_code:
        print(f"Invalid region ID: {region_id}")
        return None
    
    # Construct file paths
    series_file = os.path.join(base_dir, f"{region_code}_gla_MEAN-CAL-mass-change-series_obs_unobs.csv")
    error_file = os.path.join(base_dir, f"{region_code}_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv")
    
    if not os.path.exists(series_file):
        print(f"[!] Missing series file for {region_code}")
        return None

    # Load data
    df_series = pd.read_csv(series_file)
    df_error = pd.read_csv(error_file) if os.path.exists(error_file) else None
    
    #Process comparison only for rgi ids that exist in the model output
    if all_rgi_ids is not None:
        df_series = df_series[df_series["RGIId"].isin(all_rgi_ids)]
        # print which RGIIDs are missing
        missing_rgi_ids = set(all_rgi_ids) - set(df_series["RGIId"])
        if missing_rgi_ids:
            print(f"[!] Missing RGIIDs in series for {region_code}: {missing_rgi_ids}")
        if df_error is not None:
            df_error = df_error[df_error["RGIId"].isin(all_rgi_ids)]
            missing_rgi_ids_error = set(all_rgi_ids) - set(df_error["RGIId"])
            if missing_rgi_ids_error:
                print(f"[!] Missing RGIIDs in error for {region_code}: {missing_rgi_ids_error}")

    #print(f"Existing RGIIDs: {df_series['RGIId']}")
    #print which regions are missing
    if "Area" not in df_series.columns:
        raise ValueError("Area column not found in series file.")

    # Convert area to m²
    area_m2 = df_series["Area"].values * 1e6

    # Identify year columns and extract corresponding mass change values
    year_cols = [col for col in df_series.columns if col.isdigit()]
    years = list(map(int, year_cols))
    mwe_values = df_series[year_cols].values
    errors = df_error[year_cols].values if df_error is not None else np.zeros_like(mwe_values)
    
    # Multiply each glacier's mass change by its area 
    mass_change_kg = mwe_values * area_m2[:, np.newaxis]*1000  # shape: (n_glaciers, n_years)
    error_kg = errors * area_m2[:, np.newaxis]*1000
    
    #convert to Gt
    mass_change_kg /= 1e12 
    error_kg /= 1e12 

    # Aggregate across glaciers (per year)
    total_mass_change = mass_change_kg.sum(axis=0)
    total_std = np.sqrt((error_kg ** 2).sum(axis=0))
    ci_upper = total_mass_change + 1.96 * total_std
    ci_lower = total_mass_change - 1.96 * total_std

    return pd.DataFrame({
        "region": region_id,
        "region_code": region_code,
        "hydro_year": years,
        "dmdt": total_mass_change,
        "std": total_std,
        "dmdt_ci_lower": ci_lower,
        "dmdt_ci_upper": ci_upper
    })

#################################################################################################################
################################### Run defenition for selected regions #########################################
#################################################################################################################

# output_file = "/Users/erika/CmCt/glacier/glacier_data/combined_regional_mass_change.csv"
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
all_region_dfs = []

base_dir = os.path.join(unzipped_folder, "individual-glacier")
for region_id in sorted(selected_regions):
    print(f"Processing {region_id}...")
    region_rgi_ids = all_rgi_ids_by_region.get(region_id)
    df_region = aggregate_from_raw(region_id, base_dir, region_rgi_ids)
    if df_region is not None:
        all_region_dfs.append(df_region)

df_all = pd.concat(all_region_dfs, ignore_index=True)
# df_all.to_csv(output_file, index=False)
# print(f"\n Saved combined regional mass change data to:\n{output_file}")

#################################################################################################################
################################ plots for mass change in selected region #######################################
#################################################################################################################

plt.figure(figsize=(10, 5))
cmap = plt.get_cmap("tab20")  

for idx, region_to_plot in enumerate(selected_regions):
    df_region = df_all[df_all["region"] == region_to_plot]
    color = cmap(idx % cmap.N)  # Cycle through colors if more regions than colors in the map
    plt.plot(df_region["hydro_year"], df_region["dmdt"], label=f"Mass Change RGI{region_to_plot}", color=color)
    plt.fill_between(df_region["hydro_year"], df_region["dmdt_ci_lower"], df_region["dmdt_ci_upper"], color=color, alpha=0.3, label=None)

plt.title("Regional Mass Change")
plt.xlabel("Year")
plt.ylabel("Mass Change (Gt)")
#plt.xlim(1980,1990)
#plt.ylim(-0.5, 0.5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


