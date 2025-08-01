# import pandas as pd
# import numpy as np
# import os
# import matplotlib.pyplot as plt

#################################################################################################################
###################### Definition for cleaning up mass change data for selected region ##########################
#################################################################################################################

# definition to clean the data and export it to a csv file
def aggregate_from_raw_hugonnet(region_id, base_dir, all_rgi_ids=None):
    """
    Aggregate data from raw Hugonnet files and export to a CSV file.
    
    Parameters:
    - region_id: str, identifier for the region
    - base_dir: str, base directory where the data is stored
    - all_rgi_ids: list, list of all RGI IDs to filter the data

    """
    # Define the path to the raw data file
    raw_data_path = os.path.join(base_dir, f"dh_{region_id}_rgi60_pergla_rates.csv")

    # Load the raw data
    df = pd.read_csv(raw_data_path)

    if all_rgi_ids is not None:
        df = df[df["rgiid"].isin(all_rgi_ids)]
        # print which RGIIDs are missing
        missing_rgi_ids = set(all_rgi_ids) - set(df["rgiid"])
        if missing_rgi_ids:
            print(f"[!] Missing RGIIDs in series for {region_id}: {missing_rgi_ids}")

    # Extract start and end years as integers
    df['start_year'] = df['period'].str.split('_').str[0].str[:4].astype(int)
    df['calendar_year'] = df['period'].str.split('_').str[1].str[:4].astype(int)

    # Keep only rows where the difference is 1 year
    df_five_year = df[df['calendar_year'] - df['start_year'] == 5].copy()
    df_five_year = df_five_year.drop(columns=['start_year'])

    # Convert 'region' to string and zero-fill it to 2 digits
    df_five_year['region'] = df_five_year['reg'].astype(str).str.zfill(2)

    
    def weighted_stats(g):
        #weights = g['area']
        dhdt = g['dhdt']
        se_dhdt = g['err_dhdt']
        dmdt = g['dmdt']
        se_dmdt = g['err_dmdt']

        # dhdt weighted mean and CI
        #weighted_mean = (weights * dhdt).sum() / weights.sum()
        dhdt_mean = dhdt.mean()
        #weighted_se = np.sqrt((weights**2 * se_dhdt**2).sum()) / weights.sum()
        n = len(se_dhdt)
        simple_se = np.sqrt((se_dhdt**2).sum()) / n
        dhdt_ci_lower = dhdt_mean - 1.96 * simple_se
        dhdt_ci_upper = dhdt_mean + 1.96 * simple_se

        # dmdt sum and CI
        total_dmdt = dmdt.sum()
        total_se_dmdt = np.sqrt((se_dmdt**2).sum())
        dmdt_ci_lower = total_dmdt - 1.96 * total_se_dmdt
        dmdt_ci_upper = total_dmdt + 1.96 * total_se_dmdt

        return pd.Series({
            'dmdt': total_dmdt,
            'dmdt_ci_lower': dmdt_ci_lower,
            'dmdt_ci_upper': dmdt_ci_upper,
            'dhdt': dhdt_mean,
            'dhdt_ci_lower': dhdt_ci_lower,
            'dhdt_ci_upper': dhdt_ci_upper
        })

    summary = (
    df_five_year.groupby(['region', 'calendar_year'])
    .apply(weighted_stats, include_groups=False)
    .reset_index()
    )
    
    return summary

# selected_regions = ["06","07","09"]  
# base_dir = "/Users/erika/Desktop/hugonnet_timeseries"
if "12" in selected_regions:
     
    # File paths
    series_file = os.path.join(base_dir, "dh_12_rgi60_pergla_rates.csv")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conversion_file = os.path.join(script_dir, '12_GLIMSId_RGIId_dict.csv')
    # Load DataFrames
    hugonnet_series_df = pd.read_csv(series_file)
    conversion_df = pd.read_csv(conversion_file)

    # Process if hugonnet_series_df does not contain RGIID column
    if 'RGIId' not in hugonnet_series_df.columns:
        print("Processing Hugonnet files region 12 to add RGIId...")
       
        # Clean up: strip spaces from column names and values
        for df in [hugonnet_series_df, conversion_df]:
            df.columns = df.columns.str.strip()
            for col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Map GLIMS_ID → RGIId6
        glims_to_rgi = dict(zip(conversion_df.iloc[:,0], conversion_df.iloc[:,1]))
        hugonnet_series_df['rgiid'] = hugonnet_series_df['rgiid'].map(glims_to_rgi)

        # Save the updated DataFrames
        hugonnet_series_df.to_csv(series_file, index=False)

# cycle through regions selected by the user
#selected_regions = ["06"]
df_all_obs = []
for region in selected_regions:
    print(f"Processing region {region}...")
    df_region = aggregate_from_raw_hugonnet(region, base_dir_hugonnet, all_rgi_ids=all_rgi_ids_by_region.get(region))
    df_all_obs.append(df_region)
# Concatenate all region dataframes into a single dataframe
df_all = pd.concat(df_all_obs, ignore_index=True)
output_file_hugonnet = '/Users/erika/CmCt/glacier/glacier_data/hugonnet_aggregated_data.csv'
os.makedirs(os.path.dirname(output_file_hugonnet), exist_ok=True)
df_all.to_csv(output_file_hugonnet, index=False)
print(f"Aggregated data saved to {output_file_hugonnet}")


# plot all regions for dmdt and dhdt with CI, side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

for region in selected_regions:
    region_data = df_all[df_all['region'] == region]
    # DMDT plot
    axes[0].plot(region_data['calendar_year'], region_data['dmdt'], label=f'Region {region}')
    axes[0].fill_between(region_data['calendar_year'], 
                         region_data['dmdt_ci_lower'], 
                         region_data['dmdt_ci_upper'], 
                         alpha=0.2)
    # DHDT plot
    axes[1].plot(region_data['calendar_year'], region_data['dhdt'], label=f'Region {region}')
    axes[1].fill_between(region_data['calendar_year'], 
                         region_data['dhdt_ci_lower'], 
                         region_data['dhdt_ci_upper'], 
                         alpha=0.2)

axes[0].set_xlabel('Year')
axes[0].set_ylabel('DMDT (Gt/yr)')
axes[0].set_title('DMDT with Confidence Intervals by Region')
axes[0].legend()
axes[0].grid()

axes[1].set_xlabel('Year')
axes[1].set_ylabel('DHDT (m/yr)')
axes[1].set_title('DHDT with Confidence Intervals by Region')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()


