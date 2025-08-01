# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# import geopandas as gpd
# import os
# import pandas as pd
# from shapely.geometry import Point

#global_file_path = os.path.join(unzipped_folder, "/global-gridded/global-gridded-annual-glacier-mass-change.nc4")
#global_file_path = '/Users/erika/CmCt/glacier/glacier_data/wgms-amce-2025-02b/global-gridded/global-gridded-annual-glacier-mass-change.nc4'
global_file_path = os.path.join(extract_dir, 'wgms-amce-2025-02b/global-gridded/global-gridded-annual-glacier-mass-change.nc4')
ds = xr.open_dataset(global_file_path)
#print(ds)

# Open shape file with RGI region outline coordinates
#shapefile_path = 'global_comparison_shapefile/00_rgi60_O1Regions.shp'
gdf = gpd.read_file(shapefile_path)

# Ensure the RGI_CODE is a string and zero-padded to 2 digits
gdf['RGI_CODE'] = gdf['RGI_CODE'].astype(str).str.zfill(2)

# Extract needed variables from the dataset
data = ds['glacier_mass_change_gt'].values
lons = ds['lon'].values
lats = ds['lat'].values

# take first time step (how the regions are assigned is all the same for all years)
data = data[0]

# Get indices where data is not NaN
idx = np.where(~np.isnan(data))
# Get lon/lat for those indices
lon_points = lons[idx[1]]
lat_points = lats[idx[0]]

points_df = pd.DataFrame({'lon': lon_points, 'lat': lat_points})
points_gdf = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df['lon'], points_df['lat']),
    crs=gdf.crs  # Use the same CRS as your region polygons
) # Generate GeometryArray of shapely Point geometries


joined = gpd.sjoin(points_gdf, gdf[['RGI_CODE', 'geometry']], how='left', predicate='within')
# Now, joined['RGI_CODE'] gives the region code for each glacier grid point
# print(joined.head()) #OK #21 nan values


# Map the region codes back to the original grid shape and fill with NaN where no region is found
region_mask = np.full(data.shape, np.nan, dtype=object)
region_mask[idx] = joined['RGI_CODE'].values

ds = ds.assign(region_mask=(('lat', 'lon'), region_mask))
#print(ds)
