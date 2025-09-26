import logging

import geopandas as gpd

# Create module-specific logger
logger = logging.getLogger(__name__)


def load_basin_polygons(shapefile_path):
    """
    Load basin polygons from a shapefile and return a dictionary of polygons.

    Parameters:
    shapefile_path (str): Path to the shapefile containing basin polygons.

    Returns:
    dict: A dictionary where keys are basin names and values are Shapely Polygon objects.
    """
    gdf = gpd.read_file(shapefile_path)
    basin_polygons = {}

    for idx, row in gdf.iterrows():
        basin_name = row[
            "SUBREGION1"
        ]  # Column name in GRE_Basins_IMBIE2_v1.3 shapefile
        basin_polygons[basin_name] = row["geometry"]

    return basin_polygons
