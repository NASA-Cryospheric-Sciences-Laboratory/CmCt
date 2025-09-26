import array
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr


def compute_residuals_and_stats(gsfc_values, model_values, x_indices, y_indices):
    """Compute residuals and statistics using pure Python."""
    n_points = len(x_indices)
    residuals = [float("nan")] * n_points

    valid_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    residual_sum = 0.0

    for i in range(n_points):
        gsfc_val = gsfc_values[y_indices[i]][x_indices[i]]
        model_val = model_values[y_indices[i]][x_indices[i]]

        if not (math.isnan(gsfc_val) or math.isnan(model_val)):
            residual = gsfc_val - model_val
            residuals[i] = residual
            valid_count += 1
            residual_sum += residual
            abs_sum += abs(residual)
            sq_sum += residual * residual

    return residuals, valid_count, abs_sum, sq_sum, residual_sum


def point_in_polygon(
    x: float, y: float, polygon_x: List[float], polygon_y: List[float]
) -> bool:
    """Point-in-polygon using winding number algorithm."""
    n = len(polygon_x)
    winding_number = 0

    for i in range(n):
        j = (i + 1) % n
        if polygon_y[i] <= y:
            if polygon_y[j] > y:  # Upward crossing
                cross_product = (polygon_x[j] - polygon_x[i]) * (y - polygon_y[i]) - (
                    x - polygon_x[i]
                ) * (polygon_y[j] - polygon_y[i])
                if cross_product > 0:
                    winding_number += 1
        else:
            if polygon_y[j] <= y:  # Downward crossing
                cross_product = (polygon_x[j] - polygon_x[i]) * (y - polygon_y[i]) - (
                    x - polygon_x[i]
                ) * (polygon_y[j] - polygon_y[i])
                if cross_product < 0:
                    winding_number -= 1

    return winding_number != 0


def assign_basins_batch(
    x_coords: List[float],
    y_coords: List[float],
    basin_polygons_x: List[float],
    basin_polygons_y: List[float],
    basin_lengths: List[int],
    batch_start: int,
    batch_end: int,
) -> List[int]:
    """Assign basin IDs to a batch of points."""
    batch_size = batch_end - batch_start
    basin_ids = [-1] * batch_size
    n_basins = len(basin_lengths)

    for i in range(batch_size):
        x = x_coords[batch_start + i]
        y = y_coords[batch_start + i]

        polygon_start = 0
        for basin_idx in range(n_basins):
            polygon_end = polygon_start + basin_lengths[basin_idx]
            poly_x = basin_polygons_x[polygon_start:polygon_end]
            poly_y = basin_polygons_y[polygon_start:polygon_end]

            if point_in_polygon(x, y, poly_x, poly_y):
                basin_ids[i] = basin_idx
                break

            polygon_start = polygon_end

    return basin_ids


def prepare_basin_polygons(basin_polygons_dict, target_crs="EPSG:3413"):
    """Convert basin polygons to Python lists with proper coordinate transformation."""
    from pyproj import Transformer
    from shapely.ops import transform

    basin_names = []
    all_x = []
    all_y = []
    basin_lengths = []

    # Create transformer from WGS84 (geographic) to polar stereographic
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    for name, polygon in basin_polygons_dict.items():
        if name == "unassigned":
            continue

        # Transform polygon from geographic to projected coordinates
        try:
            projected_polygon = transform(transformer.transform, polygon)
            coords = list(projected_polygon.exterior.coords)

            basin_names.append(name)
            for coord in coords:
                all_x.append(float(coord[0]))
                all_y.append(float(coord[1]))
            basin_lengths.append(len(coords))

            logging.info(f"Transformed basin {name}: {len(coords)} points")

        except Exception as e:
            logging.error(f"Failed to transform basin {name}: {e}")
            continue

    return basin_names, all_x, all_y, basin_lengths


def searchsorted(arr: List[float], values: List[float]) -> List[int]:
    """Python implementation of searchsorted."""
    indices = []
    for val in values:
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < val:
                left = mid + 1
            else:
                right = mid
        indices.append(left)
    return indices


def clip(values: List[int], min_val: int, max_val: int) -> List[int]:
    """Clip values to range."""
    return [max(min_val, min(max_val, val)) for val in values]


def process_year_data(
    gsfc_data,
    model_data,
    x_coords: List[float],
    y_coords: List[float],
    basin_polygons_x: List[float],
    basin_polygons_y: List[float],
    basin_lengths: List[int],
    basin_names: List[str],
    year: int,
):
    """Process a single year of data with basin assignments."""

    # Get coordinate indices
    gsfc_x = list(gsfc_data.x.values)
    gsfc_y = list(gsfc_data.y.values)

    x_indices = searchsorted(gsfc_x, x_coords)
    x_indices = clip(x_indices, 0, len(gsfc_x) - 1)

    y_indices = searchsorted(gsfc_y, y_coords)
    y_indices = clip(y_indices, 0, len(gsfc_y) - 1)

    # Convert ice mask data to 2D lists
    gsfc_values = [list(row) for row in gsfc_data.ice_mask.values]
    model_values = [list(row) for row in model_data.ice_mask.values]

    # Flatten coordinates for processing
    x_flat = []
    y_flat = []
    x_idx_flat = []
    y_idx_flat = []

    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            x_flat.append(x)
            y_flat.append(y)
            x_idx_flat.append(x_indices[i])
            y_idx_flat.append(y_indices[j])

    # Compute residuals
    residuals, valid_count, abs_sum, sq_sum, residual_sum = compute_residuals_and_stats(
        gsfc_values, model_values, x_idx_flat, y_idx_flat
    )

    # Assign basins in sequential batches
    n_points = len(x_flat)
    batch_size = 50000
    basin_assignments = [-1] * n_points

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)
        batch_result = assign_basins_batch(
            x_flat,
            y_flat,
            basin_polygons_x,
            basin_polygons_y,
            basin_lengths,
            start,
            end,
        )
        basin_assignments[start:end] = batch_result

    # Reshape to 2D grids
    n_x, n_y = len(x_coords), len(y_coords)
    residual_grid = []
    basin_grid = []

    for i in range(n_y):
        residual_row = []
        basin_row = []
        for j in range(n_x):
            idx = j * n_y + i
            residual_row.append(residuals[idx])
            basin_row.append(basin_assignments[idx])
        residual_grid.append(residual_row)
        basin_grid.append(basin_row)

    # Get aligned ice masks
    gsfc_aligned = []
    model_aligned = []
    for y_idx in y_indices:
        gsfc_row = []
        model_row = []
        for x_idx in x_indices:
            gsfc_row.append(gsfc_values[y_idx][x_idx])
            model_row.append(model_values[y_idx][x_idx])
        gsfc_aligned.append(gsfc_row)
        model_aligned.append(model_row)

    # Calculate statistics
    avg_residual = abs_sum / valid_count if valid_count > 0 else 0.0
    rms_residual = math.sqrt(sq_sum / valid_count) if valid_count > 0 else 0.0

    stats = {
        "avg_abs_residual": round(avg_residual, 3),
        "rms_residual": round(rms_residual, 3),
        "sum_residual": round(residual_sum, 3),
        "valid_points": int(valid_count),
        "total_points": n_points,
    }

    return residual_grid, basin_grid, gsfc_aligned, model_aligned, stats


def create_basin_mask_optimized(
    x_coords: List[float],
    y_coords: List[float],
    basin_polygons_x: List[float],
    basin_polygons_y: List[float],
    basin_lengths: List[int],
) -> List[List[int]]:
    """Create basin mask using pure Python."""
    n_y, n_x = len(y_coords), len(x_coords)
    n_basins = len(basin_lengths)
    basin_mask = [[-1] * n_x for _ in range(n_y)]

    for i in range(n_y):
        for j in range(n_x):
            x = x_coords[j]
            y = y_coords[i]

            polygon_start = 0
            for basin_idx in range(n_basins):
                polygon_end = polygon_start + basin_lengths[basin_idx]
                poly_x = basin_polygons_x[polygon_start:polygon_end]
                poly_y = basin_polygons_y[polygon_start:polygon_end]

                if point_in_polygon(x, y, poly_x, poly_y):
                    basin_mask[i][j] = basin_idx
                    break

                polygon_start = polygon_end

    return basin_mask


def compute_residuals_vectorized(
    gsfc_data: List[List[List[float]]], model_data: List[List[List[float]]]
) -> List[List[List[float]]]:
    """Compute residuals using pure Python."""
    n_time = len(gsfc_data)
    n_y = len(gsfc_data[0])
    n_x = len(gsfc_data[0][0])

    residuals = []
    for t in range(n_time):
        time_slice = []
        for i in range(n_y):
            row = []
            for j in range(n_x):
                gsfc_val = gsfc_data[t][i][j]
                model_val = model_data[t][i][j]

                if not (math.isnan(gsfc_val) or math.isnan(model_val)):
                    row.append(gsfc_val - model_val)
                else:
                    row.append(float("nan"))
            time_slice.append(row)
        residuals.append(time_slice)

    return residuals


def compute_stats_vectorized(residuals: List[List[List[float]]]) -> List[List[float]]:
    """Compute statistics for residuals using pure Python."""
    n_time = len(residuals)
    stats = []

    for t in range(n_time):
        abs_sum = 0.0
        sq_sum = 0.0
        residual_sum = 0.0
        valid_count = 0
        total_points = len(residuals[t]) * len(residuals[t][0])

        for i in range(len(residuals[t])):
            for j in range(len(residuals[t][i])):
                val = residuals[t][i][j]
                if not math.isnan(val):
                    valid_count += 1
                    residual_sum += val
                    abs_sum += abs(val)
                    sq_sum += val * val

        if valid_count > 0:
            avg_abs = abs_sum / valid_count
            rms = math.sqrt(sq_sum / valid_count)
            stats.append(
                [avg_abs, rms, residual_sum, float(valid_count), float(total_points)]
            )
        else:
            stats.append(
                [float("nan"), float("nan"), float("nan"), 0.0, float(total_points)]
            )

    return stats


def create_calving_dataset(gsfc, model, years, basin_polygons_dict):
    """
    Create a comprehensive dataset with calving data across multiple years and basins.
    This version uses pure Python instead of NumPy.
    """

    logging.info("Starting calving dataset creation without NumPy...")
    start_time = time.time()

    # Prepare basin data once
    basin_names, basin_polygons_x, basin_polygons_y, basin_lengths = (
        prepare_basin_polygons(basin_polygons_dict)
    )

    # Get coordinate arrays - use model coordinates as reference
    model_sample = model.ds.sel(time=years[0])
    x_coords = list(model_sample.x.values)
    y_coords = list(model_sample.y.values)

    logging.info(f"Grid dimensions: {len(y_coords)} x {len(x_coords)}")

    # Collect GSFC data for all years
    gsfc_data_all = []
    for year in years:
        gsfc_year = gsfc.ds.sel(year=year)

        gsfc_x = list(gsfc_year.x.values)
        gsfc_y = list(gsfc_year.y.values)

        x_indices = searchsorted(gsfc_x, x_coords)
        x_indices = clip(x_indices, 0, len(gsfc_x) - 1)
        y_indices = searchsorted(gsfc_y, y_coords)
        y_indices = clip(y_indices, 0, len(gsfc_y) - 1)

        # Extract aligned data
        gsfc_values = gsfc_year.ice_mask.values
        gsfc_aligned = []
        for y_idx in y_indices:
            row = []
            for x_idx in x_indices:
                row.append(float(gsfc_values[y_idx, x_idx]))
            gsfc_aligned.append(row)
        gsfc_data_all.append(gsfc_aligned)

    # Collect model data for all years
    model_data_all = []
    for year in years:
        model_year = model.ds.sel(time=year)
        model_values = []
        for row in model_year.ice_mask.values:
            model_values.append([float(val) for val in row])
        model_data_all.append(model_values)

    logging.info("Computing residuals...")
    residuals_all = compute_residuals_vectorized(gsfc_data_all, model_data_all)

    logging.info("Computing statistics...")
    stats_array = compute_stats_vectorized(residuals_all)

    logging.info("Creating basin assignments...")

    # Create basin mask
    basin_mask = create_basin_mask_optimized(
        x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
    )

    # Count basin assignments
    unique_basins = set()
    for row in basin_mask:
        unique_basins.update(row)
    unique_basins = sorted(list(unique_basins))

    logging.info(f"Basin assignment complete. Unique basin IDs: {unique_basins}")

    for basin_id in unique_basins:
        if basin_id >= 0:
            count = sum(row.count(basin_id) for row in basin_mask)
            basin_name = (
                basin_names[basin_id] if basin_id < len(basin_names) else "Unknown"
            )
            logging.info(f"  Basin {basin_id} ({basin_name}): {count} points")
        else:
            count = sum(row.count(basin_id) for row in basin_mask)
            logging.info(f"  Unassigned points: {count}")

    # Broadcast basin mask to all years
    n_years = len(years)
    basins_all = []
    for _ in range(n_years):
        # Deep copy of basin_mask
        year_basins = [row[:] for row in basin_mask]
        basins_all.append(year_basins)

    # Create statistics list
    stats_list = []
    for i, year in enumerate(years):
        stats = {
            "year": year,
            "avg_abs_residual": float(stats_array[i][0])
            if not math.isnan(stats_array[i][0])
            else 0.0,
            "rms_residual": float(stats_array[i][1])
            if not math.isnan(stats_array[i][1])
            else 0.0,
            "sum_residual": float(stats_array[i][2])
            if not math.isnan(stats_array[i][2])
            else 0.0,
            "valid_points": int(stats_array[i][3])
            if not math.isnan(stats_array[i][3])
            else 0,
            "total_points": int(stats_array[i][4])
            if not math.isnan(stats_array[i][4])
            else 0,
        }
        stats_list.append(stats)

    logging.info("Creating xarray dataset...")

    # Convert lists to xarray-compatible format
    import numpy as np  # We need numpy just for xarray creation

    # Create dataset
    ds = xr.Dataset(
        {
            "residual": (["time", "y", "x"], np.array(residuals_all, dtype=np.float32)),
            "basin": (["time", "y", "x"], np.array(basins_all, dtype=np.int32)),
            "gsfc_ice_mask": (
                ["time", "y", "x"],
                np.array(gsfc_data_all, dtype=np.float32),
            ),
            "model_ice_mask": (
                ["time", "y", "x"],
                np.array(model_data_all, dtype=np.float32),
            ),
        },
        coords={
            "time": years,
            "x": x_coords,
            "y": y_coords,
            "basin_names": ("basin_id", basin_names),
        },
        attrs={
            "title": "Calving comparison analysis",
            "projection": "EPSG:3413",
            "units": "ice_mask units",
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    # Add statistics
    stats_df = pd.DataFrame(stats_list)
    for col in stats_df.columns:
        if col != "year":
            ds[f"stats_{col}"] = ("time", stats_df[col].values)

    end_time = time.time()
    logging.info(f"Dataset creation completed in {end_time - start_time:.2f} seconds")

    return ds
