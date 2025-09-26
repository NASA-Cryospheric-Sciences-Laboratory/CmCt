import logging
import time

import numpy as np
import pandas as pd
import xarray as xr
from numba import float64, int64, jit, njit, prange, types
from numba.typed import List as NbList
from pyproj import Transformer
from shapely.ops import transform


@njit(cache=True, fastmath=True)
def _is_nan(x):
    return x != x


@njit(cache=True, fastmath=True)
def _compute_residuals_and_stats_jitted(
    gsfc_values, model_values, x_indices, y_indices
):
    n_points = len(x_indices)
    residuals = NbList.empty_list(float64)
    residuals.extend([float("nan")] * n_points)

    valid_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    residual_sum = 0.0

    for i in range(n_points):
        gi = gsfc_values[y_indices[i]][x_indices[i]]
        mi = model_values[y_indices[i]][x_indices[i]]

        if not (_is_nan(gi) or _is_nan(mi)):
            r = gi - mi
            residuals[i] = r
            valid_count += 1
            residual_sum += r
            abs_sum += abs(r)
            sq_sum += r * r

    return residuals, valid_count, abs_sum, sq_sum, residual_sum


def compute_residuals_and_stats(gsfc_values, model_values, x_indices, y_indices):
    residuals_nb, valid_count, abs_sum, sq_sum, residual_sum = (
        _compute_residuals_and_stats_jitted(
            gsfc_values, model_values, x_indices, y_indices
        )
    )
    return (
        np.array(residuals_nb, dtype=np.float32),
        valid_count,
        abs_sum,
        sq_sum,
        residual_sum,
    )


@njit(cache=True, fastmath=True)
def point_in_polygon(x, y, poly_x, poly_y):
    n = len(poly_x)
    winding = 0
    for i in range(n):
        j = (i + 1) % n
        if poly_y[i] <= y < poly_y[j]:
            if (poly_x[j] - poly_x[i]) * (y - poly_y[i]) - (x - poly_x[i]) * (
                poly_y[j] - poly_y[i]
            ) > 0:
                winding += 1
        elif poly_y[j] <= y < poly_y[i]:
            if (poly_x[j] - poly_x[i]) * (y - poly_y[i]) - (x - poly_x[i]) * (
                poly_y[j] - poly_y[i]
            ) < 0:
                winding -= 1
    return winding != 0


@njit(parallel=False, cache=True, fastmath=True)
def assign_basins_batch(
    x_coords,
    y_coords,
    basin_polygons_x,
    basin_polygons_y,
    basin_lengths,
    batch_start,
    batch_end,
):
    batch_size = batch_end - batch_start
    basin_ids = NbList.empty_list(int64)
    basin_ids.extend([-1] * batch_size)

    for i in prange(batch_size):
        xi = x_coords[batch_start + i]
        yi = y_coords[batch_start + i]

        poly_start = 0
        for b, blen in enumerate(basin_lengths):
            poly_end = poly_start + blen
            if point_in_polygon(
                xi,
                yi,
                basin_polygons_x[poly_start:poly_end],
                basin_polygons_y[poly_start:poly_end],
            ):
                basin_ids[i] = b
                break
            poly_start = poly_end
        if i % 1000 == 0:
            time.sleep(0.001)
    return basin_ids


def prepare_basin_polygons(basin_polygons_dict, target_crs="EPSG:3413"):
    basin_names = []
    all_x = []
    all_y = []
    basin_lengths = []

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    for name, polygon in basin_polygons_dict.items():
        if name == "unassigned":
            continue

        try:
            projected_polygon = transform(transformer.transform, polygon)
            coords = np.array(projected_polygon.exterior.coords)

            basin_names.append(name)
            all_x.extend(coords[:, 0])
            all_y.extend(coords[:, 1])
            basin_lengths.append(len(coords))
            time.sleep(0.001)
            logging.info(f"Transformed basin {name}: {len(coords)} points")

        except Exception as e:
            logging.error(f"Failed to transform basin {name}: {e}")
            continue

    return (
        basin_names,
        np.array(all_x, dtype=np.float32),
        np.array(all_y, dtype=np.float32),
        np.array(basin_lengths, dtype=np.int32),
    )


def process_year_data(
    gsfc_data,
    model_data,
    x_coords,
    y_coords,
    basin_polygons_x,
    basin_polygons_y,
    basin_lengths,
    basin_names,
    year,
):
    gsfc_x = gsfc_data.x.values.astype(np.float32)
    gsfc_y = gsfc_data.y.values.astype(np.float32)

    x_indices = np.searchsorted(gsfc_x, x_coords)
    x_indices = np.clip(x_indices, 0, len(gsfc_x) - 1)

    y_indices = np.searchsorted(gsfc_y, y_coords)
    y_indices = np.clip(y_indices, 0, len(gsfc_y) - 1)

    gsfc_values = gsfc_data.ice_mask.values.astype(np.float32)
    model_values = model_data.ice_mask.values.astype(np.float32)

    x_flat = np.repeat(x_coords, len(y_coords))
    y_flat = np.tile(y_coords, len(x_coords))
    x_idx_flat = np.repeat(x_indices, len(y_indices))
    y_idx_flat = np.tile(y_indices, len(x_indices))

    residuals, valid_count, abs_sum, sq_sum, residual_sum = compute_residuals_and_stats(
        gsfc_values, model_values, x_idx_flat, y_idx_flat
    )

    n_points = len(x_flat)
    batch_size = 50000
    basin_assignments = np.full(n_points, -1, dtype=np.int32)

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

    n_x, n_y = len(x_coords), len(y_coords)
    residual_grid = residuals.reshape(n_x, n_y).T
    basin_grid = basin_assignments.reshape(n_x, n_y).T

    gsfc_aligned = gsfc_values[np.ix_(y_indices, x_indices)]
    model_aligned = model_values[np.ix_(y_indices, x_indices)]

    avg_residual = abs_sum / valid_count if valid_count > 0 else 0.0
    rms_residual = np.sqrt(sq_sum / valid_count) if valid_count > 0 else 0.0

    stats = {
        "avg_abs_residual": round(avg_residual, 3),
        "rms_residual": round(rms_residual, 3),
        "sum_residual": round(residual_sum, 3),
        "valid_points": int(valid_count),
        "total_points": n_points,
    }

    return residual_grid, basin_grid, gsfc_aligned, model_aligned, stats


@njit(parallel=False, cache=True)
def _create_basin_mask_optimized_jitted(
    x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
):
    ny = len(y_coords)
    nx = len(x_coords)
    n_basins = len(basin_lengths)
    mask = [[-1] * nx for _ in range(ny)]

    for i in prange(ny):
        for j in range(nx):
            x = x_coords[j]
            y = y_coords[i]

            pstart = 0
            for b in range(n_basins):
                pend = pstart + basin_lengths[b]
                if point_in_polygon(
                    x, y, basin_polygons_x[pstart:pend], basin_polygons_y[pstart:pend]
                ):
                    mask[i][j] = b
                    break
                pstart = pend
        # import time
        # if i % 1000 == 0:
        #     time.sleep(0.001)
    return mask


def create_basin_mask_optimized(
    x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
):
    mask_lol = _create_basin_mask_optimized_jitted(
        x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
    )
    return np.array(mask_lol, dtype=np.int32)


def create_basin_mask_debug(
    x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
):
    n_y, n_x = len(y_coords), len(x_coords)
    n_basins = len(basin_lengths)
    basin_mask = np.full((n_y, n_x), -1, dtype=np.int32)

    logging.info(f"Creating basin mask for {n_x} x {n_y} grid with {n_basins} basins")

    logging.info(
        f"Data coordinates: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]"
    )
    logging.info(
        f"Basin polygon coordinates: X=[{basin_polygons_x.min():.1f}, {basin_polygons_x.max():.1f}], Y=[{basin_polygons_y.min():.1f}, {basin_polygons_y.max():.1f}]"
    )

    points_assigned = 0
    total_points = n_x * n_y

    for i in range(n_y):
        if i % 500 == 0:
            logging.info(f"Processing row {i}/{n_y}")
            time.sleep(0.001)

        for j in range(n_x):
            x = x_coords[j]
            y = y_coords[i]

            polygon_start = 0
            for basin_idx in range(n_basins):
                polygon_end = polygon_start + basin_lengths[basin_idx]
                poly_x = basin_polygons_x[polygon_start:polygon_end]
                poly_y = basin_polygons_y[polygon_start:polygon_end]

                if point_in_polygon(x, y, poly_x, poly_y):
                    basin_mask[i, j] = basin_idx
                    points_assigned += 1
                    break

                polygon_start = polygon_end

    logging.info(
        f"Basin assignment complete: {points_assigned}/{total_points} points assigned to basins ({100 * points_assigned / total_points:.1f}%)"
    )

    return basin_mask


@jit(nopython=True, parallel=False, cache=True, fastmath=True)
def compute_residuals_vectorized(gsfc_data, model_data):
    n_time, n_y, n_x = gsfc_data.shape
    residuals = np.full((n_time, n_y, n_x), np.nan, dtype=np.float32)

    for t in prange(n_time):
        for i in prange(n_y):
            for j in prange(n_x):
                gsfc_val = gsfc_data[t, i, j]
                model_val = model_data[t, i, j]

                if not (np.isnan(gsfc_val) or np.isnan(model_val)):
                    residuals[t, i, j] = gsfc_val - model_val

    return residuals


@jit(nopython=True, cache=True, fastmath=True)
def compute_stats_vectorized(residuals):
    n_time, n_y, n_x = residuals.shape
    stats = np.full((n_time, 5), np.nan, dtype=np.float32)

    for t in range(n_time):
        abs_sum = 0.0
        sq_sum = 0.0
        residual_sum = 0.0
        valid_count = 0
        total_points = n_y * n_x

        for i in range(n_y):
            for j in range(n_x):
                val = residuals[t, i, j]
                if not np.isnan(val):
                    valid_count += 1
                    residual_sum += val
                    abs_sum += abs(val)
                    sq_sum += val * val

        if valid_count > 0:
            avg_abs = abs_sum / valid_count
            rms = np.sqrt(sq_sum / valid_count)

            stats[t, 0] = avg_abs
            stats[t, 1] = rms
            stats[t, 2] = residual_sum
            stats[t, 3] = valid_count
            stats[t, 4] = total_points
        else:
            stats[t, 3] = 0
            stats[t, 4] = total_points

    return stats


def create_calving_dataset(gsfc, model, years, basin_polygons_dict):
    logging.info("Starting optimized calving dataset creation...")
    start_time = time.time()

    basin_names, basin_polygons_x, basin_polygons_y, basin_lengths = (
        prepare_basin_polygons(basin_polygons_dict)
    )

    model_sample = model.ds.sel(time=years[0])
    x_coords = model_sample.x.values.astype(np.float32)
    y_coords = model_sample.y.values.astype(np.float32)

    logging.info(f"Grid dimensions: {len(y_coords)} x {len(x_coords)}")

    gsfc_data_list = []
    for year in years:
        gsfc_year = gsfc.ds.sel(year=year)

        gsfc_x = gsfc_year.x.values
        gsfc_y = gsfc_year.y.values

        x_indices = np.searchsorted(gsfc_x, x_coords)
        x_indices = np.clip(x_indices, 0, len(gsfc_x) - 1)
        y_indices = np.searchsorted(gsfc_y, y_coords)
        y_indices = np.clip(y_indices, 0, len(gsfc_y) - 1)

        gsfc_aligned = gsfc_year.ice_mask.values[np.ix_(y_indices, x_indices)].astype(
            np.float32
        )
        gsfc_data_list.append(gsfc_aligned)

    gsfc_data_all = np.stack(gsfc_data_list, axis=0)
    model_data_list = []
    for year in years:
        model_year = model.ds.sel(time=year)
        model_data_list.append(model_year.ice_mask.values.astype(np.float32))

    model_data_all = np.stack(model_data_list, axis=0)

    logging.info("Computing residuals...")
    residuals_all = compute_residuals_vectorized(gsfc_data_all, model_data_all)

    logging.info("Computing statistics...")
    stats_array = compute_stats_vectorized(residuals_all)

    logging.info("Creating basin assignments...")

    logging.info(
        f"Data coordinates: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]"
    )
    logging.info(
        f"Basin polygon coordinates: X=[{basin_polygons_x.min():.1f}, {basin_polygons_x.max():.1f}], Y=[{basin_polygons_y.min():.1f}, {basin_polygons_y.max():.1f}]"
    )

    try:
        basin_mask = create_basin_mask_optimized(
            x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
        )
    except Exception as e:
        logging.error(f"Error in optimized basin mask creation: {e}")
        logging.info("Falling back to debug version...")
        basin_mask = create_basin_mask_debug(
            x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
        )
    unique_basins = np.unique(basin_mask)
    logging.info(f"Basin assignment complete. Unique basin IDs: {unique_basins}")

    for basin_id in unique_basins:
        if basin_id >= 0:
            count = np.sum(basin_mask == basin_id)
            basin_name = (
                basin_names[basin_id] if basin_id < len(basin_names) else "Unknown"
            )
            logging.info(f"  Basin {basin_id} ({basin_name}): {count} points")
        else:
            count = np.sum(basin_mask == basin_id)
            logging.info(f"  Unassigned points: {count}")

    assigned_points = np.sum(basin_mask >= 0)
    total_points = basin_mask.size
    assignment_rate = 100 * assigned_points / total_points
    logging.info(
        f"Basin assignment rate: {assigned_points}/{total_points} ({assignment_rate:.1f}%)"
    )

    n_years = len(years)
    basins_all = np.broadcast_to(
        basin_mask[None, :, :], (n_years, basin_mask.shape[0], basin_mask.shape[1])
    ).copy()

    stats_list = []
    for i, year in enumerate(years):
        stats = {
            "year": year,
            "avg_abs_residual": float(stats_array[i, 0])
            if not np.isnan(stats_array[i, 0])
            else 0.0,
            "rms_residual": float(stats_array[i, 1])
            if not np.isnan(stats_array[i, 1])
            else 0.0,
            "sum_residual": float(stats_array[i, 2])
            if not np.isnan(stats_array[i, 2])
            else 0.0,
            "valid_points": int(stats_array[i, 3])
            if not np.isnan(stats_array[i, 3])
            else 0,
            "total_points": int(stats_array[i, 4])
            if not np.isnan(stats_array[i, 4])
            else 0,
        }
        stats_list.append(stats)

    logging.info("Creating xarray dataset...")
    logging.info(f"model_data_all.shape: {model_data_all.shape}")
    logging.info(f"gsfc_data_all.shape: {gsfc_data_all.shape}")
    logging.info(f"residuals_all.shape: {residuals_all.shape}")
    logging.info(f"basin_mask.shape: {basin_mask.shape}")
    logging.info(f"Years: {years}")

    if model_data_all.shape != gsfc_data_all.shape:
        logging.warning(
            f"Shape mismatch: model_data_all.shape={model_data_all.shape}, gsfc_data_all.shape={gsfc_data_all.shape}"
        )
        if (
            model_data_all.shape[1] == gsfc_data_all.shape[2]
            and model_data_all.shape[2] == gsfc_data_all.shape[1]
        ):
            logging.info("Transposing arrays to match gsfc_data_all dimensions...")
            model_data_all = np.transpose(model_data_all, (0, 2, 1))
            residuals_all = compute_residuals_vectorized(gsfc_data_all, model_data_all)

            x_coords, y_coords = y_coords, x_coords

            logging.info(f"Original basin_mask.shape: {basin_mask.shape}")
            basin_mask = np.transpose(basin_mask, (1, 0))
            logging.info(f"Transposed basin_mask.shape: {basin_mask.shape}")

            logging.info(
                f"After transpose - model_data_all.shape: {model_data_all.shape}"
            )
            logging.info(
                f"After transpose - gsfc_data_all.shape: {gsfc_data_all.shape}"
            )
            logging.info(
                f"After transpose - residuals_all.shape: {residuals_all.shape}"
            )

    n_years = len(years)
    expected_spatial_shape = gsfc_data_all.shape[1:]
    logging.info(f"Expected spatial shape for basin_mask: {expected_spatial_shape}")
    logging.info(f"Current basin_mask.shape: {basin_mask.shape}")

    if basin_mask.shape != expected_spatial_shape:
        logging.error(
            f"Basin mask shape {basin_mask.shape} doesn't match expected {expected_spatial_shape}"
        )
        logging.error(
            "This indicates the coordinate transformation didn't work as expected"
        )

        if basin_mask.shape == expected_spatial_shape[::-1]:
            logging.info("Forcing basin_mask transpose to match expected shape")
            basin_mask = np.transpose(basin_mask, (1, 0))
            logging.info(
                f"After forced transpose - basin_mask.shape: {basin_mask.shape}"
            )

    basins_all = np.broadcast_to(
        basin_mask[None, :, :], (n_years, basin_mask.shape[0], basin_mask.shape[1])
    ).copy()

    logging.info(f"basins_all.shape: {basins_all.shape}")

    arrays_to_check = {
        "model_data_all": model_data_all,
        "gsfc_data_all": gsfc_data_all,
        "residuals_all": residuals_all,
        "basins_all": basins_all,
    }

    expected_shape = gsfc_data_all.shape
    for name, array in arrays_to_check.items():
        logging.info(
            f"Checking {name}: shape {array.shape} vs expected {expected_shape}"
        )
        if array.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: {name} has shape {array.shape}, "
                f"expected {expected_shape}"
            )

    logging.info(f"All arrays have consistent shape: {expected_shape}")
    logging.info(f"x_coords.shape: {x_coords.shape}")
    logging.info(f"y_coords.shape: {y_coords.shape}")
    logging.info(f"Data array spatial shape: {gsfc_data_all.shape[1:]}")

    if (
        len(y_coords) != gsfc_data_all.shape[1]
        or len(x_coords) != gsfc_data_all.shape[2]
    ):
        logging.warning(
            f"Coordinate dimensions don't match data: y_coords={len(y_coords)}, x_coords={len(x_coords)}"
        )
        logging.warning(
            f"Expected: y_coords={gsfc_data_all.shape[1]}, x_coords={gsfc_data_all.shape[2]}"
        )

        if (
            len(y_coords) == gsfc_data_all.shape[2]
            and len(x_coords) == gsfc_data_all.shape[1]
        ):
            logging.info("Swapping coordinate arrays to match data dimensions")
            x_coords, y_coords = y_coords, x_coords
            logging.info(
                f"After swap - x_coords.shape: {x_coords.shape}, y_coords.shape: {y_coords.shape}"
            )

    ds = xr.Dataset(
        {
            "residual": (["time", "y", "x"], residuals_all),
            "basin": (["time", "y", "x"], basins_all),
            "gsfc_ice_mask": (["time", "y", "x"], gsfc_data_all),
            "model_ice_mask": (["time", "y", "x"], model_data_all),
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

    stats_df = pd.DataFrame(stats_list)
    for col in stats_df.columns:
        if col != "year":
            ds[f"stats_{col}"] = ("time", stats_df[col].values)

    end_time = time.time()
    logging.info(
        f"Optimized dataset creation completed in {end_time - start_time:.2f} seconds"
    )

    return ds
