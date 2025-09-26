import gc
import logging
import os
import time

import numpy as np
import pandas as pd
import xarray as xr
from numba import jit, prange

from ..ice_area_extent import configure_ice_area_extent_logging

# Create module-specific logger
logger = configure_ice_area_extent_logging()


# GPU Configuration Detection
def _detect_gpu_config():
    """Detect GPU configuration from environment or runtime."""
    gpu_config = {
        "platform": "cpu",
        "cuda_available": False,
        "metal_available": False,
        "use_gpu": False,
    }

    # Check environment variables (set by notebook)
    if os.environ.get("CMCT_GPU_PLATFORM"):
        gpu_config["platform"] = os.environ.get("CMCT_GPU_PLATFORM", "cpu")
        gpu_config["cuda_available"] = (
            os.environ.get("CMCT_CUDA_AVAILABLE", "False") == "True"
        )
        gpu_config["metal_available"] = (
            os.environ.get("CMCT_METAL_AVAILABLE", "False") == "True"
        )
        gpu_config["use_gpu"] = (
            gpu_config["cuda_available"] or gpu_config["metal_available"]
        )
        return gpu_config

    # Fallback detection
    try:
        from numba import cuda

        if cuda.is_available() and len(cuda.gpus) > 0:
            gpu_config["cuda_available"] = True
            gpu_config["platform"] = "cuda"
            gpu_config["use_gpu"] = True
    except ImportError:
        pass
    except Exception:
        pass

    return gpu_config


_GPU_CONFIG = _detect_gpu_config()

# Import CUDA if available
if _GPU_CONFIG["cuda_available"]:
    try:
        from numba import cuda
    except ImportError:
        _GPU_CONFIG["cuda_available"] = False
        _GPU_CONFIG["use_gpu"] = _GPU_CONFIG["metal_available"]


@jit(nopython=True, parallel=False, cache=True)
def compute_residuals_and_stats(
    observations_values, model_values, x_indices, y_indices
):
    n_points = len(x_indices)
    residuals = np.full(n_points, np.nan, dtype=np.float32)

    valid_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    residual_sum = 0.0

    for i in prange(n_points):
        observations_val = observations_values[y_indices[i], x_indices[i]]
        model_val = model_values[y_indices[i], x_indices[i]]

        if not (np.isnan(observations_val) or np.isnan(model_val)):
            residual = observations_val - model_val
            residuals[i] = residual
            valid_count += 1
            residual_sum += residual
            abs_sum += abs(residual)
            sq_sum += residual * residual

    return residuals, valid_count, abs_sum, sq_sum, residual_sum


# GPU-accelerated residual computation
if _GPU_CONFIG["cuda_available"]:

    @cuda.jit
    def _cuda_compute_residuals_kernel(
        observations_values, model_values, x_indices, y_indices, residuals, stats
    ):
        """CUDA kernel for residual computation."""
        idx = cuda.grid(1)
        if idx < x_indices.size:
            x_i = x_indices[idx]
            y_i = y_indices[idx]

            if (
                x_i < observations_values.shape[1]
                and y_i < observations_values.shape[0]
            ):
                observations_val = observations_values[y_i, x_i]
                model_val = model_values[y_i, x_i]

                if not (np.isnan(observations_val) or np.isnan(model_val)):
                    residual = observations_val - model_val
                    residuals[idx] = residual

                    # Atomic operations for statistics
                    cuda.atomic.add(stats, 0, 1)  # count
                    cuda.atomic.add(stats, 1, residual)  # sum
                    cuda.atomic.add(stats, 2, abs(residual))  # abs_sum
                    cuda.atomic.add(stats, 3, residual * residual)  # sq_sum


def gpu_compute_residuals_and_stats(
    observations_values, model_values, x_indices, y_indices
):
    """GPU-accelerated residual computation."""
    if not _GPU_CONFIG["use_gpu"] or len(x_indices) < 10000:
        # Use CPU for small datasets
        return compute_residuals_and_stats(
            observations_values, model_values, x_indices, y_indices
        )

    try:
        if _GPU_CONFIG["cuda_available"]:
            return _cuda_compute_residuals_and_stats(
                observations_values, model_values, x_indices, y_indices
            )
    except Exception:
        # Fallback to CPU
        pass

    # CPU fallback
    return compute_residuals_and_stats(
        observations_values, model_values, x_indices, y_indices
    )


def _cuda_compute_residuals_and_stats(
    observations_values, model_values, x_indices, y_indices
):
    """CUDA implementation of residual computation."""
    n_points = len(x_indices)

    # Prepare data for GPU
    observations_gpu = cuda.to_device(observations_values.astype(np.float32))
    model_gpu = cuda.to_device(model_values.astype(np.float32))
    x_indices_gpu = cuda.to_device(x_indices.astype(np.int32))
    y_indices_gpu = cuda.to_device(y_indices.astype(np.int32))

    # Output arrays
    residuals = cuda.device_array(n_points, dtype=np.float32)
    stats = cuda.zeros(4, dtype=np.float32)  # count, sum, abs_sum, sq_sum

    # Configure kernel
    threads_per_block = 256
    blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block

    # Launch kernel
    _cuda_compute_residuals_kernel[blocks_per_grid, threads_per_block](
        observations_gpu, model_gpu, x_indices_gpu, y_indices_gpu, residuals, stats
    )

    # Copy results back
    cuda.synchronize()
    residuals_host = residuals.copy_to_host()
    stats_host = stats.copy_to_host()

    # Fill NaN for invalid points
    for i in range(n_points):
        if np.isnan(residuals_host[i]):
            residuals_host[i] = np.nan

    return (
        residuals_host,
        int(stats_host[0]),  # valid_count
        float(stats_host[2]),  # abs_sum
        float(stats_host[3]),  # sq_sum
        float(stats_host[1]),  # residual_sum
    )


@jit(nopython=True, inline="always", fastmath=False)
def point_in_polygon(x, y, polygon_x, polygon_y):
    """Optimized point-in-polygon using winding number algorithm."""
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


def prepare_basin_polygons(basin_polygons_dict, target_crs="EPSG:3413"):
    """
    Convert basin polygons to numba-friendly arrays with proper coordinate transformation.

    Parameters:
    -----------
    basin_polygons_dict : dict
        Dictionary of basin polygons in geographic coordinates
    target_crs : str
        Target coordinate reference system (default: EPSG:3413 for polar stereographic)

    Returns:
    --------
    tuple
        (basin_names, basin_polygons_x, basin_polygons_y, basin_lengths)
    """
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
            coords = np.array(projected_polygon.exterior.coords)

            basin_names.append(name)
            all_x.extend(coords[:, 0])
            all_y.extend(coords[:, 1])
            basin_lengths.append(len(coords))

            logger.info(f"Transformed basin {name}: {len(coords)} points")

        except Exception as e:
            logger.error(f"Failed to transform basin {name}: {e}")
            continue

    return (
        basin_names,
        np.array(all_x, dtype=np.float32),
        np.array(all_y, dtype=np.float32),
        np.array(basin_lengths, dtype=np.int32),
    )


@jit(nopython=True, parallel=False, cache=True)
def create_basin_mask_optimized(
    x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
):
    """Create basin mask using vectorized operations with proper coordinate handling."""
    n_y, n_x = len(y_coords), len(x_coords)
    n_basins = len(basin_lengths)
    basin_mask = np.full((n_y, n_x), -1, dtype=np.int32)

    # Create coordinate meshgrid
    for i in prange(n_y):
        for j in prange(n_x):
            x = x_coords[j]
            y = y_coords[i]

            polygon_start = 0
            for basin_idx in range(n_basins):
                polygon_end = polygon_start + basin_lengths[basin_idx]
                poly_x = basin_polygons_x[polygon_start:polygon_end]
                poly_y = basin_polygons_y[polygon_start:polygon_end]

                if point_in_polygon(x, y, poly_x, poly_y):
                    basin_mask[i, j] = basin_idx
                    break

                polygon_start = polygon_end

    return basin_mask


# GPU-accelerated basin mask creation
if _GPU_CONFIG["cuda_available"]:

    @cuda.jit
    def _cuda_basin_mask_kernel(
        x_coords,
        y_coords,
        basin_polygons_x,
        basin_polygons_y,
        basin_starts,
        basin_lengths,
        basin_mask,
    ):
        """CUDA kernel for basin mask creation."""
        i, j = cuda.grid(2)

        if i < basin_mask.shape[0] and j < basin_mask.shape[1]:
            x = x_coords[j]
            y = y_coords[i]

            for basin_idx in range(basin_lengths.size):
                start_idx = basin_starts[basin_idx]
                length = basin_lengths[basin_idx]

                # Check if point is inside this basin polygon
                winding_number = 0
                for k in range(length):
                    k_next = (k + 1) % length
                    poly_x_k = basin_polygons_x[start_idx + k]
                    poly_y_k = basin_polygons_y[start_idx + k]
                    poly_x_next = basin_polygons_x[start_idx + k_next]
                    poly_y_next = basin_polygons_y[start_idx + k_next]

                    if poly_y_k <= y:
                        if poly_y_next > y:  # Upward crossing
                            cross_product = (poly_x_next - poly_x_k) * (
                                y - poly_y_k
                            ) - (x - poly_x_k) * (poly_y_next - poly_y_k)
                            if cross_product > 0:
                                winding_number += 1
                    else:
                        if poly_y_next <= y:  # Downward crossing
                            cross_product = (poly_x_next - poly_x_k) * (
                                y - poly_y_k
                            ) - (x - poly_x_k) * (poly_y_next - poly_y_k)
                            if cross_product < 0:
                                winding_number -= 1

                if winding_number != 0:
                    basin_mask[i, j] = basin_idx
                    break


def gpu_create_basin_mask_optimized(
    x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
):
    """GPU-accelerated basin mask creation with fallback."""
    if not _GPU_CONFIG["use_gpu"] or len(x_coords) * len(y_coords) < 50000:
        return create_basin_mask_optimized(
            x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
        )

    try:
        if _GPU_CONFIG["cuda_available"]:
            return _cuda_create_basin_mask_optimized(
                x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
            )
    except Exception:
        # Fallback to CPU
        pass

    return create_basin_mask_optimized(
        x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
    )


def _cuda_create_basin_mask_optimized(
    x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
):
    """CUDA implementation of basin mask creation."""
    n_y, n_x = len(y_coords), len(x_coords)

    # Calculate start indices for each basin polygon
    basin_starts = np.zeros(len(basin_lengths), dtype=np.int32)
    for i in range(1, len(basin_lengths)):
        basin_starts[i] = basin_starts[i - 1] + basin_lengths[i - 1]

    # Transfer data to GPU
    x_coords_gpu = cuda.to_device(x_coords.astype(np.float32))
    y_coords_gpu = cuda.to_device(y_coords.astype(np.float32))
    basin_polygons_x_gpu = cuda.to_device(basin_polygons_x.astype(np.float32))
    basin_polygons_y_gpu = cuda.to_device(basin_polygons_y.astype(np.float32))
    basin_starts_gpu = cuda.to_device(basin_starts)
    basin_lengths_gpu = cuda.to_device(basin_lengths)
    basin_mask_gpu = cuda.device_array((n_y, n_x), dtype=np.int32)

    # Initialize with -1
    basin_mask_gpu[:] = -1

    # Configure 2D grid
    threads_per_block = (16, 16)
    blocks_per_grid_x = (n_x + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_y = (n_y + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)

    # Launch kernel
    _cuda_basin_mask_kernel[blocks_per_grid, threads_per_block](
        x_coords_gpu,
        y_coords_gpu,
        basin_polygons_x_gpu,
        basin_polygons_y_gpu,
        basin_starts_gpu,
        basin_lengths_gpu,
        basin_mask_gpu,
    )

    cuda.synchronize()
    return basin_mask_gpu.copy_to_host()


def create_basin_mask_debug(
    x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
):
    """
    Create basin mask with debugging information.

    This is a non-numba version for debugging purposes that provides detailed
    information about the basin assignment process.
    """
    n_y, n_x = len(y_coords), len(x_coords)
    n_basins = len(basin_lengths)
    basin_mask = np.full((n_y, n_x), -1, dtype=np.int32)

    logger.info(f"Creating basin mask for {n_x} x {n_y} grid with {n_basins} basins")

    # Debug: Print coordinate ranges
    logger.info(
        f"Data coordinates: X=[{x_coords.min():.1f}, {x_coords.max():.1f}], Y=[{y_coords.min():.1f}, {y_coords.max():.1f}]"
    )
    logger.info(
        f"Basin polygon coordinates: X=[{basin_polygons_x.min():.1f}, {basin_polygons_x.max():.1f}], Y=[{basin_polygons_y.min():.1f}, {basin_polygons_y.max():.1f}]"
    )

    points_assigned = 0
    total_points = n_x * n_y

    for i in range(n_y):
        if i % 500 == 0:  # Progress indicator
            logger.info(f"Processing row {i}/{n_y}")

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

    logger.info(
        f"Basin assignment complete: {points_assigned}/{total_points} points assigned to basins ({100 * points_assigned / total_points:.1f}%)"
    )

    return basin_mask


@jit(nopython=True, parallel=False, cache=True, fastmath=False)
def compute_residuals_vectorized(observations_data, model_data):
    """Compute residuals in a vectorized manner."""
    n_time, n_y, n_x = observations_data.shape
    residuals = np.full((n_time, n_y, n_x), np.nan, dtype=np.float32)

    for t in prange(n_time):
        for i in prange(n_y):
            for j in prange(n_x):
                observations_val = observations_data[t, i, j]
                model_val = model_data[t, i, j]

                if not (np.isnan(observations_val) or np.isnan(model_val)):
                    residuals[t, i, j] = observations_val - model_val

    return residuals


# GPU-accelerated residuals computation
if _GPU_CONFIG["cuda_available"]:

    @cuda.jit
    def _cuda_residuals_kernel(observations_data, model_data, residuals):
        """CUDA kernel for residual computation."""
        t, i, j = cuda.grid(3)

        if (
            t < observations_data.shape[0]
            and i < observations_data.shape[1]
            and j < observations_data.shape[2]
        ):
            observations_val = observations_data[t, i, j]
            model_val = model_data[t, i, j]

            if not (np.isnan(observations_val) or np.isnan(model_val)):
                residuals[t, i, j] = observations_val - model_val


def gpu_compute_residuals_vectorized(observations_data, model_data):
    """GPU-accelerated residual computation with fallback."""
    if (
        not _GPU_CONFIG["use_gpu"] or observations_data.size < 100000
    ):  # Use CPU for small datasets
        return compute_residuals_vectorized(observations_data, model_data)

    try:
        if _GPU_CONFIG["cuda_available"]:
            return _cuda_compute_residuals_vectorized(observations_data, model_data)
    except Exception:
        # Fallback to CPU
        pass

    return compute_residuals_vectorized(observations_data, model_data)


def _cuda_compute_residuals_vectorized(observations_data, model_data):
    """CUDA implementation of vectorized residual computation."""
    n_time, n_y, n_x = observations_data.shape

    # Transfer data to GPU
    observations_gpu = cuda.to_device(observations_data.astype(np.float32))
    model_gpu = cuda.to_device(model_data.astype(np.float32))
    residuals_gpu = cuda.device_array((n_time, n_y, n_x), dtype=np.float32)

    # Initialize with NaN
    residuals_gpu[:] = np.nan

    # Configure 3D grid
    threads_per_block = (8, 8, 8)
    blocks_per_grid_x = (n_x + threads_per_block[2] - 1) // threads_per_block[2]
    blocks_per_grid_y = (n_y + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (n_time + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid = (blocks_per_grid_z, blocks_per_grid_y, blocks_per_grid_x)

    # Launch kernel
    _cuda_residuals_kernel[blocks_per_grid, threads_per_block](
        observations_gpu, model_gpu, residuals_gpu
    )

    cuda.synchronize()
    return residuals_gpu.copy_to_host()


@jit(nopython=True, cache=True, fastmath=False)
def compute_stats_vectorized(residuals):
    """Compute statistics for residuals in a vectorized manner."""
    n_time, n_y, n_x = residuals.shape
    stats = np.full(
        (n_time, 5), np.nan, dtype=np.float32
    )  # [avg_abs, rms, sum, valid_count, total_points]

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
            stats[t, 3] = 0  # valid_count
            stats[t, 4] = total_points  # total_points

    return stats


def compute_basin_mask_once(observations, model, basin_polygons_dict, year=None):
    """
    Compute basin mask once using the first available year of data.

    Parameters
    ----------
    observations : observationsice_area_extent
        observations ice_area_extent data object
    model : Modelice_area_extent
        Model ice_area_extent data object (first ensemble member)
    basin_polygons_dict : dict
        Dictionary of basin polygons
    year : int, optional
        Specific year to use for mask creation. If None, uses first available year.

    Returns
    -------
    tuple
        (basin_mask, basin_names, x_coords, y_coords)
    """
    logger.info("Computing basin mask once for all ensemble members...")

    # Prepare basin data
    basin_names, basin_polygons_x, basin_polygons_y, basin_lengths = (
        prepare_basin_polygons(basin_polygons_dict)
    )

    # Get coordinate arrays from model - use first available year if not specified
    if year is None:
        year = model.ds.time.values[0]

    model_sample = model.ds.sel(time=year)
    x_coords = model_sample.x.values.astype(np.float32)
    y_coords = model_sample.y.values.astype(np.float32)

    logger.info(f"Grid dimensions: {len(y_coords)} x {len(x_coords)}")
    logger.info(f"Using year {year} for basin mask creation")

    # Create basin mask
    logger.info("Creating basin mask...")
    try:
        basin_mask = gpu_create_basin_mask_optimized(
            x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
        )
    except Exception as e:
        logger.error(f"Error in GPU basin mask creation: {e}")
        logger.info("Falling back to CPU version...")
        try:
            basin_mask = create_basin_mask_optimized(
                x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
            )
        except Exception as e2:
            logger.error(f"Error in optimized basin mask creation: {e2}")
            logger.info("Falling back to debug version...")
            basin_mask = create_basin_mask_debug(
                x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
            )

    # Log basin assignment statistics
    unique_basins = np.unique(basin_mask)
    logger.info(f"Basin assignment complete. Unique basin IDs: {unique_basins}")

    for basin_id in unique_basins:
        if basin_id >= 0:
            count = np.sum(basin_mask == basin_id)
            basin_name = (
                basin_names[basin_id] if basin_id < len(basin_names) else "Unknown"
            )
            logger.info(f"  Basin {basin_id} ({basin_name}): {count} points")
        else:
            count = np.sum(basin_mask == basin_id)
            logger.info(f"  Unassigned points: {count}")

    assigned_points = np.sum(basin_mask >= 0)
    total_points = basin_mask.size
    assignment_rate = 100 * assigned_points / total_points
    logger.info(
        f"Basin assignment rate: {assigned_points}/{total_points} ({assignment_rate:.1f}%)"
    )

    return basin_mask, basin_names, x_coords, y_coords


def create_ice_area_extent_dataset_with_precomputed_mask(
    observations, model, years, basin_mask, basin_names, x_coords, y_coords
):
    """
    Create a comprehensive dataset with ice_area_extent data using a precomputed basin mask.

    Parameters
    ----------
    observations : observationsice_area_extent
        observations ice_area_extent data object
    model : Modelice_area_extent
        Model ice_area_extent data object
    years : list
        List of years to process
    basin_mask : np.ndarray
        Precomputed basin mask (y, x)
    basin_names : list
        List of basin names
    x_coords : np.ndarray
        X coordinates
    y_coords : np.ndarray
        Y coordinates

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions (time, y, x) and variables for residuals,
        basin assignments, and ice masks
    """
    logger.info("Creating ice_area_extent dataset with precomputed basin mask...")
    start_time = time.time()

    # Prepare observations data for all years
    observations_data_list = []
    for year in years:
        observations_year = observations.ds.sel(year=year)

        # Use xarray's interpolation to ensure exact coordinate matching
        # Interpolate observations data to the exact coordinates used for the basin mask
        observations_interp = observations_year.ice_mask.interp(
            x=x_coords, y=y_coords, method="linear"
        )
        observations_aligned = observations_interp.values.astype(np.float32)

        # Handle any NaN values that might result from interpolation
        observations_aligned = np.nan_to_num(observations_aligned, nan=0.0)

        observations_data_list.append(observations_aligned)

    observations_data_all = np.stack(observations_data_list, axis=0)
    logger.info(f"observations data shape: {observations_data_all.shape}")

    # Prepare model data for all years
    model_data_list = []
    for year in years:
        model_year = model.ds.sel(time=year)

        # Instead of using searchsorted, use xarray's interpolation to ensure exact coordinate matching
        # Interpolate model data to the exact coordinates used for the basin mask
        model_interp = model_year.ice_mask.interp(
            x=x_coords, y=y_coords, method="linear"
        )
        model_aligned = model_interp.values.astype(np.float32)

        # Handle any NaN values that might result from interpolation
        model_aligned = np.nan_to_num(model_aligned, nan=0.0)

        model_data_list.append(model_aligned)

    model_data_all = np.stack(model_data_list, axis=0)

    # Check if model data needs to be transposed to match observations data dimension order
    if model_data_all.shape != observations_data_all.shape:
        logger.info(
            f"Transposing model data from {model_data_all.shape} to match observations shape {observations_data_all.shape}"
        )
        # Transpose the spatial dimensions (keep time dimension as first)
        model_data_all = model_data_all.transpose(0, 2, 1)

    logger.info(f"Model data shape after alignment: {model_data_all.shape}")
    logger.info(f"Basin mask shape: {basin_mask.shape}")

    # Ensure all data arrays have consistent shapes
    if observations_data_all.shape != model_data_all.shape:
        logger.error(
            f"Shape mismatch after transpose: observations {observations_data_all.shape} vs Model {model_data_all.shape}"
        )
        raise ValueError(
            f"Data shape mismatch after transpose: observations {observations_data_all.shape} vs Model {model_data_all.shape}"
        )

    if observations_data_all.shape[1:] != basin_mask.shape:
        logger.error(
            f"Basin mask shape mismatch: Data {observations_data_all.shape[1:]} vs Mask {basin_mask.shape}"
        )
        raise ValueError(
            f"Basin mask shape mismatch: Data {observations_data_all.shape[1:]} vs Mask {basin_mask.shape}"
        )

    logger.info("Computing residuals...")
    residuals_all = gpu_compute_residuals_vectorized(
        observations_data_all, model_data_all
    )

    logger.info("Computing statistics...")
    stats_array = compute_stats_vectorized(residuals_all)

    # Broadcast basin mask to all years
    n_years = len(years)
    basins_all = np.broadcast_to(
        basin_mask[None, :, :], (n_years, basin_mask.shape[0], basin_mask.shape[1])
    ).copy()

    # Create statistics list
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

    logger.info("Creating xarray dataset...")

    ds = xr.Dataset(
        {
            "residual": (["time", "y", "x"], residuals_all),
            "basin": (["time", "y", "x"], basins_all),
            "observations_ice_mask": (["time", "y", "x"], observations_data_all),
            "model_ice_mask": (["time", "y", "x"], model_data_all),
        },
        coords={
            "time": years,
            "x": x_coords,
            "y": y_coords,
            "basin_names": ("basin_id", basin_names),
        },
        attrs={
            "title": "ice_area_extent comparison analysis",
            "projection": "EPSG:3413",
            "units": "ice_mask units",
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    # Add statistics as variables
    stats_df = pd.DataFrame(stats_list)
    for col in stats_df.columns:
        if col != "year":
            ds[f"stats_{col}"] = ("time", stats_df[col].values)

    end_time = time.time()
    logger.info(f"Dataset creation completed in {end_time - start_time:.2f} seconds")

    return ds


# Backward compatibility function - kept for legacy code
def create_ice_area_extent_dataset(observations, model, years, basin_polygons_dict):
    """
    Legacy function that computes basin mask and creates dataset.
    For new code, use compute_basin_mask_once() followed by
    create_ice_area_extent_dataset_with_precomputed_mask().
    """
    # Compute basin mask
    basin_mask, basin_names, x_coords, y_coords = compute_basin_mask_once(
        observations, model, basin_polygons_dict, years[0]
    )
    ()
    return create_ice_area_extent_dataset_with_precomputed_mask(
        observations, model, years, basin_mask, basin_names, x_coords, y_coords
    ), basin_mask
    
    

