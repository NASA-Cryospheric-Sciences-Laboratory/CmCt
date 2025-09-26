import logging
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from multiprocessing import Pool, cpu_count

import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4 as nc
import numba
import numpy as np
import xarray as xr
from matplotlib import rc
from numba import jit, prange
from scipy import stats

from cmct.ice_area_extent_modules import shapefile_utils


def configure_ice_area_extent_logging(log_level="ALL"):
    """
    Configure logging for ice area extent modules.

    Parameters
    ----------
    log_level : str
        Logging level. Options:
        - "ALL": Show all logs (DEBUG and above)
        - "ERROR": Show only error messages
        - "WARNING": Show warnings and errors
        - "INFO": Show info, warnings, and errors (default behavior)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Map string levels to logging constants
    level_mapping = {
        "ALL": logging.DEBUG,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
    }

    # Validate input
    if log_level not in level_mapping:
        raise ValueError(
            f"Invalid log_level '{log_level}'. Must be one of: {list(level_mapping.keys())}"
        )

    # Get the numeric level
    numeric_level = level_mapping[log_level]

    # Configure the root logger for ice area extent modules
    logger = logging.getLogger("cmct.ice_area_extent")
    logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    # Also configure loggers for the ice_area_extent_modules
    module_names = [
        "cmct.ice_area_extent_modules.residual_calculation",
        "cmct.ice_area_extent_modules.plotting_utils",
        "cmct.ice_area_extent_modules.interpolation",
        "cmct.ice_area_extent_modules.shapefile_utils",
        "cmct.ice_area_extent_modules.statistical_utils",
    ]

    for module_name in module_names:
        module_logger = logging.getLogger(module_name)
        module_logger.setLevel(numeric_level)
        module_logger.handlers.clear()
        module_logger.addHandler(console_handler)
        module_logger.propagate = False  # Prevent propagation to root logger

    # Set propagation for main logger
    logger.propagate = False

    return logger


def detect_and_configure_gpu():
    """
    Detect available GPU compute platforms and configure accordingly.
    Supports CUDA, Metal (Mac), and CPU fallback.
    """
    gpu_info = {
        "platform": "cpu",
        "device_name": "CPU",
        "cuda_available": False,
        "metal_available": False,
        "device_count": 0,
    }

    # Check for CUDA
    try:
        from numba import cuda

        if cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_info["platform"] = "cuda"
            gpu_info["device_count"] = len(cuda.gpus)
            if gpu_info["device_count"] > 0:
                gpu_info["device_name"] = (
                    f"CUDA GPU (Count: {gpu_info['device_count']})"
                )
                # Test CUDA functionality
                try:

                    @cuda.jit
                    def test_cuda_kernel(arr):
                        idx = cuda.grid(1)
                        if idx < arr.size:
                            arr[idx] = idx

                    test_arr = cuda.device_array(10, dtype=np.float32)
                    test_cuda_kernel[1, 10](test_arr)
                    cuda.synchronize()
                except Exception:
                    gpu_info["cuda_available"] = False
                    gpu_info["platform"] = "cpu"
    except ImportError:
        pass
    except Exception:
        pass

    # Check for Metal (Mac GPU)
    if platform.system() == "Darwin":  # macOS
        try:
            # Try to import Metal Performance Shaders
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "Metal" in result.stdout or "GPU" in result.stdout:
                gpu_info["metal_available"] = True
                if not gpu_info[
                    "cuda_available"
                ]:  # Only use Metal if CUDA not available
                    gpu_info["platform"] = "metal"
                    gpu_info["device_name"] = "Metal GPU (Apple Silicon)"
        except Exception:
            pass

        # Alternative Metal detection using PyTorch if available
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_info["metal_available"] = True
                if not gpu_info["cuda_available"]:
                    gpu_info["platform"] = "metal"
                    gpu_info["device_name"] = "Metal GPU (MPS)"
        except ImportError:
            pass
        except Exception:
            pass

    # Configure Numba based on available hardware
    if gpu_info["cuda_available"]:
        numba.set_num_threads(2)  # Conservative thread count for CUDA
        os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    elif gpu_info["metal_available"]:
        # Set environment variables for Metal optimization
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        numba.set_num_threads(4)  # More threads for Metal
    else:
        # Optimize for CPU
        cpu_count = os.cpu_count()
        optimal_threads = min(cpu_count, 8)  # Cap at 8 threads
        numba.set_num_threads(optimal_threads)

    return gpu_info


# GPU Configuration Detection
def _detect_gpu_capabilities():
    """
    Detect GPU capabilities for accelerated computation.
    Returns configuration dictionary.
    """
    gpu_config = {
        "platform": "cpu",
        "cuda_available": False,
        "metal_available": False,
        "use_gpu": False,
    }

    # Check environment variables first (set by notebook)
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

    # Fallback to runtime detection
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

    if platform.system() == "Darwin" and not gpu_config["cuda_available"]:
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_config["metal_available"] = True
                gpu_config["platform"] = "metal"
                gpu_config["use_gpu"] = True
        except ImportError:
            pass
        except Exception:
            pass

    return gpu_config


# Global GPU configuration
_GPU_CONFIG = _detect_gpu_capabilities()

# Import CUDA if available
if _GPU_CONFIG["cuda_available"]:
    try:
        from numba import cuda
    except ImportError:
        _GPU_CONFIG["cuda_available"] = False
        _GPU_CONFIG["use_gpu"] = _GPU_CONFIG["metal_available"]

# from cmct.shapefile_utils import *

# from .shapefile_utils import (
#     get_nonzero_indices,
#     scaling_shape_to_target,
#     shapefile_to_xy,
# )


def safe_float_conversion(data, default_value=np.nan):
    """
    Safely convert data to float64, handling non-numeric values.

    Parameters
    ----------
    data : array-like
        Data to convert
    default_value : float, optional
        Value to use for non-numeric entries (default: np.nan)

    Returns
    -------
    numpy.ndarray
        Array converted to float64 with non-numeric values replaced
    """
    try:
        return data.astype(np.float64)
    except (ValueError, TypeError):
        result = np.full(data.shape, default_value, dtype=np.float64)
        # Try to convert element by element for mixed types
        flat_data = data.flatten()
        flat_result = result.flatten()

        for i in range(len(flat_data)):
            try:
                val = float(flat_data[i])
                if i % 1000 == 0:
                    time.sleep(0.001)
                if np.isfinite(val):
                    flat_result[i] = val
                else:
                    flat_result[i] = default_value
            except (ValueError, TypeError, OverflowError):
                flat_result[i] = default_value

        return flat_result.reshape(data.shape)


logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_basins(basin_filename, basins):
    basin_data = shapefile_utils.load_basin_polygons(basin_filename)
    if basins == "all":
        selected_basins = ["CW", "NE", "SE", "SW", "NO", "NW", "unassigned"]
    else:
        selected_basins = basins

    # Filter basin_data to only include selected basins
    basin_polygons = {}

    for basin_name in selected_basins:
        if basin_name in basin_data:
            basin_polygons[basin_name] = basin_data[basin_name]

    return basin_polygons, selected_basins


def load_observations_ice_area_extent(filepath, basins=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        observations = observationsice_area_extent(filepath, basins)

    except Exception as error:
        print("Error: Failed to load observations dataset.")
        print(error)
        observations = None
        raise ValueError("Failed to load observations ice_area_extent data.")

    return observations


def load_model_ice_area_extent(filepath):
    try:
        model_res = Modelice_area_extent(filepath)
    except Exception as error:
        logging.error("Error: Failed to load Model dataset.")
        logging.error(error)
        model_res = None
    return model_res


def load_residuals(residuals):
    if isinstance(residuals, str):
        if not os.path.exists(residuals):
            raise FileNotFoundError(f"File not found: {residuals}")
        try:
            residuals = Residual(
                xr.open_dataset(residuals, autoclose=True, engine="netcdf4")
            )
        except Exception as error:
            logging.error("Error: Failed to load residuals dataset.")
            logging.error(error)
            residuals = None

    elif isinstance(residuals, xr.Dataset):
        return Residual(residuals)


class observationsice_area_extent:
    def __init__(self, nc_path, basins=None):
        # Open as xarray Dataset

        self.ds = xr.open_dataset(
            nc_path, autoclose=True, engine="netcdf4", use_cftime=True
        )
        # self.ds =
        self.ds["ice_mask"] = self.ds["ice_mask"] / 100
        # Safely convert ice_mask to float64 for np.isnan compatibility
        ice_mask_data = safe_float_conversion(self.ds["ice_mask"].values)
        self.ds["ice_mask"] = (self.ds["ice_mask"].dims, ice_mask_data)
        # self.basins = basins

    # Direct access to variables as attributes
    @property
    def time(self):
        return self.ds["year"]

    @property
    def ice_mask(self):
        return self.ds["ice_mask"]

    @property
    def x(self):
        return self.ds["x"]

    @property
    def y(self):
        return self.ds["y"]

    @property
    def basins(self):
        return self.basins

    # def _set_times_as_datetimes(self, days):
    #     return np.datetime64('2002-01-01T00:00:00') + np.array([int(d*24) for d in days], dtype='timedelta64[h]')


class Modelice_area_extent:
    def __init__(self, nc_path):
        # Open as xarray Dataset
        self.ds = xr.open_dataset(
            nc_path, autoclose=True, engine="netcdf4", use_cftime=True
        )
        self.ds["ice_mask"] = self.ds["sftgif"]
        # Safely convert ice_mask to float64 for np.isnan compatibility
        ice_mask_data = safe_float_conversion(self.ds["ice_mask"].values)
        self.ds["ice_mask"] = (self.ds["ice_mask"].dims, ice_mask_data)

        # Making the variables consistent REMOVE IF NECESSARY
        self.ds = self.ds.drop("sftgif")
        # Direct access to variables as attributes

    @property
    def x(self):
        return self.ds["x"]

    @property
    def y(self):
        return self.ds["y"]

    @property
    def lat(self):
        return self.ds["lat"]

    @property
    def lon(self):
        return self.ds["lon"]

    @property
    def ice_mask(self):
        return self.ds["ice_mask"]

    @property
    def time(self):
        return self.ds["time"]

    # def _set_times_as_datetimes(self, days):
    #     return np.datetime64('2002-01-01T00:00:00') + np.array([int(d*24) for d in days], dtype='timedelta64[h]')

    def close(self):
        self.ds.close()

    def print_info(self):
        print("observations ice_area_extent Data:")
        print(f"Latitude: {self.lat.values}")
        print(f"Longitude: {self.lon.values}")
        print(f"Time: {self.time.values}")
        print(f"ice_mask: {self.ice_mask.values}")
        print(f"X coordinates: {self.x.values}")
        print(f"Y coordinates: {self.y.values}")


class Residual:
    def __init__(self, residuals):
        self.ds = residuals

    @property
    def x(self):
        return self.ds["x"]

    @property
    def y(self):
        return self.ds["y"]

    @property
    def lat(self):
        return self.ds["lat"]

    @property
    def lon(self):
        return self.ds["lon"]

    @property
    def ice_mask(self):
        return self.ds["ice_mask"]

    @property
    def time(self):
        return self.ds["time"]

    def get_basin_data(self, year, basin_id=None):
        if basin_id is None:
            # Return all data for the year
            data = self.ds.sel(time=year).residual.values
        else:
            # Return data only for the specified basin
            basin_mask = self.ds.basin.sel(time=year) == basin_id
            data = self.ds.residual.sel(time=year).where(basin_mask).values

        return data

    def to_netCDF(self, output_path):
        """
        Save the residuals dataset to a NetCDF file.

        Parameters
        ----------
        output_path : str
            Path to save the NetCDF file.
        """
        self.ds.to_netcdf(output_path, mode="w", format="NETCDF4")
        logging.info(f"Residuals saved to {output_path}")

    def to_json(self, output_path):
        """
        Save the residuals dataset to a JSON file.

        Parameters
        ----------
        output_path : str
            Path to save the JSON file.
        """
        self.ds.to_dataframe().to_json(output_path)
        logging.info(f"Residuals saved to {output_path}")


# GPU-Accelerated Functions
if _GPU_CONFIG["cuda_available"]:

    @cuda.jit
    def _cuda_calculate_basin_stats_kernel(valid_data, stats_output):
        """CUDA kernel for basin statistics calculation."""
        idx = cuda.grid(1)
        if idx < valid_data.size:
            val = valid_data[idx]
            if not np.isnan(val):
                cuda.atomic.add(stats_output, 0, 1)  # count
                cuda.atomic.add(stats_output, 1, val)  # sum
                cuda.atomic.add(stats_output, 2, val * val)  # sum_sq


def _gpu_accelerated_basin_stats(residual_data, basin_data, basin_idx):
    """
    GPU-accelerated basin statistics calculation.

    Parameters
    ----------
    residual_data : numpy.ndarray
        Residual data array
    basin_data : numpy.ndarray
        Basin assignment data array
    basin_idx : int
        Basin index to process

    Returns
    -------
    dict
        Statistics dictionary
    """
    if not _GPU_CONFIG["use_gpu"]:
        return None

    try:
        if _GPU_CONFIG["cuda_available"]:
            return _cuda_basin_stats(residual_data, basin_data, basin_idx)
        elif _GPU_CONFIG["metal_available"]:
            return _metal_basin_stats(residual_data, basin_data, basin_idx)
    except Exception:
        # Fallback to CPU if GPU fails
        pass

    return None


def _cuda_basin_stats(residual_data, basin_data, basin_idx):
    """CUDA implementation of basin statistics."""
    try:
        # Create mask for current basin
        basin_mask = basin_data == basin_idx

        # Extract valid data
        basin_residuals = residual_data[basin_mask]
        valid_data = basin_residuals[~np.isnan(basin_residuals)]

        if len(valid_data) == 0:
            return None

        # Transfer to GPU
        d_valid_data = cuda.to_device(valid_data.astype(np.float32))

        # Allocate output array for statistics
        h_stats = np.zeros(3, dtype=np.float32)  # count, sum, sum_sq
        d_stats = cuda.to_device(h_stats)

        # Configure CUDA kernel
        threads_per_block = 256
        blocks_per_grid = (len(valid_data) + threads_per_block - 1) // threads_per_block

        # Launch kernel
        _cuda_calculate_basin_stats_kernel[blocks_per_grid, threads_per_block](
            d_valid_data, d_stats
        )

        # Copy results back
        cuda.synchronize()
        stats_result = d_stats.copy_to_host()

        count = int(stats_result[0])
        sum_val = float(stats_result[1])
        sum_sq = float(stats_result[2])

        if count > 0:
            mean_val = sum_val / count
            variance = (sum_sq / count) - (mean_val * mean_val)
            std_val = np.sqrt(max(0, variance))

            return {
                "count": count,
                "mean": mean_val,
                "std": std_val,
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "rms": np.sqrt(sum_sq / count),
                "rss": sum_sq,
                "sum": sum_val,
            }

    except Exception:
        pass

    return None


def _metal_basin_stats(residual_data, basin_data, basin_idx):
    """Metal Performance Shaders implementation of basin statistics."""
    try:
        # For Metal, we use NumPy with optimized threading
        # Metal acceleration would require PyTorch MPS or similar
        basin_mask = basin_data == basin_idx
        basin_residuals = residual_data[basin_mask]
        valid_data = basin_residuals[~np.isnan(basin_residuals)]

        if len(valid_data) == 0:
            return None

        # Use NumPy's optimized operations which can leverage Metal via BLAS
        count = len(valid_data)
        sum_val = float(np.sum(valid_data))
        sum_sq = float(np.sum(valid_data * valid_data))
        mean_val = sum_val / count
        variance = (sum_sq / count) - (mean_val * mean_val)
        std_val = float(np.sqrt(max(0, variance)))

        return {
            "count": count,
            "mean": mean_val,
            "std": std_val,
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "rms": float(np.sqrt(sum_sq / count)),
            "rss": sum_sq,
            "sum": sum_val,
        }

    except Exception:
        pass

    return None


if _GPU_CONFIG["cuda_available"]:

    @cuda.jit
    def _cuda_calculate_basin_stats_kernel(valid_data, stats_output):
        """Optimized CUDA kernel for basin statistics."""
        idx = cuda.grid(1)
        if idx < valid_data.size:
            val = valid_data[idx]
            if not np.isnan(val):
                cuda.atomic.add(stats_output, 0, 1)  # count
                cuda.atomic.add(stats_output, 1, val)  # sum
                cuda.atomic.add(stats_output, 2, val * val)  # sum_sq


def calculate_basin_statistics(residuals):
    """
    Calculate comprehensive statistics for each basin across all years.

    Parameters
    ----------
    residuals : Residual
        Residual object containing the dataset with basin assignments and residual values

    Returns
    -------
    dict
        Dictionary with structure: {year: {basin_name: {stat_name: value}}}
        Statistics include: count, mean, std, min, max, rms, rss, sum,
        winsorized_mean, outlier_weighted_mean

    """
    logger = logging.getLogger(__name__)
    logger.info("Starting basin statistics calculation")

    basin_stats = {}

    basin_names = residuals.ds.basin_names.values
    times = residuals.ds.time.values

    logger.info(f"Processing {len(times)} time steps and {len(basin_names)} basins")

    for time_idx, year in enumerate(times):
        if year not in basin_stats:
            basin_stats[year] = {}

        residual_data = residuals.ds.residual.isel(time=time_idx)
        basin_data = residuals.ds.basin.isel(time=time_idx)

        logger.debug(f"Processing year {year} (time index {time_idx})")

        for basin_idx, basin_name in enumerate(basin_names):
            basin_mask = basin_data == basin_idx

            basin_residuals = residual_data.where(basin_mask).values

            valid_data = basin_residuals[~np.isnan(basin_residuals)]

            if len(valid_data) > 0:
                logger.debug(
                    f"  Basin {basin_name}: {len(valid_data)} valid data points"
                )

                # Try GPU acceleration first
                gpu_stats = None
                if (
                    _GPU_CONFIG["use_gpu"] and len(valid_data) > 1000
                ):  # Use GPU for larger datasets
                    gpu_stats = _gpu_accelerated_basin_stats(
                        residual_data.values, basin_data.values, basin_idx
                    )

                if gpu_stats is not None:
                    # Use GPU-computed basic stats and add advanced stats
                    mean_val = gpu_stats["mean"]
                    std_val = gpu_stats["std"]
                    basic_stats = gpu_stats
                else:
                    # Fallback to CPU computation
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    basic_stats = {
                        "count": len(valid_data),
                        "mean": mean_val,
                        "std": std_val,
                        "min": np.min(valid_data),
                        "max": np.max(valid_data),
                        "rms": np.sqrt(np.mean(np.square(valid_data))),
                        "rss": np.sum(np.square(valid_data)),
                        "sum": np.sum(valid_data),
                    }

                # Calculate winsorized mean (trimmed mean with 5% limits)
                zero_fraction = np.sum(valid_data == 0) / len(valid_data)

                if zero_fraction > 0.9:  # If more than 90% are zeros
                    non_zero_data = valid_data[valid_data != 0]
                    if len(non_zero_data) > 0:
                        if len(non_zero_data) > 10:
                            winsorized_nonzero = stats.mstats.winsorize(
                                non_zero_data, limits=0.05
                            )
                            winsorized_mean_nonzero = float(np.mean(winsorized_nonzero))
                        else:
                            winsorized_mean_nonzero = np.mean(non_zero_data)
                        winsorized_mean = winsorized_mean_nonzero * (1 - zero_fraction)
                    else:
                        winsorized_mean = 0.0
                else:
                    winsorized_data = stats.mstats.winsorize(valid_data, limits=0.05)
                    winsorized_mean = float(np.mean(winsorized_data))

                median_val = np.median(valid_data)

                if zero_fraction > 0.8:  # For zero-heavy data
                    # Give more reasonable weights to avoid extreme values
                    weights = 1.0 / (np.abs(valid_data) + 0.01)  # Larger epsilon
                    weights = np.minimum(weights, 100.0)
                    outlier_weighted_mean = np.average(valid_data, weights=weights)
                else:
                    weights = 1.0 / (np.abs(valid_data - median_val) + 1e-6)
                    weights = np.minimum(weights, 1000.0)
                    outlier_weighted_mean = np.average(valid_data, weights=weights)

                # Combine basic stats with advanced stats
                basin_stats[year][basin_name] = {
                    **basic_stats,  # Use GPU-computed or CPU-computed basic stats
                    "winsorized_mean": winsorized_mean,
                    "outlier_weighted_mean": outlier_weighted_mean,
                }
            else:
                logger.warning(f"  Basin {basin_name}: No valid data for year {year}")
                basin_stats[year][basin_name] = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "rms": np.nan,
                    "rss": np.nan,
                    "sum": np.nan,
                    "winsorized_mean": np.nan,
                    "outlier_weighted_mean": np.nan,
                }

    logger.info("Basin statistics calculation completed")
    return basin_stats


def calculate_basin_statistics_with_mask(observations, basin_mask, basin_names=None):
    """
    Calculate basin statistics for observations data using a basin mask.

    This function assigns each point to a basin based on the basin mask and then
    calculates the same statistics as calculate_basin_statistics.

    Parameters
    ----------
    observations : observationsice_area_extent or Modelice_area_extent
        Input data object with ds attribute containing xarray Dataset.
    basin_mask : numpy.ndarray
        2D array where each point is assigned a basin index (or -1 if not in any basin).
        Shape should match the spatial dimensions of observations data.
    basin_names : list, optional
        List of basin names corresponding to basin indices. If None, will try to get
        from observations dataset or use default indices.

    Returns
    -------
    dict
        Dictionary with structure: {year: {basin_name: {stat_name: value}}}
        Statistics include: count, mean, std, min, max, rms, rss, sum,
        winsorized_mean, outlier_weighted_mean
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting basin statistics calculation with mask")

    # Get basin names - try from observations dataset first, then from parameter
    if basin_names is None:
        if (
            hasattr(observations.ds, "basin_names")
            and "basin_names" in observations.ds.coords
        ):
            basin_names = observations.ds.basin_names.values
        else:
            # Get unique basin indices from mask (excluding -1)
            unique_basins = np.unique(basin_mask)
            unique_basins = unique_basins[unique_basins >= 0]  # Remove -1 values
            basin_names = [f"Basin_{i}" for i in unique_basins]

    basin_stats = {}
    times = observations.time.values

    # Determine the time dimension name
    time_dim = "time"
    if (
        hasattr(observations, "ds")
        and "year" in observations.ds.dims
        and "year" in observations.ds.coords
    ):
        time_dim = "year"

    logger.info(
        f"Processing {len(times)} time steps and {len(basin_names)} basins using time dimension '{time_dim}'"
    )

    for time_idx, year in enumerate(times):
        if year not in basin_stats:
            basin_stats[year] = {}

        # Get observations data for this time step - handle different time dimension names
        if time_dim == "year":
            obs_data = observations.ds.ice_mask.isel(year=time_idx).values
        else:
            obs_data = observations.ds.ice_mask.isel(time=time_idx).values

        # Check if data dimensions match basin_mask and transpose if necessary
        if obs_data.shape != basin_mask.shape:
            logger.debug(
                f"Data shape {obs_data.shape} doesn't match basin_mask shape {basin_mask.shape}. Attempting to transpose."
            )
            if obs_data.shape == (basin_mask.shape[1], basin_mask.shape[0]):
                obs_data = obs_data.T
                logger.debug(f"Transposed data to shape {obs_data.shape}")
            else:
                logger.error(
                    f"Cannot reshape data from {obs_data.shape} to match basin_mask {basin_mask.shape}"
                )
                raise ValueError(
                    f"Data spatial dimensions {obs_data.shape} cannot be matched to basin_mask {basin_mask.shape}"
                )

        logger.debug(f"Processing year {year} (time index {time_idx})")

        for basin_idx, basin_name in enumerate(basin_names):
            # Create mask for this basin
            basin_points_mask = basin_mask == basin_idx

            # Extract data points that belong to this basin
            basin_obs_data = obs_data[basin_points_mask]

            # Remove invalid data (NaN values)
            valid_data = basin_obs_data[~np.isnan(basin_obs_data)]

            if len(valid_data) > 0:
                logger.debug(
                    f"  Basin {basin_name}: {len(valid_data)} valid data points"
                )

                # Calculate basic statistics (same as calculate_basin_statistics)
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)

                basic_stats = {
                    "count": len(valid_data),
                    "mean": mean_val,
                    "std": std_val,
                    "min": np.min(valid_data),
                    "max": np.max(valid_data),
                    "rms": np.sqrt(np.mean(np.square(valid_data))),
                    "rss": np.sum(np.square(valid_data)),
                    "sum": np.sum(valid_data),
                }

                # Calculate advanced statistics (same as calculate_basin_statistics)
                # Calculate winsorized mean (trimmed mean with 5% limits)
                zero_fraction = np.sum(valid_data == 0) / len(valid_data)

                if zero_fraction > 0.9:  # If more than 90% are zeros
                    non_zero_data = valid_data[valid_data != 0]
                    if len(non_zero_data) > 0:
                        if len(non_zero_data) > 10:
                            winsorized_nonzero = stats.mstats.winsorize(
                                non_zero_data, limits=0.05
                            )
                            winsorized_mean_nonzero = float(np.mean(winsorized_nonzero))
                        else:
                            winsorized_mean_nonzero = np.mean(non_zero_data)
                        winsorized_mean = winsorized_mean_nonzero * (1 - zero_fraction)
                    else:
                        winsorized_mean = 0.0
                else:
                    winsorized_data = stats.mstats.winsorize(valid_data, limits=0.05)
                    winsorized_mean = float(np.mean(winsorized_data))

                median_val = np.median(valid_data)

                if zero_fraction > 0.8:  # For zero-heavy data
                    # Give more reasonable weights to avoid extreme values
                    weights = 1.0 / (np.abs(valid_data) + 0.01)  # Larger epsilon
                    weights = np.minimum(weights, 100.0)
                    outlier_weighted_mean = np.average(valid_data, weights=weights)
                else:
                    weights = 1.0 / (np.abs(valid_data - median_val) + 1e-6)
                    weights = np.minimum(weights, 1000.0)
                    outlier_weighted_mean = np.average(valid_data, weights=weights)

                # Combine basic stats with advanced stats
                basin_stats[year][basin_name] = {
                    **basic_stats,
                    "winsorized_mean": winsorized_mean,
                    "outlier_weighted_mean": outlier_weighted_mean,
                }
            else:
                logger.warning(f"  Basin {basin_name}: No valid data for year {year}")
                basin_stats[year][basin_name] = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "rms": np.nan,
                    "rss": np.nan,
                    "sum": np.nan,
                    "winsorized_mean": np.nan,
                    "outlier_weighted_mean": np.nan,
                }

    logger.info("Basin statistics calculation with mask completed")
    return basin_stats


def format_basin_stats(basin_stats):
    """
    Format basin statistics in a readable format.

    Parameters
    ----------
    basin_stats : dict
        Dictionary containing basin statistics.

    Returns
    -------
    str
        Formatted string with basin statistics.
    """
    output_lines = []

    for i in basin_stats.keys():
        output_lines.append(f"=== Statistics for Year {i} ===")
        output_lines.append(
            "Basin | Sum       | Count    | Mean        | Winsorized  | Outlier Wgt | Std        | RMS"
        )
        output_lines.append("-" * 85)

        for basin_name, basin_stat in basin_stats[i].items():
            if basin_stat["count"] > 0:
                output_lines.append(
                    f"{basin_name:5} | {basin_stat['sum']:8} | {basin_stat['count']:8} | {basin_stat['mean']:11.8f} | {basin_stat['winsorized_mean']:11.8f} | {basin_stat['outlier_weighted_mean']:11.8f} | {basin_stat['std']:10.6f} | {basin_stat['rms']:10.6f}"
                )
        output_lines.append("-" * 85)
        output_lines.append("\n")

    return "\n".join(output_lines)

    # for basin_name, stats in basin_stats[2007].items():
    #     if stats["count"] > 0:
    #         print(
    #             f"{basin_name:5} | {stats['count']:8} | {stats['mean']:11.8f} | {stats['winsorized_mean']:11.8f} | {stats['outlier_weighted_mean']:11.8f} | {stats['std']:10.6f} | {stats['rms']:10.6f}"
    #         )


def calculate_observations_statistics(observations, basin_polygons_dict):
    """
    Calculate comprehensive statistics for each basin across all years for observations data.

    Parameters
    ----------
    observations : observationsice_area_extent
        observations ice_area_extent object containing the dataset with ice mask values
    basin_polygons_dict : dict
        Dictionary containing basin polygons for spatial assignment

    Returns
    -------
    dict
        Dictionary with structure: {basin_name: {year: {stat_name: value}}}
        Statistics include: count, mean, std, min, max, rms, rss, sum,
        winsorized_mean, outlier_weighted_mean
    """

    logger = logging.getLogger(__name__)
    logger.info("Starting observations basin statistics calculation")

    # Import the required modules for basin assignment
    from cmct.ice_area_extent_modules.residual_calculation import (
        # create_basin_mask_debug,
        create_basin_mask_optimized,
        prepare_basin_polygons,
    )

    # Get times from observations data
    times = observations.time.values
    logger.info(f"Processing {len(times)} time steps")

    # Get coordinate arrays from observations data
    x_coords = observations.x.values.astype(np.float32)
    y_coords = observations.y.values.astype(np.float32)

    logger.info(f"observations grid dimensions: {len(y_coords)} x {len(x_coords)}")

    # Prepare basin data for assignment
    basin_names, basin_polygons_x, basin_polygons_y, basin_lengths = (
        prepare_basin_polygons(basin_polygons_dict)
    )

    logger.info(f"Found {len(basin_names)} basins: {basin_names}")

    # Create basin mask for observations coordinates
    logger.info("Creating basin assignments for observations coordinates...")

    basin_mask = create_basin_mask_optimized(
        x_coords, y_coords, basin_polygons_x, basin_polygons_y, basin_lengths
    )

    # Debug: Check basin assignment results
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

    # Initialize nested dictionary structure: {basin_name: {year: {stats}}}
    basin_stats = {}
    for basin_name in basin_names:
        basin_stats[basin_name] = {}

    # Process each time step
    for time_idx, year in enumerate(times):
        logger.debug(f"Processing year {year} (time index {time_idx})")
        ice_mask_data = observations.ds.ice_mask.isel(year=time_idx).values

        # Process each basin
        for basin_idx, basin_name in enumerate(basin_names):
            basin_mask_current = basin_mask == basin_idx

            basin_ice_mask = ice_mask_data[basin_mask_current]

            valid_data = basin_ice_mask[~np.isnan(basin_ice_mask)]

            if len(valid_data) > 0:
                logger.debug(
                    f"  Basin {basin_name}: {len(valid_data)} valid data points"
                )

                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)

                zero_fraction = np.sum(valid_data == 0) / len(valid_data)

                if zero_fraction > 0.9:  # If more than 90% are zeros
                    non_zero_data = valid_data[valid_data != 0]
                    if len(non_zero_data) > 0:
                        if len(non_zero_data) > 10:
                            winsorized_nonzero = stats.mstats.winsorize(
                                non_zero_data, limits=0.05
                            )
                            winsorized_mean_nonzero = float(np.mean(winsorized_nonzero))
                        else:
                            winsorized_mean_nonzero = np.mean(non_zero_data)
                        winsorized_mean = winsorized_mean_nonzero * (1 - zero_fraction)
                    else:
                        winsorized_mean = 0.0
                else:
                    winsorized_data = stats.mstats.winsorize(valid_data, limits=0.05)
                    winsorized_mean = float(np.mean(winsorized_data))

                # Calculate outlier-weighted mean
                median_val = np.median(valid_data)

                if zero_fraction > 0.8:  # For zero-heavy data
                    # Use squared distance to reduce extreme weighting

                    weights = 1.0 / (np.abs(valid_data) + 0.01)  # Larger epsilon
                    weights = np.minimum(weights, 100.0)  # Cap at 100x weight
                    outlier_weighted_mean = np.average(valid_data, weights=weights)
                else:
                    # Standard outlier weighting for more balanced data
                    weights = 1.0 / (np.abs(valid_data - median_val) + 1e-6)
                    weights = np.minimum(weights, 1000.0)
                    outlier_weighted_mean = np.average(valid_data, weights=weights)

                basin_stats[basin_name][year] = {
                    "count": len(valid_data),
                    "mean": mean_val,
                    "std": std_val,
                    "min": np.min(valid_data),
                    "max": np.max(valid_data),
                    "rms": np.sqrt(np.mean(np.square(valid_data))),
                    "rss": np.sum(np.square(valid_data)),
                    "sum": np.sum(valid_data),
                    "winsorized_mean": winsorized_mean,
                    "outlier_weighted_mean": outlier_weighted_mean,
                }
            else:
                # No valid data for this basin/year combination
                logger.warning(f"  Basin {basin_name}: No valid data for year {year}")
                basin_stats[basin_name][year] = {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "rms": np.nan,
                    "rss": np.nan,
                    "sum": np.nan,
                    "winsorized_mean": np.nan,
                    "outlier_weighted_mean": np.nan,
                }

    logger.info("observations basin statistics calculation completed")
    return basin_stats


def format_observations_basin_stats(basin_stats):
    """
    Format observations basin statistics in a readable format.

    Parameters
    ----------
    basin_stats : dict
        Dictionary containing observations basin statistics with structure:
        {basin_name: {year: {stat_name: value}}}

    Returns
    -------
    str
        Formatted string with basin statistics organized by basin and year.
    """
    output_lines = []
    output_lines.append("=== observations Basin Statistics ===\n")

    for basin_name, yearly_stats in basin_stats.items():
        if not yearly_stats:  # Skip basins with no data
            continue

        output_lines.append(f"Basin: {basin_name}")
        output_lines.append("-" * 100)
        output_lines.append(
            "Year | Count    | Mean        | Winsorized  | Outlier Wgt | Std        | RMS        | Sum      | Min      | Max"
        )
        output_lines.append("-" * 100)

        # Sort years for consistent output
        sorted_years = sorted(yearly_stats.keys())

        for year in sorted_years:
            stats = yearly_stats[year]
            if stats["count"] > 0:
                output_lines.append(
                    f"{year:4} | {stats['count']:8} | {stats['mean']:11.8f} | {stats['winsorized_mean']:11.8f} | {stats['outlier_weighted_mean']:11.8f} | {stats['std']:10.6f} | {stats['rms']:10.6f} | {stats['sum']:8.4f} | {stats['min']:8.4f} | {stats['max']:8.4f}"
                )
            else:
                output_lines.append(
                    f"{year:4} | {stats['count']:8} | {'N/A':>11} | {'N/A':>11} | {'N/A':>11} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}"
                )

        output_lines.append("-" * 100)
        output_lines.append("")  # Add spacing between basins

    return "\n".join(output_lines)


def get_gpu_config():
    """
    Get current GPU configuration.

    Returns
    -------
    dict
        GPU configuration dictionary with platform, availability flags, and usage status
    """
    return _GPU_CONFIG.copy()


def is_gpu_available():
    """
    Check if GPU acceleration is available and enabled.

    Returns
    -------
    bool
        True if GPU acceleration is available and enabled
    """
    return _GPU_CONFIG["use_gpu"]
