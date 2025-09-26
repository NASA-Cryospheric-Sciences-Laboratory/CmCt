import gc
import logging
import os
import sys

import netCDF4 as nc
import numpy as np
import rioxarray as rxr
import xarray as xr
from PIL import Image
from ..ice_area_extent import configure_ice_area_extent_logging
# Create module-specific logger
logger = configure_ice_area_extent_logging()


def detect_coordinate_alignment(model_coords, target_coords, tolerance=1e-6):
    """
    Detect if there's a fixed offset between coordinate systems and calculate alignment parameters.

    Parameters
    ----------
    model_coords : numpy.ndarray
        Model coordinate values
    target_coords : numpy.ndarray
        Target coordinate values
    tolerance : float
        Tolerance for detecting uniform spacing

    Returns
    -------
    dict
        Dictionary containing alignment information:
        - 'can_align': bool, whether coordinates can be aligned through trimming
        - 'model_start_idx': int,
        - 'model_end_idx': int,
        - 'target_start_idx': int
        - 'target_end_idx': int
        - 'offset': float
    """

    # Check if both coordinate arrays have uniform spacing
    model_spacing = np.diff(model_coords)
    target_spacing = np.diff(target_coords)

    model_uniform = np.allclose(model_spacing, model_spacing[0], rtol=tolerance)
    target_uniform = np.allclose(target_spacing, target_spacing[0], rtol=tolerance)

    if not (model_uniform and target_uniform):
        return {"can_align": False, "reason": "Non-uniform spacing detected"}

    if not np.allclose(model_spacing[0], target_spacing[0], rtol=tolerance):
        return {"can_align": False, "reason": "Different spacing between coordinates"}

    offset = model_coords[0] - target_coords[0]

    model_min, model_max = model_coords[0], model_coords[-1]
    target_min, target_max = target_coords[0], target_coords[-1]

    overlap_min = max(model_min, target_min)
    overlap_max = min(model_max, target_max)

    if overlap_min >= overlap_max:
        return {"can_align": False, "reason": "No overlap between coordinate ranges"}

    model_start_idx = np.argmin(np.abs(model_coords - overlap_min))
    model_end_idx = np.argmin(np.abs(model_coords - overlap_max)) + 1

    target_start_idx = np.argmin(np.abs(target_coords - overlap_min))
    target_end_idx = np.argmin(np.abs(target_coords - overlap_max)) + 1

    return {
        "can_align": True,
        "model_start_idx": model_start_idx,
        "model_end_idx": model_end_idx,
        "target_start_idx": target_start_idx,
        "target_end_idx": target_end_idx,
        "offset": offset,
        "overlap_min": overlap_min,
        "overlap_max": overlap_max,
    }


def trim_to_align_shapes(model_ds, target_ds, x_dim="x", y_dim="y"):
    """
    Trim model dataset to align with target dataset when there's a fixed offset.

    Parameters
    ----------
    model_ds : xarray.Dataset
        Model dataset to trim
    target_ds : xarray.Dataset
        Target dataset to align with
    x_dim : str
        Name of x dimension (default: 'x')
    y_dim : str
        Name of y dimension (default: 'y')

    Returns
    -------
    xarray.Dataset
        Trimmed model dataset aligned with target, or original if alignment not possible
    dict
        Information about the alignment process
    """
    logger.warning("Checking coordinate alignment for shape trimming...")

    # Get coordinate arrays
    model_x = model_ds[x_dim].values
    model_y = model_ds[y_dim].values
    target_x = target_ds[x_dim].values
    target_y = target_ds[y_dim].values

    # Check x-coordinate alignment
    x_alignment = detect_coordinate_alignment(model_x, target_x)
    y_alignment = detect_coordinate_alignment(model_y, target_y)

    alignment_info = {
        "x_alignment": x_alignment,
        "y_alignment": y_alignment,
        "trimmed": False,
        "original_shape": (len(model_x), len(model_y)),
        "target_shape": (len(target_x), len(target_y)),
    }

    if not x_alignment["can_align"] or not y_alignment["can_align"]:
        logger.info(
            f"Cannot align coordinates: X - {x_alignment.get('reason', 'Unknown')}, Y - {y_alignment.get('reason', 'Unknown')}"
        )
        return model_ds, alignment_info

    try:
        x_slice = slice(x_alignment["model_start_idx"], x_alignment["model_end_idx"])
        y_slice = slice(y_alignment["model_start_idx"], y_alignment["model_end_idx"])

        trimmed_ds = model_ds.isel({x_dim: x_slice, y_dim: y_slice})
        target_x_overlap = target_x[
            x_alignment["target_start_idx"] : x_alignment["target_end_idx"]
        ]
        target_y_overlap = target_y[
            y_alignment["target_start_idx"] : y_alignment["target_end_idx"]
        ]

        trimmed_ds = trimmed_ds.assign_coords(
            {x_dim: target_x_overlap, y_dim: target_y_overlap}
        )

        alignment_info.update(
            {
                "trimmed": True,
                "final_shape": (len(target_x_overlap), len(target_y_overlap)),
                "x_offset": x_alignment["offset"],
                "y_offset": y_alignment["offset"],
            }
        )

        logger.info(
            f"Successfully trimmed model from {alignment_info['original_shape']} to {alignment_info['final_shape']}"
        )
        logger.info(
            f"X offset: {x_alignment['offset']:.2f}, Y offset: {y_alignment['offset']:.2f}"
        )

        return trimmed_ds, alignment_info

    except Exception as e:
        logger.error(f"Error during trimming: {e}")
        return model_ds, alignment_info


def need_for_interpolation(observations, model):
    """
    Check if interpolation is needed between model and observation data.

    Parameters
    ----------
    observations : observationsice_area_extent
        Observation data object
    model : Modelice_area_extent
        Model data object

    Returns
    -------
    dict
        Dictionary containing:
        - 'needs_processing': bool
        - 'can_trim': bool
        - 'needs_interpolation': bool
        - 'reason': str
    """
    model_x, model_y = model.ds.x.values, model.ds.y.values
    observations_x, observations_y = observations.ds.x.values, observations.ds.y.values

    if np.array_equal(model_x, observations_x) and np.array_equal(
        model_y, observations_y
    ):
        return {
            "needs_processing": False,
            "can_trim": False,
            "needs_interpolation": False,
            "reason": "Coordinates are identical",
        }

    x_alignment = detect_coordinate_alignment(model_x, observations_x)
    y_alignment = detect_coordinate_alignment(model_y, observations_y)

    if x_alignment["can_align"] and y_alignment["can_align"]:
        return {
            "needs_processing": True,
            "can_trim": True,
            "needs_interpolation": False,
            "reason": "Coordinate alignment possible through trimming",
            "x_offset": x_alignment["offset"],
            "y_offset": y_alignment["offset"],
        }

    if model_x.size != observations_x.size or model_y.size != observations_y.size:
        logger.warning(
            f"Model x size: {model_x.size}, observations x size: {observations_x.size}, "
            f"Model y size: {model_y.size}, observations y size: {observations_y.size}. "
            "Interpolation needed."
        )
        return {
            "needs_processing": True,
            "can_trim": False,
            "needs_interpolation": True,
            "reason": "Different grid sizes - interpolation required",
        }

    # Check if spacing is different
    model_dx = model_x[1] - model_x[0] if len(model_x) > 1 else 0
    model_dy = model_y[1] - model_y[0] if len(model_y) > 1 else 0
    observations_dx = (
        observations_x[1] - observations_x[0] if len(observations_x) > 1 else 0
    )
    observations_dy = (
        observations_y[1] - observations_y[0] if len(observations_y) > 1 else 0
    )

    if not np.allclose(model_dx, observations_dx) or not np.allclose(
        model_dy, observations_dy
    ):
        return {
            "needs_processing": True,
            "can_trim": False,
            "needs_interpolation": True,
            "reason": "Different grid spacing - interpolation required",
        }

    # If we get here, the grids have same size and spacing but different origins
    return {
        "needs_processing": True,
        "can_trim": False,
        "needs_interpolation": True,
        "reason": "Different coordinate origins - interpolation required",
    }


class Interpolater:
    def __init__(
        self,
        input_data,
        target_data=None,
        target_resolution=None,
        interpolation_method="linear",
    ):
        self.input_data = input_data
        self.target_data = target_data
        self.target_resolution = target_resolution
        self.interpolation_method = interpolation_method

        if isinstance(target_resolution, str):
            raise ValueError(
                f"ERROR: interpolation_method '{target_resolution}' was passed as target_resolution. "
                f"Use keyword arguments: Interpolater(input_data, target_data=target, interpolation_method='{target_resolution}')"
            )

    def interpolate(self):
        """
        First attempts to align coordinates through trimming if possible,
        then falls back to interpolation if needed.
        """
        input_ds = self.input_data.ds

        obs_x = input_ds.x
        obs_y = input_ds.y

        logger.info(f"Input x coordinates: {obs_x.values}")
        logger.info(f"Input y coordinates: {obs_y.values}")

        if self.target_data is not None:
            target_ds = self.target_data.ds
            tgt_x = target_ds.x
            tgt_y = target_ds.y

            x_same = np.array_equal(tgt_x.values, obs_x.values)
            y_same = np.array_equal(tgt_y.values, obs_y.values)

            logger.debug(f"Target x coordinates: {tgt_x.values}")
            logger.debug(f"Target y coordinates: {tgt_y.values}")

            if x_same and y_same:
                return input_ds

            trimmed_ds, alignment_info = trim_to_align_shapes(input_ds, target_ds)

            if alignment_info["trimmed"]:
                logger.info("Successfully aligned datasets through coordinate trimming")
                return trimmed_ds
            else:
                logger.info(
                    "Coordinate trimming not possible, falling back to interpolation"
                )
                ds_resampled = self._perform_interpolation(input_ds, tgt_x, tgt_y)

        elif self.target_resolution is not None:
            target_x = np.linspace(obs_x.min(), obs_x.max(), self.target_resolution[0])
            target_y = np.linspace(obs_y.min(), obs_y.max(), self.target_resolution[1])

            ds_resampled = self._perform_interpolation(input_ds, target_x, target_y)

        else:
            raise ValueError(
                "Either target_data or target_resolution must be provided for resampling."
            )

        ds_resampled = ds_resampled.fillna(0)

        del obs_x, obs_y

        if self.target_data is not None:
            del target_ds, tgt_x, tgt_y
        gc.collect()

        return ds_resampled

    def _perform_interpolation(self, input_ds, target_x, target_y):
        """
        Perform the actual interpolation based on the selected method.

        Parameters
        ----------
        input_ds : xarray.Dataset
            Input dataset to interpolate
        target_x and target_y : xarray.DataArray or numpy.ndarray
            Target x and y coordinates

        Returns
        -------
        xarray.Dataset
            Interpolated dataset
        """
        if self.interpolation_method == "linear":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="linear")

        elif self.interpolation_method == "nearest":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="nearest")

        elif self.interpolation_method == "akima":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="akima")

        elif self.interpolation_method == "pchip":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="pchip")

        elif self.interpolation_method == "cubic":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="cubic")

        elif self.interpolation_method == "slinear":
            ds_resampled = input_ds.interp(x=target_x, y=target_y, method="slinear")

        elif self.interpolation_method == "lanczos3d":
            ds_resampled_vars = {}
            ds_resampled_coords = {}

            target_x_values = (
                target_x.values if hasattr(target_x, "values") else target_x
            )
            target_y_values = (
                target_y.values if hasattr(target_y, "values") else target_y
            )

            for var_name, var_data in input_ds.data_vars.items():
                if len(var_data.dims) == 2 and (
                    "y" in var_data.dims and "x" in var_data.dims
                ):
                    ice_array = var_data.values

                    if var_data.dims == ("x", "y"):
                        ice_array = ice_array.T

                    ice_array_clean = np.nan_to_num(ice_array, nan=0.0)
                    ice_min, ice_max = ice_array_clean.min(), ice_array_clean.max()

                    if ice_max > ice_min:
                        ice_normalized = (
                            (ice_array_clean - ice_min) / (ice_max - ice_min) * 255
                        ).astype(np.uint8)
                    else:
                        ice_normalized = np.zeros_like(ice_array_clean, dtype=np.uint8)

                    target_width = len(target_x_values)
                    target_height = len(target_y_values)

                    resized_normalized = np.array(
                        Image.fromarray(ice_normalized).resize(
                            (target_width, target_height), Image.LANCZOS
                        )
                    )

                    if ice_max > ice_min:
                        resized = (resized_normalized.astype(np.float32) / 255.0) * (
                            ice_max - ice_min
                        ) + ice_min
                    else:
                        resized = np.full_like(
                            resized_normalized, ice_min, dtype=np.float32
                        )

                    ds_resampled_vars[var_name] = (["y", "x"], resized)

                    del (
                        ice_array,
                        ice_array_clean,
                        ice_normalized,
                        resized_normalized,
                        resized,
                    )

                elif len(var_data.dims) == 3 and (
                    "y" in var_data.dims and "x" in var_data.dims
                ):
                    spatial_dims = ["x", "y"]
                    other_dims = [
                        dim for dim in var_data.dims if dim not in spatial_dims
                    ]

                    if len(other_dims) == 1:
                        other_dim = other_dims[0]
                        other_coord = input_ds.coords[other_dim]

                        processed_slices = []

                        for i in range(len(other_coord)):
                            slice_data = var_data.isel({other_dim: i})

                            ice_array = slice_data.values
                            if slice_data.dims == ("x", "y"):
                                ice_array = ice_array.T

                            ice_array_clean = np.nan_to_num(ice_array, nan=0.0)
                            ice_min, ice_max = (
                                ice_array_clean.min(),
                                ice_array_clean.max(),
                            )

                            if ice_max > ice_min:
                                ice_normalized = (
                                    (ice_array_clean - ice_min)
                                    / (ice_max - ice_min)
                                    * 255
                                ).astype(np.uint8)
                            else:
                                ice_normalized = np.zeros_like(
                                    ice_array_clean, dtype=np.uint8
                                )

                            resized_normalized = np.array(
                                Image.fromarray(ice_normalized).resize(
                                    (len(target_x_values), len(target_y_values)),
                                    Image.LANCZOS,
                                )
                            )

                            if ice_max > ice_min:
                                resized = (
                                    resized_normalized.astype(np.float32) / 255.0
                                ) * (ice_max - ice_min) + ice_min
                            else:
                                resized = np.full_like(
                                    resized_normalized, ice_min, dtype=np.float32
                                )

                            processed_slices.append(resized)

                            del (
                                ice_array,
                                ice_array_clean,
                                ice_normalized,
                                resized_normalized,
                                resized,
                            )

                        stacked_data = np.stack(processed_slices, axis=0)

                        if var_data.dims == (other_dim, "y", "x"):
                            ds_resampled_vars[var_name] = (
                                [other_dim, "y", "x"],
                                stacked_data,
                            )
                        elif var_data.dims == ("y", "x", other_dim):
                            stacked_data = np.transpose(stacked_data, (1, 2, 0))
                            ds_resampled_vars[var_name] = (
                                ["y", "x", other_dim],
                                stacked_data,
                            )
                        else:
                            ds_resampled_vars[var_name] = (
                                [other_dim, "y", "x"],
                                stacked_data,
                            )

                        ds_resampled_coords[other_dim] = other_coord

                        del processed_slices, stacked_data
                    else:
                        var_interp = var_data.interp(
                            x=target_x, y=target_y, method="nearest"
                        )
                        ds_resampled_vars[var_name] = var_interp
                else:
                    try:
                        var_interp = var_data.interp(
                            x=target_x, y=target_y, method="nearest"
                        )
                        ds_resampled_vars[var_name] = var_interp
                    except Exception as e:
                        logger.warning(
                            f"Failed to interpolate variable {var_name}: {e}"
                        )
                        ds_resampled_vars[var_name] = var_data

            coords_dict = {"y": target_y_values, "x": target_x_values}
            coords_dict.update(ds_resampled_coords)

            ds_resampled = xr.Dataset(ds_resampled_vars, coords=coords_dict)

            del ds_resampled_vars, ds_resampled_coords
            gc.collect()

        else:
            raise ValueError(
                f"Unsupported interpolation method: {self.interpolation_method}. "
                f"Supported methods are: 'linear', 'nearest', 'cubic', 'akima', 'pchip', 'slinear', 'lanczos3d'"
            )

        return ds_resampled
