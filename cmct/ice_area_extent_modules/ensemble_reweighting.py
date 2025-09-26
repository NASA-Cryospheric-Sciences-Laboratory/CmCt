"""
Ensemble Reweighting Module for Ice Area Extent Analysis

This module provides functionality to:
1. Create scores for each ensemble member based on how well it reproduces observations
2. Implement KDE (Kernel Density Estimation) weighting using normalized scores
3. Reweight ensemble statistics based on performance scores

Author: CmCt Ice Area Extent Tool
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import gaussian_kde

# Create module-specific logger
logger = logging.getLogger(__name__)


class EnsembleReweighter:
    """
    A class to handle ensemble member scoring and reweighting based on
    observation-model comparison statistics.
    """

    def __init__(self, observations_stats: Dict, scoring_method: str = "rmse"):
        """
        Initialize the EnsembleReweighter

        Parameters
        ----------
        observations_stats : dict
            Dictionary containing observation statistics by year and basin
        scoring_method : str, default 'rmse'
            Method to use for scoring. Options: 'rmse', 'mae', 'correlation', 'combined'
        """
        self.observations_stats = observations_stats
        self.scoring_method = scoring_method
        self.ensemble_scores = None
        self.normalized_scores = None
        self.weights = None

    def calculate_ensemble_scores(
        self,
        basin_stats_array: List[Dict],
        basin_list: List[str] = None,
        years: List[int] = None,
    ) -> np.ndarray:
        """
        Calculate performance scores for each ensemble member

        Parameters
        ----------
        basin_stats_array : list
            List of dictionaries containing basin statistics for each model
        basin_list : list, optional
            List of basin names to include in scoring
        years : list, optional
            List of years to include in scoring

        Returns
        -------
        np.ndarray
            Array of scores for each ensemble member (lower is better)
        """
        n_models = len(basin_stats_array)
        scores = np.zeros(n_models)

        # Default to all available basins and years if not specified
        if basin_list is None:
            basin_list = self._get_available_basins(basin_stats_array)
        if years is None:
            years = self._get_available_years(basin_stats_array)

        logger.info(
            f"Calculating scores for {n_models} models using {self.scoring_method} method"
        )
        logger.info(f"Including basins: {basin_list}")
        logger.info(f"Including years: {years}")

        for i, model_stats in enumerate(basin_stats_array):
            model_score = self._calculate_model_score(model_stats, basin_list, years)
            scores[i] = model_score

        self.ensemble_scores = scores
        logger.info(f"Score range: {np.min(scores):.4f} to {np.max(scores):.4f}")

        return scores

    def _calculate_model_score(
        self, model_stats: Dict, basin_list: List[str], years: List[int]
    ) -> float:
        """
        Calculate score for a single model based on comparison with observations

        Parameters
        ----------
        model_stats : dict
            Model statistics by year and basin
        basin_list : list
            List of basin names
        years : list
            List of years

        Returns
        -------
        float
            Model score (lower is better)
        """
        scores_per_basin_year = []

        for year in years:
            for basin in basin_list:
                try:
                    # Get model statistics
                    model_mean = (
                        model_stats.get(year, {}).get(basin, {}).get("mean", np.nan)
                    )
                    model_std = (
                        model_stats.get(year, {}).get(basin, {}).get("std", np.nan)
                    )

                    # Get observation statistics - handle basin-first data structure
                    obs_mean = np.nan
                    obs_std = np.nan

                    if basin in self.observations_stats:
                        year_key = float(
                            year
                        )  # Convert to float as numpy stores years as float64
                        if year_key in self.observations_stats[basin]:
                            obs_data = self.observations_stats[basin][year_key]
                            obs_mean = obs_data.get("mean", np.nan)
                            obs_std = obs_data.get("std", np.nan)

                    # Skip if any values are missing
                    if (
                        np.isnan(model_mean)
                        or np.isnan(model_std)
                        or np.isnan(obs_mean)
                        or np.isnan(obs_std)
                    ):
                        continue

                    # Calculate score based on method
                    if self.scoring_method == "rmse":
                        score = np.sqrt((model_mean - obs_mean) ** 2)
                    elif self.scoring_method == "mae":
                        score = abs(model_mean - obs_mean)
                    elif self.scoring_method == "correlation":
                        # Use inverse of normalized covariance as score
                        score = abs((model_mean - obs_mean) / (obs_std + 1e-10))
                    elif self.scoring_method == "combined":
                        # Combined score: mean bias + std bias
                        mean_bias = abs(model_mean - obs_mean) / (abs(obs_mean) + 1e-10)
                        std_bias = abs(model_std - obs_std) / (obs_std + 1e-10)
                        score = mean_bias + std_bias
                    else:
                        raise ValueError(
                            f"Unknown scoring method: {self.scoring_method}"
                        )

                    scores_per_basin_year.append(score)

                except Exception as e:
                    logger.warning(
                        f"Error calculating score for year {year}, basin {basin}: {e}"
                    )
                    continue

        if not scores_per_basin_year:
            logger.warning("No valid scores calculated for this model")
            return np.inf

        # Return mean score across all basin-year combinations
        mean_score = np.mean(scores_per_basin_year)
        return mean_score

    def normalize_scores(self, method: str = "minmax") -> np.ndarray:
        """
        Normalize scores to create weights (higher normalized score = better model)

        Parameters
        ----------
        method : str, default 'minmax'
            Normalization method: 'minmax', 'zscore', 'rank', or 'inverse'

        Returns
        -------
        np.ndarray
            Normalized scores (higher values indicate better models)
        """
        if self.ensemble_scores is None:
            raise ValueError("Must calculate ensemble scores first")

        scores = self.ensemble_scores.copy()

        # Handle infinite or invalid scores
        valid_mask = np.isfinite(scores)
        if not np.any(valid_mask):
            logger.warning("No valid scores found, using uniform weights")
            normalized = np.ones(len(scores)) / len(scores)
        else:
            if method == "minmax":
                # Convert to 0-1 range, with 1 being best (lowest original score)
                min_score = np.min(scores[valid_mask])
                max_score = np.max(scores[valid_mask])
                if max_score == min_score:
                    normalized = np.ones(len(scores))
                else:
                    normalized = (max_score - scores) / (max_score - min_score)

            elif method == "inverse":
                # Use inverse of scores (with small offset for numerical stability)
                normalized = 1.0 / (scores + np.min(scores[valid_mask]) * 0.01 + 1e-10)

            elif method == "zscore":
                # Z-score normalization, then convert to positive weights
                mean_score = np.mean(scores[valid_mask])
                std_score = np.std(scores[valid_mask])
                if std_score == 0:
                    normalized = np.ones(len(scores))
                else:
                    z_scores = (
                        mean_score - scores
                    ) / std_score  # Flip sign so higher is better
                    normalized = np.exp(z_scores)  # Convert to positive values

            elif method == "rank":
                # Rank-based normalization
                ranks = stats.rankdata(-scores)  # Negative for descending order
                normalized = ranks / len(scores)

            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Set invalid scores to minimum weight
            normalized[~valid_mask] = np.min(normalized[valid_mask]) * 0.1

        # Ensure all weights are positive
        normalized = np.maximum(normalized, 1e-10)

        self.normalized_scores = normalized
        logger.info(
            f"Normalized scores range: {np.min(normalized):.4f} to {np.max(normalized):.4f}"
        )

        return normalized

    def calculate_kde_weights(
        self, bandwidth: Optional[float] = None, adaptive: bool = True
    ) -> np.ndarray:
        """
        Calculate KDE-based weights using normalized scores

        Parameters
        ----------
        bandwidth : float, optional
            KDE bandwidth. If None, uses Scott's rule
        adaptive : bool, default True
            Whether to use adaptive bandwidth based on local density

        Returns
        -------
        np.ndarray
            KDE-based weights for ensemble members
        """
        if self.normalized_scores is None:
            raise ValueError("Must normalize scores first")

        scores = self.normalized_scores.copy()

        # Handle edge cases
        if len(np.unique(scores)) < 2:
            logger.warning("All scores are identical, using uniform weights")
            weights = np.ones(len(scores)) / len(scores)
        else:
            try:
                # Create KDE
                if bandwidth is None:
                    kde = gaussian_kde(scores)
                else:
                    kde = gaussian_kde(scores, bw_method=bandwidth)

                # Evaluate KDE at each score point
                densities = kde(scores)

                if adaptive:
                    # Adaptive weighting: higher density = higher weight
                    # But also consider the score value itself
                    weights = densities * scores
                else:
                    # Simple density-based weighting
                    weights = densities

                # Normalize to sum to 1
                weights = weights / np.sum(weights)

            except Exception as e:
                logger.warning(f"Error calculating KDE weights: {e}")
                weights = self.normalized_scores / np.sum(self.normalized_scores)

        self.weights = weights
        logger.info(
            f"KDE weights range: {np.min(weights):.4f} to {np.max(weights):.4f}"
        )
        logger.info(f"Effective sample size: {1 / np.sum(weights**2):.2f}")

        return weights

    def get_reweighted_statistics(
        self,
        basin_stats_array: List[Dict],
        basin_list: List[str] = None,
        years: List[int] = None,
    ) -> Dict:
        """
        Calculate reweighted ensemble statistics

        Parameters
        ----------
        basin_stats_array : list
            List of model statistics dictionaries
        basin_list : list, optional
            List of basin names
        years : list, optional
            List of years

        Returns
        -------
        dict
            Reweighted ensemble statistics
        """
        if self.weights is None:
            raise ValueError("Must calculate weights first")

        if basin_list is None:
            basin_list = self._get_available_basins(basin_stats_array)
        if years is None:
            years = self._get_available_years(basin_stats_array)

        reweighted_stats = {}

        for year in years:
            reweighted_stats[year] = {}
            for basin in basin_list:
                # Collect all model values for this year/basin
                values = []
                model_weights = []

                for i, model_stats in enumerate(basin_stats_array):
                    try:
                        value = (
                            model_stats.get(year, {}).get(basin, {}).get("mean", np.nan)
                        )
                        if not np.isnan(value):
                            values.append(value)
                            model_weights.append(self.weights[i])
                    except Exception as e:
                        logger.warning(f"Error accessing model stats: {e}")
                        continue

                if values:
                    values = np.array(values)
                    model_weights = np.array(model_weights)

                    # Normalize weights for this subset
                    if np.sum(model_weights) > 0:
                        model_weights = model_weights / np.sum(model_weights)
                    else:
                        model_weights = np.ones(len(values)) / len(values)

                    # Calculate weighted statistics
                    weighted_mean = np.average(values, weights=model_weights)
                    weighted_var = np.average(
                        (values - weighted_mean) ** 2, weights=model_weights
                    )
                    weighted_std = np.sqrt(weighted_var)

                    # Calculate quantiles
                    sorted_idx = np.argsort(values)
                    cumulative_weights = np.cumsum(model_weights[sorted_idx])

                    def weighted_quantile(q):
                        idx = np.searchsorted(cumulative_weights, q)
                        if idx >= len(values):
                            return values[sorted_idx[-1]]
                        return values[sorted_idx[idx]]

                    reweighted_stats[year][basin] = {
                        "mean": weighted_mean,
                        "std": weighted_std,
                        "count": len(values),
                        "q25": weighted_quantile(0.25),
                        "q50": weighted_quantile(0.50),
                        "q75": weighted_quantile(0.75),
                        "effective_n": 1 / np.sum(model_weights**2),
                    }
                else:
                    reweighted_stats[year][basin] = {
                        "mean": np.nan,
                        "std": np.nan,
                        "count": 0,
                        "q25": np.nan,
                        "q50": np.nan,
                        "q75": np.nan,
                        "effective_n": 0,
                    }

        return reweighted_stats

    def get_weight_summary(self) -> pd.DataFrame:
        """
        Get a summary of the weights assigned to each ensemble member

        Returns
        -------
        pd.DataFrame
            Summary dataframe with scores and weights
        """
        if self.weights is None:
            raise ValueError("Must calculate weights first")

        summary_df = pd.DataFrame(
            {
                "model_index": range(len(self.weights)),
                "raw_score": self.ensemble_scores
                if self.ensemble_scores is not None
                else np.nan,
                "normalized_score": self.normalized_scores
                if self.normalized_scores is not None
                else np.nan,
                "weight": self.weights,
                "weight_percentile": stats.rankdata(self.weights, method="average")
                / len(self.weights)
                * 100,
            }
        )

        return summary_df.sort_values("weight", ascending=False)

    def _get_available_basins(self, basin_stats_array: List[Dict]) -> List[str]:
        """Get list of available basins from the first model"""
        if not basin_stats_array:
            return []

        first_model = basin_stats_array[0]
        if not first_model:
            return []

        first_year = list(first_model.keys())[0]
        return list(first_model[first_year].keys())

    def _get_available_years(self, basin_stats_array: List[Dict]) -> List[int]:
        """Get list of available years from the first model"""
        if not basin_stats_array:
            return []

        first_model = basin_stats_array[0]
        return [int(year) for year in first_model.keys()]


def create_performance_comparison_plot(
    reweighter: EnsembleReweighter,
    model_names: List[str] = None,
    title: str = "Ensemble Member Performance",
) -> go.Figure:
    """
    Create an interactive plot comparing model performance scores and weights

    Parameters
    ----------
    reweighter : EnsembleReweighter
        Fitted reweighter object
    model_names : list, optional
        List of model names for labeling
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if reweighter.weights is None:
        raise ValueError("Reweighter must have calculated weights")

    summary_df = reweighter.get_weight_summary()

    if model_names is None:
        model_names = [f"Model_{i}" for i in range(len(reweighter.weights))]

    summary_df["model_name"] = [model_names[i] for i in summary_df["model_index"]]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Raw Scores (lower=better)",
            "Normalized Scores",
            "Final Weights",
            "Weight vs Score",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Raw scores
    fig.add_trace(
        go.Scatter(
            x=summary_df["model_index"],
            y=summary_df["raw_score"],
            mode="markers",
            name="Raw Score",
            text=summary_df["model_name"],
            hovertemplate="%{text}<br>Score: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Normalized scores
    fig.add_trace(
        go.Scatter(
            x=summary_df["model_index"],
            y=summary_df["normalized_score"],
            mode="markers",
            name="Normalized Score",
            text=summary_df["model_name"],
            hovertemplate="%{text}<br>Norm Score: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Final weights
    fig.add_trace(
        go.Bar(
            x=summary_df["model_index"],
            y=summary_df["weight"],
            name="Weight",
            text=summary_df["model_name"],
            hovertemplate="%{text}<br>Weight: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Weight vs Score scatter
    fig.add_trace(
        go.Scatter(
            x=summary_df["normalized_score"],
            y=summary_df["weight"],
            mode="markers",
            name="Weight vs Score",
            text=summary_df["model_name"],
            hovertemplate="%{text}<br>Score: %{x:.4f}<br>Weight: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(title=title, height=800, showlegend=False)

    return fig


def calculate_ensemble_uncertainty(
    basin_stats_array: List[Dict],
    weights: np.ndarray = None,
    basin_list: List[str] = None,
    years: List[int] = None,
) -> Dict:
    """
    Calculate ensemble uncertainty metrics

    Parameters
    ----------
    basin_stats_array : list
        List of model statistics dictionaries
    weights : np.ndarray, optional
        Model weights (if None, uses equal weights)
    basin_list : list, optional
        List of basin names
    years : list, optional
        List of years

    Returns
    -------
    dict
        Uncertainty metrics by year and basin
    """
    if weights is None:
        weights = np.ones(len(basin_stats_array)) / len(basin_stats_array)

    # Normalize weights
    weights = weights / np.sum(weights)

    if basin_list is None:
        first_model = basin_stats_array[0] if basin_stats_array else {}
        first_year = list(first_model.keys())[0] if first_model else None
        basin_list = list(first_model.get(first_year, {}).keys()) if first_year else []

    if years is None:
        first_model = basin_stats_array[0] if basin_stats_array else {}
        years = [int(year) for year in first_model.keys()] if first_model else []

    uncertainty_metrics = {}

    for year in years:
        uncertainty_metrics[year] = {}
        for basin in basin_list:
            values = []
            valid_weights = []

            for i, model_stats in enumerate(basin_stats_array):
                try:
                    value = model_stats.get(year, {}).get(basin, {}).get("mean", np.nan)
                    if not np.isnan(value):
                        values.append(value)
                        valid_weights.append(weights[i])
                except Exception as e:
                    logger.warning(f"Error processing model stats: {e}")
                    continue

            if len(values) > 1:
                values = np.array(values)
                valid_weights = np.array(valid_weights)
                valid_weights = valid_weights / np.sum(valid_weights)

                # Weighted statistics
                weighted_mean = np.average(values, weights=valid_weights)
                weighted_var = np.average(
                    (values - weighted_mean) ** 2, weights=valid_weights
                )

                # Ensemble spread
                spread = np.max(values) - np.min(values)

                # Agreement metrics
                agreement = 1.0 - (np.sqrt(weighted_var) / (spread + 1e-10))

                uncertainty_metrics[year][basin] = {
                    "ensemble_mean": weighted_mean,
                    "ensemble_std": np.sqrt(weighted_var),
                    "ensemble_spread": spread,
                    "agreement": max(0, agreement),
                    "n_models": len(values),
                    "effective_n": 1 / np.sum(valid_weights**2),
                }
            else:
                uncertainty_metrics[year][basin] = {
                    "ensemble_mean": np.nan,
                    "ensemble_std": np.nan,
                    "ensemble_spread": np.nan,
                    "agreement": np.nan,
                    "n_models": len(values),
                    "effective_n": 0,
                }

    return uncertainty_metrics
