import logging

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create module-specific logger
logger = logging.getLogger(__name__)


def rotated_data_year(observations, year):
    """
    Get the residual data for a specific year.

    Parameters
    ----------
    year : int
        The year for which to retrieve the residual data.

    Returns
    -------
    xarray.DataArray
        The residual data for the specified year.
    """
    if "year" not in observations.ds.dims:
        raise ValueError("The dataset does not contain a 'year' dimension.")

    data = observations.ds.sel(time=year).residual
    return np.rot90(data, k=0)


def create_interactive_residual_plot(
    residuals, year, basin_id=None, preserve_zoom=False, existing_fig=None
):
    """
    Create an interactive Plotly heatmap for residual data

    Parameters:
    -----------
    year : int, year to plot
    basin_id : int or None, basin ID to plot (None for all data)
    preserve_zoom : bool, whether to preserve zoom state from existing figure
    existing_fig : go.FigureWidget or None, existing figure to update instead of creating new one
    """
    try:
        x_coords = residuals.ds.x.values
        y_coords = residuals.ds.y.values

        # Get data for the specified basin/year
        data = residuals.get_basin_data(year, basin_id)

        if basin_id is not None:
            basin_name = residuals.ds.basin_names.values[basin_id]
            title = (
                f"Residual Ice Mask for {year} - Basin {basin_name} (ID: {basin_id})"
            )
        else:
            title = f"Residual Ice Mask for {year} - All Basins"

        if existing_fig is not None and preserve_zoom:
            with existing_fig.batch_update():
                # Update the heatmap data
                existing_fig.data[0].z = data
                existing_fig.data[0].x = x_coords
                existing_fig.data[0].y = y_coords
                # Update title
                existing_fig.layout.title.text = title
            return existing_fig
        else:
            FigureClass = go.FigureWidget if preserve_zoom else go.Figure
            fig = FigureClass(
                data=go.Heatmap(
                    z=data,
                    x=x_coords,
                    y=y_coords,
                    colorscale="RdBu",  # Red-Blue colorscale, good for residuals
                    zmid=0,
                    colorbar=dict(
                        title="Ice Mask Residual",
                    ),
                    hoverongaps=False,
                    hovertemplate="X: %{x:.0f}<br>Y: %{y:.0f}<br>Residual: %{z:.4f}<extra></extra>",
                )
            )

            fig.update_layout(
                title=title,
                xaxis_title="X Coordinate (m)",
                yaxis_title="Y Coordinate (m)",
                width=800,
                height=600,
                font=dict(size=12),
            )

            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            return fig

    except Exception as e:
        print(f"Error creating plot for year {year}: {e}")
        print(f"Available years in dataset: {residuals.ds.time.values}")
        return None


def create_basin_statistics_plot(basin_stats, year):
    """
    Create an interactive bar plot for basin statistics
    """
    if year not in basin_stats:
        print(f"No data for year {year}")
        return None

    basins = []
    means = []
    stds = []
    counts = []
    rms_values = []

    for basin_name, stats in basin_stats[year].items():
        if stats["count"] > 0:
            basins.append(basin_name)
            means.append(stats["mean"])
            stds.append(stats["std"])
            counts.append(stats["count"])
            rms_values.append(stats["rms"])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Mean Residual", "Standard Deviation", "Data Count", "RMS"),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    fig.add_trace(
        go.Bar(x=basins, y=means, name="Mean", marker_color="lightblue"), row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=basins, y=stds, name="Std Dev", marker_color="lightcoral"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(x=basins, y=counts, name="Count", marker_color="lightgreen"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(x=basins, y=rms_values, name="RMS", marker_color="lightyellow"),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=f"Basin Statistics for Year {year}",
        showlegend=False,
        height=600,
        width=900,
    )

    return fig


def interactive_plot(residuals, basin_stats, year, basin_id, plot_type):
    """Interactive plotting function with multiple plot types"""

    if plot_type == "residual":
        basin_id = None if basin_id == -1 else int(basin_id)
        fig = create_interactive_residual_plot(residuals, year, basin_id)
        if fig:
            fig.show()

    elif plot_type == "stats":
        fig = create_basin_statistics_plot(basin_stats, year)
        if fig:
            fig.show()


def create_zoom_preserving_residual_widget(
    residuals, basin_stats, available_years=None, available_basins=None
):
    """
    Create an interactive widget for residual plots that preserves zoom when switching years

    Parameters:
    -----------
    residuals : object with get_basin_data method and ds attribute
    basin_stats : dict, basin statistics
    available_years : list, optional, list of available years
    available_basins : list, optional, list of available basins

    Returns:
    --------
    ipywidgets.VBox : Interactive widget with zoom preservation
    """
    import ipywidgets as widgets
    from IPython.display import display

    # Get available years and basins from data if not provided
    if available_years is None:
        available_years = sorted([y for y in residuals.ds.time.values])
    if available_basins is None:
        available_basins = [None] + list(range(len(residuals.ds.basin_names.values)))

    year_slider = widgets.IntSlider(
        value=available_years[0] if available_years else 2007,
        min=min(available_years) if available_years else 2007,
        max=max(available_years) if available_years else 2015,
        step=1,
        description="Year:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="400px"),
    )

    basin_dropdown = widgets.Dropdown(
        options=[("All Basins", None)]
        + [
            (f"Basin {i}", i)
            for i in range(len(residuals.ds.basin_names.values))
            if i is not None
        ],
        value=None,
        description="Basin:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="200px"),
    )

    plot_type_dropdown = widgets.Dropdown(
        options=[("Residual Map", "residual"), ("Basin Statistics", "stats")],
        value="residual",
        description="Plot Type:",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="200px"),
    )

    output = widgets.Output()

    # Store the current figure to enable zoom preservation
    current_fig = {"fig": None, "plot_type": None}

    def update_plot(change=None):
        with output:
            year = year_slider.value
            basin_id = basin_dropdown.value
            plot_type = plot_type_dropdown.value

            # Check if we're switching plot types or if this is the first plot
            switching_plot_type = current_fig["plot_type"] != plot_type

            if plot_type == "residual":
                if switching_plot_type or current_fig["fig"] is None:
                    output.clear_output(wait=True)
                    fig = create_interactive_residual_plot(
                        residuals, year, basin_id, preserve_zoom=True
                    )
                    current_fig["fig"] = fig
                    current_fig["plot_type"] = plot_type
                    display(fig)
                else:
                    # Update existing figure while preserving zoom
                    fig = create_interactive_residual_plot(
                        residuals,
                        year,
                        basin_id,
                        preserve_zoom=True,
                        existing_fig=current_fig["fig"],
                    )
                    # No need to display again, the figure is already shown and updated in place

            elif plot_type == "stats":
                # For stats plots, we create new plots since they're bar charts
                output.clear_output(wait=True)
                fig = create_basin_statistics_plot(basin_stats, year)
                current_fig["fig"] = fig
                current_fig["plot_type"] = plot_type
                if fig:
                    display(fig)

    # Initial plot
    update_plot()

    # Connect widgets to update function
    year_slider.observe(update_plot, names="value")
    basin_dropdown.observe(update_plot, names="value")
    plot_type_dropdown.observe(update_plot, names="value")

    controls = widgets.HBox([year_slider, basin_dropdown, plot_type_dropdown])

    # Add informational text
    info_text = widgets.HTML(value="<b>Interactive Residual Plot</b><br>")

    return widgets.VBox([info_text, controls, output])


# Also create a combined view function
def create_combined_dashboard(residuals, basin_stats, year):
    """Combined dashboard with both residual map and statistics"""

    residual_fig = create_interactive_residual_plot(residuals, year, basin_id=None)

    stats_fig = create_basin_statistics_plot(basin_stats, year)

    if residual_fig:
        residual_fig.show()

    if stats_fig:
        stats_fig.show()


def create_example_zoom_preserving_dashboard(residuals, basin_stats):
    """
    Example of how to create a dashboard with zoom preservation

    Usage:
    ------

    widget = create_example_zoom_preserving_dashboard(residuals, basin_stats)

    # Display it
    display(widget)
    """
    return create_zoom_preserving_residual_widget(residuals, basin_stats)


def create_time_series_plot(basin_stats, statistic="mean", colors=None):
    """
    Interactive time series plot showing how statistics change over time

    Parameters:
    -----------
    basin_stats : dict
        Basin statistics dictionary
    statistic : str
        Which statistic to plot ('mean', 'std', 'rms', 'count', etc.)
    colors : dict
        Color mapping for basins
    """

    # Default colors if not provided
    if colors is None:
        colors = {
            "CW": "blue",
            "NE": "red",
            "SE": "green",
            "SW": "orange",
            "NO": "purple",
            "NW": "brown",
        }

    # Prepare data
    years = sorted(basin_stats.keys())
    basin_names = list(basin_stats[years[0]].keys())

    fig = go.Figure()

    # Add a trace for each basin
    for basin_name in basin_names:
        values = []
        for year in years:
            if (
                basin_name in basin_stats[year]
                and basin_stats[year][basin_name]["count"] > 0
            ):
                values.append(basin_stats[year][basin_name][statistic])
            else:
                values.append(None)

        # Use consistent color
        color = colors.get(basin_name, "gray")

        fig.add_trace(
            go.Scatter(
                x=years,
                y=values,
                mode="lines+markers",
                name=basin_name,
                line=dict(width=2, color=color),
                marker=dict(size=8, color=color),
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>{statistic.title()}: %{{y:.6f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Time Series of {statistic.title()} by Basin",
        xaxis_title="Year",
        yaxis_title=statistic.title(),
        hovermode="x unified",
        width=900,
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    return fig


def create_relative_time_series_plot(basin_stats, statistic="mean", colors=None):
    """
    Create a relative time series plot where each basin's first value is Relative to 0

    Parameters:
    -----------
    basin_stats : dict
        Basin statistics dictionary
    statistic : str
        Which statistic to plot ('mean', 'std', 'rms', 'count', etc.)
    colors : dict
        Color mapping for basins
    """

    # Default colors if not provided
    if colors is None:
        colors = {
            "CW": "blue",
            "NE": "red",
            "SE": "green",
            "SW": "orange",
            "NO": "purple",
            "NW": "brown",
        }

    # Prepare data
    years = sorted(basin_stats.keys())
    basin_names = list(basin_stats[years[0]].keys())

    fig = go.Figure()

    for basin_name in basin_names:
        values = []
        first_value = None

        for year in years:
            if (
                basin_name in basin_stats[year]
                and basin_stats[year][basin_name]["count"] > 0
            ):
                val = basin_stats[year][basin_name][statistic]
                values.append(val)
                if first_value is None:
                    first_value = val
            else:
                values.append(None)

        if first_value is not None:
            relative_values = []
            for val in values:
                if val is not None:
                    relative_values.append(val - first_value)
                else:
                    relative_values.append(None)
        else:
            relative_values = values

        # Use consistent color
        color = colors.get(basin_name, "gray")

        fig.add_trace(
            go.Scatter(
                x=years,
                y=relative_values,
                mode="lines+markers",
                name=basin_name,
                line=dict(width=2, color=color),
                marker=dict(size=8, color=color),
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Relative {statistic.title()}: %{{y:.6f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Relative Time Series of {statistic.title()} by Basin (Relative to First Value)",
        xaxis_title="Year",
        yaxis_title=f"Relative {statistic.title()} (Change from First Year)",
        hovermode="x unified",
        width=900,
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    return fig


def create_observations_model_residual_grid(
    basin_stats, obs_stats, mod_stats, statistic="mean", colors=None
):
    """
    Create a 2x3 grid of plots showing observations, Model, and Residual statistics
    First row: Standard plots
    Second row: Relative plots

    Parameters:
    -----------
    basin_stats : dict
        Basin statistics dictionary
    statistic : str
        Which statistic to plot ('mean', 'std', 'rms', 'count', etc.)
    colors : dict
        Color mapping for basins
    """

    # Default colors if not provided
    if colors is None:
        colors = {
            "CW": "blue",
            "NE": "red",
            "SE": "green",
            "SW": "orange",
            "NO": "purple",
            "NW": "brown",
        }

    # Prepare data
    years = sorted(basin_stats.keys())
    basin_names = list(basin_stats[years[0]].keys())

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[
            "Observations (Standard)",
            "Model (Standard)",
            "Residual (Standard)",
            "Observations (Relative)",
            "Model (Relative)",
            "Residual (Relative)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # For each basin, add traces to all subplots
    for basin_name in basin_names:
        # Get consistent color for this basin
        color = colors.get(basin_name, "gray")

        # Collect observations, Model, and Residual data
        observations_values = []
        model_values = []
        residual_values = []

        for year in years:
            if (
                basin_name in basin_stats[year]
                and basin_stats[year][basin_name]["count"] > 0
            ):
                # Use the actual residual statistic
                residual_val = basin_stats[year][basin_name][statistic]
                residual_values.append(residual_val)

                # Use actual observations data if available
                if (
                    obs_stats
                    and year in obs_stats
                    and basin_name in obs_stats[year]
                    and obs_stats[year][basin_name]["count"] > 0
                ):
                    observations_val = obs_stats[year][basin_name][statistic]
                    observations_values.append(observations_val)
                else:
                    observations_values.append(None)

                # Use actual model data if available
                if (
                    mod_stats
                    and year in mod_stats
                    and basin_name in mod_stats[year]
                    and mod_stats[year][basin_name]["count"] > 0
                ):
                    model_val = mod_stats[year][basin_name][statistic]
                    model_values.append(model_val)
                else:
                    model_values.append(None)
            else:
                observations_values.append(None)
                model_values.append(None)
                residual_values.append(None)

        # Calculate Relative values (relative to first value)
        def normalize_values(values):
            first_val = next((v for v in values if v is not None), None)
            if first_val is not None:
                return [(v - first_val) if v is not None else None for v in values]
            return values

        observations_norm = normalize_values(observations_values)
        model_norm = normalize_values(model_values)
        residual_norm = normalize_values(residual_values)

        # Add traces for standard plots (row 1)
        fig.add_trace(
            go.Scatter(
                x=years,
                y=observations_values,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>observations: %{{y:.6f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=model_values,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Model: %{{y:.6f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=residual_values,
                mode="lines+markers",
                name=basin_name,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=True,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Residual: %{{y:.6f}}<extra></extra>",
            ),
            row=1,
            col=3,
        )

        # Add traces for Relative plots (row 2)
        fig.add_trace(
            go.Scatter(
                x=years,
                y=observations_norm,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>observations (Relative): %{{y:.6f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=model_norm,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Model (Relative): %{{y:.6f}}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=years,
                y=residual_norm,
                mode="lines+markers",
                name=f"{basin_name}" if basin_name == basin_names[0] else None,
                line=dict(width=2, color=color),
                marker=dict(size=6, color=color),
                showlegend=False,
                hovertemplate=f"Year: %{{x}}<br>Basin: {basin_name}<br>Residual (Relative): %{{y:.6f}}<extra></extra>",
            ),
            row=2,
            col=3,
        )

    fig.update_layout(
        title=f"observations, Model, and Residual {statistic.title()} Statistics by Basin",
        height=800,
        width=1400,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.01),
    )

    # Update axes labels
    for i in range(1, 4):
        fig.update_xaxes(title_text="Year", row=2, col=i)

    fig.update_yaxes(title_text=f"observations {statistic.title()}", row=1, col=1)
    fig.update_yaxes(title_text=f"Model {statistic.title()}", row=1, col=2)
    fig.update_yaxes(title_text=f"Residual {statistic.title()}", row=1, col=3)

    fig.update_yaxes(
        title_text=f"observations {statistic.title()} (Relative)", row=2, col=1
    )
    fig.update_yaxes(title_text=f"Model {statistic.title()} (Relative)", row=2, col=2)
    fig.update_yaxes(
        title_text=f"Residual {statistic.title()} (Relative)", row=2, col=3
    )

    return fig


def create_correlation_matrix(basin_stats, year):
    """
    Create a correlation matrix heatmap for different statistics

    Parameters:
    -----------
    basin_stats : dict
        Basin statistics dictionary
    year : int
        Year to create correlation matrix for
    """
    if year not in basin_stats:
        return None

    # Prepare data for correlation
    stats_data = []
    basin_names = []

    for basin_name, stats in basin_stats[year].items():
        if stats["count"] > 0:
            basin_names.append(basin_name)
            stats_data.append(
                [
                    stats["mean"],
                    stats["std"],
                    stats["rms"],
                    stats["winsorized_mean"],
                    stats["outlier_weighted_mean"],
                    stats["sum"],
                ]
            )

    if len(stats_data) < 2:
        return None

    df = pd.DataFrame(
        stats_data,
        index=basin_names,
        columns=[
            "Mean",
            "Std Dev",
            "RMS",
            "Winsorized Mean",
            "Outlier Weighted Mean",
            "Sum",
        ],
    )

    # Calculate correlation matrix
    corr_matrix = df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.3f}",
            textfont={"size": 12},
            hoverongaps=False,
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Statistics Correlation Matrix for Year {year}", width=600, height=600
    )

    return fig


# =============================================================================
# ENSEMBLE PLOTTING UTILITIES
# =============================================================================


def create_ensemble_time_series_plot(
    basin_stats_array,
    model_names,
    statistic="mean",
    basin_list=None,
    observations_stats=None,
    important_models=None,
    use_figure_widget=False,
):
    """
    Interactive time series plot for ensemble basin statistics with improved visual hierarchy.

    Parameters
    ----------
    basin_stats_array : list
    model_names : list
    statistic : str, default 'mean'
        Statistic to plot ('mean', 'std', 'rms', 'sum', 'winsorized_mean', 'outlier_weighted_mean')
    basin_list : list, optional
    observations_stats : dict, optional
    important_models : list, optional
        List of 3-5 model names that should be highlighted with bolder lines
    use_figure_widget : bool, default False
        If True, return FigureWidget instead of Figure for interactive updates

    Returns
    -------
    plotly.graph_objects.Figure or plotly.graph_objects.FigureWidget
        Interactive time series plot with improved visual hierarchy
    """
    if not basin_stats_array:
        raise ValueError("basin_stats_array cannot be empty")

    # Extract years from first model
    years = sorted(list(basin_stats_array[0].keys()))

    # Get all basin names if not specified
    if basin_list is None:
        basin_list = list(basin_stats_array[0][years[0]].keys())

    # Auto-select top 3-5 important models if not specified
    if important_models is None:
        # Use first 3-5 models as important by default
        important_models = model_names[: min(5, len(model_names))]

    fig = make_subplots(
        rows=len(basin_list),
        cols=1,
        subplot_titles=[f"Basin {basin}" for basin in basin_list],
        shared_xaxes=True,
        vertical_spacing=0.04,  # Increased spacing for better readability
    )

    # Convert to FigureWidget if requested
    if use_figure_widget:
        fig = go.FigureWidget(fig)

    # Define colors for different models with better contrast
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ]

    # Plot each basin
    for basin_idx, basin_name in enumerate(basin_list):
        row = basin_idx + 1

        # Plot each model
        for model_idx, (basin_stats, model_name) in enumerate(
            zip(basin_stats_array, model_names)
        ):
            values = []
            for year in years:
                if year in basin_stats and basin_name in basin_stats[year]:
                    values.append(basin_stats[year][basin_name][statistic])
                else:
                    values.append(np.nan)

            # Determine visual properties based on importance
            is_important = model_name in important_models
            line_width = 4 if is_important else 1
            opacity = 1.0 if is_important else 0.2
            marker_size = 8 if is_important else 5

            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=values,
                    mode="lines+markers",
                    name=f"{model_name}" if basin_idx == 0 else None,
                    line=dict(color=colors[model_idx % len(colors)], width=line_width),
                    marker=dict(size=marker_size),
                    opacity=opacity,
                    showlegend=(basin_idx == 0),  # Only show legend for first subplot
                    hovertemplate=f"<b>{model_name}</b><br>"
                    + f"Basin: {basin_name}<br>"
                    + "Year: %{x}<br>"
                    + f"{statistic.replace('_', ' ').title()}: %{{y:.6f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    # Add observations data if provided (always highlighted as most important)
    if observations_stats is not None:
        for basin_idx, basin_name in enumerate(basin_list):
            row = basin_idx + 1

            if basin_name in observations_stats:
                observations_values = []
                for year in years:
                    if year in observations_stats[basin_name]:
                        observations_values.append(
                            observations_stats[basin_name][year][statistic]
                        )
                    else:
                        observations_values.append(np.nan)

                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=observations_values,
                        mode="lines+markers",
                        name="observations (Reference)" if basin_idx == 0 else None,
                        line=dict(color="black", width=4),
                        marker=dict(size=10, color="black"),
                        showlegend=(basin_idx == 0),
                        hovertemplate="<b>observations (Reference)</b><br>"
                        + f"Basin: {basin_name}<br>"
                        + "Year: %{x}<br>"
                        + f"{statistic.replace('_', ' ').title()}: %{{y:.6f}}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )

    # Improved layout with legend positioned outside plot area
    fig.update_layout(
        title=dict(
            text=f"Ensemble Time Series: {statistic.replace('_', ' ').title()} by Basin",
            font=dict(size=16, family="Arial, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        height=400
        * len(basin_list),  # Reduced height per subplot for better proportions
        width=1400,  # Increased width to accommodate external legend
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,  # Position legend outside plot area
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=150, t=80, b=80),  # Increased margins for better spacing
    )

    # Update x-axis labels with improved formatting
    fig.update_xaxes(
        title_text="Year",
        row=len(basin_list),
        col=1,
        title_font=dict(size=14, family="Arial, sans-serif"),
        tickfont=dict(size=12),
        showgrid=True,
        gridcolor="rgba(128,128,128,0.2)",
    )

    # Update y-axis labels with improved formatting
    for i in range(len(basin_list)):
        fig.update_yaxes(
            title_text=f"{statistic.replace('_', ' ').title()}",
            row=i + 1,
            col=1,
            title_font=dict(size=12, family="Arial, sans-serif"),
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        )

    return fig


def update_ensemble_time_series_plot(
    existing_fig,
    basin_stats_array,
    model_names,
    statistic="mean",
    basin_list=None,
    observations_stats=None,
    important_models=None,
):
    """
    Update an existing FigureWidget with new data while preserving zoom

    Parameters
    ----------
    existing_fig : plotly.graph_objects.FigureWidget
        Existing figure to update
    basin_stats_array : list
    model_names : list
    statistic : str, default 'mean'
    basin_list : list, optional
    observations_stats : dict, optional
    important_models : list, optional

    Returns
    -------
    plotly.graph_objects.FigureWidget
        Updated figure widget
    """
    if not basin_stats_array:
        return existing_fig

    # Extract years from first model
    years = sorted(list(basin_stats_array[0].keys()))

    # Get all basin names if not specified
    if basin_list is None:
        basin_list = list(basin_stats_array[0][years[0]].keys())

    # Auto-select top 3-5 important models if not specified
    if important_models is None:
        important_models = model_names[: min(5, len(model_names))]

    with existing_fig.batch_update():
        # Update title
        existing_fig.layout.title.text = (
            f"Ensemble Time Series: {statistic.replace('_', ' ').title()} by Basin"
        )

        # Update each trace
        trace_idx = 0

        # Update model traces for each basin
        for basin_idx, basin_name in enumerate(basin_list):
            for model_idx, (basin_stats, model_name) in enumerate(
                zip(basin_stats_array, model_names)
            ):
                values = []
                for year in years:
                    if year in basin_stats and basin_name in basin_stats[year]:
                        values.append(basin_stats[year][basin_name][statistic])
                    else:
                        values.append(np.nan)

                # Determine visual properties based on importance
                is_important = model_name in important_models
                line_width = 4 if is_important else 1
                opacity = 1.0 if is_important else 0.2
                marker_size = 8 if is_important else 5

                if trace_idx < len(existing_fig.data):
                    # Update existing trace
                    existing_fig.data[trace_idx].x = years
                    existing_fig.data[trace_idx].y = values
                    existing_fig.data[trace_idx].line.width = line_width
                    existing_fig.data[trace_idx].opacity = opacity
                    existing_fig.data[trace_idx].marker.size = marker_size
                    existing_fig.data[trace_idx].hovertemplate = (
                        f"<b>{model_name}</b><br>"
                        + f"Basin: {basin_name}<br>"
                        + "Year: %{x}<br>"
                        + f"{statistic.replace('_', ' ').title()}: %{{y:.6f}}<extra></extra>"
                    )

                trace_idx += 1

        # Update observations traces if provided
        if observations_stats is not None:
            for basin_idx, basin_name in enumerate(basin_list):
                if basin_name in observations_stats:
                    observations_values = []
                    for year in years:
                        if year in observations_stats[basin_name]:
                            observations_values.append(
                                observations_stats[basin_name][year][statistic]
                            )
                        else:
                            observations_values.append(np.nan)

                    if trace_idx < len(existing_fig.data):
                        # Update existing observations trace
                        existing_fig.data[trace_idx].x = years
                        existing_fig.data[trace_idx].y = observations_values
                        existing_fig.data[trace_idx].hovertemplate = (
                            "<b>observations (Reference)</b><br>"
                            + f"Basin: {basin_name}<br>"
                            + "Year: %{x}<br>"
                            + f"{statistic.replace('_', ' ').title()}: %{{y:.6f}}<extra></extra>"
                        )

                    trace_idx += 1

        # Update y-axis labels for each subplot
        for i in range(len(basin_list)):
            y_axis_name = f"yaxis{i + 1 if i > 0 else ''}"
            if hasattr(existing_fig.layout, y_axis_name):
                getattr(existing_fig.layout, y_axis_name)["title"]["text"] = (
                    f"{statistic.replace('_', ' ').title()}"
                )

    return existing_fig


def create_ensemble_statistics_summary(basin_stats_array, model_names, basin_list=None):
    """
    Create summary statistics across ensemble members for each basin and year.

    Parameters
    ----------
    basin_stats_array : list
        List of basin statistics dictionaries from multiple models
    model_names : list
        List of model names corresponding to each basin_stats
    basin_list : list, optional
        List of basin names to analyze. If None, analyzes all basins

    Returns
    -------
    dict
        Dictionary with ensemble statistics (mean, std, min, max) for each basin, year, and statistic
    """
    if not basin_stats_array:
        raise ValueError("basin_stats_array cannot be empty")

    # Extract years from first model
    years = sorted(list(basin_stats_array[0].keys()))

    # Get all basin names if not specified
    if basin_list is None:
        basin_list = list(basin_stats_array[0][years[0]].keys())

    # Available statistics
    available_stats = [
        "mean",
        "std",
        "rms",
        "sum",
        "winsorized_mean",
        "outlier_weighted_mean",
    ]

    ensemble_summary = {}

    for basin_name in basin_list:
        ensemble_summary[basin_name] = {}

        for year in years:
            ensemble_summary[basin_name][year] = {}

            for stat in available_stats:
                values = []
                for basin_stats in basin_stats_array:
                    if year in basin_stats and basin_name in basin_stats[year]:
                        values.append(basin_stats[year][basin_name][stat])

                if values:
                    ensemble_summary[basin_name][year][stat] = {
                        "ensemble_mean": np.mean(values),
                        "ensemble_std": np.std(values),
                        "ensemble_min": np.min(values),
                        "ensemble_max": np.max(values),
                        "ensemble_count": len(values),
                    }

    return ensemble_summary


def create_interactive_ensemble_plot(
    basin_stats_array,
    model_names,
    basin_list=None,
    observations_stats=None,
    important_models=None,
):
    """
    Create an interactive ensemble plot with dropdown for statistic selection and improved visual hierarchy.

    Parameters
    ----------
    basin_stats_array : list
    model_names : list
    basin_list : list, optional
    observations_stats : dict, optional
        observations statistics dictionary with structure {basin_name: {year: {stat_name: value}}}
        If provided, will be plotted as a bold black line
    important_models : list, optional
        List of 3-5 model names that should be highlighted with bolder lines.
        If None, automatically selects first 3-5 models as important.

    Returns
    -------
    ipywidgets.VBox
        Interactive widget with dropdown and plot featuring visual hierarchy
    """

    # Available statistics
    stats_options = [
        ("Mean", "mean"),
        ("Standard Deviation", "std"),
        ("RMS", "rms"),
        ("Sum", "sum"),
        ("Winsorized Mean", "winsorized_mean"),
        ("Outlier Weighted Mean", "outlier_weighted_mean"),
    ]

    # Auto-select important models if not provided
    if important_models is None:
        important_models = model_names[: min(5, len(model_names))]

    stat_dropdown = widgets.Dropdown(
        options=stats_options,
        value="mean",
        description="Statistic:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="300px"),
    )

    important_models_widget = widgets.SelectMultiple(
        options=model_names,
        value=important_models,
        style={"description_width": "100px"},
        layout=widgets.Layout(width="400px", height="150px"),
    )

    output = widgets.Output()

    # Store current figure to enable zoom preservation
    current_figure = {"fig": None}

    def update_plot(change=None):
        with output:
            current_important = list(important_models_widget.value)

            if current_figure["fig"] is None:
                output.clear_output(wait=True)
                fig = create_ensemble_time_series_plot(
                    basin_stats_array,
                    model_names,
                    statistic=stat_dropdown.value,
                    basin_list=basin_list,
                    observations_stats=observations_stats,
                    important_models=current_important,
                    use_figure_widget=True,  # We'll add this parameter
                )
                current_figure["fig"] = fig
                from IPython.display import display

                display(fig)
            else:
                # to support updating existing FigureWidget
                fig = update_ensemble_time_series_plot(
                    current_figure["fig"],
                    basin_stats_array,
                    model_names,
                    statistic=stat_dropdown.value,
                    basin_list=basin_list,
                    observations_stats=observations_stats,
                    important_models=current_important,
                )

    # Initial plot
    with output:
        fig = create_ensemble_time_series_plot(
            basin_stats_array,
            model_names,
            statistic="mean",
            basin_list=basin_list,
            observations_stats=observations_stats,
            important_models=important_models,
            use_figure_widget=True,
        )
        current_figure["fig"] = fig
        from IPython.display import display

        display(fig)

    # Connect widgets to update function
    stat_dropdown.observe(update_plot, names="value")
    important_models_widget.observe(update_plot, names="value")

    controls = widgets.HBox([stat_dropdown, important_models_widget])

    info_text = widgets.HTML(
        value="<b>Visual Hierarchy:</b> Selected models are highlighted with bolder lines and full opacity. "
        "Other models are shown with thinner lines and reduced opacity for better focus."
    )

    return widgets.VBox([info_text, controls, output])


def calculate_ensemble_accuracy_metrics(
    basin_stats_array, model_names, statistic="mean"
):
    """
    Calculate accuracy metrics for ensemble members.

    Parameters
    ----------
    basin_stats_array : list
        List of basin statistics for each model
    model_names : list
        List of model names
    statistic : str
        Statistic to analyze

    Returns
    -------
    dict
        Dictionary with accuracy metrics for each model
    """
    accuracy_metrics = {}

    # Get all years and basins from first model
    if not basin_stats_array:
        return accuracy_metrics

    first_model = basin_stats_array[0]
    years = sorted(first_model.keys())
    basins = list(first_model[years[0]].keys())

    for basin in basins:
        accuracy_metrics[basin] = {}

        for year in years:
            # Collect all model values for this year/basin/statistic
            values = []
            for stats in basin_stats_array:
                if year in stats and basin in stats[year]:
                    value = stats[year][basin].get(statistic, np.nan)
                    if not np.isnan(value):
                        values.append(value)

            if len(values) > 1:
                ensemble_mean = np.mean(values)
                ensemble_std = np.std(values)

                # Calculate accuracy metrics for each model
                model_accuracy = []
                for i, stats in enumerate(basin_stats_array):
                    if year in stats and basin in stats[year]:
                        value = stats[year][basin].get(statistic, np.nan)
                        if not np.isnan(value):
                            # Distance from ensemble mean (closer to 0 is better for residuals)
                            distance_from_mean = abs(value - ensemble_mean)
                            # Relative distance (within 1 std = good, within 2 std = acceptable)
                            normalized_distance = distance_from_mean / (
                                ensemble_std + 1e-10
                            )
                            # Absolute value (closer to 0 is better for residuals)
                            abs_value = abs(value)

                            model_accuracy.append(
                                {
                                    "model_name": model_names[i],
                                    "value": value,
                                    "distance_from_mean": distance_from_mean,
                                    "normalized_distance": normalized_distance,
                                    "abs_value": abs_value,
                                    "percentile_rank": 0,  # Will calculate below
                                }
                            )

                # Calculate percentile ranks
                if model_accuracy:
                    abs_values = [m["abs_value"] for m in model_accuracy]
                    for i, model_metric in enumerate(model_accuracy):
                        # Rank based on absolute value (lower is better)
                        rank = sum(
                            1 for v in abs_values if v < model_metric["abs_value"]
                        )
                        percentile = (rank / len(abs_values)) * 100
                        model_accuracy[i]["percentile_rank"] = percentile

                accuracy_metrics[basin][year] = {
                    "ensemble_mean": ensemble_mean,
                    "ensemble_std": ensemble_std,
                    "model_accuracy": model_accuracy,
                    "num_models": len(values),
                }

    return accuracy_metrics


def create_advanced_ensemble_comparison_plot(
    basin_stats_array, model_names, basin_list=None, start_year=2007, end_year=2015
):
    """
    Create an advanced interactive ensemble comparison plot with multiple dropdowns and accuracy metrics.

    Parameters
    ----------
    basin_stats_array : list
        List of basin statistics for each model
    model_names : list
        List of model names
    basin_list : list, optional
        List of basins to include
    start_year : int
        Start year for analysis
    end_year : int
        End year for analysis

    Returns
    -------
    ipywidgets.VBox
        Interactive widget with multiple controls and advanced plotting
    """

    # Available statistics
    stats_options = [
        ("Mean", "mean"),
        ("Standard Deviation", "std"),
        ("RMS", "rms"),
        ("Sum", "sum"),
        ("Winsorized Mean", "winsorized_mean"),
        ("Outlier Weighted Mean", "outlier_weighted_mean"),
    ]

    # Get years and basins from data
    if basin_stats_array:
        first_model = basin_stats_array[0]
        available_years = sorted([y for y in first_model.keys() if isinstance(y, int)])
        if available_years:
            available_basins = list(first_model[available_years[0]].keys())
        else:
            available_basins = []
    else:
        available_years = list(range(start_year, end_year + 1))
        available_basins = basin_list or ["NW"]

    # Filter by basin_list if provided
    if basin_list:
        available_basins = [b for b in available_basins if b in basin_list]

    # Ensure vars
    if not available_years:
        available_years = list(range(start_year, end_year + 1))

    if not available_basins:
        available_basins = ["NW"]

    stat_dropdown = widgets.Dropdown(
        options=stats_options,
        value="mean",
        description="Statistic:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="200px"),
    )

    year_dropdown = widgets.IntSlider(
        min=min(available_years),
        max=max(available_years),
        value=available_years[0],
        description="Year:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="300px"),
    )

    basin_dropdown = widgets.Dropdown(
        options=available_basins,
        value=available_basins[0],
        description="Basin:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="150px"),
    )

    compare_mode_dropdown = widgets.Dropdown(
        options=[
            ("Individual vs Ensemble Mean", "individual"),
            ("Time Series Comparison", "timeseries"),
            ("Accuracy Ranking", "ranking"),
        ],
        value="individual",
        description="Mode:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="250px"),
    )

    model_select = widgets.SelectMultiple(
        options=model_names,
        value=[model_names[0]] if model_names else [],
        description="Models:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="300px", height="120px"),
    )

    show_ensemble_stats = widgets.Checkbox(
        value=True,
        description="Show Ensemble Statistics",
        style={"description_width": "initial"},
    )

    show_accuracy_metrics = widgets.Checkbox(
        value=True,
        description="Show Accuracy Metrics",
        style={"description_width": "initial"},
    )

    output = widgets.Output()

    def update_plot(change=None):
        with output:
            output.clear_output(wait=True)

            selected_stat = stat_dropdown.value
            selected_year = year_dropdown.value
            selected_basin = basin_dropdown.value
            selected_models = list(model_select.value)
            mode = compare_mode_dropdown.value

            if not selected_models and mode != "ranking":
                print("Please select at least one model to compare.")
                return

            # Calculate accuracy metrics
            accuracy_metrics = calculate_ensemble_accuracy_metrics(
                basin_stats_array, model_names, selected_stat
            )

            if mode == "individual":
                fig = create_individual_vs_ensemble_plot(
                    basin_stats_array,
                    model_names,
                    selected_models,
                    selected_stat,
                    selected_year,
                    selected_basin,
                    accuracy_metrics,
                    show_ensemble_stats.value,
                    show_accuracy_metrics.value,
                )
            elif mode == "timeseries":
                fig = create_model_timeseries_comparison_plot(
                    basin_stats_array,
                    model_names,
                    selected_models,
                    selected_stat,
                    selected_basin,
                    available_years,
                    accuracy_metrics,
                    show_ensemble_stats.value,
                )
            elif mode == "ranking":
                fig = create_accuracy_ranking_plot(
                    accuracy_metrics,
                    selected_stat,
                    selected_year,
                    selected_basin,
                    model_names,
                )

            if fig:
                fig.show()

    # Initial plot
    update_plot()

    # Connect widgets
    for widget in [
        stat_dropdown,
        year_dropdown,
        basin_dropdown,
        compare_mode_dropdown,
        model_select,
        show_ensemble_stats,
        show_accuracy_metrics,
    ]:
        widget.observe(update_plot, names="value")

    row1 = widgets.HBox(
        [stat_dropdown, year_dropdown, basin_dropdown, compare_mode_dropdown]
    )
    row2 = widgets.HBox([show_ensemble_stats, show_accuracy_metrics])
    row3 = widgets.HBox([model_select])

    controls = widgets.VBox([row1, row2, row3])

    title = widgets.HTML("<h3>Advanced Ensemble Comparison Analysis</h3>")
    info = widgets.HTML(
        "<b>Tips:</b> Select models to compare against ensemble mean. "
        "Accuracy metrics show how close each model is to zero (better for residuals). "
        "Ranking mode shows all models sorted by performance."
    )

    return widgets.VBox([title, info, controls, output])


def create_individual_vs_ensemble_plot(
    basin_stats_array,
    model_names,
    selected_models,
    statistic,
    year,
    basin,
    accuracy_metrics,
    show_ensemble_stats,
    show_accuracy_metrics,
):
    """Create individual model vs ensemble mean comparison plot."""

    fig = go.Figure()

    # Get ensemble statistics for the year/basin
    if basin in accuracy_metrics and year in accuracy_metrics[basin]:
        ensemble_data = accuracy_metrics[basin][year]
        ensemble_mean = ensemble_data["ensemble_mean"]
        ensemble_std = ensemble_data["ensemble_std"]
        model_accuracy = ensemble_data["model_accuracy"]

        model_acc_dict = {m["model_name"]: m for m in model_accuracy}

        # Plot ensemble mean line
        if show_ensemble_stats:
            fig.add_hline(
                y=ensemble_mean,
                line_dash="dash",
                line_color="black",
                line_width=3,
                annotation_text=f"Ensemble Mean: {ensemble_mean:.3f}",
                annotation_position="top left",
            )

            # Add ensemble standard deviation bands
            fig.add_hrect(
                y0=ensemble_mean - ensemble_std,
                y1=ensemble_mean + ensemble_std,
                fillcolor="lightgray",
                opacity=0.3,
                line_width=0,
                annotation_text=f"±1 STD ({ensemble_std:.3f})",
            )

        # Plot selected models
        for model_name in selected_models:
            if model_name in model_acc_dict:
                model_data = model_acc_dict[model_name]
                value = model_data["value"]

                # Color based on performance (closer to 0 is better)
                abs_val = abs(value)
                if abs_val < abs(ensemble_mean):
                    color = "green"
                elif model_data["normalized_distance"] < 1:
                    color = "orange"
                else:
                    color = "red"

                # Add bar for model
                fig.add_trace(
                    go.Bar(
                        x=[model_name],
                        y=[value],
                        name=model_name,
                        marker_color=color,
                        text=f"{value:.3f}"
                        if not show_accuracy_metrics
                        else f"{value:.3f}<br>Rank: {model_data['percentile_rank']:.1f}%<br>Dist: {model_data['distance_from_mean']:.3f}",
                        textposition="outside",
                        showlegend=False,
                    )
                )

    fig.update_layout(
        title=f"Model Comparison - {statistic.title()} for {basin} Basin in {year}",
        xaxis_title="Models",
        yaxis_title=f"{statistic.title()} Value",
        height=600,
        showlegend=False,
    )

    # Add zero line (ideal for residuals)
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="blue",
        annotation_text="Ideal (Zero)",
        annotation_position="bottom right",
    )

    return fig


def create_model_timeseries_comparison_plot(
    basin_stats_array,
    model_names,
    selected_models,
    statistic,
    basin,
    years,
    accuracy_metrics,
    show_ensemble_stats,
):
    """Create time series comparison plot for selected models."""

    fig = go.Figure()

    # Plot ensemble mean and std bands
    if show_ensemble_stats and basin in accuracy_metrics:
        ensemble_means = []
        ensemble_stds = []
        valid_years = []

        for year in years:
            if year in accuracy_metrics[basin]:
                ensemble_means.append(accuracy_metrics[basin][year]["ensemble_mean"])
                ensemble_stds.append(accuracy_metrics[basin][year]["ensemble_std"])
                valid_years.append(year)

        if ensemble_means:
            # Plot ensemble mean
            fig.add_trace(
                go.Scatter(
                    x=valid_years,
                    y=ensemble_means,
                    mode="lines+markers",
                    name="Ensemble Mean",
                    line=dict(color="black", width=3, dash="dash"),
                    marker=dict(size=8),
                )
            )

            # Add standard deviation bands
            upper_band = [m + s for m, s in zip(ensemble_means, ensemble_stds)]
            lower_band = [m - s for m, s in zip(ensemble_means, ensemble_stds)]

            fig.add_trace(
                go.Scatter(
                    x=valid_years + valid_years[::-1],
                    y=upper_band + lower_band[::-1],
                    fill="toself",
                    fillcolor="rgba(128,128,128,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="±1 STD",
                    showlegend=True,
                )
            )

    # Plot selected models
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    for i, model_name in enumerate(selected_models):
        model_idx = model_names.index(model_name) if model_name in model_names else -1
        if model_idx >= 0 and model_idx < len(basin_stats_array):
            model_stats = basin_stats_array[model_idx]

            x_vals = []
            y_vals = []

            for year in years:
                if year in model_stats and basin in model_stats[year]:
                    value = model_stats[year][basin].get(statistic, np.nan)
                    if not np.isnan(value):
                        x_vals.append(year)
                        y_vals.append(value)

            if x_vals:
                color = colors[i % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        name=model_name,
                        line=dict(color=color, width=2),
                        marker=dict(size=6),
                    )
                )

    # Add zero line
    fig.add_hline(
        y=0, line_dash="dot", line_color="blue", annotation_text="Ideal (Zero)"
    )

    fig.update_layout(
        title=f"Time Series Comparison - {statistic.title()} for {basin} Basin",
        xaxis_title="Year",
        yaxis_title=f"{statistic.title()} Value",
        height=600,
        legend=dict(x=1.05, y=1),
    )

    return fig


def create_accuracy_ranking_plot(accuracy_metrics, statistic, year, basin, model_names):
    """Create accuracy ranking plot for all models."""

    if basin not in accuracy_metrics or year not in accuracy_metrics[basin]:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected basin/year",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    model_accuracy = accuracy_metrics[basin][year]["model_accuracy"]

    # Sort by absolute value (best performance first)
    sorted_models = sorted(model_accuracy, key=lambda x: x["abs_value"])

    fig = go.Figure()

    colors = []
    for model_data in sorted_models:
        if model_data["percentile_rank"] <= 25:  # Top quartile
            colors.append("green")
        elif model_data["percentile_rank"] <= 50:  # Second quartile
            colors.append("lightgreen")
        elif model_data["percentile_rank"] <= 75:  # Third quartile
            colors.append("orange")
        else:  # Bottom quartile
            colors.append("red")

    fig.add_trace(
        go.Bar(
            x=[m["model_name"] for m in sorted_models],
            y=[m["value"] for m in sorted_models],
            marker_color=colors,
            text=[
                f"{m['value']:.3f}<br>Rank: {m['percentile_rank']:.1f}%"
                for m in sorted_models
            ],
            textposition="outside",
            showlegend=False,
        )
    )

    # Calculate median and mean of all model values
    all_values = [m["value"] for m in sorted_models]
    median_value = np.median(all_values)
    mean_value = np.mean(all_values)

    # Add zero line
    fig.add_hline(
        y=0, line_dash="dot", line_color="blue", annotation_text="Ideal (Zero)"
    )

    # Add median line
    fig.add_hline(
        y=median_value,
        line_dash="dash",
        line_color="purple",
        line_width=1,
        opacity=0.6,
        annotation_text=f"Median: {median_value:.3f}",
        annotation_position="top right",
    )

    # Add mean line
    fig.add_hline(
        y=mean_value,
        line_dash="dashdot",
        line_color="darkred",
        line_width=1,
        opacity=0.6,
        annotation_text=f"Mean: {mean_value:.3f}",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Model Accuracy Ranking - {statistic.title()} for {basin} Basin in {year}",
        xaxis_title="Models (Sorted by Performance)",
        yaxis_title=f"{statistic.title()} Value",
        height=700,
        xaxis_tickangle=-45,
    )

    return fig


def create_box_whiskers_plot(
    residuals, statistic="residual", colors=None, basin_stats=None
):
    """
    Create a box and whiskers plot aggregating across basins for each time period.

    Parameters:
    -----------
    residuals : Residual object
        Contains the residual data with basin_stats
    statistic : str
        The statistic to plot ('residual', 'mean', 'std', etc.)
    colors : dict, optional
        Color mapping for basins

    Returns:
    --------
    plotly figure
    """
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Get the basin statistics
    basin_stats_data = (
        residuals.basin_stats if hasattr(residuals, "basin_stats") else basin_stats
    )

    if basin_stats_data is None:
        raise ValueError(
            "No basin statistics available. Either pass basin_stats parameter or ensure residuals object has basin_stats attribute."
        )

    plot_data = []

    # Get all years and basins
    years = sorted(basin_stats_data.keys())
    all_basins = set()
    for year_data in basin_stats_data.values():
        all_basins.update(year_data.keys())

    for year in years:
        year_data = basin_stats_data[year]
        values = []
        basin_names = []

        for basin_name, basin_data in year_data.items():
            if statistic in basin_data:
                values.append(basin_data[statistic])
                basin_names.append(basin_name)

        if values:  # Only add if we have data
            plot_data.append(
                {"year": year, "values": values, "basin_names": basin_names}
            )

    fig = go.Figure()

    for data in plot_data:
        fig.add_trace(
            go.Box(
                y=data["values"],
                x=[str(data["year"])] * len(data["values"]),
                name=str(data["year"]),
                boxpoints=False,  # Don't show points on box plot
                fillcolor="lightblue",
                line=dict(color="darkblue"),
                opacity=0.7,
                showlegend=False,
            )
        )

    # Now add individual colored points as scatter plots
    if colors:
        # Group data by basin for colored scatter points
        basin_data = {}
        for data in plot_data:
            for i, basin_name in enumerate(data["basin_names"]):
                if basin_name not in basin_data:
                    basin_data[basin_name] = {"x": [], "y": [], "years": []}
                basin_data[basin_name]["x"].append(str(data["year"]))
                basin_data[basin_name]["y"].append(data["values"][i])
                basin_data[basin_name]["years"].append(data["year"])

        # Add scatter trace for each basin
        for basin_name, basin_info in basin_data.items():
            basin_color = colors.get(basin_name, "gray")
            fig.add_trace(
                go.Scatter(
                    x=basin_info["x"],
                    y=basin_info["y"],
                    mode="markers",
                    name=f"Basin {basin_name}",
                    marker=dict(
                        color=basin_color,
                        size=8,
                        line=dict(width=1, color="white"),
                        opacity=0.8,
                    ),
                    hovertemplate=f"<b>Basin {basin_name}</b><br>"
                    + "Year: %{x}<br>"
                    + f"{statistic.title()}: %{{y:.4f}}<br>"
                    + "<extra></extra>",
                    legendgroup="basins",
                )
            )
    else:
        # If no colors provided, add simple scatter points
        for data in plot_data:
            fig.add_trace(
                go.Scatter(
                    x=[str(data["year"])] * len(data["values"]),
                    y=data["values"],
                    mode="markers",
                    name="Data Points",
                    marker=dict(color="blue", size=6),
                    text=data["basin_names"],
                    hovertemplate="<b>%{text}</b><br>"
                    + "Year: %{x}<br>"
                    + f"{statistic.title()}: %{{y:.4f}}<br>"
                    + "<extra></extra>",
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"Distribution of {statistic.title()} Across Basins by Year<br><sub>Individual points colored by basin</sub>",
        xaxis_title="Year",
        yaxis_title=f"{statistic.title()} Value",
        width=1200,
        height=700,
        template="plotly_white",
        legend=dict(title="Basins", yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    return fig


def create_interactive_box_whiskers_plot(residuals, colors=None, basin_stats=None):
    """
    Create an interactive box and whiskers plot with dropdown for different statistics.
    """
    from ipywidgets import Dropdown, VBox, interact

    statistic_options = [
        ("Mean", "mean"),
        ("Standard Deviation", "std"),
        ("RMS", "rms"),
        ("Winsorized Mean", "winsorized_mean"),
        ("Outlier Weighted Mean", "outlier_weighted_mean"),
        ("Sum", "sum"),
    ]

    statistic_dropdown = Dropdown(
        options=statistic_options, value="mean", description="Statistic:"
    )

    def interactive_box_plot(statistic):
        """Interactive box plotting function"""
        fig = create_box_whiskers_plot(
            residuals, statistic, colors=colors, basin_stats=basin_stats
        )
        if fig:
            fig.show()

    interact(interactive_box_plot, statistic=statistic_dropdown)


def create_ensemble_box_whiskers_plot(
    basin_stats_array,
    model_names,
    statistic="mean",
    basin_name="NW",
    start_year=2007,
    end_year=2015,
):
    """
    Create a box and whiskers plot aggregating across ensemble members for each year.

    Parameters:
    -----------
    basin_stats_array : list
        List of basin statistics for each model
    model_names : list
        List of model names
    statistic : str
        The statistic to plot ('mean', 'std', 'rms', etc.)
    basin_name : str
        The basin to analyze

    Returns:
    --------
    plotly figure
    """
    import numpy as np
    import plotly.graph_objects as go

    # Extract data for all models and years
    plot_data = []

    # Get all available years from first model
    if basin_stats_array and len(basin_stats_array) > 0:
        first_model = basin_stats_array[0]
        years = sorted(first_model.keys())
    else:
        years = list(range(start_year, end_year + 1))

    # For each year, collect values across all ensemble members
    for year in years:
        year_values = []
        model_names_for_year = []

        for i, model_stats in enumerate(basin_stats_array):
            if year in model_stats and basin_name in model_stats[year]:
                basin_data = model_stats[year][basin_name]
                if statistic in basin_data and not np.isnan(basin_data[statistic]):
                    year_values.append(basin_data[statistic])
                    model_names_for_year.append(
                        model_names[i] if i < len(model_names) else f"Model_{i}"
                    )

        if year_values:  # Only add if we have data
            plot_data.append(
                {
                    "year": year,
                    "values": year_values,
                    "model_names": model_names_for_year,
                }
            )

    # Create the box plot
    fig = go.Figure()

    # Add box plots for each year (without individual points first)
    for data in plot_data:
        fig.add_trace(
            go.Box(
                y=data["values"],
                x=[str(data["year"])] * len(data["values"]),
                name=str(data["year"]),
                boxpoints=False,  # Don't show points on box plot
                fillcolor="lightgreen",
                line=dict(color="darkgreen"),
                opacity=0.7,
                showlegend=False,
            )
        )

    # Add individual points as scatter plots colored by model performance
    for data in plot_data:
        # Color points based on their value (relative to median)
        median_val = np.median(data["values"])
        point_colors = []
        for val in data["values"]:
            if val < median_val - np.std(data["values"]):
                point_colors.append("red")  # Below average
            elif val > median_val + np.std(data["values"]):
                point_colors.append("blue")  # Above average
            else:
                point_colors.append("orange")  # Average

        fig.add_trace(
            go.Scatter(
                x=[str(data["year"])] * len(data["values"]),
                y=data["values"],
                mode="markers",
                name=f"Models {data['year']}",
                marker=dict(
                    color=point_colors,
                    size=6,
                    line=dict(width=1, color="white"),
                    opacity=0.8,
                ),
                text=data["model_names"],
                hovertemplate="<b>%{text}</b><br>"
                + "Year: %{x}<br>"
                + f"{statistic.title()}: %{{y:.4f}}<br>"
                + "<extra></extra>",
                showlegend=False,
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Ensemble Distribution of {statistic.title()} for Basin {basin_name} by Year<br><sub>Each point represents one ensemble member</sub>",
        xaxis_title="Year",
        yaxis_title=f"{statistic.title()} Value",
        width=1200,
        height=700,
        template="plotly_white",
    )

    return fig


def create_interactive_ensemble_box_whiskers_plot(
    basin_stats_array, model_names, basin_list
):
    """
    Create an interactive ensemble box and whiskers plot with dropdowns for basin and statistic.
    """
    from ipywidgets import Dropdown, VBox, interact

    # Create dropdown for statistics
    statistic_options = [
        ("Mean", "mean"),
        ("Standard Deviation", "std"),
        ("RMS", "rms"),
        ("Count", "count"),
        ("Sum", "sum"),
    ]

    # Create dropdown for basins
    basin_options = [(basin, basin) for basin in basin_list]

    statistic_dropdown = Dropdown(
        options=statistic_options, value="mean", description="Statistic:"
    )

    basin_dropdown = Dropdown(
        options=basin_options,
        value=basin_list[0] if basin_list else "NW",
        description="Basin:",
    )

    def interactive_ensemble_box_plot(statistic, basin):
        """Interactive ensemble box plotting function"""
        fig = create_ensemble_box_whiskers_plot(
            basin_stats_array, model_names, statistic, basin
        )
        if fig:
            fig.show()

    interact(
        interactive_ensemble_box_plot,
        statistic=statistic_dropdown,
        basin=basin_dropdown,
    )
