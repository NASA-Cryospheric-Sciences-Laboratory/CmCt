# Shared utilities
import datetime

import cftime
import numpy as np


def check_datarange(time_var, start_date_cftime, end_date_cftime):
    # Get the minimum and maximum values directly from the time variable
    min_time = time_var.values.min()
    max_time = time_var.values.max()

    # Check if the selected start and end dates are within the range
    if (
        min_time <= start_date_cftime <= max_time
        and min_time <= end_date_cftime <= max_time
    ):
        print(
            f"The selected dates {start_date_cftime} and {end_date_cftime} are within the range of the model data."
        )
    else:
        raise ValueError(
            f"Error: The selected dates {start_date_cftime} or {end_date_cftime} are out of range. Model data time range is from {min_time} to {max_time}."
        )


def days_in_year(date):
    if date.calendar == "365_day" or date.calendar == "noleap":
        diy = 365
    elif date.calendar == "366_day" or date.calendar == "all_leap":
        diy = 366
    elif date.calendar == "360_day":
        diy = 360
    else:
        if cftime.is_leap_year(date.year, date.calendar):
            diy = 366
        else:
            diy = 365

    return diy


def checking_ice_area_extent_daterange(
    observations_time: list, model_time: list, start_date: int, end_date: int
):
    """
    Check if the requested date range is available in both datasets.
    Handles different time formats flexibly (cftime, datetime, float years, etc.)

    Parameters
    ----------
    observations_time : list or array
        Time values from observations dataset
    model_time : list or array
        Time values from model dataset
    start_date : int
        Start year for comparison
    end_date : int
        End year for comparison
    """
    try:
        import pandas as pd

        HAS_PANDAS = True
    except ImportError:
        HAS_PANDAS = False

    def extract_year(time_val):
        """Extract year from various time formats"""
        if isinstance(time_val, (int, float)):
            # Assume it's already a year (possibly decimal year)
            return float(time_val)
        elif hasattr(time_val, "year"):
            # cftime, datetime, or similar objects with year attribute
            return float(time_val.year)
        elif isinstance(time_val, np.datetime64):
            # numpy datetime64
            if HAS_PANDAS:
                return float(pd.to_datetime(time_val).year)
            else:
                # Fallback without pandas
                return float(str(time_val)[:4])
        elif isinstance(time_val, str):
            # Try to parse string dates
            if HAS_PANDAS:
                try:
                    dt = pd.to_datetime(time_val)
                    return float(dt.year)
                except:
                    pass
            # If pandas parsing fails or not available, try to extract year from string
            import re

            year_match = re.search(r"\d{4}", str(time_val))
            if year_match:
                return float(year_match.group())

        # Fallback: try to convert to float directly
        try:
            return float(time_val)
        except Exception:
            raise ValueError(
                f"Cannot extract year from time value: {time_val} (type: {type(time_val)})"
            )

    # Convert time arrays to years
    try:
        observations_years = [extract_year(t) for t in observations_time]
        model_years = [extract_year(t) for t in model_time]
    except Exception as e:
        print(f"Error processing time values: {e}")
        print(f"observations time sample: {observations_time[:3] if len(observations_time) > 3 else observations_time}")
        print(
            f"Model time sample: {model_time[:3] if len(model_time) > 3 else model_time}"
        )
        raise

    # Sort and get min/max years
    observations_years.sort()
    observations_time_min = observations_years[0]
    observations_time_max = observations_years[-1]

    model_years.sort()
    model_time_min = model_years[0]
    model_time_max = model_years[-1]

    minimum_time = max(observations_time_min, model_time_min)
    maximum_time = min(observations_time_max, model_time_max)

    if not (minimum_time <= start_date <= end_date <= maximum_time):
        raise ValueError(
            f"Date range {start_date} to {end_date} is outside the available overlapping data range: {minimum_time:.1f} to {maximum_time:.1f}."
        )
    else:
        print(
            f"The selected dates {start_date} to {end_date} are within the overlapping data range."
        )


def standardising_time_var(observations_time):
    """
    Standardize time variables to a common format (list of years as floats).
    Handles cftime, datetime, numpy datetime64, pandas datetime, and other formats.

    Parameters
    ----------
    observations_time : list, array, or time series
        Time values from observations dataset
    model_time : list, array, or time series
        Time values from model dataset

    Returns
    -------
    tuple of lists
        Standardized observations and model time as lists of float years.
    """
    from datetime import datetime

    import numpy as np
    import pandas as pd

    def to_year_list(time_var):
        """Convert various time formats to a list of years (floats)."""
        if time_var is None:
            return []

        time_array = np.asarray(time_var)
        years = []

        for time_val in time_array:
            try:
                if hasattr(time_val, "year"):
                    year = time_val.year
                    if hasattr(time_val, "dayofyr"):
                        # Add fractional year based on day of year
                        days_in_year = (
                            366
                            if hasattr(time_val, "calendar")
                            and "leap" in str(time_val.calendar).lower()
                            else 365
                        )
                        year += (time_val.dayofyr - 1) / days_in_year
                    elif hasattr(time_val, "month") and hasattr(time_val, "day"):
                        import calendar

                        days_in_year = 366 if calendar.isleap(time_val.year) else 365
                        day_of_year = time_val.timetuple().tm_yday
                        year += (day_of_year - 1) / days_in_year
                    years.append(float(year))

                elif isinstance(time_val, np.datetime64):
                    pd_time = pd.to_datetime(time_val)
                    year = pd_time.year
                    day_of_year = pd_time.dayofyear
                    days_in_year = 366 if pd_time.is_leap_year else 365
                    year += (day_of_year - 1) / days_in_year
                    years.append(float(year))

                elif isinstance(time_val, pd.Timestamp):
                    year = time_val.year
                    day_of_year = time_val.dayofyear
                    days_in_year = 366 if time_val.is_leap_year else 365
                    year += (day_of_year - 1) / days_in_year
                    years.append(float(year))

                elif isinstance(time_val, datetime):
                    year = time_val.year
                    day_of_year = time_val.timetuple().tm_yday
                    import calendar

                    days_in_year = 366 if calendar.isleap(time_val.year) else 365
                    year += (day_of_year - 1) / days_in_year
                    years.append(float(year))

                elif isinstance(time_val, str):
                    try:
                        dt = pd.to_datetime(time_val)
                        year = dt.year
                        day_of_year = dt.dayofyear
                        days_in_year = 366 if dt.is_leap_year else 365
                        year += (day_of_year - 1) / days_in_year
                        years.append(float(year))
                    except:
                        year_str = str(time_val)[:4]
                        years.append(float(year_str))

                elif isinstance(time_val, (int, float, np.integer, np.floating)):
                    years.append(float(time_val))

                else:
                    try:
                        dt = pd.to_datetime(time_val)
                        year = dt.year
                        day_of_year = dt.dayofyear
                        days_in_year = 366 if dt.is_leap_year else 365
                        year += (day_of_year - 1) / days_in_year
                        years.append(float(year))
                    except:
                        years.append(float(time_val))
            except Exception as e:
                print(
                    f"Warning: Could not convert time value {time_val} to year. Error: {e}"
                )
                continue

        return years

    # Process both time arrays
    years = to_year_list(observations_time)

    return years
