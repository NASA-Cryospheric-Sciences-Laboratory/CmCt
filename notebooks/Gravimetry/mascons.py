import numpy as np
import xarray as xr
import h5py
from scipy.interpolate import interp1d
import cftime
import datetime

class GSFCmascons:
    def __init__(self, f, lon_wrap='pm180'):
        self.lat_centers = f['/mascon/lat_center'][0][:]
        self.lat_spans = f['/mascon/lat_span'][0][:]
        self.lon_centers = f['/mascon/lon_center'][0][:]
        self.lon_spans = f['/mascon/lon_span'][0][:]
        self.locations = f['/mascon/location'][0][:]
        self.basins = f['/mascon/basin'][0][:]
        self.areas = f['/mascon/area_km2'][0][:]
        self.cmwe = f['/solution/cmwe'][:]
        
        self.days_start = f['/time/ref_days_first'][0][:]
        self.days_middle = f['/time/ref_days_middle'][0][:]
        self.days_end = f['/time/ref_days_last'][0][:]
        self.times_start = self._set_times_as_datetimes(self.days_start)
        self.times_middle = self._set_times_as_datetimes(self.days_middle)
        self.times_end = self._set_times_as_datetimes(self.days_end)

        self.N_mascons = len(self.lat_centers)
        self.N_times = len(self.days_middle)
        self.labels = np.array([i for i in range(self.N_mascons)])
        
        self.reset_lon_bounds(lon_wrap)
        
        self.min_lats = self.lat_centers - self.lat_spans/2
        self.max_lats = self.lat_centers + self.lat_spans/2
        self.max_lats[self.min_lats < -90.0] = -89.5
        self.min_lats[self.min_lats < -90.0] = -90.0
        self.min_lats[self.max_lats > 90.0] = 89.5
        self.max_lats[self.max_lats > 90.0] = 90.0
        self.min_lons = self.lon_centers - self.lon_spans/2
        self.max_lons = self.lon_centers + self.lon_spans/2

    def reset_lon_bounds(self, lon_wrap):
        if lon_wrap == 'pm180':
            self.lon_centers[self.lon_centers > 180] -= 360
        elif lon_wrap == '0to360':
            self.lon_centers[self.lon_centers < 0] += 360

    def _set_times_as_datetimes(self, days):
        return np.datetime64('2002-01-01T00:00:00') + np.array([int(d*24) for d in days], dtype='timedelta64[h]')
    
    def as_dataset(self):
        ds = xr.Dataset({'cmwe': (['label', 'time'], self.cmwe),
                         'lat_centers': ('label', self.lat_centers),
                         'lat_spans': ('label', self.lat_spans),
                         'lon_centers': ('label', self.lon_centers),
                         'lat_spans': ('label', self.lon_spans),
                         'areas': ('label', self.areas),
                         'basins': ('label', self.basins),
                         'locations': ('label', self.locations),
                         'basins': ('label', self.basins),
                         'lats_max': ('label', self.max_lats),
                         'lats_min': ('label', self.min_lats),
                         'lons_max': ('label', self.max_lons),
                         'lons_min': ('label', self.min_lons),
                         'times_start': ('time', self.times_start),
                         'times_end': ('time', self.times_end),
                         'days_start': ('time', self.days_start),
                         'days_middle': ('time', self.days_middle),
                         'days_end': ('time', self.days_end)
                        }, coords={'label': self.labels, 'time': self.times_middle})
        return ds

def load_gsfc_solution(h5_filename, lon_wrap='pm180'):
    with h5py.File(h5_filename, mode='r') as f:
        mascons = GSFCmascons(f, lon_wrap)
    return mascons

def points_to_mascons(mascons, I_, lats, lons, values):
    """
    For each mascon i such that I_[i] == true, this function will set mscn_mean[j] to be the 
    average of all values[z] such that lats[z] and lons[z] is contained within the bounds of 
    mascon[i] as defined by mascons.lat_center[i], mascons.lon_center[i], mascons.lat_span[i], and
    mascons.lon_span[i]. 
    Note that j does not have to be equal to i, and in fact it often is not. mscn_mean[0] will 
    correspond to the first entry of I_ which is true, mscn_mean[1] will correspond to the second 
    entry of I_ which is true, and so on.
    
    Parameters:
    mascons: A mascons object
    I_ : A boolean array with shape (mascons.N_mascons,)
    lats, lons, values: All three are 1D arrays which must all have the same length. value[z] 
    corresponds to the point at lats[z], lons[z]. The function np.nanmean must be able to 
    operate on the entries of the values array

    Returns:
    mscn_mean: A 1D array of length np.sum(I_) and whose entries are the same datatype as the 
    entries of the values array
    """
    N_GIS_mascons = np.sum(I_)   # number of mascons that are on the greenland ice sheet
    d2r = np.pi/180
    
    min_lats = mascons.lat_centers - mascons.lat_spans/2
    max_lats = mascons.lat_centers + mascons.lat_spans/2
    min_lons = mascons.lon_centers - mascons.lon_spans/2
    max_lons = mascons.lon_centers + mascons.lon_spans/2
    
    mscn_mean = np.nan * np.ones(N_GIS_mascons)

    indices = np.array(range(mascons.N_mascons))[I_]
    j = 0
    for i in indices:
    
        if np.min(lats) > max_lats[i]:
            continue
        if np.max(lats) < min_lats[i]:
            continue
        if np.min(lons) > max_lons[i]:
            continue
        if np.max(lons) < min_lons[i]:
            continue
        
        K_ = (lats >= min_lats[i]) & (lats < max_lats[i]) & (lons >= min_lons[i]) & (lons < max_lons[i])
        m = values[K_]
        m_lats = lats[K_]
        
        m_lats = m_lats[~np.isnan(m)]
        m = m[~np.isnan(m)]
        
        if len(m) == 0:
            continue
        if np.sum(~np.isnan(m)) == 0:
            continue

        cos_weight = np.cos(m_lats*d2r)
        mscn_mean[j] = np.nanmean(m) # * cos_weight) / (np.sum(cos_weight) * len(m))
        
        j = j + 1
    return mscn_mean

def YYYY_MM_DD_to_days_since_Jan_0_2002(date_str):
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    return cftime.date2num(date_obj, "days since 2001-12-31")

def calc_mascon_delta_cmwe(mascon, start_date, end_date, I_):
    """
    Given a mascons object, strings indicating start and end dates, and a boolean array, this function
    returns the change in the cmwe variable for the mascons object between the start and end dates.
    Only rows i of the cmwe array for which I_[i] is True will be considered, and the output array will
    have shape (np.sum(I_),). The function will interpolate the cmwe at start_date or end_date if either 
    start_date or end_date respectively do not fall on one of the entries in mascon.days_middle.

    Parameters:
    mascon: A mascons object
    start_date: A string of the form "YYYY-MM-DD". An error will be printed and the function will return 
    None if start_date occurs before mascon.days_start[0]
    end_date: A string of the form "YYYY-MM-DD". An error will be printed and the function will return 
    None if end_date occurs after mascon.days_end[-1]
    I_: A 1D boolean array with the same size as the first axis of mascon.cmwe

    Returns:
    A 1D array with shape (np.sum(I_)). This represents (for the rows of mascon.cmwe indicated by I_, 
    and in the same relative order as they appear in mascon.cmwe) the change in mascon.cmwe between 
    start_date and end_date, estimated with linear interpolation.
    """
    
    t_0 = YYYY_MM_DD_to_days_since_Jan_0_2002(start_date)
    t_1 = YYYY_MM_DD_to_days_since_Jan_0_2002(end_date)

    if t_0 < mascon.days_start[0]:
        print(f"Error: Inputted start_date ({start_date}) is before the earliest date in GRACE MASCONS data ({mascon.times_start[0]})")
        return None
    elif t_1 > mascon.days_end[-1]:
        print(f"Error: Inputted end_date ({end_date}) is after the latest date in GRACE MASCONS data ({mascon.times_end[-1]})")
        return None

    cmwe_I_ = mascon.cmwe[I_]   # Only include those mascons indicated in I_
    
    # Create an interpolation object to estimate the ice thickness on the start date and end date
    interp_obj = interp1d(mascon.days_middle, cmwe_I_, axis = 1, assume_sorted = True, bounds_error = False, fill_value = "extrapolate")
    
    # Note that the only case in which interp1d will need to extrapolate is when days_start[0] <= t_0 < days_middle[0] or 
    # days_middle[-1] < t_1 < days_end[-1], in which case the extrapolation should not introduce too much error
    start_cmwe = interp_obj(t_0)
    end_cmwe = interp_obj(t_1)
    return np.array(end_cmwe - start_cmwe)

    