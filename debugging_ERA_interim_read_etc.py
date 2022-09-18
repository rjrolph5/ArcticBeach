

import pandas as pd
import numpy as np
from netCDF4 import Dataset
import cftime


wave_period_ifile = '/permarisk/data/ERA_Data/ERAint_Arctic/pp1d/ERAintforcing_Arctic_pp1d_20140101_20141231.nc'

fh = Dataset(wave_period_ifile, mode= 'r')
wave_period = fh.variables['pp1d'][:]
timestamp_wave_period = fh.variables['time'] # from ncdump: time:units = "hours since 1900-01-01 00:00:00.0"
dates_wave_period = cftime.num2pydate(timestamp_wave_period[:],timestamp_wave_period.units,calendar=timestamp_wave_period.calendar) # convert to python datetime
fh.close()

lat_idx=18
lon_idx=261

wave_period_site = wave_period[:, lat_idx, lon_idx]

wave_period_site_df = pd.DataFrame({'timestamps': dates_wave_period, 'wave_period_site': wave_period_site.data}, columns = ['timestamps', 'wave_period_site'])
# set index of the dataframe to be the timestamps
wave_period_site_df = wave_period_site_df.set_index(pd.DatetimeIndex(wave_period_site_df['timestamps']))
# make anything less than 0 nan
#wave_period_site_df_no_interp = wave_period_site_df.mask(wave_period_site_df < 0) # defaults to nan where condition is met if mask value not specified
wave_period_site_df = wave_period_site_df.mask(wave_period_site_df.wave_period_site < 0) # defaults to nan where condition is met if mask value not specified. in this case it puts nan >
print(wave_period_site_df)
wave_period_site_df_check = wave_period_site_df.copy()

wave_period_site_df = wave_period_site_df.resample('3H').ffill()

# add the timestamps to the df
first_date_waves_interpd = wave_period_site_df.index[0]
last_date_waves_interpd = wave_period_site_df.index[-1]
wave_period_site_df['timestamps'] = pd.date_range(first_date_waves_interpd,last_date_waves_interpd,freq='3H')

'''
import matplotlib.pyplot as plt
plt.plot(wave_period_site_df_no_interp.wave_period_site,'o') # 730 rows
plt.plot(wave_period_site_df.wave_period_site,'*') # 2917 rows
plt.show()
# this interactive plotting check on interpolation function checks out, ok to use the interpolated data.
'''
# should be in one dataframe
ws_wd_sst_waveperiod_df = pd.merge_asof(df_ws_wd_sst, wave_period_site_df, on='timestamps')

# set new indices
ws_wd_sst_waveperiod_df = ws_wd_sst_waveperiod_df.set_index(pd.DatetimeIndex(ws_wd_sst_waveperiod_df['timestamps']))



