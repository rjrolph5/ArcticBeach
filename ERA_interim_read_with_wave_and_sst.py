# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:24:35 2020

Reads ERA-interim reanalysis data.  Wave height, wave period and sea surface temperature data can be used to
force the erosion model and wind data can be used to force storm surge module (that is coupled to the erosion
model).

@author: rebecca.rolph@awi.de
"""

import numpy as np
from netCDF4 import Dataset
import pandas as pd
import cftime
from datetime import datetime

def geo_idx(dd, dd_array):
	"""
	Search for nearest decimal degree in an array of decimal degrees and return the index.
	np.argmin returns the indices of minimum value along an axis.
	so subtract dd (decimal degree of selected location) from all values in dd_array, take absolute value and find index of minimum.
	"""
	geo_idx = (np.abs(dd_array - dd)).argmin()
	return geo_idx

def read_ERAInterim(lat_site, lon_site, start_datetime, end_datetime):

	## specify input filenames
	ncfile_path = '/permarisk/data/ERA_Data/ERAint_Arctic/' # for pls11.  Put path of where your ERA data is here.

	# u-component of wind
	uwind_ifile = ncfile_path + 'uwind/ERAintforcing_Arctic_uwind_'+ str(start_datetime.year) +'0101_'+str(start_datetime.year)+'1231.nc'
	fh = Dataset(uwind_ifile, mode = 'r')
	lons = fh.variables['longitude'][:]
	lats= fh.variables['latitude'][:]
	timestamp = fh.variables['time'] # from ncdump: time:units = "hours since 1900-01-01 00:00:00.0"
	dates_wind = cftime.num2pydate(timestamp[:],timestamp.units,calendar=timestamp.calendar) # convert to python datetime
	u_wind10m = fh.variables['u10'][:][:]
	fh.close()

	# v-component of wind
	vwind_ifile = ncfile_path + 'vwind/ERAintforcing_Arctic_vwind_' + str(start_datetime.year) + '0101_' + str(start_datetime.year) + '1231.nc'
	fh = Dataset(vwind_ifile, mode = 'r')
	v_wind10m = fh.variables['v10'][:][:]
	fh.close()

	# find indices that are closest to the selected lat/lon
	lat_idx = geo_idx(lat_site,lats)
	lon_idx = geo_idx(lon_site,lons)

	# check function is working correctly
	lat_site_ERAI = lats[lat_idx]
	lon_site_ERAI = lons[lon_idx]

	# find u-component of wind at selected lat/lon
	u_wind10m_site = u_wind10m[:, lat_idx, lon_idx]

	# find v-component of wind at selected lat/lon
	v_wind10m_site = v_wind10m[:, lat_idx, lon_idx]

	# calculate the wind direction but just for plotting, but it is not used in the storm surge model (see technical note, Grange 2014) .. Calculate the average wind vectors
	wind_direction_ERAI = (np.arctan2(u_wind10m_site,v_wind10m_site)*360/2/np.pi) + 180 # add 180 to convert to oceanographic convention (vector is where wind is blowing to, not from) for input into storm surge model.

	# calculate vector averaged wind speed
	wind_speed_ERAI = (u_wind10m_site**2 + v_wind10m_site**2)**0.5

	## sea surface temperature
	sst_ifile = ncfile_path + 'sst/ERAintforcing_Arctic_sst_' + str(start_datetime.year) + '0101_' + str(start_datetime.year) + '1231.nc'
	fh = Dataset(sst_ifile, mode='r')
	sst = fh.variables['sst'][:][:]
	lons = fh.variables['longitude'][:]
	lats = fh.variables['latitude'][:]
	timestamp = fh.variables['time'] # from ncdump: time:units = "hours since 1900-01-01 00:00:00.0"
	dates_sst = cftime.num2pydate(timestamp[:],timestamp.units,calendar=timestamp.calendar) # convert to python datetime
	fh.close()

	# find sst at the selected lat/lon
	sst_site = sst[:, lat_idx, lon_idx]

	# put wind_speed, wind_dir, and sst in the same dataframe since they all have the same timestamps.
	data = {'timestamps': dates_wind, 'u_wind10m_site': u_wind10m_site, 'v_wind10m_site': v_wind10m_site, 'wind_direction': wind_direction_ERAI, 'wind_speed': wind_speed_ERAI, 'sst_site': sst_site}
	df_ws_wd_sst = pd.DataFrame(data, columns = ['timestamps', 'u_wind10m_site', 'v_wind10m_site','wind_direction', 'wind_speed', 'sst_site'])
	df_ws_wd_sst = df_ws_wd_sst.set_index(pd.DatetimeIndex(df_ws_wd_sst['timestamps']))

	df_ws_wd_sst_check_before_interp = df_ws_wd_sst.copy()
	# drop the column of timestamps now that it is the index
	#df_ws_wd_sst = df_ws_wd_sst.drop(columns=['timestamps'])

	## peak wave period ! only every 12h not every 3h like the rest of the datasets.
	wave_period_ifile = ncfile_path + 'pp1d/ERAintforcing_Arctic_pp1d_' + str(start_datetime.year) + '0101_' + str(start_datetime.year) + '1231.nc'
	fh = Dataset(wave_period_ifile, mode= 'r')
	wave_period = fh.variables['pp1d'][:]
	lons_wave_period = fh.variables['longitude'][:]
	lats_wave_period= fh.variables['latitude'][:]
	timestamp_wave_period = fh.variables['time'] # from ncdump: time:units = "hours since 1900-01-01 00:00:00.0"
	dates_wave_period = cftime.num2pydate(timestamp_wave_period[:],timestamp_wave_period.units,calendar=timestamp_wave_period.calendar) # convert to python datetime
	fh.close()

	# checked that the lon and lats for wave period are the same as they are for winds.. they are. 

	# find peak wave period at the selected lat/lon
	wave_period_site = wave_period[:, lat_idx, lon_idx]

	# interpolate to every 3h so the size of the other variable arrays are the same.
	# make into a dataframe
	wave_period_site_df = pd.DataFrame({'timestamps': dates_wave_period, 'wave_period_site': wave_period_site.data}, columns = ['timestamps', 'wave_period_site'])
	# set index of the dataframe to be the timestamps
	wave_period_site_df = wave_period_site_df.set_index(pd.DatetimeIndex(wave_period_site_df['timestamps']))
	# make anything less than 0 nan
	wave_period_site_df_no_interp = wave_period_site_df.mask(wave_period_site_df < 0) # defaults to nan where condition is met if mask value not specified
	wave_period_site_df = wave_period_site_df.mask(wave_period_site_df < 0) # defaults to nan where condition is met if mask value not specified. in this case it puts nan in negative wave period values (nan in original dataset for wave period is -32767.0).
	wave_period_site_df_check = wave_period_site_df.copy()
	# interp the wave period df to 3h, so it is the same as the wind timesteps.
	i = pd.DatetimeIndex(start=wave_period_site_df.index.min(), end=wave_period_site_df.index.max(), freq = '3H')
	# limit the number of nans to the max number that can be filled by the 12h to 3h gap
	wave_period_site_df = wave_period_site_df.reindex(i).interpolate(method='linear', limit=9)
	#>>> np.where(wave_period_site_df.wave_period_site==np.nanmax(wave_period_site_df.wave_period_site))
	#(array([1808, 2288]),)
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

	# check that the timestamps of the merged dataframe and the indiv. dataframes match the same data.
	#df_ws_wd_sst.loc['1994-08-01':'1994-08-31']
	#ws_wd_sst_waveperiod_df.loc['1994-08-01':'1994-08-31']
	#wave_period_site_df.loc['1994-08-01':'1994-08-31']
	# yes the variables are the same

	# also checked that wave period is nan except during the open water season, so that means the dataframe has been concatenated correctly.

	# peak swh (signifcant wave height). only every 6h not every 3h is the orig input file
	swh_ifile = ncfile_path + 'swh/ERAintforcing_Arctic_swh_' + str(start_datetime.year) + '0101_' + str(start_datetime.year) + '1231.nc' # this ifile contains all lats, unlike the rest of the ifiles which contain only arctic lats.
	fh = Dataset(swh_ifile, mode= 'r')
	swh = fh.variables['swh'][:]
	lats_swh = fh.variables['latitude'][:] # lats are from 90, 89.25, ..., 0, ... -89.25, -90
	#lons_swh = fh.variables['longitude'][:]
	lons_swh_orig = fh.variables['longitude'][:]
	timestamp_swh = fh.variables['time'] # from ncdump: time:units = "hours since 1900-01-01 00:00:00.0"
	dates_swh = cftime.num2pydate(timestamp_swh[:],timestamp_swh.units,calendar=timestamp_swh.calendar) # convert to python datetime
	fh.close()

	# lons for swh are from 0, 0.75, ... 358.5, 359.25, so have to subtract becuase neg lons are west (above 180 on orig frame)
	lons_swh = lons_swh_orig
	lons_swh[int(np.where(lons_swh_orig>180)[0][0]):] = lons_swh_orig[int(np.where(lons_swh_orig>180)[0][0]):]-360.

	# find indices that are closest to the selected lat/lon
	lat_idx_swh = int(np.where(lat_site_ERAI==lats_swh)[0])
	lon_idx_swh = int(np.where(lon_site_ERAI==lons_swh)[0])

	# check function is working correctly
	lat_site_ERAI_swh = lats_swh[lat_idx_swh]
	lon_site_ERAI_swh = lons_swh[lon_idx_swh]

	# find significant wave height at the selected lat/lon
	swh_site = swh[:, lat_idx_swh, lon_idx_swh]

	# interpolate to every 3h so the size of the other variable arrays are the same.
	# make into a dataframe
	swh_site_df = pd.DataFrame({'timestamps': dates_swh, 'swh_site': swh_site.data}, columns = ['timestamps', 'swh_site'])
	# set index of the dataframe to be the timestamps
	swh_site_df = swh_site_df.set_index(pd.DatetimeIndex(swh_site_df['timestamps']))
	# make anything less than 0 nan
	swh_site_df = swh_site_df.mask(swh_site_df < 0) # defaults to nan where condition is met if mask value not specified

	swh_site_check = swh_site_df.copy()

	# interp the wave period df to 3h, so it is the same as the wind timesteps.
	i = pd.DatetimeIndex(start=swh_site_df.index.min(), end=swh_site_df.index.max(), freq = '3H')
	swh_site_df = swh_site_df.reindex(i).interpolate(method='linear', limit=3)

	# add the timestamps to the intpd df
	first_date_swh_interpd = swh_site_df.index[0]
	last_date_swh_interpd = swh_site_df.index[-1]
	# adding a timestamps column with the same daterange index so you can merge df below
	swh_site_df['timestamps'] = pd.date_range(first_date_swh_interpd,last_date_swh_interpd,freq='3H')

	# merge all variables into the final dataframe
	ws_wd_sst_waveperiod_swh_df = pd.merge_asof(ws_wd_sst_waveperiod_df, swh_site_df, on='timestamps')

	# set new indices
	ws_wd_sst_waveperiod_swh_df = ws_wd_sst_waveperiod_swh_df.set_index(pd.DatetimeIndex(ws_wd_sst_waveperiod_swh_df['timestamps']))

	##### read in sea ice concentration
	sicn_ifile = ncfile_path + 'ci_with_mask/ERAintforcing_Arctic_ci_' + str(start_datetime.year) + '0101_' + str(start_datetime.year) + '1231_with_mask.nc'

	# read sea ice concentration from ifile
	fh = Dataset(sicn_ifile, mode = 'r')
	lons_sicn = fh.variables['longitude'][:] # [degrees north]
	lats_sicn = fh.variables['latitude'][:] # [degrees east]
	timestamp = fh.variables['time'] # hours since 1900-01-01 00:00:00.0, calendar = gregorian
	sicn = fh.variables['siconc'][:,:,:] # fraction of ice cover within the grid cell shape is: (time, lat, lon)
	dates = cftime.num2pydate(timestamp[:],timestamp.units,calendar=timestamp.calendar) # convert to python datetime
	fh.close()

	sicn_offshore_all_timesteps = sicn[:,lat_idx, lon_idx]

	## check plot of sicn overlaid with wave and sst parameters
	'''
	import matplotlib.pyplot as plt
	fig,ax1 = plt.subplots()
	#ax1.plot(ws_wd_sst_waveperiod_swh_df.swh_site, color='b', label='swh')
	ax1.plot(ws_wd_sst_waveperiod_swh_df.wave_period_site, color='b', label='wave period')
	ax2 = ax1.twinx()
	ax2.plot(dates, sicn_offshore_all_timesteps, color='r', label='sicn')

	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()

	lines = lines1 + lines2
	labels = labels1 + labels2

	ax1.legend(lines,labels)

	plt.show()
	'''
	# return also actual lat lons being used for the data (does not match lat/lon inputs exactly due to grid resolution)
	return ws_wd_sst_waveperiod_swh_df, lat_site_ERAI, lon_site_ERAI





















