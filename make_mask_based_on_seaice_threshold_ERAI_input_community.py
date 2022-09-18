
# Create a mask to apply to water levels based on sea ice concentration threshold of a close-to-shore grid cell.

# Save the masked water level in yearly files.

import numpy as np
from netCDF4 import Dataset
import pandas as pd
import cftime
from datetime import datetime
import math
import matplotlib.pyplot as plt
import cmocean # Colormaps for sea ice from Kristen M. Thyng, Chad A. Greene, Robert D. Hetland, Heather M. Zimmerle, and Steven F. DiMarco (2016). True colors of oceanography: Guidelines for effective and accurate colormap selection. Oceanography, 29(3), 10. doi:10.5670/oceanog.2016.66
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap, addcyclic
import ERA_interim_read_with_wave_and_sst

#study_site = 'Mamontovy_Khayata'
#study_site = 'prudhoe_bay'
#study_site = 'Drew_Point'
study_site = 'veslobogen'

print(study_site)

##### path names

basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'

# put where your ERA-Interim data is stored here.
ncfile_path = '/permarisk/data/ERA_Data/ERAint_Arctic/'


# Put the lat lon of the ERAI cell offshore, used to create the mask applied to the water levels in this script.

if study_site == 'prudhoe_bay':
	npy_path = basepath + 'input_data/storm_surge/prudhoe_bay/'
	lat_site = 70.402 # also can load the first two indices (but hardcoded here are the same lat/lon values) of input_array_to_produce_modelled_water_levels_prudhoe_bay_era.npy
	lon_site = -148.519
	#year_range = np.arange(2007,2008) # up to (not including) year end
	year_range = np.arange(2007,2017) # up to (not including) year end

if study_site == 'Drew_Point':
	npy_path = basepath + 'input_data/storm_surge/Drew_Point/'
	lat_site = np.load(npy_path + 'lat_offshore_site_ERAI_Drew_Point.npy')
	lon_site = np.load(npy_path + 'lon_offshore_site_ERAI_Drew_Point.npy')
	#year_range = np.arange(2007,2008) # up to (not including) year end
	year_range = np.arange(2007,2017) # up to (not including) year end

if study_site == 'Mamontovy_Khayata':
	npy_path = basepath + 'input_data/storm_surge/Mamontovy_Khayata/'
	lat_site = np.load(npy_path + 'lat_offshore_site_ERAI_mamontovy_hayata.npy') # [degrees North] this is the point offshore
	lon_site = np.load(npy_path + 'lon_offshore_site_ERAI_mamontovy_hayata.npy')
	#year_range = np.arange(1995,1996) # up to year end
	year_range = np.arange(1995,2019) # up to year end

if study_site == 'veslobogen':
	npy_path = basepath + 'input_data/storm_surge/veslobogen/'
	lat_site = np.load(npy_path + 'lat_offshore_site_ERAI_veslobogen.npy') # [degrees North] this is the point offshore
	lon_site = np.load(npy_path + 'lon_offshore_site_ERAI_veslobogen.npy')
	year_range = np.arange(2014,2018) # up to year end


def geo_idx(dd, dd_array):
	"""
	Search for nearest decimal degree in an array of decimal degrees and return the index.
	np.argmin returns the indices of minimum value along an axis.
	so subtract dd (decimal degree of selected location) from all values in dd_array, take absolute value and find index of minimum.
	"""
	geo_idx = (np.abs(dd_array - dd)).argmin()
	return geo_idx

for year in year_range:
	print(year)
	start_date = str(year) + '-01-01'  # make month padded with preceding zero (e.g. july is 07 and not 7) , same with daynumber e.g. 2007-07-01 is july 1st).
	end_date = str(year) + '-12-31'

	# start def
	start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
	end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

	# sea ice concentration input file
	sicn_ifile = ncfile_path + 'ci_with_mask/ERAintforcing_Arctic_ci_' + str(year) + '0101_' + str(year) + '1231_with_mask.nc'
	print(sicn_ifile)

	# read sea ice concentration from ifile
	fh = Dataset(sicn_ifile, mode = 'r')
	lons = fh.variables['longitude'][:] # [degrees north]
	lats = fh.variables['latitude'][:] # [degrees east]
	timestamp = fh.variables['time'] # hours since 1900-01-01 00:00:00.0, calendar = gregorian
	sicn = fh.variables['siconc'][:,:,:] # fraction of ice cover within the grid cell shape is: (time, lat, lon)
	dates = cftime.num2pydate(timestamp[:],timestamp.units,calendar=timestamp.calendar) # convert to python datetime
	fh.close()

	# find indices that are closest to the selected lat/lon
	lat_idx = geo_idx(lat_site,lats)
	lon_idx = geo_idx(lon_site,lons)

	# check function is working correctly
	lat_site_ERAI = lats[lat_idx]
	lon_site_ERAI = lons[lon_idx]

	sicn_offshore_all_timesteps = sicn[:,lat_idx, lon_idx]

	# you have to match the timesteps of the unmasked water levels with the timesteps of the era interim sicn, and then apply the mask.

	# water levels calculated use the ERAI winds from the same offshore lat/lon
	npy_path = basepath + 'input_data/storm_surge/'+ study_site +'/'
	unmasked_water_levels = pd.read_pickle(npy_path + 'ERAI_forced_water_levels/ERAI_forced_WL_' + str(year) + '.pkl')

	# find indices in sicn that match the first and last datetimes of the calcd water level file
	first_wl_datetime = unmasked_water_levels.index[0]
	last_wl_datetime = unmasked_water_levels.index[-1]

	first_wl_datetime_ind = int(np.where(first_wl_datetime == dates)[0])
	last_wl_datetime_ind = int(np.where(last_wl_datetime == dates)[0])

	sicn_offshore_timesubset = sicn_offshore_all_timesteps[first_wl_datetime_ind:last_wl_datetime_ind+1] # +1 bc not inclusive

	# create bool mask where average sicn in that domain crosses a user-input threshold

	sicn_threshold = 0.15 # era-interim gives fraction as a unit, not percent, of sea ice cover per grid cell.

	# find open water timesteps
	timesteps_bt = np.where(sicn_offshore_timesubset<sicn_threshold)

	# create empty array of nans as placeholders for full timestep mask
	mask_for_wl = np.zeros(sicn_offshore_timesubset.shape[0])
	mask_for_wl[:] = np.nan

	# make the bool value of 1 for timesteps of open water (e.g. below sicn threshold) for this year (current ncfile) and this will be your mask
	mask_for_wl[timesteps_bt] = 1

	# apply mask to calcd water levels in the npy path and resave it to use for a now seasonal forcing of the erosion model.
	# check that mask to be applied to dataset is the same shape as the dataset
	'''
	>>> mask_for_wl.shape
	(977,)
	>>> unmasked_water_levels.shape
	(977,) .. good its the same
	'''

	# multiply mask and calcd water levels to get new array that will be applied to erosion model.
	water_levels_masked = mask_for_wl*unmasked_water_levels

	# save the new water levels you will use to force the erosion model
	water_levels_masked.to_pickle(npy_path + 'ERAI_forced_water_levels_masked/water_levels_' + study_site + '_masked_' + str(year) + '.pkl')

	# load the dataframe of the other variables and mask where sicn is above the threshold.
	df, lat_site_ERAI, lon_site_ERAI = ERA_interim_read_with_wave_and_sst.read_ERAInterim(lat_site, lon_site, start_datetime, end_datetime)

	## mask where above sicn threshold
	# add new column to df so you can multiply

	df['mask_for_wl'] = mask_for_wl

	df = df.drop(columns=['timestamps', 'times_x', 'times_y'])
	df_masked = df.loc[:,'u_wind10m_site':'swh_site'].multiply(df['mask_for_wl'],axis='index')
	# append the columns that were not mutliplied 
	df_masked['sicn'] = sicn_offshore_timesubset

	# save the dataframe by year
	df_masked.to_pickle(npy_path + 'ERAI_forcing_variables_masked/ERAI_forcings_' + study_site + '_masked_' + str(year) + '.pkl')





















































































































