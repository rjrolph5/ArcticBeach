import numpy as np
import pandas as pd

# Interpolate the masked pickle (.pkl) files produced from make_mask_based_on_seaice_threshold_ERAI_input_community.py

# Loop through years, save forcing as yearly files

# set paths
basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'

#community_name = 'Drew_Point'
community_name = 'veslobogen'

if community_name == 'Drew_Point':
	year_range = np.arange(2007,2017)

if community_name == 'Mamontovy_Khayata':
	year_range = np.arange(1995,2019)

if community_name == 'veslobogen':
	year_range = np.arange(2014,2018)

for year in year_range:
	print(year)
	# set community
	npy_path = basepath + 'input_data/storm_surge/' + community_name + '/'
	## load in the masked arrays (based on sea ice concentration threshold)
	water_level_meters = pd.read_pickle(npy_path + 'ERAI_forced_water_levels_masked/water_levels_' + community_name + '_masked_' + str(year) + '.pkl')
	# interpolate the water_level_meters into hourly timeseries
	#i = pd.DatetimeIndex(start=water_level_meters.index.min(), end=water_level_meters.index.max(), freq = 'H')
	# rewrite the water_level_meters so that it is in hourly timesteps
	#water_level_meters = water_level_meters.reindex(i).interpolate(method='linear', limit=2)
	water_level_meters = water_level_meters.resample('1H').ffill()
	# save the hourly interpolated dataset
	water_level_meters.to_pickle(npy_path + 'ERAI_forced_water_levels_masked/hourly/water_levels_' + community_name + '_masked_' + str(year) + '_hourly.pkl')

	## load the wave height and period, sst. Interpolate the dataframe to hourly
	df = pd.read_pickle(npy_path + 'ERAI_forcing_variables_masked/ERAI_forcings_' + community_name + '_masked_' + str(year) + '.pkl')
	#df_hourly = df.reindex(i).interpolate(method='linear', limit = 2)
	df_hourly = df.resample('1H').ffill()
	# write the hourly df to pkl
	df_hourly.to_pickle(npy_path + 'ERAI_forcing_variables_masked/hourly/ERAI_forcings_' + community_name + '_masked_' + str(year) + '_hourly.pkl')



