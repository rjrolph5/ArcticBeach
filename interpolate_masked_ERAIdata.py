import numpy as np
import pandas as pd

# interpolate the masked pkl files produced from make_mask_based_on_seaice_threshold_ERAI_input_community.py
# these are read in by the script global_variables.py to run the erosion model. 

# loop through years, save forcing as yearly files

# set paths
basepath = '/home/rrolph/erosion_model/'

community_name = 'Drew_Point'
#community_name = 'Mamontovy_Khayata'

if community_name == 'Mamontovy_Khayata':
	year_range = np.arange(1995,2019)

if community_name == 'Drew_Point':
	year_range = np.arange(2007,2017)

'''
# specific to mamontovy khayata (bykovsky)
for year in np.arange(1995,2018+1): # bykovsky is 1995 through 2018
	print(year)
	# set community
	npy_path = basepath + 'input_data/storm_surge/' + community_name + '/'
	## load in the masked arrays (based on sea ice concentration threshold)
	water_level_meters = pd.read_pickle(npy_path + 'ERAI_forced_water_levels_masked/water_levels_Mamontovy_Khayata_masked_' + str(year) + '.pkl')
	# interpolate the water_level_meters into hourly timeseries
	i = pd.DatetimeIndex(start=water_level_meters.index.min(), end=water_level_meters.index.max(), freq = 'H')
	# rewrite the water_level_meters so that it is in hourly timesteps
	water_level_meters = water_level_meters.reindex(i).interpolate(method='linear', limit=2)
	# save the hourly interpolated dataset
	water_level_meters.to_pickle(npy_path + 'ERAI_forced_water_levels_masked/hourly/water_levels_' + community_name + '_masked_' + str(year) + '_hourly.pkl')

	## load the wave height and period, sst. Interpolate the dataframe to hourly
	df = pd.read_pickle(npy_path + 'ERAI_forcing_variables_masked/ERAI_forcings_Mamontovy_Khayata_masked_' + str(year) + '.pkl')
	df_hourly = df.reindex(i).interpolate(method='linear', limit = 2)
	# write the hourly df to pkl
	df_hourly.to_pickle(npy_path + 'ERAI_forcing_variables_masked/hourly/ERAI_forcings_' + community_name + '_masked_' + str(year) + '_hourly.pkl')
'''

# specific to drew point
for year in year_range:
	print(year)
	# set community
	npy_path = basepath + 'input_data/storm_surge/' + community_name + '/'
	## load in the masked arrays (based on sea ice concentration threshold)
	water_level_meters = pd.read_pickle(npy_path + 'ERAI_forced_water_levels_masked/water_levels_' + community_name + '_masked_' + str(year) + '.pkl')
	# interpolate the water_level_meters into hourly timeseries
	i = pd.DatetimeIndex(start=water_level_meters.index.min(), end=water_level_meters.index.max(), freq = 'H')
	# rewrite the water_level_meters so that it is in hourly timesteps
	water_level_meters = water_level_meters.reindex(i).interpolate(method='linear', limit=2)
	# save the hourly interpolated dataset
	water_level_meters.to_pickle(npy_path + 'ERAI_forced_water_levels_masked/hourly/water_levels_' + community_name + '_masked_' + str(year) + '_hourly.pkl')

	## load the wave height and period, sst. Interpolate the dataframe to hourly
	df = pd.read_pickle(npy_path + 'ERAI_forcing_variables_masked/ERAI_forcings_' + community_name + '_masked_' + str(year) + '.pkl')
	df_hourly = df.reindex(i).interpolate(method='linear', limit = 2)
	# write the hourly df to pkl
	df_hourly.to_pickle(npy_path + 'ERAI_forcing_variables_masked/hourly/ERAI_forcings_' + community_name + '_masked_' + str(year) + '_hourly.pkl')



