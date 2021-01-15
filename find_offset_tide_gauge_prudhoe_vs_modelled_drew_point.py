import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

'''
Calculates and saves the offset between the masked modelled water level and masked tide gauge data.

'''


basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'

# function used for getting offset between obs and model water levels
def func(x,a):
        return x + a


for year in np.arange(2007, 2017):
	print(year)
	##### read in the tide gauge data
	tide_gauge_wl_sample_year = pd.read_pickle(basepath + '/input_data/storm_surge/prudhoe_bay/tide_gauge_pkld_water_levels/water_level_meters_tidegauge' + str(year) + '.pkl')

	##### read in the modelled water level data
	modelled_water_levels = pd.read_pickle(basepath + 'input_data/storm_surge/Drew_Point/ERAI_forced_water_levels_masked/water_levels_Drew_Point_masked_' + str(year) + '.pkl')

	# mask the tide gauge data. find where modelled wl are nan, so you can apply it to tide gauge timesteps
	first_timestamp_open_water_model = modelled_water_levels.first_valid_index()
	last_timestamp_open_water_model = modelled_water_levels.last_valid_index()
	#inds_where_ice = np.where(np.isnan(modelled_water_levels))[0]
	tide_gauge_open_water_at_drew_point = tide_gauge_wl_sample_year[first_timestamp_open_water_model:last_timestamp_open_water_model]

	##### find the offset between the tide gauge and the modelled water level
	tide_gauge_level = tide_gauge_open_water_at_drew_point.dropna()
	modelled_wl = modelled_water_levels.dropna()
	idx = tide_gauge_level.index.intersection(modelled_wl.index)
	tide_gauge_level_truncated = tide_gauge_level.loc[idx]
	surge_model_level_truncated = modelled_wl.loc[idx]
	offset_to_add_to_storm_surge_ERAI, pcov_met = curve_fit(func, surge_model_level_truncated, tide_gauge_level_truncated)

	##### save the offset
	np.save(basepath + 'input_data/storm_surge/Drew_Point/offset/offset_ERAI/offset_from_prudhoe_gauge_to_add_to_modelled_drew_point_' +str(year)+ '.npy', offset_to_add_to_storm_surge_ERAI)



