

# This script has been developed in response to Rev #2, where we should 'quantify how well the model performs'.

# At both sites, quantify modelled vs observed 1) retreat rates 2) water level

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#### select community to find RMSE for
#community_name = 'Mamontovy_Khayata'
community_name = 'Drew_Point'


##### find RMSE for retreat rates ########

##### Load data
basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
plot_path = basepath + 'plots/'


###### load the modelled retreat rates from water_level_solver
modelled_retreat = np.load(basepath + 'R_all_modelled_using_median_calcd_water_level_' + community_name + '.npy')

###### load observed retreat
observed_retreat_years = np.load(basepath + 'input_data/observed_retreat_years_' + community_name + '.npy')
retreat_observed = np.load(basepath + 'input_data/observed_retreat_rates_allyears_' + community_name + '.npy')

##### calculate RMSE
def rmse(observed, modelled):
	rmse = np.sqrt(np.average((observed - modelled)**2))
	return rmse

RMSE = rmse(retreat_observed, modelled_retreat)

print('RMSE for retreat at ' + community_name + ' is: ' + str(RMSE))

''' # dont have to make a plot for now
#### scatter plot of modelled vs observed retreat rates
fig, ax = plt.subplots(figsize=(3,4))
if community_name == 'Mamontovy_Khayata':
	fig, ax = plt.subplots(figsize=(6,4))

ax.scatter(retreat_observed, modelled_retreat)
plt.show()
'''


##### find RMSE for modelled water level ######

##### Prudhoe bay (2007 year, to correspond with Figure 5)

##### load data
community_name = 'prudhoe_bay'
year = 2007
npy_path = basepath + 'input_data/storm_surge/'+ community_name + '/'

#### modelled
modelled_water_levels_erai_sample_year = pd.read_pickle(npy_path + 'ERAI_forced_water_levels_masked/water_levels_'+ community_name + '_masked_' + str(year) + '.pkl')

#### observed
tide_gauge_wl_sample_year = pd.read_pickle(basepath + 'input_data/storm_surge/prudhoe_bay/tide_gauge_pkld_water_levels/water_level_meters_tidegauge' + str(year) + '.pkl')
# find where modelled wl are nan, so you can apply it to tide gauge timesteps
first_timestamp_open_water_model = modelled_water_levels_erai_sample_year.first_valid_index()
last_timestamp_open_water_model = modelled_water_levels_erai_sample_year.last_valid_index()
#inds_where_ice = np.where(np.isnan(modelled_water_levels_erai_sample_year))[0]
tide_gauge_obs_wl = tide_gauge_wl_sample_year[first_timestamp_open_water_model:last_timestamp_open_water_model]

### extract the same timestamps from the modelled water level as you have in the observed
modelled_water_levels_same_timestamps_as_obs = modelled_water_levels_erai_sample_year[first_timestamp_open_water_model:last_timestamp_open_water_model]

### resample to every 3h
tide_gauge_obs_wl_3h = tide_gauge_obs_wl.resample('3H').mean()

### apply the offset for the water level so that the baselines are the same (this was also done in Fig.5 and described in the caption).
def func(x,a):
	return x + a

tide_gauge_obs_wl_3h_drop_na = tide_gauge_obs_wl_3h.dropna()
modelled_water_levels_same_timestamps_as_obs_drop_na = modelled_water_levels_same_timestamps_as_obs.dropna()
idx = tide_gauge_obs_wl_3h_drop_na.index.intersection(modelled_water_levels_same_timestamps_as_obs_drop_na.index)
tide_gauge_truncated = tide_gauge_obs_wl_3h_drop_na.loc[idx]
model_truncated = modelled_water_levels_same_timestamps_as_obs_drop_na.loc[idx]
offset_to_add_to_modelled_wl, pcov = curve_fit(func, model_truncated, tide_gauge_truncated)

## apply offset
modelled_wl_prudhoe_2007_with_offset = offset_to_add_to_modelled_wl + model_truncated

RMSE = rmse(tide_gauge_truncated, modelled_wl_prudhoe_2007_with_offset)
print('RMSE for water level at ' + community_name + ' is: ' + str(RMSE))

#### MK (2007 year, the only year we have high freq water level obs)

## see read_and_compare_water_depth_sensor2007_tiksi_with_ERAI_bykovsky.py (search RMSE)




