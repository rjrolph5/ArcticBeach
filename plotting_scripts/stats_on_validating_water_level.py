import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
plot_path = basepath + 'plots/'



######### Prudhoe Bay #############

community_name = 'prudhoe_bay'


##### select sample year for figure
year = 2007

npy_path = basepath + 'input_data/storm_surge/'+ community_name + '/'


### modelled water levels (masked where sea ice)
## load the calculated water levels for 1 chosen example year
modelled_water_levels_erai_sample_year = pd.read_pickle(npy_path + 'ERAI_forced_water_levels_masked/water_levels_'+ community_name + '_masked_' + str(year) + '.pkl')


### tide gauge data
# load the tide gauge water levels for 1 example year 
tide_gauge_wl_sample_year = pd.read_pickle(basepath + 'input_data/storm_surge/prudhoe_bay/tide_gauge_pkld_water_levels/water_level_meters_tidegauge' + str(year) + '.pkl')
# find where modelled wl are nan, so you can apply it to tide gauge timesteps
first_timestamp_open_water_model = modelled_water_levels_erai_sample_year.first_valid_index()
last_timestamp_open_water_model = modelled_water_levels_erai_sample_year.last_valid_index()
#inds_where_ice = np.where(np.isnan(modelled_water_levels_erai_sample_year))[0]
# final array of tide gauge data (masked where sea ice)
tide_gauge_obs_wl = tide_gauge_wl_sample_year[first_timestamp_open_water_model:last_timestamp_open_water_model]

#plt.plot(tide_gauge_obs_wl, label = 'Observed', color='b')
#plt.plot(modelled_water_levels_erai_sample_year, label = 'Modelled', color='r')
#plt.show()

## find the range of modelled water level
range_modelled_wl = np.nanmax(modelled_water_levels_erai_sample_year) - np.nanmin(modelled_water_levels_erai_sample_year)
print('range of modelled water level: ' + str(range_modelled_wl))

## find the range of the observed water level
range_obs_wl = np.nanmax(tide_gauge_obs_wl) - np.nanmin(tide_gauge_obs_wl)
print('range of observed water level: ' + str(range_obs_wl))

# interpolate to hourly

## correlate observed with modelled
tide_gauge_obs_wl.corr(modelled_water_levels_erai_sample_year)

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

'''
Timestamp('2007-09-11 09:00:00')
>>> modelled_water_levels_erai_sample_year.argmin()
Timestamp('2007-10-06 00:00:00')
>>> modelled_water_levels_erai_sample_year.argmax()
Timestamp('2007-09-10 21:00:00')
'''


df_modelled_wl = pd.DataFrame(data=modelled_water_levels_erai_sample_year.values, index=modelled_water_levels_erai_sample_year.index)
df_modelled_wl.columns = ['wl_modelled']

df_measured_wl = pd.DataFrame(data=tide_gauge_obs_wl.values, index=tide_gauge_obs_wl.index)
df_measured_wl.columns = ['wl_measured']


# truncate the modelled water levels so it only includes those timesteps that were measured.
df_modelled_wl = df_modelled_wl[df_measured_wl.index.min(): df_measured_wl.index.max()].copy()

# interpolate the modelled water level so that it is hourly, into the same timestamps as measured.
df_interpol_modelled = df_modelled_wl.resample('H') \
			.mean()

df_interpol_modelled['wl_modelled'] = df_interpol_modelled['wl_modelled'].interpolate()

# rename the index so they match with the modelled df
df_measured_wl.index.name = 'timestamps'

# combine the df into one array
df_merged = pd.merge(left=df_interpol_modelled, left_index=True, right=df_measured_wl, right_index=True, how='inner')

corr, p = scipy.stats.pearsonr(df_merged['wl_modelled'],df_merged['wl_measured'])
print('corr: ' + str(corr) + 'p_value: ' + str(p))

xcov_hourly = [crosscorr(df_merged.wl_modelled, df_merged.wl_measured, lag=i) for i in range(24)]


############ Mamontovy Khayata

# see script read_and_compare_water_depth_sensor2007_tiksi_with_ERAI_bykovsky.py


