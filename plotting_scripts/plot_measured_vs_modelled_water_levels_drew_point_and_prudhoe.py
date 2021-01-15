'''

# Plot the measured and modelled water levels, calculated from:

	# save_measured_vs_modelled_water_level_drew_point_or_prudhoe.py

# offset values between the tide gauge and modelled water leve are calculated from:
 	# find_offset_tide_gauge_prudhoe_vs_modelled_drew_point.py

rebecca.rolph@awi.de
13 Oct 2020
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import optimize
import matplotlib.ticker as ticker

basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
plot_path = basepath + 'plots/'

community_name = 'Drew_Point'

year_range = np.arange(2007,2017)

##### select sample year for figure
year = 2007

npy_path = basepath + 'input_data/storm_surge/'+ community_name + '/'

## load the calculated water levels for 1 chosen example year
modelled_water_levels_erai_sample_year = pd.read_pickle(npy_path + 'ERAI_forced_water_levels_masked/water_levels_'+ community_name + '_masked_' + str(year) + '.pkl')

## load the tide gauge water levels for 1 example year only if Drew Point bc then there is a tide gauge available. The available tide gauge data is hourly (not interpolated).
tide_gauge_wl_sample_year = pd.read_pickle(basepath + 'input_data/storm_surge/prudhoe_bay/tide_gauge_pkld_water_levels/water_level_meters_tidegauge' + str(year) + '.pkl')
# find where modelled wl are nan, so you can apply it to tide gauge timesteps
first_timestamp_open_water_model = modelled_water_levels_erai_sample_year.first_valid_index()
last_timestamp_open_water_model = modelled_water_levels_erai_sample_year.last_valid_index()
#inds_where_ice = np.where(np.isnan(modelled_water_levels_erai_sample_year))[0]
tide_gauge_open_water_at_drew_point = tide_gauge_wl_sample_year[first_timestamp_open_water_model:last_timestamp_open_water_model]

## plot the timeseries of both masked tide gauge and modelled water levels for the example year
fig, ax = plt.subplots()
plt.plot(modelled_water_levels_erai_sample_year, label = 'Modelled for Drew Point', color='r')
plt.plot(tide_gauge_open_water_at_drew_point, label = 'Prudhoe Bay tide gauge', color='b')
ax.tick_params('x',rotation=90,pad=1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax.set_ylabel('Water level [m]', fontsize='large', fontweight= 'bold')
ax.set_ylim(-1.1,1.8)

# the number decides how many timesteps should be skipped before making a tickmark
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
# this number decides how many tickmarks are skipped before the label is put on
every_nth = 2
for n, label in enumerate(ax.xaxis.get_ticklabels()):
	if n % every_nth != 0:
		label.set_visible(False)

for label in ax.get_xticklabels():
	label.set_rotation(90)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

plt.legend(loc='lower left')
plt.savefig(plot_path + 'modelled_vs_measured_wl_' + community_name + '.png', bbox_inches='tight')
plt.show()

'''
## plot the timeseries of masked tide gauge and modelled water levels, but with an offset applied to the modelled water levels

# load the offset to add to the modelled water level so the mean matches the tide gauge data
offset_to_add_to_storm_surge_ERAI = np.load(basepath + 'input_data/storm_surge/Drew_Point/offset/offset_ERAI/offset_from_prudhoe_gauge_to_add_to_modelled_drew_point_' + str(year) + '.npy')

# re-plot tide gauge vs modelled water level but with offset this time.
fig,ax = plt.subplots()
#plt.title('Tide Gauge relative to MSL and offset \n applied to ERAI forced modelled water level ' + str(year) + '\n ' + community_name)
plt.plot(modelled_water_levels_erai_sample_year + offset_to_add_to_storm_surge_ERAI, label = 'Modelled for Drew Point, with offset', color='r')
plt.plot(tide_gauge_open_water_at_drew_point, label = 'Prudhoe Bay tide gauge', color='b')
ax.tick_params('x',rotation=90,pad=1)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax.set_ylabel('Water level [m]', fontsize='large', fontweight= 'bold')
ax.set_xlim(-1,1.7)

# the number decides how many timesteps should be skipped before making a tickmark
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

# this number decides how many tickmarks are skipped before the label is put on
every_nth = 2
for n, label in enumerate(ax.xaxis.get_ticklabels()):
	if n % every_nth != 0:
		label.set_visible(False)

for label in ax.get_xticklabels():
	label.set_rotation(90)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

plt.legend(loc= 'lower left')
plt.savefig(plot_path + 'modelled_vs_measured_wl_' + community_name + '_with_offset'+ str(year) +'.png', bbox_inches = 'tight')
plt.show()
# close figure
#plt.cla()
'''

## plot a histogram of all years between the OFFSET modelled and observed water levels

## load the calculated water levels for all years
# initialize the big dataframe that will contain all years of modelled data
df_modelled_water_levels_with_offset_erai_all_years = modelled_water_levels_erai_sample_year.copy()
df_gauge_prudhoe_all_years = tide_gauge_open_water_at_drew_point.copy()

for yr in year_range:
	# read modelled water level for each year
	yr_df = pd.read_pickle(npy_path + 'ERAI_forced_water_levels_masked/water_levels_'+ community_name + '_masked_' + str(yr) + '.pkl')
	# load the offset for each year 
	offset_to_add_to_storm_surge_ERAI = np.load(basepath + 'input_data/storm_surge/Drew_Point/offset/offset_ERAI/offset_from_prudhoe_gauge_to_add_to_modelled_drew_point_' + str(yr) + '.npy')

	# load the observed water level values
	tide_gauge_wl_yr = pd.read_pickle(basepath + 'input_data/storm_surge/prudhoe_bay/tide_gauge_pkld_water_levels/water_level_meters_tidegauge' + str(yr) + '.pkl')
	# find where modelled wl are nan, so you can apply it to tide gauge timesteps
	first_timestamp_open_water_model = yr_df.first_valid_index()
	last_timestamp_open_water_model = yr_df.last_valid_index()
	#inds_where_ice = np.where(np.isnan(modelled_water_levels_erai_sample_year))[0]
	tide_gauge_open_water_at_drew_point_yr = tide_gauge_wl_sample_year[first_timestamp_open_water_model:last_timestamp_open_water_model]

	# add the offset to the modelled water levels
	modelled_wl_with_offset_yr = yr_df + offset_to_add_to_storm_surge_ERAI

	# append the file to a master dataframe
	# bigdata = data1.append(data2, ignore_index=True)
	df_modelled_water_levels_with_offset_erai_all_years = df_modelled_water_levels_with_offset_erai_all_years.append(modelled_wl_with_offset_yr, ignore_index=False)
	df_gauge_prudhoe_all_years = df_gauge_prudhoe_all_years.append(tide_gauge_open_water_at_drew_point_yr, ignore_index=False)
	# create the histogram not here but in the 'if community_name == ' loop below..

# remove the initializing row
df_modelled_water_levels_with_offset_erai_all_years.index.drop_duplicates()
df_gauge_prudhoe_all_years.index.drop_duplicates()

# remove nan values for hist
df_modelled_water_levels_with_offset_erai_all_years_dropped_nan = df_modelled_water_levels_with_offset_erai_all_years.dropna()
df_gauge_prudhoe_all_years_dropped_nan = df_gauge_prudhoe_all_years.dropna()

fig, ax = plt.subplots()
n_modelled, bins_modelled, patches_modelled = plt.hist(x=df_modelled_water_levels_with_offset_erai_all_years_dropped_nan,density= True, bins='auto',color='r', alpha=0.7, rwidth=0.85, label = 'Modelled for Drew Point \nwith offset')
n_modelled_gauge, bins_modelled_gauge, patches_modelled_gauge = plt.hist(x=df_gauge_prudhoe_all_years_dropped_nan,density= True, bins='auto',color='b', alpha=0.7, rwidth=0.85, label = 'Prudhoe Bay tide gauge')
ax.legend()
ax.set_ylabel('Frequency')
ax.set_xlabel('Water level [m]')
ax.set_ylim(0,3.1)
plt.savefig(plot_path + 'modelled_vs_measured_wl_' + community_name + '_histogram_with_offset.png', bbox_inches = 'tight')
plt.show()

'''
############# plot cumulative postive modelled vs measured water levels ####################

# calculate cumulative postive modelled water levels

##### select the community
#community_name = 'Mamontovy_Khayata'
community_name = 'Drew_Point'

npy_path = basepath + 'input_data/storm_surge/' + community_name + '/'

if community_name == 'Mamontovy_Khayata':
	year_range = np.arange(1995, 2019)
	ifilebase = npy_path + 'ERAI_forced_water_levels_masked/water_levels_mamontovy_khayata_masked_'
	# this number decides how many tickmarks are skipped before the label is put on
	every_nth = 2

if community_name == 'Drew_Point':
	year_range = np.arange(2007, 2017)
	ifilebase = npy_path + 'ERAI_forced_water_levels_masked/water_levels_Drew_Point_masked_'
	# this number decides how many tickmarks are skipped before the label is put on
	every_nth = 1

####### load the modelled water levels and calc the sum
# initialize summed water level array
summed_modelled_wl_allyrs = np.empty((year_range.shape[0],year_range.shape[0]))
summed_modelled_wl_allyrs[:,0] = year_range

summed_pos_wl_allyrs = np.empty((year_range.shape[0],year_range.shape[0]))
summed_pos_wl_allyrs[:,0] = year_range

ind = -1
for year in year_range:
#for year in np.arange(2007,2008):
	ind = ind +1
	##### load the modelled water levels
	modelled_water_levels_erai_yr = pd.read_pickle(ifilebase + str(year) + '.pkl')

	#### find the positive water level indices
	inds_where_pos = np.where(modelled_water_levels_erai_yr > 0)[0]

	###  make array of pos water levels
	pos_wl_yr = modelled_water_levels_erai_yr[inds_where_pos]

	## sum of only positive water levels
	summed_pos_wl_yr = np.nansum(pos_wl_yr)
	## put into master array for all yrs
	summed_pos_wl_allyrs[ind,1] = summed_pos_wl_yr

	##### sum the water levels
	summed_modelled_wl_yr = np.nansum(modelled_water_levels_erai_yr)

	#### put the summed water levels into a master array for all years
	summed_modelled_wl_allyrs[ind,1] = summed_modelled_wl_yr

## plot the timeseries of both masked tide gauge and modelled water levels for the example year
fig, ax = plt.subplots()
ax.plot(summed_modelled_wl_allyrs[:,0], summed_modelled_wl_allyrs[:,1], color='b')
#ax.plot(summed_pos_wl_allyrs[:,0], summed_pos_wl_allyrs[:,1], color='b')
'''








