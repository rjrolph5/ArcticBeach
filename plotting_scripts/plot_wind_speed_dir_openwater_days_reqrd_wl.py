import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


community_name = 'Mamontovy_Khayata'
#community_name = 'Drew_Point'

# basepaths
basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
npy_path = basepath + 'input_data/storm_surge/'+ community_name +'/'

if community_name == 'Mamontovy_Khayata':
	year_range = np.arange(1995,2019) # up to year end
	ifile_forcing_base = 'ERAI_forcings_Mamontovy_Khayata_masked_'

if community_name == 'Drew_Point':
	year_range = np.arange(2007,2017)
	ifile_forcing_base = 'ERAI_forcings_Drew_Point_masked_'

# initialize the open water days array
open_water_days_all_years = np.empty((year_range.shape[0],2))
open_water_days_all_years[:,0] = year_range

# initialize the wind speed array
wind_speed_masked_all_years = np.empty((year_range.shape[0],2))
wind_speed_masked_all_years[:,0] = year_range

# initialize the wind direction array
wind_direction_masked_all_years = np.empty((year_range.shape[0],2))
wind_direction_masked_all_years[:,0] = year_range

# calculate the number of open water days, and average wind speed and vector averaged wind directions for each year. then save each to file with years

ind = -1 # init
for year in year_range:
	print(year)
	ind = ind + 1

	# load the pkld file that has the forcing variables.
	df_erai_forcing = pd.read_pickle(npy_path + 'ERAI_forcing_variables_masked/' + ifile_forcing_base  + str(year) + '.pkl')

	##### calculate the number of open water days
	# extract the sea ice concentration
	df_sicn = df_erai_forcing.sicn

	# find the number of open water days
	threshold = 0.15

	# average the sicn to daily
	df_sicn_daily = df_sicn.groupby(pd.Grouper(freq='d')).mean()

	# find the number of days below the threshold
	number_open_water_days = np.where(df_sicn_daily<threshold)[0].shape[0]

	# fill in the number of open water days to a dataframe with the year
	open_water_days_all_years[ind,1] = number_open_water_days

	##### vector averaged wind speed and directions .  relative to shorenormal oceanographic convention was used when these ws and wd were calculated in storm_surge_ERA_Interim.py)
	# wind speed
	df_wind_speed = df_erai_forcing.wind_speed

	# wind direction
	df_wind_dir = df_erai_forcing.wind_direction - 180 #  convert to met convention for plots.  oceanographic convention was needed for the storm surge model and so this was what was stored in the pickled file.

	## average the wind speed and direction so they each have one value per each year
	u_average = np.nanmean(df_wind_speed*np.sin(np.radians(df_wind_dir)))
	v_average = np.nanmean(df_wind_speed*np.cos(np.radians(df_wind_dir)))

	# compute scalar average wind speed (recommendation of Tech. Note. Grange 2014)
	df_wind_speed_avg = np.nanmean(df_wind_speed)

	# find the average wind direction
	df_wind_dir_avg = np.arctan2(u_average,v_average)*360/2/np.pi

	# append the wind speed and directions to a master array to be saved
	wind_speed_masked_all_years[ind,1] = df_wind_speed_avg
	wind_direction_masked_all_years[ind,1] = df_wind_dir_avg

# save the number of open water days
np.save(npy_path + 'openwater_days/open_water_days_all_years_' + community_name + '.npy', open_water_days_all_years)
np.save(npy_path + 'wind_dir/wind_dir_all_years_' + community_name + '.npy', wind_direction_masked_all_years)
np.save(npy_path + 'wind_speed/wind_speed_all_years_' + community_name + '.npy', wind_speed_masked_all_years)


## load the water level offset
experiment_name = 'avg'

wl_offset_path = basepath + 'input_data/storm_surge/' + community_name + '/required_offset_to_match_observed_retreat/' + experiment_name + '/'

if community_name == 'Mamontovy_Khayata':
        year_range = np.arange(1995, 2019)

if community_name == 'Drew_Point':
        year_range = np.arange(2007, 2017)

water_offset_ts = np.ones(1)
for year in year_range:
        ifile = wl_offset_path + 'water_offset_reqrd_to_match_obs_' + str(year) + '.npy'
        water_offset1 = np.load(ifile)
        water_offset_ts = np.append(water_offset_ts, water_offset1)

water_offset_ts = water_offset_ts[1:]



## make reqrd water level offset and open water days into a bar plot
#fig, ax = plt.subplots(figsize=(3, 2))
fig,ax = plt.subplots()

width = 0.35

ax.bar(open_water_days_all_years[:,0].astype(int), open_water_days_all_years[:,1], width, color='blue')

#plt.xlim([year_range[0],year_range[-1] + 1])

ax2 = ax.twinx()

ax2.scatter(year_range, water_offset_ts, marker = '*', color='red', s= 120)

# calculate the median water level and add it as a horizontal line to the plot
median_reqrd_wl = np.median(water_offset_ts)
plt.axhline(median_reqrd_wl, color='r',lw=2,ls='dashed', label = 'Median reqrd water level')

# calculate the average water level and add it as a horizontal line to the plot
avg_reqrd_wl = np.mean(water_offset_ts)
plt.axhline(avg_reqrd_wl,color='k', lw= 2, label = 'Avg reqrd water level')

# the number decides how many timesteps should be skipped before making a tickmark
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# this number decides how many tickmarks are skipped before the label is put on
if community_name == 'Mamontovy_Khayata':
	every_nth = 2
else:
	every_nth = 1
for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
                label.set_visible(False)
for label in ax.get_xticklabels():
	label.set_rotation(90)
	label.set_fontsize(16)

for label in ax.get_yticklabels():
	label.set_fontsize(16)

for label in ax2.get_yticklabels():
	label.set_fontsize(16)

#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.set_ylabel('Required water level offset [m]', color = 'red', fontsize=16, fontweight='bold', rotation = 270, labelpad = 18)
ax2.tick_params(axis='y', colors='red')
ax.set_ylabel('Number open water days', color = 'blue', fontsize=16, fontweight='bold')
ax.tick_params(axis='y', colors='blue')

#plt.axis('tight')
ax.set_xlim([year_range[0] - width, year_range[-1]+ width])
ax2.set_xlim([year_range[0] - width ,year_range[-1]+ width])

ax.set_ylim([0,139])
ax2.set_ylim([-0.3,2.7])
plt.legend()
plt.savefig(basepath + 'plots/wind_speed_dir_open_water/owdays_with_wl_offset_' + community_name + '.png', bbox_inches = 'tight')
plt.show()

######### make a plot to compare the average and median required water levels ##########

fig, ax = plt.subplots()
# plot the yearly required water level offset
ax.scatter(year_range, water_offset_ts, marker = '*', color='r', s= 120)
ax.set_ylabel('Required water level offset [m]', color = 'red', fontsize=16, fontweight='bold', labelpad = 18)
ax.tick_params(axis='y', colors='red')

# calculate the median water level and add it as a horizontal line to the plot
median_reqrd_wl = np.median(water_offset_ts)
plt.axhline(median_reqrd_wl, color='b', label = 'Median reqrd water level')

# calculate the average water level and add it as a horizontal line to the plot
avg_reqrd_wl = np.mean(water_offset_ts)
plt.axhline(avg_reqrd_wl,color='k', label = 'Avg reqrd water level')
# the number decides how many timesteps should be skipped before making a tickmark
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

# this number decides how many tickmarks are skipped before the label is put on
if community_name == 'Mamontovy_Khayata':
	every_nth = 2
else:
	every_nth = 1
for n, label in enumerate(ax.xaxis.get_ticklabels()):
	if n % every_nth != 0:
		label.set_visible(False)
for label in ax.get_xticklabels():
	label.set_rotation(90)
	label.set_fontsize(16)

for label in ax.get_yticklabels():
	label.set_fontsize(16)

for label in ax2.get_yticklabels():
	label.set_fontsize(16)

plt.ylim(-0.3,2.7)
plt.legend()
plt.savefig(basepath + 'plots/reqrd_wl_avg_vs_median_' + community_name + '.png',bbox_inches='tight')
plt.show()













