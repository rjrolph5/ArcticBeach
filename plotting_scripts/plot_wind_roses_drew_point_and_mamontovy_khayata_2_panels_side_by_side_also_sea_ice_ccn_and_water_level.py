import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from windrose import WindroseAxes
import windrose

# plot wind roses

# specify community name

#community_name = 'Mamontovy_Khayata'
community_name = 'Drew_Point'

# basepaths
basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
npy_path = basepath + 'input_data/storm_surge/'+ community_name +'/'
plot_path = basepath + 'plots/wind_roses_with_modelled_water_levels/'

if community_name == 'Mamontovy_Khayata':
	#year_range = np.arange(1995,2019) # up to year end
	year_range = np.array([1999, 2002])
	ifile_forcing_base = 'ERAI_forcings_Mamontovy_Khayata_masked_'
	ifile_water_level_base = 'water_levels_Mamontovy_Khayata_masked_'

if community_name == 'Drew_Point':
	#year_range = np.arange(2007,2017)
	year_range = np.array([2007, 2009])
	ifile_forcing_base = 'ERAI_forcings_Drew_Point_masked_'
	ifile_water_level_base = 'water_levels_Drew_Point_masked_'

print(community_name)

#def(year):

nrows, ncols = 1, 2

#fig = plt.figure(figsize=(12,4))
fig = plt.figure(figsize=(8,6))
bins = np.arange(1,16,2)
# change the default axes title format
plt.rcParams.update({'axes.titlesize': 12})
plt.rcParams.update({'axes.titleweight': 'bold'})

#def wind_rose_plot(year):
for (idx,year) in enumerate(year_range):
	print(year)

	# add subplot
	ax = fig.add_subplot(nrows, ncols, idx+1, projection='windrose')
	ax.set_title(str(year), pad=18)

	## read in data
	df_erai_forcing_filename = npy_path + 'ERAI_forcing_variables_masked/' + ifile_forcing_base  + str(year) + '.pkl'
	df_erai_forcing = pd.read_pickle(df_erai_forcing_filename)
	
	# extract the vector averaged wind speed and directions .  relative to shorenormal oceanographic convention was used when these ws and wd were calculated in storm_surge_ERA_Interim.py)
	# wind speed
	df_wind_speed = df_erai_forcing.wind_speed
	# wind direction
	df_wind_dir = df_erai_forcing.wind_direction - 180 #  convert to met convention for plots.  oceanographic convention was needed for the storm surge model and so this was what was stored in the pickled file.
	#
	## plot data onto subplot
	ax.bar(df_wind_dir, df_wind_speed, normed=True, bins=bins)

	## customize the tick labels and spacing
	ax.set_yticks(np.arange(5,31,step=5))
	ax.set_yticklabels(np.arange(5,31,step=5))
	ax.tick_params(axis='both', labelsize=9)

	## add a legend onto the rightmost subplot
	if year == year_range[-1]:
		ax.legend(title='Wind speed [m/s]', title_fontsize=14, prop={'size': 10}, loc='center left', bbox_to_anchor = (1.14,0.5)) # the loc parameter specifies which corner of the bounding box for the legend is placed.

fig.tight_layout()

# save the figure
plt.savefig(plot_path + community_name + '_wind_roses_2_panels' + '.png', bbox_inches = 'tight', dpi=300)

plt.show()

####### plot the corresponding modelled water levels ... with the same time period .. (e.g. from april to 
####### november..) showing sea ice concetration as well

nrows, ncols = 1, 2

fig = plt.figure(figsize=(12,4))
#plt.rcParams.update({'axes.titlesize': 16})
#plt.rcParams.update({'axes.titleweight': 'bold'})
yaxis_title_fontsize = 16

#def wind_rose_plot(year):
for (idx,year) in enumerate(year_range):
	print(year)

	# add subplot
	ax = fig.add_subplot(nrows, ncols, idx+1)
	ax.set_title(str(year), pad=18)

	## read in modelled water levesls
	ifile_water_level = npy_path + 'ERAI_forced_water_levels_masked/' + ifile_water_level_base + str(year) + '.pkl'
	modelled_water_levels_erai_yr = pd.read_pickle(ifile_water_level)

	# plot data
	ax.plot(modelled_water_levels_erai_yr, color='b')

	## set titles and axes formats
	ax.set_title(str(year), fontweight='bold', fontsize=20)
	# x-axis
	ax.set_xlim(pd.Timestamp(str(year)+'-06-01'), pd.Timestamp(str(year)+'-12-01'))
	ax.tick_params('x',rotation=90,pad=1,labelsize=14)
	ax.xaxis.label.set_color('b')
	# left (1st) y-axis
	if year == year_range[0]:
		ax.set_ylabel('Modelled water level [m]', fontsize=yaxis_title_fontsize, fontweight= 'bold', color='b')
	
	ax.tick_params('y', colors='b',labelsize=14)
	ax.set_ylim(-1.11,1.01)
	# set hline to differentiate between pos and neg water levels
	ax.axhline(y=0, color='b', linestyle='dashed')

	# add an hline to mean water level
	mean_wl = np.nanmean(modelled_water_levels_erai_yr)
	ax.axhline(y=mean_wl, color='r', linestyle = 'solid')

	## add sea ice concentration as a second axis
	# extract the sea ice concentration
	df_erai_forcing_filename = npy_path + 'ERAI_forcing_variables_masked/' + ifile_forcing_base  + str(year) + '.pkl'
	df_erai_forcing = pd.read_pickle(df_erai_forcing_filename)
	df_sicn = df_erai_forcing.sicn

	ax2 = ax.twinx()
	# only label the second y-axis if it is the last year (far right plot)
	if year == year_range[-1]:
		ax2.set_ylabel('Sea ice concentration [%]', fontsize = yaxis_title_fontsize, fontweight='bold', color='k')
	
	ax2.plot(df_sicn*100, color='k')
	ax2.axhline(y=15, color='k', linestyle='dashed')

	ax2.tick_params('y', colors='k',labelsize=14)
	ax2.xaxis.label.set_color('k')

	for label in ax.get_xticklabels():
		label.set_rotation(90)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

	if year == 2007:
		y_pos = 0.30
	if year == 2009:
		y_pos = 0.37
	if year == 1999:
		y_pos = 0.42
	if year == 2002:
		y_pos = 0.62
	# drew point 2007, x = 0.55, y= 0.38
	# drew point 2009, x = 0.55, y = 0.36
	# mk 1999, x = 0.55, y = 0.4
	# mk 2002, x = 0.55, y= 0.57

	plt.text(0.6, y_pos,'Mean water level', ha='left', va='center', color='r', transform=ax.transAxes, fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', edgecolor='none'))
	

fig.tight_layout()

plt.savefig(plot_path + community_name + '_modelled_water_level_2_panels.png', bbox_inches = 'tight', dpi=300)
plt.show()
