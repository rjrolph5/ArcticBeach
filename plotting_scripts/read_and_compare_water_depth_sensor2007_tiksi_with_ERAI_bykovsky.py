# -*- coding: utf-8 -*-
"""
Created on Wed May 27 10:55:43 2020

Sensitive water depth sensor installed in Tiksi Bay for Marita Scheller (data sent by Paul Overduin).
Coordinates:  71°31'52.3"   129°33'33" (71.53 , 129.56 decimal degrees)
Hourly values relative to null

@author: rrolph
"""
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.xmargin'] = 0 # removes whitespace around data in plots.
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import scipy.stats

basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'

def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

def read_july_thru_oct_2007_tiksi_water_level_data():

	ifile = basepath + 'input_data/storm_surge/lapt0708.pgl'

	df = pd.read_csv(ifile,parse_dates={'date':[0]},index_col=['date'],delim_whitespace=True)

	# add a column with the datetime per the thesis, since the gregorian datetimes do not seem to add up to starting at ('2007-07-31 00:00:00') .. "Wasserstand aller 15 Minuten beginnend am 31. Juli 
	# 2007 bis zum 31. Juli 2008. In der Datei sind in der ersten Spalte das Gregorianische Datum (+2400000) und in der zweiten Spalte die Wassersäule (-höhe) in Metern angegeben. "

	# create timeseries, every 15 minutes from ('2007-07-31 00:00:00') through 31 July 2008. 
	start = pd.to_datetime('2007-07-31 00:00:00')
	end = pd.to_datetime('2008-07-31 00:00:00')

	dts = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in datetime_range(start,end,timedelta(minutes=15))]

	# put datetimes as the index for the dataframe 
	datetimes = pd.to_datetime(dts)

	df = df.set_index(datetimes)

	# make a new, truncated dataframe of water levels showing only 3 hour means (since ERA-I is a 3 hour average of winds)
	df_3hourly_mean = df.resample('3H').mean()

	return df_3hourly_mean


########## compare with ERA-I forced water levels at bykovsky location #####################

start_date = '2007-07-31'
end_date = '2007-10-01'

# based on ERA-Interim, calculate storm surge water level
npy_path = basepath + 'input_data/storm_surge/Mamontovy_Khayata/'
plot_path = basepath + 'plots/'

lat_site = np.load(npy_path + 'lat_offshore_site_ERAI_mamontovy_hayata.npy') # this is the lat used to calculate the bykovsky water levels in the erosion model  (72.75 N, closest on the selected ERAI grid)
#lat_site = 71.53 # from Marita Scheller's thesis
print(lat_site)
lon_site = np.load(npy_path + 'lon_offshore_site_ERAI_mamontovy_hayata.npy') # lon for erosion model (129.75)
#lon_site = 129.56 # from Scheller's thesis.
print(lon_site)

start_datetime = pd.to_datetime(start_date)
end_datetime = pd.to_datetime(end_date)

bathymetry_ifile = npy_path + 'depth_bykovsky.npy'

shorenormal_angle_from_north = 45 # [shore-normal, degrees clockwise from true north]
storm_surge_threshold = 999 # does not affect value of total water level timeseries, but is need to determine the start & stop points for the storm surges, which is used when coupled to the erosion model.

# call the function above to read in the meaured water level data
df_3hourly_mean_obs = read_july_thru_oct_2007_tiksi_water_level_data()

# take subset of the measured water level data
df_3hourly_mean_obs_subset = df_3hourly_mean_obs.loc[start_date:end_date]

# load the modelled water level (masked)
water_level_meters_erai = pd.read_pickle(npy_path + 'ERAI_forced_water_levels_masked/water_levels_Mamontovy_Khayata_masked_2007.pkl')



################  plot high-temporal observed vs modelled water levels.
fig, ax = plt.subplots(figsize=(5,4))
plt.title('Tiksi water level data no offset \n July 31 2007 - Oct 31 2007')
ax.plot(df_3hourly_mean_obs_subset.obs_water_level,label = '3h mean of 15 minute observations off Tiksi',color='b')
# save the df 3hour mean of 15 min obs for tiksi
#x = df_3hourly_mean_obs_subset.obs_water_level
#x.to_csv('/home/rrolph/erosion_model/input_data/storm_surge/tiksi/scheller_obs_tiksi_3hmean_water_level_2007.csv', encoding='utf-8')

ax.plot(water_level_meters_erai[start_datetime:end_datetime],label = 'ERA-I forced modelled water level Bykovsky',color='r')
df_modelled_to_save = water_level_meters_erai[start_datetime:end_datetime]
# save the modelled water level for tiksi
#df_modelled_to_save.to_csv('/home/rrolph/erosion_model/input_data/storm_surge/tiksi/modelled_water_level_tiksi2007.csv', encoding='utf-8')

for label in ax.get_xticklabels():
    label.set_rotation(90)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%y'))

ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend(loc='upper left', prop={'size': 12})
#ax.set_ylim(-1.1,1.8)
fig.tight_layout()
plt.savefig(plot_path + 'modelled_vs_obs_water_level_no_offset_Tiksi2007.png', bbox_inches='tight', dpi=300)
plt.show()


# find the mean offset in this dataset
mean_offset = np.nanmean(df_3hourly_mean_obs_subset.obs_water_level) - np.nanmean(water_level_meters_erai[start_datetime:end_datetime])

# find the mean offset of the observations to subtract from both datasets
mean_offset_from_obs = np.nanmean(df_3hourly_mean_obs_subset.obs_water_level)



################ apply mean offset and plot comparison btwn obs and modelled again.
fig, ax = plt.subplots(figsize=(5,4))
#plt.title('Observed and modelled water level using bykovsky bathy')
ax.plot(df_3hourly_mean_obs_subset.obs_water_level - mean_offset_from_obs,label = 'Observed',color='b')
ax.plot(water_level_meters_erai[start_datetime:end_datetime] + mean_offset - mean_offset_from_obs,label = 'Modelled',color='r')
ax.set_ylabel('Water level [m]',fontsize=25) #fontweight='bold')

# the number decides how many timesteps should be skipped before making a tickmark
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))

# this number decides how many tickmarks are skipped before the label is put on
every_nth = 2
for n, label in enumerate(ax.xaxis.get_ticklabels()):
	if n % every_nth != 0:
		label.set_visible(False)

for label in ax.get_xticklabels():
	label.set_rotation(90)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%y'))

ax.tick_params(axis='both', which='major', labelsize=15)
ax.legend(loc='upper left', prop={'size': 12})
fig.tight_layout()
plt.savefig(plot_path + 'modelled_vs_obs_water_level_offset_applied_Tiksi2007.png', bbox_inches='tight',dpi=300)
plt.show()

############### calculated RMS in response to Reviewer #2 ################
obs_with_offset = df_3hourly_mean_obs_subset.obs_water_level - mean_offset_from_obs
modelled = water_level_meters_erai[start_datetime:end_datetime] + mean_offset - mean_offset_from_obs

## have to find intersection of the timestamps
idx = obs_with_offset.index.intersection(modelled.index)

## truncate based on those intersections
obs_with_offset_truncated = obs_with_offset[idx]
modelled_truncated = modelled[idx]

def rmse(observed, modelled):
	rmse = np.sqrt(np.average((observed - modelled)**2))
	return rmse

RMSE = rmse(obs_with_offset_truncated, modelled_truncated)
print('RMSE for water level at MK is :' + str(RMSE))




#### compute stats of obs vs modelled water levels:  range and pearson correlation coeff

wl_obs = df_3hourly_mean_obs_subset.obs_water_level - mean_offset_from_obs
wl_modelled = water_level_meters_erai[start_datetime:end_datetime] + mean_offset - mean_offset_from_obs

# find range of modelled water level
range_modelled_wl = np.nanmax(wl_modelled) - np.nanmin(wl_modelled)
print('range of modelled water level: ' + str(range_modelled_wl))

## find the range of the observed water level
range_obs_wl = np.nanmax(wl_obs) - np.nanmin(wl_obs)
print('range of observed water level: ' + str(range_obs_wl))

## correlate the modelled and observed
df_modelled_wl = pd.DataFrame(data=wl_modelled.values, index=wl_modelled.index)
df_modelled_wl.columns = ['wl_modelled']
df_measured_wl = pd.DataFrame(data=wl_obs.values, index=wl_obs.index)
df_measured_wl.columns = ['wl_measured']

df_merged = pd.merge(left=df_modelled_wl, left_index=True, right=df_measured_wl, right_index=True, how='inner')
corr, p = scipy.stats.pearsonr(df_merged['wl_modelled'],df_merged['wl_measured'])
print('corr: ' + str(corr) + 'p_value: ' + str(p))




#### plot a histogram for the 1 year that you have measured vs modelled water level data

# remove nan values for hist
df_3h_mean_obs_subset_minus_mean_obs = df_3hourly_mean_obs_subset.obs_water_level - mean_offset_from_obs
#df_obs_water_level_dropped_nan = df_3hourly_mean_obs_subset.obs_water_level.dropna() # observed
df_obs_water_level_dropped_nan = df_3h_mean_obs_subset_minus_mean_obs.dropna() # observed
water_level_meters_erai_dropped_nan = water_level_meters_erai[start_datetime:end_datetime]+mean_offset-mean_offset_from_obs # modelled
water_level_meters_erai_dropped_nan = water_level_meters_erai_dropped_nan.dropna()

# modelled
fig, ax = plt.subplots(figsize=(5,4))
n_modelled_gauge, bins_modelled_gauge, patches_modelled_gauge = plt.hist(x=df_obs_water_level_dropped_nan,density= True, bins='auto',color='b', alpha=0.7, rwidth=0.85, label = 'Observed')
n_modelled, bins_modelled, patches_modelled = plt.hist(x=water_level_meters_erai_dropped_nan, density= True, bins='auto',color='r', alpha=0.7, rwidth=0.85, label = 'Modelled')
ax.legend(loc='upper left', prop={'size': 12})
ax.set_ylabel('Frequency', fontsize=25) #fontweight='bold')
ax.set_xlabel('Water level [m]', fontsize=25) # fontweight='bold')
ax.set_ylim(0,3.2)
ax.set_xlim(-1.01,1.7)

props = dict(boxstyle='round', facecolor='wheat',alpha=0.5)
ax.text(0.70,0.95,'MK (2007)', transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

ax.tick_params(axis='both', which='major', labelsize=15)
fig.tight_layout()
plt.savefig(plot_path + 'modelled_vs_measured_wl_tiksi_histogram_with_offset.png', bbox_inches = 'tight',dpi=300)
plt.show()


















