"""
Created on Tue Jun  9 15:34:03 2020

Plot modelled water levels (ERAI-forced), assuming no sea ice cover.  The full timeseries produced in this script (modelled water levels and other data (sea surface temperature, winds, waves)) are subsequently masked in another script.

Mamontovy Khayata site:  site 3 ("C") on Bykovsky Peninsula in Overduin et al. (2007).

@author: rebecca.rolph@awi.de
"""

import storm_surge_ERA_Interim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
#from scipy import stats
from numpy.polynomial.polynomial import polyfit
from datetime import datetime
from scipy.optimize import curve_fit

## specify paths
basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
#plot_path = 

npy_path = basepath + 'input_data/storm_surge/Mamontovy_Khayata/'

bathymetry_ifile = npy_path + 'depth_bykovsky.npy'


# Input lat/lon of offshore site, which is chosen from the script: generate_grid_maps_and_choose_forcing_grid_cells_mamontovy_khayata.py. (The RA grid cell selected for winds can be visualized on a
# map there) These lat lon should be the same as in the make_mask_based_on_seaice_threshold_ERAI_input_community.py.  Note when masking the output used in this script (e.g the sea ice concentration data used for making water 
# levels produced by winds should come from the same grid cell as the wind data comes from).
lat_site = np.load(npy_path + 'lat_offshore_site_ERAI_mamontovy_hayata.npy')
lon_site = np.load(npy_path + 'lon_offshore_site_ERAI_mamontovy_hayata.npy')

## input shore-normal angle of coastline from true north
shorenormal_angle_from_north = 45 # [shore-normal, degrees clockwise from true north].

## input the threshold at which the erosion model will start to exponentially decrease potential transport rate away from shore (qp)
storm_surge_threshold = 0.3 # just required as placeholder input when calculating water level from wind data, but does not affect results here (same function is called in other places where this input does matter, so it is kept here)

for year in np.arange(1995,1996):
#for year in np.arange(1979,1995):
#for year in np.arange(1995,2019):  # these are the years observed in https://doi.org/10.1594/PANGAEA.905519 , 1995 up to 2019.

	print(year)

	start_date = str(year) + '-01-01'  # make month padded with preceding zero (e.g. july is 07 and not 7) , same with daynumber e.g. 2007-07-01 is july 1st).
	end_date = str(year) + '-12-31'

	start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
	end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

	## ERA forced water level calc: based on ERA-Interim, calculate storm surge water level
	water_level_meters_erai, ws_vector_average_ERAI, wd_vector_avg_ERAI,max_water_level_values_ERAI, max_water_level_index_in_total_array_ERAI, start_storm_index_in_total_array_ERAI,arrays_of_inds_consec_above_threshold_ERAI,start_storm_surge_level_ERAI,needed_index_between_start_and_peak_ERAI, lat_site_ERAI, lon_site_ERAI = storm_surge_ERA_Interim.storm_surge_from_era_interim(lat_site, lon_site, start_datetime, end_datetime, bathymetry_ifile,shorenormal_angle_from_north, storm_surge_threshold)

	# save the ERAI-forced water levels and other associated values
	water_level_meters_erai.to_pickle(npy_path + 'ERAI_forced_water_levels/ERAI_forced_WL_' + str(year) + '.pkl')

	'''
	# plot wind speed forcing for ERA-I (it has been converted relative to shorenormal)
	fig,ax = plt.subplots()
	plt.title('Wind speed forcing ' + str(year))
	ax.plot(ws_vector_average_ERAI,label='ERA-I relative to shorenormal (Bykovsky) 3 hourly')
	for label in ax.get_xticklabels():
		label.set_rotation(90)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
	plt.legend()
	plt.savefig(plot_path + 'wind_speed/ws_met_vs_ERAI_' + start_date + '_thru_' + end_date + '.png',bbox_inches='tight')
	plt.show()
	# close figure
	#plt.cla()

	# plot wind direction forcing for ERA-I
	fig,ax = plt.subplots()
	plt.title('Wind direction forcing, relative to shorenormal, \n oceanographic convention: ' + str(year))
	ax.plot(wd_vector_avg_ERAI, label = 'ERA-I Bykovsky')
	for label in ax.get_xticklabels():
		label.set_rotation(90)
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
	plt.legend()
	plt.savefig(plot_path + 'wind_direction/wind_dir_met_vs_ERAI_' + start_date + '_thru_' + end_date + '.png', bbox_inches = 'tight')
	plt.show()
	# close figure
	#plt.cla()
	'''
