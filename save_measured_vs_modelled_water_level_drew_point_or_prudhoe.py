# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:34:03 2020

Plot modelled water levels (ERAI-forced) against tide gauge data.
Save the modelled water levels (to be masked depending on sea ice concentration) for use in the erosion model.

Use this full timeseries which assumes no sea ice as an input for the script that masks these water level data
if the sea ice ccn exceeds a certain threshold.

@author: rrolph
"""

import get_storm_surge_values_from_TideGauge_prudhoe
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
basepath = '/home/rrolph/erosion_model/'
plot_path = '/home/rrolph/erosion_model_output_figures_too_large_for_github/'

bathymetry_ifile = basepath + 'input_data/storm_surge/Drew_Point/depthHR_drewPoint.npy' # specify bathymetry file to be used in the storm surge module.  Drew Point bathy file produced in script called 'bathymetry_produced_for_drew_point.py'
# filename of tide gauge data relative to MSL

## input lat/lon
'''Barrow met station
lat_site = 71.3230 # [degrees North]
lon_site = -156.6114 # [degrees West]
'''

'''
# Prudhoe bay tide gauge
lat_site = 70.402 # [degrees North]
lon_site = -148.519 # [degrees West]
community_name = 'prudhoe_bay'
'''

# Drew Point offshore site 
#lat_site = 70.88 # [degrees North]
#lon_site = -153.92 # [degrees West]
lat_site = np.load('/home/rrolph/erosion_model/input_data/storm_surge/Drew_Point/lat_offshore_site_ERAI_Drew_Point.npy')
lon_site = np.load('/home/rrolph/erosion_model/input_data/storm_surge/Drew_Point/lon_offshore_site_ERAI_Drew_Point.npy')
community_name = 'Drew_Point'

# choose an offshore grid cell where the sicn, winds, SST will be taken from.. this is visulalized on a map and produced from make_mask_with_extra_figures_single_year_sicn_threshold_ERAI_drew_point.py
#community_name = 'Drew_Point' # community name here also refers to the offshore point
#lat_site = np.load(npy_path + 'lat_offshore_site_ERAI_drew_point.npy')
#lon_site = np.load(npy_path + 'lon_offshore_site_ERAI_drew_point.npy')

npy_path = basepath + 'input_data/storm_surge/'+ community_name + '/'

## input shore-normal angle of coastline from true north
shorenormal_angle_from_north = 351 # [shore-normal, degrees clockwise from true north]. 351 used for Drew Point as a rough approximation.

## input the threshold at which the erosion model will start to exponentially decrease potential transport rate away from shore (qp)
storm_surge_threshold = 0.3 # just required as placeholder input when calculating water level from wind data, but does not affect results here (same function is called in other places where this input does matter, so it is kept here)

# save the inputs used to calculate the modelled water levels
input_array_to_produce_water_levels = np.array([lat_site, lon_site, shorenormal_angle_from_north, storm_surge_threshold])
np.save(npy_path + 'input_array_to_produce_modelled_water_levels_' + community_name + '.npy', input_array_to_produce_water_levels)

# function used for getting offset between obs and model water levels
def func(x,a):
        return x + a

for year in np.arange(2007,2008):
#for year in np.arange(2007,2017):

	print(year)

	start_date = str(year) + '-01-01'  # make month padded with preceding zero (e.g. july is 07 and not 7) , same with daynumber e.g. 2007-07-01 is july 1st).
	end_date = str(year) + '-12-31'

	start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
	end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

	if community_name == 'prudhoe_bay':
		# prudhoe bay tide gauge is used to compare modelled drew point water levels since that is the closest (to drew point) tide gauge data available
		tide_gauge_input_file =  basepath + 'input_data/storm_surge/prudhoe_bay/tide_gauge_data/Jan1_thru_Dec31_' + str(year) + '_Prudhoe_MSL.csv'

		## read in tide gauge data
		raw_surge_values, df_rel_waterLevel, water_level_meters_tidegauge, max_tide_gauge_values, max_water_level_index_in_total_array, start_storm_index_in_total_array,arrays_of_inds_consec_above_threshold,start_storm_surge_level,needed_index_between_start_and_peak = get_storm_surge_values_from_TideGauge_prudhoe.get_storm_surge_values(tide_gauge_input_file, storm_surge_threshold)
		# pickle the water level from the tide gauge data
		water_level_meters_tidegauge.to_pickle(npy_path + 'tide_gauge_pkld_water_levels/water_level_meters_tidegauge' + str(year) + '.pkl')

	## ERA forced water level calc: based on ERA-Interim, calculate storm surge water level
	water_level_meters_erai, ws_vector_average_ERAI, wd_vector_avg_ERAI,max_water_level_values_ERAI, max_water_level_index_in_total_array_ERAI, start_storm_index_in_total_array_ERAI,arrays_of_inds_consec_above_threshold_ERAI,start_storm_surge_level_ERAI,needed_index_between_start_and_peak_ERAI, lat_site_ERAI, lon_site_ERAI = storm_surge_ERA_Interim.storm_surge_from_era_interim(lat_site, lon_site, start_datetime, end_datetime, bathymetry_ifile,shorenormal_angle_from_north, storm_surge_threshold)
	# save the ERAI-forced water levels and other associated values
	water_level_meters_erai.to_pickle(npy_path + 'ERAI_forced_water_levels/ERAI_forced_WL_' + str(year) + '.pkl')

	print('lat_site_ERAI: ' + str(lat_site_ERAI))
	print('lon_site_ERAI: ' + str(lon_site_ERAI)) # this should be the same as the saved arrays in /home/rrolph/erosion_model/input_data/storm_surge/Drew_Point/lat_offshore*, which were produced by make_mask_with_extra_figures_single_year_sicn_threshold_ERAI_drew_point.py



