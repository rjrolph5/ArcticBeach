# -*- coding: utf-8 -*-
"""
Reads Prudhoe Bay tide gauge data. An optional forcing for water levels used by the erosion model is tide gauge data.
Created on Wed Apr 15 16:10:11 2020
@author: rrolph
"""

import pandas as pd
import numpy as np

def consecutive(data, stepsize=1):
	return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def get_storm_surge_values(tide_gauge_input_file,storm_surge_threshold):

	# create dataframe from tide gauge datafile
	df_rel_waterLevel = pd.read_csv(tide_gauge_input_file, sep=",", header=0, parse_dates={'date':[0,1]}, index_col=['date'], dayfirst=False, squeeze=True,na_values='-')

	# convert ft from datafile to meters
	water_level_meters = df_rel_waterLevel['Verified (ft)']*0.3048

	# add to meter water level dataframe
	df_rel_waterLevel['rel_water_level_m'] = water_level_meters # the 'raw' data read in from tide gauge

	# change variable name
	tide_gauge_values = df_rel_waterLevel['rel_water_level_m']

	# get storm surge start and peak indices and values. 
	# find the storm surge peak because this is the point where you invoke the exponentially-decaying behavior of qp (eqn. 20 in Larson 1989a_Sbeach_Report1)

	# find all the sequential timesteps that the storm surge is above specified relative water level threshold for storms. if there is a gap between the timesteps of the storm, then that 'storm surge occurrence' is over. 

	# find indices where relative water level is above this threshold
	ind_water_above_threshold = np.where(tide_gauge_values>storm_surge_threshold)[0]

	# initialize arrays
	start_storm_index_in_total_array = np.zeros(1)
	max_water_level_index_in_total_array = np.zeros(1)

	# set what happens when there is no timestep that exceeded the storm surge threshold for the selected timeslice span.
	if ind_water_above_threshold.shape[0] == 0:
		print('storm surge threshold not exceeded for selected timespan')
		## maybe make the below in an 'else' statement, make sure required outputs are nan if necssary
		max_tide_gauge_values = np.nan
		max_water_level_index_in_total_array = np.nan 
		start_storm_index_in_total_array = np.nan
		arrays_of_inds_consec_above_threshold = np.nan
		start_storm_surge_level = np.nan
		needed_index_between_start_and_peak = np.nan
	else:
		# to find consecutive indices of storm surge values (those above specified threshold of water level)
		arrays_of_inds_consec_above_threshold = consecutive(ind_water_above_threshold)

		for array in arrays_of_inds_consec_above_threshold: # array is the storm occurrence and contains the indices of the ifile data where the relative water level exceeds the specified threshold.
			# find the index in the total tide gauge values array where the start of the storm surge water level is 
			start_storm_index_in_total_array1 = array[0]

			# append the start index that corresponds to master storm surge array to an array of start indices
			start_storm_index_in_total_array = np.append(start_storm_index_in_total_array,start_storm_index_in_total_array1)

			# find max water level of storm in current loop.
			max_water_level_in_array1 = np.nanmax(tide_gauge_values[array])

			#max_water_level_in_array = np.append(max_water_level_in_array,max_water_level_in_array1)

			# find the index in the total tide gauge values array where that max water level is 
			max_water_level_index_subarray = np.where(max_water_level_in_array1==tide_gauge_values[array])[0]
			# if there are two max water levels because they are the same value, take the last one
			max_water_level_index_in_total_array1 = array[max_water_level_index_subarray[-1]]

			max_water_level_index_in_total_array = np.append(max_water_level_index_in_total_array,max_water_level_index_in_total_array1).astype(int)

			# find the peak storm surge water level values 
			max_tide_gauge_values = tide_gauge_values[max_water_level_index_in_total_array]

			# find the start of the storm surge water level values
			start_storm_index_in_total_array = start_storm_index_in_total_array.astype(int)
			start_storm_surge_level = tide_gauge_values[start_storm_index_in_total_array]

		# get rid of leading zero from array initialization
		start_storm_index_in_total_array = start_storm_index_in_total_array[1:]
		max_water_level_index_in_total_array = max_water_level_index_in_total_array[1:]

		needed_index_between_start_and_peak = np.array([0])
		for ind,value_start in enumerate(start_storm_index_in_total_array):
			#print(value_start)
			#print(ind)
			value_peak = max_water_level_index_in_total_array[ind] # get the next peak_surge that corresponds to the current start_surge
			if value_peak > value_start:
				needed_index_between_start_and_peak1 = np.arange(value_start,value_peak)
			else:
				needed_index_between_start_and_peak1 = value_start
				needed_index_between_start_and_peak = np.append(needed_index_between_start_and_peak,needed_index_between_start_and_peak1)
				needed_index_between_start_and_peak = needed_index_between_start_and_peak[1:]
	return df_rel_waterLevel.rel_water_level_m, df_rel_waterLevel, tide_gauge_values, max_tide_gauge_values, max_water_level_index_in_total_array, start_storm_index_in_total_array,arrays_of_inds_consec_above_threshold,start_storm_surge_level,needed_index_between_start_and_peak

