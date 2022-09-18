# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:44:52 2020

Read ERA-Interim RA data and calculate storm surge or relative water level. Much of this storm surge code has been translated from matlab into python by R. Rolph. Original matlab code from T. Ravens 
(2012).  Edits include calculating vector averaged winds instead of scalar averaged, and using ERA instead of met station data, plus a different method for adjusting for beach direction reference frame.

@author: rebecca.rolph@awi.de
"""

import numpy as np
#import ERA_interim_read
import ERA_interim_read_with_wave_and_sst
from datetime import datetime


def consecutive(data, stepsize=1):
	return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

#def storm_surge_from_era_interim(lat_site, lon_site, start_datetime, end_datetime, bathymetry_ifile,shorenormal_angle_from_north, storm_surge_threshold):

## specify paths
basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
#plot_path =

npy_path = basepath + 'input_data/storm_surge/veslobogen/'

lat_site = np.load(npy_path + 'lat_offshore_site_ERAI_veslobogen.npy')
lon_site = np.load(npy_path + 'lon_offshore_site_ERAI_veslobogen.npy')
bathymetry_ifile = npy_path + 'depth_veslobogen.npy'

year = 2014
start_date = str(year) + '-01-01'  # make month padded with preceding zero (e.g. ju>
end_date = str(year) + '-12-31'


## input shore-normal angle of coastline from true north
shorenormal_angle_from_north = 80 # [shore-normal, degrees clockwise from true north].

## input the threshold at which the erosion model will start to exponentially decrease pote>
storm_surge_threshold = 0.3 

start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

# read in ERA-interim wind components from function file ERA_interim_read_and_plot.py. returns dataframe of vector-calculated wind speed and direction at the latitude/lon of site specfied.  beach coordinate system not implemented here (e.g. wind directions need to be adjusted after calling this function if shorenormal is not due north).
df, lat_site_ERA, lon_site_ERA = ERA_interim_read_with_wave_and_sst.read_ERAInterim(lat_site, lon_site, start_datetime, end_datetime)

# load bathymetry
depthHR =  np.load(bathymetry_ifile)[:-2] # see script called 'bathymetry_produce_for_drew_point.py' to see how this higher resolution bathy file was produced.

# change wind direction from reanalysis met convention (where wind is coming from, degrees clockwise from north) to the beach-specific coordinate system. in other words, find the angle between onshore (x-axis) and local wind vector. 
wind_dir_new = df.wind_direction + 360 - shorenormal_angle_from_north

inds_where_new_wd_over_360 = np.where(wind_dir_new > 360)[0]
winds_minus_360_df = wind_dir_new - 360
wind_dir_new.iloc[inds_where_new_wd_over_360] = winds_minus_360_df.iloc[inds_where_new_wd_over_360]

ws = df.wind_speed

### below is the same as the storm surge function that reads from met.

# calculate u and v wind components in the reference frame of the beach.
u = -ws*np.sin(2*np.pi*wind_dir_new/360) # alongshore wind speed (parallel to beach)  .the negative sign negates the direction. this negation is because wind direction, by met convention, is defined from where the wind is blowing from, while the vectors define the direction where the flow is heading to. (see technical note, Grange 2014). positive is blowing towards the right along the beach from the perspective of standing on the beach looking toward the ocean. (not necessarily blowing to the east, depending on the beach orientation.  but if the shorenormal is pointing due north then a wind blowing toward the east would have a positive u-direction.)

v = -ws*np.cos(2*np.pi*wind_dir_new/360) # crosshore wind speed (offshore/onshore, perpendicular to beach), positive away from coast

# since reanalysis is not as high of a temporal resolution as met station data (e.g. 6 hourly instead of 1 hourly), we do not take the mean here but leave it as an option (this is why the variables have not been changed from _mean or ave labels below)

# average the wind components according to the user-defined input time-span for average (e.g. 12h averages)
#freq_str = str(number_hours_to_avg_wind_over)+'H'

'''
# calculate means of the components of wind speeds
#u_mean = u.groupby(pd.Grouper(freq=freq_str)).mean()  # average alongshore wind speed
u_mean = u.rolling(freq_str,min_periods=1).mean()
#v_mean = v.groupby(pd.Grouper(freq=freq_str)).mean()  # average cross-shore wind speed
v_mean = v.rolling(freq_str,min_periods=1).mean()
'''
u_mean = u.copy()

v_mean = v.copy()

# calculate the mean of the barometric pressure 
#Pave = df_met_data.barometric_pressure_hPa.groupby(pd.Grouper(freq=freq_str)).mean()
#Pave = df_met_data.barometric_pressure_hPa.rolling(freq_str,min_periods=1).mean()
Pave = u.copy() # copy for shape
Pave[:] = np.nan # Pressure is neglected from the reanalysis data for now (it only makes a few cm difference for the water level and is not significant in the erosion model).

# find the vector averaged wind direction
wd_vector_avg = (np.arctan2(u_mean,v_mean)* 360/2/np.pi) # do not change wind direction vector back to met convention since now oceanographic convention is used.

# find vector averaged wind speed
ws_vector_average = (u_mean**2 + v_mean**2)**0.5

# calculate wind speed in the longshore direction
Wy = ws_vector_average*np.sin(np.radians(wd_vector_avg))

# calculate wind speed in the onshore/offshore (cross shore) direction
Wx = ws_vector_average*np.cos(np.radians(wd_vector_avg))

# calculate means of the components of wind stresses
Tsy = (1030*2e-6)*Wy**2*Wy/np.abs(Wy)
Tsx = (1030*2e-6)*Wx**2*Wx/np.abs(Wx)

'''
# manually take the vector average of the first 12 hour period to see if the grouper function is working as you want it to
ws_first_12h = ws[0:12]
# take mean first 12h of vector components
Wy_first12h = np.nanmean(u[0:12])
Wx_first12h = np.nanmean(v[0:12])
# find vector avgd wind dir for first 12h
wd_vector_avg_first12h = (np.arctan2(Wy_first12h,Wx_first12h)*360/2/np.pi) + 180
# find vector avgd wind speed for first 12h
ws_vector_avg_first12h = (Wy_first12h**2 + Wx_first12h**2)**0.5

# wd_vector_avg[0]
# Out: 95.45385597252502

# wd_vector_avg_first12h
# Out: 95.45385597252502

# ws_vector_avg[0]
# Out: 5.79299342647685

# ws_vector_avg_first12h
# Out: 5.79299342647685
'''

# define number of timesteps of averaged timeseries. 
Nhdays = ws_vector_average.shape[0]

# define cross-shore bathymetry 
# distance_where_surge_zero = 125000 # [m] distance offshore where storm surge assumed to have no effect. 125 km used in Ravens et al. (2012)
dx = 1000

## calculate V(x) alongshore current from cross-shore bathymetry. (starts at lines 1591 in .m script)

# define constants
Ngrid = depthHR.shape[0]  # number of grid points along high resolution bathymetry transect. 
RO = 0.1
f = 2*7.272e-5*np.sin(lat_site*np.pi/180) # coriolis parameter assumed constant on cross-shore transect

# initialize arrays
h = np.ones(Ngrid) # water depth from bathymetry (ie. MSL)
V = np.ones(Ngrid) # alongshore current velocity
aida = np.zeros(Ngrid) # relative water level due to storm surge. Assumed 0 far from shore.
WLcalc = np.ones((Nhdays,2)) # water level index, relative water level

jglobal = -1  # % global index on final calculations of water level, breaking height etc.
for j in np.arange(0,Nhdays): # iterates through number of timesteps in average array
#for j in np.arange(0,1):
	#print(j)
	jglobal = jglobal + 1
	for i in np.arange(0,Ngrid):  # iterates through number of spacesteps in transect array. NOTE: i=0 is the FURTHEST position offshore.
	#for i in np.arange(0,1):
		#print(i)
		h[i] = depthHR[i]
		V[i] = 0.5
		Re = np.abs(V[i]*4*h[i]/1e-6)
		ff = 0.25/(np.log10(RO/(3.7*4*h[i])+5.74/Re**.9))**2 # friction factor
		V[i] = (np.abs(Tsy[j])*8/(1030*ff))**.5*Tsy[j]/np.abs(Tsy[j])
		#print(V)
		Re = np.abs(V[i]*4*h[i]/1e-6) 
		ff = .25/(np.log10(RO/(3.7*4*h[i])+5.74/Re**.9))**2
		V[i] = (np.abs(Tsy[j])*8/(1030*ff))**.5*Tsy[j]/np.abs(Tsy[j])
		#print(V)
		Re = np.abs(V[i]*4*h[i]/1e-6)
		ff=.25/(np.log10(RO/(3.7*4*h[i])+5.74/Re**.9))**2
		V[i]=(np.abs(Tsy[j])*8/(1030*ff))**.5*Tsy[j]/np.abs(Tsy[j]) # final V[i] that will be used to calculate water level in the equations of motion. 
		#print(V)

	# Determine water level as function of x, cross-shore position
	for i in np.arange(0,Ngrid-1):
		aida[i+1] = aida[i] + dx*(f*V[i]/9.8 + Tsx[j]/(1030*9.8*(h[i] + aida[i])))
	WLcalc[jglobal,0] = jglobal*0.5 + 0.75
	# if Pave is nan, then make it 1013. Since it doesnt make much difference in water level, it should not cancel out what we have calculated from wind stress.
	if np.isnan(Pave[j]):
		Pave[j] = 1013
	# calculate water level
	WLcalc[jglobal,1] = aida[Ngrid-1] # + (1013-Pave[j])*100/(1030*9.8)
	#print(aida[Ngrid-1])
	#distance = 1125-WLcalc[jglobal,1]/.00267 # Determine effective distance offshore of 3 m site

# make WLcalc a dataframe
df_WLcalc = Tsy.copy() # copy to get template df of same size 
df_WLcalc[:] = WLcalc[:,1]
#df_WLcalc[:] = WLcalc[:,0]

ind_water_above_threshold = np.where(df_WLcalc>storm_surge_threshold)[0]

# initialize arrays
start_storm_index_in_total_array = np.zeros(1)
max_water_level_index_in_total_array = np.zeros(1)

# set what happens when there is no timestep that exceeded the storm surge threshold for the selected timeslice span.
if ind_water_above_threshold.shape[0] == 0:
	print('storm surge threshold not exceeded for selected timespan')
	## maybe make the below in an 'else' statement, make sure required outputs are nan if necssary
	max_water_level_values = np.nan
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
		max_water_level_in_array1 = np.nanmax(df_WLcalc[array])

		#max_water_level_in_array = np.append(max_water_level_in_array,max_water_level_in_array1)

		# find the index in the total tide gauge values array where that max water level is 
		max_water_level_index_subarray = np.where(max_water_level_in_array1==df_WLcalc[array])[0]
		# if there are two max water levels because they are the same value, take the last one
		max_water_level_index_in_total_array1 = array[max_water_level_index_subarray[-1]]

		max_water_level_index_in_total_array = np.append(max_water_level_index_in_total_array,max_water_level_index_in_total_array1).astype(int)

		# find the peak storm surge water level values 
		max_water_level_values = df_WLcalc[max_water_level_index_in_total_array]

		# find the start of the storm surge water level values
		start_storm_index_in_total_array = start_storm_index_in_total_array.astype(int)
		start_storm_surge_level = df_WLcalc[start_storm_index_in_total_array]

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

#return df_WLcalc, ws_vector_average, wd_vector_avg,max_water_level_values, max_water_level_index_in_total_array, start_storm_index_in_total_array,arrays_of_inds_consec_above_threshold,start_storm_surge_level,needed_index_between_start_and_peak, lat_site_ERA, lon_site_ERA












