# This script uses a solver to find a numerical solution for the tuning parameter called 'water_level_offset', in order to match (within tolerance) of the observed retreat rate. This is a calibration script.  It also has the 
# user-defined option to set the tuning parameter to the median of the annually-calculated values, and then run the erosion model using that median value.  This approach will produce a timeseries of retreat rates. Note: You must 
# update global_variables.py before running this script to the study site and parameters you would like to evaulate retreat rates for.

import sys
import global_variables_for_batch as global_variables
from scipy.optimize import fsolve # numerical python solver
import numpy as np
import rbeach_for_batch as rbeach
import pandas as pd
from pathlib import Path

use_median_water_level_offset = 'True'
calc_reqrd_water_level = 'False'


##### Read the location of the study site from the global_variables_for_batch file and assign the relevant years where observations were made at that study site.
if global_variables.community_name == 'Drew_Point':
	# Drew Point
	year_init = 2007
	year_init_for_median = 2007
	year_final = 2017  # up to (not including) year end
	year_final_for_median = 2017 # keep the year range constant when comparing median water levels

if global_variables.community_name == 'Mamontovy_Khayata':
	# Mamontovy
	year_init = 1995
	year_init_for_median = 1995
	year_final = 2019 # up to (not including) year end
	year_final_for_median = 2019

##### This indicates if you would like to run the erosion model using the median of the cailibrated water level offsets. This corresponds to the default water level offset values used for the default run when comparing Monte 
##### Carlo sensitivity tests.
wl_all = np.array([1]) if use_median_water_level_offset == 'True':
	# find median water level
	for yr in np.arange(year_init_for_median, year_final_for_median):
		# specify the paths the calculated water level offset should be saved in,  directory by experiment, filename by year.
		water_level_offset_path = global_variables.npy_path + 'required_offset_to_match_observed_retreat/avg/'
		wl = np.load(water_level_offset_path + 'water_offset_reqrd_to_match_obs_' + str(yr) + '.npy')
		wl_all = np.append(wl_all,wl)

	wl_all = wl_all[1:]
	median_wl = np.median(wl_all)
	water_offset_required = median_wl
	np.save('median_wl_' + global_variables.community_name +'.npy', median_wl)

##### Provide an initial guess for the solver
initial_guess_for_water_level_offset = 0.2

##### Load the observed retreat rates for the given community
# These numpy files of observed retreat rates are generated from save_observed_retreat_rates.py
observed_retreat_years = np.load('/home/rrolph/erosion_model/input_data/observed_retreat_years_' + global_variables.community_name + '.npy')
retreat_observed = np.load('/home/rrolph/erosion_model/input_data/observed_retreat_rates_allyears_' + global_variables.community_name + '.npy')

def read_observed_erosion_rates(year):
	#global observed_retreat_years
	#global retreat_observed
	retreat_observed_year_selected = retreat_observed[int(np.where(observed_retreat_years==year)[0])]
	return retreat_observed_year_selected

def modelled_minus_observed_retreat(water_level_offset): # the input is a single value guess of water_level_offset
	global_variables.water_level_offset = water_level_offset # this updates the global variable. in fsolve, this is resolved to a closer value that solves the equation than the input argument guess. the global variable water_level_offset is rewritten in each iteration fsolve uses.
	observed_retreat_current_year = read_observed_erosion_rates(global_variables.year)
	return rbeach.erosion_main(global_variables.water_level_offset)- observed_retreat_current_year # uses the most updated value of water_level_offset to calculate the modelled retreat rate, then subtracts it from the obs to give a final output of modelled minus obs retreat rate.

def rbeach_erosion_main(water_level_offset):
	global_variables.water_level_offset = water_level_offset
	return rbeach.erosion_main(global_variables.water_level_offset)


##### Assign paths and variables from global to pass to erosion model.
R_all_modelled = np.ones(1)
for year in np.arange(year_init, year_final):
	global_variables.year = year # This updates the global variable 'year', and this is important when modelled_minus_observed_retreat is called since it calls global_variables.observed_retreat_current_year
	print(global_variables.year)

	## Load in the masked forcing arrays (based on sea ice concentration threshold), interpolated to hourly
	global_variables.water_level_meters = pd.read_pickle(global_variables.npy_path + 'ERAI_forced_water_levels_masked/hourly/water_levels_' + global_variables.community_name + '_masked_' + str(year) + '_hourly.pkl')

	## Update the path for model outputs, different paths for each year of model output.
	global_variables.path_parameters_tested_outputs_per_year = global_variables.basepath + 'data_io/' + global_variables.community_name + '/' + global_variables.parameters_tested_name + '/' + str(year) + '/'
	Path(global_variables.path_parameters_tested_outputs_per_year).mkdir(parents=True, exist_ok=True)

	# load the wave height and period, sst. Interpolated to hourly
	global_variables.df_hourly = pd.read_pickle(global_variables.npy_path + 'ERAI_forcing_variables_masked/hourly/ERAI_forcings_' + global_variables.community_name + '_masked_' + str(global_variables.year) + '_hourly.pkl')
	global_variables.Hrms = global_variables.df_hourly.swh_site # this is an example of a call for the columns in the pandas dataframe df_hourly
	global_variables.Tr = global_variables.df_hourly.wave_period_site
	global_variables.Tw = global_variables.df_hourly.sst_site - 273.15 # convert K to deg C

	# the first argument in fsolve (a function) should be equal to 0 within the tolerance (i.e. 'solved')
	#print(global_variables.water_level_offset) # print the updated water level solution.

	# rbeach is called by global_variables.water_level_offset. You subtract the observed_retreat_current_year because you are trying to calculate the water level required such that modeled_retreat = obs_retreat, e.g. the function is modelled_retreat - obs_retreat = 0.
	if calc_reqrd_water_level == 'True':
		water_offset_required = fsolve(rbeach_erosion_main, initial_guess_for_water_level_offset) # water_level_offset is the unknown, but provided an initial guess
		np.save(water_level_offset_path + 'water_offset_reqrd_to_match_obs_' + str(global_variables.year) + '.npy', water_offset_required)

	if use_median_water_level_offset == 'True': # here is the option to run the model using the median of pre-calculated required water offset values.
		# Since rbeach returns modelled-observed, to get the modelled retreat, you need to add back the observed:
		R_modelled_current_loop_year = rbeach.erosion_main(water_offset_required) + read_observed_erosion_rates(global_variables.year)
		R_all_modelled = np.append(R_all_modelled, R_modelled_current_loop_year)

R_all_modelled = R_all_modelled[1:]

task_id = sys.argv[1] # argv[1:]  # this corresponds to run number of monte carlo iteration. 

global_variables.experiment_name = str(task_id) # this is a STRING of of the run number from the batch script and subsequently passed to this script.

# update the datapath with the iteration experiment name
global_variables.datapath_io = global_variables.path_parameters_tested

##### save the modelled retreat rate array
# make the directory if it does not exist.
#Path(global_variables.datapath_io).mkdir(parents=True, exist_ok=True)
np.save(global_variables.path_parameters_tested + '/R_all_modelled_' + global_variables.experiment_name + '.npy', R_all_modelled)


'''
# plot the observed vs modelled retreat rates.


import matplotlib.pyplot as plt


x = observed_retreat_years  # the label locations

fig, ax = plt.subplots()
if use_median_water_level_offset == 'True':
	plt.title('Cliff retreat using calculated median water level: ' + str(round(water_offset_required,2)) + 'm, ' + global_variables.community_name)

width = 0.35

ax.bar(observed_retreat_years - width/2 ,R_all_modelled, width, label = 'Modelled retreat')
ax.bar(observed_retreat_years+ width/2, retreat_observed, width, label = 'Observed retreat')

ax.set_ylabel('Retreat [m]')
ax.set_xticks(x)
ax.tick_params('x',rotation=90,pad=1)
plt.legend()

plt.savefig('/home/rrolph/erosion_model_output_figures_too_large_for_github/modelled_vs_observed_retreat_using_median_reqrd_wl' + str(global_variables.community_name) + '.png', bbox_inches = 'tight')
plt.show()
'''





















