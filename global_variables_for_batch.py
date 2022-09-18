from pathlib import Path
import pandas as pd
import numpy as np
import os, shutil

# Use this script, global_variables_for_batch.py, to specify study site, cliff and beach parameters (with the option of a uniform distribution within a range of values for the Monte Carlo sensitivity tests), some model constants, 
# as well as paths for input/output.

community_name = 'veslobogen'
#community_name = 'Mamontovy_Khayata'
#community_name = 'Drew_Point'

# Initialize year, updated in the water_level_solver.py module
if community_name == 'Mamontovy_Khayata':
	year=1995 # the starting observed year for retreat rates for Mamontovy_Khayata is 1995

if community_name == 'Drew_Point':
	year=2007 # initial year retreat rate obs exist that you will use.

if community_name == 'veslobogen':
	year=2014

# Define start and end dates, but make sure they correspond to your available forcing data.
start_month = 'Jan'
start_day='1'
end_month = 'Dec'
end_day = '31'

## Basepaths (Change these based on your system set-up)
basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'

### paths that use basepaths above
npy_path = basepath + 'input_data/storm_surge/' + community_name + '/'

# Indicate the name of the experiment.  This name is also used to define the output pathname. UNCOMMENT the parameters below. these could be replaced with if statements.
parameters_tested_name = 'avg' # e.g. full_uncertainty , cliff_height_change, beach_thaw_depth_change, cliff_ice_content_change, cliff_angle_change are examples.

# Create a path to save this script, which has global variables defined, that has the parameters_tested_name
path_parameters_tested = basepath + 'data_io/' + community_name + '/' + parameters_tested_name + '/'
# if the path does not exist, create it
Path(path_parameters_tested).mkdir(parents=True, exist_ok=True)

path_parameters_tested_outputs_per_year = basepath + 'data_io/' + community_name + '/' + parameters_tested_name + '/' + str(year) + '/'
# if the path does not exist, create it
Path(path_parameters_tested_outputs_per_year).mkdir(parents=True, exist_ok=True)

# Save this script in the path_parameters_tested as a txt file to keep track of the parameters tested
shutil.copy('global_variables_for_batch.py', path_parameters_tested + 'global_variables_for_batch_record_of_' + parameters_tested_name + '.txt')

# initialize the experiment_name.  this will be changed based on the batch script iteration each time water_level_solver_for_batch.py is run (the passing command from bash to python is in that script).
experiment_name = parameters_tested_name + '_initial_run_number/'
datapath_io = 'initialization_placeholder' # this is updated in the water_level_solver_for_batch.py

# the path is created in the water_level_solver script also.

# set classes of input variables for easier readability. see rbeach.py for input variable description.
class cliff_parameters:
	def __init__(self, Hc, thc, Bc, Pc, vc, nic, ksc):
		self.Hc = Hc
		self.thc = thc
		self.Bc = Bc
		self.Pc = Pc
		self.vc = vc
		self.nic = nic
		self.ksc = ksc

class beach_parameters:
	def __init__(self, Wb, thb, Bb, Pb, vb, ksb):
		self.Wb = Wb
		self.thb = thb
		self.Bb = Bb
		self.Pb = Pb
		self.vb = vb
		self.ksb = ksb

# change some cliff and beach parameters based on community name
if community_name == 'Mamontovy_Khayata':
	# Specify cliff, beach, and water parameter values.
	######## cliff ########
	#                              (Hc, thc, Bc,  Pc,   vc, nic,  ksc)
	#cliff_params = cliff_parameters(10, 60, 0.2, 0.1, 0.43, 0.8, 0.002)

	# cliff_height_change experiment (Hc)
	#cliff_height = np.random.uniform(5,20) # this will only generate one number
	#cliff_params = cliff_parameters(cliff_height, 60, 0.2, 0.1, 0.43, 0.8, 0.002)

	# cliff angle change experiment (thc)
	#cliff_angle = np.random.uniform(45, 90) # ice volume per unit volume of frozen cliff sediment
	#cliff_params = cliff_parameters(10, cliff_angle, 0.2, 0.1, 0.43, 0.8, 0.002)

	# unfrozen cliff sediment thickness (Bc)
	cliff_thaw_depth = np.random.uniform(0.1,0.5)
	cliff_params = cliff_parameters(10, 60, cliff_thaw_depth, 0.1, 0.43, 0.8, 0.002)

	# cliff coarse sediment volume per unit volume unfrozen cliff sediment (Pc) (if thaw depth is small, then this does not matter 
	#cliff_coarse_fraction_unfrozen_sed = np.random.uniform(0.05, 0.2)
	#cliff_params = cliff_parameters(10, 60, 0.2, 0.1, 0.43, 0.8, 0.002)

	# cliff ice content change experiment (nic)
	#cliff_ice_content = np.random.uniform(0.6, 0.9) # ice volume per unit volume of frozen cliff sediment
	#cliff_params = cliff_parameters(10, 60, 0.2, 0.1, 0.43, cliff_ice_content, 0.002)

	# full uncertainty experiment (Hc, thc, Bc, Pc, nic)
	#cliff_params = cliff_parameters(cliff_height, cliff_angle, cliff_thaw_depth, cliff_coarse_fraction_unfrozen_sed, 0.43, cliff_ice_content, 0.002)

	##### beach #####

	#			       (Wb,    thb,   Bb,   Pb,  vb, ksb)
	beach_params = beach_parameters(0.01, 0.06,   1,   0.6, 0.6, 0.002)

if community_name == 'Drew_Point':

	# Specify cliff, beach, and water parameter values.

	####### cliff #########
	#                              (Hc, thc, Bc,  Pc,   vc, nic,  ksc)
	#cliff_params = cliff_parameters(3, 60, 0.2, 0.1, 0.43, 0.8, 0.002)

	# cliff height change experiment
	#cliff_height = np.random.uniform(1,10)
	#cliff_params = cliff_parameters(cliff_height, 60, 0.2, 0.1, 0.43, 0.8, 0.002)

	# cliff angle change experiment
	#cliff_angle = np.random.uniform(45, 90) # ice volume per unit volume of frozen cliff sediment
	#cliff_params = cliff_parameters(3, cliff_angle, 0.2, 0.1, 0.43, 0.8, 0.002)

	# unfrozen cliff sediment thickness (Bc)
	cliff_thaw_depth = np.random.uniform(0.1,0.5)
	cliff_params = cliff_parameters(3, 60, cliff_thaw_depth, 0.1, 0.43, 0.8, 0.002)

	# cliff coarse sediment volume per unit volume unfrozen cliff sediment (Pc) (if thaw depth is small, then this does not matter 
	#cliff_coarse_fraction_unfrozen_sed = np.random.uniform(0.05, 0.2)
	#cliff_params = cliff_parameters(3, 60, 0.2, cliff_coarse_fraction_unfrozen_sed, 0.43, 0.8, 0.002)

	# cliff ice content change experiment
	#cliff_ice_content = np.random.uniform(0.6, 0.9) # ice volume per unit volume of frozen cliff sediment
	#cliff_params = cliff_parameters(3, 60, 0.2, 0.1, 0.43, cliff_ice_content, 0.002)

	# full uncertainty
	#cliff_params = cliff_parameters(cliff_height, cliff_angle, cliff_thaw_depth, cliff_coarse_fraction_unfrozen_sed, 0.43, cliff_ice_content, 0.002)

	######## beach ##########
	#			       (Wb,    thb,   Bb,   Pb,  vb, ksb)
	beach_params = beach_parameters(0.01, 0.06,   1,   0.6, 0.6, 0.002)


if community_name == 'veslobogen':

	# Specify cliff, beach, and water parameter values.

	####### cliff #########
	#                              (Hc, thc, Bc,  Pc,   vc, nic,  ksc)
	cliff_params = cliff_parameters(8, 60, 0.2, 0.1, 0.43, 0.01, 0.002)

	# cliff height change experiment
	#cliff_height = np.random.uniform(1,10)
	#cliff_params = cliff_parameters(cliff_height, 60, 0.2, 0.1, 0.43, 0.8, 0.002)

	# cliff angle change experiment
	#cliff_angle = np.random.uniform(45, 90) # ice volume per unit volume of frozen cliff sediment
	#cliff_params = cliff_parameters(3, cliff_angle, 0.2, 0.1, 0.43, 0.8, 0.002)

	# unfrozen cliff sediment thickness (Bc)
	#cliff_thaw_depth = np.random.uniform(0.0000001,0.00000005)
	#cliff_params = cliff_parameters(3, 60, cliff_thaw_depth, 0.1, 0.43, 0.8, 0.002)

	# cliff coarse sediment volume per unit volume unfrozen cliff sediment (Pc) (if thaw depth is small, then this does not matter 
	#cliff_coarse_fraction_unfrozen_sed = np.random.uniform(0.05, 0.2)
	#cliff_params = cliff_parameters(3, 60, 0.2, cliff_coarse_fraction_unfrozen_sed, 0.43, 0.8, 0.002)

	# cliff ice content change experiment
	#cliff_ice_content = np.random.uniform(0.6, 0.9) # ice volume per unit volume of frozen cliff sediment
	#cliff_params = cliff_parameters(3, 60, 0.2, 0.1, 0.43, cliff_ice_content, 0.002)

	# full uncertainty
	#cliff_params = cliff_parameters(cliff_height, cliff_angle, cliff_thaw_depth, cliff_coarse_fraction_unfrozen_sed, 0.43, cliff_ice_content, 0.002)

	######## beach ##########
	#			       (Wb,    thb,   Bb,   Pb,  vb, ksb)
	beach_params = beach_parameters(0.01, 0.06,   1,   0.6, 0.6, 0.002)

## Load in the masked arrays (based on sea ice concentration threshold), interpolated to hourly
water_level_meters = pd.read_pickle(npy_path + 'ERAI_forced_water_levels_masked/hourly/water_levels_' + community_name + '_masked_' + str(year) + '_hourly.pkl')

## load the wave height and period, sst. Interpolated to hourly
df_hourly = pd.read_pickle(npy_path + 'ERAI_forcing_variables_masked/hourly/ERAI_forcings_' + community_name + '_masked_' + str(year) + '_hourly.pkl')
Hrms = df_hourly.swh_site # this is an example of a call for the columns in the pandas dataframe df_hourly
Tr = df_hourly.wave_period_site
Tw = df_hourly.sst_site - 273.15 # convert K to deg C

# Profile empirical parameters, where A is more established as 0.1, but calibrated together with alpha. Determines the rate of potnetial cross-shore sediment transport.
A = 0.1 # empirical parameter for equilibrium beach profile
alpha = 0.001 # alpha = empirical parameter for sediment transport rate parameter (decreasing alpha decreases potential sediment transport supplied by beach to ocean)
adjust = 0.1 # heat adjustment parameter (given in Kob. '99)

Sw = 15.0 # seawater salinity
gammab = 0.4 # empirical breaking parameter
beta = 1.0 # wave runup parameter (1.0) given in Kobayashi et al. (1999) for a storm !!! this might need to be adjusted using an empirical formula with wave height as an input for example.
