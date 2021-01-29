# Readme file for ArcticBeach.

Code for Rolph et al., 'Towards A physics-based parameterization of pan-Arctic erosion'

#### Set up initial model parameters
Use global_variables_for_batch.py to specify study site, cliff and beach parameters (with the option of a
uniform distribution within a range of values for the Monte Carlo sensitivity tests), some model constants,
as well as paths for input/output.

#### Run the water offset solver and calculate the retreat rates
batch_monte_carlo.sh is the slurm batch script used to run the 
water_level_solver_for_batch.py, described below.

water_level_solver_for_batch.py simultaneously calculates the required water level offset as described in
the paper, and then uses this offset to simulate the retreat rates.  Once it calculates the water level
offset, it calls the rbeach_for_batch.py, which is the main erosion model.

##  Preparing forcing data

- ERA_interim_read_with_wave_and_sst.py:
        - Reads the ERA-Interim reanalysis data
        - Calculates the vector averaged wind speed and directions
        - Averages the forcing to 3 hourly means if not supplied in that time resolution already.
        - Outputs a dataframe of forcing data, latitude and longitude of the grid cell the reanalysis data
was taken from

## Use the plotting scripts to identify the offshore grid cell for each community
- Mamontovy Khayata:  make_mask_with_extra_figures_single_year_sicn_threshold_ERAI_bykovsky_mamontovy_khayata.py
- Drew Point:  make_mask_with_extra_figures_single_year_sicn_threshold_ERAI_drew_point.py

## Generate and/or load a bathymetry file for use in the storm surge model.
- bathymetry_produce_for_drew_point_and_bykovsky.py

## Calculate the unmasked water levels at the offshore grid cell you selected from the plotting script above.
## Both of these scripts below call storm_surge_ERA_Interim.py, which is the storm surge model.
- Mamontovy Khayata: calculate_modelled_water_levels_bykovsky.py
- Drew Point: save_measured_vs_modelled_water_level_drew_point_or_prudhoe.py
- Prudhoe Bay: save_measured_vs_modelled_water_level_prudhoe_only.py

## Create and apply a mask at timesteps when sea ice concentration is greater than 15%. Save the masked
## arrays that will be fed into the erosion model.
- make_mask_based_on_seaice_threshold_ERAI_input_community.py

## Interpolate into hourly timesteps to use for input into the model
- interpolate_masked_ERAIdata.py

##### Validation data #############

## For model tuning and validation of retreat rates, create a file of observed retreat rates.
- save_observed_retreat_rates.py

#### Plotting scripts #########
plotting_scripts/*, including an own readme file in that path.
