import numpy as np
import matplotlib.pyplot as plt

basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
plot_path = basepath + 'plots/'

#community_name = 'Mamontovy_Khayata'
community_name = 'Drew_Point'

###### load the modelled retreat rates from water_level_solver
modelled_retreat = np.load(basepath + 'R_all_modelled_using_median_calcd_water_level_' + community_name + '.npy')

###### load observed retreat
observed_retreat_years = np.load(basepath + 'input_data/observed_retreat_years_' + community_name + '.npy')
retreat_observed = np.load(basepath + 'input_data/observed_retreat_rates_allyears_' + community_name + '.npy')

##### plot the cumulative modelled retreat
fig, ax = plt.subplots(figsize=(3,4))
if community_name == 'Mamontovy_Khayata':
	fig, ax = plt.subplots(figsize=(6,4))

#plt.title('Cumulative cliff retreat using calculated \n median water level: ' + community_name)

width = 0.35

ax.bar(observed_retreat_years - width/2, np.cumsum(modelled_retreat), width, label = 'Modelled retreat', color='blue')
ax.bar(observed_retreat_years+ width/2, np.cumsum(retreat_observed), width, label = 'Observed retreat', color = 'orange')

ax.set_ylabel('Cumulative retreat [m]', fontsize=20)

x = observed_retreat_years  # the label locations

ax.set_xticks(x)

plt.xticks(np.arange(min(x), max(x)+1, 2.0))


ax.tick_params('x',rotation=90,pad=1, labelsize=15)
ax.set_yticks(np.arange(0,176,25))
ax.tick_params('y', pad=1, labelsize=15)
fig.tight_layout()
plt.savefig(plot_path + 'Cumulative_modelled_vs_observed_retreat_using_median_reqrd_wl' + str(community_name) + '.png', bbox_inches = 'tight', dpi=300)
plt.show()


##### plot the actual retreat rates
#fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(3,4))
if community_name == 'Mamontovy_Khayata':
        fig, ax = plt.subplots(figsize=(6,4))


#plt.title('Cliff retreat using calculated \n median water level: ' + community_name)

width = 0.35

ax.bar(observed_retreat_years - width/2, modelled_retreat, width, label = 'Modelled retreat', color='blue')
ax.bar(observed_retreat_years+ width/2, retreat_observed, width, label = 'Observed retreat', color='orange')

ax.set_ylabel('Retreat [m]',fontsize=20)

x = observed_retreat_years  # the label locations

if community_name == 'Mamontovy_Khayata':
	plt.legend(loc='upper left', prop={'size': 12})

ax.set_xticks(x)

#if community_name == 'Mamontovy_Khayata':
plt.xticks(np.arange(min(x), max(x)+1, 2.0))

ax.tick_params('x',rotation=90,pad=1, labelsize=15)
ax.tick_params('y', pad=1, labelsize=15)
ax.set_ylim([0,32])
#plt.legend()
fig.tight_layout()
plt.savefig(plot_path + 'modelled_vs_observed_retreat_using_median_reqrd_wl' + str(community_name) + '.png', bbox_inches = 'tight', dpi=300)
plt.show()

##### add an error bar based on cliff height change #########################
npy_filepath_cliff_height_change = basepath + 'data_io/' + community_name + '/cliff_height_change/'


# load the range of retreat rates from the uniform distribution cliff height changes (500 realizations)
retreat_2d_array = np.empty((500,observed_retreat_years.size))

for i in np.arange(1,501):
	ifile = npy_filepath_cliff_height_change + 'R_all_modelled_' + str(i) + '.npy'
	print(ifile)
	retreat_timeseries = np.load(ifile)
	retreat_2d_array[i-1,:] = retreat_timeseries


##### plot the box plot
fig, ax = plt.subplots()

ax.boxplot(retreat_2d_array,  notch=True, positions=observed_retreat_years)
ax.scatter(observed_retreat_years, retreat_observed, color='orange', label='Observed')
ax.scatter(observed_retreat_years, modelled_retreat, color='b', marker='*', label='Modelled using \nfixed parameters')

ax.set_ylim([0,61])

year_xlabels = observed_retreat_years.astype(str)

ax.set_xticklabels(year_xlabels, rotation=90)
ax.tick_params('x', pad=1, labelsize=14)
ax.tick_params('y', pad=1, labelsize=14)

#ax.set_title('Retreat showing sensitivity to cliff height change for ' + community_name)
ax.set_ylabel('Retreat [m]', fontsize=25)

plt.legend(loc='upper left')

# save to the same path that the global varaibles for the experiment is in
path_parameters_tested = basepath + 'data_io/' + community_name + '/cliff_height_change/'
plt.savefig(path_parameters_tested + 'modelled_vs_observed_retreat_using_median_reqrd_wl' + community_name + '_cliff_height_change.png', bbox_inches = 'tight')
plt.show()


##### add an error bar based on cliff coarse sediment fraction change #########################
npy_filepath_coarse_sed_frac_change = basepath + 'data_io/' + community_name + '/coarse_sediment_fraction_unfrozen_cliff/'


# load the range of retreat rates from the uniform distribution cliff height changes (500 realizations)
retreat_2d_array = np.empty((500,observed_retreat_years.size))

for i in np.arange(1,501):
	ifile = npy_filepath_coarse_sed_frac_change + 'R_all_modelled_' + str(i) + '.npy'
	print(ifile)
	retreat_timeseries = np.load(ifile)
	retreat_2d_array[i-1,:] = retreat_timeseries

##### plot the box plot for coarse sediment fraciton in cliff
fig, ax = plt.subplots()

ax.boxplot(retreat_2d_array,  notch=True, positions=observed_retreat_years)
ax.scatter(observed_retreat_years, retreat_observed, color='orange', label='Observed')
ax.scatter(observed_retreat_years, modelled_retreat, color='b', marker='*', label='Modelled using \nfixed parameters')

ax.set_ylim([0,61])

year_xlabels = observed_retreat_years.astype(str)

ax.set_xticklabels(year_xlabels, rotation=90)
ax.tick_params('x', pad=1, labelsize=14)
ax.tick_params('y', pad=1, labelsize=14)

#ax.set_title('Retreat showing sensitivity to cliff height change for ' + community_name)
ax.set_ylabel('Retreat [m]', fontsize=25)

plt.legend(loc='upper left')

# save to the same path that the global varaibles for the experiment is in
path_parameters_tested = basepath + 'data_io/' + community_name + '/coarse_sediment_fraction_unfrozen_cliff/'
plt.savefig(path_parameters_tested + 'modelled_vs_observed_retreat_using_median_reqrd_wl' + community_name + '_cliff_unfrozen_cliff_sed_fraction.png', bbox_inches = 'tight')
plt.show()


##### add an error bar based on changing beach thaw depth ################
npy_filepath_beach_thaw_depth_change = basepath + 'data_io/' + community_name + '/beach_thaw_depth_change/'


# load the range of retreat rates from the uniform distribution cliff height changes (500 realizations)
retreat_2d_array = np.empty((500,observed_retreat_years.size))

for i in np.arange(1,501):
	ifile = npy_filepath_beach_thaw_depth_change + 'R_all_modelled_' + str(i) + '.npy'
	print(ifile)
	retreat_timeseries = np.load(ifile)
	retreat_2d_array[i-1,:] = retreat_timeseries


##### plot the box plot
fig, ax = plt.subplots()

ax.boxplot(retreat_2d_array, notch=True, positions=observed_retreat_years)
ax.scatter(observed_retreat_years, retreat_observed, color='orange', label='Observed')
ax.scatter(observed_retreat_years, modelled_retreat, color='b', marker='*', label='Modelled using \nfixed parameters')


year_xlabels = observed_retreat_years.astype(str)

ax.set_xticklabels(year_xlabels, rotation=90)
ax.tick_params('x', pad=1, labelsize=14)
ax.tick_params('y', pad=1, labelsize=14)

#ax.set_title('Retreat showing sensitivity to beach thaw depth for ' + community_name)
ax.set_ylabel('Retreat [m]', fontsize=25)

plt.legend(loc='upper left')

# save to the same path that the global varaibles for the experiment is in
path_parameters_tested = basepath + 'data_io/' + community_name + '/beach_thaw_depth_change/'
plt.savefig(path_parameters_tested + 'modelled_vs_observed_retreat_using_median_reqrd_wl' + community_name + '_beach_thaw_depth_change.png', bbox_inches = 'tight')
plt.show()


##### add an error bar based on changing cliff thaw depth ################
npy_filepath_cliff_thaw_depth_change = basepath + 'data_io/' + community_name + '/cliff_thaw_depth/'


# load the range of retreat rates from the uniform distribution cliff height changes (500 realizations)
retreat_2d_array = np.empty((500,observed_retreat_years.size))

for i in np.arange(1,501):
	ifile = npy_filepath_cliff_thaw_depth_change + 'R_all_modelled_' + str(i) + '.npy'
	print(ifile)
	retreat_timeseries = np.load(ifile)
	retreat_2d_array[i-1,:] = retreat_timeseries


##### plot the box plot
fig, ax = plt.subplots()

ax.boxplot(retreat_2d_array, notch=True, positions=observed_retreat_years)
ax.scatter(observed_retreat_years, retreat_observed, color='orange', label='Observed')
ax.scatter(observed_retreat_years, modelled_retreat, color='b', marker='*', label='Modelled using \nfixed parameters')


year_xlabels = observed_retreat_years.astype(str)

ax.set_xticklabels(year_xlabels, rotation=90)
ax.tick_params('x', pad=1, labelsize=14)
ax.tick_params('y', pad=1, labelsize=14)

#ax.set_title('Retreat showing sensitivity to cliff thaw depth for ' + community_name)
ax.set_ylabel('Retreat [m]', fontsize=25)

plt.legend(loc='upper left')

# save to the same path that the global varaibles for the experiment is in
path_parameters_tested = basepath + 'data_io/' + community_name + '/cliff_thaw_depth/'
plt.savefig(path_parameters_tested + 'modelled_vs_observed_retreat_using_median_reqrd_wl' + community_name + '_cliff_thaw_depth_change.png', bbox_inches = 'tight')
plt.show()


##### add an error bar based on changing cliff ice content ################
npy_filepath_cliff_ice_content_change = basepath + 'data_io/' + community_name + '/cliff_ice_content_change/'


# load the range of retreat rates from the uniform distribution cliff height changes (500 realizations)
retreat_2d_array = np.empty((500,observed_retreat_years.size))

for i in np.arange(1,501):
	ifile = npy_filepath_cliff_ice_content_change + 'R_all_modelled_' + str(i) + '.npy'
	print(ifile)
	retreat_timeseries = np.load(ifile)
	retreat_2d_array[i-1,:] = retreat_timeseries


##### plot the box plot
fig, ax = plt.subplots()

ax.boxplot(retreat_2d_array, notch=True, positions=observed_retreat_years)
ax.scatter(observed_retreat_years, retreat_observed, color='orange', label='Observed')
ax.scatter(observed_retreat_years, modelled_retreat, color='b', marker='*', label='Modelled using \nfixed parameters')


year_xlabels = observed_retreat_years.astype(str)

ax.set_xticklabels(year_xlabels, rotation=90)

plt.legend(loc='upper left')

#ax.set_title('Retreat showing sensitivity to cliff ice content for ' + community_name)
ax.set_ylabel('Retreat [m]', fontsize=25)
ax.set_ylim([0,41])

ax.tick_params('x',rotation=90,pad=1, labelsize=14)
ax.tick_params('y', pad=1, labelsize=14)


# save to the same path that the global varaibles for the experiment is in
path_parameters_tested = basepath + 'data_io/' + community_name + '/cliff_ice_content_change/'
plt.savefig(path_parameters_tested + 'modelled_vs_observed_retreat_using_median_reqrd_wl' + community_name + '_cliff_ice_content_change.png', bbox_inches = 'tight')
plt.show()



##### add an error bar based on changing cliff angle ################
npy_filepath_cliff_angle_change = basepath + 'data_io/' + community_name + '/cliff_angle_change/'

# load the range of retreat rates from the uniform distribution cliff angle changes (500 realizations)
retreat_2d_array = np.empty((500,observed_retreat_years.size))

for i in np.arange(1,501):
	ifile = npy_filepath_cliff_angle_change + 'R_all_modelled_' + str(i) + '.npy'
	print(ifile)
	retreat_timeseries = np.load(ifile)
	retreat_2d_array[i-1,:] = retreat_timeseries


##### plot the box plot
fig, ax = plt.subplots()

ax.boxplot(retreat_2d_array, notch=True, positions=observed_retreat_years)
ax.scatter(observed_retreat_years, retreat_observed, color='orange', label='Observed')
ax.scatter(observed_retreat_years, modelled_retreat, color='b', marker='*', label='Modelled using fixed parameters')


year_xlabels = observed_retreat_years.astype(str)

ax.set_xticklabels(year_xlabels, rotation=90)

#ax.set_title('Retreat showing sensitivity to cliff angle for ' + community_name)
ax.set_ylabel('Retreat [m]', fontsize=25)
ax.set_yticks(np.arange(0,39,5))
ax.set_ylim([0,38])

ax.tick_params('x',rotation=90, pad=1, labelsize=14)
ax.tick_params('y', pad=1, labelsize=14)

plt.legend(loc='upper left')

# save to the same path that the global varaibles for the experiment is in
path_parameters_tested = basepath + 'data_io/' + community_name + '/cliff_angle_change/'
plt.savefig(path_parameters_tested + 'modelled_vs_observed_retreat_using_median_reqrd_wl' + community_name + '_cliff_angle_change.png', bbox_inches = 'tight')
plt.show()


##### add an error bar based on full uncertainty ################
npy_filepath_full_uncertainty = basepath + 'data_io/' + community_name + '/full_uncertainty/'

# load the range of retreat rates from the full uncertainty (500 realizations)
retreat_2d_array = np.empty((500,observed_retreat_years.size))

for i in np.arange(1,501):
	ifile = npy_filepath_full_uncertainty + 'R_all_modelled_' + str(i) + '.npy'
	print(ifile)
	retreat_timeseries = np.load(ifile)
	retreat_2d_array[i-1,:] = retreat_timeseries


##### plot the box plot
fig, ax = plt.subplots()

ax.boxplot(retreat_2d_array, notch=True, positions=observed_retreat_years) # bool value can also be 1 for notch true/false
ax.scatter(observed_retreat_years, retreat_observed, color='orange', label='Observed')
ax.scatter(observed_retreat_years, modelled_retreat, color='b', marker='*', label='Modelled using \nfixed parameters')

year_xlabels = observed_retreat_years.astype(str)

ax.set_xticklabels(year_xlabels, rotation=90)

#ax.set_title('Retreat showing sensitivity to cliff angle for ' + community_name)
ax.set_ylabel('Retreat [m]', fontsize=25)
ax.set_ylim(0,79)
ax.set_yticks(np.arange(0,79,10))

ax.tick_params('x',rotation=90, pad=1, labelsize=14)
ax.tick_params('y', pad=1, labelsize=14)

plt.legend(loc='upper left')

# save to the same path that the global varaibles for the experiment is in
path_parameters_tested = basepath + 'data_io/' + community_name + '/full_uncertainty/'
plt.savefig(path_parameters_tested + 'modelled_vs_observed_retreat_using_median_reqrd_wl' + community_name + '_full_uncertainty.png', bbox_inches = 'tight')
plt.show()


