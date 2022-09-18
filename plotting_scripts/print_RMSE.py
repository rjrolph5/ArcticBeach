

import numpy as np



basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
plot_path = basepath + 'plots/'

#community_name = 'Mamontovy_Khayata'
#community_name = 'Drew_Point'
community_name = 'veslobogen'

###### load the modelled retreat rates from water_level_solver
modelled_retreat = np.load(basepath + 'R_all_modelled_using_median_calcd_water_level_' + community_name + '.npy')

###### load observed retreat
observed_retreat_years = np.load(basepath + 'input_data/observed_retreat_years_' + community_name + '.npy')
retreat_observed = np.load(basepath + 'input_data/observed_retreat_rates_allyears_' + community_name + '.npy')




def rmse(observed, modelled):
        rmse = np.sqrt(np.average((observed - modelled)**2))
        return rmse


print(rmse(retreat_observed, modelled_retreat))
