import numpy as np

basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'


## Bykovsky-- Mamontovy_Khayata

# bykovsky retreat rates
# from Coastal section (CSEC) * COMMENT: Bykovsky Peninsula key site, at its Northeast coast at Mamontovy-Hayata
# https://doi.org/10.1594/PANGAEA.905519
observed_retreat_years = np.arange(1995,2019) # 1995 starts the continuous retreat rate timeseries
retreat_observed = np.array(np.ones(observed_retreat_years.shape[0]))
retreat_observed[np.where(observed_retreat_years==1995)[0]] = 5.7
retreat_observed[np.where(observed_retreat_years==1996)[0]] = 4.5
retreat_observed[np.where(observed_retreat_years==1997)[0]] = 3.8
retreat_observed[np.where(observed_retreat_years==1998)[0]] = 3.8
retreat_observed[np.where(observed_retreat_years==1999)[0]] = 3.1
retreat_observed[np.where(observed_retreat_years==2000)[0]] = 3.7
retreat_observed[np.where(observed_retreat_years==2001)[0]] = 3.9
retreat_observed[np.where(observed_retreat_years==2002)[0]] = 6.2
retreat_observed[np.where(observed_retreat_years==2003)[0]] = 6.6
retreat_observed[np.where(observed_retreat_years==2004)[0]] = 6.5
retreat_observed[np.where(observed_retreat_years==2005)[0]] = 5.2
retreat_observed[np.where(observed_retreat_years==2006)[0]] = 5.4
retreat_observed[np.where(observed_retreat_years==2007)[0]] = 6.8
retreat_observed[np.where(observed_retreat_years==2008)[0]] = 6.9
retreat_observed[np.where(observed_retreat_years==2009)[0]] = 5.0
retreat_observed[np.where(observed_retreat_years==2010)[0]] = 5.7
retreat_observed[np.where(observed_retreat_years==2011)[0]] = 11
retreat_observed[np.where(observed_retreat_years==2012)[0]] = 4.2
retreat_observed[np.where(observed_retreat_years==2013)[0]] = 2.3
retreat_observed[np.where(observed_retreat_years==2014)[0]] = 4.9
retreat_observed[np.where(observed_retreat_years==2015)[0]] = 7.4
retreat_observed[np.where(observed_retreat_years==2016)[0]] = 1.3
retreat_observed[np.where(observed_retreat_years==2017)[0]] = 3.2
retreat_observed[np.where(observed_retreat_years==2018)[0]] = 2.7

np.save(basepath + 'observed_retreat_years_Mamontovy_Khayata.npy', observed_retreat_years)
np.save(basepath + 'observed_retreat_rates_allyears_Mamontovy_Khayata.npy', retreat_observed)


## Drew Point
# from Jones et al. (2018).
retreat_observed = np.array([22.2,15.9,19.4,6.7,17.0,22.6,13.4,16.5,16.2,22.0])
observed_retreat_years = np.arange(2007,2017)

np.save(basepath + 'observed_retreat_years_Drew_Point.npy', observed_retreat_years)
np.save(basepath + 'observed_retreat_rates_allyears_Drew_Point.npy', retreat_observed)




