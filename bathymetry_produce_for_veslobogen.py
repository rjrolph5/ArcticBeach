# -*- coding: utf-8 -*-
"""
Created on Sunday 14 Aug 2022

Produce bathymetry NumPy arrays for use in the bathystrophic storm surge model.

rjrolph@alaska.edu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set basepath
basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/'
# set path to save npy bathymetry output file.
ofilepath = basepath + 'input_data/storm_surge/'


# From the publication Moskalik et al (2014), Oceanologia, the bathymetry from the 
# Veslobogen cliffs roughly goes from 0 to 110m depth across 5km.  The coastline 
# angle there is roughly east to west, with the shoreline facing south (if you are 
# standing onshore looking toward the ocean, you would be looking south).  Looking 
# at some sources further out, directly south does not get too much deeper.
depth_veslobogen = np.zeros(127)
for i in np.arange(1,depth_veslobogen.shape[0]-1):
	depth_veslobogen[i-1] = 110/127 * i

np.save(ofilepath + 'veslobogen/depth_veslobogen.npy', depth_veslobogen)

depth_veslobogen = np.flip(depth_veslobogen[:-2])


# Mamontovy Khayata for comparison
depth_bykovsky = np.zeros(127) # number of data points from coast to offshore
for i in np.arange(1,depth_bykovsky.shape[0]-1):
        depth_bykovsky[i-1] = 80/126 * i

depth_bykovsky = np.flip(depth_bykovsky[:-2])

# the numpy file needs to be saved from offshore inwards, so the greatest depth has
# index 0.  however, for plotting purposes, we will reverse this so index 0 is the
# beach depth.

fig, ax = plt.subplots()
plt.title('Bathymetry input used to calculate storm surge')
ax.plot(np.flip(depth_veslobogen),'*', label='Veslobogen')
ax.plot(np.flip(depth_bykovsky),'o', label='Bykovsky')
ax.set_xlabel('Distance from beach [km]' )
ax.set_ylabel('Water depth [m]' )
#plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.legend()
plt.show()
