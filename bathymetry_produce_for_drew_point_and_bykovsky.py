# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:58:04 2020

Produce the bathymetry for Drew Point , used in the storm surge module. 

@author: rrolph
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# increase resolution of bathymetry. commented out after loop is run and depthHR is saved to a file which is loaded at the top of the script. 

bathyfile = '/home/rrolph/erosion_model/depth.txt'

depth = pd.read_csv(bathyfile,header=None)
depthHR =  np.zeros(126)
k = -1
for i in np.arange(0,depth.shape[0]-1):
    k = k + 1
    depthHR[k] = depth.iloc[i]
    k = k + 1
    depthHR[k] = 0.8*depth.iloc[i] + 0.2*depth.iloc[i+1]
    k = k + 1
    depthHR[k] = .6*depth.iloc[i]+.4*depth.iloc[i+1]
    k = k + 1
    depthHR[k] = .4*depth.iloc[i]+.6*depth.iloc[i+1]
    k = k + 1
    depthHR[k] = .2*depth.iloc[i]+.8*depth.iloc[i+1]

depthHR = depthHR[:-1]
# save 
np.save('/home/rrolph/erosion_model/input_data/storm_surge/Drew_Point/depthHR_drewPoint.npy',depthHR)

# bykovsky
depth_bykovsky = np.zeros(127)
for i in np.arange(1,depth_bykovsky.shape[0]-1):
	depth_bykovsky[i-1] = 80/126 * i

depth_bykovsky = np.flip(depth_bykovsky[:-2])

np.save('/home/rrolph/erosion_model/input_data/storm_surge/Mamontovy_Khayata/depth_bykovsky.npy',depth_bykovsky)

# the numpy file needs to be saved from offshore inwards, so the greatest depth has index 0.  
# however, for plotting purposes, we will reverse this so index 0 is the beach depth. 

fig, ax = plt.subplots()
plt.title('Bathymetry input used to calculate storm surge')
ax.plot(np.flip(depthHR),'*', label='Drew point')
ax.plot(np.flip(depth_bykovsky),'o', label='Bykovsky')
ax.set_xlabel('Distance from beach [km]' )
ax.set_ylabel('Water depth [m]' )
#plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.legend()
plt.show()


