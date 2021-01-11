
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import cftime
from datetime import datetime
import math
import matplotlib.pyplot as plt
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#import cartopy.crs as ccrs
#import cartopy
import cmocean # Colormaps for sea ice from Kristen M. Thyng, Chad A. Greene, Robert D. Hetland, Heather M. Zimmerle, and Steven F. DiMarco (2016). True colors of oceanography: Guidelines for effective and accurate colormap selection. Oceanography, 29(3), 10. doi:10.5670/oceanog.2016.66
#from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap, addcyclic

# This is a plotting script to identify the offshore grid cell for each community

# At the moment, there are separate scripts for each study site but they can easily be combined into one script with a user-input community name and lat lon.

###### set paths ######

# path where you save the lat lon and index information of the grid cells selected in this script.
#basepath = '/home/rrolph/erosion_model/input_data/storm_surge/'
basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/input_data/storm_surge/'

# path where the maps generated in this script are saved
#plot_basepath = '/home/rrolph/erosion_model_output_figures_too_large_for_github/'
plot_basepath = '/permarisk/output/becca_erosion_model/ArcticBeach/plots/'

# Reanalysis data basepath
ncfile_path = '/permarisk/data/ERA_Data/ERAint_Arctic/' # for pls11. Change path to wherever your reanalysis data is stored.

######################

def geo_idx(dd, dd_array):
	"""
	Search for nearest decimal degree in an array of decimal degrees and return the index.
	np.argmin returns the indices of minimum value along an axis.
	so subtract dd (decimal degree of selected location) from all values in dd_array, take absolute value and find index of minimum.
	"""
	geo_idx = (np.abs(dd_array - dd)).argmin()
	return geo_idx

year = 2011

## input lat lon of site where retreat rates are measured
lat_site = 70.88 # [degrees North]
lon_site = -153.92 # [positive means degrees East, negative is degrees West]

start_date = str(year) + '-01-01'  # make month padded with preceding zero (e.g. july is 07 and not 7) , same with daynumber e.g. 2007-07-01 is july 1st).
end_date = str(year) + '-12-31'

## start def
start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

# sea ice concentration input file
sicn_ifile = ncfile_path + 'ci_with_mask/ERAintforcing_Arctic_ci_' + str(year) + '0101_' + str(year) + '1231_with_mask.nc'
print(sicn_ifile)

# read sea ice concentration from ifile

fh = Dataset(sicn_ifile, mode = 'r')
lons = fh.variables['longitude'][:] # [degrees north]
lats = fh.variables['latitude'][:] # [degrees east]
timestamp = fh.variables['time'] # hours since 1900-01-01 00:00:00.0, calendar = gregorian
sicn = fh.variables['siconc'][:,:,:] # fraction of ice cover within the grid cell shape is: (time, lat, lon)
dates = cftime.num2pydate(timestamp[:],timestamp.units,calendar=timestamp.calendar) # convert to python datetime
fh.close()

# load land-sea mask 
fh = Dataset(ncfile_path + 'land_sea_mask.nc', mode= 'r')
mask = fh.variables['lsm'][:]
lats_mask = fh.variables['latitude'][:]
lons_mask_orig = fh.variables['longitude'][:] # lons go from 0 to 360
fh.close()

# have to correct the lons from teh mask to the same convention as is used for the sicn variable. 
lons_mask = lons_mask_orig
lons_mask[int(np.where(lons_mask_orig>180)[0][0]):] = lons_mask_orig[int(np.where(lons_mask_orig>180)[0][0]):]-360.

# find indices that are closest to the selected lat/lon
lat_idx = geo_idx(lat_site,lats)
lon_idx = geo_idx(lon_site,lons)

# check function is working correctly
lat_site_ERAI = lats[lat_idx]
lon_site_ERAI = lons[lon_idx]

### save lat and lon of site on ERA grid
np.save(basepath + 'Drew_Point/lat_site_ERAI_Drew_Point.npy',lat_site_ERAI)
np.save(basepath + 'Drew_Point/lon_site_ERAI_Drew_Point.npy',lon_site_ERAI)


###### offshore grid cell #######


# choose an offshore grid cell where the sicn, winds, SST will be taken from ######## this is an input variable later used in ArcticBeach that determines which grid cell of ERA-I that the forcing data comes from! This script is 
# used so you can look around the map produced here if you want to change grid cell locations from which you drive ArcticBeach ###############
lat_offshore_idx = lat_idx - 1
lon_offshore_idx = lon_idx

# save the offshore indices 
lat_offshore_site_ERAI = lats[lat_offshore_idx]
lon_offshore_site_ERAI = lons[lon_offshore_idx]

np.save(basepath + 'Drew_Point/lat_offshore_site_ERAI_Drew_Point.npy',lat_offshore_site_ERAI)
np.save(basepath + 'Drew_Point/lon_offshore_site_ERAI_Drew_Point.npy',lon_offshore_site_ERAI)

# plot the sicn and grid cells of site and offshore selected point in a snapshot of example sicn
m = Basemap(projection='npstere', boundinglat=65, lon_0=0, resolution='l')

# wrap data to cover all lons
sicn_wrapped, lons_wrapped = addcyclic(sicn[30*5*8,:,:],lons) # select timestamp in 0 index for which sea ice snapshot you wish to plot.
mask_wrapped, lons_wrapped_mask = addcyclic(mask[0,:,:],lons_mask)

# get lat/lon formatted
x, y = np.meshgrid(lons_wrapped,lats)

# get indices on map projection 
px,py=m(x,y)

# create figure and axes handles
fig = plt.figure(figsize=(10,10))

cmap = cmocean.cm.ice

# plot data over map
m.pcolor(px,py, sicn_wrapped,cmap=cmap)

## mark where site is
# get indices of site on projection coords
px_site, py_site = m(lon_site_ERAI, lat_site_ERAI)
px_site_label, py_site_label = m(lon_site_ERAI+16, lat_site_ERAI-6)
# plot site
m.scatter(px_site, py_site, marker='o', s=100, color='r', zorder=5)
#plt.text(px_site_label,py_site_label, 'Drew Point', fontsize=12, fontweight='bold', color='r')

# add extras
m.drawcoastlines()
m.bluemarble()
#m.fillcontinents(color='gray')
#m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)

plt.show()

##### make a zoomed in plot of the crossshore transect area

def polar_stere(lon_w, lon_e, lat_s, lat_n, **kwargs):
	'''Returns a Basemap object (NPS/SPS) focused in a region.

	lon_w, lon_e, lat_s, lat_n -- Graphic limits in geographical coordinates.
	W and S directions are negative.
	**kwargs -- Aditional arguments for Basemap object.

	'''
	lon_0 = lon_w + (lon_e - lon_w) / 2.
	ref = lat_s if abs(lat_s) > abs(lat_n) else lat_n
	lat_0 = math.copysign(90., ref)
	proj = 'npstere' if lat_0 > 0 else 'spstere'
	prj = Basemap(projection=proj, lon_0=lon_0, lat_0=lat_0, boundinglat=0, resolution='c')
	#prj = pyproj.Proj(proj='stere', lon_0=lon_0, lat_0=lat_0)
	lons = [lon_w, lon_e, lon_w, lon_e, lon_0, lon_0]
	lats = [lat_s, lat_s, lat_n, lat_n, lat_s, lat_n]
	x, y = prj(lons, lats)
	ll_lon, ll_lat = prj(min(x), min(y), inverse=True)
	ur_lon, ur_lat = prj(max(x), max(y), inverse=True)
	return Basemap(projection='stere', lat_0=lat_0, lon_0=lon_0,llcrnrlon=ll_lon, llcrnrlat=ll_lat,urcrnrlon=ur_lon, urcrnrlat=ur_lat, **kwargs)


# create figure and axes handles
fig = plt.figure(figsize=(10,10))

lllon = -160
urlon = -145
lllat = 69
urlat = 73

m_zoom = polar_stere(lllon, urlon, lllat, urlat)

# get meter distances on the map projection 
px,py=m_zoom(x,y)

# set min and max of map
xmin, ymin = m_zoom(lllon, lllat)
xmax, ymax = m_zoom(urlon, urlat)

# plot data over map
m_zoom.pcolor(px,py, sicn_wrapped,cmap=cmap)

# plot ERAI land mask
#m_zoom.pcolor(px,py, mask_wrapped[0:43,:])

# add extras
m_zoom.drawcoastlines()
m_zoom.bluemarble()

## mark where site is
# get indices of site on projection coords

# site retreat loc/marker
px_site, py_site = m_zoom(lon_site_ERAI + 0.75/2, lat_site_ERAI - 0.75/2)
#px_site, py_site = m_zoom(lon_site_ERAI, lat_site_ERAI)
m_zoom.scatter(px_site, py_site, marker='*', s=100, color ='r', zorder=5)
# drew point text label
#px_site_label, py_site_label = m_zoom(lon_site_ERAI-7, lat_site_ERAI+3)
#plt.text(px_site_label, py_site_label, 'Drew Point', fontsize=16, fontweight='bold',color='r')

# plot the offshore point
# loc/marker
px_site, py_site = m_zoom(lons[lon_offshore_idx] + 0.75/2, lats[lat_offshore_idx] - 0.75/2)
m_zoom.scatter(px_site, py_site, marker='o', s=100, color ='r', zorder=5)
# label
#px_site_label, py_site_label = m_zoom(lons[lon_offshore_idx]-7, lats[lat_offshore_idx]+3)
#plt.text(px_site_label, py_site_label, 'Offshore point', fontsize=16, fontweight='bold',color='r')

# draw grid lines
m_zoom.drawmeridians(np.arange(np.min(lons_mask),np.max(lons_mask)+0.75,0.75), color='k') # last value is exclusive so have to add 0.75
m_zoom.drawparallels(np.arange(-90, 90.75,0.75), color='k')

# set axes limits for plotting so that the map is zoomed in
ax = plt.gca()

ax.set_xlim([xmin,xmax])
ax.set_ylim([ymin,ymax])

plt.savefig(plot_basepath + 'study_sites/grid_drew_point' + '.png', bbox_inches='tight', dpi=300)
plt.show()




















































































































