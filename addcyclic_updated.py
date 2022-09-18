
from __future__ import (absolute_import, division, print_function)
from distutils.version import LooseVersion

try:
    from urllib import urlretrieve
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlretrieve, urlopen

from matplotlib import __version__ as _matplotlib_version
from matplotlib.cbook import dedent
# check to make sure matplotlib is not too old.
_matplotlib_version = LooseVersion(_matplotlib_version)
_mpl_required_version = LooseVersion('0.98')
if _matplotlib_version < _mpl_required_version:
    msg = dedent("""
    your matplotlib is too old - basemap requires version %s or
    higher, you have version %s""" %
    (_mpl_required_version,_matplotlib_version))
    raise ImportError(msg)
from matplotlib import rcParams, is_interactive
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Ellipse, Circle, Polygon, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import pyproj
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import imread
import sys, os, math
from .proj import Proj
import numpy as np
import numpy.ma as ma
import _geoslib
import functools

def addcyclic(*arr,**kwargs):
    """
    Adds cyclic (wraparound) points in longitude to one or several arrays,
    the last array being longitudes in degrees. e.g.

   ``data1out, data2out, lonsout = addcyclic(data1,data2,lons)``

    ==============   ====================================================
    Keywords         Description
    ==============   ====================================================
    axis             the dimension representing longitude (default -1,
                     or right-most)
    cyclic           width of periodic domain (default 360)
    ==============   ====================================================
    """
    # get (default) keyword arguments
    axis = kwargs.get('axis',-1)
    cyclic = kwargs.get('cyclic',360)
    # define functions
    def _addcyclic(a):
        """addcyclic function for a single data array"""
        npsel = np.ma if np.ma.is_masked(a) else np
        slicer = [slice(None)] * np.ndim(a)
        try:
            slicer[axis] = slice(0, 1)
        except IndexError:
            raise ValueError('The specified axis does not correspond to an '
                    'array dimension.')
        return npsel.concatenate((a,a[slicer]),axis=axis)
    def _addcyclic_lon(a):
        """addcyclic function for a single longitude array"""
        # select the right numpy functions
        npsel = np.ma if np.ma.is_masked(a) else np
        # get cyclic longitudes
        clon = (np.take(a,[0],axis=axis)
                + cyclic * np.sign(np.diff(np.take(a,[0,-1],axis=axis),axis=axis)))
        # ensure the values do not exceed cyclic
        clonmod = npsel.where(clon<=cyclic,clon,np.mod(clon,cyclic))
        return npsel.concatenate((a,clonmod),axis=axis)
    # process array(s)
    if len(arr) == 1:
        return _addcyclic_lon(arr[-1])
    else:
        return list(map(_addcyclic,arr[:-1]) + [_addcyclic_lon(arr[-1])])
