import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal


path = '/home/slava/science/codes/python/jwst/output/detector3/J1007+2853_dith=4_ch1-medium_s3d.fits'

hdulist = fits.open(path)
hdu = hdulist[1].data
hdulist.close()
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import numpy as np
data = datamodels.open(path)
wcs = data.meta.wcs
conv = wcs.pixel_to_world(20,20,600)
ra= conv[0].ra.deg
print('')
#pixel_to_world_values
#https://jdaviz.readthedocs.io/en/latest/_modules/jdaviz/configs/imviz/plugins/coords_info/coords_info.html
#hasattr(getattr(image, 'coords', None), 'pixel_to_world_values'):
#sky = image.coords.pixel_to_world(x, y).icrs