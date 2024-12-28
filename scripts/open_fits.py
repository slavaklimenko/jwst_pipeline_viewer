import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal


path = './../output/detector2/bkgr_subtracted/jw02155011001_05101_00002_mirifulong__bkgr_sub.fits'

hdulist = fits.open(path)
hdu = hdulist[1].data
hdulist.close()
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import numpy as np
data = datamodels.open(path)

plt.subplots()
vmin=np.nanquantile(data.data.flatten(),0.05)
vmax=np.nanquantile(data.data.flatten(),0.95)
plt.imshow(data.data,vmin=-0.1,vmax=vmax)
plt.show()
#pixel_to_world_values
#https://jdaviz.readthedocs.io/en/latest/_modules/jdaviz/configs/imviz/plugins/coords_info/coords_info.html
#hasattr(getattr(image, 'coords', None), 'pixel_to_world_values'):
#sky = image.coords.pixel_to_world(x, y).icrs