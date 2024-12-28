import warnings
import tempfile

from astroquery.mast import Observations
from photutils.aperture import CircularAperture
#from regions import PixCoord, CirclePixelRegion

from jdaviz import Cubeviz

cubeviz = Cubeviz()
cubeviz.show()