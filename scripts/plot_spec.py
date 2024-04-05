from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import numpy as np
from astropy.io import fits

path = '/home/slava/science/codes/python/jwst/output/specviewer/'
spec_files = [path+'J1007+2853_new.fits',path+'J0901+2044_new.fits']
spec_redsh = [0.8839,1.0191]

nfigs = len(spec_files)
fig,ax = plt.subplots(nfigs,1,sharex=True)
for i in range(nfigs):
    hdu = fits.open(spec_files[i])
    prihdr = hdu[1].data
    flux = prihdr['FLUX']
    wave = prihdr['WAVELENGTH']/(1+spec_redsh[i])
    hdu.close()
    ax[i].plot(wave,flux)
plt.show()