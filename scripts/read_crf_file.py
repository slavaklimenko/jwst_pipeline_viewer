from astropy.io import ascii, fits
import matplotlib.pyplot as plt
path = '/home/slava/science/codes/python/jwst/output/detector3/jw02155004001_03104_00001_mirifushort_a3001_crf.fits'


from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import numpy as np
data = datamodels.open(path)
#f = np.loadtxt('./../data/miri_psf_pix_Argyriou_2023.dat')
mask_CR = np.bitwise_and(data.dq,dqflags.pixel['OUTLIER'])
print(np.sum(mask_CR))
fig,ax = plt.subplots(1,3,sharey=True,sharex=True)
c1 = ax[0].imshow(data.data, cmap='viridis', origin='lower',vmin=-30,vmax=200)
ax[1].imshow(data.err, cmap='viridis', origin='lower',vmin=-30,vmax=200)
ax[2].imshow(mask_CR)
fig.colorbar(c1)


path = '/home/slava/science/codes/python/jwst/output/detector3/jw02155004001_03104_00004_mirifushort_a3001_crf.fits'


from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import numpy as np
data = datamodels.open(path)
#f = np.loadtxt('./../data/miri_psf_pix_Argyriou_2023.dat')
mask_CR = np.bitwise_and(data.dq,dqflags.pixel['OUTLIER'])
print(np.sum(mask_CR))
fig,ax = plt.subplots(1,3,sharey=True,sharex=True)
c1 = ax[0].imshow(data.data, cmap='viridis', origin='lower',vmin=-30,vmax=200)
ax[1].imshow(data.err, cmap='viridis', origin='lower',vmin=-30,vmax=200)
ax[2].imshow(mask_CR)
fig.colorbar(c1)

plt.show()