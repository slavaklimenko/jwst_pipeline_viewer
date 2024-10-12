import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import glob,os
from stdatamodels.jwst import datamodels
import scipy
path = '/home/slava/science/codes/python/jwst/output/detector3/'
filename = 'AO0235+164_dith=2_3A_ch3-short_s3d.fits'

data = datamodels.open(path+filename)
x = np.arange(data.data.shape[1])
y = np.arange(data.data.shape[2])
X,Y = np.meshgrid(y,x)


data_mean = np.nanmedian(data.data,axis=0)
result = data.data.copy()
xc,yc, rad = 15,15,6
xc2,yc2, rad2 = 12,5,3

mask_95 = (np.sqrt(np.power(X-xc,2)+np.power(Y-yc,2))<rad) + (np.sqrt(np.power(X-xc2,2)+np.power(Y-yc2,2))<rad2)
fig,ax = plt.subplots(1,2)
ax[0].imshow(data_mean)
circle1 = plt.Circle((xc, yc), rad, color='yellow', lw=2, fill=False)
circle2 = plt.Circle((xc2, yc2), rad2, color='red', lw=2, fill=False)
ax[0].add_patch(circle1)
ax[0].add_patch(circle2)
ax[1].imshow(mask_95)
plt.show()

#mask_95 = data_mean<0.05*np.nanmax(data_mean)


radius = 2
filter_kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
for i in range(filter_kernel.shape[0]):
    for j in range(filter_kernel.shape[1]):
        if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
            filter_kernel[i, j] = 1

for l in range(100): #(data.data.shape[0]):
    print('l: ', l)
    data_orig =data.data[l].copy()
    data_tmp = data_orig.copy()

    # smooth diff
    npix = scipy.signal.convolve2d(np.ones_like(data_tmp), filter_kernel, mode='same', boundary='fill', fillvalue=0)
    data_smoothed = scipy.signal.convolve2d(data_tmp, filter_kernel, mode='same', boundary='fill', fillvalue=0) / npix
    data_smoothed[mask_95] = np.nan
    data_tmp = data_smoothed

    #
    from scipy.interpolate import RBFInterpolator
    mask = ~np.isnan(data_tmp.flatten())
    xobs = np.c_[X.flatten()[mask], Y.flatten()[mask]]
    yobs = data_tmp.flatten()[mask]
    m = RBFInterpolator(xobs, yobs, kernel='linear',epsilon=5,degree=2)
    #m = RBFInterpolator(xobs, yobs, kernel='thin_plate_spline', epsilon=0.5)
    #
    model = np.zeros_like(data_tmp)
    for i, xi in enumerate(x):
        for k, yi in enumerate(y):
            model[i,k] = m([[yi,xi]])
    model[np.isnan(data_mean)] = np.nan

    #
    result[l,:,:] = model

result_mean = np.nanmean(result,axis=0)
fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
vmin,vmax = np.nanquantile(data_mean,0.05),np.nanquantile(data_mean,0.98)
ax[0].imshow(data_mean,vmin=vmin,vmax = vmax)
circle1 = plt.Circle((xc, yc), rad, color='yellow', lw=2, fill=False)
circle2 = plt.Circle((xc2, yc2), rad2, color='red', lw=2, fill=False)
ax[0].add_patch(circle1)
ax[0].add_patch(circle2)
ax[1].imshow(result_mean,vmin=vmin,vmax = vmax)
ax[2].imshow(data_mean-result_mean,vmin=vmin,vmax = vmax)

l=45
fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
vmin,vmax = np.nanquantile(data_mean,0.05),np.nanquantile(data_mean,0.98)
ax[0].imshow(data.data[l,:,:],vmin=vmin,vmax = vmax)
circle1 = plt.Circle((xc, yc), rad, color='yellow', lw=2, fill=False)
circle2 = plt.Circle((xc2, yc2), rad2, color='red', lw=2, fill=False)
ax[0].add_patch(circle1)
ax[0].add_patch(circle2)
ax[1].imshow(result[l,:,:],vmin=vmin,vmax = vmax)
circle1 = plt.Circle((xc, yc), rad, color='yellow', lw=2, fill=False)
circle2 = plt.Circle((xc2, yc2), rad2, color='red', lw=2, fill=False)
ax[1].add_patch(circle1)
ax[1].add_patch(circle2)
ax[2].imshow(data.data[l,:,:]-result[l,:,:],vmin=vmin,vmax = vmax)


plt.show()