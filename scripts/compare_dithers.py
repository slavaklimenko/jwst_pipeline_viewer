import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import os, glob

####################
def read_settings(init_file='./../init.dat'):
    init_settings = {}
    with open(init_file) as f:
        for k, line in enumerate(f):
            values = [s for s in line.split()]
            if line[0] != '#':
                if values[0] == 'input1_dir:':
                    input_dir = values[1]
                    init_settings['input1_dir'] = values[1]
                if values[0] == 'input2_dir:':
                    init_settings['input2_dir'] = values[1]
                if values[0] == 'spec2_cachedir:':
                    init_settings['spec2_cachedir'] = values[1]
                if values[0] == 'output1_dir:':
                    init_settings['output1_dir'] = values[1]
                if values[0] == 'output2_dir:':
                    init_settings['output2_dir'] = values[1]
                if values[0] == 'CRDS_PATH:':
                    init_settings['CRDS_PATH'] = values[1]
                if values[0] == 'CRDS_SERVER_URL:':
                    init_settings['CRDS_SERVER_URL'] = values[1]
                if values[0] == 'CRDS_CONTEXT:':
                    init_settings['CRDS_CONTEXT'] = values[1]
    return init_settings
settings =  read_settings()
os.environ["CRDS_PATH"] = settings['CRDS_PATH']
os.environ["CRDS_SERVER_URL"] = settings['CRDS_SERVER_URL']
if 'CRDS_CONTEXT' in  settings.keys():
    os.environ["CRDS_CONTEXT"] = settings['CRDS_CONTEXT']
    ##################


def miri_psf_arcsec(lam):
    # interpolation of miri psf https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-performance/miri-point-spread-functions
    f = np.loadtxt('../data/miri_psf_arcsec_Argyriou_2023.dat')

    print(f[:, 0])
    f1d = interp1d(f[:, 0], f[:, 1], fill_value='extrapolate')
    print('miri psf =', f1d(lam))
    return f1d(lam)



path = '/home/slava/science/codes/python/jwst/output/tmp/'
#filename = 'step_Spec2Pipeline_rate_1.fits'
#data1 = datamodels.open(path+filename)

filename = 'jw02155003001_05101_00001_mirifulong__rate.fits'
data2 = datamodels.open(path+filename)

#filename = 'step_Spec2Pipeline_rate_3.fits'
#data3 = datamodels.open(path+filename)
#filename = 'step_Spec2Pipeline_rate_4.fits'
#data4 = datamodels.open(path+filename)


if 1:
    band = data2.meta.instrument.band
    channel = data2.meta.instrument.channel
    import glob

    photom_list = sorted(glob.glob(os.environ["CRDS_PATH"] + '/references/jwst/miri/*photom*'))
    for f in photom_list:
        hdulist = fits.open(f)
        header = hdulist[0].header
        f_band, f_ch = header['BAND'], header['CHANNEl']
        if band == f_band and f_ch == channel:
            photom_file = f
            break

    hdulist = fits.open(photom_file)
    hdu = hdulist[1].data
    hdulist.close()

    mask_flux = hdu.copy()
    mask_flux[np.isnan(mask_flux)] = 0
    mask_flux[mask_flux>0] = 1
if 1:
    wcs2 = data2.meta.wcs
    #wcs4 = data1.meta.wcs

    cal_world_to_detector2 = wcs2.get_transform('world','detector')
    cal_detector_to_world2 = wcs2.get_transform('detector','world')
    #cal_world_to_detector4 = wcs4.get_transform('world','detector')
    #cal_detector_to_world4 = wcs4.get_transform('detector','world')

    if 0:
        fig,ax = plt.subplots(1,2)
        X,Y,F = [],[],[]
        for i in np.arange(500, 501):
            print(i)
            for j in np.arange(10, data2.data.shape[0] - 10):
                if ~np.isnan(data2.data[i,j]) and mask_flux[i,j]>0:
                    (x,y,l) = cal_detector_to_world2(j,i)
                    if ~np.isnan(x) and ~np.isnan(y):
                        X.append(x)
                        Y.append(y)
                        F.append(data2.data[i,j])
        X, Y, F = np.array(X), np.array(Y), np.array(F)
        #X = (X-135.344498)*3600
        #Y = (Y-20.74626)*3600
        ax[0].plot(X,Y,'o',c='blue')
        ax[1].plot(Y,F,'o',c='blue')
        plt.show()

    mask = np.zeros((data2.data.shape[0],data2.data.shape[1]))
    tr_x = [100,250,500,750,900]
    l1 = np.nanmean([cal_detector_to_world2(l, 500)[2] for l in np.arange(100, 400, 10)])
    miri_psf_fwhm1 = miri_psf_arcsec(l1)
    l2 = np.nanmean([cal_detector_to_world2(l,500)[2] for l in np.arange(500,800,10)])
    miri_psf_fwhm2 = miri_psf_arcsec(l2)


    q_ra, q_dec = 135.3445, 20.746259
    for i in tr_x:
        print(i)
        for j in np.arange(10, data2.data.shape[0] - 10):
            if mask_flux[i,j]>0:
                (x,y,l) = cal_detector_to_world2(j,i)
                if ~np.isnan(x) and ~np.isnan(y):
                    if j<500:
                        if np.sqrt((x-q_ra)**2+ (y-q_dec)**2)*3600<2*miri_psf_fwhm1:
                            mask[i,j] = 1
                    else:
                        if np.sqrt((x - q_ra) ** 2 + (y - q_dec) ** 2) * 3600 < 1 * 2*miri_psf_fwhm2:
                            mask[i, j] = 1

        tmp = np.array([mask[i, e] - mask[i, e + 1] for e in np.arange(0, data2.data.shape[1] - 1)])
        ntraces = np.sum(tmp == -1)
        print(np.where(tmp == -1)[0])
        print('number of traces(at i), ', ntraces)

    tmp = np.array([mask[500,e]-mask[500,e+1] for e in np.arange(0,data2.data.shape[1]-1)])
    ntraces = np.sum(tmp==-1)
    print('i=500: ',np.where(tmp==-1)[0])
    print('number of traces(at i=500), ',ntraces)
    left_border = np.zeros((ntraces,len(tr_x)))
    right_border = np.zeros((ntraces,len(tr_x)))

    fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
    ax[0].imshow(data2.data,vmin=-0.5,vmax=0.5)
    ax[1].imshow(mask)
    plt.show()


    for k,i in enumerate(tr_x):
        tmp = np.array([mask[i, e] - mask[i, e + 1] for e in np.arange(0, data2.data.shape[1] - 1)])
        s = np.where(tmp==-1)[0]
        print(s)
        left_border[:,k] =np.where(tmp==-1)[0]
        right_border[:,k] = np.where(tmp == 1)[0]

    for k in range(ntraces):
        zl = np.polyfit(tr_x, left_border[k,:], 2)
        pl = np.poly1d(zl)
        zr = np.polyfit(tr_x, right_border[k,:], 2)
        pr = np.poly1d(zr)
        for i in range(mask.shape[0]):
            l,r = int(np.rint(pl(i))),int(np.rint(pr(i)))
            mask[i,l:r] = 1






    fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
    ax[0].imshow(data2.data,vmin=-0.5,vmax=0.5)
    data2.data[mask.astype(bool)] = -10
    ax[1].imshow(data2.data,vmin=-0.5,vmax=0.5)
    plt.show()



fig,ax = plt.subplots(1,4)
ax[0].imshow(data1.data,vmin=-0.5,vmax=0.5)
ax[1].imshow(data2.data,vmin=-0.5,vmax=0.5)
ax[2].imshow(data3.data,vmin=-0.5,vmax=0.5)
ax[3].imshow(data4.data,vmin=-0.5,vmax=0.5)
ax[0].plot(500,100,'o',markersize=20)

fig,ax = plt.subplots(1,3)
ax[0].imshow(data2.data,vmin=-0.5,vmax=0.5)
ax[1].imshow(data4.data,vmin=-0.5,vmax=0.5)
ax[2].imshow(data2.data-data4.data,vmin=-0.5,vmax=0.5)

if 1:
    band = data1.meta.instrument.band
    channel = data1.meta.instrument.channel
    import glob
    photom_list = sorted(glob.glob(os.environ["CRDS_PATH"] +'/references/jwst/miri/*photom*'))
    for f in photom_list:
        hdulist = fits.open(f)
        header = hdulist[0].header
        f_band,f_ch = header['BAND'],header['CHANNEl']
        if band == f_band and f_ch == channel:
            photom_file = f
            break
    from flat_field import get_trace_mask
    trace_mask = get_trace_mask(path=photom_file)
    trace_shape = trace_mask.shape
    nrows = data4.shape[0]
    for t in range(trace_shape[0]):
        ax[2].plot(trace_mask[t, :, 0], np.arange(nrows), color='red', lw=2)
        ax[2].plot(trace_mask[t, :, 1], np.arange(nrows), color='red', lw=2)

plt.show()
print('')