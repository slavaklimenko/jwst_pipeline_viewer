#import webbpsf
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import os
import astropy,scipy,pickle
from scipy.interpolate import interp1d
import matplotlib.patches as patches
from lmfit import Minimizer, Parameters

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
                if values[0] == 'input3_dir:':
                    init_settings['input3_dir'] = values[1]
                if values[0] == 'spec2_cachedir:':
                    init_settings['spec2_cachedir'] = values[1]
                if values[0] == 'output1_dir:':
                    init_settings['output1_dir'] = values[1]
                if values[0] == 'output2_dir:':
                    init_settings['output2_dir'] = values[1]
                if values[0] == 'output3_dir:':
                    init_settings['output3_dir'] = values[1]
                if values[0] == 'CRDS_PATH:':
                    init_settings['CRDS_PATH'] = values[1]
                if values[0] == 'CRDS_SERVER_URL:':
                    init_settings['CRDS_SERVER_URL'] = values[1]
                if values[0] == 'WEBBPSF_PATH:':
                    init_settings['WEBBPSF_PATH'] = values[1]
    return init_settings
settings =  read_settings()
os.environ["CRDS_PATH"] = settings['CRDS_PATH']
os.environ["CRDS_SERVER_URL"] = settings['CRDS_SERVER_URL']
if 'CRDS_CONTEXT' in  settings.keys():
    os.environ["CRDS_CONTEXT"] = settings['CRDS_CONTEXT']
os.environ["WEBBPSF_PATH"] = settings['WEBBPSF_PATH']

#from stage3_pipeline import detector3


# HD159222
if 0:
    fname = 'HD-159222_1C_ch1-long_s3d.fits'
    cal_name = '/home/slava/science/codes/python/jwst/output/detector2/' + 'jw01050003001_03106_00001_mirifushort_cal.fits'
    #psf_center = (18 + 0.25 * 2, 14 - 0.25 * 1)  # sA
    psf_center = (25, 24)  # sA



def rebin_2d(array, factor):
    array = np.array(array)
    # Ensure the array shape is divisible by the rebinning factor
    shape = (array.shape[0] // factor, factor, array.shape[1] // factor, factor)
    # Reshape and compute the mean along the rebinned axes
    return array.reshape(shape).mean(axis=(1, 3))*factor**2


def upsample_2d(array, factor):
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)

from JWST_cube_analyser import miri_psf_arcsec

if __name__ == '__main__':

    mode = 'star_psf'
    debug = True

    #create star psf for HD152999
    ch_list = ['1A_ch1-short']
    if 1:
        ch_list = ['1A_ch1-short','1B_ch1-medium','1C_ch1-long',
                   '2A_ch2-short','2B_ch2-medium','2C_ch2-long',
                   '3A_ch3-short','3B_ch3-medium','3C_ch3-long',
                   '4A_ch4-short','4B_ch4-medium','4C_ch4-long']


    path = '/home/slava/science/codes/python/jwst/'
    for ch in ch_list:

        if mode == 'star_psf':
            print('ch',ch)
            filename = (path + 'output/detector3/HD159222_ATCN6_N6/'+
                        'HD-159222_ATCN6_N6_'+ch+'__CORR_s3d.fits')
            star_psf_cube = datamodels.open(filename)
            data = np.array(star_psf_cube.data)
            #wcs = star_psf_cube.meta.wcs
            image = np.nanmedian(data[:300,:,:], axis=0)
            #set wavelength
            with fits.open(filename) as hdu:
                hdr = hdu['SCI'].header
            wavelength = (np.arange(hdr['NAXIS3']) + hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']

            from astropy.wcs import WCS
            wcs = WCS(hdr)


            print('cenA:', np.argwhere(image == np.nanmax(image)))
            cenA = np.argwhere(image == np.nanmax(image))[0]
            # set 1 sigma aperture
            miri_psf_fwhm = miri_psf_arcsec(np.nanmean(wavelength)) #in arcsec
            miri_psf_sigma = miri_psf_fwhm / 2.355 #in arcsec
            cdelt1 = wcs.wcs.cdelt[0]
            pix_size = cdelt1*3600 #in arcsec
            miri_psf_sigma_pix =miri_psf_sigma / (pix_size)
            print('1sigma rad =', miri_psf_sigma_pix, 'pix')

            # pixel grids
            yy, xx = np.indices(image.shape)
            # distance from center
            rr = np.sqrt((xx - cenA[1]) ** 2 + (yy - cenA[0]) ** 2)

            # circular mask within 1 sigma
            mask = rr <= miri_psf_sigma_pix

            # mean value in the 2D image
            norm_flux_1sigma = np.nansum(data[:,mask],axis=1)
            fig,ax = plt.subplots()
            plt.plot(wavelength,norm_flux_1sigma)
            plt.show()

            hdu1 = fits.open(filename)

            from astropy.io import fits

            norm_data = data / norm_flux_1sigma[:, None, None]

            print(norm_data.shape)
            fig, ax = plt.subplots()
            plt.imshow(norm_data[100,:,:],origin='lower')
            plt.plot(cenA[1],cenA[0],'o',color='red')
            plt.show()
            #save data
            if 0:
                half_size = 20
                y0, x0 = cenA
                ymin, ymax = y0 - half_size, y0 + half_size
                xmin, xmax = x0 - half_size, x0 + half_size
                # primary science cube
                hdu_sci = fits.PrimaryHDU(data=norm_data[:, ymin:ymax, xmin:xmax], header=hdr)
                # wavelength "cube" (broadcast to 3D for consistency)
                hdu_wave = fits.ImageHDU(data=wavelength, name='WAVELENGTH')

                hdul = fits.HDUList([hdu_sci, hdu_wave])
                hdul.writeto(path+'data_local/miri_psf/psf_v2.0/'+ch+'_test.fits', overwrite=True)



print('Ok!')