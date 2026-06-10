
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from statsmodels.tsa.arima import params
from stdatamodels.jwst import datamodels
from lmfit import Minimizer, Parameters


path_to_jwst_folder = '/home/slava/science/codes/python/jwst_Viewer-2.0.0/'



if __name__ == '__main__':


    #subtract two images iteratively
    if 1:
        ch_list = ['1A_ch1-short', '1B_ch1-medium', '1C_ch1-long',
                   '2A_ch2-short', '2B_ch2-medium', '2C_ch2-long',
                   '3A_ch3-short', '3B_ch3-medium', '3C_ch3-long',
                   '4A_ch4-short', '4B_ch4-medium', '4C_ch4-long']
        ch_list = [  '1A_ch1-short', '1B_ch1-medium', '1C_ch1-long',
                   '2A_ch2-short', '2B_ch2-medium', '2C_ch2-long',
                   '3A_ch3-short', '3B_ch3-medium', '3C_ch3-long',
                   '4A_ch4-short', '4B_ch4-medium'
                   ]
        params_list = np.zeros((len(ch_list), 7))
        for i,ch in enumerate(ch_list):
            # read source
            if 0:
                fname = path_to_jwst_folder + 'output/detector3/' + 'B0218-ATCN6_N6/TXS0218+357ATCN6_N6_'+ch+'__CORR_s3d.fits'
            if 1:
                fname = path_to_jwst_folder + 'output/detector3/QSO-B1830-211-SIGHTLINEB_ATCN6_N6/' + 'QSO-B1830-211-SIGHTLINEBATCN6_N6_' + ch + '__CORR_s3d.fits'
                #QSO-B1830-211-SIGHTLINEBATCN6_N6_1B_ch1-medium__CORR_s3d.fits
            else:
                fname = path_to_jwst_folder + 'output/detector3/' + 'J0134_ATCN6_N6/J0134-0931_ATCN6_N6_' + ch + '__CORR_s3d.fits'
            field_name = (fname.split('s3d.fits')[0]).split('/')[-1]
            psf_name = 'HD-163466'

            cube = datamodels.open(fname)
            with fits.open(fname) as hdu:
                hdr = hdu['SCI'].header
            wavelength = (np.arange(hdr['NAXIS3']) + hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']


            from astropy.wcs import WCS
            wcs = WCS(hdr)


            # Load parameters for A
            params_A = Parameters()
            params_A.load(open(fname.split('s3d.fits')[0] +
                               'subtr_A_based_' + psf_name + '_paramsA.json'))

            # Load parameters for B
            params_B = Parameters()
            params_B.load(open(fname.split('s3d.fits')[0] +
                               'subtr_B_based_' + psf_name + '_paramsB.json'))

            xc,yc = params_A['xc'].value,params_A['yc'].value
            l= np.nanmean(wavelength)
            psf_center_A = cube.meta.wcs.pixel_to_world_values(yc, xc, l)
            xc, yc = params_B['xc'].value, params_B['yc'].value
            psf_center_B = cube.meta.wcs.pixel_to_world_values(yc, xc, l)
            print('psf_center_A',psf_center_A)
            params_list[i, 0] = psf_center_A[2]
            params_list[i,1] = psf_center_A[0]
            params_list[i,2] = psf_center_A[1]
            params_list[i,3] =  params_A['amp'].value
            params_list[i,4] = psf_center_B[0]
            params_list[i,5] = psf_center_B[1]
            params_list[i,6] =  params_B['amp'].value
        np.savetxt(path_to_jwst_folder + 'output/detector3/QSO-B1830-211-SIGHTLINEB_ATCN6_N6/'+'fit_parameters_values.txt', params_list)
        print(params_list)


