import webbpsf
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

#HD163466
if 0:
    fname = 'HD-163466_1C_ch1-long_BKG_SUBTR_s3d.fits'
    psf_center = (21 , 23)  # sA


if 0:
    cube = datamodels.open(path+fname)
    #read wavelenght
    sstring = fname.split('_s3d')[0] +  '_x1d.fits'
    hdu2 = fits.open(path+sstring)
    wavelengths = hdu2['EXTRACT1D'].data['WAVELENGTH']
    hdu2.close()

    # --- Step 1: Define the 2B Channel Wavelength Range ---
    cube_shape = cube.data.shape   # Data cube: 5 slices, 60x60 pixels each
    data_slice = np.nanmedian(cube.data,axis=0) #[slice_numer]
    #data_slice = cube.data[slice_numer]
    #lam = wavelengths[slice_numer]


def rebin_2d(array, factor):
    array = np.array(array)
    # Ensure the array shape is divisible by the rebinning factor
    shape = (array.shape[0] // factor, factor, array.shape[1] // factor, factor)
    # Reshape and compute the mean along the rebinned axes
    return array.reshape(shape).mean(axis=(1, 3))*factor**2


def upsample_2d(array, factor):
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


if __name__ == '__main__':

    mode = 'star_psf'
    debug = True

    ch_list = ['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']
    #ch_list = ['4C']

    path = '/home/slava/science/codes/python/jwst/'
    for ch in ch_list:
        if mode == 'webb_psf':
            #make the psf function using webbpsf
            print('channel = ',ch)
            cal_list = path+'data/miri_psf/webb_psf/cal_files.txt'
            with open(cal_list, "r") as file:
                #content = file.readline().strip()
                for line in file:
                    if line.strip().split(',')[0]==ch:
                        cal_filename = line.split(',')[1]
                        cal_filename=cal_filename.split("\n")[0]
                        cal_filename = cal_filename.replace(" ","")


            #miri = webbpsf.MIRI()
            #miri.channel = ch[0]  # MIRI MRS Channel 2

            hdu = fits.open(path+'output/detector2/' + cal_filename) #'jw02441001001_04106_00001_mirifushort_cal.fits') # cal_filename)
            if hdu[0].header['INSTRUME'] == 'MIRI':
                hdu[0].header['CHANNEL'] = ch[0]

            hdu.writeto('./tmp_output/tmp.fits', overwrite=True)
            inst = webbpsf.setup_sim_to_match_file('./tmp_output/tmp.fits')

            psf = inst.calc_psf(nlambda=1)#monochromatic=, nlambda=1
            psf.info()
            #read: https://webbpsf.readthedocs.io/en/stable/usage.html
            #https://webbpsf.readthedocs.io/en/stable/jwst_matching_psfs_to_data.html

            if debug:
                fig, axes = plt.subplots(figsize=(12, 3), ncols=4,sharex=True,sharey=True)
                for i in range(len(psf)):
                    webbpsf.display_psf(psf, ext=i, ax=axes[i], title=f'Ext {i}: {psf[i].header["EXTNAME"]}',
                                        imagecrop=2, colorbar=False)

            psf_image = psf['DET_DIST'].data.copy()
            psf_image_overdist = psf['OVERDIST'].data.copy()
            #psf_image= astropy.nddata.Cutout2D(psf_image, position=(46,46), size=12).data

            #save psf
            with open( path+'data/miri_psf/webb_psf/psf_'+ch+'.pkl', 'wb') as f:
              pickle.dump(psf_image,f)
            with open( path+'data/miri_psf/webb_psf/psf_'+ch+'_overdist.pkl', 'wb') as f:
                pickle.dump(psf_image_overdist, f)
            plt.show()

        if mode == 'star_psf':
            print('ch',ch)
            star_psf_cube_A = datamodels.open(path + 'output/detector3/'+ 'HD-159222_'+ch+'_BKG_SUBTR_s3d.fits')
            if 1:
                if ch!='4C':
                    star_psf_img_full_A = np.nanmedian(star_psf_cube_A.data, axis=0)
                else:
                    star_psf_img_full_A = np.nanmedian(star_psf_cube_A.data[:300,:,:], axis=0)
                star_psf_img_full_A /= np.nansum(star_psf_img_full_A)
                print('cenA:', np.argwhere(star_psf_img_full_A == np.nanmax(star_psf_img_full_A)))
                cenA = np.argwhere(star_psf_img_full_A == np.nanmax(star_psf_img_full_A))[0]
                print(cenA[0] - 10, cenA[0] + 11, cenA[1] - 9, cenA[1] + 10)
                star_psf_img_A = np.array(star_psf_img_full_A[cenA[0] - 10:cenA[0] + 11, cenA[1] - 9:cenA[1] + 10])

            star_psf_cube_B = datamodels.open(path + 'output/detector3/'+ 'HD-163466_'+ch+'_BKG_SUBTR_s3d.fits')
            if 1:
                if ch!='4C':
                    star_psf_img_full_B = np.nanmedian(star_psf_cube_B.data, axis=0)
                else:
                    star_psf_img_full_B = np.nanmedian(star_psf_cube_B.data[:300,:,:], axis=0)
                star_psf_img_full_B /= np.nansum(star_psf_img_full_B)
                print('cenB:', np.argwhere(star_psf_img_full_B == np.nanmax(star_psf_img_full_B)))
                cenB = np.argwhere(star_psf_img_full_B == np.nanmax(star_psf_img_full_B))[0]
                print(cenB[0] - 10, cenB[0] + 11, cenB[1] - 9, cenB[1] + 10)
                star_psf_img_B = np.array(star_psf_img_full_B[cenB[0] - 10:cenB[0] + 11, cenB[1] - 9:cenB[1] + 10])

            star_psf_img = np.nanmean([star_psf_img_A,star_psf_img_B],axis=0)
            star_psf_img /= np.nansum(star_psf_img)
             #save psf
            with open( path+'data/miri_psf/custom_psf/star_psf_'+ch+'.pkl', 'wb') as f:
                pickle.dump(star_psf_img,f)
            if debug:
                fig,ax = plt.subplots(1,4,sharex=True,sharey=True)
                ax[0].imshow(np.log10(np.abs(star_psf_img_A)))
                ax[0].set_title('A')
                ax[1].imshow(np.log10(np.abs(star_psf_img_B)))
                ax[1].set_title('B')

                ax[2].imshow(np.log10(np.abs(star_psf_img)))
                ax[2].set_title(ch)

                im1= ax[3].imshow(np.log10(np.abs(star_psf_img_A))-np.log10(np.abs(star_psf_img_B)),vmin=-0.3,vmax=0.3)
                fig.colorbar(im1)
                plt.show()


print('Ok!')