
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import os
import astropy,scipy,pickle
from scipy.interpolate import interp1d
import matplotlib.patches as patches





# function to correct spectral leak
def correct_miri_mrs_spectral_leak(ch3spec, ch1spec, leakreffile):
    """
    Corrects the MRS spectra at 12.2 um for a leak that comes from 6.1 micron.

    Parameters
    ----------
    ch3file: FITS filename with spectrum containing 12.2 micron (assumed to be in Jy)
             can be full spectra, channel 3 only, or even just channel 3A
    ch1file: FITS filename with spectrum containing 6.1 micron (assumed to be in Jy)
             can be full spectra, channel 1 only, or even just channel 1B
    leakreffile: FITS filename giving the reference file of the spectral leak response function
                 function assumed to be in percentage terms and applys to spectra in Jy

    Outputs
    -------
    Saves the ch3file spectrum corrected for the leak in ch3file_leakcor.fits.
    """
    # read in the fractional spectral leak
    cdata = fits.getdata(leakreffile)
    leak_wave = cdata["wavelength"]
    leak_frac = cdata["frac_leak"]
    lmin, lmax = 11.6, 13.4

    # read in the spectral segment with the leak that needs correcting (includes 12.2 micron)
    #hdul = fits.open(ch3file)
    #cdata = hdul[1].data
    #orig_wave = cdata["WAVELENGTH"]
    #orig_flux = cdata["FLUX"]
    (orig_wave, orig_flux) = ch3spec

    # cut the wavelength to focus on just the leak wavelengths
    gvals = (orig_wave > lmin) & (orig_wave < lmax)
    wave = orig_wave[gvals]
    flux = orig_flux[gvals]

    # read in the spectral segment with the wavelengths that leak (includes 6.1 micron)
    #cdata = fits.getdata(ch1file, 1)
    #wave1b = cdata["WAVELENGTH"]
    #flux1b = cdata["FLUX"]
    (wave1b,flux1b) = ch1spec

    interp = np.interp(wave, wave1b * 2, flux1b)
    interp_leak = np.interp(wave, leak_wave, leak_frac)

    # Apply spectral leak calibration to the 1B spectra
    leak = interp * interp_leak

    # remove the leak from the full channel 3 spectrum and save it
    orig_flux_corr = np.array(orig_flux)
    orig_flux_corr[gvals] = flux - leak

    return (orig_wave,orig_flux_corr)

if __name__ == '__main__':

    if 0:

        folder = '/home/slava/science/codes/python/jwst/output/detector3/HD159222_ATCN6_N6/'
        filename = 'HD-159222_ATCN6_N6_3A_ch3-short_s3d.fits'
        cube3A = datamodels.open(folder+filename)
        with fits.open(folder+filename) as hdu:
            hdr = hdu['SCI'].header
        wave3A = (np.arange(hdr['NAXIS3']) + hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']
        hdr = hdu['SCI'].header
        wcs3A = {}
        wcs3A['CRPIX1'] = hdr['CRPIX1']
        wcs3A['CRPIX2'] = hdr['CRPIX2']
        wcs3A['CRPIX3'] = hdr['CRPIX3']
        wcs3A['CRVAL1'] = hdr['CRVAL1']
        wcs3A['CRVAL2'] = hdr['CRVAL2']
        wcs3A['CRVAL3'] = hdr['CRVAL3']
        wcs3A['CDELT1'] = hdr['CDELT1']
        wcs3A['CDELT2'] = hdr['CDELT2']
        wcs3A['CDELT3'] = hdr['CDELT3']
        wcs3A['NAXIS1'] = hdr['NAXIS1']
        wcs3A['NAXIS2'] = hdr['NAXIS2']
        wcs3A['NAXIS3'] = hdr['NAXIS3']


        filename = 'HD-159222_ATCN6_N6_1B_ch1-medium_s3d.fits'
        cube1B = datamodels.open(folder+filename)
        with fits.open(folder+filename) as hdu:
            hdr = hdu['SCI'].header
        wave1B = (np.arange(hdr['NAXIS3']) + hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']
        hdr = hdu['SCI'].header
        wcs1B = {}
        wcs1B['CRPIX1'] = hdr['CRPIX1']
        wcs1B['CRPIX2'] = hdr['CRPIX2']
        wcs1B['CRPIX3'] = hdr['CRPIX3']
        wcs1B['CRVAL1'] = hdr['CRVAL1']
        wcs1B['CRVAL2'] = hdr['CRVAL2']
        wcs1B['CRVAL3'] = hdr['CRVAL3']
        wcs1B['CDELT1'] = hdr['CDELT1']
        wcs1B['CDELT2'] = hdr['CDELT2']
        wcs1B['CDELT3'] = hdr['CDELT3']
        wcs1B['NAXIS1'] = hdr['NAXIS1']
        wcs1B['NAXIS2'] = hdr['NAXIS2']
        wcs1B['NAXIS3'] = hdr['NAXIS3']



        image3A = np.nanmean(cube3A.data[:300,:,:], axis=0)
        image3A_snr = np.nanmean(cube3A.data[:300, :, :]/cube3A.err[:300, :, :], axis=0)
        snr_limit= 150

        image1B = np.nanmean(cube1B.data[:300,:,:], axis=0)


        fig,ax = plt.subplots(1,2)
        ax[0].imshow(image1B,origin='lower')
        ax[0].set_title('1B')
        ax[1].imshow(image3A,origin='lower')
        ax[1].set_title('3A')
        ax[1].contour(
            image3A_snr,
            levels=[snr_limit],
            colors='white',
            linewidths=1.5
        )
        plt.show()

        mask_snr = np.where( image3A_snr>snr_limit)
        for x,y in zip(mask_snr[1], mask_snr[0]):
            print(x,y)
            #psf_center_A = cube3A.meta.wcs.world_to_pixel_values(raq, deq, l)
            world_coord_3A_pix = cube3A.meta.wcs.pixel_to_world(x, y, 300)
            pix_coord_1B_pix = cube1B.meta.wcs.world_to_pixel_values(world_coord_3A_pix[0].ra, world_coord_3A_pix[0].dec, 300)
            print(world_coord_3A_pix,pix_coord_1B_pix)
            print(cube3A.meta.wcs.pixel_to_world(pix_coord_1B_pix[0],pix_coord_1B_pix[1],300))

            fig, ax = plt.subplots(1, 4,figsize=(12,3))
            ax[0].imshow(np.log10(np.abs(image1B)), origin='lower')
            ax[0].plot(pix_coord_1B_pix[0],pix_coord_1B_pix[1],'o',color='red')
            #
            ax[1].imshow(np.log10(np.abs(image3A)), origin='lower')
            ax[1].plot( x,y, 'o', color='red')
            ax[1].contour(
                image3A_snr,
                levels=[snr_limit],
                colors='white',
                linewidths=1.5
            )

            leak_filename = '/home/slava/science/codes/python/jwst/data_local/leak/MRS_spectral_leak_fractional.fits'
            #sp1B = (wave1B,cube1B.data[:,int(pix_coord_1B_pix[1]),int(pix_coord_1B_pix[0])])

            # correction for differences in solid angle
            pix_solid_angle_1B = wcs1B['CDELT1'] * wcs1B['CDELT2'] * (np.pi / 180) ** 2 * 1e6
            pix_solid_angle_3A = wcs3A['CDELT1'] * wcs3A['CDELT2'] * (np.pi / 180) ** 2 * 1e6

            angle_corr = pix_solid_angle_3A / pix_solid_angle_1B

            if 1:
                xx = int(np.round(pix_coord_1B_pix[0]))
                yy = int(np.round(pix_coord_1B_pix[1]))
                r = 10
                sp1B = (
                    wave1B,
                    np.nanmean(
                        cube1B.data[:, yy - r-1:yy + r, xx - r-1:xx + r],
                        axis=(1, 2)
                    )/angle_corr
                )
            else:
                sp1B = (wave1B, cube1B.data[:, pix_coord_1B_pix[1], pix_coord_1B_pix[0]])


            if 0:
                sp3A = (
                    wave3A,
                    np.nanmean(
                        cube3A.data[:, y - r-1:y + r, x - r-1:x + r],
                        axis=(1, 2)
                    )
                )
            else:
                sp3A = (wave3A, cube3A.data[:, y, x])

            sp3A_corr = correct_miri_mrs_spectral_leak(ch3spec=sp3A, ch1spec=sp1B, leakreffile=leak_filename)

            ax[2].plot(sp1B[0],sp1B[1])
            ax[2].set_title('1B:' +str(int(pix_coord_1B_pix[1]))+
                            ','+str(int(pix_coord_1B_pix[0])))
            ax[3].plot(sp3A[0],sp3A[1],label='orig')
            ax[3].set_title('3A:' +str(y)+
                            ','+str(x))

            ax[3].plot(sp3A_corr[0],sp3A_corr[1],label='corrected')
            ax[3].legend()
        plt.show()

    if 1:
        leak_filename = '/home/slava/science/codes/python/jwst/data_local/leak/MRS_spectral_leak_fractional.fits'
        sp1 = np.loadtxt('/home/slava/science/projects/jwst/ID5491/v2.0/calibration_stars/'+'HD159222-level3-12s.txt')
        sp = (sp1[:,0],sp1[:,1])
        sp3A_corr = correct_miri_mrs_spectral_leak(ch3spec=sp, ch1spec=sp, leakreffile=leak_filename)
        plt.subplots()
        plt.plot(sp[0],sp[1])
        plt.plot(sp3A_corr[0], sp3A_corr[1])
        sp1[:,1] = sp3A_corr[1]
        #np.savetxt('/home/slava/science/projects/jwst/ID5491/v2.0/calibration_stars/'+'HD159222-level3-12s_leak_corr.txt',sp1)

        plt.show()
