
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import os
import astropy,scipy,pickle
from scipy.interpolate import interp1d
import matplotlib.patches as patches

module_dir = os.path.dirname(os.path.abspath(__file__))

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

    return (orig_flux_corr,leak)

if __name__ == '__main__':

    if 1:

        roi_names = ['green'] #,'red']

        folder = module_dir +'/../output/detector3/roi_spectra/'
        filename = 'HD-163466_ATCN6_N6'
        #HD-163466_ATCN6_N6_4C_ch4-long_s3d_(A)_green
        leak_filename = '/home/slava/science/codes/python/jwst/data_local/leak/MRS_spectral_leak_fractional.fits'

        for roi in roi_names:
            f1B = folder+filename+'_1B_ch1-medium_' + 's3d_(A)_' + roi +'.spec1d'
            sp1B = np.loadtxt(f1B)
            f3A = folder+filename+'_3A_ch3-short_' + 's3d_(A)_' + roi +'.spec1d'
            sp3A = np.loadtxt(f3A)

            (sp3A_corr,leak) = correct_miri_mrs_spectral_leak(ch3spec=(sp3A[:,0],sp3A[:,1]), ch1spec=(sp1B[:,0],sp1B[:,1]),
                                                       leakreffile=leak_filename)
            plt.subplots()
            plt.plot(sp3A[:,0],sp3A[:,1])
            plt.plot(sp3A[:,0], sp3A_corr)
            plt.show()

            sp3A[:,1] = sp3A_corr
            np.savetxt(f3A,sp3A)
