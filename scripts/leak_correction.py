
import numpy as np
from astropy.io import fits


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
                #'/home/slava/science/codes/python/jwst/data_local/leak/MRS_spectral_leak_fractional.fits'

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