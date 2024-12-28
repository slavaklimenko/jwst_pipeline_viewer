#!/usr/bin/env python


from bisect import bisect_left
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import numpy as np
from pathlib import Path
from scipy import interpolate
import sys
sys.path.append('/home/slava/science/codes/python/spectro/')
sys.path.append('/home/slava/anaconda3')
sys.path.append('/home/slava/science/codes/python/spectro/sviewer/')
from astropy import constants as const
import os
import pickle
from scipy.interpolate import interp1d
from matplotlib import rcParams
from astropy.io import ascii, fits
rcParams['font.family'] = 'serif'
import emcee
from chainconsumer import ChainConsumer

from astropy.io import fits
from scipy import signal
import scipy.signal
import time, glob
from numpy.polynomial.polynomial import polyval
from scipy.signal import savgol_filter
import csv
#define classes and functions

class spectrum():
    def __init__(self, x=None, y=None, err=None,name=None):
        if any([v is not None for v in [x, y, err]]):
            self.set_data(x=x, y=y, err=err,name=name)

    def set_data(self, x=None, y=None, err=None,name=None):
        if x is not None:
            self.x = np.asarray(x)
        if y is not None:
            self.y = np.asarray(y)
        if y is not None:
            self.y = np.asarray(y)
        if err is not None:
            self.err = np.asarray(err)
        if name is not None:
            self.name = name

    def normalize(self,x0=12,delta_x = 0.1):
        if x0>self.x[0] and x0<self.x[-1]:
            mask = (self.x >= x0-delta_x)*(self.x <= x0+delta_x)
            norm_f = np.mean(self.y[mask])
            self.y /=norm_f
            self.err /=norm_f

    def append(self,s,mode = 'mean disp',debug=False):
        if not hasattr(self, 'x'):
            if hasattr(s,'x'):
                self.x = s.x.copy()
                self.y = s.y.copy()
                self.err = s.err.copy()
        else:
            if np.sum(s.x)>0:
                mask_intersection = s.x <= self.x[-1]
                mask_extension = ~mask_intersection
                if debug:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 2)
                    ax[0].plot(s.x,s.y/s.err,label='s')
                    ax[0].plot(self.x,self.y/self.err,label='self')
                    f = np.linspace(0.1,10,100)
                    ax[1].plot(f,(f+5)*5/(25 + f**2))
                    ax[0].legend()
                    plt.show()


                if np.sum(mask_intersection) > 0:
                    s_interp = interp1d(s.x, s.y, bounds_error=False, fill_value=np.NaN)
                    s_interp_err = interp1d(s.x, s.err, bounds_error=False, fill_value=np.NaN)
                    mask_selfx_intersection = self.x>=s.x[0]
                    comb = [self.y[mask_selfx_intersection],s_interp(self.x[mask_selfx_intersection])]
                    e_comb = [self.err[mask_selfx_intersection],s_interp_err(self.x[mask_selfx_intersection])]
                    if mode == 'mean weighted':
                        w = np.power(e_comb,-2)
                        f2 = np.nansum(comb * w, axis=0) / np.nansum(w, axis=0)
                        self.y[mask_selfx_intersection] = f2
                        self.err[mask_selfx_intersection] = np.power(np.nansum(w, axis=0), -0.5)
                    elif mode == 'mean':
                        self.y[mask_selfx_intersection] = np.nanmean(comb)
                        self.err[mask_selfx_intersection] = np.power(np.nansum(np.power(e_comb, -2), axis=0), -0.5)
                    elif mode == 'mean disp':
                        self.y[mask_selfx_intersection] = np.nansum(comb,axis=0)/2
                        f = comb-self.y[mask_selfx_intersection]
                        f1 = np.power(f,2)
                        self.err[mask_selfx_intersection] = np.power(np.nansum(f1,axis=0)/2, 0.5)

                self.x = np.append(self.x,s.x[mask_extension])
                self.y = np.append(self.y,s.y[mask_extension])
                self.err = np.append(self.err,s.err[mask_extension])


    def copy(self):
        return spectrum(self.x,self.y,self.err)


def convolveflux(l, f, res, vel=False, kind='direct-vary-res', verbose=False, debug=False):
    """
    Convolve flux with instrument function.
    Data can be unevenly spaced.
    There are several types of convolution

    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - res       : float
                        resolution of instrument function
        - vel       : boolean
                        if true velocity format for l, otherwise wavelength
        - kind      : str
                        specified type of convolution:
                           'astropy'     : convolution from astropy package (for evenly spaced data)
                           'gauss'       : convolution with fixed gaussian function
                           'direct'      : brut force convolution (can be used for non evenly spaced data)

    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """

    if kind == 'astropy':
        if vel:
            pixels = const.c.cgs.value / res / 1e5 / ((l[-1] - l[0]) / (len(l) - 1))
        else:
            pixels = (l[-1] + l[0]) / 2 / res / ((l[-1] - l[0]) / (len(l) - 1))

        # print(pixels)

        gauss_kernel = conv.Gaussian1DKernel(pixels / 3)

        fc = conv.convolve(f, gauss_kernel, boundary='extend')

    if kind == 'hst-cos':
        lw = 1260  # corresponds to studied range
        print('convolution with the tabulated HST-COS LSF at the wavelength closest to ', lw)
        # choose the wavelength for the LSF (inside the considered wavelength range)
        # read the LSF function and bin size
        hst_kernel, step = hstcos_lsf(lw=lw)
        # save the lsf to *.dat ascii file (to read it late with julia)
        if 0:
            with open('./hst-kernel-lp3-cen1220-lw-1260.dat', 'w') as fout2:
                print(step)
                fout2.write('{:f}\n'.format(step))
                for el in hst_kernel:
                    fout2.write('{0:10.4e} \n'.format(el))

        wavelength = np.array(l)
        spec = np.array(f)
        # rescale  wavelength array to cos resolution (bin size) -> wave_cos
        nstep = round((max(wavelength) - min(wavelength)) / step) - 1
        wave_cos = min(wavelength) + np.arange(nstep) * step

        # builds interpolated initial spectrum at COS' wavelength scale for convolution
        interp_func = interp1d(wavelength, spec)
        spec_cos = interp_func(wave_cos)
        # print('wave_cos ',wave_cos)
        # print('spec_cos ', spec_cos)
        # print('hst_kernel ', hst_kernel)
        # make convolution with the COS LSF at COS resolution - bin to bin
        final_spec = conv.convolve(spec_cos, hst_kernel, boundary="extend", normalize_kernel=True, )
        # return back to original wavelength scale and recalculate convolved spectrum at original wavelength array - fc
        interp_func1 = interp1d(wave_cos, final_spec, fill_value=1, bounds_error=False)
        fc = interp_func1(l)

    if kind == 'hst-cos-temp':
        print('convolution with the HST-COS LSF, using wavelength-variable LSF function')
        # read kernel data from file (default)
        new_lsf, new_w, step = hstcos_lsf_all()

        # resampling onto the input spectrum's wavelength scale
        wavelength = np.array(l)
        spec = np.array(f)
        # sets up new wavelength scale used in the convolution
        nstep = round((max(wavelength) - min(wavelength)) / step) - 1
        wave_cos = min(wavelength) + np.arange(nstep) * step
        interp_func = interp1d(wavelength, spec)
        spec_cos = interp_func(wave_cos)
        # Initializes final spectrum to the interpolated input spectrum
        final_spec = interp_func(wave_cos)

        for i, w in enumerate(new_w):  # Loop through the redefined LSF kernels
            # First need to find the boundaries of each kernel's "jurisdiction": where it applies
            # The first and last elements need to be treated separately
            if i == 0:  # First kernel
                diff_wave_left = 500
                diff_wave_right = (new_w[i + 1] - w) / 2.0
            elif i == len(new_w) - 1:  # Last kernel
                diff_wave_right = 500
                diff_wave_left = (w - new_w[i - 1]) / 2.0
            else:  # All other kernels
                diff_wave_left = (w - new_w[i - 1]) / 2.0
                diff_wave_right = (new_w[i + 1] - w) / 2.0

            # splitting up the spectrum into slices around the redefined LSF kernel wvlns
            # will apply the kernel corresponding to that chunk to that region of the spectrum - its "jurisdiction"
            chunk = np.where((wave_cos < w + diff_wave_right) & (wave_cos >= w - diff_wave_left))[0]
            if len(chunk) == 0:
                # off the edge, go to the next chunk
                continue
            # build wider chunk to account boarder effect
            chunk_larged = np.where((wave_cos < w + 2 * diff_wave_right) & (wave_cos >= w - 2 * diff_wave_left))[0]

            current_lsf = new_lsf[:, i]  # selects the current kernel
            if 1:
                mask = []
                for xi in chunk_larged:
                    k = 0
                    for xj in chunk:
                        if xi == xj:
                            k = 1
                    if k == 1:
                        mask.append(True)
                    else:
                        mask.append(False)
                # print(mask)

            if 1:
                final_spec[chunk] = \
                conv.convolve(spec_cos[chunk_larged], current_lsf, boundary="extend", normalize_kernel=True, )[mask]
            interp_func1 = interp1d(wave_cos, final_spec, fill_value=1, bounds_error=False)
            fc = interp_func1(l)

    if kind == 'gauss':

        # >>> renormalize res to satisfy dispersion of Gauss set to be <l/R>
        R = res * 2 * np.sqrt(2 * np.log(2))

        fc = np.zeros_like(f)

        # expand the regions of wavelength array and flux
        delta = 4
        addl = np.linspace(delta, 0, 21)
        lt = np.concatenate((l[0] * (1 - addl / R), l, l[-1] * (1 + addl / R)), axis=0)
        ft = np.concatenate((f[0] * np.ones_like(addl), f, np.ones_like(addl) * f[-1]), axis=0)

        # print(len(lt), len(ft), lt[0], lt[-1], lt[0]*(4/R))

        def gauss(x, s):
            return 1 / np.sqrt(2 * np.pi) / s * np.exp(-.5 * (x / s) ** 2)

        for i in range(len(l)):
            mask = (lt < l[i] * (1 + delta / R)) & (lt > l[i] * (1 - delta / R))
            fl = ft[mask]
            # print(np.sum(f1), sum(mask))
            if (np.sum(fl) < 0.998 * sum(mask)):
                x = lt[mask] / l[i]
                # print(1-l1/l[i], gauss(1-l1/l[i], 1.0/R))
                # input()
                fc[i] = simps(fl * gauss(1 - x, 1.0 / R), x)
            else:
                fc[i] = 1
        # print(fc)

    if kind == 'direct':
        return convolve_res2(l, f, res)

    if kind == 'direct-vary-res':
        return convolve_res3(l, f, res)

    return fc


# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
def gauss(x, s):
    return 1 / np.sqrt(2 * np.pi) / s * np.exp(-.5 * (x / s) ** 2)


def errf(x):
    a = [0.3480242, -0.0958798, 0.7478556]
    t = 1 / (1 + 0.47047 * np.abs(x))
    return np.sign(x) * (1 - t * (a[0] + t * (a[1] + t * a[2])) * np.exp(-x ** 2))


def errf_v2(x):
    a = [-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587, -0.82215223,
         0.17087277]
    t = 1 / (1 + 0.5 * np.abs(x))
    tau = t * np.exp(-x ** 2 + a[0] + t * (a[1] + t * (
                a[2] + t * (a[3] + t * (a[4] + t * (a[5] + t * (a[6] + t * (a[7] + t * (a[8] + t * a[9])))))))))
    if x >= 0:
        return 1 - tau
    else:
        return tau - 1

def convolve_res(l, f, R):
    """
    Convolve flux with instrument function specified by resolution R
    Data can be unevenly spaced.

    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - R         : float
                        resolution of the instrument function. Assumed to be constant with wavelength.
                        i.e. the width of the instrument function is linearly dependent on wavelenth.

    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """
    #sig = 127301 / R
    delta = 3.0

    n = len(l)
    fc = np.zeros_like(f)

    d = [l[1] - l[0]]
    for i in range(1, n-1):
        d.append((l[i + 1] - l[i - 1]) / 2)
    d.append(l[-1]-l[-2])

    il = 0
    for i, x in enumerate(l):
        sig = x / R / 2.355
        k = il
        while l[k] < x - delta * sig:
            k += 1
        il = k
        s = f[k] * (1 - errf(np.abs(l[k] - x - d[0]/2) / np.sqrt(2) / sig)) / 2
        while k < n and l[k] < x + delta * sig:
            #s += f[k] * 1 / np.sqrt(2 * np.pi) / sig * np.exp(-.5 * ((l[k] - x) / sig) ** 2) * d[k]
            s += f[k] * gauss(l[k] - x, sig) * d[k]
            k += 1

        k -= 1
        s += f[k] * (1 - errf(np.abs(l[k] - x + d[k]/2) / np.sqrt(2) / sig)) / 2
        fc[i] = s

    return fc

def convolve_res2(l, f, R):
    """
    Convolve flux with instrument function specified by resolution R
    Data can be unevenly spaced.

    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - R         : float
                        resolution of the instrument function. Assumed to be constant with wavelength.
                        i.e. the width of the instrument function is linearly dependent on wavelenth.

    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """
    #sig = 127301 / R
    delta = 3.0

    n = len(l)
    fc = np.zeros_like(f)

    f = 1 - f

    il = 0
    for i, x in enumerate(l):
        sig = x / R / 2.355
        k = il
        while l[k] < x - delta * sig:
            k += 1
        il = k
        s = f[il] * (1 - errf_v2((x - l[il]) / np.sqrt(2) / sig)) / 2
        #ie = il + 30
        while k < n-1 and l[k+1] < x + delta * sig:
            #s += f[k] * 1 / np.sqrt(2 * np.pi) / sig * np.exp(-.5 * ((l[k] - x) / sig) ** 2) * d[k]
            s += (f[k+1] * gauss(l[k+1] - x, sig) + f[k] * gauss(l[k] - x, sig)) / 2 * (l[k+1] - l[k])
            #print(i, k , gauss(l[k] - x, sig))
            k += 1
        #input()
        s += f[k] * (1 - errf_v2(np.abs(l[k] - x) / np.sqrt(2) / sig)) / 2
        fc[i] = s

    return 1 - fc

def convolve_res2(l, f, R):
    """
    Convolve flux with instrument function specified by resolution R
    Data can be unevenly spaced.

    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - R         : float
                        resolution of the instrument function. Assumed to be constant with wavelength.
                        i.e. the width of the instrument function is linearly dependent on wavelenth.

    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """
    #sig = 127301 / R
    delta = 3.0

    n = len(l)
    fc = np.zeros_like(f)

    f = 1 - f

    il = 0
    for i, x in enumerate(l):
        sig = x / R / 2.355
        k = il
        while l[k] < x - delta * sig:
            k += 1
        il = k
        s = f[il] * (1 - errf_v2((x - l[il]) / np.sqrt(2) / sig)) / 2
        #ie = il + 30
        while k < n-1 and l[k+1] < x + delta * sig:
            #s += f[k] * 1 / np.sqrt(2 * np.pi) / sig * np.exp(-.5 * ((l[k] - x) / sig) ** 2) * d[k]
            s += (f[k+1] * gauss(l[k+1] - x, sig) + f[k] * gauss(l[k] - x, sig)) / 2 * (l[k+1] - l[k])
            #print(i, k , gauss(l[k] - x, sig))
            k += 1
        #input()
        s += f[k] * (1 - errf_v2(np.abs(l[k] - x) / np.sqrt(2) / sig)) / 2
        fc[i] = s

    return 1 - fc

def convolve_res3(l, f, R):
    """
    Convolve flux with instrument function specified by resolution R
    Data can be unevenly spaced.

    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - R         : float array, shape(N)
                        resolution of the instrument function.

    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """
    #sig = 127301 / R
    delta = 3.0

    n = len(l)
    fc = np.zeros_like(f)

    f = 1 - f

    il = 0
    for i, x in enumerate(l):
        sig = x / R[i] / 2.355
        k = il
        while l[k] < x - delta * sig:
            k += 1
        il = k
        s = f[il] * (1 - errf_v2((x - l[il]) / np.sqrt(2) / sig)) / 2
        #ie = il + 30
        while k < n-1 and l[k+1] < x + delta * sig:
            #s += f[k] * 1 / np.sqrt(2 * np.pi) / sig * np.exp(-.5 * ((l[k] - x) / sig) ** 2) * d[k]
            s += (f[k+1] * gauss(l[k+1] - x, sig) + f[k] * gauss(l[k] - x, sig)) / 2 * (l[k+1] - l[k])
            #print(i, k , gauss(l[k] - x, sig))
            k += 1
        #input()
        s += f[k] * (1 - errf_v2(np.abs(l[k] - x) / np.sqrt(2) / sig)) / 2
        fc[i] = s
        #print(x,s, il, k, k - il)
    return 1 - fc

def Miri_res(l):
    """
       Approxiamtion of MIRI spectral resolution.

       parameters:
           - l         : float array, shape(N)
                           wavelength array (in micron)

       returns:
           - R        : float array, shape(N)
                           spectral resolution
       """

    R = 4603 -128*l

    return R

def dust_abs_model(f):
    '''
    How to convert a dust absorption template to observed flux:
        (a) 1- f
        (b) np.exp(-f)
    :param f: dust template
    :return: observed flux
    '''
    return np.exp(-f)

def dust_emiss_line_model(f,debug=False):
    '''
    How to convert a dust emission template to observed flux:
    :param f: dust template
    :return: observed flux
    '''
    if debug:
        plt.subplots()
        plt.plot(f)
        plt.plot(np.exp(f)-1)
        plt.show()

    return np.exp(f)-1

#

def rebin_arr(a, factor):
    n = a.shape[0] // factor
    return a[:n*factor].reshape(a.shape[0] // factor, factor).sum(1)/factor

def rebin_weight_mean(y, err,factor):
    w = np.array(np.power(err,-2))
    a = np.array(y)
    n = a.shape[0] // factor
    a*=w
    a = a[:n*factor].reshape(n, factor)
    w = w[:n * factor].reshape(n, factor)
    a = a.sum(1)
    w = w.sum(1)
    return a/w,np.power(w,-0.5)

spectra = {}

#folder= '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/spectro/'
#qname='normalized_spectrum_fit1_AO0235.dat'
#qname = ('normalized_spectrum_J0901.dat')
#qname='normalized_spectrum_PAH6_2_J0901.dat'
#qname=('normalized_spectrum_J0900.dat')
#qname= 'normalized_spectrum_J1007-Polynomial.dat'
#qname= 'normalized_spectrum_PowerLaw_J1007.dat'
#qname = 'normalized_spectrum_fit2_AO0235.dat'
qname= 'normalized_spectrum_J1017.dat'
folder = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/normalized_spectra/'

#qname = 'normalized_spectrum_J1017.dat'
rebin = False
normalize = False
correct_err=False

if qname.endswith('fits'):
    hdu = fits.open(folder+qname)
    data = hdu[1].data
    col1 = data['WAVELENGTH']
    col2 = data['FLUX']
    col3 = data['ERROR']
    spec = np.zeros((int(np.size(col1)),3))
    spec[:,0] = col1
    spec[:,1] = col2
    spec[:,2] = col3
elif qname.endswith('.dat'):
    data = np.loadtxt(folder+qname)
    col1 = data[:,0]
    col2 = data[:, 1]
    col3 = data[:, 2]



if rebin:
    n = 1
    x = rebin_arr(col1, n)
    y, err = rebin_weight_mean(col2,col3,n)
    plt.step(x,y,'red')
    plt.step(x,err,'red')
    plt.title('rebinning')
    plt.show()
else:
    x,y,err = col1,col2,col3
sp  = spectrum(x=x, y=y, err=1*err,name=qname)



if normalize:
    sp.normalize(12)

mask_lines = sp.x<0
if 'J1007' in qname:
    sp.z_abs = 0.8839
    sp.z_qso = 1.047
    mask_lines = (sp.x>14.2)*(sp.x<14.4) + (sp.x>15.5)*(sp.x<15.8) + (sp.x>17.4)*(sp.x<17.7)+ (sp.x>18.3)*(sp.x<18.55)+ (sp.x>21.4)*(sp.x<21.62)
    sp.mask = ~mask_lines * (sp.x > 7 * (1 + sp.z_abs)) * (sp.x < 12.2 * (1 + sp.z_abs)) * (sp.y != 0)
    sp.cont_ref_points = np.array([6.6,8.08, 11, 12.2])

    sp.fit_emiss=True
    sp.ston = 80


if 'J1017' in qname:
    sp.z_abs = 1.118
    sp.z_qso = 1.222
    mask_lines =  ((sp.x>11.86)*(sp.x<11.91) + (sp.x>12.15)*(sp.x<12.3) + (sp.x>13.55)*(sp.x<14.3)+ (sp.x>15.36)*
                   (sp.x<15.41)+ (sp.x>15.5)*(sp.x<15.8)+(sp.x > 16.6) * (sp.x < 18.2)+ (sp.x > 21.1) * (sp.x < 21.7)+ (sp.x > 24.8) *
                   (sp.x < 25.4) +(sp.x>20.7)*(sp.x<20.9)+(sp.x > 18.8) * (sp.x < 19.1))
    sp.mask = ~mask_lines * (sp.x > 7 * (1 + sp.z_abs)) * (sp.x < 12.3 * (1 + sp.z_abs)) * (sp.y != 0)
    sp.cont_ref_points = np.array([7,9.1, 11, 11.6])

    folder = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/J1017/'
    filename = 'continuum_j1017.dat'
    continuum = np.loadtxt(folder + filename)
    sp.cont_interp = interp1d(continuum[:, 0], continuum[:, 1],fill_value='extrapolate')

    sp.fit_emiss = True
    sp.ston = 60

if 'AO0235' in qname:
    sp.z_abs = 0.524
    sp.z_qso = 0.94
    mask_lines = (sp.x > 11.88) * (sp.x < 12.44) + (sp.x > 14.73) * (sp.x < 14.94) + (sp.x > 22.9) * (sp.x < 23.4) + (sp.x > 24.6) * (sp.x < 24.8)
    sp.mask = ~mask_lines * (sp.x > 7 * (1 + sp.z_abs)) * (sp.x < 13 * (1 + sp.z_abs)) * (sp.y != 0)
    sp.cont_ref_points = np.array([7, 8.5, 11,12.6])

    sp.fit_emiss = False
    sp.ston = 80

if 'J0901' in qname:
    sp.z_abs = 1.019
    sp.z_qso = 2.0934
    mask_lines = sp.x<0
    #mask_lines = (sp.x > 13.4) * (sp.x < 15.4) + (sp.x > 23.4) * (sp.x < 23.8)
    sp.mask = ~mask_lines * (sp.x > 8 * (1 + sp.z_abs)) * (sp.x < 12.3 * (1 + sp.z_abs)) * (sp.y != 0)
    sp.cont_ref_points = np.array([7, 8.5, 9.5, 11.4])

    if 1:
        sp.fit_emiss=True
        sp.ston = 45
    else:
        sp.fit_emiss = False
        sp.ston = 45

if 'J0900' in qname:
    sp.z_abs = 1.051
    sp.z_qso = 1.992
    sp.mask = ~mask_lines * (sp.x > 6.8 * (1 + sp.z_abs)) * (sp.x < 12.6 * (1 + sp.z_abs)) * (sp.y != 0)
    sp.cont_ref_points = np.array([6.6,8.08, 10, 12.04])

    folder = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/J0900/'
    filename = 'continuum_j0900.dat'
    continuum = np.loadtxt(folder + filename)
    sp.cont_interp = interp1d(continuum[:, 0], continuum[:, 1])

    sp.fit_emiss=False
    sp.ston = 35

if 1:
    abs_model_folder1 = './../data/dust_templates/Templates/profiles/Lab-Templates/'
    emission_template = 'SpoonAmOliv.dat'
    filename = abs_model_folder1 + emission_template
    sp.em_line_model = np.loadtxt(filename)

if 1:
    def gaussian(x, mu, sig):
        return (1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
        )
    x = np.linspace(5,7,100)
    pah_6_2_profile = gaussian(x=x,mu=6.25,sig=0.2)
    pah_6_2_tau = np.log(pah_6_2_profile+1)
    if 0:
        plt.subplots()
        plt.plot(x,pah_6_2_profile)
        plt.plot(x, pah_6_2_tau)
        plt.plot(x, np.exp(pah_6_2_tau)-1,ls='--')
        plt.show()
    sp.pah_em_line_model = np.zeros((100,2))
    sp.pah_em_line_model[:,0] =x
    sp.pah_em_line_model[:,1] =np.array(pah_6_2_tau)

# mask fit regions

if 1:
    plt.subplots()
    plt.step(sp.x,sp.y,color='black')
    plt.step(sp.x,sp.err,color='red',ls=':')
    plt.plot(sp.x[sp.mask],sp.x[sp.mask]*0+1.2,'o',color='blue',markersize=5)
    plt.show()

# show emission lines
if 0:
    fig,ax = plt.subplots(2,1)
    ax[0].plot(sp.x/(1+sp.z_abs),sp.y)
    ax[1].plot(sp.x/(1+sp.z_qso),sp.y)


    filename = '/home/slava/science/research/kulkarni/JWST-DLAs/lines_data/NGC7469.csv'
    l_w,l_n = [],[]
    import csv
    with open(filename, mode='r') as file:
        csvFile = csv.reader(file)
        for k, lines in enumerate(csvFile):
            print(lines)
            l_n.append(lines[0])
            l_w.append(float(lines[1]))
    for axs in ax[:]:
        for l,n in zip(l_w,l_n):
            axs.axvline(l,ls='--')
            axs.text(l,1,n,rotation=90,color='red')

    ax[0].set_ylabel('Flux')
    ax[0].set_xlabel('Wavelength (Galaxy)')
    ax[1].set_xlabel('Wavelength (Quasar)')

    plt.show()

# plot profiles for all dust templates
if 0:
    #abs_model = np.loadtxt('./../data/dust_templates/Templates/profiles/Lab-Templates/SpoonAmOliv.dat')
    path = './../data/dust_templates/Templates/profiles/Obs-Template/'
    fig,ax = plt.subplots(1,2)
    for (dirpath, dirname, filenames) in os.walk(path):
        print(filenames)
        for f in filenames:
            abs_model = np.loadtxt(dirpath+'/'+f)
            abs_model[:,1]/=np.nanmax(abs_model[:,1])
            ax[0].plot(abs_model[:,0],abs_model[:,1],label=f)
            ax[0].axhline(0)
            ax[0].set_ylabel('flux')
            ax[0].set_xlabel('wavelength')
            ax[0].set_title('Profile of the template: f(x)')
            An=1
            ax[1].axhline(1)
            ax[1].plot(abs_model[:,0],dust_abs_model(An*abs_model[:,1]),label=f)
            ax[1].set_ylabel('flux')
            ax[1].set_xlabel('wavelength')
            ax[1].set_title('Absorption profile exp(-A*f(x))')
            ax[0].legend()
            ax[1].legend()

    plt.show()


    path = './../data/dust_templates/Templates/profiles/Lab-Templates/'
    fig,ax = plt.subplots()
    for (dirpath, dirname, filenames) in os.walk(path):
        print(filenames)
        for f in filenames:
            abs_model = np.loadtxt(dirpath+'/'+f)
            abs_model[:,1]/=np.nanmax(abs_model[:,1])
            An=2
            ax.axhline(1)
            ax.plot(abs_model[:,0],dust_abs_model(An*abs_model[:,1]),label=f)
            ax.set_ylabel('flux')
            ax.set_xlabel('wavelength')
            ax.set_title('Absorption profile exp(-A*f(x))')
            ax.legend()
            ax.set_xlim(5,20)
            ax.set_ylim(-0.1, 1.2)
    plt.show()

def color_for_fit(abs_label):
    filename = './../data/dust_templates/Templates/profiles/namelist.csv'
    names, mode, label = [], [],[]
    import csv
    with open(filename, mode='r') as file:
        csvFile = csv.reader(file)
        for k, lines in enumerate(csvFile):
            names.append(lines[0])
            mode.append(lines[1])
            label.append(lines[2])
    names,mode,label = np.array(names),np.array(mode),np.array(label)
    mask = abs_label == names
    color = {}
    color['1']='red'
    color['2']='blue'
    color['3']='tab:green'
    color['4'] = 'tab:orange'
    n = mode[mask][0]
    return color[n],label[mask][0]
def plot_fit(sp=sp, abs_label = 'model',sa_fit = sp.y, sc_fit=-1, chiqred =1.0,save_fig=True,plot_sigma=True,continuum=-1,theta=[0,0,0]):
    fig, ax = plt.subplots(figsize=(6, 6))
    fontsize = 10

    lcolor,abs_label = color_for_fit(abs_label=abs_label)
    #abs_label = abs_label.split('.')[0]
    ax.errorbar(x=sp.x[sp.mask] / (1 + sp.z_abs), y=sp.y[sp.mask], yerr=sp.err[sp.mask], ds='steps-mid', ls='-', color='black', lw=0.5,
            label='MIRI/MRS',ecolor='lightgrey')
    ax.errorbar(x=sp.x / (1 + sp.z_abs), y=sp.y, yerr=sp.err, ds='steps-mid', ls='-',
                color='grey', lw=0.5,zorder=-10)

    if np.sum(continuum)!=-1:
        ax.plot(sp.x / (1 + sp.z_abs),continuum,'-',color='red')
        x_ref = sp.cont_ref_points
        An, c1, c2, c3,c4 = sp.theta
        yscale = []
        for e in x_ref:
            mask = (sp.x > e * (1 + sp.z_abs) - 0.2) * (sp.x < e * (1 + sp.z_abs) + 0.2)
            yscale.append(np.nanmedian(sp.err[mask]))
        yscale = np.array(yscale)
        ax.plot(x_ref,np.array([c1,c2,c3,c4])*yscale+1,'o',markersize=10,order=10,lw=2)

    # ax.set_title(abs_label + ';   $\\chi$/dof='+str(chiq),color=lcolor)

    if 'J1007+2853' in qname:
        for l, w in zip(['[ArII]', '[NeVI]', '[SIV]'], [6.98, 7.65, 10.51]):
            win = signal.windows.hann(50)
            filtered = interp1d(sp.x,signal.convolve(sp.y, win, mode='same') / sum(win))
            #ax.text(w * (1 + sp.z_qso) / (1 + sp.z_abs) * (1 - 10000 / 3e5), 1.2, l, fontsize=fontsize - 1, rotation=90, color='red')
            ax.text(w * (1 + sp.z_qso) / (1 + sp.z_abs) * (1 - 10000 / 3e5),
                    filtered(w * (1 + sp.z_qso))+0.1, l, fontsize=fontsize - 1, rotation=90, color='red')
    if 'J1017+4749' in qname:
        for l, w in zip(['PAH 7.6'], [7.55]):
            ax.text(w * (1 + sp.z_qso) / (1 + sp.z_abs) * (1 - 10000 / 3e5),
                    1.2, l, fontsize=fontsize - 1, rotation=90, color='red')

    plot_emiss = True
    if np.sum(sc_fit)==-1:
        plot_emiss = False
        sc_fit = np.ones_like(sp.x)
    sp_fit=sa_fit*sc_fit

    ax.plot(sp.x / (1 + sp.z_abs), sp_fit, color=lcolor, ls='-', label=abs_label,zorder=10)
    if plot_emiss:
        ax.plot(sp.x / (1 + sp.z_abs), sa_fit, color=lcolor, ls='--', label=abs_label+'(Abs)', zorder=10)
        ax.plot(sp.x / (1 + sp.z_abs), sc_fit, color='tab:orange', ls='-', label=abs_label+'(Emis)',zorder=10)

    if np.sum(continuum) !=-1:
        ax.fill_between(sp.x / (1 + sp.z_abs), continuum, sp_fit, color=lcolor,
                    alpha=0.3)
        ax.axhline(1,ls=':',color='tab:green')
    else:
        ax.fill_between(sp.x / (1 + sp.z_abs), sp_fit[0], sp_fit, color=lcolor,
                        alpha=0.3,zorder=10)

    if plot_sigma:
        sigma = (sp.y[sp.mask] - sp_fit[sp.mask])/sp.err[sp.mask]*0.02 + 1.4
        ax.plot(sp.x[sp.mask] / (1 + sp.z_abs), sigma, 'o', markersize=1,color=lcolor,alpha=0.5)
        ax.axhline(1.4,color='tab:green',lw=0.7)
        ax.axhline(1.4+0.06,color='tab:green',lw=0.5)
        ax.axhline(1.4 - 0.06, color='tab:green',lw=0.5)
        ax.fill_between(ax.get_xlim(), 1.4-0.06, 1.4+0.06,color='grey',alpha=0.2 )
    ax.axhline(0, ls=':', c='black')


    ax.set_title(abs_label + '; $\\chi^2/dof$=' + str(round(chiqred, 2))+'; $\\tau_{10}=}$'+str(round(theta[1], 2)))
    ax.set_ylabel('Normalized flux', fontsize=fontsize)
    ax.set_xlabel('Absorption restframe wavelength, $\\mu$m',
                  fontsize=fontsize)

    ax.axvline(9.7 , color='tab:green', ls='--')
    ax.axvline(9.7 * (1 + sp.z_qso)/(1 + sp.z_abs), color='tab:orange', ls='--')

    ax.set_xlim(6.5, 13.5)
    #ax.set_xlim(6.5, 18)
    if plot_sigma:
        ax.set_ylim(-0.1, 1.6)
    else:
        ax.set_ylim(0.5, 1.2)
    ax.tick_params(which='both', width=1, direction='in',
                        labelsize=fontsize,
                        right='True',
                        top='True')
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', length=3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(loc='lower left')

    #plt.show()

    if save_fig:
        fig.savefig('./../output/scripts/fit/' + (qname.split('.')[0]).split('_')[-1] + '/' + abs_model_name.split('.dat')[0] + '.pdf',
                bbox_inches='tight')
    else:
        plt.show()
        print()



if __name__ == '__main__':

    mode = 'run_mcmc'
    if mode == 'run_mcmc':
        # run for templates in the list
        abs_model_folder1 = './../data/dust_templates/Templates/profiles/Lab-Templates/'
        abs_model_folder2 = './../data/dust_templates/Templates/profiles/Obs-Template/'
        fn = 1
        for abs_model_folder in [abs_model_folder1,abs_model_folder2]:
            for (dirpath, dirname, filenames) in os.walk(abs_model_folder):
                for f in filenames:
                    print(fn,f)
                    fn+=1
                    #############
                    abs_model_folder = dirpath
                    abs_model_name = f
                    debug = False
                    calc_models = True
                    run_mcmc =True
                    ##############

                    class silicate_model():
                        def __init__(self, folder ='', template_name='',spec=sp):
                            if template_name != '':
                                self.folder = folder
                                self.template_name = template_name
                                self.read_abs_template()
                                self.spec =spec


                        def read_abs_template(self,normalize=True,set_xlim=True):
                            abs_model = np.loadtxt(self.folder+self.template_name)
                            if normalize:
                                abs_model[:, 1] /= np.nanmax(abs_model[:, 1])
                            if set_xlim:
                                mask = (abs_model[:, 0] > 0) * (abs_model[:, 0] < 20)
                                temp = np.zeros((np.sum(mask), 2))
                                temp[:, 0] = abs_model[:, 0][mask]
                                temp[:, 1] = abs_model[:, 1][mask]
                            if 0:
                                plt.subplots()
                                plt.plot(abs_model[:, 0], abs_model[:, 1])
                                plt.show()

                            self.abs_model = np.array(temp)

                        def continuum_variation_using_ref_points(self,theta,sp=sp,debug=True):
                            An,c1,c2,c3,c4 = theta
                            c = np.array([c1,c2,c3,c4])
                            # x-coord of continuum ref points
                            x_ref_points = sp.cont_ref_points
                            # set y-range for continuum variation
                            yscale = []
                            for e in x_ref_points:
                                mask = (sp.x>e*(1+sp.z_abs)-0.2)*(sp.x<e*(1+sp.z_abs)+0.2)
                                yscale.append(np.nanmedian(sp.err[mask]))
                            yscale = np.array(yscale)
                            y_ref_points = c*yscale + 1
                            z = np.polyfit(x_ref_points*(1+sp.z_abs), y_ref_points, 3)
                            p = np.poly1d(z)

                            if debug:
                                plt.subplots()
                                plt.plot(sp.x, sp.y)
                                plt.plot(x_ref_points*(1+sp.z_abs),y_ref_points,'o')
                                plt.plot(sp.x, p(sp.x))
                                plt.ylim(0, 1.2)
                                plt.show()
                            self.continuum_vary_function = p
                            return p

                        def simple_continuum_variation_model(self,x=sp.x,cont_level=0):
                            self.cont_level = cont_level
                            f = np.ones_like(x)*(1+cont_level)
                            self.continuum = f
                            return f

                        #calculate absorption spectrum
                        def absorption_model(self, x=sp.x, zabs=0, tau_abs=1, smoothing=False,debug=False):
                            f = np.ones_like(x)
                            mask = (x >= self.abs_model[0, 0] * (1 + zabs)) * (x <=  self.abs_model[-1, 0] * (1 + zabs))
                            if np.sum(mask) > 0:
                                f_interp = interp1d(self.abs_model[:, 0],  self.abs_model[:, 1],fill_value=(0,0),bounds_error=False)
                                f[mask] = dust_abs_model(tau_abs * f_interp(x[mask] / (1 + zabs)))
                            if smoothing:
                                from scipy import signal
                                win_size = 1000
                                win = signal.windows.hann(win_size)
                                filtered = signal.convolve(f, win, mode='same') / sum(win)
                                s = np.arange(len(x))
                                mask = (s > win_size / 2) * (s < len(x) - win_size / 2)*f>0.98
                                f[mask] = filtered[mask]
                                f = savgol_filter(f, 100, 1)
                            if debug:
                                plt.subplots()
                                plt.plot(x,dust_abs_model(tau_abs * f_interp(x / (1 + zabs))), label='orig')
                                plt.plot(x,f, label='smoothed')
                                plt.legend()
                                plt.legend()
                                #plt.show()

                            self.absorption_profile = f
                            return f

                        def emission_line_model(self,x=sp.x, zem = 0, tau_em = 0,smoothing=True,debug=False, case = 'silicate'):
                            #read_template
                            if case == 'silicate':
                                self.spec.em_line_model[:, 1] /= np.nanmax(self.spec.em_line_model[:, 1])
                                mask = (self.spec.em_line_model[:, 0] > 0) * (self.spec.em_line_model[:, 0] < 20)
                                temp = np.zeros((np.sum(mask), 2))
                                temp[:, 0] = self.spec.em_line_model[:, 0][mask]
                                temp[:, 1] = self.spec.em_line_model[:, 1][mask]
                            elif case == 'pah6_2':
                                temp = np.zeros_like(self.spec.pah_em_line_model)
                                temp[:,0] = self.spec.pah_em_line_model[:,0]
                                temp[:,1] = self.spec.pah_em_line_model[:,1]

                            self.em_line_template = np.array(temp)
                            #
                            #calc emission line
                            #
                            f = np.zeros_like(x)
                            mask = (x >= self.em_line_template[0, 0] * (1 + zem)) * (x <= self.em_line_template[-1, 0] * (1 + zem))
                            if np.sum(mask) > 0:
                                f_interp = interp1d(self.em_line_template[:, 0], self.em_line_template[:, 1])
                                f[mask] = dust_emiss_line_model(tau_em * f_interp(x[mask] / (1 + zem)))
                            if smoothing and 1:
                                f = savgol_filter(f, int(0.1/(x[1]-x[0])), 1)
                            self.emission_profile = f
                            if debug:
                                plt.subplots()
                                plt.plot(x, f)
                                plt.show()
                            #read and divide to continuum
                            if 'J1017' in self.spec.name and 1:
                                f = f / (self.spec.cont_interp(sp.x) / self.spec.cont_interp(9.7 * (1 + self.spec.z_qso)))
                                if 0:
                                    plt.subplots()
                                    plt.plot(sp.x,f)
                                    plt.plot(sp.x,self.spec.cont_interp(sp.x)/self.spec.cont_interp(9.7*(1+sp.z_qso)))
                                    plt.plot(sp.x,f/(self.spec.cont_interp(sp.x)/self.spec.cont_interp(9.7*(1+sp.z_qso))))
                                    plt.show()
                            return f

                        #calculate convolution
                        def conv_abs_model(self, x, f, mask=None):
                            f_interp = interp1d(x, f)
                            xnew = [x[0]]
                            npix = 1
                            for k in range(int(1e6)):
                                resolution = Miri_res(xnew[k])
                                delta = xnew[k] / resolution / 2.35 / npix
                                if xnew[k] + delta > x[-1]:
                                    xnew.append(x[-1])
                                    break
                                # print(xnew[k]+delta,f_interp(xnew[k]+delta))
                                if f_interp(xnew[k] + delta) < 1 - 1e-2:
                                    npix = 1
                                else:
                                    npix = 8
                                delta = xnew[k] / resolution / 2.35 / npix
                                x1 = xnew[k] + delta
                                if x1 > x[-1]:
                                    xnew.append(x[-1])
                                    break
                                else:
                                    xnew.append(x1)
                            xnew = np.array(xnew)
                            fnew = f_interp(xnew)

                            # mask pixel where to calculate convolution with PSF
                            mask_conv = fnew < 1

                            composed_flux = np.zeros_like(fnew)
                            composed_flux[~mask_conv] = 1
                            composed_flux[mask_conv] = convolveflux(xnew[mask_conv], fnew[mask_conv], Miri_res(xnew[mask_conv]))
                            f_interp = interp1d(xnew, composed_flux)

                            return f_interp
                        def calc_absorption(self, l=sp.x, tau=0.5, timer=False, method='direct',x_grid=sp.x,zabs=sp.z_abs):
                            if timer:
                                start = time.time()

                            if method == 'interpolation':
                                y = np.zeros_like(x_grid)
                                for i, x in enumerate(x_grid):
                                    f = interp1d(tau_grid, models_grid[:, i])
                                    y[i] = f(tau)

                                f = interp1d(x_grid, y,fill_value='extrapolate')
                                y = f(l)
                            elif method == 'direct':
                                y = absorption_model(x=l, zabs=zabs, An=tau)

                            if timer:
                                end = time.time()
                                print('Time/per run:', end - start)
                            return y

                    spec_model = silicate_model(folder = abs_model_folder, template_name=abs_model_name)

                    def log_likelihood_for_abs(theta, sp=sp, timer=False, debug=False,spec_model=spec_model):
                        tau_abs = theta
                        l = sp.x
                        flux = sp.y
                        flux_err = sp.err

                        if timer:
                            start = time.time()

                        abs_spectrum = spec_model.absorption_model(x=l, zabs=sp.z_abs, tau_abs=tau_abs)

                        chi = -0.5 * np.nansum(np.power(flux[sp.mask] - abs_spectrum[sp.mask], 2) / 0.05**2) #np.power(flux_err[sp.mask], 2))

                        if timer:
                            end = time.time()
                            print('Time/per run:', end - start, ' chiq:', chi)
                        if debug:
                            return abs_spectrum, chi
                        else:
                            return chi

                    def log_likelihood_for_cont_abs_and_emiss(theta, sp=sp, debug=False,spec_model=spec_model,show_model=False,fit_emiss=True):
                        cont_level, tau_abs, tau_em = theta
                        l = sp.x
                        flux = sp.y
                        flux_err = sp.err

                        fit_emiss = sp.fit_emiss

                        cont_model  = spec_model.simple_continuum_variation_model(x=l,cont_level=cont_level)
                        abs_model =  spec_model.absorption_model(x=l, tau_abs=tau_abs,zabs=sp.z_abs)
                        emiss_model = np.zeros_like(abs_model)
                        if fit_emiss:
                            emiss_model =  spec_model.emission_line_model(x=l, tau_em=tau_em,zem=sp.z_qso)
                            y = cont_model * abs_model*(emiss_model+1)
                        else:
                            y = cont_model * abs_model
                        chi = -0.5 * np.nansum(np.power(flux[sp.mask] - y[sp.mask], 2) / np.power(flux_err[sp.mask], 2))
                        #chi = -0.5 * np.nansum(np.power(flux[sp.mask] - y[sp.mask], 2) / np.power(0.02, 2))

                        if show_model:
                            plt.subplots()
                            plt.plot(l,cont_model,label='cont')
                            plt.plot(l,abs_model,label='abs')
                            plt.plot(l,emiss_model,label='em')
                            plt.plot(l[sp.mask],flux[sp.mask] ,label='spec')
                            plt.plot(l[sp.mask],flux_err[sp.mask], label='spec err')
                            plt.plot(l,y,label='Total',color='black')
                            plt.axvline(9.7*(1+sp.z_abs),color='orange',ls='--')
                            plt.axvline(9.7 * (1 + sp.z_qso),color='green',ls='--')

                            plt.title(str(theta))
                            plt.legend()
                            plt.show()
                        if debug:
                            print('chi', chi, np.sum(sp.mask))
                            sc = cont_model*(emiss_model+1)
                            sa = abs_model
                            return sc, sa, chi
                        else:
                            return chi

                    def log_prior_for_abs(tau_abs =1):
                        #An = theta
                        if (tau_abs < 0) + (tau_abs >2):
                            return -np.inf
                        else:
                            return 0.0

                    def log_prior_for_cont_varitation_using_ref_points(theta):
                        [c1,c2,c3,c4] = theta
                        if np.abs(c1)>1 or np.abs(c2)>1 or np.abs(c3)>1 or np.abs(c4)>1:
                            return -np.inf
                        else:
                            return 0.0

                    def log_prior_for_cont_abs_and_emiss(theta):
                        cont_level, tau_abs, tau_em = theta
                        ston=sp.ston
                        if sp.fit_emiss == True:
                            if np.abs(cont_level)>(1/ston) or (tau_abs)<=0.04 or (tau_abs)>2 or (tau_em)<0: # or (tau_em)>1e-2:
                                return -np.inf
                            else:
                                return 0.0
                        elif sp.fit_emiss == False:
                            if np.abs(cont_level)>(1/ston) or (tau_abs)<=0.02 or (tau_abs)>2 or (tau_em)<0 or (tau_em)>1e-4:
                                return -np.inf
                            else:
                                return 0.0

                    def log_likelihood_for_cont_variation_and_abs(theta, sp=sp, timer=False, debug=False):
                        An, c1,c2,c3,c4 = theta
                        l = sp.x[sp.mask]
                        flux = sp.y
                        flux_err = sp.err

                        if timer:
                            start = time.time()

                        abs_spectrum = absorption_model(x=l, zabs=sp.z_abs, An=An)
                        cont_model =  continuum_variation_ref_points(theta=theta,sp=sp,debug=False)

                        synthetic = abs_spectrum*cont_model(l)

                        chi = -0.5 * np.nansum(
                            np.power(flux[sp.mask] - synthetic, 2) / np.power(flux_err[sp.mask], 2))

                        if timer:
                            end = time.time()
                            print('Time/per run:', end - start, ' chiq:', chi)
                        if debug:
                            abs_spectrum = absorption_model(x=sp.x, zabs=sp.z_abs, An=An)
                            synthetic = abs_spectrum * cont_model(sp.x)
                            return synthetic, chi
                        else:
                            return chi


                    if 1:
                        ''' run mcmc for model with free parameters:
                        alpha = slope power law continuum,
                        F0 = continuum normalization factor
                        An = amplitude of absorption
                        '''
                        nwalkers = 200
                        nsteps =300

                        # continuum fit + absorption + emiss
                        case = 'continuum fit + absorption + emiss'
                        #case = 'continuum fit + absorption'
                        if case == 'absorption':
                            ndim = 1
                            par_names = ["tau_abs"]
                            init = [0.5]
                            init_range = [0.5]

                            def log_probability(theta):
                                tau_abs = theta
                                lp = log_prior_for_abs(tau_abs)
                                if not np.isfinite(lp):
                                    return -np.inf
                                return lp + log_likelihood_for_abs(theta)

                        elif case == 'continuum fit + absorption + emiss':
                            ndim = 3
                            par_names = ["cont_level", "tau_abs", "tau_em"]
                            if sp.fit_emiss:
                                init = [0.0, 0.2, 0.2]
                                init_range = [0.2, 0.1, 0.1]
                            else:
                                init = [0.0, 1, 1.0e-5]
                                init_range = [0.1, 0.5, 1e-6]
                            def log_probability(theta):
                                lp = log_prior_for_cont_abs_and_emiss(theta)
                                if not np.isfinite(lp):
                                    return -np.inf
                                chi = log_likelihood_for_cont_abs_and_emiss(theta)
                                return lp + chi

                        if case == 'continuum fit + absorption':
                            ndim = 5
                            par_names = ["An", "c1", "c2", "c3","c4"]
                            init = [1.01, 0, 0, 0,0]
                            init_range = [0.1, 1, 1, 1,1]


                            def log_probability(theta):
                                An, c1, c2, c3, c4 = theta
                                lp = log_prior_for_abs(An) + log_prior_for_cont_varitation_using_ref_points(
                                    [c1, c2, c3, c4])
                                if not np.isfinite(lp):
                                    return -np.inf
                                chi = log_likelihood_for_cont_variation_and_abs(theta)
                                return lp + chi



                        initial_pos = []

                        # runc mcmc
                        if run_mcmc:
                            for i in range(nwalkers):
                                prob = -np.inf
                                while prob == -np.inf:
                                    rndm = np.random.randn(ndim)
                                    wal_pos = init + init_range * rndm
                                    prob = log_probability(theta=wal_pos)
                                initial_pos.append(wal_pos)
                                pos = [init + init_range * np.random.randn(ndim) for i in range(nwalkers)]

                            from multiprocessing import Pool

                            with Pool() as pool:
                                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
                                start = time.time()
                                sampler.run_mcmc(initial_pos, nsteps, progress=True)
                                end = time.time()
                                multi_time = end - start
                                print("Multiprocessing took {0:.1f} seconds".format(multi_time))

                            samples = sampler.chain[:, :, :]

                            with open('./../temp/mcmc.pkl', 'wb') as f:
                                pickle.dump(samples, f)

                        with open('./../temp/mcmc.pkl', 'rb') as f:
                            samples = pickle.load(f)

                        means = np.zeros((ndim, nsteps))
                        vars = np.zeros((ndim, nsteps))
                        single = np.zeros((ndim, nsteps))
                        for i in range(nsteps):
                            for j in range(ndim):
                                means[j, i] = np.mean(samples[:, i, j])
                                vars[j, i] = np.std(samples[:, i, j])
                                single[j, i] = samples[5, i, j]

                        print('chain stats')
                        if 1:
                            fig, ax = plt.subplots(nrows=1, ncols=ndim)
                            if ndim > 1:
                                i = 0
                                for col in ax:
                                    print(i)
                                    if i < ndim:
                                        col.errorbar(np.arange(nsteps), means[i, :], yerr=vars[i, :],
                                                     fmt='-', color='black',
                                                     markeredgecolor='black', markeredgewidth=2, capsize=2,
                                                     ecolor='royalblue', alpha=0.7)
                                        col.plot(np.arange(nsteps), single[i, :], color='red')
                                    i += 1
                            #plt.show()

                        chain = samples[:, int(nsteps * 0.7):, :].reshape((-1, ndim))


                        if 1:
                            c = ChainConsumer()

                            c.add_chain(chain, parameters=par_names)
                            c.plotter.plot(filename="example.png", figsize="column")
                            res = c.analysis.get_summary(parameters=par_names)
                            print('res',res)


                            if ndim == 4 and 0:
                                alpha = res['alpha'][1]
                                F0 = res['F0'][1]
                                An = res['An'][1]
                                c0 = res['c0'][1]
                                theta = alpha, F0, c0, An

                                chiq *= -2
                                chiqred = chiq / (np.sum(sp.mask) - ndim)
                                if 1:
                                    fig, ax = plt.subplots(2, 1, sharex=True)
                                    fontsize = 12

                                    ax[0].plot(sp.x, cont * abs_m, label='Fit', c='red')
                                    ax[0].plot(sp.x, sp.y, zorder=-10, c='black', label='MIRI/MRS')
                                    ax[1].plot(sp.x, abs_m, label='Fit', c='red')
                                    ax[1].plot(sp.x, sp.y / cont, zorder=-10, c='black', label='MIRI/MRS')

                                    ax[0].set_xlim(4, 30)
                                    ax[1].set_xlim(4, 30)
                                    ax[1].set_ylim(-0.1, 1.5)
                                    ax[0].set_ylim(-5, 15)
                                    ax[0].set_ylabel('Flux', fontsize=fontsize)
                                    ax[1].set_ylabel('Normalized flux', fontsize=fontsize)

                                    ax[0].set_title(abs_model_name + ' chi2/dof=' + str(round(chiqred, 1)))
                                    for col in ax[:]:
                                        col.axhline(0, ls=':', c='black')
                                        col.tick_params(which='both', width=1, direction='in', labelsize=fontsize,
                                                        right='True',
                                                        top='True')
                                        col.tick_params(which='major', length=5)
                                        col.tick_params(which='minor', length=3)
                                        col.xaxis.set_minor_locator(AutoMinorLocator(4))
                                        col.xaxis.set_major_locator(MultipleLocator(2))
                                        # col.yaxis.set_minor_locator(AutoMinorLocator(5))
                                        # col.yaxis.set_major_locator(MultipleLocator(0.1))
                                        col.set_xlabel('Observed wavelength, $\\mu$m', fontsize=fontsize)
                                        col.legend(loc='lower right', fontsize=fontsize)
                            elif ndim == 5:
                                An = res['An'][1]
                                c1 = res['c1'][1]
                                c2 = res['c2'][1]
                                c3 = res['c3'][1]
                                c4 = res['c4'][1]
                                theta = An, c1,c2,c3,c4
                                sp.theta = theta
                                if res['An'][2] != None:
                                    An_err = res['An'][2] - res['An'][1]
                                else:
                                    An_err = None
                                fit_model, chiq = log_likelihood_for_cont_variation_and_abs(theta, debug=True)
                                chiq *= -2
                                chiqred = chiq / (np.sum(sp.mask) - ndim)
                                if 1:
                                    cont_model = continuum_variation_ref_points(theta=theta, sp=sp, debug=False)
                                    continuum = cont_model(sp.x)
                                    plot_fit(sp=sp, abs_label=abs_model_name, sp_fit=fit_model, chiqred=chiqred,
                                             save_fig=True,continuum=continuum)

                            if case == 'absorption':
                                theta = [res['tau_abs'][1]]
                                tau_abs = theta
                                if res['tau_abs'][2] != None:
                                    tau_abs_err = res['tau_abs'][2]-res['tau_abs'][1]
                                else:
                                    tau_abs_err= None
                                fit_model, chiq = log_likelihood_for_abs(theta, debug=True)
                                chiq *= -2
                                chiqred = chiq / (np.sum(sp.mask) - ndim)
                                if 0:

                                    fig, ax = plt.subplots()
                                    fontsize = 12

                                    ax.plot(sp.x/(1+sp.z_abs), abs_m, label='Fit', c='red')
                                    ax.plot(sp.x/(1+sp.z_abs), sp.y, zorder=-10, c='black', label='MIRI/MRS')

                                    ax.plot(sp.x[sp.mask]/(1+sp.z_abs), sp.x[sp.mask] * 0 + 1.2, 'o', color='blue', markersize=5)

                                    ax.set_xlim(4/(1+sp.z_abs), 30/(1+sp.z_abs))
                                    ax.set_ylim(-0.1, 1.5)
                                    ax.set_ylabel('Normalized flux', fontsize=fontsize)

                                    ax.set_title(abs_model_name + ' chi2/dof=' + str(round(chiqred, 1)))
                                    for col in [ax]:
                                        col.axhline(0, ls=':', c='black')
                                        col.tick_params(which='both', width=1, direction='in', labelsize=fontsize,
                                                        right='True',
                                                        top='True')
                                        col.tick_params(which='major', length=5)
                                        col.tick_params(which='minor', length=3)
                                        col.xaxis.set_minor_locator(AutoMinorLocator(4))
                                        col.xaxis.set_major_locator(MultipleLocator(2))
                                        # col.yaxis.set_minor_locator(AutoMinorLocator(5))
                                        # col.yaxis.set_major_locator(MultipleLocator(0.1))
                                        col.set_xlabel('Galaxy restrframe wavelength, $\\mu$m', fontsize=fontsize)
                                        col.legend(loc='lower right', fontsize=fontsize)
                                if 1:
                                    plot_fit(sp=sp, abs_label=abs_model_name, sa_fit=fit_model, chiqred=chiqred, save_fig=True)

                            elif case == 'continuum fit + absorption + emiss':
                                par_names = ["cont_level", "tau_abs", "tau_em"]
                                theta = [res[el][1] for el in par_names]
                                theta_err = []
                                for el in par_names:
                                    err= None
                                    if res[el][2] != None:
                                        err = res[el][2]-res[el][1]
                                    theta_err.append(err)

                                tau_abs = theta[1]
                                tau_abs_err =theta_err[1]


                                sc, sa, chiq = log_likelihood_for_cont_abs_and_emiss(theta, debug=True,show_model=False)
                                chiq *= -2
                                chiqred = chiq / (np.sum(sp.mask) - ndim)
                                print('chiq:',chiq, ' n=',np.sum(sp.mask), ' c_red ', chiqred )
                                print('theta',theta)
                                plot_fit(sp=sp, abs_label=abs_model_name, sa_fit=sa, sc_fit=sc, chiqred=chiqred, save_fig=True,theta=theta)

                            #plt.show()
                            if 1:
                                #alpha,F0,An,chqred = silicate_absorption_fit(debug=False, calc_models = True,run_mcmc=True)
                                with open('./../output/scripts/fit/'+(qname.split('.')[0]).split('_')[-1]+'/'+'fit_results.txt', 'a') as file:
                                    #file.write(abs_model_name+','+str(chiqred)+','+str(alpha)+','+str(F0)+','+str(c0)+','+str(An)+'\n')
                                    line = ''
                                    for el in theta:
                                        line +=', '+str(el)
                                    file.write(abs_model_name + ',' + str(chiqred) + ',' + str(tau_abs)+ ',' + str(tau_abs_err) + line + '\n')
                                    #plt.show()

    elif mode == 'plot_fig':
        abs_model_folder1 = './../data/dust_templates/Templates/profiles/Lab-Templates/'
        abs_model_folder2 = './../data/dust_templates/Templates/profiles/Obs-Template/'
        abs_model_folder_list =   [abs_model_folder1]
        if 0:
            abs_label = 'Amorph. Pyroxene'
            lcolor = 'tab:green'
        elif 0:
            abs_label = 'Mol. Cloud'
            lcolor = 'red'
        elif 1:
            abs_label = 'Crystal. Pyroxene'
            lcolor = 'blue'
        if 'J1007+2853' in qname:
            #J1007
            if 1:
                abs_model_name_list = ['Mont.dat']
                chiq = 2.36
                theta_list = [0.11]
            if 0:
                abs_model_name_list = ['SpoonAmOliv.dat']
                chiq = 2.77
                theta_list = [0.09]

        if 'J1017+4749' in qname:
            if 0:
                abs_model_name_list = ['AmOliv2.0.dat']
                chiq = 2.61
                theta_list = [0.074]
            if 0:
                abs_model_name_list = ['Levy.dat']
                chiq = 2.74
                theta_list = [0.086]
            if 1:
                abs_model_name_list = ['FayaT10.dat']
                chiq = 1.81
                theta_list = [0.25]
        if 'AO0235+164' in qname:
            if 0:
                abs_model_name_list = ['hypers.dat']
                chiq = 3.12
                theta_list = [0.069]
            if 0:
                abs_model_name_list = ['AmOliv1.5.dat']
                chiq = 2.74
                theta_list = [0.086]
            if 1:
                abs_model_name_list = ['SpoonU4N.dat']
                chiq = 2.58
                theta_list = [0.067]
        if 'J0901+2044' in qname:
            if 1:
                abs_model_name_list = ['vB-SSTc2d.dat']
                chiq = 2.59
                theta_list = [0.31]
            if 0:
                abs_model_name_list = ['C06-CDEporPYR.dat']
                chiq = 4.01
                theta_list = [0.266]
            if 0:
                abs_model_name_list = ['B2-FER21.dat']
                chiq = 3.52
                theta_list = [0.301]
        if 'J0900+0214' in qname:
            if 1:
                abs_model_name_list = ['SpoonU4N.dat']
                chiq = 2.47
                theta_list = [0.23]
            if 0:
                abs_model_name_list = ['C06-CDEporOLIV.dat']
                chiq = 2.37
                theta_list = [0.24]
            if 0:
                abs_model_name_list = ['bronzite.dat']
                chiq = 2.54
                theta_list = [0.27]



        label_list = [abs_label]
        line_ls = ['-']


        fig,ax = plt.subplots(figsize=(4,4))
        fontsize = 10
        ax.step(x=sp.x/(1+sp.z_abs),y=sp.y,where='mid',color='black',lw=1,label='MIRI/MRS')
        #ax.step(x=sp.x/(1+sp.z_abs),y=sp.y,where='mid',color='grey',lw=1,zorder=-1)

        ax.set_xlim(7 ,12)
        ax.set_ylim(0.5, 1.4)
        ax.set_ylabel('Normalized flux', fontsize=fontsize)
        ax.set_xlabel('Absorption restfrrame wavelength, $\\mu$m', fontsize=fontsize)
        ax.axhline(0, ls=':', c='black')
        #ax.set_title(abs_label + ';   $\\chi$/dof='+str(chiq),color=lcolor)

        if 'J1007+2853' in qname:
            for l,w in zip(['[ArII]','[NeVI]','[SIV]'],[6.98,7.65,10.51]):
                ax.text(w*(1+sp.z_qso)/(1+sp.z_abs)*(1-10000/3e5),1.2,l,fontsize=fontsize-1,rotation=90,color='red')
                #ax.axvline(w*(1+sp.z_qso)/(1+sp.z_abs),ls='--',color='red')
        if 'J1017+4749' in qname:
            for l,w in zip(['PAH 7.6'],[7.55]):
                ax.text(w*(1+sp.z_qso)/(1+sp.z_abs)*(1-10000/3e5),1.2,l,fontsize=fontsize-1,rotation=90,color='red')


        #ax.set_title(abs_model_name + ' chi2/dof=' + str(round(chiqred, 1)))
        for col in [ax]:
            col.tick_params(which='both', width=1, direction='in', labelsize=fontsize,
                            right='True',
                            top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=3)
            col.xaxis.set_minor_locator(AutoMinorLocator(5))
            col.xaxis.set_major_locator(MultipleLocator(1))
            col.yaxis.set_minor_locator(AutoMinorLocator(5))
            col.yaxis.set_major_locator(MultipleLocator(0.1))
            col.legend(loc='lower right', fontsize=fontsize)



        for abs_model_folder,abs_model_name,theta,label_f,ls_f in zip(abs_model_folder_list,abs_model_name_list,theta_list,label_list,line_ls):
            abs_model = np.loadtxt(abs_model_folder + abs_model_name)
            abs_model[:, 1] /= np.nanmax(abs_model[:, 1])
            mask = (abs_model[:, 0] > 0) * (abs_model[:, 0] < 20)
            temp = np.zeros((np.sum(mask), 2))
            temp[:, 0] = abs_model[:, 0][mask]
            temp[:, 1] = abs_model[:, 1][mask]
            abs_model = np.array(temp)
            sp.abs_model = np.array(abs_model)
            # calculate absorption spectrum
            def absorption_model(x=sp.x, zabs=0, An=1, template=abs_model, smoothing=True):
                f = np.ones_like(x)
                mask = (x >= template[0, 0] * (1 + zabs)) * (x <= template[-1, 0] * (1 + zabs))
                if np.sum(mask) > 0:
                    f_interp = interp1d(template[:, 0], template[:, 1])
                    f[mask] = dust_abs_model(An * f_interp(x[mask] / (1 + zabs)))
                if smoothing:
                    from scipy import signal
                    win_size = 100
                    win = signal.windows.hann(win_size)
                    filtered = signal.convolve(f, win, mode='same') / sum(win)
                    s = np.arange(len(x))
                    mask = sp.x<9*(1+sp.z_abs)
                    f[mask] = filtered[mask]

                return f
            def log_likelihood_for_abs(theta, sp=sp, timer=False, debug=False):
                An = theta
                l = sp.x
                flux = sp.y
                flux_err = sp.err

                if timer:
                    start = time.time()

                abs_spectrum = absorption_model(x=l, zabs=sp.z_abs, An=An)

                chi = -0.5 * np.nansum(np.power(flux[sp.mask] - abs_spectrum[sp.mask], 2) / np.power(flux_err[sp.mask], 2))

                if timer:
                    end = time.time()
                    print('Time/per run:', end - start, ' chiq:', chi)
                if debug:
                    return abs_spectrum, chi
                else:
                    return chi

            abs_m, chiq = log_likelihood_for_abs(theta, debug=True)
            print(chiq)
            ax.plot(sp.x/(1+sp.z_abs),abs_m,color=lcolor,label=label_f,ls=ls_f)
            ax.fill_between(sp.x/(1+sp.z_abs),abs_m*0+1,abs_m,color=lcolor,alpha=0.3)

            ax.legend(loc='lower left')


        plt.show()


    elif mode == 'plot_fig2':
        folder = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/spectro/J1007/'
        spec = np.loadtxt(folder+'fit.dat')
        cont = np.loadtxt(folder+'fit_cont.dat')

        qname = 'J1007+2853'
        coef = 0.5e-3
        sp = spectrum(x=spec[:,0], y=spec[:,1]*coef, err=spec[:,2], name=qname)
        sp.z_abs = 0.8839
        sp.z_qso = 1.047
        mask_lines = (np.abs(sp.x - 14.3) < 0.1) + (np.abs(sp.x - 15.67) < 0.15) + (np.abs(sp.x - 21.53) < 0.2)
        sp.mask = ~mask_lines * (sp.x > 8 * (1 + sp.z_abs)) * (sp.x < 24) * (sp.y != 0)

        sp_c = spectrum(x=cont[:,0], y=cont[:,1]*coef, name=qname)
        sp_c.z_abs = 0.8839
        sp_c.z_qso = 1.047


        abs_model_folder1 = './../data/dust_templates/Templates/profiles/Lab-Templates/'
        abs_model_folder2 = './../data/dust_templates/Templates/profiles/Obs-Template/'
        abs_model_folder_list =   [abs_model_folder1]

        if 1:
            abs_label = 'Crystal. Pyroxene'
            lcolor = 'blue'
        if 'J1007+2853' in qname:
            #J1007
            if 1:
                abs_model_name_list = ['Mont.dat']
                chiq = 2.36
                theta_list = [0.11]



        label_list = [abs_label]
        line_ls = ['-']


        fig,ax = plt.subplots(figsize=(4,4))
        fontsize = 10
        ax.step(x=sp.x,y=sp.y,where='mid',color='black',lw=1,label='MIRI/MRS')
        ax.plot(sp_c.x,sp_c.y , color='red', lw=1,label='Quasar cont.')

        #ax.step(x=sp.x/(1+sp.z_abs),y=sp.y,where='mid',color='grey',lw=1,zorder=-1)

        ax.set_xlim(5 ,25)
        ax.set_ylim(0, 40000*coef)
        ax.set_ylabel('Flux, mJy', fontsize=fontsize)
        ax.set_xlabel('Observed wavelength, $\\mu$m', fontsize=fontsize)
        ax.axhline(0, ls=':', c='black')
        #ax.set_title(abs_label + ';   $\\chi$/dof='+str(chiq),color=lcolor)

        if 'J1007+2853' in qname:
            for l,w,y in zip(['[ArII]','[NeVI]','[SIV]'],[6.98,7.65,10.51],[2e4,2.1e4,3.4e4]):
                ax.text(w*(1+sp.z_qso)*(1-10000/3e5),y*coef,l,fontsize=fontsize-1,rotation=90,color='red')
                #ax.axvline(w*(1+sp.z_qso)/(1+sp.z_abs),ls='--',color='red')
        if 'J1017+4749' in qname:
            for l,w in zip(['PAH 7.6'],[7.55]):
                ax.text(w*(1+sp.z_qso)*(1-10000/3e5),1.2,l,fontsize=fontsize-1,rotation=90,color='red')


        #ax.set_title(abs_model_name + ' chi2/dof=' + str(round(chiqred, 1)))
        for col in [ax]:
            col.tick_params(which='both', width=1, direction='in', labelsize=fontsize,
                            right='True',
                            top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=3)
            #col.xaxis.set_minor_locator(AutoMinorLocator(5))
            #col.xaxis.set_major_locator(MultipleLocator(1))
            #col.yaxis.set_minor_locator(AutoMinorLocator(5))
            #col.yaxis.set_major_locator(MultipleLocator(0.1))
            col.legend(loc='lower right', fontsize=fontsize)



        for abs_model_folder,abs_model_name,theta,label_f,ls_f in zip(abs_model_folder_list,abs_model_name_list,theta_list,label_list,line_ls):
            abs_model = np.loadtxt(abs_model_folder + abs_model_name)
            abs_model[:, 1] /= np.nanmax(abs_model[:, 1])
            mask = (abs_model[:, 0] > 0) * (abs_model[:, 0] < 20)
            temp = np.zeros((np.sum(mask), 2))
            temp[:, 0] = abs_model[:, 0][mask]
            temp[:, 1] = abs_model[:, 1][mask]
            abs_model = np.array(temp)
            sp.abs_model = np.array(abs_model)
            # calculate absorption spectrum
            def absorption_model(x=sp.x, zabs=0, An=1, template=abs_model, smoothing=True):
                f = np.ones_like(x)
                mask = (x >= template[0, 0] * (1 + zabs)) * (x <= template[-1, 0] * (1 + zabs))
                if np.sum(mask) > 0:
                    f_interp = interp1d(template[:, 0], template[:, 1])
                    f[mask] = dust_abs_model(An * f_interp(x[mask] / (1 + zabs)))
                if smoothing:
                    from scipy import signal
                    win_size = 100
                    win = signal.windows.hann(win_size)
                    filtered = signal.convolve(f, win, mode='same') / sum(win)
                    s = np.arange(len(x))
                    mask = sp.x<9*(1+sp.z_abs)
                    f[mask] = filtered[mask]

                return f
            def log_likelihood_for_abs(theta, sp=sp, timer=False, debug=False):
                An = theta
                l = sp.x
                flux = sp.y
                flux_err = sp.err

                if timer:
                    start = time.time()

                abs_spectrum = absorption_model(x=l, zabs=sp.z_abs, An=An)

                chi = -0.5 * np.nansum(np.power(flux[sp.mask] - abs_spectrum[sp.mask], 2) / np.power(flux_err[sp.mask], 2))

                if timer:
                    end = time.time()
                    print('Time/per run:', end - start, ' chiq:', chi)
                if debug:
                    return abs_spectrum, chi
                else:
                    return chi

            abs_m, chiq = log_likelihood_for_abs(theta, debug=True)
            print(chiq)
            f_int = interp1d(sp.x,sp.y)
            mask = (sp_c.x>8.4*(1+sp_c.z_abs)) * (sp_c.x<11*(1+sp_c.z_abs))
            ax.fill_between(sp_c.x[mask],sp_c.y[mask], f_int(sp_c.x[mask]),color=lcolor,alpha=0.3)

            ax.legend(loc='upper left')

            ax.text(6,3e4*coef,'z(Abs)=0.88')
            ax.text(6,2.8e4*coef,'z(QSO)=1.05')
            ax.text(16.2, 2.3e4*coef, '  Silicate\n absorption', rotation=30)


        plt.show()

    if mode == 'fit_cont':
        from numpy.polynomial.polynomial import polyval
        x= [7,10,12]
        x0 = np.linspace(x[0],x[-1],100)
        y = np.random.normal(1,0.03,3)
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        plt.subplots()
        plt.plot(x,y,'o')
        x0 = np.linspace(6,14,100)
        plt.plot(x0,p(x0))
        plt.ylim(0,1.2)
        plt.show()