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

    def normalize(self,x0=5,delta_x = 0.1):
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
    :return: obsevrd flux
    '''
    return np.exp(-f)

#


spectra = {}
#folder = '/home/slava/science/codes/python/jwst/output/specviewer/'
#f = 'J1007_fringe_corrected_2sigma_aperture.fits'
folder= '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/J1007/'
f='spectrum_6pix_rebinned.fits'
if f.endswith('fits'):
    hdu = fits.open(folder+f)
    data = hdu[1].data
    col1 = data['WAVELENGTH']
    col2 = data['FLUX']
    col3 = data['ERROR']
    spec = np.zeros((int(np.size(col1)),3))
    spec[:,0] = col1
    spec[:,1] = col2
    spec[:,2] = col3

sp  = spectrum(x=col1, y=col2, err=col3,name=f)
sp.normalize(5)
sp.z_abs = 0.8839
sp.z_qso = 1.047

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
            axs.text(l,6,n,rotation=90,color='red')

    ax[0].set_ylabel('Flux')
    ax[0].set_xlabel('Wavelength (Galaxy)')
    ax[1].set_xlabel('Wavelength (Quasar)')

    plt.show()

# show dust templates
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




abs_model_folder ='./../data/dust_templates/Templates/profiles/Lab-Templates/'
os.walk(abs_model_folder)
abs_model_name = 'SpoonAmOliv.dat'
#abs_model_name = 'fayalite.dat'
abs_model_name = 'horton.dat'



if __name__ == '__main__':

# run individual
    if 0:
        abs_model_folder ='./../data/dust_templates/Templates/profiles/Lab-Templates/'
        os.walk(abs_model_folder)
        #abs_model_name = 'SpoonAmOliv.dat'
        #abs_model_name = 'fayalite.dat'
        abs_model_name = 'horton.dat'
        if 1:
            debug = True
            calc_models = False
            run_mcmc = False
            abs_model = np.loadtxt(abs_model_folder + abs_model_name)
            abs_model[:, 1] /= np.nanmax(abs_model[:, 1])
            mask = (abs_model[:, 0] > 0) * (abs_model[:, 0] < 20)
            tmp = np.zeros((np.sum(mask), 2))
            tmp[:, 0] = abs_model[:, 0][mask]
            tmp[:, 1] = abs_model[:, 1][mask]
            abs_model = np.array(tmp)


            def continuum_model(x=sp.x, x0=5, alpha=1, y0=1):
                f = y0 * (x / x0) ** alpha
                return f


            def absorption_model(x=sp.x, zabs=0, An=1, template=abs_model, smoothing=False):
                f = np.ones_like(x)
                mask = (x >= template[0, 0] * (1 + zabs)) * (x <= template[-1, 0] * (1 + zabs))
                if np.sum(mask) > 0:
                    f_interp = interp1d(template[:, 0], template[:, 1])
                    f[mask] = dust_abs_model(An * f_interp(x[mask] / (1 + zabs)))
                if smoothing:
                    from scipy import signal
                    win_size = 10
                    win = signal.windows.hann(win_size)
                    filtered = signal.convolve(f, win, mode='same') / sum(win)
                    s = np.arange(len(x))
                    mask = (s > win_size / 2) * (s < len(x) - win_size / 2)
                    f[mask] = filtered[mask]

                return f


            def conv_abs_model(x, f, mask=None):
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


            # convolution test
            if debug:
                fig, ax = plt.subplots()
                An = 0.3
                ax.axhline(0, ls=':')
                ax.plot(abs_model[:, 0], dust_abs_model(An * abs_model[:, 1]), label=abs_model_name)
                ax.set_ylabel('flux')
                ax.set_xlabel('Restframe wavelength')
                ax.set_title('Absorption profile exp(-A*f(x))')
                ax.legend()
                plt.show()

                fig, ax = plt.subplots()
                f = absorption_model(x=sp.x, zabs=sp.z_abs, An=0.3)
                m = conv_abs_model(sp.x, absorption_model(x=sp.x, zabs=sp.z_abs, An=0.3))

                ax.plot(sp.x, absorption_model(x=sp.x, zabs=sp.z_abs, An=0.3, smoothing=False), label='model', ls='-')
                ax.plot(sp.x, f, label='model smooth', ls='--')
                ax.plot(sp.x, m(sp.x), label='PSF convloved')
                ax.set_xlabel('Observed wavelength')
                ax.set_title('Absorption profile exp(-A*f(x))')
                ax.legend()
                plt.show()

            # continuum test
            if 0:
                fig, ax = plt.subplots(3, 1)
                ax[0].plot(sp.x / (1 + sp.z_abs), sp.y, color='black')
                cont = continuum_model(x=sp.x / (1 + sp.z_abs), x0=5 / (1 + sp.z_abs), alpha=1.25)
                mask = (sp.x / (1 + sp.z_abs) <= 7) + (sp.x / (1 + sp.z_abs) > 7.5) * (sp.x / (1 + sp.z_abs) < 8) + (
                            sp.x / (1 + sp.z_abs) > 8.5)

                f = absorption_model(x=sp.x, zabs=sp.z_abs, An=0.3)
                m = conv_abs_model(sp.x, f)

                ax[0].plot(sp.x / (1 + sp.z_abs), cont)

                ax[1].plot(sp.x / (1 + sp.z_abs), sp.y / cont, color='black')
                ax[1].plot(sp.x[mask] / (1 + sp.z_abs), m(sp.x[mask]), color='red')

                ax[2].plot(sp.x / (1 + sp.z_abs), sp.y, color='black')
                ax[2].plot(sp.x / (1 + sp.z_abs), cont)
                ax[2].plot(sp.x / (1 + sp.z_abs), cont * m(sp.x))
                plt.show()

            # grid for An
            if 1:
                nmodels = 100
                fname = abs_model_name.split('.')[0] + '.pkl'
                gridpath = './../temp/dust/' + fname
                if calc_models:
                    lst = glob.glob('./../temp/dust/' + '*.pkl')
                    if './../temp/dust/' + fname not in lst:
                        tau = np.linspace(0, 2, nmodels)
                        models_grid = []
                        for i, t in enumerate(tau):
                            start = time.time()
                            m = conv_abs_model(sp.x, absorption_model(x=sp.x, zabs=sp.z_abs, An=t))
                            models_grid.append(m(sp.x))
                            end = time.time()
                            print('Time/per run', i, ': ', end - start)
                        models_grid = np.array(models_grid)
                        with open(gridpath, 'wb') as f:
                            pickle.dump([sp.x, tau, models_grid], f)
                    else:
                        with open(gridpath, 'rb') as f:
                            x_grid, tau_grid, models_grid = pickle.load(f)

                with open(gridpath, 'rb') as f:
                    x_grid, tau_grid, models_grid = pickle.load(f)


                def calc_absorption(l=sp.x, tau=0.5, timer=0):
                    if timer:
                        start = time.time()

                    y = np.zeros_like(x_grid)
                    for i, x in enumerate(x_grid):
                        f = interp1d(tau_grid, models_grid[:, i])
                        y[i] = f(tau)

                    f = interp1d(x_grid, y)

                    if timer:
                        end = time.time()
                        print('Time/per run:', end - start)
                    return f(l)


                # test
                if 0:
                    print('plot')
                    plt.subplots()
                    for i in range(nmodels):
                        plt.plot(sp.x, models_grid[i, :])
                    plt.plot(sp.x, calc_absorption(sp.x, 0.4), ls='--')
                    plt.show()


            def log_probability(theta):
                lp = log_prior(theta)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_likelihood(theta)


            def log_prior(theta):
                alpha, F0, An = theta
                if (alpha < 0) + (alpha > 3) + (An < 0.1) + (An > 2) + (F0 <= 0):
                    return -np.inf
                else:
                    return 0.0


            def log_likelihood(theta, sp=sp, timer=False, debug=False):
                alpha, F0, An = theta
                l = sp.x
                flux = sp.y
                flux_err = sp.err

                if timer:
                    start = time.time()
                cont = continuum_model(x=l, x0=5, alpha=alpha, y0=F0)
                abs_model = calc_absorption(l[sp.mask], An)

                y = cont[sp.mask] * abs_model
                chi = -0.5 * np.nansum(np.power(flux[sp.mask] - y, 2) / np.power(flux_err[sp.mask], 2))

                if timer:
                    end = time.time()
                    print('Time/per run:', end - start, ' chiq:', chi)
                if debug:
                    return cont, calc_absorption(l, An), chi
                else:
                    return chi


            if 1:
                ''' run mcmc for model with free parameters:
                alpha = slope power law continuum,
                F0 = continuum normalization factor
                An = amplitude of absorption
                '''
                nwalkers = 200
                nsteps = 50
                ndim = 3
                # nask regions
                mask_lines = (np.abs(sp.x - 12.72) < 0.06) + (np.abs(sp.x - 15.67) < 0.17) + (
                            np.abs(sp.x - 21.53) < 0.6)
                sp.mask = ~mask_lines * (sp.x < 24) * (sp.y != 0)
                # set effective errorbar
                sp.err[:] = 0.1

                init = [1.04, 1.1, 0.5]
                init_range = [0.5, 0.5, 0.2]
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
                if 1:
                    with open('./../temp/mcmc.pkl', 'rb') as f:
                        samples = pickle.load(f)

                if 1:
                    means = np.zeros((ndim, nsteps))
                    vars = np.zeros((ndim, nsteps))
                    single = np.zeros((ndim, nsteps))
                    for i in range(nsteps):
                        for j in range(ndim):
                            means[j, i] = np.mean(samples[:, i, j])
                            vars[j, i] = np.std(samples[:, i, j])
                            single[j, i] = samples[5, i, j]

                    print('chain stats')
                    if debug:
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

                    chain = samples[:, int(nsteps * 0.9):, :].reshape((-1, ndim))

                if 1:
                    c = ChainConsumer()
                    par_names = ["alpha", "F0", "An"]
                    c.add_chain(chain, parameters=par_names)
                    c.plotter.plot(filename="example.png", figsize="column")
                    res = c.analysis.get_summary(parameters=par_names)
                    print(res)

                    alpha = res['alpha'][1]
                    F0 = res['F0'][1]
                    An = res['An'][1]
                    theta = alpha, F0, An
                    cont, abs_m, chiq = log_likelihood(theta, debug=True)
                    chiq *= -2
                    chiqred = chiq / (np.sum(sp.mask) - ndim)
                    if debug:
                        fig, ax = plt.subplots(2, 1, sharex=True)
                        fontsize = 12

                        ax[0].plot(sp.x, cont * abs_m, label='Fit', c='red')
                        ax[0].plot(sp.x, sp.y, zorder=-10, c='black', label='MIRI/MRS')
                        ax[1].plot(sp.x, abs_m, label='Fit', c='red')
                        ax[1].plot(sp.x, sp.y / cont, zorder=-10, c='black', label='MIRI/MRS')

                        ax[0].set_xlim(4, 30)
                        ax[1].set_xlim(4, 30)
                        ax[1].set_ylim(0.1, 1.5)
                        ax[0].set_ylabel('Flux', fontsize=fontsize)
                        ax[1].set_ylabel('Normalized flux', fontsize=fontsize)

                        ax[0].set_title(abs_model_name + ' chi2/dof=' + str(round(chiqred, 1)))
                        for col in ax[:]:
                            col.axhline(0, ls=':', c='black')
                            col.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                            top='True')
                            col.tick_params(which='major', length=5)
                            col.tick_params(which='minor', length=3)
                            col.xaxis.set_minor_locator(AutoMinorLocator(4))
                            col.xaxis.set_major_locator(MultipleLocator(2))
                            # col.yaxis.set_minor_locator(AutoMinorLocator(5))
                            # col.yaxis.set_major_locator(MultipleLocator(0.1))
                            col.set_xlabel('Observed wavelength, $\\mu$m', fontsize=fontsize)
                            col.legend(loc='lower right', fontsize=fontsize)
                        plt.show()
                    # f_name = 'figs/'+name+'_'+mode+'_tau='+str(tau0)[:4]+'.pdf'
                    # fig.savefig(f_name, bbox_inches='tight')


# run for the list of files
    else:

        #abs_model_folder = './../data/dust_templates/Templates/profiles/Lab-Templates/'
        abs_model_folder = './../data/dust_templates/Templates/profiles/Obs-Template/'
        for (dirpath, dirname, filenames) in os.walk(abs_model_folder):
            for f in filenames:
                print(f)
                #############
                abs_model_folder = dirpath
                abs_model_name = f
                debug = False
                calc_models = True
                run_mcmc = True
                ##############
                abs_model = np.loadtxt(abs_model_folder + abs_model_name)
                abs_model[:, 1] /= np.nanmax(abs_model[:, 1])
                mask = (abs_model[:, 0] > 0) * (abs_model[:, 0] < 20)
                tmp = np.zeros((np.sum(mask), 2))
                tmp[:, 0] = abs_model[:, 0][mask]
                tmp[:, 1] = abs_model[:, 1][mask]
                abs_model = np.array(tmp)


                def continuum_model(x=sp.x, x0=5, alpha=1, y0=1):
                    f = y0 * (x / x0) ** alpha
                    return f


                def absorption_model(x=sp.x, zabs=0, An=1, template=abs_model, smoothing=False):
                    f = np.ones_like(x)
                    mask = (x >= template[0, 0] * (1 + zabs)) * (x <= template[-1, 0] * (1 + zabs))
                    if np.sum(mask) > 0:
                        f_interp = interp1d(template[:, 0], template[:, 1])
                        f[mask] = dust_abs_model(An * f_interp(x[mask] / (1 + zabs)))
                    if smoothing:
                        from scipy import signal
                        win_size = 10
                        win = signal.windows.hann(win_size)
                        filtered = signal.convolve(f, win, mode='same') / sum(win)
                        s = np.arange(len(x))
                        mask = (s > win_size / 2) * (s < len(x) - win_size / 2)
                        f[mask] = filtered[mask]

                    return f


                def conv_abs_model(x, f, mask=None):
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


                # convolution test
                if debug:
                    fig, ax = plt.subplots()
                    An = 0.3
                    ax.axhline(0, ls=':')
                    ax.plot(abs_model[:, 0], dust_abs_model(An * abs_model[:, 1]), label=abs_model_name)
                    ax.set_ylabel('flux')
                    ax.set_xlabel('Restframe wavelength')
                    ax.set_title('Absorption profile exp(-A*f(x))')
                    ax.legend()
                    plt.show()

                    fig, ax = plt.subplots()
                    f = absorption_model(x=sp.x, zabs=sp.z_abs, An=0.3)
                    m = conv_abs_model(sp.x, absorption_model(x=sp.x, zabs=sp.z_abs, An=0.3))

                    ax.plot(sp.x, absorption_model(x=sp.x, zabs=sp.z_abs, An=0.3, smoothing=False), label='model', ls='-')
                    ax.plot(sp.x, f, label='model smooth', ls='--')
                    ax.plot(sp.x, m(sp.x), label='PSF convloved')
                    ax.set_xlabel('Observed wavelength')
                    ax.set_title('Absorption profile exp(-A*f(x))')
                    ax.legend()
                    plt.show()

                # continuum test
                if 0:
                    fig, ax = plt.subplots(3, 1)
                    ax[0].plot(sp.x / (1 + sp.z_abs), sp.y, color='black')
                    cont = continuum_model(x=sp.x / (1 + sp.z_abs), x0=5 / (1 + sp.z_abs), alpha=1.25)
                    mask = (sp.x / (1 + sp.z_abs) <= 7) + (sp.x / (1 + sp.z_abs) > 7.5) * (sp.x / (1 + sp.z_abs) < 8) + (
                                sp.x / (1 + sp.z_abs) > 8.5)

                    f = absorption_model(x=sp.x, zabs=sp.z_abs, An=0.3)
                    m = conv_abs_model(sp.x, f)

                    ax[0].plot(sp.x / (1 + sp.z_abs), cont)

                    ax[1].plot(sp.x / (1 + sp.z_abs), sp.y / cont, color='black')
                    ax[1].plot(sp.x[mask] / (1 + sp.z_abs), m(sp.x[mask]), color='red')

                    ax[2].plot(sp.x / (1 + sp.z_abs), sp.y, color='black')
                    ax[2].plot(sp.x / (1 + sp.z_abs), cont)
                    ax[2].plot(sp.x / (1 + sp.z_abs), cont * m(sp.x))
                    plt.show()

                # grid for An
                if 1:
                    nmodels = 100
                    fname = abs_model_name.split('.')[0] + '.pkl'
                    gridpath = './../temp/dust/' + fname
                    if calc_models:
                        lst = glob.glob('./../temp/dust/' + '*.pkl')
                        if './../temp/dust/' + fname not in lst:
                            tau = np.linspace(0, 0.7, 15)
                            models_grid = []
                            for i, t in enumerate(tau):
                                start = time.time()
                                m = conv_abs_model(sp.x, absorption_model(x=sp.x, zabs=sp.z_abs, An=t))
                                models_grid.append(m(sp.x))
                                end = time.time()
                                print('Time/per run', i, ': ', end - start)
                            models_grid = np.array(models_grid)
                            with open(gridpath, 'wb') as f:
                                pickle.dump([sp.x, tau, models_grid], f)
                        else:
                            with open(gridpath, 'rb') as f:
                                x_grid, tau_grid, models_grid = pickle.load(f)

                    with open(gridpath, 'rb') as f:
                        x_grid, tau_grid, models_grid = pickle.load(f)


                    def calc_absorption(l=sp.x, tau=0.5, timer=0):
                        if timer:
                            start = time.time()

                        y = np.zeros_like(x_grid)
                        for i, x in enumerate(x_grid):
                            f = interp1d(tau_grid, models_grid[:, i])
                            y[i] = f(tau)

                        f = interp1d(x_grid, y)

                        if timer:
                            end = time.time()
                            print('Time/per run:', end - start)
                        return f(l)


                    # test
                    if 0:
                        print('plot')
                        plt.subplots()
                        for i in range(nmodels):
                            plt.plot(sp.x, models_grid[i, :])
                        plt.plot(sp.x, calc_absorption(sp.x, 0.4), ls='--')
                        plt.show()


                def log_probability(theta):
                    lp = log_prior(theta)
                    if not np.isfinite(lp):
                        return -np.inf
                    return lp + log_likelihood(theta)


                def log_prior(theta):
                    alpha, F0, An = theta
                    if (alpha < 0) + (alpha > 3) + (An < 0.1) + (An > 0.7) + (F0 <= 0):
                        return -np.inf
                    else:
                        return 0.0


                def log_likelihood(theta, sp=sp, timer=False, debug=False):
                    alpha, F0, An = theta
                    l = sp.x
                    flux = sp.y
                    flux_err = sp.err

                    if timer:
                        start = time.time()
                    cont = continuum_model(x=l, x0=5, alpha=alpha, y0=F0)
                    abs_model = calc_absorption(l[sp.mask], An)

                    y = cont[sp.mask] * abs_model
                    chi = -0.5 * np.nansum(np.power(flux[sp.mask] - y, 2) / np.power(flux_err[sp.mask], 2))

                    if timer:
                        end = time.time()
                        print('Time/per run:', end - start, ' chiq:', chi)
                    if debug:
                        return cont, calc_absorption(l, An), chi
                    else:
                        return chi


                if 1:
                    ''' run mcmc for model with free parameters:
                    alpha = slope power law continuum,
                    F0 = continuum normalization factor
                    An = amplitude of absorption
                    '''
                    nwalkers = 200
                    nsteps = 50
                    ndim = 3
                    # nask regions
                    mask_lines = (np.abs(sp.x - 12.72) < 0.06) + (np.abs(sp.x - 15.67) < 0.17) + (
                                np.abs(sp.x - 21.53) < 0.6)
                    sp.mask = ~mask_lines * (sp.x < 24) * (sp.y != 0)
                    # set effective errorbar
                    sp.err[:] = 0.1

                    init = [1.04, 1.1, 0.5]
                    init_range = [0.5, 0.5, 0.2]
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
                    if 1:
                        with open('./../temp/mcmc.pkl', 'rb') as f:
                            samples = pickle.load(f)

                    if 1:
                        means = np.zeros((ndim, nsteps))
                        vars = np.zeros((ndim, nsteps))
                        single = np.zeros((ndim, nsteps))
                        for i in range(nsteps):
                            for j in range(ndim):
                                means[j, i] = np.mean(samples[:, i, j])
                                vars[j, i] = np.std(samples[:, i, j])
                                single[j, i] = samples[5, i, j]

                        print('chain stats')
                        if debug:
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

                        chain = samples[:, int(nsteps * 0.9):, :].reshape((-1, ndim))

                    if 1:
                        c = ChainConsumer()
                        par_names = ["alpha", "F0", "An"]
                        c.add_chain(chain, parameters=par_names)
                        c.plotter.plot(filename="example.png", figsize="column")
                        res = c.analysis.get_summary(parameters=par_names)
                        print(res)

                        alpha = res['alpha'][1]
                        F0 = res['F0'][1]
                        An = res['An'][1]
                        theta = alpha, F0, An
                        cont, abs_m, chiq = log_likelihood(theta, debug=True)
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
                            ax[1].set_ylim(0.1, 1.5)
                            ax[0].set_ylabel('Flux', fontsize=fontsize)
                            ax[1].set_ylabel('Normalized flux', fontsize=fontsize)

                            ax[0].set_title(abs_model_name + ' chi2/dof=' + str(round(chiqred, 1)))
                            for col in ax[:]:
                                col.axhline(0, ls=':', c='black')
                                col.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                                top='True')
                                col.tick_params(which='major', length=5)
                                col.tick_params(which='minor', length=3)
                                col.xaxis.set_minor_locator(AutoMinorLocator(4))
                                col.xaxis.set_major_locator(MultipleLocator(2))
                                # col.yaxis.set_minor_locator(AutoMinorLocator(5))
                                # col.yaxis.set_major_locator(MultipleLocator(0.1))
                                col.set_xlabel('Observed wavelength, $\\mu$m', fontsize=fontsize)
                                col.legend(loc='lower right', fontsize=fontsize)

                        fig.savefig('./../output/scripts/'+abs_model_name.split('.dat')[0]+'.pdf', bbox_inches='tight')



                        #alpha,F0,An,chqred = silicate_absorption_fit(debug=False, calc_models = True,run_mcmc=True)
                        with open('./../output/scripts/fit_results.txt', 'a') as file:
                            file.write(abs_model_name+','+str(chiqred)+','+str(alpha)+','+str(F0)+','+str(An)+'\n')
