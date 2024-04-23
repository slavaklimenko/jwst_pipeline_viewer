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



spectra = {}
folder = '/home/slava/science/codes/python/jwst/output/specviewer/'
f = 'J1007_fringe_corrected_2sigma_aperture.fits'
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

abs_model = np.loadtxt('./../data/dust_templates/Templates/profiles/Lab-Templates/SpoonAmOliv.dat')
print('int:= ',np.trapz(abs_model[:,1],abs_model[:,0]))
plt.subplots()
plt.plot(abs_model[:,0],abs_model[:,1])
plt.ylabel('flux')
plt.xlabel('wavelength')
plt.show()

def continuum_model(x=sp.x, x0=5, alpha=1,y0=1):
    f =  y0* (x / x0) ** alpha
    return f

fig,ax = plt.subplots()
ax.plot(sp.x/(1+sp.z_abs),sp.y)
ax.plot(sp.x/(1+sp.z_abs),continuum_model(x=sp.x/(1+sp.z_abs),x0=5/(1+sp.z_abs),alpha=1.25))
plt.show()





def fit_model(w=10, tau0=0,model='trapezium'):
    if model == 'trapezium':
        m = trapezium_model
    elif model == 'olivine':
        m = olivine_model
    x,y = m[:,0],m[:,1]
    y/=np.max(y)
    fit = interp1d(x,y,fill_value='extrapolate')
    win = signal.windows.hann(400)
    m = fit(w)
    m[w<x[0]] = 0
    m[w>x[-1]] = 0
    convloved = signal.convolve(m, win, mode='same') / sum(win)
    if 0:
        fig, ax = plt.subplots()
        ax.plot(x,y)
        ax.plot(w,m)
        ax.plot(w,convloved,color='red')
        #ax.plot(win)
        plt.show()
    return np.exp(-tau0*m)


def log_probability(theta,wave=None, flux=None, mode='trapezium'):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, mode=mode)

def log_prior(theta):
    cont,tau0,v = theta
    chi = 0
    if (np.abs(cont) > 0.02)+(tau0<0)+np.abs(v)>0.01:
        return -np.inf
    else:
        return 0.0

def log_likelihood(theta,mode='olivine',wave=wave,flux=flux,flux_err=flux_err,debug=False):
    cont,tau0,v = theta
    fit = fit_model(w=wave*(1+v), tau0=tau0,model=mode)
    mask = (fit<1)*(wave>8)*(wave<12)
    y = fit[mask] - cont
    chi = 0
    chi = -0.5 * np.nansum(np.power(flux[mask] - y, 2) / np.power(flux_err[mask], 2))
    if debug:
        print('theta',theta, chi)
        plt.subplots()
        plt.axhline(1-cont)
        plt.plot(wave,flux,c='black')
        plt.plot(wave,-cont+fit,ls='--')
        plt.plot(wave[mask], y)
        if debug:
            plt.show()


    return chi



if 0:
    nwalkers = 500
    nsteps =300
    ndim = 3

    init = [0, 0.2,0.03]
    init_range = [0.002, 0.05,-0.01]
    pos2 = []
    #mode = 'trapezium' #'olivine'
    mode = 'olivine'

    for i in range(nwalkers):
        prob = -np.inf
        while prob == -np.inf:
            rndm = np.random.randn(ndim)
            wal_pos = init + init_range * rndm
            prob = log_probability(theta=wal_pos, mode=mode)
        pos2.append(wal_pos)

        pos = [init + init_range * np.random.randn(ndim) for i in range(nwalkers)]

    if 0:
        from multiprocessing import Pool

        pool = Pool(8)
        # with Pool(processes=2) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(wave, flux), pool=pool)
        if 1:
            sampler.run_mcmc(pos2, nsteps, progress=True)

            if 1:
                samples = sampler.chain[:, :, :]
                with open('data/chain'+name+'_'+mode+'.pkl', 'wb') as f:
                    pickle.dump(samples, f)

    if 1:
        with open('data/chain'+name+'_'+mode+'.pkl', 'rb') as f:
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
        par_names = ["cont", "tau0","vz"]
        c.add_chain(chain, parameters=par_names)
        c.plotter.plot(filename="example.png", figsize="column")
        res = c.analysis.get_summary(parameters=par_names)
        print(res)



        cont = res['cont'][1]
        tau0 = res['tau0'][1]
        vz = res['vz'][1]
        theta = cont,tau0,vz
        log_likelihood(theta,mode=mode,debug=True)

        fig, ax = plt.subplots()


        fit = fit_model(w=wave*(1+vz), tau0=tau0, model=mode)
        mask = fit < 1
        y = 0- cont + fit
        fontsize=12

        ax.plot(wave,y,label='9.7$\\mu$m Abs',c='red')
        ax.plot(wave , flux,zorder=-10,c='black',label='MIRI/MRS')
        ax.set_xlim(5,15)
        ax.set_ylim(0.5, 1.2)
        ax.legend(loc='lower left',fontsize=fontsize)
        ax.set_title(name + ', $\\tau_0=$'+str(tau0)[:4],fontsize=fontsize)


        for col in [ax]:
            col.axhline(0, ls=':', c='black')
            col.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                            top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=3)
            col.xaxis.set_minor_locator(AutoMinorLocator(4))
            col.xaxis.set_major_locator(MultipleLocator(2))
            col.yaxis.set_minor_locator(AutoMinorLocator(5))
            col.yaxis.set_major_locator(MultipleLocator(0.1))
            col.set_ylabel('Normalized flux', fontsize=fontsize)
            col.set_xlabel('Restframe wavelength, $\\mu$m', fontsize=fontsize)
            col.legend(loc='lower right', fontsize=fontsize)
            col.set_xlim(6, 12)
            col.set_ylim(-0.1, 2.1)

        plt.show()
        f_name = 'figs/'+name+'_'+mode+'_tau='+str(tau0)[:4]+'.pdf'
        fig.savefig(f_name, bbox_inches='tight')