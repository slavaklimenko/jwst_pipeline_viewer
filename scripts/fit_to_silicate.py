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

spectra = {}
folder = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/Normalized_spectra/'
for (dirpath, dirname, filenames) in os.walk(folder):
    print(dirpath, dirname, filenames)
    for k, f in enumerate(filenames):
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
            specname = f.split('.fits')[0]
            spectra[specname] = spec

trapezium_model = np.loadtxt('./../data/silicate_profiles/hannersilem.txt')
olivine_model = np.loadtxt('./../data/silicate_profiles/olivinesilem.txt')

spec_names= []
for el in spectra.keys():
    spec_names.append(el)
print(spec_names)
name = spec_names[6]
print(name)
sp = spectra[name]
wave,flux,flux_err = sp[:,0],sp[:,1],sp[:,2]+0.03
redshift =   0.524
wave /= (1+redshift)

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



if 1:
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