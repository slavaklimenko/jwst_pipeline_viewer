#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import numpy as np
#from pathlib import Path
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

# run for templates in the list
abs_model_folder1 = './../data/dust_templates/Templates/profiles/Lab-Templates/'
abs_model_folder2 = './../data/dust_templates/Templates/profiles/Obs-Template/'
#for abs_model_folder in [abs_model_folder1,abs_model_folder2]:
for (dirpath, dirname, filenames) in os.walk(abs_model_folder):
    for f in filenames:
        print(f)
        #############
        abs_model_folder = dirpath
        abs_model_name = f
        debug = False
        calc_models = True
        run_mcmc =True
        ##############

        abs_model = np.loadtxt(abs_model_folder + abs_model_name)
        abs_model[:, 1] /= np.nanmax(abs_model[:, 1])
        mask = (abs_model[:, 0] > 0) * (abs_model[:, 0] < 20)
        temp = np.zeros((np.sum(mask), 2))
        temp[:, 0] = abs_model[:, 0][mask]
        temp[:, 1] = abs_model[:, 1][mask]
        abs_model = np.array(temp)
        sp.abs_model = np.array(abs_model)


        def continuum_model(x=sp.x, x0=12, alpha=1, y0=1,c0 = 0):
            f = c0 + y0 * (x / x0) ** alpha
            return f


        #calculate absorption spectrum
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
                mask = (s > win_size / 2) * (s < len(x) - win_size / 2)*f>0.98
                f[mask] = filtered[mask]

            return f

        #calculate convolution
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

        # calculate and read grid for PSF convoclved absorption templates
        if 0:
            fname = abs_model_name.split('.')[0] + '.pkl'
            gridpath = './../output/scripts/' + qname.split('.')[0]+'/grids/' +fname
            if calc_models:
                lst = glob.glob('./../output/scripts/' + q_name+'/grids/' + '*.pkl')
                if gridpath not in lst:
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


        def calc_absorption(l=sp.x, tau=0.5, timer=0, method='direct'):
            if timer:
                start = time.time()

            if method == 'inteepolation':
                y = np.zeros_like(x_grid)
                for i, x in enumerate(x_grid):
                    f = interp1d(tau_grid, models_grid[:, i])
                    y[i] = f(tau)

                f = interp1d(x_grid, y,fill_value='extrapolate')
                y = f(l)
            elif method == 'direct':
                y = absorption_model(x=l, zabs=sp.z_abs, An=tau)

            if timer:
                end = time.time()
                print('Time/per run:', end - start)
            return y


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
        def log_prior_for_abs(theta):
            An = theta
            if (An < 0) + (An >2):
                return -np.inf
            else:
                return 0.0


        def log_likelihood_for_cont_and_abs(theta, sp=sp, timer=False, debug=False):
            alpha, F0, c0, An = theta
            l = sp.x
            flux = sp.y
            flux_err = sp.err

            if timer:
                start = time.time()
            cont = continuum_model(x=l, x0=12, alpha=alpha, y0=F0, c0=c0)
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

        def log_prior_for_cont_and_abs(theta):
            alpha, F0, c0, An = theta
            if (alpha < 0) + (alpha > 3) + (An < 0.1) + (An > 0.7) + (F0 <= 0):
                return -np.inf
            else:
                return 0.0


        def log_probability(theta):
            lp = log_prior_for_abs(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood_for_abs(theta)



        if 1:
            ''' run mcmc for model with free parameters:
            alpha = slope power law continuum,
            F0 = continuum normalization factor
            An = amplitude of absorption
            '''
            nwalkers = 500
            nsteps =100
            ndim = 1

            # set effective errorbar
            #sp.err[:] = 0.05*sp.y

            if ndim == 4:
                par_names = ["alpha", "F0", "c0", "An"]
                init = [1.01, 1.1, 0.0, 0.5]
                init_range = [0.1, 0.5, 0.2, 0.2]
            elif ndim == 1:
                par_names = ["An"]
                init = [0.5]
                init_range = [0.5]

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
            if 0:
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


            chain = samples[:, int(nsteps * 0.7):, :].reshape((-1, ndim))


            if 1:
                c = ChainConsumer()

                c.add_chain(chain, parameters=par_names)
                c.plotter.plot(filename="example.png", figsize="column")
                res = c.analysis.get_summary(parameters=par_names)
                print(res)
                #plt.show()

                if ndim == 4:
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

                elif ndim == 1:
                    theta = res['An'][1]
                    theta_err = res['An'][2]-res['An'][1]
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
                        plot_fit(sp=sp, abs_label=abs_model_name, sp_fit=fit_model, chiqred=chiqred, save_fig=True)


                #alpha,F0,An,chqred = silicate_absorption_fit(debug=False, calc_models = True,run_mcmc=True)
                with open('./../output/scripts/'+qname.split('.')[0]+'/'+'fit_results.txt', 'a') as file:
                    #file.write(abs_model_name+','+str(chiqred)+','+str(alpha)+','+str(F0)+','+str(c0)+','+str(An)+'\n')
                    file.write(abs_model_name + ',' + str(chiqred) + ',' + str(theta)+ ',' + str(theta_err) + '\n')
