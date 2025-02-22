import os, glob
import statistics

import numpy as np
from astropy.units.quantity_helper.function_helpers import histogram
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from stdatamodels.jwst import datamodels
import scipy
from scipy import signal
from scipy.fft import fft, fftfreq
from astropy import modeling
from jwst.residual_fringe.utils import fit_residual_fringes_1d
from lmfit import Minimizer, Parameters, report_fit, fit_report, conf_interval, printfuncs, Model
from scipy.signal import savgol_filter
from statsmodels.tsa.stattools import ccf
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import f_oneway
import scipy.stats as st
from collections import Counter


def generate_red_noise(N, alpha=1,freq_0=1):
    """Generates red noise (1/f^alpha) using the inverse Fourier transform method."""
    freqs = np.fft.rfftfreq(N)

    # Avoid division by zero for DC component
    freqs[0] = freq_0

    # Generate power-law spectrum
    spectrum = (1.0 / freqs ** alpha)
    spectrum /= np.max(spectrum)  # Normalize

    # Generate random phases
    phases = np.exp(2j * np.pi * np.random.rand(len(freqs)))

    # Create complex spectrum
    fft_values = np.sqrt(spectrum) * phases

    # Perform inverse FFT to get time series
    time_series = np.fft.irfft(fft_values, n=N)
    return time_series

def fft_analysis(signal,length,N):
    T = length / N
    L = N
    s = np.array(signal)
    fft_values = np.fft.fft(s)
    frequencies = np.fft.fftfreq(L, T)  # Frequency bins

    positive_freqs = frequencies[:L // 2]
    positive_fft_values = np.abs(fft_values[:L // 2])

    return positive_freqs,positive_fft_values



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
        self.norm_factor = 1

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
        return spectrum(np.array(self.x),np.array(self.y),np.array(self.err))

    def normalize(self):
        norm = np.mean(self.y[10:40])
        self.y/=norm
        if hasattr(self,'err'):
            self.err/=norm
        self.norm_factor = norm

    def renormalize(self):
        self.y *= self.norm_factor
        if hasattr(self, 'err'):
            self.err *= self.norm_factor
        if hasattr(self,'fringe_corrected'):
            self.fringe_corrected *=self.norm_factor

    def rebin(self,n=5):
        x = rebin_arr(self.x, n)
        y, err = rebin_weight_mean(self.y, self.err, n)
        self.sp_rebinned = spectrum(x,y,err)

    def nan_interpolation(self, debug=True ,label='none'):
        def isNaN(x):
            return x != x

        if np.sum(isNaN(self.y))>0:
            if debug:
                plt.subplots()
                plt.plot(self.x,self.y,label='spectrum')
            mask_nan = isNaN(self.y)
            x_range = np.where(isNaN(self.y))
            f_interp = interp1d(self.x[~mask_nan],self.y[~mask_nan])
            print(x_range)
            for i in x_range[0]:
                self.y[i] = f_interp(self.x[i])
                if debug:
                    plt.axvline(x=self.x[i], ls='--',color='tab:blue')
            if debug:
                plt.plot(self.x, self.y, label='interpolation')
                plt.title(label)
                plt.show()


class fringe_model():
    def __init__(self, num_chunks=1,a=None, fq=None, p=None,b=None,chiq=None,x=None,chunk_size=0.1, chunk_size_delta=0.05):
        self.set_data(num_chunks=num_chunks,a=a, fq=fq, p=p,b=b,chiq=chiq,x=x,chunk_size=chunk_size, chunk_size_delta=chunk_size_delta)

    def set_data(self, num_chunks=1,a=None, fq=None, p=None,b=None,chiq=None,x=None,chunk_size=0.1, chunk_size_delta=0.05):
        if num_chunks is not None:
            self.num_chunks = np.asarray(num_chunks)
        if a is not None:
            self.a = np.asarray(a)
        if fq is not None:
            self.fq = np.asarray(fq)
        if p is not None:
            self.p = np.asarray(p)
        if b is not None:
            self.b = np.asarray(b)
        if chiq is not None:
            self.chiq = np.asarray(chiq)
        if x is not None:
            self.x = np.asarray(x)
        if chunk_size is not None:
            self.chunk_size = np.asarray(chunk_size)
        if chunk_size_delta is not None:
            self.chunk_size_delta = np.asarray(chunk_size_delta)



def fringe_custom_correction_1d_old(s_pix, s_mean,debug=False,show_fit_chunks=False,
                             smooth_scale=50,chunk_size = 0.20,chunk_size_delta = 0.18,chiqlimit=7):

    s_mean.normalize()
    s_pix.normalize()


    #smooth integrated spec
    #s_mean.rebin(n=50)
    #f = interp1d(s_mean.x,s_mean.y,fill_value='extrapolate')
    #s_mean.smoothed = f(s_mean.x)
    #win = signal.windows.hann(smooth_scale)
    #s_mean.smoothed = signal.convolve(s_mean.y, win, mode='same') / sum(win)
    #s_mean.smoothed[:25]=np.mean(s_mean.y[:25])
    #s_mean.smoothed[s_mean.x.shape[0]-25:]=np.mean(s_mean.y[s_mean.x.shape[0]-25:])
    #s_mean.y = np.array(s_mean.smoothed)

    # calc polyfit correction
    mask_nan = np.isnan(s_pix.y) + np.isnan(s_mean.y)
    z = np.polyfit(s_pix.x[~mask_nan], (s_pix.y-s_mean.y)[~mask_nan], 5)
    polynomial_fit = np.poly1d(z)
    if debug and 1:
        plt.subplots()
        plt.plot(s_pix.x,s_pix.y,label='spaxel',color='black')
        plt.plot(s_mean.x,s_mean.y,label='integrated')
        plt.plot(s_pix.x,s_mean.y+polynomial_fit(s_pix.x),label='spaxel polyfit corrected')
        plt.plot(s_pix.x, polynomial_fit(s_pix.x), label='polyfit function')
        plt.title('Polynomial Fit correction')
        plt.legend()
        plt.show()
    # apply polyfit correction
    s_mean.y += polynomial_fit(s_mean.x)


    wave = s_pix.x
    flux = s_pix.y-s_mean.y
    flux_err = s_pix.err



    def fit_to_wiggles(wave,flux,flux_err=None, plot_results=0,plot_parameter_stats=0,return_params_model=0):

        # set x range binning
        delta_x = wave[-1]-wave[0]
        num_chunks = np.where(np.array([chunk_size*i - chunk_size_delta*(i-1) for i in range(5000)])<delta_x)[0][-1]+1
        params_vals = np.zeros((num_chunks,4))
        params_chiq = np.zeros(num_chunks)
        params_x = np.zeros(num_chunks)


        for i in range(num_chunks):
            l_left = (chunk_size-chunk_size_delta)*i
            l_right = chunk_size*(i+1) - chunk_size_delta*i
            mask = (wave>=wave[0]+l_left)*(wave<wave[0]+l_right)

            # calc fringe signal in chunk
            data_x = wave[mask] #- s_pix.x[mask][0]
            data_y = flux[mask] #-polynomial_fit(s_pix.x[mask])
            data_err = flux_err[mask]


            #par_names = ["a", "f", "p",'b']
            #lmfit to fringe signal
            from lmfit import Model
            if i ==0:
                init = [0.2, 20, 0, 0]

            #set frequence using fft method
            yf = fft(data_y) # - np.mean(data_y))
            N = data_x.shape[0]
            T=chunk_size/N
            xf = fftfreq(N, T)[:N // 2]
            yf_line = 2.0 / N * np.abs(yf[0:N // 2])
            yf_line = savgol_filter(yf_line, 7, 2)

            if 0:
                fig,ax = plt.subplots(1,2)
                ax[1].plot(xf,yf_line)
                #ax[1].plot(xf,2.0 / N * np.abs(fft(data_ref)[0:N // 2]))
                ax[1].fill_betweenx(x1=init[1]-10,x2=init[1]+10,y=[0,np.max(yf_line)],alpha=0.3,color='green')
                ax[0].plot(data_y)
                #ax[0].plot(data_ref)
                plt.show()

            if 0:
                mask_yf = np.abs(yf_line - np.mean(yf_line))<2*np.std(yf_line)
                yf_mean = np.mean(yf_line[mask_yf])
                yf_disp = np.std(yf_line[mask_yf])
                # set fq value if significance is above 5sigma
                yf_line[0] = 0
                if yf_line[np.argmax(yf_line)]>yf_mean+ 4.5*yf_disp:
                    fq = xf[np.argmax(yf_line)]
                else:
                    # set minimum value of fq as 5
                    fq = 5
                print('fq=',fq)
                if i>0 and fq - init[1]>5:
                    fq = init[1]
            else:
                if i>0:
                    mask_yf = np.abs(xf-fq)<=20
                else:
                    mask_yf = xf>0
                fq = xf[np.where(yf_line==np.max(yf_line[mask_yf]))[0]]

                if fq<5:
                    fq = 5
                print('fq=', fq)
                if i > 0 and fq - init[1] > 5:
                    fq = init[1]


            if 1:
                def func(x, a, p, b):
                    return a * np.sin(2 * np.pi * x * fq + p) + b

                fmodel = Model(func)
                result = fmodel.fit(data_y, x=data_x, a=init[0], p=init[2],b=init[3])
                init = [result.best_values['a'],float(fq), result.best_values['p'], result.best_values['b']]
            else:
                fitter = modeling.fitting.LevMarLSQFitter()
                model = modeling.models.Sine1D(frequency=fq)  # depending on the data you need to give some initial values
                #model.frequency.fixed = True
                mask_nan = np.isnan(data_y)
                fitted_model = fitter(model, data_x[~mask_nan], data_y[~mask_nan])
                a = fitted_model.amplitude.value
                fq = fitted_model.frequency.value
                p = fitted_model.phase.value
                b=0
                def func(x, a, p, b):
                    return a * np.sin(2 * np.pi * x * fq + p) + b
                init = [a, fq, 2*np.pi*p, b]
            print(init)





            fit= func(data_x, init[0], init[2], init[3])
            #calc chi2:
            chiq = np.sum(np.power((data_y-fit)/np.std(data_y),2))
            chiqred = chiq/data_y.shape[0]
            #print(i,init, 'chiq_red:',chiq/data_y.shape[0])

            params_chiq[i] = chiqred
            params_vals[i, :] = init
            params_x[i] = wave[0]+l_left
            if show_fit_chunks:
                fig, ax = plt.subplots(1, 2)
                #ax[0].axhline(yf_mean, ls='-')
                #ax[0].axhline(yf_mean+ 1 * yf_disp,ls='--')
                #ax[0].axhline(yf_mean+ 5* yf_disp)
                ax[0].axvline(fq,ls='--')
                ax[0].plot(xf, 2.0 / N * np.abs(yf[0:N // 2]), color='black')
                ax[0].plot(xf, yf_line, color='red',ls='--')
                ax[1].errorbar(data_x,data_y,yerr=data_err,color='black',label='data')
                ax[1].plot(data_x, func(data_x, init[0], init[2], init[3]), label='lmfit: fq='+str(round(fq,2)))
                ax[0].set_title('chunk'+str(i)+ ': '+str(wave[0]+l_left))
                ax[1].set_title('chiq: '+str(chiqred))
                ax[1].legend()
                plt.show()


        #combine chunks
        y_fit = spectrum(wave.copy(),flux.copy()*0,flux_err.copy())
        y_fit.npix = np.zeros_like(y_fit.x)
        def func(x, a, fq,p, b):
            return a * np.sin(2 * np.pi * x * fq + p) + b
        for i in range(num_chunks):
            res = params_vals[i, :]
            chiq= params_chiq[i]
            if 1: #chiq<chiqlimit:
                l_left = (chunk_size - chunk_size_delta) * i
                l_right = chunk_size * (i + 1) - chunk_size_delta * i
                mask = (wave >= wave[0] + l_left) * (wave < wave[0] + l_right)

                data_x = wave[mask]
                y = func(data_x, res[0], res[1], res[2], res[3])
                y_fit.y[mask] +=y
                y_fit.npix[mask]+=1
        y_fit.y/=y_fit.npix
        #print('zeros_pix', np.sum(y_fit.npix==0),wave[np.where(y_fit.npix==0)[0]])


        if plot_results:
            plt.subplots()
            plt.plot(wave,flux,label='data')
            plt.plot(y_fit.x,y_fit.y,label='model')
            plt.plot(y_fit.x, flux-y_fit.y-0.2, label='corrected data')
            plt.legend()
            plt.show()

        if plot_parameter_stats:
            # correct params for outlliers
            for i in range(4):
                p = params_vals[:, i]
                for j in range(p.shape[0]):
                    if j > 0 and j < p.shape[0] - 1:
                        if np.abs(p[j] - p[j - 1]) > np.abs(p[j + 1] - p[j - 1]) and np.abs(p[j + 1] - p[j]) > np.abs(
                                p[j + 1] - p[j - 1]):
                            f_interp_p = interp1d([j - 1, j + 1], [p[j - 1], p[j + 1]])
                            p[j] = f_interp_p(j)
            # show relation fit parameters with coordinate
            if 1:
                px = np.array([(chunk_size - chunk_size_delta) * i for i in range(num_chunks)]) + s_pix.x[0]
                mask_good_points = params_vals[:, 1] > 0  # 5*np.median(params_vals[:, 1])
                fig, ax = plt.subplots(1, 4, sharex=True)
                # npoints = params_vals[:,0].shape[0]
                x = np.linspace(px[0], px[-1], 100)
                p_interp = []
                for i in range(4):
                    ax[i].plot(px[mask_good_points], params_vals[:, i][mask_good_points], 'o')
                    f_interp = interp1d(px[mask_good_points], params_vals[:, i][mask_good_points],
                                        fill_value='extrapolate')
                    ax[i].plot(x, f_interp(x))
                    p_interp.append(f_interp)

                plt.show()

        if return_params_model:
            return y_fit, num_chunks,params_x,params_vals,params_chiq
        else:
            return y_fit

    (y_fit, num_chunks,params_x,params_vals,params_chiq) = fit_to_wiggles(wave=wave,flux=flux,flux_err=flux_err,plot_parameter_stats=0,return_params_model=1)

    #second iteration
    #y_fit_2 = fit_to_wiggles(wave=y_fit.x, flux=flux-y_fit.y, flux_err=y_fit.err)
    #y_fit.y += y_fit_2.y

    #third iteration
    #y_fit_3 = fit_to_wiggles(wave=y_fit.x, flux=flux-y_fit.y, flux_err=y_fit.err)
    #y_fit.y += y_fit_3.y

    if debug:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
        ax[0].plot(s_pix.x, s_pix.y, label='SPAXEL(input)')
        ax[0].plot(s_mean.x, s_mean.y, label='continuum (correct by polynom)')
        ax[0].plot(s_mean.x, s_pix.y - s_mean.y, label='fringes = spaxel-continuum')
        ax[0].plot(s_pix.x, y_fit.y, color='red', label='fit to fringes')
        ax[0].plot(s_pix.x, s_pix.y - s_mean.y - y_fit.y - 0.2, color='tab:blue', label='fit residuals')

        ax[1].plot(s_pix.x, s_pix.y, ls='-', color='black', label='SPAXEL(input)',alpha=0.3)
        ax[1].plot(s_pix.x, s_mean.y+y_fit.y, ls='-', color='red', label='continuum+fringes fit')
        ax[1].plot(s_mean.x, s_pix.y - y_fit.y - 0.2, color='green', ls='-', label='SPAXEL-fringes fit')
        ax[1].plot(s_mean.x, s_mean.y-0.2, label='continuum (correct by polynom)')
        #ax[1].plot(s_mean.x, s_pix.y - y_fit.y - 0.2 - s_mean.y, color='tab:blue', ls='-', label='spaxel-fringe_correction-s_mean')
        #ax[1].axhline(-0.2)

        ax[0].legend(fontsize=15)
        ax[1].legend(fontsize=15)
    if 1:
        fig,ax = plt.subplots()
        ax.set_title('frequency')
        ax.plot(params_vals[:,1])

        plt.show()


    s_pix.fringe_corrected = np.array(s_pix.y-y_fit.y)

    wiggle_model = fringe_model(num_chunks=num_chunks,a= params_vals[:,0], fq=params_vals[:,1], p=params_vals[:,2],b=params_vals[:,3],
                     chiq=params_chiq,x=params_x,chunk_size=chunk_size,chunk_size_delta=chunk_size_delta)

    s_pix.renormalize()
    s_corrected =s_pix.copy()
    s_corrected.y = np.array(s_pix.fringe_corrected)
    return s_corrected,wiggle_model


def fringe_custom_multipix_model(s_pix_array, s_mean,debug=False,show_fit_chunks=False,
                             chunk_size = 0.30,chunk_size_delta = 0.18):
    s_pix_array_tmp = []
    for s in s_pix_array:
        s_pix_array_tmp.append(spectrum(x=np.array(s.x),y=np.array(s.y),err=np.array(s.err)))
    del(s_pix_array)
    s_pix_array = s_pix_array_tmp
    s_mean = spectrum(x=np.array(s_mean.x),y=np.array(s_mean.y),err=np.array(s_mean.err))

    s_mean.normalize()
    # normalize spectra of pixels and save to wave,flux,err
    sp_list_normalized = []
    wave,flux,flux_err = [],[],[]
    if debug:
        fig,ax = plt.subplots(2,len(s_pix_array),figsize=(5*len(s_pix_array),5),sharex=True)
    for i,s in enumerate(s_pix_array):
        s.normalize()
        mask_nan = np.isnan(s.y) + np.isnan(s_mean.y)
        z = np.polyfit(s.x[~mask_nan], (s.y-s_mean.y)[~mask_nan], 5)
        polynomial_fit = np.poly1d(z)
        if debug:
            ax[0,i].plot(s.x,s.y,label='spaxel',color='black')
            ax[0,i].plot(s_mean.x,s_mean.y,label='Continuum')
            ax[0,i].plot(s.x,s_mean.y+polynomial_fit(s.x),label='Continuum (polyfit corrected)')
            ax[1, i].plot(s.x, s.y - (s_mean.y + polynomial_fit(s.x)), label='Fringes (input) model',color='green')
            ax[0,i].plot(s.x, polynomial_fit(s.x), label='polyfit function')
            ax[0,i].set_title('Polynomial Fit correction '+str(i))
            ax[0,i].legend()
            ax[1, i].legend()
        sp_list_normalized.append(spectrum(x=s.x,y=s.y-(s_mean.y+polynomial_fit(s_mean.x)),err=s.err))

    if debug:
        plt.show()


    def fit_to_wiggles_multi(sp_list = sp_list_normalized, plot_results=0,plot_parameter_stats=0,return_params_model=0,debug=True):
        n_spec = len(sp_list)
        if n_spec>0:
            # set x range binning
            wave = sp_list[0].x
            delta_x = wave[-1]-wave[0]
            num_chunks = np.where(np.array([chunk_size*i - chunk_size_delta*(i-1) for i in range(5000)])<delta_x)[0][-1]+1
            params_values = []
            params_chiq = []
            params_x = np.zeros(num_chunks)
            fq_values = []
            fq_values_2 = []
            wiggles_model = spectrum(x=wave,y=np.zeros_like(wave))


            for i in range(num_chunks):
                l_left = (chunk_size-chunk_size_delta)*i
                l_right = chunk_size*(i+1) - chunk_size_delta*i
                mask =  (wave>=wave[0]+l_left)*(wave<wave[0]+l_right)

                # calc fringe signal in chunk
                sp_chunk = []
                for j in range(n_spec):
                    sp_chunk.append(spectrum(x=wave[mask], y= sp_list[j].y[mask], err = sp_list[j].err[mask]))


                #estimate fq from the first chunk
                npix = sp_chunk[0].x.shape[0]


                p = []
                for j in range(n_spec):
                    c = ccf(sp_chunk[0].y ,sp_chunk[j].y )
                    p.append(find_peaks(c,height=0)[0][0])

                sp_chunk_combined = np.zeros((npix,n_spec))
                for j in range(n_spec):
                    if j==0:
                        sp_chunk_combined[:,j] = sp_chunk[j].y
                    if j>0:
                        f = np.zeros(npix)
                        f[p[j]:] = sp_chunk[j].y[:npix - p[j]]
                        f[:p[j]] = sp_chunk[j].y[:p[j]]
                        sp_chunk_combined[:,j] = f
                sp_chunk_median = spectrum(x=sp_chunk[0].x,y=np.median(sp_chunk_combined,axis=1))
                sp_chunk_median.y = savgol_filter(sp_chunk_median.y, 5, 1)
                wiggles_model.y[mask] = np.array(sp_chunk_median.y)
                del(sp_chunk_combined)

                # snow correlation between chunks
                if 0:
                    fig,ax = plt.subplots(2,4)
                    for j in range(4):

                        if j==0:
                            ax[1,0].plot(sp_chunk[j].x, sp_chunk[j].y,
                                       label=str(j))
                        if j>0:


                            ax[0,j].plot(ccf(sp_chunk[0].y, sp_chunk[j].y))
                            ax[0,j].axvline(p[j])
                            ax[1,j].plot(sp_chunk[0].x, sp_chunk[0].y)

                            ax[1,j].plot(sp_chunk[j].x+(sp_chunk[1].x[p[j]]-sp_chunk[j].x[0]), sp_chunk[j].y,label=str(j))

                            ax[1,j].plot(sp_chunk[j].x, f,label=str(j)+'edited')
                            ax[1, j].plot(sp_chunk[j].x, sp_chunk_median, label='median')

                            ax[1,0].plot(sp_chunk[j].x + (sp_chunk[1].x[p[j]] - sp_chunk[j].x[0]), sp_chunk[j].y,
                                           label=str(j))

                            ax[1,j].legend()
                    ax[0, 0].plot(sp_chunk[0].x, sp_chunk[0].y,
                                  label=str(j))
                    ax[0, 0].plot(sp_chunk[0].x, sp_chunk_median, label='median')
                    ax[1,0].legend()
                    plt.show()




                #determine fq using the fft method
                if debug and 0:
                    fig, ax = plt.subplots(1, 2,figsize=(12, 6))
                yf_line = np.zeros(10)
                for j in range(n_spec):
                    yf = fft(sp_chunk[j].y)
                    N = sp_chunk[j].x.shape[0]
                    T=chunk_size/N
                    xf = fftfreq(N, T)[:N // 2]
                    yf_i = 2.0 / N * np.abs(yf[0:N // 2])
                    if debug  and 0:
                        ax[0].plot(sp_chunk[j].y)
                        ax[1].plot(xf, yf_i)
                    if j == 0:
                        yf_line =  2.0 / N * np.abs(yf[0:N // 2])
                    else:
                        yf_line *= 2.0 / N * np.abs(yf[0:N // 2])


                #smooth fft signal to reduce a noise
                yf_line = savgol_filter(yf_line, 5, 2)
                if debug and 0:
                    ax[1].plot(xf, yf_line)

                if 1:
                    if i>0:
                        mask_yf = np.abs(xf-fq)<=20
                    else:
                        mask_yf = xf>0
                    fq = float(xf[np.where(yf_line==np.max(yf_line[mask_yf]))[0]])

                    if fq<5:
                        fq = 5
                    fq_values.append(fq)

                if debug  and 0:
                    ax[1].axvline(fq,ls='--')


                if 1:
                    red_noise = generate_red_noise(N=N,alpha=1.5)
                    red_positive_freqs, red_positive_fft_values = fft_analysis(signal=red_noise,length=chunk_size,N=N)

                    import scipy.stats as st
                    dof = 10
                    fstat = st.f.ppf(.99, dof, 1000)
                    spec99 = [fstat * m for m in red_positive_fft_values]
                    fstat = st.f.ppf(.95, dof, 1000)
                    spec95 = [fstat * m for m in red_positive_fft_values]
                    fstat = st.f.ppf(.68, dof, 1000)
                    spec68 = [fstat * m for m in red_positive_fft_values]

                if 1:
                    positive_freqs, positive_fft_values = fft_analysis(signal=sp_chunk_median.y,length=chunk_size,N=N)
                    h = np.array(positive_fft_values)

                    if debug:
                        fig,ax = plt.subplots(1,4,figsize=(18, 6))

                        #ax[0].plot(sp_chunk[0].x, sp_chunk[0].y,label='j=0')
                        ax[0].plot(sp_chunk_median.x, sp_chunk_median.y,label='median')
                        ax[0].plot(sp_chunk_median.x, red_noise,label='red noise',color='red')
                        rspec_norm = spec68/np.sum(positive_fft_values)
                        ax[1].plot(positive_freqs, positive_fft_values/np.sum(positive_fft_values)/rspec_norm ,label='fft(median)')
                        ax[1].plot(red_positive_freqs, red_positive_fft_values/np.sum(red_positive_fft_values)/rspec_norm , label='fft(red noise)',color='red')
                        ax[1].plot(red_positive_freqs, spec99 / np.sum(red_positive_fft_values)/rspec_norm , '--', label='99% confidence', color='red')
                        ax[1].plot(red_positive_freqs, spec95 / np.sum(red_positive_fft_values)/rspec_norm , ':',
                                   label='95% confidence', color='red')
                        ax[1].set_ylim(0,5)
                        #plt.show()





                    peak_pos = find_peaks(h,height=0.1)[0]
                    print('peak_pos',peak_pos,[positive_freqs[p] for p in peak_pos])
                    peak_pos = find_peaks(h,height=0.1)[0]
                    print('peak_pos',peak_pos,[positive_freqs[p] for p in peak_pos])

                    peak_height = find_peaks(h,height=0.1)[1]['peak_heights']
                    arr1inds = peak_height.argsort()
                    peak_pos = peak_pos[arr1inds[::-1]]
                    peak_height = peak_height[arr1inds[::-1]]
                    print('peak_pos',peak_pos,[positive_freqs[p] for p in peak_pos])
                    print('peak_height',peak_height)


                    fq2 = positive_freqs[peak_pos[0]]
                    if fq2<5:
                        fq2 = positive_freqs[peak_pos[1]]
                    # check for oscillation on multiple frequencies
                    if len(fq_values_2)>0 and fq_values_2[-1]%fq2 <1 and fq_values_2[-1]>fq2:
                        fq2 = fq_values_2[-1]
                    print('fq2 = ', fq2)
                    fq_values_2.append(fq2)



                    if debug:
                        ax[2].axvline(fq2, color='red', ls='--')

                        if 1:
                            # plot sinusoidal fit
                            a = np.std(sp_chunk_median.y)
                            f = a * np.sin(2 * np.pi * sp_chunk_median.x * fq2)
                            c = ccf(sp_chunk_median.y, f)
                            print(find_peaks(c, height=0)[0])
                            p = find_peaks(c, height=0)[0][0]

                            ff = a * np.sin(2 * np.pi * (sp_chunk_median.x - (sp_chunk_median.x[p] - sp_chunk_median.x[0])) *fq2)
                            ax[0].plot(sp_chunk_median.x, ff, ls='--', color='orange')


                    if debug:
                        ax[0].set_title("Original Signal (Time Domain)")
                        ax[0].set_xlabel("Time [s]")
                        ax[0].set_ylabel("Amplitude")
                        ax[0].legend()

                        # Plot the FFT result
                        ax[1].set_title("FFT of Signal (Frequency Domain)")
                        ax[1].set_xlabel("Frequency [Hz]")
                        ax[1].set_ylabel("Magnitude")

                        if i>0:
                            #ax[1].plot(positive_freqs, previuos_fft, label='fft(median) - previous',ls ='--')
                            #ax[1].plot(positive_freqs, previuos_fft*positive_fft_values, label='fft(median) - previous')
                            previuos_fft = np.array(positive_fft_values)
                        else:
                            previuos_fft = np.array(positive_fft_values)

                        ax[1].legend()

                        ax[2].plot(positive_freqs, h/np.nanmax(h),color='red',label='fft(median signal)')
                        ax[2].plot(xf, yf_line/np.nanmax(yf_line),color='blue',label='combined fft(j)')
                        ax[2].legend()

                        ax[3].plot(fq_values_2,color='red',label='median fq',marker='o')
                        ax[3].plot(fq_values,color='blue',label='combined fq')
                        ax[3].legend()
                        plt.tight_layout()
                        plt.show()



                if 0:
                    #make fit to sp_chunk_median
                    def func(x, a, p, b, fq=fq2):
                        return a * np.sin(2 * np.pi * x * fq + p) + b


                    def fcn2min(params):
                        err = 0.005
                        fj = func(x=sp_chunk[j].x, a=params['A'].value, p= params['p'].value,
                                  b=params['b'].value,fq=params['fq'].value)
                        delta =  sp_chunk_median.y - fj
                        return delta/err


                    amp = np.std(sp_chunk_median.y)
                    if 1:
                        c = ccf(sp_chunk_median.y, amp * np.sin(2 * np.pi * sp_chunk_median.x * fq2))
                        print(find_peaks(c, height=0)[0])
                        p = find_peaks(c, height=0)[0][0]

                        ff = a * np.sin(
                            2 * np.pi * (sp_chunk_median.x - (sp_chunk_median.x[p] - sp_chunk_median.x[0])) * fq2)
                        ax[0].plot(sp_chunk_median.x, ff, ls='--', color='orange')


                    params = Parameters()
                    names = ['fq']
                    names.append('A')
                    names.append( 'p')
                    names.append('b')
                    params.add('fq',value=float(fq2), min=5,max=np.inf)
                    params.add('A', value = amp, min=0, max=3*amp)
                    params.add('p', value  = np.pi/2,min=-np.pi,max=np.pi)
                    params.add('b', value  = 0.0, min=-3*amp, max=3*amp) #,min=-2*np.std(sp_chunk[j].y),max=2*np.std(sp_chunk[j].y))




                    minner = Minimizer(fcn2min, params,nan_policy='propagate', calc_covar=True) #, fcn_args=sp_chunk ) #nan_policy='propagate', calc_covar=True) #,fcn_args=sp_chunk)
                    #Minimizer(fcn2min_integ, params, fcn_args=(x_bin, y, err, cont))
                    result = minner.minimize(method='leastsq') #method='emcee') #(method='leastsq',params=params)
                    # write error report
                    #report_fit(result)
                    for par in params:
                        params[par].value = result.params[par].value



                    #params_chiq[i] = chiqred
                    params_values.append(params)
                    params_chiq.append(result.chisqr)
                    params_x[i] = wave[0]+l_left
                    if 1:
                        fig, ax = plt.subplots(1, n_spec)
                        for j in range(n_spec):
                            ax[j].plot(sp_chunk[j].x,sp_chunk[j].y)
                            ax[j].plot(sp_chunk[j].x, func(x=sp_chunk[j].x, a=params['A_' + str(j)].value, p= params['p_' + str(j)].value,
                                          b=params['b_' + str(j)].value,fq=params['fq'].value))
                        plt.show()

            if 1:
                plt.subplots()
                plt.plot(fq_values)
                plt.plot(fq_values_2)
                plt.show()

            #combine chunks
            y_fit = spectrum(wave.copy(),flux.copy()*0,flux_err.copy())
            y_fit.npix = np.zeros_like(y_fit.x)
            def func(x, a, fq,p, b):
                return a * np.sin(2 * np.pi * x * fq + p) + b
            for i in range(num_chunks):
                res = params_vals[i, :]
                chiq= params_chiq[i]
                if 1: #chiq<chiqlimit:
                    l_left = (chunk_size - chunk_size_delta) * i
                    l_right = chunk_size * (i + 1) - chunk_size_delta * i
                    mask = (wave >= wave[0] + l_left) * (wave < wave[0] + l_right)

                    data_x = wave[mask]
                    y = func(data_x, res[0], res[1], res[2], res[3])
                    y_fit.y[mask] +=y
                    y_fit.npix[mask]+=1
            y_fit.y/=y_fit.npix
            #print('zeros_pix', np.sum(y_fit.npix==0),wave[np.where(y_fit.npix==0)[0]])

            if plot_results:
                plt.subplots()
                plt.plot(wave,flux,label='data')
                plt.plot(y_fit.x,y_fit.y,label='model')
                plt.plot(y_fit.x, flux-y_fit.y-0.2, label='corrected data')
                plt.legend()
                plt.show()

            if plot_parameter_stats:
                # correct params for outlliers
                for i in range(4):
                    p = params_vals[:, i]
                    for j in range(p.shape[0]):
                        if j > 0 and j < p.shape[0] - 1:
                            if np.abs(p[j] - p[j - 1]) > np.abs(p[j + 1] - p[j - 1]) and np.abs(p[j + 1] - p[j]) > np.abs(
                                    p[j + 1] - p[j - 1]):
                                f_interp_p = interp1d([j - 1, j + 1], [p[j - 1], p[j + 1]])
                                p[j] = f_interp_p(j)
                # show relation fit parameters with coordinate
                if 1:
                    px = np.array([(chunk_size - chunk_size_delta) * i for i in range(num_chunks)]) + s_pix.x[0]
                    mask_good_points = params_vals[:, 1] > 0  # 5*np.median(params_vals[:, 1])
                    fig, ax = plt.subplots(1, 4, sharex=True)
                    # npoints = params_vals[:,0].shape[0]
                    x = np.linspace(px[0], px[-1], 100)
                    p_interp = []
                    for i in range(4):
                        ax[i].plot(px[mask_good_points], params_vals[:, i][mask_good_points], 'o')
                        f_interp = interp1d(px[mask_good_points], params_vals[:, i][mask_good_points],
                                            fill_value='extrapolate')
                        ax[i].plot(x, f_interp(x))
                        p_interp.append(f_interp)

                    plt.show()

            if return_params_model:
                return y_fit, num_chunks,params_x,params_vals,params_chiq
            else:
                return y_fit

    def fit_to_wiggles_multi_freq_list_copy(sp_list = sp_list_normalized, plot_results=0,plot_parameter_stats=0,return_params_model=0,debug=show_fit_chunks):
        n_spec = len(sp_list)
        if n_spec>0:
            # set x range binning
            wave = sp_list[0].x
            delta_x = wave[-1]-wave[0]
            #########
            chunk_size = delta_x / 80 / 0.1
            chunk_size_delta = chunk_size * 0.9
            num_chunks = np.where(np.array([chunk_size*i - chunk_size_delta*(i-1) for i in range(5000)])<delta_x)[0][-1]+1
            params_values = []
            params_chiq = []
            chunks_xleft_pos = np.zeros(num_chunks)
            fq_values = []
            fq_strengths = []
            wiggles_model = spectrum(x=wave,y=np.zeros_like(wave))


            for i in range(num_chunks):
                l_left = (chunk_size-chunk_size_delta)*i
                l_right = chunk_size*(i+1) - chunk_size_delta*i
                mask =  (wave>=wave[0]+l_left)*(wave<wave[0]+l_right)
                chunks_xleft_pos[i] = wave[0] + l_left

                # read chunks
                sp_chunk = []
                for j in range(n_spec):
                    sp_chunk.append(spectrum(x=wave[mask], y= sp_list[j].y[mask], err = sp_list[j].err[mask]))
                npix = sp_chunk[0].x.shape[0]

                #calc phase shift between chunks and the first chunk
                p = []
                for j in range(n_spec):
                    c = ccf(sp_chunk[0].y ,sp_chunk[j].y )
                    p.append(find_peaks(c,height=0)[0][0])
                #combine chunks
                sp_chunk_combined = np.zeros((npix,n_spec))
                for j in range(n_spec):
                    if j==0:
                        sp_chunk_combined[:,j] = sp_chunk[j].y
                    if j>0:
                        f = np.zeros(npix)
                        f[p[j]:] = sp_chunk[j].y[:npix - p[j]]
                        f[:p[j]] = sp_chunk[j].y[:p[j]]
                        sp_chunk_combined[:,j] = f

                #create median model for chunks
                sp_chunk_median = spectrum(x=sp_chunk[0].x,y=np.median(sp_chunk_combined,axis=1))
                sp_chunk_median.y = savgol_filter(sp_chunk_median.y, 5, 1)
                wiggles_model.y[mask] = np.array(sp_chunk_median.y)
                del(sp_chunk_combined)

                # snow the procedure of averaging
                if 0:
                    fig,ax = plt.subplots(2,n_spec)
                    for j in range(n_spec):

                        if j==0:
                            ax[1,0].plot(sp_chunk[j].x, sp_chunk[j].y,
                                       label=str(j))
                        if j>0:


                            ax[0,j].plot(ccf(sp_chunk[0].y, sp_chunk[j].y))
                            ax[0,j].axvline(p[j])
                            ax[1,j].plot(sp_chunk[0].x, sp_chunk[0].y)

                            ax[1,j].plot(sp_chunk[j].x+(sp_chunk[1].x[p[j]]-sp_chunk[j].x[0]), sp_chunk[j].y,label=str(j))

                            ax[1,j].plot(sp_chunk[j].x, f,label=str(j)+'edited')
                            ax[1, j].plot(sp_chunk[j].x, sp_chunk_median.y, label='median')

                            ax[1,0].plot(sp_chunk[j].x + (sp_chunk[1].x[p[j]] - sp_chunk[j].x[0]), sp_chunk[j].y,
                                           label=str(j))

                            ax[1,j].legend()
                    ax[0, 0].plot(sp_chunk[0].x, sp_chunk[0].y,
                                  label=str(j))
                    ax[0, 0].plot(sp_chunk[0].x, sp_chunk_median.y, label='median')
                    ax[1,0].legend()
                    plt.show()


                #generate red noise model
                if 1:
                    N = npix
                    red_noise = generate_red_noise(N=N,alpha=1.5)
                    red_positive_freqs, red_positive_fft_values = fft_analysis(signal=red_noise,length=chunk_size,N=N)
                    #find the significance level
                    dof = 5 #10? #dof = 2. * (ir + 1.)
                    fstat = st.f.ppf(.99, dof, 1000)
                    spec99 = [fstat * m for m in red_positive_fft_values]
                    fstat = st.f.ppf(.95, dof, 1000)
                    spec95 = [fstat * m for m in red_positive_fft_values]
                    fstat = st.f.ppf(.68, dof, 1000)
                    spec68 = [fstat * m for m in red_positive_fft_values]

                #calculate fft for the median chunk
                if 1:
                    positive_freqs, positive_fft_values = fft_analysis(signal=sp_chunk_median.y,length=chunk_size,N=N)
                    chunk_power_spec = np.array(positive_fft_values) / np.sum(positive_fft_values)
                    chunk_power_spec_normalized =chunk_power_spec /(spec68 / np.sum(positive_fft_values))
                    chunk_power_spec_95_limit = spec95 / np.sum(red_positive_fft_values) /(spec68 / np.sum(positive_fft_values))
                    chunk_power_spec_99_limit = spec99 / np.sum(red_positive_fft_values) / (spec68 / np.sum(positive_fft_values))
                    if debug:
                        fig,ax = plt.subplots(1,5,figsize=(18, 6))

                        #ax[0].plot(sp_chunk[0].x, sp_chunk[0].y,label='j=0')
                        ax[0].plot(sp_chunk_median.x, sp_chunk_median.y,label='median',color='black',lw=2)
                        ax[3].plot(sp_chunk_median.x, sp_chunk_median.y, label='median', color='black', lw=2)
                        ax[0].plot(sp_chunk_median.x, red_noise,label='red noise',color='red',lw=1)
                        rspec_norm = 1
                        ax[1].plot(positive_freqs, positive_fft_values / np.sum(positive_fft_values) / rspec_norm,
                                   label='fft(median)')
                        ax[1].plot(red_positive_freqs,
                                   red_positive_fft_values / np.sum(red_positive_fft_values) / rspec_norm,
                                   label='fft(red noise)', color='red')
                        ax[1].plot(red_positive_freqs, spec99 / np.sum(red_positive_fft_values) / rspec_norm, '--',
                                   label='99% confidence', color='red')
                        ax[1].plot(red_positive_freqs, spec95 / np.sum(red_positive_fft_values) / rspec_norm, ':',
                                   label='95% confidence', color='red')
                        ax[1].plot(red_positive_freqs, spec68 / np.sum(red_positive_fft_values) / rspec_norm, ':',
                                   label='68% confidence', color='red')
                        #
                        rspec_norm = spec68/np.sum(positive_fft_values)
                        ax[2].plot(positive_freqs, positive_fft_values/np.sum(positive_fft_values)/rspec_norm ,label='fft(median)')
                        ax[2].plot(red_positive_freqs, red_positive_fft_values/np.sum(red_positive_fft_values)/rspec_norm , label='fft(red noise)',color='red')
                        ax[2].plot(red_positive_freqs, spec99 / np.sum(red_positive_fft_values)/rspec_norm , '--', label='99% confidence', color='red')
                        ax[2].plot(red_positive_freqs, spec95 / np.sum(red_positive_fft_values)/rspec_norm , ':',
                                   label='95% confidence', color='red')
                        ax[2].set_ylim(0,5)


                    height = chunk_power_spec_99_limit[0]
                    peak_pos = find_peaks(chunk_power_spec_normalized,height=height)[0]
                    peak_height = find_peaks(chunk_power_spec_normalized,height=height)[1]['peak_heights']
                    arr1inds = peak_height.argsort()
                    peak_pos = peak_pos[arr1inds[::-1]]
                    peak_height = peak_height[arr1inds[::-1]]
                    if len(peak_pos) ==0:
                        print('95LIMIT, chunk number', i)
                        height = chunk_power_spec_95_limit[0]
                        peak_pos = find_peaks(chunk_power_spec_normalized, height=height)[0]
                        peak_height = find_peaks(chunk_power_spec_normalized, height=height)[1]['peak_heights']
                        arr1inds = peak_height.argsort()
                        peak_pos = peak_pos[arr1inds[::-1]]
                        peak_height = peak_height[arr1inds[::-1]]
                        #plt.show()

                    #print('peak_pos',peak_pos,[positive_freqs[p] for p in peak_pos])
                    #print('peak_height',peak_height)
                    #mask_peak = chunk_power_spec_normalized[]>chunk_power_spec_95_limit[]

                    print('fq_add_values:', [positive_freqs[p] for p in peak_pos])
                    print('fq_add_strengths:', peak_height)

                    if len(peak_pos) >0:
                        fq_values.append([positive_freqs[p] for p in peak_pos])
                        fq_strengths.append(peak_height)
                    elif len(peak_pos) == 0:
                        fq_values.append(fq_values[-1])
                        fq_strengths.append(fq_strengths[-1])
                    fq2 = fq_values[-1][np.argmax(fq_strengths[-1])]

                    print('fq_values:',fq_values[-1])
                    print('fq_strengths:',fq_strengths[-1])
                    print('main fq:', fq2)



                    if debug:
                        ax[2].axvline(fq2, color='red', ls='--')
                        for k,fk in enumerate(fq_values):
                            for el in fk:
                                ax[4].plot(k,el, color='red', marker='o')

                        if 1:
                            # plot sinusoidal fit
                            a = np.std(sp_chunk_median.y)
                            f = a * np.sin(2 * np.pi * sp_chunk_median.x * fq2)
                            c = ccf(sp_chunk_median.y, f)
                            print(find_peaks(c, height=0)[0])
                            p = find_peaks(c, height=0)[0][0]

                            ff = a * np.sin(2 * np.pi * (sp_chunk_median.x - (sp_chunk_median.x[p] - sp_chunk_median.x[0])) *fq2)
                            ax[3].plot(sp_chunk_median.x, ff, ls='-', color='tab:blue')


                    if debug:
                        ax[0].set_title("Signal "+str(i))
                        ax[0].set_xlabel("Wavelength [micron]")
                        ax[0].set_ylabel("Amplitude")
                        ax[0].legend()

                        # Plot the FFT result
                        ax[1].set_title("FFT")
                        ax[1].set_xlabel("Frequency[1/micron]")
                        ax[1].set_ylabel("Magnitude")

                        ax[2].set_title("FFT normalized")
                        ax[2].set_xlabel("Frequency[1/micron]")
                        ax[2].set_ylabel("Magnitude normalized")

                        ax[3].set_title("Oscillation example")
                        ax[3].set_ylabel("Amplitude")
                        ax[3].set_ylabel("Wavelenght")

                        ax[4].set_title("Freq. list")
                        ax[4].set_ylabel("Frequency[1/micron]")
                        ax[4].set_ylabel("Chunk number #")

                        ax[0].legend()
                        ax[1].legend()
                        ax[2].legend()

                        plt.tight_layout()
                        plt.show()

            #create a model of the main frequency
            if 1:
                main_frequency_list = np.zeros(num_chunks)
                for k, fk in enumerate(fq_values):
                    main_frequency_list[k] = fq_values[k][np.argmax(fq_strengths[k])]
                # correct for 1 pix jumps
                for k in range(num_chunks):
                    if k==0:
                        if main_frequency_list[k+1]== main_frequency_list[k+2] and main_frequency_list[k] != main_frequency_list[k+1]:
                            main_frequency_list[k] = main_frequency_list[k+1]
                    elif k<num_chunks-1:
                        if main_frequency_list[k+1]== main_frequency_list[k-1] and main_frequency_list[k] != main_frequency_list[k-1]:
                            main_frequency_list[k] = main_frequency_list[k - 1]
                # correct for positive jumps
                for k in range(num_chunks):
                    if k>0 and main_frequency_list[k]> main_frequency_list[k-1]:
                        main_frequency_list[k] =  main_frequency_list[k-1]


            if 1:
                fig,ax = plt.subplots(1,2,sharex=True,figsize=(9,3))
                fig.subplots_adjust(wspace=0.3)
                for k, fk in enumerate(fq_values):
                    for el in fk:
                        ax[0].plot(chunks_xleft_pos[k], el, color='red', marker='o')
                        ax[0].plot(chunks_xleft_pos[k],fq_values[k][np.argmax(fq_strengths[k])], color='blue', marker='o')
                ax[0].plot(chunks_xleft_pos,main_frequency_list,label='Model',color='black',lw=2)
                ax[0].set_ylabel("Frequency[1/micron]")
                ax[0].set_xlabel("Wavelength [micron]")
                ax[1].plot(wiggles_model.x,wiggles_model.y,color='black',lw=1)
                ax[1].set_ylabel("Aver.Wiggles amplitude")
                ax[1].set_xlabel("Wavelength [micron]")

                plt.show()


            if 0:
                    #make fit to sp_chunk_median
                    def func(x, a, p, b, fq=fq2):
                        return a * np.sin(2 * np.pi * x * fq + p) + b


                    def fcn2min(params):
                        err = 0.005
                        fj = func(x=sp_chunk[j].x, a=params['A'].value, p= params['p'].value,
                                  b=params['b'].value,fq=params['fq'].value)
                        delta =  sp_chunk_median.y - fj
                        return delta/err


                    amp = np.std(sp_chunk_median.y)
                    if 1:
                        c = ccf(sp_chunk_median.y, amp * np.sin(2 * np.pi * sp_chunk_median.x * fq2))
                        print(find_peaks(c, height=0)[0])
                        p = find_peaks(c, height=0)[0][0]

                        ff = a * np.sin(
                            2 * np.pi * (sp_chunk_median.x - (sp_chunk_median.x[p] - sp_chunk_median.x[0])) * fq2)
                        ax[0].plot(sp_chunk_median.x, ff, ls='--', color='orange')


                    params = Parameters()
                    names = ['fq']
                    names.append('A')
                    names.append( 'p')
                    names.append('b')
                    params.add('fq',value=float(fq2), min=5,max=np.inf)
                    params.add('A', value = amp, min=0, max=3*amp)
                    params.add('p', value  = np.pi/2,min=-np.pi,max=np.pi)
                    params.add('b', value  = 0.0, min=-3*amp, max=3*amp) #,min=-2*np.std(sp_chunk[j].y),max=2*np.std(sp_chunk[j].y))




                    minner = Minimizer(fcn2min, params,nan_policy='propagate', calc_covar=True) #, fcn_args=sp_chunk ) #nan_policy='propagate', calc_covar=True) #,fcn_args=sp_chunk)
                    #Minimizer(fcn2min_integ, params, fcn_args=(x_bin, y, err, cont))
                    result = minner.minimize(method='leastsq') #method='emcee') #(method='leastsq',params=params)
                    # write error report
                    #report_fit(result)
                    for par in params:
                        params[par].value = result.params[par].value



                    #params_chiq[i] = chiqred
                    params_values.append(params)
                    params_chiq.append(result.chisqr)
                    params_x[i] = wave[0]+l_left
                    if 1:
                        fig, ax = plt.subplots(1, n_spec)
                        for j in range(n_spec):
                            ax[j].plot(sp_chunk[j].x,sp_chunk[j].y)
                            ax[j].plot(sp_chunk[j].x, func(x=sp_chunk[j].x, a=params['A_' + str(j)].value, p= params['p_' + str(j)].value,
                                          b=params['b_' + str(j)].value,fq=params['fq'].value))
                        plt.show()


            if 0:
                #combine chunks
                y_fit = spectrum(wave.copy(),flux.copy()*0,flux_err.copy())
                y_fit.npix = np.zeros_like(y_fit.x)
                def func(x, a, fq,p, b):
                    return a * np.sin(2 * np.pi * x * fq + p) + b
                for i in range(num_chunks):
                    res = params_vals[i, :]
                    chiq= params_chiq[i]
                    if 1: #chiq<chiqlimit:
                        l_left = (chunk_size - chunk_size_delta) * i
                        l_right = chunk_size * (i + 1) - chunk_size_delta * i
                        mask = (wave >= wave[0] + l_left) * (wave < wave[0] + l_right)

                        data_x = wave[mask]
                        y = func(data_x, res[0], res[1], res[2], res[3])
                        y_fit.y[mask] +=y
                        y_fit.npix[mask]+=1
                y_fit.y/=y_fit.npix
                #print('zeros_pix', np.sum(y_fit.npix==0),wave[np.where(y_fit.npix==0)[0]])

                if plot_results:
                    plt.subplots()
                    plt.plot(wave,flux,label='data')
                    plt.plot(y_fit.x,y_fit.y,label='model')
                    plt.plot(y_fit.x, flux-y_fit.y-0.2, label='corrected data')
                    plt.legend()
                    plt.show()

                if plot_parameter_stats:
                    # correct params for outlliers
                    for i in range(4):
                        p = params_vals[:, i]
                        for j in range(p.shape[0]):
                            if j > 0 and j < p.shape[0] - 1:
                                if np.abs(p[j] - p[j - 1]) > np.abs(p[j + 1] - p[j - 1]) and np.abs(p[j + 1] - p[j]) > np.abs(
                                        p[j + 1] - p[j - 1]):
                                    f_interp_p = interp1d([j - 1, j + 1], [p[j - 1], p[j + 1]])
                                    p[j] = f_interp_p(j)
                    # show relation fit parameters with coordinate
                    if 1:
                        px = np.array([(chunk_size - chunk_size_delta) * i for i in range(num_chunks)]) + s_pix.x[0]
                        mask_good_points = params_vals[:, 1] > 0  # 5*np.median(params_vals[:, 1])
                        fig, ax = plt.subplots(1, 4, sharex=True)
                        # npoints = params_vals[:,0].shape[0]
                        x = np.linspace(px[0], px[-1], 100)
                        p_interp = []
                        for i in range(4):
                            ax[i].plot(px[mask_good_points], params_vals[:, i][mask_good_points], 'o')
                            f_interp = interp1d(px[mask_good_points], params_vals[:, i][mask_good_points],
                                                fill_value='extrapolate')
                            ax[i].plot(x, f_interp(x))
                            p_interp.append(f_interp)

                        plt.show()

                if return_params_model:
                    return y_fit, num_chunks,params_x,params_vals,params_chiq
                else:
                    return y_fit

    def fit_to_wiggles_multi_freq_list(sp_list = sp_list_normalized, debug=show_fit_chunks,  num_chunks = 50):
        n_spec = len(sp_list)
        if n_spec>0:
            # set x range binning
            wave = sp_list[0].x
            delta_x = wave[-1]-wave[0]
            #########
            if 1:
                chunks_central_pos = np.array([wave[0] + delta_x / num_chunks * i for i in range(num_chunks)])
                chunks_left_pos = np.zeros(num_chunks)
                chunks_right_pos = np.zeros(num_chunks)
                chunks_size = np.zeros(num_chunks)

            params_values = []
            params_chiq = []
            fq_values = []
            fq_strengths = []
            wiggles_model = spectrum(x=wave,y=np.zeros_like(wave))
            #init frequency of wiggles
            if wave[0]<7.5:
                fq_i = 70 #40
                chunks_size_up_lim = 0.2
            elif wave[0]<11:
                fq_i = 40 #40
                chunks_size_up_lim = 0.4
            elif wave[0]<17:
                fq_i = 20  # 40
                chunks_size_up_lim = 0.7
            else:
                fq_i = 4  # 40
                chunks_size_up_lim = 1
            for i in range(num_chunks):
                #set size
                chunks_size[i] = float(4 / fq_i)
                if chunks_size[i]<0.15:
                    chunks_size[i] = 0.15
                elif chunks_size[i]>chunks_size_up_lim:
                    chunks_size[i] = chunks_size_up_lim
                #set position: l_left,l_right
                l_central = chunks_central_pos[i]
                if l_central-chunks_size[i]/2<wave[0]:
                        l_left = wave[0]
                else :
                     l_left = l_central-chunks_size[i]/2
                l_right = l_left+chunks_size[i]
                if l_right>wave[-1]:
                    l_right = wave[-1]
                    l_left = wave[-1] - chunks_size[i]
                chunks_left_pos[i] = l_left
                chunks_right_pos[i] = l_right
                # set mask for a chunk position
                mask_chunk_pos =  (wave>=l_left)*(wave<l_right)
                npix = np.sum(mask_chunk_pos)  # number of pixels in the chunk
                chunk_size = chunks_size[i]
                print('chunk#', i, ' pos:',l_left,':',l_right, ' size:',chunk_size)

                # calculated median chunk model "sp_chunk_median"
                # read chunks from spaxels list
                sp_chunk = []
                for j in range(n_spec):
                    sp_chunk.append(spectrum(x=wave[mask_chunk_pos], y=sp_list[j].y[mask_chunk_pos],
                                             err=sp_list[j].err[mask_chunk_pos]))
                if 1:
                    #calc phase shift between chunks and the first chunk
                    p = []
                    for j in range(n_spec):
                        c = ccf(sp_chunk[0].y ,sp_chunk[j].y )
                        p.append(find_peaks(c,height=0)[0][0])
                    #combine chunks
                    sp_chunk_combined = np.zeros((npix,n_spec))
                    for j in range(n_spec):
                        if j==0:
                            sp_chunk_combined[:,j] = sp_chunk[j].y
                        if j>0:
                            f = np.zeros(npix)
                            f[p[j]:] = sp_chunk[j].y[:npix - p[j]]
                            f[:p[j]] = sp_chunk[j].y[:p[j]]
                            sp_chunk_combined[:,j] = f

                    #create median model for chunks
                    sp_chunk_median = spectrum(x=sp_chunk[0].x,y=np.median(sp_chunk_combined,axis=1))
                    sp_chunk_median.y = savgol_filter(sp_chunk_median.y, 5, 1)
                    wiggles_model.y[mask_chunk_pos] = np.array(sp_chunk_median.y)
                    del(sp_chunk_combined)

                    # snow the procedure of averaging
                    if 0:
                        fig,ax = plt.subplots(2,n_spec)
                        for j in range(n_spec):

                            if j==0:
                                ax[1,0].plot(sp_chunk[j].x, sp_chunk[j].y,
                                           label=str(j))
                            if j>0:


                                ax[0,j].plot(ccf(sp_chunk[0].y, sp_chunk[j].y))
                                ax[0,j].axvline(p[j])
                                ax[1,j].plot(sp_chunk[0].x, sp_chunk[0].y)

                                ax[1,j].plot(sp_chunk[j].x+(sp_chunk[1].x[p[j]]-sp_chunk[j].x[0]), sp_chunk[j].y,label=str(j))

                                ax[1,j].plot(sp_chunk[j].x, f,label=str(j)+'edited')
                                ax[1, j].plot(sp_chunk[j].x, sp_chunk_median.y, label='median')

                                ax[1,0].plot(sp_chunk[j].x + (sp_chunk[1].x[p[j]] - sp_chunk[j].x[0]), sp_chunk[j].y,
                                               label=str(j))

                                ax[1,j].legend()
                        ax[0, 0].plot(sp_chunk[0].x, sp_chunk[0].y,
                                      label=str(j))
                        ax[0, 0].plot(sp_chunk[0].x, sp_chunk_median.y, label='median')
                        ax[1,0].legend()
                        plt.show()


                #generate a red noise model
                if 1:
                    N = npix
                    red_noise = generate_red_noise(N=N,alpha=1.5)
                    red_positive_freqs, red_positive_fft_values = fft_analysis(signal=red_noise,length=chunk_size,N=N)
                    #find the significance level
                    dof = 5 #10? #dof = 2. * (ir + 1.)
                    fstat = st.f.ppf(.99, dof, 1000)
                    spec99 = [fstat * m for m in red_positive_fft_values]
                    fstat = st.f.ppf(.95, dof, 1000)
                    spec95 = [fstat * m for m in red_positive_fft_values]
                    fstat = st.f.ppf(.68, dof, 1000)
                    spec68 = [fstat * m for m in red_positive_fft_values]

                #calculate fft for the median chunk
                if 1:
                    positive_freqs, positive_fft_values = fft_analysis(signal=sp_chunk_median.y,length=chunk_size,N=N)

                    chunk_power_spec = np.array(positive_fft_values) / np.nansum(positive_fft_values)
                    #
                    power_spec_normalization = (spec68 / np.nansum(positive_fft_values))
                    chunk_power_spec_normalized =chunk_power_spec /power_spec_normalization
                    chunk_power_spec_95_limit = spec95 / np.sum(red_positive_fft_values) /power_spec_normalization
                    chunk_power_spec_99_limit = spec99 / np.sum(red_positive_fft_values) / power_spec_normalization

                    if debug:
                        fig,ax = plt.subplots(1,5,figsize=(18, 6))
                        ax[0].plot(sp_chunk_median.x, sp_chunk_median.y,label='median',color='black',lw=2)
                        ax[3].plot(sp_chunk_median.x, sp_chunk_median.y, label='median', color='black', lw=2)
                        ax[0].plot(sp_chunk_median.x, red_noise,label='red noise',color='red',lw=1)
                        rspec_norm = 1
                        ax[1].plot(positive_freqs, chunk_power_spec,  label='fft(median)')
                        ax[1].plot(red_positive_freqs,red_positive_fft_values / np.sum(red_positive_fft_values) / rspec_norm,
                                   label='fft(red noise)', color='red')
                        ax[1].plot(red_positive_freqs, spec99 / np.sum(red_positive_fft_values) / rspec_norm, '--',
                                   label='99% confidence', color='red')
                        ax[1].plot(red_positive_freqs, spec95 / np.sum(red_positive_fft_values) / rspec_norm, ':',
                                   label='95% confidence', color='red')
                        ax[1].plot(red_positive_freqs, spec68 / np.sum(red_positive_fft_values) / rspec_norm, ':',
                                   label='68% confidence', color='red')
                        #

                        ax[2].plot(positive_freqs, chunk_power_spec_normalized ,label='fft(median)')
                        ax[2].plot(red_positive_freqs, red_positive_fft_values/np.sum(red_positive_fft_values)/rspec_norm , label='fft(red noise)',color='red')
                        ax[2].plot(red_positive_freqs, chunk_power_spec_99_limit , '--', label='99% confidence', color='red')
                        ax[2].plot(red_positive_freqs, chunk_power_spec_95_limit , ':',
                                   label='95% confidence', color='red')
                        ax[2].set_ylim(0,5)



                    height = spec99 / np.sum(red_positive_fft_values)
                    peak_pos = find_peaks(chunk_power_spec,height=height)[0]
                    peak_height = np.array([chunk_power_spec_normalized[el] for el in [peak_pos]])[0]#find_peaks(chunk_power_spec, height=height)[1]['peak_heights']
                    arr1inds = peak_height.argsort()
                    peak_pos = peak_pos[arr1inds[::-1]]
                    peak_height = peak_height[arr1inds[::-1]]

                    if len(peak_pos) ==0:
                        print('search for peaks within 95% confidence')
                        height = spec95 / np.sum(red_positive_fft_values)
                        peak_pos = find_peaks(chunk_power_spec, height=height)[0]
                        print('peak_pos', peak_pos)
                        peak_height = np.array([chunk_power_spec_normalized[el] for el in [peak_pos]])[
                            0]  # find_peaks(chunk_power_spec, height=height)[1]['peak_heights']
                        print('peak_height', peak_height)
                        arr1inds = peak_height.argsort()
                        peak_pos = peak_pos[arr1inds[::-1]]
                        peak_height = peak_height[arr1inds[::-1]]

                    #print('peak_pos',peak_pos,[positive_freqs[p] for p in peak_pos])
                    #print('peak_height',peak_height)
                    #mask_peak = chunk_power_spec_normalized[]>chunk_power_spec_95_limit[]

                    print('peak position (fq_values):', [positive_freqs[p] for p in peak_pos])
                    print('peak height (fq_strengths):', peak_height)
                    if len(peak_pos) >0:
                        fq_tmp = [positive_freqs[p] for p in peak_pos][np.argmax(peak_height)]
                        if np.abs(fq_tmp-fq_i)<2*fq_i:
                            fq_values.append([positive_freqs[p] for p in peak_pos])
                            fq_strengths.append(peak_height)
                        else:
                            fq_values.append(fq_values[-1])
                            fq_strengths.append(fq_strengths[-1])
                    elif len(peak_pos) == 0:
                        #fq_values.append(fq_values[-1])
                        #fq_strengths.append(fq_strengths[-1])
                        fq_values.append([-1])
                        fq_strengths.append([-1])
                    if fq_values[-1][np.argmax(fq_strengths[-1])]>0:
                        fq_i = fq_values[-1][np.argmax(fq_strengths[-1])]

                    print('main fq_values:',fq_values[-1])
                    print('main fq_strengths:',fq_strengths[-1])
                    print('main fq:', fq_i)



                    if debug:
                        ax[1].axvline(fq_i, color='red', ls='--')
                        ax[2].axvline(fq_i, color='red', ls='--')
                        for k,fk in enumerate(fq_values):
                            for el in fk:
                                ax[4].plot(chunks_central_pos[k],el, color='red', marker='o')
                            ax[4].plot(chunks_central_pos[k], fq_values[k][np.argmax(fq_strengths[k])], color='blue',
                                       marker='o')

                        if 1:
                            # plot sinusoidal fit
                            a = np.std(sp_chunk_median.y)
                            f = a * np.sin(2 * np.pi * sp_chunk_median.x * fq_i)
                            c = ccf(sp_chunk_median.y, f)
                            print(find_peaks(c, height=0)[0])
                            p = find_peaks(c, height=0)[0][0]

                            ff = a * np.sin(2 * np.pi * (sp_chunk_median.x - (sp_chunk_median.x[p] - sp_chunk_median.x[0])) *fq_i)
                            ax[3].plot(sp_chunk_median.x, ff, ls='-', color='tab:blue')


                    if debug:
                        ax[0].set_title("Signal "+str(i))
                        ax[0].set_xlabel("Wavelength [micron]")
                        ax[0].set_ylabel("Amplitude")
                        ax[0].legend()

                        # Plot the FFT result
                        ax[1].set_title("FFT")
                        ax[1].set_xlabel("Frequency[1/micron]")
                        ax[1].set_ylabel("Magnitude")

                        ax[2].set_title("FFT normalized")
                        ax[2].set_xlabel("Frequency[1/micron]")
                        ax[2].set_ylabel("Magnitude normalized")

                        ax[3].set_title("Oscillation example")
                        ax[3].set_ylabel("Amplitude")
                        ax[3].set_ylabel("Wavelenght")

                        ax[4].set_title("Freq. list")
                        ax[4].set_ylabel("Frequency[1/micron]")
                        ax[4].set_ylabel("Chunk number #")

                        ax[0].legend()
                        ax[1].legend()
                        ax[2].legend()

                        plt.tight_layout()
                        plt.show()

            #create a model of the main frequency
            if 1:
                main_frequency_list = np.zeros(num_chunks)
                for k, fk in enumerate(fq_values):
                    main_frequency_list[k] = fq_values[k][np.argmax(fq_strengths[k])]
                # correct for cases of non-detections
                #for k in range(num_chunks):
                #    if k > 0:
                #        if main_frequency_list[k] < 0:
                #            main_frequency_list[k] = main_frequency_list[k - 1]
                # correct for 1 pix jumps
                for k in range(num_chunks):
                    if k==0:
                        if main_frequency_list[k+1]== main_frequency_list[k+2] and main_frequency_list[k] != main_frequency_list[k+1]:
                            main_frequency_list[k] = main_frequency_list[k+1]
                    elif k<num_chunks-1:
                        if main_frequency_list[k+1]== main_frequency_list[k-1] and main_frequency_list[k] != main_frequency_list[k-1]:
                            main_frequency_list[k] = main_frequency_list[k - 1]
                # leave most frequent numbers
                numbers = [l for l in main_frequency_list]
                counter = Counter(numbers)
                # Select numbers with frequency more than 5
                frequent_numbers = [num for num, count in counter.items() if count >= 3]
                for k in range(num_chunks):
                    if k < num_chunks - 1:
                        if main_frequency_list[k + 1] != main_frequency_list[k] and main_frequency_list[k + 1] not in frequent_numbers:
                            main_frequency_list[k + 1] = main_frequency_list[k]
                    else:
                        if main_frequency_list[k] != main_frequency_list[k-1] and main_frequency_list[k] not in frequent_numbers:
                            main_frequency_list[k] = main_frequency_list[k-1]
                # allow jump if next three numbers are equal
                for k in range(num_chunks):
                    if k < num_chunks - 3 and main_frequency_list[k + 1] != main_frequency_list[k] and 1:
                        if len(set(main_frequency_list[k+1:k+3])) > 1:
                            main_frequency_list[k + 1] = main_frequency_list[k]
                #cacl fit with parabola (fringes_fr)
                data_x,data_y = chunks_central_pos[main_frequency_list>0],main_frequency_list[main_frequency_list>0]
                #
                for k, fk in enumerate(fq_values):
                    data_x,data_y = np.append(data_x,chunks_central_pos[k]),  np.append(data_y,fq_values[k][np.argmax(fq_strengths[k])])
                #
                coefficients = np.polyfit(data_x,data_y, 2)
                a, b, c = coefficients
                x_fit = np.linspace(wave[0], wave[-1], 100)
                y_fit = a * x_fit ** 2 + b * x_fit + c
                fringes_fr = interp1d(x_fit, y_fit)
                #
            if 1:
                fig,ax = plt.subplots(1,2,sharex=True,figsize=(9,3))
                fig.subplots_adjust(wspace=0.3)
                for k, fk in enumerate(fq_values):
                    for el in fk:
                        if el>0:
                            ax[0].plot(chunks_central_pos[k], el, color='red', marker='o')
                            ax[0].plot(chunks_central_pos[k],fq_values[k][np.argmax(fq_strengths[k])], color='blue', marker='o')
                        else:
                            ax[0].plot(chunks_central_pos[k], el, color='black', marker='x')
                for l in frequent_numbers:
                    ax[0].axhline(l,ls=':')
                ax[0].plot(chunks_central_pos,main_frequency_list,label='Model',color='black',lw=2)
                ax[0].plot(wave,fringes_fr(wave))
                ax[0].set_ylabel("Frequency[1/micron]")
                ax[0].set_xlabel("Wavelength [micron]")
                ax[1].plot(wiggles_model.x,wiggles_model.y,color='black',lw=1)
                ax[1].set_ylabel("Aver.Wiggles amplitude")
                ax[1].set_xlabel("Wavelength [micron]")



                plt.show()

            if 1:
                return fringes_fr


            if 0:
                    #make fit to sp_chunk_median
                    def func(x, a, p, b, fq=fq_i):
                        return a * np.sin(2 * np.pi * x * fq + p) + b


                    def fcn2min(params):
                        err = 0.005
                        fj = func(x=sp_chunk[j].x, a=params['A'].value, p= params['p'].value,
                                  b=params['b'].value,fq=params['fq'].value)
                        delta =  sp_chunk_median.y - fj
                        return delta/err


                    amp = np.std(sp_chunk_median.y)
                    if 1:
                        c = ccf(sp_chunk_median.y, amp * np.sin(2 * np.pi * sp_chunk_median.x * fq_i))
                        print(find_peaks(c, height=0)[0])
                        p = find_peaks(c, height=0)[0][0]

                        ff = a * np.sin(
                            2 * np.pi * (sp_chunk_median.x - (sp_chunk_median.x[p] - sp_chunk_median.x[0])) * fq_i)
                        ax[0].plot(sp_chunk_median.x, ff, ls='--', color='orange')


                    params = Parameters()
                    names = ['fq']
                    names.append('A')
                    names.append( 'p')
                    names.append('b')
                    params.add('fq',value=float(fq_i), min=5,max=np.inf)
                    params.add('A', value = amp, min=0, max=3*amp)
                    params.add('p', value  = np.pi/2,min=-np.pi,max=np.pi)
                    params.add('b', value  = 0.0, min=-3*amp, max=3*amp) #,min=-2*np.std(sp_chunk[j].y),max=2*np.std(sp_chunk[j].y))




                    minner = Minimizer(fcn2min, params,nan_policy='propagate', calc_covar=True) #, fcn_args=sp_chunk ) #nan_policy='propagate', calc_covar=True) #,fcn_args=sp_chunk)
                    #Minimizer(fcn2min_integ, params, fcn_args=(x_bin, y, err, cont))
                    result = minner.minimize(method='leastsq') #method='emcee') #(method='leastsq',params=params)
                    # write error report
                    #report_fit(result)
                    for par in params:
                        params[par].value = result.params[par].value



                    #params_chiq[i] = chiqred
                    params_values.append(params)
                    params_chiq.append(result.chisqr)
                    params_x[i] = wave[0]+l_left
                    if 1:
                        fig, ax = plt.subplots(1, n_spec)
                        for j in range(n_spec):
                            ax[j].plot(sp_chunk[j].x,sp_chunk[j].y)
                            ax[j].plot(sp_chunk[j].x, func(x=sp_chunk[j].x, a=params['A_' + str(j)].value, p= params['p_' + str(j)].value,
                                          b=params['b_' + str(j)].value,fq=params['fq'].value))
                        plt.show()


            if 0:
                #combine chunks
                y_fit = spectrum(wave.copy(),flux.copy()*0,flux_err.copy())
                y_fit.npix = np.zeros_like(y_fit.x)
                def func(x, a, fq,p, b):
                    return a * np.sin(2 * np.pi * x * fq + p) + b
                for i in range(num_chunks):
                    res = params_vals[i, :]
                    chiq= params_chiq[i]
                    if 1: #chiq<chiqlimit:
                        l_left = (chunk_size - chunk_size_delta) * i
                        l_right = chunk_size * (i + 1) - chunk_size_delta * i
                        mask = (wave >= wave[0] + l_left) * (wave < wave[0] + l_right)

                        data_x = wave[mask]
                        y = func(data_x, res[0], res[1], res[2], res[3])
                        y_fit.y[mask] +=y
                        y_fit.npix[mask]+=1
                y_fit.y/=y_fit.npix
                #print('zeros_pix', np.sum(y_fit.npix==0),wave[np.where(y_fit.npix==0)[0]])

                if plot_results:
                    plt.subplots()
                    plt.plot(wave,flux,label='data')
                    plt.plot(y_fit.x,y_fit.y,label='model')
                    plt.plot(y_fit.x, flux-y_fit.y-0.2, label='corrected data')
                    plt.legend()
                    plt.show()

                if plot_parameter_stats:
                    # correct params for outlliers
                    for i in range(4):
                        p = params_vals[:, i]
                        for j in range(p.shape[0]):
                            if j > 0 and j < p.shape[0] - 1:
                                if np.abs(p[j] - p[j - 1]) > np.abs(p[j + 1] - p[j - 1]) and np.abs(p[j + 1] - p[j]) > np.abs(
                                        p[j + 1] - p[j - 1]):
                                    f_interp_p = interp1d([j - 1, j + 1], [p[j - 1], p[j + 1]])
                                    p[j] = f_interp_p(j)
                    # show relation fit parameters with coordinate
                    if 1:
                        px = np.array([(chunk_size - chunk_size_delta) * i for i in range(num_chunks)]) + s_pix.x[0]
                        mask_good_points = params_vals[:, 1] > 0  # 5*np.median(params_vals[:, 1])
                        fig, ax = plt.subplots(1, 4, sharex=True)
                        # npoints = params_vals[:,0].shape[0]
                        x = np.linspace(px[0], px[-1], 100)
                        p_interp = []
                        for i in range(4):
                            ax[i].plot(px[mask_good_points], params_vals[:, i][mask_good_points], 'o')
                            f_interp = interp1d(px[mask_good_points], params_vals[:, i][mask_good_points],
                                                fill_value='extrapolate')
                            ax[i].plot(x, f_interp(x))
                            p_interp.append(f_interp)

                        plt.show()

                if return_params_model:
                    return y_fit, num_chunks,params_x,params_vals,params_chiq
                else:
                    return y_fit


    #(y_fit, num_chunks,params_x,params_vals,params_chiq) = fit_to_wiggles_multi_freq_list(sp_list = sp_normalized,plot_parameter_stats=0,return_params_model=1)
    fringe_fq_model = fit_to_wiggles_multi_freq_list(sp_list=sp_list_normalized)

    return fringe_fq_model

    if debug and 0:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
        ax[0].plot(s_pix.x, s_pix.y, label='SPAXEL(input)')
        ax[0].plot(s_mean.x, s_mean.y, label='continuum (correct by polynom)')
        ax[0].plot(s_mean.x, s_pix.y - s_mean.y, label='fringes = spaxel-continuum')
        ax[0].plot(s_pix.x, y_fit.y, color='red', label='fit to fringes')
        ax[0].plot(s_pix.x, s_pix.y - s_mean.y - y_fit.y - 0.2, color='tab:blue', label='fit residuals')

        ax[1].plot(s_pix.x, s_pix.y, ls='-', color='black', label='SPAXEL(input)',alpha=0.3)
        ax[1].plot(s_pix.x, s_mean.y+y_fit.y, ls='-', color='red', label='continuum+fringes fit')
        ax[1].plot(s_mean.x, s_pix.y - y_fit.y - 0.2, color='green', ls='-', label='SPAXEL-fringes fit')
        ax[1].plot(s_mean.x, s_mean.y-0.2, label='continuum (correct by polynom)')
        #ax[1].plot(s_mean.x, s_pix.y - y_fit.y - 0.2 - s_mean.y, color='tab:blue', ls='-', label='spaxel-fringe_correction-s_mean')
        #ax[1].axhline(-0.2)

        ax[0].legend(fontsize=15)
        ax[1].legend(fontsize=15)
    if 0:
        fig,ax = plt.subplots()
        ax.set_title('frequency')
        ax.plot(params_vals[:,1])

        plt.show()

    if 0:
        s_pix.fringe_corrected = np.array(s_pix.y-y_fit.y)

        wiggle_model = fringe_model(num_chunks=num_chunks,a= params_vals[:,0], fq=params_vals[:,1], p=params_vals[:,2],b=params_vals[:,3],
                         chiq=params_chiq,x=params_x,chunk_size=chunk_size,chunk_size_delta=chunk_size_delta)

        s_pix.renormalize()
        s_corrected =s_pix.copy()
        s_corrected.y = np.array(s_pix.fringe_corrected)
        return s_corrected,wiggle_model


def fringe_custom_correction_1d(s_pix, s_mean,debug=False,show_fit_chunks=False, fringe_fq_model=fringe_custom_multipix_model,num_chunks=400,
                                n_iter_minimize = 3, title='report',channel=1):
    print_out = False
    continuum_smoothing = True

    s_pix = spectrum(x=np.array(s_pix.x),y=np.array(s_pix.y),err=np.array(s_pix.err))
    s_mean = spectrum(x=np.array(s_mean.x),y=np.array(s_mean.y),err=np.array(s_mean.err))


    s_mean.normalize()
    #smooth
    if continuum_smoothing:
        s_mean.y = savgol_filter(s_mean.y,50,2)

    s_pix.normalize()

    mask_nan = np.isnan(s_pix.y) + np.isnan(s_mean.y)
    z = np.polyfit(s_pix.x[~mask_nan], (s_pix.y - s_mean.y)[~mask_nan], 5)
    polynomial_fit = np.poly1d(z)
    s_fringe = s_pix.copy()
    s_fringe.y = s_pix.y - (s_mean.y + polynomial_fit(s_pix.x))

    if debug and 0:
        fig,ax = plt.subplots(2,1)
        ax[0].plot(s_pix.x, s_pix.y, label='spaxel', color='black')
        ax[0].plot(s_mean.x, s_mean.y, label='Continuum')
        ax[0].plot(s_pix.x, s_mean.y + polynomial_fit(s_pix.x), label='Continuum (polyfit corrected)')
        ax[1].plot(s_pix.x, s_fringe.y , label='Fringes (input) model', color='green')
        ax[0].plot(s_pix.x, polynomial_fit(s_pix.x), label='polyfit function')
        ax[0].set_title('Polynomial Fit correction ')
        ax[0].legend()
        ax[1].legend()
        plt.show()

    wave = s_fringe.x
    #########
    if 1:
        chunks_central_pos = np.array([wave[0] + (wave[-1] - wave[0]) / num_chunks * i for i in range(num_chunks)])
        chunks_left_pos = np.zeros(num_chunks)
        chunks_right_pos = np.zeros(num_chunks)
        chunks_size = np.zeros(num_chunks)
        chunks_fqi = np.zeros(num_chunks)
        chunks_params = []
        for i in range(num_chunks):
            #set a chunk position
            fq_i = fringe_fq_model(chunks_central_pos[i])
            chunks_size[i] = float(2 / fq_i)
            if chunks_size[i]<0.1:
                chunks_size[i] = 0.1
            l_central = chunks_central_pos[i]
            if l_central - chunks_size[i] / 2 < wave[0]:
                l_left = wave[0]
            else:
                l_left = l_central - chunks_size[i] / 2
            l_right = l_left + chunks_size[i]
            if l_right > wave[-1]:
                l_right = wave[-1]
                l_left = wave[-1] - chunks_size[i]
            chunks_left_pos[i] = l_left
            chunks_right_pos[i] = l_right
            # set mask for a chunk position
            mask_chunk_pos = (wave >= l_left) * (wave < l_right)
            npix = np.sum(mask_chunk_pos)  # number of pixels in the chunk
            chunk_size = chunks_size[i]
            if print_out:
                print('chunk#', i, ' pos:', l_left, ':', l_right, ' size:', chunk_size)
            # crate a chunk
            sp_chunk = spectrum(x=s_fringe.x[mask_chunk_pos], y=s_fringe.y[mask_chunk_pos],
                                             err=s_fringe.err[mask_chunk_pos])
            #smooth
            #sp_chunk.y = savgol_filter(sp_chunk.y, 5, 1)

            #define sinusoidal function
            def sin_func(x, amp, p, b, fq=fq_i):
                return amp * np.sin(2 * np.pi * (x-p) * fq) + b

            amp_init = np.std(sp_chunk.y)
            c = ccf(sp_chunk.y, sin_func(sp_chunk.x,amp_init,0,0,fq_i))
            j = find_peaks(c, height=0)[0][0]
            phase_init = sp_chunk.x[j] - sp_chunk.x[0]
            if debug and 0:
                print(find_peaks(c, height=0)[0])
                fig,ax = plt.subplots()
                ax.plot(sp_chunk.x, sp_chunk.y, ls='-', color='black')
                ax.plot(sp_chunk.x, sin_func(sp_chunk.x,amp_init,phase_init,0,fq_i), ls='--', color='red')
                plt.show()

            #minimization

            def fcn2min(params,err = 1):

                fval = sin_func(x=sp_chunk.x, amp=params['amp'].value, p=params['p'].value,
                          b=params['b'].value, fq=params['fq'].value)
                delta = sp_chunk.y - fval
                return delta /0.02

            def calc_chi(params):
                chi = np.sum(np.power(fcn2min(params, err=1),2))
                return chi

            def calc_model(x,params):
                return sin_func(x=x, amp=params['amp'].value, p=params['p'].value,
                          b=params['b'].value, fq=params['fq'].value)


            params = Parameters()
            names = ['fq']
            names.append('amp')
            names.append('phase')
            names.append('b')
            #params.add('fq', value=float(fq_i), min=float(fq_i)*0.9, max=float(fq_i)*1.1,vary=False)
            params.add('fq', value=float(fq_i), min=float(fq_i) * 0.5, max=float(fq_i) * 2, vary=False)
            params.add('amp', value=amp_init, min=0, max=5 * amp_init)
            params.add('p', value=phase_init, vary=True, min=phase_init*0.5, max=phase_init*1.5)
            params.add('b', value=0.0, min=-5 * amp_init, max=5 * amp_init)  # ,min=-2*np.std(sp_chunk[j].y),max=2*np.std(sp_chunk[j].y))
            chi_init =  calc_chi(params)

            for j in range(n_iter_minimize):
                minner = Minimizer(fcn2min, params, nan_policy='propagate',
                                   calc_covar=True)
                result = minner.minimize(method='leastsq')  # method='emcee') #(method='leastsq',params=params)
                # write error report
                #report_fit(result)
                for par in params:
                    params[par].value = result.params[par].value

            chunks_params.append(params)
            if print_out:
                print('chi_fit:', calc_chi(params), 'vs chi_init:', chi_init)
                print('fq_fit:', params['fq'].value, 'vs fq_init:', fq_i)
            chunks_fqi[i] = params['fq'].value


            if debug and 0:

                fig, ax = plt.subplots()
                ax.errorbar(x=sp_chunk.x, y=sp_chunk.y, yerr=sp_chunk.err,fmt='none', color='black')
                ax.plot(sp_chunk.x, sp_chunk.y, ls='-', color='black')
                ax.plot(sp_chunk.x,sin_func(x=sp_chunk.x, amp=params['amp'].value, p=params['p'].value,
                          b=params['b'].value, fq=params['fq'].value), ls='--', color='red')
                plt.show()


        #combine chunks
        fringe_fit_model = spectrum(x=s_fringe.x,y=np.zeros_like(s_fringe.y))
        fringe_fit_model.npix = np.zeros_like(fringe_fit_model.x)
        if debug:
            fringe_fit_model_full = spectrum(x=s_fringe.x, y=np.zeros_like(s_fringe.y))
        for i in range(num_chunks):
            # set mask for a chunk position
            mask_chunk_pos = (fringe_fit_model.x >= chunks_left_pos[i]) * (fringe_fit_model.x <= chunks_right_pos[i])
            par_i =  chunks_params[i].copy()
            par_i['b'].value = 0
            fringe_fit_model.y[mask_chunk_pos]+=calc_model(fringe_fit_model.x[mask_chunk_pos], params=par_i)
            if debug:
                fringe_fit_model_full.y[mask_chunk_pos] +=calc_model(fringe_fit_model.x[mask_chunk_pos], params=chunks_params[i].copy())
            fringe_fit_model.npix[mask_chunk_pos] += 1
            if debug and 0:
                fig, ax = plt.subplots()
                ax.errorbar(x=s_fringe.x, y=s_fringe.y, yerr=s_fringe.err, fmt='none', color='black')
                ax.plot(s_fringe.x, s_fringe.y, ls='-', color='black')
                ax.plot(fringe_fit_model.x, fringe_fit_model.y, ls='--', color='red')
                plt.show()

        fringe_fit_model.y/=fringe_fit_model.npix
        if debug:
            fringe_fit_model_full.y /=fringe_fit_model.npix


        if debug:
            fig, ax = plt.subplots(6,1,sharex=True)
            ax[0].plot(s_pix.x,s_pix.y,label='pixel')
            ax[0].plot(s_pix.x, s_mean.y, label='integrated')
            ax[0].plot(s_pix.x, s_mean.y + polynomial_fit(s_pix.x), label='polyfit adjusted')
            ax[0].plot(s_pix.x, polynomial_fit(s_pix.x), label='polyfit')


            y_shift = 1
            ax[1].errorbar(x=s_fringe.x, y=s_fringe.y, yerr=s_fringe.err,fmt='none', color='black')
            ax[1].plot(s_fringe.x, s_fringe.y, ls='-', color='black',label='fringes in data')
            ax[1].plot(fringe_fit_model.x,fringe_fit_model.y-y_shift, ls='--', color='red',label='fit to fringes')
            ax[1].plot(fringe_fit_model.x, fringe_fit_model_full.y, ls='-', color='red', label='fit to fringes')
            ax[1].plot(fringe_fit_model.x,s_fringe.y-fringe_fit_model.y-2*y_shift, ls='--', color='red',label='difference')
            ax[1].axhline(-y_shift,ls=':',color='grey')

            if 0:
                print('calc pipeline')
                temp_flux = fit_residual_fringes_1d(s_pix.y,s_pix.x, channel=channel,
                                                    dichroic_only=False, max_amp=None)
                #ax[1].plot(fringe_fit_model.x,s_pix.y,label='data',color='black')
                #ax[1].plot(fringe_fit_model.x, temp_flux, label='after pipeline', color='red')
                ax[2].plot(s_fringe.x, s_fringe.y, ls='-', color='black', label='fringes in data')
                ax[2].plot(fringe_fit_model.x, s_pix.y - temp_flux, label='pipeline fit', color='green')
                ax[2].plot(fringe_fit_model.x, s_fringe.y-(s_pix.y - temp_flux)-y_shift, label='difference', color='green')
                #ax[0].plot(fringe_fit_model.x,  s_pix.y - temp_flux, label='difference', color='green')
            else:
                ax[2].plot(s_pix.x, s_pix.y, label='data')
                ax[2].plot(s_pix.x, s_pix.y - fringe_fit_model.y, color='red', label='data corrected')

            ax[3].plot(s_pix.x,s_pix.y,label='data')
            ax[3].plot(s_pix.x, (s_mean.y + polynomial_fit(s_pix.x)), color='black',label='continuum')
            #ax[1].plot(s_pix.x,s_pix.y - fringe_fit_model.y,color='red')
            ax[4].plot(s_pix.x, s_pix.y - fringe_fit_model.y, color='red',label='data corrected')
            ax[4].plot(s_pix.x, (s_mean.y + polynomial_fit(s_pix.x)), color='black',label='continuum')
            ax[5].plot(chunks_central_pos, chunks_fqi, marker='o', color='red', label='fq from the fit')
            ax[5].plot(chunks_central_pos, fringe_fq_model(chunks_central_pos), color='black', label='fq model')
            ax[0].set_title(title)
            for axs in ax[:]:
                axs.legend()

            plt.show()
        fringe_fit_model.y*=s_pix.norm_factor
        return fringe_fit_model


def fringe_custom_correction_second_pixel(s_pix,s_mean,debug=False,show_fit_chunks=False,
                             smooth_scale=50, fringe_init=None,label=''):
    s_mean.normalize()
    s_pix.normalize()

    # correction for nan
    #s_pix.nan_interpolation(label='spaxel')
    #s_mean.nan_interpolation(label='mean')

    # smooth integrated spec
    #win = signal.windows.hann(smooth_scale)
    #s_mean.smoothed = signal.convolve(s_mean.y, win, mode='same') / sum(win)
    #s_mean.smoothed[:25] = s_mean.y[25]
    #s_mean.smoothed[s_mean.x.shape[0] - 25:] = s_mean.smoothed[s_mean.x.shape[0] - 25]

    # calc POLYFIT correction
    #X, Y = s_pix.x, s_pix.y - s_mean.smoothed
    X, Y = s_pix.x, s_pix.y - s_mean.y
    mask_nan = np.isnan(s_pix.y) + np.isnan(s_mean.y)
    z = np.polyfit(X[~mask_nan], Y[~mask_nan], 5)
    polynomial_fit = np.poly1d(z)
    if debug and 1:
        plt.subplots()
        plt.plot(s_pix.x, s_pix.y, label='spaxel')
        plt.plot(s_mean.x, s_mean.y, label='integrated')
        plt.plot(s_pix.x, s_mean.y + polynomial_fit(s_pix.x), label='spaxel polyfit corrected')
        plt.plot(s_pix.x, polynomial_fit(s_pix.x), label='polyfit function')
        plt.title('Polyfit: '+label)
        plt.legend()
        plt.show()
    # correct s_mean for polynomial correction
    s_mean.y += polynomial_fit(s_mean.x)

    wave = s_pix.x
    flux = s_pix.y - s_mean.y
    flux_err = s_pix.err

    def fit_to_wiggles_second(wave, flux, flux_err=None, plot_results=0, plot_parameter_stats=0, return_params_model=0,fringe_init=fringe_init):

        # set x range binning
        delta_x = wave[-1] - wave[0]
        num_chunks = fringe_init.num_chunks
        chunk_size = fringe_init.chunk_size
        chunk_size_delta = fringe_init.chunk_size_delta
        params_vals = np.zeros((num_chunks, 4))
        params_chiq = np.zeros(num_chunks)
        params_x = np.zeros(num_chunks)

        for i in range(num_chunks):
            l_left = (chunk_size - chunk_size_delta) * i
            l_right = chunk_size * (i + 1) - chunk_size_delta * i
            mask = (wave >= wave[0] + l_left) * (wave < wave[0] + l_right)

            # calc fringe signal in chunk
            data_x = wave[mask]  # - s_pix.x[mask][0]
            data_y = flux[mask]  # -polynomial_fit(s_pix.x[mask])
            data_err = flux_err[mask]

            # par_names = ["a", "f", "p",'b']
            # lmfit to fringe signal
            from lmfit import Model
            if i == 0:
                init = [0.2, 5, 0, 0]

            # set frequence from fit to brightest pixel
            fq = fringe_init.fq[i]

            if 0:
                def func(x, a, p, b):
                    return a * np.sin(2 * np.pi * x * fq + p) + b
                fmodel = Model(func)
                result = fmodel.fit(data_y, x=data_x, a=init[0], p=init[2], b=init[3])
                #print(result.fit_report())
                init = [result.best_values['a'], fq, result.best_values['p'], result.best_values['b']]
            else:
                fitter = modeling.fitting.LevMarLSQFitter()
                model = modeling.models.Sine1D(frequency=fq)  # depending on the data you need to give some initial values
                model.frequency.fixed = True
                mask_nan = np.isnan(data_y)
                fitted_model = fitter(model, data_x[~mask_nan], data_y[~mask_nan])
                a = fitted_model.amplitude.value
                fq = fitted_model.frequency.value
                p = fitted_model.phase.value
                b = 0
                init = [a, fq, 2 * np.pi * p, b]

                def func(x, a, p, b):
                    return a * np.sin(2 * np.pi * x * fq + p) + b


            fit = func(data_x, init[0], init[2], init[3])
            # calc chi2:
            chiq = np.sum(np.power((data_y - fit) / np.std(data_y), 2))
            chiqred = chiq / data_y.shape[0]
            #print(i, init, 'chiq_red:', chiq / data_y.shape[0])

            params_chiq[i] = chiqred
            params_vals[i, :] = init
            params_x[i] = wave[0] + l_left
            if show_fit_chunks:
                fig, ax = plt.subplots(1, 2)
                ax[1].errorbar(data_x, data_y, yerr=data_err, color='black', label='data')
                ax[1].plot(data_x, func(data_x, init[0], init[2], init[3]), label='lmfit: fq=' + str(round(fq, 2)))
                ax[1].legend()
                plt.show()

        # combine chunks
        y_fit = spectrum(wave.copy(), flux.copy() * 0, flux_err.copy())
        y_fit.npix = np.zeros_like(y_fit.x)

        def func(x, a, f, p, b):
            return a * np.sin(2 * np.pi * x * f + p) + b

        for i in range(num_chunks):
            res = params_vals[i, :]
            chiq = params_chiq[i]
            if 1:  # chiq<chiqlimit:
                l_left = (chunk_size - chunk_size_delta) * i
                l_right = chunk_size * (i + 1) - chunk_size_delta * i
                mask = (wave >= wave[0] + l_left) * (wave < wave[0] + l_right)

                data_x = wave[mask]
                y = func(data_x, res[0], res[1], res[2], res[3])
                y_fit.y[mask] += y
                y_fit.npix[mask] += 1
        y_fit.y /= y_fit.npix
        #print('zeros_pix', np.sum(y_fit.npix == 0), wave[np.where(y_fit.npix == 0)[0]])

        if plot_results:
            plt.subplots()
            plt.plot(wave, flux, label='data')
            plt.plot(y_fit.x, y_fit.y, label='model')
            plt.plot(y_fit.x, flux - y_fit.y - 0.2, label='corrected data')
            plt.legend()
            plt.show()

        if plot_parameter_stats:
            # correct params for outlliers
            for i in range(4):
                p = params_vals[:, i]
                for j in range(p.shape[0]):
                    if j > 0 and j < p.shape[0] - 1:
                        if np.abs(p[j] - p[j - 1]) > np.abs(p[j + 1] - p[j - 1]) and np.abs(p[j + 1] - p[j]) > np.abs(
                                p[j + 1] - p[j - 1]):
                            f_interp_p = interp1d([j - 1, j + 1], [p[j - 1], p[j + 1]])
                            p[j] = f_interp_p(j)
            # show relation fit parameters with coordinate
            if 1:
                px = np.array([(chunk_size - chunk_size_delta) * i for i in range(num_chunks)]) + s_pix.x[0]
                mask_good_points = params_vals[:, 1] > 0  # 5*np.median(params_vals[:, 1])
                fig, ax = plt.subplots(1, 4, sharex=True)
                # npoints = params_vals[:,0].shape[0]
                x = np.linspace(px[0], px[-1], 100)
                p_interp = []
                for i in range(4):
                    ax[i].plot(px[mask_good_points], params_vals[:, i][mask_good_points], 'o')
                    f_interp = interp1d(px[mask_good_points], params_vals[:, i][mask_good_points],
                                        fill_value='extrapolate')
                    ax[i].plot(x, f_interp(x))
                    p_interp.append(f_interp)

                plt.show()

        if return_params_model:
            return y_fit, num_chunks, params_x, params_vals, params_chiq
        else:
            return y_fit

    (y_fit, num_chunks, params_x, params_vals, params_chiq) = fit_to_wiggles_second(wave=wave, flux=flux, flux_err=flux_err,
                                                                             plot_parameter_stats=0, return_params_model=1)



    #second iteration
    #y_fit_2 = fit_to_wiggles(wave=y_fit.x, flux=flux - y_fit.y, flux_err=y_fit.err)
    y_fit_2 = fit_to_wiggles_second(wave=y_fit.x, flux=flux - y_fit.y, flux_err=y_fit.err,plot_parameter_stats=0, return_params_model=0)
    y_fit.y += y_fit_2.y

    # third iteration
    # y_fit_3 = fit_to_wiggles(wave=y_fit.x, flux=flux-y_fit.y, flux_err=y_fit.err)
    # y_fit.y += y_fit_3.y

    if debug:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
        ax[0].plot(s_pix.x, s_pix.y, label='pixel')
        ax[0].plot(s_mean.x, s_mean.y, label='integrated')
        ax[0].plot(s_mean.x, s_pix.y - s_mean.y, label='fringes')
        ax[0].plot(s_pix.x, y_fit.y, color='red', label='fit')
        ax[0].plot(s_pix.x, s_pix.y - s_mean.y - y_fit.y - 0.2, color='tab:blue', label='subtracted')

        ax[1].plot(s_pix.x, s_pix.y, ls='-', color='black', label='spaxel')
        ax[1].plot(s_pix.x, s_mean.y + y_fit.y, ls='--', color='red', label='mean+fringes_fit')
        # ax[1].plot(s_pix.x, y_fit.y, ls='--', color='red', label='model')

        ax[1].plot(s_mean.x, s_pix.y - y_fit.y, color='green', ls='-', label='after')
        ax[1].plot(s_mean.x, s_pix.y - y_fit.y - 0.2 - s_mean.y, color='tab:blue', ls='-',
                   label='diff with integrated corrected for polyfit')
        ax[1].axhline(-0.2)
        ax[0].set_title('Fringefit: '+label)
        ax[0].legend()
        ax[1].legend()
        plt.show()

    s_pix.fringe_corrected = np.array(s_pix.y - y_fit.y)
    wiggle_model = fringe_model(num_chunks=num_chunks, a=params_vals[:, 0], fq=params_vals[:, 1], p=params_vals[:, 2],
                                b=params_vals[:, 3],
                                chiq=params_chiq, x=params_x, chunk_size=fringe_init.chunk_size, chunk_size_delta=fringe_init.chunk_size_delta)

    s_pix.renormalize()
    s_corrected = s_pix.copy()
    s_corrected.y = np.array(s_pix.fringe_corrected)
    return s_corrected, wiggle_model


if __name__ == '__main__':

    if 1:

        s_pix = []
        #f = np.loadtxt('/home/slava/science/codes/python/jwst/output/tmp/fringe/QSO-B1830-211-SIGHTLINEB_Jan24_2C_ch2-long_s3d_(A)_1.spec1d')
        f = np.loadtxt(
            '/home/slava/science/codes/python/jwst/output/tmp/fringe/spaxel_0.spec1d')
        s_pix.append(spectrum(x=f[:, 0], y=f[:, 1], err=f[:, 2]))

        f = np.loadtxt(
            '/home/slava/science/codes/python/jwst/output/tmp/fringe/spaxel_1.spec1d')
        s_pix.append(spectrum(x=f[:, 0], y=f[:, 1], err=f[:, 2]))

        f = np.loadtxt(
            '/home/slava/science/codes/python/jwst/output/tmp/fringe/spaxel_2.spec1d')
        s_pix.append(spectrum(x=f[:, 0], y=f[:, 1], err=f[:, 2]))

        f = np.loadtxt(
            '/home/slava/science/codes/python/jwst/output/tmp/fringe/spaxel_3.spec1d')
        s_pix.append(spectrum(x=f[:, 0], y=f[:, 1], err=f[:, 2]))

        f = np.loadtxt(
            '/home/slava/science/codes/python/jwst/output/tmp/fringe/spaxel_4.spec1d')
        s_pix.append(spectrum(x=f[:, 0], y=f[:, 1], err=f[:, 2]))

        f = np.loadtxt(
            '/home/slava/science/research/kulkarni/JWST-DLAs/ID2441/Continuum/fit_cont.dat')
        f_interp = interp1d(f[:, 0], f[:, 1], fill_value='extrapolate')

        s_mean = spectrum(x=s_pix[0].x, y=f_interp(s_pix[0].x), err=f_interp(s_pix[0].x) * 0.01)


        #run pipeline 1d fringe correction
        if 1:
            ii = 0
            temp_flux = fit_residual_fringes_1d(np.array(s_pix[ii].y), s_pix[ii].x, channel=3,
                                                    dichroic_only=False, max_amp=None)
            plt.subplots()
            plt.plot(s_pix[ii].x,s_pix[ii].y,label='data')
            plt.plot(s_pix[ii].x,temp_flux,label='data corrected')
            plt.plot(s_pix[ii].x,s_pix[ii].y-temp_flux,label='fringes')
            plt.legend()
            plt.show()


        # run custom 1pix fringe correction
        if 0:
            s_corrected,wiggle_model = fringe_custom_correction_1d(s_pix=s_pix[0].copy(), s_mean=s_mean.copy(), debug=True,
                                 show_fit_chunks=0, chiqlimit=7)
            plt.show()

        # run custom multipixels fringe correction
        if 1:
            fringe_fq_model = fringe_custom_multipix_model(s_pix_array=s_pix, s_mean=s_mean.copy(),
                                        debug=False, show_fit_chunks=False)

            fringe_custom_correction_1d(s_pix[2], s_mean.copy(),debug=True,show_fit_chunks=False, fringe_fq_model=fringe_fq_model)

        if 1:
            x=1

        if 0:
            plt.subplots()
            plt.plot(s_pix[0].x, s_pix[0].y)
            plt.plot(s_pix[0].x, temp_flux*0.8)
            plt.plot(s_pix[0].x, s_pix[0].y - temp_flux)

            #plt.plot(s_corrected.x, s_corrected.y*0.8)
            #plt.plot(s_pix.x, s_pix.y - s_corrected.y)
            plt.show()
            #(s_model, fr_model) = fringe_custom_correction(s_pix=s_pix,s_mean=s_mean,debug=True,
            #                         show_fit_chunks=0,chiqlimit=7)



    if 0:
        f = np.loadtxt(
            '/home/slava/science/codes/python/jwst/output/tmp/fringe_corr/J1007+2853_dith=1_ch3-short_s3d_(A)_bkgr.spec1d')
        s_mean = spectrum(x=f[:, 0], y=f[:, 1], err=f[:, 2])

        f = np.loadtxt(
             '/home/slava/science/codes/python/jwst/output/tmp/fringe_corr/J1007+2853_dith=1_ch3-short_s3d_(A)_sci.spec1d')
        #s_pix = spectrum(x=f[:, 0], y=f[:, 1], err=f[:, 2])

        (s_model, fr_model1) = fringe_custom_correction_second_pixel(s_pix=s_pix,s_mean=s_mean,debug=True,
                                 show_fit_chunks=0,fringe_init=fr_model)

        f = np.loadtxt(
            '/home/slava/science/codes/python/jwst/output/tmp/fringe_corr/J1007+2853_3A_ch3-short_s3d_(A)_bkgr_v3.spec1d')
        #s_pix = spectrum(x=f[:, 0], y=f[:, 1], err=f[:, 2])

        (s_model, fr_model2) = fringe_custom_correction(s_pix=s_pix,s_mean=s_mean,debug=True,
                                 show_fit_chunks=0,chiqlimit=7)


        #(x,y,yerr, fr_model2) = fringe_custom_correction(f_x=s_pix_2.x,f_y=s_pix_2.y,f_err=s_pix_2.err,f_int_x=s_mean.x,fint_y=s_mean.y,debug=True,
        #                         show_fit_chunks=False,chiqlimit=7)

        #(x, y, yerr, fr_model3) = fringe_custom_correction(f_x=s_pix_3.x, f_y=s_pix_3.y, f_err=s_pix_3.err,
        #                                                   f_int_x=s_mean.x, fint_y=s_mean.y, debug=True,
        #                                                   show_fit_chunks=False, chiqlimit=7)

        #(x, y, yerr, fr_model4) = fringe_custom_correction(f_x=s_pix_4.x, f_y=s_pix_4.y, f_err=s_pix_4.err,
        #                                                   f_int_x=s_mean.x, fint_y=s_mean.y, debug=True,
        #                                                   show_fit_chunks=False, chiqlimit=7)
        #(x, y, yerr, fr_model5) = fringe_custom_correction(f_x=s_pix_5.x, f_y=s_pix_5.y, f_err=s_pix_5.err,
        #                                                   f_int_x=s_mean.x, fint_y=s_mean.y, debug=True,
        #                                                   show_fit_chunks=False, chiqlimit=7)

        plt.subplots()
        plt.plot(fr_model.x,fr_model.fq,label='0')
        plt.plot(fr_model2.x,fr_model2.fq,label='2')
        plt.plot(fr_model1.x, fr_model1.fq,label='1')
        plt.legend()
    plt.show()
    print('')
