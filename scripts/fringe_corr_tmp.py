import os, glob
import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from stdatamodels.jwst import datamodels
import scipy




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

    def normalize(self):
        norm = np.mean(self.y[10:40])
        self.y/=norm
        self.err/=norm
        self.norm_factor = norm

f = np.loadtxt('/home/slava/science/codes/python/jwst/output/detector3/roi_spectra/fringe_corr/J1007+2853_dith=1_ch2-long_s3d_(A)_sci.spec1d')
s_mean = spectrum(x=f[:,0],y=f[:,1],err=f[:,2])

f =  np.loadtxt('/home/slava/science/codes/python/jwst/output/detector3/roi_spectra/fringe_corr/J1007+2853_dith=1_ch2-long_s3d_(A)_bkgr.spec1d')
s_pix = spectrum(x=f[:,0],y=f[:,1],err=f[:,2])

s_mean.normalize()
s_pix.normalize()

from scipy import signal

win = signal.windows.hann(50)

s_mean.smoothed = signal.convolve(s_mean.y, win, mode='same') / sum(win)
s_mean.smoothed[:25]=s_mean.y[25]
s_mean.smoothed[s_mean.x.shape[0]-25:]=s_mean.smoothed[s_mean.x.shape[0]-25]
plt.subplots()
plt.plot(s_mean.x,s_mean.smoothed)
plt.plot(s_pix.x,s_pix.y)
plt.plot(s_mean.x,s_mean.y)
plt.plot(s_pix.x,s_pix.y-s_mean.y)

s_mean.y = np.array(s_mean.smoothed)

plt.show()


from scipy.optimize import curve_fit
def func(x,a,f,p,b):
    return a*np.sin(2*np.pi*x*f+p)+b


def log_prior(theta):
    a, b, c, d = theta
    if a > 0 and 0 < c < 2*np.pi and 0 < b < 0.1:
        return 0.0
    return -np.inf

def log_likelihood(theta,x,y,err):
    a, b, c, d = theta

    def func(x, a, b, c, d):
        return a * np.sin(2 * np.pi * x / b + c) + d

    delta = y - func(x, a, b, c, d)
    ln = -0.5 * np.sum(np.power(delta / err, 2))
    return ln


def log_probability(theta,x,y,err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta,x,y,err)

def mcmcm_fit(data_x, data_y,data_err, init=[0.5, 0.1, -1.3, 0]):
    import emcee

    ndim = 4
    nwalkers = 500
    nsteps = 1000


    # run sampler
    if 1:

        # set start position
        init_range = [0.5, 0.02, 3, 0.1]
        pos2 = []
        for i in range(nwalkers):
            prob = -np.inf
            while prob == -np.inf:
                rndm = np.random.randn(ndim)
                wal_pos = init + init_range * rndm
                prob = log_probability(theta=wal_pos,x=data_x,y=data_y,err=data_err)
            pos2.append(wal_pos)
        #pos = [init + init_range * np.random.randn(ndim) for i in range(nwalkers)]
        if 1:
            from multiprocessing import Pool
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,pool=pool,args=(data_x,data_y,data_err))
                sampler.run_mcmc(pos2, nsteps, progress=True)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
            for i, result in enumerate(sampler.sample(pos2, iterations=nsteps)):
                if i % int(nsteps / 10) == 0:
                    print("{0:5.1%}".format(float(i) / nsteps))

    if 1:
        samples = sampler.chain[:, :, :]
        means = np.zeros((ndim, nsteps))
        vars = np.zeros((ndim, nsteps))
        single = np.zeros((ndim, nsteps))
        for i in range(nsteps):
            for j in range(ndim):
                means[j, i] = np.mean(samples[:, i, j])
                vars[j, i] = np.std(samples[:, i, j])
                single[j, i] = samples[5, i, j]

        # chain stats
        if 0:
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
            else:
                i = 0
                ax[0].errorbar(np.arange(nsteps), means[i, :], yerr=vars[i, :],
                               fmt='-', color='black',
                               markeredgecolor='black', markeredgewidth=2, capsize=2,
                               ecolor='royalblue', alpha=0.7)
                ax[0].plot(np.arange(nsteps), single[i, :], color='red')

        chain = sampler.chain[:, int(nsteps * 0.9):, :].reshape((-1, ndim))

        # analyse the chain using chainconsumer
        from chainconsumer import ChainConsumer

        c = ChainConsumer()

        if 0:
            par_names = ["a", "b", "c", "d"]
            c.add_chain(chain, parameters=par_names)
            c.plotter.plot(filename="example.png", figsize="column")
            res = c.analysis.get_summary(parameters=par_names)
            print(res)

            def func(x, a, b, c, d):
                return a * np.sin(2 * np.pi * x / b + c) + d

            plt.subplots()
            popt = [res[r][1] for r in par_names]
            plt.plot(data_x, data_y)
            plt.plot(data_x, func(data_x, popt[0], popt[1], popt[2], popt[3]))
        if 1:
            par_names = ["a", "b", "c", "d"]
            c.add_chain(chain, parameters=par_names)
            res = c.analysis.get_summary(parameters=par_names)
            print(res)
        return res

#POLYFIT
X,Y = s_pix.x,s_pix.y-s_mean.y
z = np.polyfit(X, Y, 3)
polinimial_fit = np.poly1d(z)
print(z)
plt.subplots()
plt.plot(X,Y)
plt.plot(X,polinimial_fit(X))
plt.show()

chunk_size = 0.12
chunk_size_delta = 0.11
delta_x = s_pix.x[-1]-s_pix.x[0]
num_chunks = np.where(np.array([chunk_size*i - chunk_size_delta*(i-1) for i in range(500)])<delta_x)[0][-1]+1

params_vals = np.zeros((num_chunks,4))
params_chiq = np.zeros(num_chunks)

debug = False

for i in range(num_chunks):
    l_left = (chunk_size-chunk_size_delta)*i
    l_right = chunk_size*(i+1) - chunk_size_delta*i
    mask = (s_pix.x>s_pix.x[0]+l_left)*(s_pix.x<s_pix.x[0]+l_right)
    print(i, np.sum(mask), 'region:', l_left, l_right)
    data_x = s_pix.x[mask]
    data_y = s_pix.y[mask]-s_mean.y[mask]-polinimial_fit(s_pix.x[mask])
    s = np.mean(data_y)
    data_err = np.power(s_pix.err[mask]**2 + s_mean.err[mask]**2,0.5)

    par_names = ["a", "f", "p",'b']
    case = 'lmfit'
    if case == 'mcmc':
        if i == 0:
            res = mcmcm_fit(data_x,data_y,data_err)
        else:
            init = [res[r][1] for r in par_names]
            res = mcmcm_fit(data_x, data_y,data_err)
        init = [res[r][1] for r in par_names]
        #params_vals[i,:] = init
    elif case == 'lmfit':
        from lmfit import Model
        if i ==0:
            init = [0.2, 0.1, 0, 0]
        #fft
        if 0:
            from scipy.fft import fft, fftfreq
            yf = fft(data_y)
            N = data_x.shape[0]
            T=1/800
            xf = fftfreq(N, T)[:N // 2]

            yf_line = 2.0 / N * np.abs(yf[0:N // 2])
            y_disp = np.std(yf_line)

            if yf_line[np.argmax(yf_line)]>5.*y_disp:
                fq = xf[np.argmax(yf_line)]
                print('fft:', 1 / fq)
                def func(x, a, p,b):
                    return a * np.sin(2 * np.pi * x  * fq+p) + b
                fmodel = Model(func)
                result = fmodel.fit(data_y, x=data_x, a=init[0], c=init[2])
                print(result.fit_report())
            else:
                fq = init[1]
                def func(x, a, c):
                    return a * np.sin(2 * np.pi * (x - c) * fq)
                fmodel = Model(func)
                result = fmodel.fit(data_y, x=data_x, a=init[0], c=init[2])
                print(result.fit_report())
            init = [result.best_values['a'],fq, result.best_values['c']]
        if 1:
            from scipy.fft import fft, fftfreq
            yf = fft(data_y-np.mean(data_y))
            N = data_x.shape[0]
            T=1/800
            xf = fftfreq(N, T)[:N // 2]

            yf_line = 2.0 / N * np.abs(yf[0:N // 2])
            y_disp = np.std(yf_line)

            if yf_line[np.argmax(yf_line)]>5.*y_disp:
                fq = xf[np.argmax(yf_line)]
            else:
                fq = init[1]
            def func(x, a, p,b):
                return a * np.sin(2 * np.pi * x  * fq+p) + b

            fmodel = Model(func)
            result = fmodel.fit(data_y, x=data_x, a=init[0], p=init[2],b=init[3])
            print(result.fit_report())
            init = [result.best_values['a'],fq, result.best_values['p'], result.best_values['b']]

        print('init',init)


        fit= func(data_x, init[0], init[2], init[3])
        chiq = np.sum(np.power((data_y-fit)/data_err,2))
        chiqred = chiq/data_y.shape[0]
        print('chiq_red:',chiq/data_y.shape[0])
        params_chiq[i] = chiqred
    print(i,init)
    params_vals[i, :] = init
    if debug:
        fig, ax = plt.subplots(1, 2)
        ax[0].axhline(1 * y_disp,ls='--')
        ax[0].axhline(5.5 * y_disp)
        ax[0].plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
        ax[1].plot(data_x,data_y)
        ax[1].plot(data_x, func(data_x, init[0], init[2], init[3]), label='mcmcfit')
        plt.show()




plt.subplots()

plt.plot(s_pix.x,s_pix.y - s_mean.y-polinimial_fit(s_pix.x),label='data')

#calc_mean_fit
y_fit = spectrum(s_pix.x.copy(),s_pix.y.copy(),s_pix.err.copy())
y_fit.y *=0
y_fit.npix = np.zeros_like(y_fit.x)


def func(x, a, f,p, b):
    return a * np.sin(2 * np.pi * x * f + p) + b
for i in range(num_chunks):
    res = params_vals[i, :]
    chiq= params_chiq[i]
    if chiq<5:
        l_left = (chunk_size - chunk_size_delta) * i
        l_right = chunk_size * (i + 1) - chunk_size_delta * i
        mask = (s_pix.x > s_pix.x[0] + l_left) * (s_pix.x < s_pix.x[0] + l_right)

        data_x = s_pix.x[mask]
        y = func(data_x, res[0], res[1], res[2], res[3])
        y_fit.y[mask] +=y
        y_fit.npix[mask]+=1
        #plt.subplots()
        #plt.plot(y_fit.x,y_fit.y/y_fit.npix)
        #plt.plot(s_pix.x[mask],y)
        #plt.plot(s_pix.x[mask], s_pix.y[mask]-s_mean.y[mask])
        #plt.show()

y_fit.y/=y_fit.npix

for i in range(num_chunks):
    res = params_vals[i,:]
    def func(x, a, b, c):
        return a * np.sin(2 * np.pi * (x-c) * b )


    l_left = (chunk_size - chunk_size_delta) * i
    l_right = chunk_size * (i + 1) - chunk_size_delta * i
    mask = (s_pix.x > s_pix.x[0] + l_left) * (s_pix.x < s_pix.x[0] + l_right)

    data_x = s_pix.x[mask] - s_pix.x[0]
    data_y = s_pix.y[mask] - np.mean(s_pix.y[mask])
    #plt.plot(s_pix.x[mask], data_y,color='black')
    #plt.plot(s_pix.x[mask], func(data_x, res[0], res[1], res[2]))

    #def f_second(x, c, a=res[0], b=res[1]):
    #    return a * np.sin(2 * np.pi * (x-c) / b )
    #popt = res[2]
    #for i in range(5):
    #    popt, pcov = curve_fit(f_second, data_x-s_pix.x[0],  s_pix.y[mask],p0=popt)
    #    print(popt)
    #res[2] = popt
    #plt.plot(data_x, func(data_x - data_x[0], res[0], res[1], res[2], res[3]), label='mcmcfit',ls='--')
plt.plot(y_fit.x,y_fit.y)
plt.legend()
plt.show()


fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
ax[0].plot(s_pix.x,s_pix.y,label='pixel')
ax[0].plot(s_mean.x,s_mean.y,label='integrated')
ax[0].plot(s_mean.x,s_pix.y-s_mean.y-polinimial_fit(s_mean.x),label='fringes')
ax[0].plot(s_mean.x,polinimial_fit(s_mean.x),label='polyfit')
ax[0].plot(s_pix.x,y_fit.y,color='red',label='fit')

#ax[1].plot(s_pix.x,s_pix.y,label='pixel',ls=':')
ax[1].plot(s_mean.x,s_pix.y-y_fit.y,color='green',ls='-')
#ax[1].plot(s_mean.x,1.05+y_fit.y,color='red',ls='-')
#ax[1].plot(s_mean.x,s_mean.y+polinimial_fit(s_mean.x)+y_fit.y,color='green')

ax[0].legend()
ax[1].legend()

plt.show()


#fit parameter evolution with wavelength

px = np.array([(chunk_size - chunk_size_delta) * i for i in range(num_chunks)])+ s_pix.x[0]
mask_good_points =  params_vals[:, 1]>0 #5*np.median(params_vals[:, 1])

#make correctioo for flipped values
mask_flip = params_vals[:,0]<0
#params_vals[:,0][mask_flip] *=-1
#params_vals[:,2][mask_flip] -= np.pi

fig,ax = plt.subplots(1,4,sharex=True)
#npoints = params_vals[:,0].shape[0]
x = np.linspace(px[0],px[-1],100)
p_interp = []
for i in range(4):
    ax[i].plot(px[mask_good_points],params_vals[:,i][mask_good_points],'o')
    f_interp = interp1d(px[mask_good_points], params_vals[:,i][mask_good_points],fill_value='extrapolate')
    ax[i].plot(x,f_interp(x))
    p_interp.append(f_interp)

plt.show()





fp_smoothed = []
for i in range(4):
    win = signal.windows.hann(10)
    y = p_interp[i](x)
    f_par_smoothed = signal.convolve(y, win, mode='same') / sum(win)
    ax[i].plot(x,f_par_smoothed)
    fp_smoothed.append(interp1d(x,f_par_smoothed,fill_value='extrapiolate'))

fig,ax = plt.subplots(1,3,sharex=True)
xx = np.linspace(x[0],x[-1],1000)
[a,b,c] = [fp_smoothed[i](xx) for i in range(3)]


def func(x, a, b, c):
    return 0.1 * np.sin(2 * np.pi * (x-c) / b )
ax[0].plot(xx,func(xx,a,b,c))
ax[1].plot(xx,b)
ax[1].plot(xx,a)
ax[1].plot(px[mask_good_points],params_vals[:,1][mask_good_points],'o')
ax[2].plot(xx,2 * np.pi * (xx-xx[0]) / b)
ax[2].plot(xx,2 * np.pi * (xx-xx[0]))
plt.show()