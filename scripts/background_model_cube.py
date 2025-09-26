import numpy as np
from jwst.background.background_sub import mask_from_source_cat
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import glob,os
from stdatamodels.jwst import datamodels
import scipy
from statsmodels.tsa.stattools import ccf
from spectrum_model import *
from lmfit import Minimizer, Parameters


def fit_components_contribution(sp,fit_comps,algorithm = 'mcmc',mask=None,debug=True):

    if mask is None:
        mask = sp.x>-1
    data = spectrum(sp.x[mask],sp.y[mask],sp.err[mask])
    s_comp = {}
    for i,el in enumerate(fit_comps):
        if np.nansum(el.y)>0:
            s_comp['c'+str(i)] = spectrum(el.x[mask],el.y[mask],el.err[mask])
    n_comp = len(s_comp)

    params = Parameters()
    names = ['c'+str(i) for i in range(n_comp)]
    init_values = [1 + i*0 for i in range(n_comp)]
    for name, value in zip(names, init_values):
        params.add(name, value=value, min=-np.inf, max=np.inf)


    #prior for pixels position
    def lnprior(params, px, py):
        #prior for known position of source (c0 component), e.g. 11,11
        print()
        rad_1 = np.sqrt((px - 11) ** 2 + (py - 11) ** 2)
        # rad_2 = (px-14)**2+(py-16)**2
        if rad_1 > 5:
            f = params['c0'] * 100
            return -f
        else:
            return 0

    def fit_model(params, s_comp):
        # params is a vocabulary
        pi = [p.value for p in params.values()]
        f = np.nansum([s.y * pi[i] for i, s in enumerate(s_comp.values())], axis=0)
        return f

    def fit_model_mcmc(params, s_comp):
        # params is an array
        f = np.nansum([s.y * params[i] for i, s in enumerate(s_comp.values())], axis=0)
        return f

    if algorithm == 'leastsq':
        xtol = 1e-10
        ftol = 1e-10
        gtol = 1e-10
        epsfcn = 1e-10
        stderr= 0.1

        def fcn2min(params):
            add_prior = 0
            verbose = 0
            m_i = fit_model(params=params,s_comp=s_comp)
            res = (data.y - m_i) / data.err

            #if rebin_spec:
            #    m_i = rebin_arr(m_i, factor)
            #    res = (y_rebinned - m_i) / err_rebinned

            if add_prior:
                res += lnprior(params,px,py)
            weight = np.ones_like(res)
            weight[res<0] = 1
            if verbose:
                print(params)
                print('residuals:',np.sum(res))
            return res*weight

        for i in range(3):
            minner = Minimizer(fcn2min, params)
            result = minner.leastsq(xtol=xtol,ftol=ftol,gtol=gtol,epsfcn=epsfcn)
            if i <2:
                for par in params:
                    params[par].value = result.params[par].value + np.random.normal(scale=stderr)

    elif algorithm == 'mcmc':
        stderr = 0.5
        plot_mcmc = False

        def log_prior(params=[1,2,3]):
            #prior for known position of source (c0 component), e.g. 11,11
            if params[0]<0:
                return -np.inf
                #if par>1 and i == 1:
                #    return -np.inf
            chi_0 = -(params[0]-1)**2/0.01**2 #   +(params[1]-0)**2/0.5**2)
            return 5*chi_0

        def log_prob_data(params=[1,2,3]):
            m_i = fit_model_mcmc(params=params, s_comp=s_comp)
            mask_nan = ~np.isnan(data.y)
            residuals = (data.y[mask_nan] - m_i[mask_nan])
            err = data.err[mask_nan]
            chi = -0.5 * np.nansum(np.power(residuals / err, 2))
            return chi

        def log_probability(params=[1,2,3]):
            chi_prior = log_prior(params)
            if ~np.isinf(chi_prior):
                #m_i = fit_model_mcmc(params=params, s_comp=s_comp)
                #residuals = (data.y - m_i)
                #err = data.err
                #chi = -0.5 * np.nansum(np.power(residuals / err, 2))
                chi = log_prob_data(params)
                return chi+chi_prior
            else:
                return chi_prior

        def fcn2min(params):
            m_i = fit_model(params=params, s_comp=s_comp)
            mask_nan = ~np.isnan(data.y)
            res = (data.y[mask_nan] - m_i[mask_nan]) / data.err[mask_nan]
            return res

        #print('sum_isnan',np.sum(~np.isnan(data.y)))

        minner = Minimizer(fcn2min, params)
        result = minner.minimize(method='leastsq')
        for par in params:
           result.params[par].stderr = stderr

        result.params['c0'].value=1
        result.params['c1'].value=np.nanmedian(data.y[data.x<500])
        print('c1_0',result.params['c1'].value)

        print('run emcee')
        ndim, nwalkers, nsteps = n_comp, 100, 100
        init = [result.params[par].value for par in params]
        init_range = [result.params[par].stderr for par in  params]
        p0 = []
        for i in range(nwalkers):
            prob = -np.inf
            while prob == -np.inf:
                rndm = np.random.randn(ndim)
                wal_pos = init + init_range * rndm
                prob = log_probability(params=wal_pos)
            p0.append(wal_pos)
        p0 = np.array(p0)

        if 1:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
            sampler.run_mcmc(p0, nsteps, progress=plot_mcmc)
        else:
            #multiprocessing
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn=log_probability, pool=pool)
                sampler.run_mcmc(p0, nsteps, progress=plot_mcmc)

        samples = sampler.chain[:, int(nsteps / 2):, :].reshape((-1, ndim))

        if plot_mcmc:
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            samples_image = sampler.get_chain()
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples_image[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples_image))
                ax.set_ylabel(names[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
            plt.show()

        c = ChainConsumer()
        c.add_chain(samples, walkers=nwalkers, parameters=names)
        if plot_mcmc:
            c.configure(smooth=True,
                        cloud=True,
                        sigmas=[0, 1, 2, 3],
                        )
            #c.configure_truth(ls='--', lw=1., c='lightblue')  # c='darkorange')
            c.plotter.plot(figsize=(30, 30),
                           # filename="output/fit.png",
                           display=True,
                           )

        res_mcmc = c.analysis.get_summary(parameters=names)
        print('res_mcmc:', res_mcmc)
        for el in names:
            params[el].value = res_mcmc[el][1]
            if None in res_mcmc[el]:
                stderr = np.inf
            else:
                stderr = (res_mcmc[el][2] - res_mcmc[el][0]) / 2

            params[el].stderr = stderr

    if debug:
        fit = fit_model(params=params,s_comp=s_comp)
        fig,ax = plt.subplots(1,2,sharex=True,sharey=True)

        ax[0].plot(data.x, data.y, label='data',lw=0.5)
        ax[0].errorbar(x=data.x, y=data.y,yerr=data.err, fmt='none',lw=0.5)

        ax[0].plot(data.x, fit, label='fit')
        #for i,s in enumerate(s_comp.values()):
        #    ax[0].plot(s.x, s.y * params['c'+str(i)].value, label='c'+str(i), lw=0.5)
        ax[0].axhline(0, ls=':', color='black')
        ax[0].legend()
        ax[1].plot(data.x, data.y-fit, label='Residuals')
        ax[1].plot(data.x, data.y, label='Data',lw=1,zorder=-10)
        ax[1].legend()
        ax[1].set_title(str(int(log_prob_data([params['c'+str(i)].value for i in range(2)]))))
        plt.show()

    return params,int(log_prob_data([params['c'+str(i)].value for i in range(2)]))



def subtract_comp(cube, mask_spaxels,fit_comps,amp_comps,
                  active_comps_list, debug=False):

    npars = len(fit_comps)
    cube_data = np.array(cube.data)
    cube_err = np.array(cube.err)
    mask_spaxels = np.array(mask_spaxels)
    amp_comps = np.array(amp_comps)



    pos = np.where(mask_spaxels==True)

    for i in range(pos[0].shape[0]):
        print('iter',i, 'out of', pos[0].shape[0])
        px, py = pos[0][i],  pos[1][i]
        print('px',px,' py',py)
        fit_comps_local = [fit_comps[k].copy() for k in range(npars)]

        fc = np.zeros_like(cube_data[:, px, py])
        for k in active_comps_list:
            amp = amp_comps[k,px,py]
            fc += fit_comps_local[k].y*amp

        if debug:
            plt.subplots()
            plt.plot(cube_data[:, px, py])
            plt.plot(fc)
            plt.show()
        cube_data[:, px, py] -= fc

    return cube_data, cube_err
    #return pars_array,pars_std_array

def save_local_cube(cube_data, cube_err, orig_name='', new_name=''):

    #filename = cube.cubename.split('_s3d.fits')[0] + '_' + new_cube_filename + local_name + '_s3d.fits'
    hdu1 = fits.open(orig_name)
    hdu1['SCI'].data = cube_data
    hdu1['ERR'].data = cube_err
    hdu1.writeto(new_name, overwrite=True)
    hdu1.close()
    specname = orig_name.split('_s3d.fits')[0] + '_x1d.fits'
    hdu1 = fits.open(specname)
    hdu1.writeto(new_name.split('_s3d.fits')[0] + '_x1d.fits', overwrite=True)
    hdu1.close()

    print('New cube was saved to', new_name)



path_to_cube = '/home/slava/science/codes/python/jwst/output/detector3/'
filename = 'QSO-B1830-211-SIGHTLINEB_2B_ch2-medium_Fringe_free_s3d.fits'
path_to_bckg = '/home/slava/science/codes/python/jwst/output/detector3/background/'
bkg_filename = filename.split('_s3d')[0] + '_bkgr.pkl'
new_cubename = filename.split('_s3d')[0] + '_bkgr_sbtr_s3d.fits'

cube = datamodels.open(path_to_cube+filename)
x = np.arange(cube.data.shape[1]) #vertical - rows (Y)
y = np.arange(cube.data.shape[2]) # horizontal - columns (X)
X,Y = np.meshgrid(y,x)

result = cube.data.copy()
derr  = cube.err.copy()
data = cube.data.copy()


#
npix = data.shape[0]
ncol = data.shape[2]
nrow = data.shape[1]
# Detector pixels coord
x = np.arange(ncol)
y = np.arange(nrow)
X, Y = np.meshgrid(x, y)
#
data_image = np.nanmedian(data,axis=0)
wavelength = np.arange(npix)

if 1:
    #2A
    #posA = 11, 13
    #posB = 15, 16
    posA = 11, 11
    posB = 16, 14

    rad = 7
    mask_radA = (np.sqrt((X - posA[0]) ** 2 + (Y - posA[1]) ** 2)).astype(int)
    mask_compA = (mask_radA <= rad)
    mask_radB = (np.sqrt((X - posB[0]) ** 2 + (Y - posB[1]) ** 2)).astype(int)
    mask_compB = (mask_radB <= rad)
    mask_spaxels = ~mask_compA*(~mask_compB)


#smooth data
if 1:
    filter_kernel = np.ones((2, 2))
    data_smoothed = data.copy()
    for i in range(npix):
        x = data_smoothed[i].copy()
        x[np.isnan(x)] = 0
        ncounts = scipy.signal.convolve2d(np.ones_like(x), filter_kernel, mode='same', boundary='fill', fillvalue=0)
        data_smoothed[i] = scipy.signal.convolve2d(x, filter_kernel, mode='same', boundary='fill', fillvalue=0) / ncounts

    #new_name = 'QSO-B1830-211-SIGHTLINEB_2A_2X_smoothed_s3d.fits'
    #save_local_cube(data_smoothed, derr, orig_name=path_to_cube + filename, new_name=path_to_cube + new_name)
    fig,ax = plt.subplots(1,2,sharey=True,sharex=True)
    ax[0].imshow(x,origin='lower')
    ax[1].imshow(x,origin='lower')
    ax[1].set_title('smoothed')
    plt.show()

#create a bkg model
print(data.shape,2+0.7*(data.shape[1]-4))
delta_edge = 1
mask_for_edge = (Y>delta_edge)*(Y<nrow-delta_edge)*(X>delta_edge)*(X<ncol-delta_edge)
mask_nan  = ~np.isnan(np.sum(data,axis=0))
if 0:
    mask_nan *=  ~np.isnan(np.sum(data_smoothed,axis=0))
mask_for_edge *= mask_nan
nlow = delta_edge*3+ int(0.7*(nrow-delta_edge*3*2))
# create a mask
mask_for_bkg = (Y>1*delta_edge)*(Y<nrow-1*delta_edge)*(X>1*delta_edge)*(X<ncol-1*delta_edge)*mask_spaxels*mask_nan
mask_spaxels *= mask_for_edge
#
#mask_for_bkg2 = np.sqrt((X-20)**2 + (Y-29)**2)<2

# set mask for spaxels
snr = np.abs(np.nanmedian(cube.data / np.nanstd(cube.data, axis=0), axis=0))
#mask_spaxels = (snr <= 2)*mask_for_edge


#plot mask
plt.imshow(np.log10(np.abs(data_image)),origin='lower')
plt.contour(X, Y, mask_for_bkg.astype(float), levels=[0], colors='green', linewidths=1.5)

plt.contour(X, Y, mask_spaxels.astype(float), levels=[0], colors='red', linewidths=1.5)
plt.show()


pos = np.where(mask_for_bkg==True)
lst= np.zeros((npix,ncol))
for i in range(ncol):
    lst[:,i] = np.nanmedian(data[:,i,nlow:nrow-delta_edge],axis=1)
comp_bkg = spectrum(x=wavelength,y= np.nanmedian(lst,axis=1),err= np.nanstd(lst,axis=1))

#comp_bkg.normalize(delta_x=-1)
comp_flat = spectrum(x=wavelength,y= np.ones(npix),err= np.zeros(npix))
#comp_flat.normalize(delta_x=-1)
fit_comps = [comp_bkg,comp_flat]


if 0:
    res = np.zeros((3,nrow,ncol))

    pos = np.where(mask_spaxels == True)
    for i in range(pos[0].shape[0]):
        print('iter', i, 'out of', pos[0].shape[0])
        px, py = pos[0][i], pos[1][i]
        print('px', px, ' py', py)
        icol = py
        irow = px

        sp = spectrum(x=wavelength, y=data_smoothed[:, irow, icol].copy(), err=derr[:, irow, icol].copy() * 2)
        #sp.normalize(delta_x=-1)
        if 0:
            plt.subplots()
            plt.plot(sp.y)
            plt.plot(comp_bkg.y,lw=0.5)
            plt.show()

        pars,chiq = fit_components_contribution(sp,fit_comps,algorithm = 'mcmc',mask=None,debug=False)
        res[0,irow, icol] = pars['c0'].value
        res[1, irow, icol] = pars['c1'].value
        res[2, irow, icol] = -chiq/npix


    with open(path_to_bckg+bkg_filename, 'wb') as f:
        pickle.dump(res, f)

    fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
    im1 =ax[0].imshow(res[0],origin='lower')
    im2 = ax[1].imshow(res[1],origin='lower')
    ax[2].imshow(res[2])
    fig.colorbar(im1)
    fig.colorbar(im2)


if 1:
    with open(path_to_bckg+bkg_filename, 'rb') as f:
        (res) = pickle.load(f)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    im1 = ax[0].imshow(res[0], origin='lower')
    im2 = ax[1].imshow(res[1], origin='lower')
    ax[2].imshow(res[2], origin='lower',vmin=0.5,vmax=1.5)
    fig.colorbar(im1)
    fig.colorbar(im2)
    ax[1].contour(X, Y, res[1]<3, levels=[0], colors='red', linewidths=1.5)
    ax[0].set_title('Comp1')
    ax[1].set_title('Comp2')
    ax[2].set_title('Chi2')
    plt.show()
#make 2D interpolations
if 1:
    from scipy.interpolate import RectBivariateSpline,RBFInterpolator

    xs,ds = [],[]
    pos = np.where(mask_spaxels==True)
    for i in range(pos[0].shape[0]):
        px, py = pos[0][i],  pos[1][i]
        xs.append([px,py])
        ds.append(res[1,px,py])
    xs = np.asarray(xs)
    f = RBFInterpolator(y=xs,d=ds,kernel='multiquadric',epsilon=1)

    xgrid = np.mgrid[0:nrow, 0:ncol]
    xflat = xgrid.reshape(2, -1).T
    yflat = f(xflat)
    ygrid = yflat.reshape(nrow, ncol)


    fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
    ax[0].imshow(res[1],vmin= 0,vmax =5,origin='lower')
    ax[1].imshow(ygrid,vmin= 0,vmax =5,origin='lower')
    ax[2].imshow(res[1]-ygrid, vmin=-2, vmax=2, origin='lower')

    ax[0].contour(X, Y, snr>1, levels=[0], colors='blue', linewidths=1.5)
    ax[0].contour(X, Y, mask_spaxels, levels=[0], colors='red', linewidths=1.5)
    ax[1].contour(X, Y, mask_spaxels, levels=[0], colors='red', linewidths=1.5)
    ax[2].contour(X, Y, mask_spaxels, levels=[0], colors='red', linewidths=1.5)

    fig,ax = plt.subplots(5,1,sharex=True,sharey=True)
    for i in np.arange(5):
        ax[0].set_title('rows')
        ax[i].plot(res[1][5+i*5,:],label=str(5+i*5))
        ax[i].plot(ygrid[5+i*5, :])
        ax[i].legend()

    fig, ax = plt.subplots(5, 1, sharex=True, sharey=True)
    for i in np.arange(5):
        ax[0].set_title('cols')
        ax[i].plot(res[1][:,5 + i * 5], label=str(5 + i * 5))
        ax[i].plot(ygrid[:,5 + i * 5])
        ax[i].legend()

    plt.show()
    res[1] = ygrid

if 1:
    amp_comps = [res[0],res[1]]
    mask_spaxels = mask_for_edge
    cube_data, cube_err = subtract_comp(cube=cube, mask_spaxels=mask_spaxels,
                                        fit_comps=fit_comps,
                                        amp_comps=amp_comps, active_comps_list=[0, 1],
                                         debug=False)


    save_local_cube(cube_data, cube_err, orig_name=path_to_cube + filename, new_name=path_to_cube + new_cubename)
plt.show()