import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from crds.core.log import verbose
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import os
import astropy,scipy,pickle
from scipy.interpolate import interp1d
import matplotlib.patches as patches
from lmfit import Minimizer, Parameters
from spectrum_model import *
from multiprocessing import Pool
import emcee
from chainconsumer import ChainConsumer
from scipy.signal import savgol_filter


def read_settings(init_file='./../init.dat'):
    init_settings = {}
    with open(init_file) as f:
        for k, line in enumerate(f):
            values = [s for s in line.split()]
            if line[0] != '#':
                if values[0] == 'input1_dir:':
                    input_dir = values[1]
                    init_settings['input1_dir'] = values[1]
                if values[0] == 'input2_dir:':
                    init_settings['input2_dir'] = values[1]
                if values[0] == 'input3_dir:':
                    init_settings['input3_dir'] = values[1]
                if values[0] == 'spec2_cachedir:':
                    init_settings['spec2_cachedir'] = values[1]
                if values[0] == 'output1_dir:':
                    init_settings['output1_dir'] = values[1]
                if values[0] == 'output2_dir:':
                    init_settings['output2_dir'] = values[1]
                if values[0] == 'output3_dir:':
                    init_settings['output3_dir'] = values[1]
                if values[0] == 'CRDS_PATH:':
                    init_settings['CRDS_PATH'] = values[1]
                if values[0] == 'CRDS_SERVER_URL:':
                    init_settings['CRDS_SERVER_URL'] = values[1]
                if values[0] == 'WEBBPSF_PATH:':
                    init_settings['WEBBPSF_PATH'] = values[1]
    return init_settings
settings =  read_settings()
os.environ["CRDS_PATH"] = settings['CRDS_PATH']
os.environ["CRDS_SERVER_URL"] = settings['CRDS_SERVER_URL']
if 'CRDS_CONTEXT' in  settings.keys():
    os.environ["CRDS_CONTEXT"] = settings['CRDS_CONTEXT']
os.environ["WEBBPSF_PATH"] = settings['WEBBPSF_PATH']

path_to_jwst_folder = '/home/slava/science/codes/python/jwst/'



def rebin_2d(array, factor):
    array = np.array(array)
    # Ensure the array shape is divisible by the rebinning factor
    shape = (array.shape[0] // factor, factor, array.shape[1] // factor, factor)
    # Reshape and compute the mean along the rebinned axes
    return array.reshape(shape).mean(axis=(1, 3))*factor**2

def upsample_2d(array, factor):
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def fit_model(params,s_comp):
    #params is a vocabulary
    pi = [p.value for p in params.values()]
    f = np.nansum([s.y*pi[i] for i,s in enumerate(s_comp.values())],axis=0)
    return f

def fit_model_mcmc(params,s_comp):
    #params is an array
    f = np.nansum([s.y*params[i] for i,s in enumerate(s_comp.values())],axis=0)
    return f



def fit_components_contribution(sp,fit_comps,px,py,algorithm = 'leastsq',mask=None,debug=False):

    if mask is None:
        mask = np.ones_like(sp.x)
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
        params.add(name, value=value, min=0, max=np.inf)


    rebin_spec = False
    if rebin_spec:
        snr = np.nanmedian(sp.y) / np.nanstd(sp.y)
        print('snr:',snr)
        if snr<5 and 1:
            factor = 10
            x_rebinned = rebin_arr(sp.x, factor)
            y_rebinned = rebin_arr(sp.y, factor)
            err_rebinned = np.zeros_like(y_rebinned)+ np.std(y_rebinned)/3
            rebin_spec = True
    if rebin_spec:
        mask = ~(np.isnan(y_rebinned))


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
        stderr = 0.1
        plot_mcmc = False

        def log_prior(params=[1,2,3]):
            #prior for known position of source (c0 component), e.g. 11,11
            for par in params:
                if par<0:
                    return -np.inf
            return 0

        def log_prob_data(params=[1,2,3]):
            m_i = fit_model_mcmc(params=params, s_comp=s_comp)
            residuals = (data.y - m_i)
            err = data.err
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
            res = (data.y - m_i) / data.err
            return res

        minner = Minimizer(fcn2min, params)
        result = minner.minimize(method='leastsq')
        for par in params:
           result.params[par].stderr = stderr

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
        plt.subplots()

        plt.plot(data.x, data.y, label='data')
        plt.errorbar(x=data.x, y=data.y,yerr=data.err, fmt='none')

        plt.plot(data.x, fit, label='fit')
        for i,s in enumerate(s_comp.values()):
            plt.plot(s.x, s.y * params['c'+str(i)].value, label='c'+str(i), lw=0.5)
        plt.axhline(0, ls=':', color='black')
        plt.legend()
        plt.show()

    return params


def psf_fit(cube, wave, mask_spaxels,fit_comps,mask_spectrum,mask_comps=None,debug=False,plot_pixel_fit =False):

    npars = len(fit_comps)
    cube_data = np.array(cube.data)
    cube_err = np.array(cube.err)
    mask_spaxels = np.array(mask_spaxels)
    pars_array = np.zeros((npars,cube_data.shape[1],cube_data.shape[2]))
    pars_std_array = np.zeros((npars, cube_data.shape[1], cube_data.shape[2]))
    wave = np.array(wave)


    pos = np.where(mask_spaxels==True)
    if mask_comps==None:
        mask_comps = [mask_spaxels for i in range(npars)]
    for i in range(pos[0].shape[0]):
        print('iter',i, 'out of', pos[0].shape[0])
        px, py = pos[0][i],  pos[1][i]
        print('px',px,' py',py)
        sp = spectrum(x=wave, y=cube_data[:, px, py],err=cube_err[:, px, py])
        sp.normalize(delta_x=-1)

        fit_comps_local = [fit_comps[k].copy() for k in range(npars)]

        for k in range(npars):
            print('mask_comps['+str(k)+'[px,py]',mask_comps[k][px,py])
            fit_comps_local[k].y *= float(mask_comps[k][px,py])
        if px>16:
            plot_pixel_fit = False
            pars = fit_components_contribution(sp, fit_comps=fit_comps_local, algorithm='mcmc',debug=plot_pixel_fit,px=px,py=py,
                                               mask=mask_spectrum)
        else:
            pars = fit_components_contribution(sp, fit_comps=fit_comps_local, algorithm='mcmc', debug=plot_pixel_fit, px=px,
                                               py=py, mask=mask_spectrum)
        ind = 0
        for k in range(npars):
            if mask_comps[k][px,py]==True:
                pars_array[k,px,py] = pars['c'+str(ind)].value
                pars_std_array[k, px, py] = pars['c' + str(ind)].stderr
                ind+=1
    if debug:
        import matplotlib.colors as colors
        cmap = plt.cm.viridis
        cmap.set_bad('black')
        vmin,vmax = -1.5,0
        fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
        def plot_pars(x,mode='log'):
            if mode == 'linear':
                return x
            elif mode == 'log':
                return np.log10(np.abs(x))

        for i in range(npars):
            im = ax[i].imshow(plot_pars(pars_array[i]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
            ax[i].set_title('c' + str(i))
        fig.colorbar(im)

        vmin, vmax = 0, 0.05
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        for i in range(npars):
            im = ax[i].imshow((pars_std_array[i]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
            ax[i].set_title('c' + str(i))
        fig.colorbar(im)


        vmin, vmax = 0, 100
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
        for i in range(npars):
            im = ax[i].imshow((pars_array[i]/pars_std_array[i]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
            ax[i].set_title('c'+str(i))
        #ax[1].imshow((pars_std_array[1]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
        #im = ax[2].imshow((pars_std_array[2]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
        fig.colorbar(im)
        plt.show()

    return pars_array,pars_std_array


def read_psf(channel='1C',source='custom_psf',overdist = False):
    if source == 'webbpsf':
        filename = path_to_jwst_folder+'data/miri_psf/webb_psf/'+'psf_'+channel
        if overdist==False:
            with open(filename+'.pkl', 'rb') as f:
                (psf_image) = pickle.load(f)
            return psf_image
        if overdist==True:
            with open(filename+'_overdist.pkl', 'rb') as f:
                (psf_image_overdist) = pickle.load(f)
            return psf_image_overdist
    elif source == 'custom_psf':
        filename = path_to_jwst_folder + 'data/miri_psf/custom_psf/' + 'star_psf_' + channel
        with open(filename + '.pkl', 'rb') as f:
            (psf_image) = pickle.load(f)
        return  psf_image


def subtract_comp(cube, wave, mask_spaxels,fit_comps,amp_comps,
                  active_comps_list, mask_comps=None,debug=False):

    npars = len(fit_comps)
    cube_data = np.array(cube.data)
    cube_err = np.array(cube.err)
    mask_spaxels = np.array(mask_spaxels)
    wave = np.array(wave)
    amp_comps = np.array(amp_comps)



    pos = np.where(mask_spaxels==True)

    for i in range(pos[0].shape[0]):
        print('iter',i, 'out of', pos[0].shape[0])
        px, py = pos[0][i],  pos[1][i]
        print('px',px,' py',py)
        sp = spectrum(x=wave, y=np.array(cube_data[:, px, py]),err=np.array(cube_err[:, px, py]))
        sp.normalize(delta_x=-1)
        pix_norm_flux = np.array(sp.norm_f)
        fit_comps_local = [fit_comps[k].copy() for k in range(npars)]

        for k in active_comps_list:
            fc = fit_comps_local[k].y*pix_norm_flux*amp_comps[k,px,py]
            if 0:
                plt.subplots()
                plt.plot(cube_data[:, px, py])
                plt.plot(fc)

                plt.title('c'+str(k))
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



if __name__ == '__main__':

    #2B
    if 0:
        # PKS1830
        if 0:
            fname = 'QSO-B1830-211-SIGHTLINEB_1C_ch1-long_for_PSF_s3d.fits'
            path_cube = path_to_jwst_folder + '/scripts/tmp_output/'
            path_spec1d = path_to_jwst_folder + '/scripts/tmp_output/'
            specA_name = 'QSO-B1830-211-SIGHTLINEB_1C_ch1-long_s3d_(A)_green.spec1d'
            specB_name = 'QSO-B1830-211-SIGHTLINEB_1C_ch1-long_s3d_(A)_red.spec1d'
        if 1:
            fname = 'QSO-B1830-211-SIGHTLINEB_2B_ch2-medium_BKG_SUB_s3d.fits'
            path_cube = path_to_jwst_folder + 'output/detector3/'
            path_spec1d = path_to_jwst_folder + 'scripts/tmp_output/'
            specA_name = 'QSO-B1830-211-SIGHTLINEB_2B_ch2-medium_s3d_(A)_green.spec1d'
            specB_name = 'QSO-B1830-211-SIGHTLINEB_2B_ch2-medium_s3d_(A)_red.spec1d'

        cube = datamodels.open(path_cube + fname)
        cube_shape = cube.data.shape  # Data cube: 5 slices, 60x60 pixels each
        # read wavelenght
        sstring = fname.split('_s3d')[0] + '_x1d.fits'
        hdu2 = fits.open(path_cube + sstring)
        wavelength = hdu2['EXTRACT1D'].data['WAVELENGTH']
        hdu2.close()

        #define CompC
        if 1:
            specC2 = np.loadtxt(path_spec1d+'tmp/' + 'QSO-B1830-211-SIGHTLINEB_2B_ch2-medium_BKG_SUB_s3d_(A)_red.spec1d')
            specC1 = np.loadtxt(path_spec1d+'tmp/' + 'QSO-B1830-211-SIGHTLINEB_2B_ch2-medium_BKG_SUB_s3d_(A)_green.spec1d')
            specC1 = spectrum(x=specC1[:, 0], y=specC1[:, 1], err=specC1[:, 2])
            specC2 = spectrum(x=specC2[:, 0], y=specC2[:, 1], err=specC2[:, 2])
            specC1.normalize(delta_x=-1)
            specC2.normalize(delta_x=-1)
            specC = specC1.copy()
            yf_line = savgol_filter(specC1.y, 200, 3)
            plt.subplots()
            plt.plot(specC1.x,specC1.y)
            plt.plot(specC1.x,yf_line)
            plt.plot(specC2.x,specC2.y)
            plt.show()
            specC.y = np.array(yf_line)

        #read qso spectra
        specA = np.loadtxt(path_spec1d+specA_name)
        specA = spectrum(x=specA[:,0],y=specA[:,1],err=specA[:,2])
        specA.normalize(delta_x = -1)
        specB = np.loadtxt(path_spec1d+specB_name)
        specB = spectrum(x=specB[:,0],y=specB[:,1],err=specB[:,2])
        specB.normalize(delta_x = -1)


        plt.subplots()
        plt.plot(specA.x, specA.y)
        plt.plot(specB.x, specB.y)
        plt.plot(specC.x, specC.y)
        plt.title('Components')
        plt.show()

        plt.subplots()
        plt.imshow(np.nanmedian(cube.data,axis=0),origin='lower')

        plt.subplots()
        snr = np.abs(np.nanmedian(cube.data/np.nanstd(cube.data,axis=0),axis=0))
        plt.imshow(np.log10(snr),origin='lower')
        plt.title('SNR')
        #set mask for spaxels
        mask_spaxels = snr > 5
        #set mask for spaxels near the edge
        x = np.arange(snr.shape[1])
        y = np.arange(snr.shape[0])
        X, Y = np.meshgrid(x, y)
        mask_spaxels*=(X>1)*(Y>1)
        #plot mask
        plt.contour(X, Y, mask_spaxels.astype(float), levels=[0], colors='green', linewidths=1.5)
        plt.show()

        mask_spectrum = (wavelength >8.7)*(wavelength<10)
        if 0:
            pars_array, pars_std_array = psf_fit(cube,wave=wavelength, mask_spaxels=mask_spaxels, fit_comps = [specA, specB,specC],mask_spectrum =mask_spectrum)
            with open('./tmp_output/psf_fit.pkl', 'wb') as f:
                pickle.dump((pars_array, pars_std_array),f)
        ########################
        if 1:
            with open('./tmp_output/psf_fit.pkl', 'rb') as f:
                (pars_array, pars_std_array) = pickle.load(f)
            npars = pars_array.shape[0]
            for i in range(npars):
                pars_array[i][pars_array[i]<=0] = np.nan
            cmap = plt.cm.viridis
            cmap.set_bad('black')
            vmin, vmax = 0., 1
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)


            def plot_pars(x, mode='linear'):
                if mode == 'linear':
                    return x
                elif mode == 'log':
                    return np.log10(np.abs(x))


            for i in range(2):
                im = ax[i].imshow(plot_pars(pars_array[i]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
                ax[i].set_title('c' + str(i))
            im2 = ax[2].imshow(plot_pars(pars_array[2]), vmin=0, vmax=0.3, origin='lower', cmap=cmap)
            print('median(c2):',np.nanmedian(pars_array[2]))

            fig.colorbar(im)
            fig.colorbar(im2)

            vmin, vmax = 0, 0.05
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
            for i in range(npars):
                im = ax[i].imshow((pars_std_array[i]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
                ax[i].set_title('c' + str(i))
            fig.colorbar(im)

            vmin, vmax = 0, 100
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
            for i in range(npars):
                im = ax[i].imshow((pars_array[i] / pars_std_array[i]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
                ax[i].set_title('c' + str(i))
            # ax[1].imshow((pars_std_array[1]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
            # im = ax[2].imshow((pars_std_array[2]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
            fig.colorbar(im)
            plt.show()

        if 0:
            px,py = 33, 1
            sp = cube.data[:,px,py]
            sp = spectrum(x=wavelength,y=cube.data[:,px,py],err=cube.err[:,px,py])
            sp.normalize(delta_x = 0.01)

            plt.subplots()
            plt.plot(specA.x,specA.y,label='sA')
            plt.plot(specB.x,specB.y,label='sB')
            plt.plot(sp.x, sp.y, label='sp')
            plt.legend()
            plt.show()

            pars = fit_components_contribution(sp,comps = [specA,specB,specC],algorithm = 'leastsq')


            fit = fit_model(params=pars, sA=specA, sB=specB)

            plt.subplots()
            plt.plot(sp.x, sp.y, label='data')
            plt.plot(sp.x, fit, label='fit')
            plt.plot(sp.x, specA.y*pars['ca'].value, label='A',lw=0.5)
            plt.plot(sp.x, specB.y*pars['cb'].value, label='B',lw=0.5)
            plt.axhline(0,ls=':',color='black')
            plt.legend()
        plt.show()


    #2A
    if 1:
        # PKS1830
        channel = '2B'
        if 1:
            if channel == '2A':
                fname = 'QSO-B1830-211-SIGHTLINEB_2A_ch2-short_Fringe_free_bkgr_sbtr_s3d.fits'

                path_cube = path_to_jwst_folder + 'output/detector3/'
                path_spec1d =  path_cube+'roi_spectra/'
                #path_spec1d = path_to_jwst_folder + 'scripts/tmp_output/'
                specA_name = fname.split('.fits')[0]+'_(A)_green.spec1d'
                specB_name = fname.split('.fits')[0]+'_(A)_red.spec1d'
                posA = 11,13
                posB = 15,16
            if channel == '2B':
                fname = 'QSO-B1830-211-SIGHTLINEB_2B_ch2-medium_Fringe_free_bkgr_sbtr_s3d.fits'
                path_cube = path_to_jwst_folder + 'output/detector3/'
                path_spec1d = path_cube + 'roi_spectra/'
                # path_spec1d = path_to_jwst_folder + 'scripts/tmp_output/'
                specA_name = fname.split('.fits')[0] + '_(A)_green.spec1d'
                specB_name = fname.split('.fits')[0] + '_(A)_red.spec1d'
                posA = 11,11
                posB = 16,14


        cube = datamodels.open(path_cube + fname)
        cube_shape = cube.data.shape  # Data cube: 5 slices, 60x60 pixels each
        # read wavelenght
        sstring = fname.split('_s3d')[0] + '_x1d.fits'
        hdu2 = fits.open(path_cube + sstring)
        wavelength = hdu2['EXTRACT1D'].data['WAVELENGTH']
        hdu2.close()

        x = np.arange(cube.data.shape[2]) #rows - Y
        y = np.arange(cube.data.shape[1]) #cols - X
        X, Y = np.meshgrid(x, y)

        # read qso spectra
        specA = np.loadtxt(path_spec1d + specA_name)
        specA = spectrum(x=specA[:, 0], y=specA[:, 1], err=specA[:, 2])
        specA.normalize(delta_x=-1)

        specB = np.loadtxt(path_spec1d + specB_name)
        specB = spectrum(x=specB[:, 0], y=specB[:, 1], err=specB[:, 2])
        specB.normalize(delta_x=-1)

        # define CompC
        if channel == '2A':
            mask_spectrumA = (specA.x < 7.8)
            mask_spectrumB = (specB.x >7.9)*(specB.x < 8.2)
        elif channel == '2B':
            mask_spectrumA = (specA.x < 9.3)*(specA.x > 9.1)
            mask_spectrumB = (specB.x < 8.9)
        if 1:
            rad = 6 #pix
            mask_regA = np.sqrt((X - posA[0])**2 +(Y -posA[1])**2)<rad
            mask_regB = np.sqrt((X - posB[0]) ** 2 + (Y - posB[1]) ** 2) < rad
            snr = np.array(np.abs(np.nanmedian(cube.data / np.nanstd(cube.data, axis=0), axis=0)))
            mask_snr = snr>0.5
            mask_compA = mask_regA*(~mask_regB)*mask_snr
            print('npix in mask_compA:',np.sum(mask_compA))
            #plot mask
            if 1:
                fig, ax = plt.subplots(1, 4, sharey=True, sharex=True)
                ax[0].imshow(mask_regA, origin='lower')
                ax[0].set_title('Mask_compA')
                ax[1].imshow(mask_regB, origin='lower')
                ax[1].set_title('Mask_compB')
                ax[2].imshow((mask_snr), origin='lower')
                ax[2].set_title('SNR')
                ax[3].imshow(np.nanmedian(cube.data, axis=0), origin='lower')
                for axs in ax[:]:
                    axs.contour(X, Y, (mask_compA).astype(float), levels=[0], colors='green', linewidths=1.5)
                plt.show()

            def derive_compC_mcmc(cube=cube,mask_spaxels=mask_regA,spq=specA,show_fit = False,mask_spectrum=None):

                spq = spq.copy()
                spq.normalize(delta_x=-1)
                wave = spq.x
                sp_flat = spectrum(x=wave, y=np.ones_like(wave),err=np.ones_like(wave)*0.1)
                sp_flat.normalize(delta_x=-1)

                pars_array, pars_std_array = psf_fit(cube=cube, wave=wave, mask_spaxels=mask_spaxels,
                                                     fit_comps=[spq, sp_flat], mask_spectrum=mask_spectrum,plot_pixel_fit=show_fit)

                #make composite
                p0 = pars_array[0]
                pos = np.where(mask_spaxels == True)
                nspec = pos[0].shape[0]
                npix = len(wave)
                composite = np.zeros((nspec,npix)) #spectrum(x=wave,y=np.zeros_like(wave),err=np.zeros_like(wave))
                data = np.array(cube.data)
                data_err=  np.array(cube.err)
                for i in range(nspec):
                    px, py = pos[0][i], pos[1][i]
                    qxy = spectrum(x=wave,y=data[:,px,py],err=data_err[:,px,py])
                    qxy.normalize(delta_x=-1)
                    composite[i,:] = qxy.y - p0[px,py]*spq.y
                    if show_fit:
                        fig, ax = plt.subplots(1,2)
                        ax[0].plot(qxy.x,qxy.y)
                        ax[0].plot(qxy.x, p0[px,py]*spq.y,ls='--')
                        ax[0].plot(composite.x,qxy.y - p0[px,py]*spq.y,color='black')
                        ax[1].plot(composite.x,composite.y,color='red')
                        ax[1].plot(composite.x, qxy.y - p0[px, py] * spq.y, color='black')
                        plt.show()

                composite = spectrum(x=wave,y=np.nanmedian(composite,axis=0),err=np.zeros_like(wave))

                composite.normalize(delta_x=1)
                spq.normalize(delta_x=1)
                #smoothed_line = savgol_filter(composite.y, 50, 1)
                if 1:
                    plt.subplots()
                    plt.plot(composite.x,composite.y)
                    #plt.plot(composite.x, smoothed_line)
                    #composite.smooth_by_rebinning(window=10)
                    #plt.plot(composite.x,composite.y,color='black',lw=2)
                    plt.plot(spq.x, spq.y,c='red')
                    plt.show()
                    #composite.y = smoothed_line
                return composite




            comp_gal_A = derive_compC_mcmc(cube=cube,mask_spaxels=mask_compA,spq=specA,show_fit = False,mask_spectrum=mask_spectrumA)
            #comp_gal_A = derive_compC(show_fit = False, mask_wave = (specA.x<8))

            #mask_regA = np.sqrt((X - posA[0])**2 +(Y -posA[1])**2)<rad*1
            #mask_regB = np.sqrt((X - posB[0]) ** 2 + (Y - posB[1]) ** 2) < rad
            mask_compB = mask_regB*(~mask_regA)*mask_snr
            print('npix in mask_compB:',np.sum(mask_compB))
            if 1:
                fig, ax = plt.subplots(1, 4, sharey=True, sharex=True)
                ax[0].imshow(mask_regA, origin='lower')
                ax[0].set_title('Mask_compA')
                ax[1].imshow(mask_regB, origin='lower')
                ax[1].set_title('Mask_compB')
                ax[2].imshow((mask_snr), origin='lower')
                ax[2].set_title('SNR')
                ax[3].imshow(np.nanmedian(cube.data, axis=0), origin='lower')
                for axs in ax[:]:
                    axs.contour(X, Y, (mask_compB).astype(float), levels=[0], colors='green', linewidths=1.5)
                plt.show()

            #comp_gal_B = derive_compC(cube=cube,mask=mask_regB,spq=specB,show_fit = False, mask_wave = (specB.x>7.9)*(specB.x<8.2))
            comp_gal_B = derive_compC_mcmc(cube=cube, mask_spaxels=mask_compB, spq=specB, show_fit=False,
                                      mask_spectrum=mask_spectrumB)


            specC = spectrum(x=wavelength,y=(comp_gal_A.y+comp_gal_B.y)/2,err=(comp_gal_A.y+comp_gal_B.y)/2/100)

            plt.subplots()
            plt.plot(comp_gal_A.x,comp_gal_A.y,label='comp_gal_A')
            plt.plot(comp_gal_B.x,comp_gal_B.y,label='comp_gal_B')
            plt.plot(specC.x,specC.y,label='comp_mean')

            plt.legend()
            plt.show()
            with open('./tmp_output/'+fname.split('.fits')[0]+'_compC.pkl', 'wb') as f:
                pickle.dump(specC,f)
            specC.normalize(delta_x=-1)
        #read CompC from file
        else:
            with open('./tmp_output/'+fname.split('.fits')[0]+'_compC.pkl', 'rb') as f:
                specC = pickle.load(f)
                specC.normalize(delta_x=-1)

        plt.subplots()
        plt.plot(specA.x, specA.y)
        plt.plot(specB.x, specB.y)
        plt.plot(specC.x, specC.y)
        specC.y = savgol_filter(specC.y, 100, 1)
        plt.plot(specC.x, specC.y)
        plt.title('Components')
        plt.show()

        # set mask for spaxels
        snr = np.abs(np.nanmedian(cube.data / np.nanstd(cube.data, axis=0), axis=0))
        mask_spaxels = snr >= 1
        # mask for spaxels near the edge
        x = np.arange(snr.shape[1])
        y = np.arange(snr.shape[0])
        X, Y = np.meshgrid(x, y)
        mask_spaxels *= (X > 1) * (Y > 1)


        # plot mask
        if 1:
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(np.nanmedian(cube.data, axis=0), origin='lower')
            ax[1].imshow(np.log10(snr), origin='lower')
            ax[1].set_title('mask SNR')
            ax[0].contour(X, Y, mask_spaxels.astype(float), levels=[0], colors='green', linewidths=1.5)
            ax[1].contour(X, Y, mask_spaxels.astype(float), levels=[0], colors='green', linewidths=1.5)
            #plt.show()

        #mask for spectral region
        mask_spectrum = (wavelength >0)

        # mask for components
        star_psf = read_psf(channel=channel)
        if 1:
            plt.subplots()
            plt.title('PSF'+channel)
            zmax = np.nanmax(star_psf)
            xc, yc = np.argwhere(star_psf == zmax)[0]
            (xmax, ymax) = star_psf.shape
            plt.plot(np.arange(ymax) - yc, star_psf[xc, :] / zmax, label='xc')
            plt.plot(np.arange(xmax) - xc, star_psf[:, yc] / zmax, label='yc')
            plt.axhline(1e-2, ls=':')
            plt.yscale('log')
            plt.legend()
            plt.show()


        rad = 6
        mask_radA = (np.sqrt((X - posA[0]) ** 2 + (Y - posA[1]) ** 2)).astype(int)
        mask_compA = np.array(mask_spaxels) *  (mask_radA <= rad)
        mask_radB = (np.sqrt((X - posB[0]) ** 2 + (Y - posB[1]) ** 2)).astype(int)
        mask_compB = np.array(mask_spaxels) * (mask_radB <= rad)
        mask_radB = np.sqrt((X - posB[0]) ** 2 + (Y - posB[1]) ** 2)
        mask_compC = np.array(mask_spaxels)
        if 1:
            fig, ax = plt.subplots(1, 4, sharey=True, sharex=True)
            ax[0].imshow(mask_compA, origin='lower')
            ax[0].set_title('Mask_compA')

            for i in range(mask_radA.shape[0]):
                for j in range(mask_radA.shape[1]):
                    if mask_compA[i,j]:
                        ax[0].text(j,i,int(mask_radA[i,j]))
            ax[1].imshow(mask_compB, origin='lower')
            ax[1].set_title('Mask_compB')
            for i in range(mask_radB.shape[0]):
                for j in range(mask_radB.shape[1]):
                    if mask_compB[i,j]:
                        ax[1].text(j,i,int(mask_radB[i,j]))
            ax[2].imshow(mask_compC, origin='lower')
            ax[2].set_title('Mask_compC')
            ax[3].imshow(np.nanmedian(cube.data, axis=0), origin='lower')
            for axs in ax[:]:
                axs.contour(X, Y, mask_spaxels.astype(float), levels=[0], colors='green', linewidths=1.5)
            plt.show()

        if 1:
            amp_comps, std_amp_comps = psf_fit(cube, wave=wavelength, mask_spaxels=mask_spaxels,
                                fit_comps=[specA, specB, specC], mask_spectrum=mask_spectrum,
                                                 mask_comps=[mask_compA,mask_compB,mask_compC],plot_pixel_fit =False)
            with open('./tmp_output/'+fname.split('.fits')[0]+'_psf_fit.pkl', 'wb') as f:
                pickle.dump((amp_comps, std_amp_comps), f)
            ########################
        else:
            with open('./tmp_output/'+fname.split('.fits')[0]+'_psf_fit.pkl', 'rb') as f:
                (amp_comps, std_amp_comps) = pickle.load(f)


        #plot results
        if 1:
            def plot_map(x, mode='linear'):
                if mode == 'linear':
                    return x
                elif mode == 'log':
                    return np.log10(np.abs(x))




            fig, ax = plt.subplots(1, 4, sharey=True, sharex=True)
            cmap = plt.cm.viridis
            cmap.set_bad('black')
            vmin,vmax=0,1
            for i in range(3):  #amp_comps, std_amp_comps
                im = ax[i].imshow(plot_map(amp_comps[i]), vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
                ax[i].set_title('Comp' + str(i))
            for axs in ax[:]:
                axs.contour(X, Y, mask_spaxels.astype(float), levels=[0], colors='green', linewidths=1.5)
            ax[0].contour(X, Y, mask_compA.astype(float), levels=[0], colors='red', linewidths=1.5)
            ax[1].contour(X, Y, mask_compB.astype(float), levels=[0], colors='red', linewidths=1.5)

            fig.colorbar(im)
            plt.show()


        #subtract components
        if 1:
            cube_data, cube_err = subtract_comp(cube=cube, wave=wavelength, mask_spaxels=mask_spaxels, fit_comps=[specA, specB, specC],
                          amp_comps=amp_comps, active_comps_list=[0,2],
                          mask_comps=[mask_compA, mask_compB, mask_compC], debug=True)

            new_name =  fname.split('_s3d')[0]+'_QB_s3d.fits'

            save_local_cube(cube_data, cube_err, orig_name=path_cube + fname, new_name=path_cube + new_name)

        if 1:

            cube_data, cube_err = subtract_comp(cube=cube, wave=wavelength, mask_spaxels=mask_spaxels, fit_comps=[specA, specB, specC],
                          amp_comps=amp_comps, active_comps_list=[1,2],
                          mask_comps=[mask_compA, mask_compB, mask_compC], debug=True)

            new_name = fname.split('_s3d')[0]+'_QA_s3d.fits'

            save_local_cube(cube_data, cube_err, orig_name=path_cube + fname, new_name=path_cube + new_name)

        if 1:
            cube_data, cube_err = subtract_comp(cube=cube, wave=wavelength, mask_spaxels=mask_spaxels, fit_comps=[specA, specB, specC],
                          amp_comps=amp_comps, active_comps_list=[0, 1],
                          mask_comps=[mask_compA, mask_compB, mask_compC], debug=True)

            new_name = fname.split('_s3d')[0]+'_GAL_s3d.fits'

            save_local_cube(cube_data, cube_err, orig_name=path_cube + fname, new_name=path_cube + new_name)

            #psf_fit(cube, wave=wavelength, mask_spaxels=mask_spaxels,
            #        fit_comps=[specA, specB, specC], mask_spectrum=mask_spectrum,
            #        mask_comps=[mask_compA, mask_compB, mask_compC])
            plt.show()