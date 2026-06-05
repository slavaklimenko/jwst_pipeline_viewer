#import webbpsf
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels import dqflags
import os
import astropy,scipy,pickle
from scipy.interpolate import interp1d
import matplotlib.patches as patches
from lmfit import Minimizer, Parameters
import emcee
from chainconsumer import ChainConsumer
from multiprocessing import Pool
import copy
from scipy.signal import fftconvolve

path_to_jwst_folder = '/home/slava/science/codes/python/jwst/'

if 0:
    def read_settings(init_file=path_to_jwst_folder+'/init.dat'):
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

    miri.monochromatic = lam
    #psf = miri.calc_psf(fov_arcsec=3)  # Adjust FOV to ~3 arcsec
    psf = miri.calc_psf(fov_arcsec=6)  # Adjust FOV to ~3 arcsec
    #psf = miri.calc_psf(fov_pixels=data_slice.shape[0])  # Adjust FOV to ~3 arcsec
    # --- Crop the PSF to 60x60 pixels ---
    psf_data = psf[0].data/np.nanmax(psf[0].data)
    center_y, center_x = psf_data.shape[0] // 2, psf_data.shape[1] // 2

    print('psf_data: ',np.sum(psf_data))

    plt.subplots()
    plt.imshow(np.log10(psf_data), origin='lower',vmin=-3,vmax=0)
    plt.show()

    half_size = 30  # Half of 60 pixels

    # Crop around the center of the PSF
    cropped_psf = psf_data[center_y - half_size:center_y + half_size,
                           center_x - half_size:center_x + half_size]

    # Normalize and assign to the cube
    psf_cube[i] = cropped_psf / np.max(cropped_psf)

def read_psf(channel='1A_ch1-short_test.fits',source='custom_psf'):
    if source == 'custom_psf':
        fname = path_to_jwst_folder + 'data_local/miri_psf/psf_v2.0/'  + channel
        with fits.open(fname) as hdul:
            # science cube
            data = hdul[0].data
            hdr = hdul[0].header

            # wavelength array
            wavelength = hdul['WAVELENGTH'].data
        return data,wavelength

def model_img(params , cube_shape, psf_cube, spectrum_model,
              debug=False, get_qso_pos=False):

    psf_center = (params['xc'].value, params['yc'].value)
    intensity = params['amp'].value*spectrum_model
    model_cube = np.zeros(shape=cube_shape)

    def add_qso(m=model_cube, center=psf_center, intensity=intensity, mode='smoothed'):
        m = np.array(m)
        if mode == 'single pixel':
            cen_int = [int(np.rint(c)) for c in center]
            m[:,cen_int[0], cen_int[1]] = intensity
        if mode == 'smoothed':
            cen_int = [int(np.rint(c)) for c in center]
            cen_delta = [center[i] - cen_int[i] for i in range(len(center))]
            #print('cen_int',cen_int,'; center',center,'; shape_m',m.shape)
            if (cen_int[0] - 1 >= 0 and cen_int[0] + 2 <= m.shape[1] and
                    cen_int[1] - 1 >= 0 and cen_int[1] + 2 <= m.shape[2]):
                short_img = m[0,cen_int[0] - 1:cen_int[0] + 2, cen_int[1] - 1:cen_int[1] + 2]
            else:
                cen_int = [int(np.rint(c)) for c in center]
                #print('Cutout exceeds image boundaries - single pix model')
                #print('cen_int', cen_int, '; center', center, '; shape_m', m.shape)
                cen_int = [int(np.floor(c)) for c in center]
                #print('floor_int', cen_int, '; center', center, '; shape_m', m.shape)
                m[:, cen_int[0], cen_int[1]] = intensity
                return m

            short_img[1, 1] = (1 - np.abs(cen_delta[0])) * (1 - np.abs(cen_delta[1]))
            short_img[1, 0] = (1 - np.abs(cen_delta[0])) * np.abs(cen_delta[1])
            short_img[1, 2] = (1 - np.abs(cen_delta[0])) * np.abs(cen_delta[1])
            short_img[0, 1] = (1 - np.abs(cen_delta[1])) * np.abs(cen_delta[0])
            short_img[2, 1] = (1 - np.abs(cen_delta[1])) * np.abs(cen_delta[0])
            short_img[2, 2] = (np.abs(cen_delta[1])) * (np.abs(cen_delta[0]))
            short_img[0, 2] = short_img[2, 2]
            short_img[2, 0] = short_img[2, 2]
            short_img[0, 0] = short_img[2, 2]
            if cen_delta[0] > 0:
                short_img[0, :] = 0
            else:
                short_img[2, :] = 0

            if cen_delta[1] > 0:
                short_img[:, 0] = 0
            else:
                short_img[:, 2] = 0

            m[:,cen_int[0] - 1:cen_int[0] + 2, cen_int[1] - 1:cen_int[1] + 2] += intensity[:, None, None] * short_img

        return m

    model_cube = add_qso(model_cube)
    model_convolved = np.zeros_like(model_cube)
    #colnvolve cube with the psf model
    for i in range(cube_shape[0]):
        model_convolved[i] = fftconvolve(
            model_cube[i],
            psf_cube[i],
            mode='same'
        )



    if debug:
        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)


        ax[0].imshow(np.nanmean(psf_cube, axis=0), origin='lower')
        ax[0].set_title("PSF (mean over wavelength)")

        ax[1].imshow(np.nanmean(model_cube, axis=0), origin='lower')
        ax[1].set_title("Model (unconvolved)")

        ax[2].imshow(np.nanmean(model_convolved, axis=0), origin='lower')
        ax[2].set_title("Model (convolved)")

        plt.tight_layout()
        plt.show()

    if get_qso_pos:
        return model_cube
    else:
        return model_convolved

def plot_comparison(data, params,psf_cube,mask_fitting, spectrum_model,params_init,filename=None):
    cmap = plt.cm.coolwarm
    cmap.set_bad('black')

    data = np.array(data)
    print('plot, params:',[( params[p].name, params[p].value) for p in params])
    print('init, params:',[( params_init[p].name, params_init[p].value) for p in params_init])
    model =  model_img(params, data.shape, psf_cube=psf_cube, spectrum_model=spectrum_model)
    image = np.nansum(data, axis=0)
    image_model = np.nansum(model, axis=0)

    fmax = np.nanmax(image)
    vmin,vmax = np.log10(fmax) - 3.5,np.log10(fmax)

    fig, ax = plt.subplots(1, 4, sharey=True, sharex=True,figsize=(20,4))
    im0 = ax[0].imshow(np.log10(np.abs(image)), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0].set_title('Data')

    ax[1].imshow(np.log10(np.abs(image_model)), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax[1].set_title('Model')

    im2 = ax[2].imshow(image - image_model, origin='lower', vmin=-0.1 * fmax, vmax=0.1 * fmax, cmap=cmap)
    ax[2].set_title('Data-Model\n linear')

    ax[3].imshow(np.log10(np.abs(image - image_model)), origin='lower', vmin=vmin,  vmax=vmax, cmap=cmap)
    ax[3].set_title('Data-Model\n log')

    for axs in ax[:]:
        axs.plot(params['yc'].value,params['xc'].value,'o',color='red',label='proposed')
        axs.plot(params_init['yc'].value,params_init['xc'].value,'X',color='black',label='init')
        axs.legend()



    if 1:
        x = np.arange(mask_fitting.shape[1])
        y = np.arange(mask_fitting.shape[0])
        X, Y = np.meshgrid(x, y)
        ax[0].contour(X, Y, mask_fitting.astype(float), levels=[0], colors='green', linewidths=1.5)
        ax[2].contour(X, Y, mask_fitting.astype(float), levels=[0], colors='green', linewidths=1.5)
        ax[3].contour(X, Y, mask_fitting.astype(float), levels=[0], colors='green', linewidths=1.5)

    if 1:
        fig.colorbar(im0, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)


    if filename is None:
        filename = 'psf_subtraction.pdf'
    fig.savefig(path_to_jwst_folder + 'output/scripts/' + filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    #plt.show()


def lnprior(parameters):

    for p in parameters.values():
        if not (p.min <= p.value <= p.max):
            return -np.inf

    return 0



def log_probability(theta, data_tmp, mask_tmp, parameters, psf_cube, spectrum_model):

    params1 = copy.deepcopy(parameters)
    vary_params1 = [p for p in params1.values() if p.vary]

    for val, p in zip(theta, vary_params1):
        p.value = val
        if not (p.min <=val <= p.max):
            return -np.inf

    #if lnprior(params1) == -np.inf:
    #    return -np.inf

    m_i = model_img(params1, data_tmp.shape,
                     psf_cube=psf_cube,
                     spectrum_model=spectrum_model)

    residuals = data_tmp - m_i
    residuals[:, ~mask_tmp] = np.nan

    snr = 30
    err = np.ones_like(data_tmp)
    median_err = np.median(data_tmp[:, mask_tmp], axis=1) / snr
    err[:, mask_tmp] = median_err[:, None]
    weights = np.ones_like(residuals)
    weights[residuals<-10*err] *= 10
    #print(np.sum(residuals<-10*err))
    chiq = np.nansum(np.power(residuals / err, 2) * weights)

    return -0.5 * chiq

'''
def log_probability(theta, data_tmp,mask_tmp,parameters,psf_cube,spectrum_model):
    vary_params = [p for p in parameters.values() if p.vary]
    for val, p in zip(theta, vary_params):
        p.value = val

    chi_prior = lnprior(parameters= parameters)

    if chi_prior == 0:

        m_i = model_img(parameters, data_tmp.shape, psf_cube=psf_cube, spectrum_model=spectrum_model,
                        debug=False, get_qso_pos=False)

        residuals = data_tmp - m_i
        residuals[:, ~mask_tmp] = np.nan

        snr = 20
        err = np.abs(data_tmp) / snr
        weights = np.ones_like(residuals)
        # weights[residuals<-err] *= 10
        chiq = np.nansum(np.power(residuals / err * weights, 2))
        return -0.5 * chiq
    else:
        return chi_prior
'''

def subtract_psf(data,psf_cube,spectrum_model,params,mask_fitting,
                 debug = False,save_results=True, nwalkers=100, nsteps = 100,arcsec_pix_scale=1):

    #copy input
    data = np.array(data)
    y, x = np.where(mask_fitting)
    # cut to match fitting area
    data_new = data[:,y.min():y.max() + 1,x.min():x.max() + 1]
    mask_new = mask_fitting[y.min():y.max() + 1,x.min():x.max() +1]
    y0, x0 = y.min(), x.min()
    #(xc_new, yc_new) = (xc - y.min(), yc - x.min())
    params_new = copy.deepcopy(params)
    params_init = copy.deepcopy(params)
    params_new['xc'].value = params['xc'].value - y0
    params_new['yc'].value = params['yc'].value - x0


    #find init values for qso center
    data_tmp = np.array(data_new)
    if 0:
        plt.subplots()
        plt.imshow(np.nanmean(data_tmp,axis=0))
        plt.plot(params_new['xc'].value,params_new['xc'].value,'o',color='red')
        plt.title('fitting area')
        plt.show()
    print('init parameters values:', [(params_new[p].name, params_new[p].value) for p in params_new])
    #use leastsq for init guess
    if 0:
        def fcn2min(params):
            m_i = model_img(params=params, cube_shape=data_tmp.shape, psf_cube=psf_cube, spectrum_model=spectrum_model,
                              debug=False, get_qso_pos=False)
            res = data_tmp[:,mask_new] - m_i[:,mask_new]
            return res

        minner = Minimizer(fcn2min, params_new)
        result = minner.minimize(method='leastsq')
        for par in params_new:
            params_new[par].value = result.params[par].value
        print('leastsq res:',[(par,params_new[par].value) for par in params_new])

    if 1:
        # set uncertanties for mcmc
        params_new['xc'].stderr = 1
        params_new['yc'].stderr = 1
        params_new['xc'].max = params_new['xc'].value+0.1*arcsec_pix_scale    #mask_new.shape[0]-1
        params_new['yc'].max = params_new['yc'].value+0.1*arcsec_pix_scale    #mask_new.shape[1]-1
        params_new['xc'].min = params_new['xc'].value - 0.1*arcsec_pix_scale  # mask_new.shape[0]-1
        params_new['yc'].min = params_new['yc'].value - 0.1*arcsec_pix_scale  # mask_new.shape[1]-1
        params_new['amp'].stderr = 0.5
        params_new['psf_rot'].stderr = 20

    print('prepare emcee')
    par_names = [p.name for p in params_new.values() if p.vary]
    ndim = len(par_names)


    #init mcmc
    init = [params_new[p].value for p in par_names]
    init_range = [params_new[p].stderr  for p in par_names]
    p0 = []
    for i in range(nwalkers):
        prob = -np.inf
        while prob == -np.inf:
            rndm = np.random.randn(ndim)
            wal_pos = init + init_range * rndm
            prob = log_probability(theta=wal_pos,data_tmp=data_tmp,mask_tmp=mask_new,parameters=params_new,
                                   psf_cube=psf_cube,spectrum_model=spectrum_model)
        p0.append(wal_pos)
    p0 = np.array(p0)

    print('start emcee')
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    args=[data_tmp,mask_new,params_new,psf_cube,spectrum_model], pool=pool)
        sampler.run_mcmc(p0, nsteps, progress=True)

    samples = sampler.chain[:, int(0.75*nsteps):, :].reshape((-1, ndim))
    #samples = sampler.chain[:, :, :].reshape((-1, ndim))

    if debug:
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples_image = sampler.get_chain()
        labels = par_names
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples_image[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples_image))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.axhline(params_new[labels[i]].value,color='red',ls='--')
        axes[-1].set_xlabel("step number")
        plt.show()

    c = ChainConsumer()
    names, truth = par_names, init
    c.add_chain(samples, walkers=nwalkers, parameters=names)

    if debug:
        c.configure(smooth=True,cloud=True,sigmas=[0, 1, 2, 3])
        c.configure_truth(ls='--', lw=1., c='lightblue')
        c.plotter.plot(figsize=(30, 30),
                       display=True,
                       truth=truth)

    res_mcmc = c.analysis.get_summary(parameters=names)
    print('res_mcmc:', res_mcmc)

    for name in par_names:
        params_new[name].value = res_mcmc[name][1]

    #plot_comparison(data_new, params_new, psf_cube, mask_new, spectrum_model)

    #convert to original scale
    params['xc'].value = params_new['xc'].value + y0
    params['yc'].value= params_new['yc'].value + x0
    params['amp'].value = params_new['amp'].value
    params['psf_rot'].value = params_new['psf_rot'].value

    plot_comparison(data, params, psf_cube, mask_fitting, spectrum_model, params_init)

    if save_results:
        model = model_img(params, data.shape, psf_cube=psf_cube, spectrum_model=spectrum_model)
        return copy.deepcopy(model), copy.deepcopy(params)

from JWST_cube_analyser import miri_psf_arcsec
from scipy.signal import convolve2d

if __name__ == '__main__':

    #subtract a single image
    if 0:
        # upload a source
        if 1:
            fname = (path_to_jwst_folder + '/output/detector3/'
                     + 'HD159222_ATCN6_N6/HD-159222_ATCN6_N6_2C_ch2-long__CORR_s3d.fits')
            #         + 'HD163466_ATCN6N6/HD-163466_ATCN6_N6_2C_ch2-long__CORR_s3d.fits')
            psf_center = (43,56)
            ch_name = '2C_ch2-long'

        cube = datamodels.open(fname)
        with fits.open(fname) as hdu:
            hdr = hdu['SCI'].header
        wavelength = (np.arange(hdr['NAXIS3']) + hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']
        cube_shape = cube.data.shape  # Data cube: 5 slices, 60x60 pixels each
        image = np.nanmedian(cube.data, axis=0)  # [slice_numer]
        data = np.array(cube.data)
        from astropy.wcs import WCS

        wcs = WCS(hdr)

        cenA = psf_center

        # set 1 sigma aperture
        miri_psf_fwhm = miri_psf_arcsec(np.nanmean(wavelength))  # in arcsec
        miri_psf_sigma = miri_psf_fwhm / 2.355  # in arcsec
        cdelt1 = wcs.wcs.cdelt[0]
        pix_size = cdelt1 * 3600  # in arcsec
        miri_psf_sigma_pix = miri_psf_sigma / (pix_size)
        print('1sigma rad =', miri_psf_sigma_pix, 'pix')

        # pixel grids
        yy, xx = np.indices(image.shape)
        # distance from center
        rr = np.hypot(xx - psf_center[1], yy - psf_center[0])


        # circular mask within 1 sigma
        mask = rr <= miri_psf_sigma_pix
        # integrated fliux within 1sigma aperture for A
        norm_flux_1sigma = np.nansum(data[:, mask], axis=1)

        #define PSF
        (psf, psf_w) = read_psf(channel=ch_name+'_based_HD-163466.fits')
        params = Parameters()
        names = ['xc', 'yc', 'amp', 'psf_rot']
        for name, value in zip(names, [psf_center[0], psf_center[1], 1.0, 0.0]):
            params.add(name, value=value, min=0, max=np.inf)
        params['amp'].max = 2
        params['psf_rot'].min = -90
        params['psf_rot'].max = 90
        params['psf_rot'].vary = False





        model = np.zeros_like(data)
        mask = rr <= miri_psf_sigma_pix
        fitting_radius = 3 * miri_psf_sigma_pix  # pix
        mask_fitting = rr < fitting_radius
        flux_1sigma = np.nansum((data)[:, mask], axis=1)
        save_model = False

        if 1:
            plt.subplots()
            plt.imshow(np.log10(np.abs(image)), origin='lower')
            plt.plot(psf_center[1], psf_center[0], 'o', color='red')
            cen_max = np.argwhere(image == np.nanmax(image))[0]
            print('cen_max', cen_max)
            plt.plot(cen_max[1], cen_max[0], 'x', color='black')

            x = np.arange(mask.shape[1])
            y = np.arange(mask.shape[0])
            X, Y = np.meshgrid(x, y)
            plt.contour(X, Y, mask.astype(float), levels=[0], colors='green', linewidths=1.5)
            plt.contour(X, Y, mask_fitting.astype(float), levels=[0], colors='black', linewidths=1.5)

            plt.subplots()
            plt.plot(wavelength, flux_1sigma, color='red')

            plt.show()


        if 1:
            print('Subtract PSF')
            params_ = copy.deepcopy(params)
            data_fit = data
            model, params = subtract_psf(data=data_fit, psf_cube=psf, spectrum_model=flux_1sigma, params=params,
                                            mask_fitting=mask_fitting,debug=True)

            # save model
            if save_model:
                params.dump(open('params_single.json', 'w'))
                cube.data = model
                # (optional but recommended) update history
                cube.add_history_entry("Replaced data with PSF-convolved model")
                # save new file
                cube.save(fname.split('s3d.fits')[0] + '_model_s3d.fits')
                # save psf subtracted cube
                cube.data = data - model
                # (optional but recommended) update history
                cube.add_history_entry("Replaced data with PSF-subtracted data")
                # cube.history.append("Replaced data with PSF-subtracted data")
                # save new file
                cube.save(fname.split('s3d.fits')[0] + 'subtracted_s3d.fits')



    #subtract two images iteratively
    if 1:

        ch_list = ['1A_ch1-short', '1B_ch1-medium', '1C_ch1-long',
                   '2A_ch2-short', '2B_ch2-medium', '2C_ch2-long',
                   '3A_ch3-short', '3B_ch3-medium', '3C_ch3-long',
                   '4A_ch4-short', '4B_ch4-medium', '4C_ch4-long']
        ch_list = ['4A_ch4-short', '4B_ch4-medium', '4C_ch4-long']
        for ch in ch_list:
            # read source
            if 1:
                fname = path_to_jwst_folder + 'output/detector3/' + 'B0218-ATCN6_N6/TXS0218+357ATCN6_N6_'+ch+'__CORR_s3d.fits'
            else:
                fname = path_to_jwst_folder + '/output/detector3/' + 'J0134_ATCN6_N6/J0134-0931_ATCN6_N6_' + ch + '__CORR_s3d.fits'


            cube = datamodels.open(fname)
            with fits.open(fname) as hdu:
                hdr = hdu['SCI'].header
            wavelength = (np.arange(hdr['NAXIS3']) + hdr['CRPIX3'] - 1) * hdr['CDELT3'] + hdr['CRVAL3']
            cube_shape = cube.data.shape  # Data cube: 5 slices, 60x60 pixels each
            image = np.nanmedian(cube.data, axis=0)  # [slice_numer]
            data = np.array(cube.data)

            if 1:
                l, raq, deq = np.nanmean(wavelength), 35.272790, 35.937148  # 35.272754, 35.937156 #B0218
                del_ra,del_dec =  0.307, 0.126
                asec = 1 / 3600.
                psf_center_A = cube.meta.wcs.world_to_pixel_values(raq, deq, l)
                psf_center_B = cube.meta.wcs.world_to_pixel_values(raq + del_ra * asec, deq + del_dec * asec, l)
            if 0:
                l, raq, deq = np.nanmean(wavelength), 23.648599, -9.517474  # J0134
                del_ra, del_dec = 0.539, -0.415
                asec = 1 / 3600.
                psf_center_A = cube.meta.wcs.world_to_pixel_values(raq, deq, l)
                psf_center_B = cube.meta.wcs.world_to_pixel_values(raq + del_ra * asec, deq + del_dec * asec, l)
                del_ra, del_dec = 0.258, 0.205
                psf_center_C = cube.meta.wcs.world_to_pixel_values(raq + del_ra * asec, deq + del_dec * asec, l)
                del_ra, del_dec = -0.082, -0.156
                psf_center_D = cube.meta.wcs.world_to_pixel_values(raq + del_ra * asec, deq + del_dec * asec, l)

            from astropy.wcs import WCS
            wcs = WCS(hdr)

            # set 1 sigma aperture
            miri_psf_fwhm = miri_psf_arcsec(np.nanmean(wavelength))  # in arcsec
            miri_psf_sigma = miri_psf_fwhm / 2.355  # in arcsec
            cdelt1 = wcs.wcs.cdelt[0]
            pix_size = cdelt1 * 3600  # in arcsec
            miri_psf_sigma_pix = miri_psf_sigma / (pix_size)
            print('1sigma rad =', miri_psf_sigma_pix, 'pix')

            # pixel grids
            yy, xx = np.indices(image.shape)
            # distance from center
            cenA = (psf_center_A[1],psf_center_A[0])
            cenB = (psf_center_B[1],psf_center_B[0])
            rr = np.hypot(xx - cenA[1], yy - cenA[0])
            rr_B = np.hypot(xx - cenB[1], yy - cenB[0])

            #define psf
            #(psf,psf_w) = read_psf(channel=ch+'_test.fits')
            (psf, psf_w) = read_psf(channel=ch + '_based_HD-163466.fits')

            #define parameters
            params = Parameters()
            names = ['xc', 'yc', 'amp','psf_rot']
            for name, value in zip(names, [cenA[0], cenA[1], 1.0, 0.0]):
                params.add(name, value=value, min=0, max=np.inf)
            params['amp'].max = 10
            params['psf_rot'].min = -90
            params['psf_rot'].max = 90


            #prepare data
            if 1:
                params_A = copy.deepcopy(params)
                params_B = copy.deepcopy(params)
                params_A['xc'].value = cenA[0]
                params_A['yc'].value = cenA[1]
                params_B['xc'].value = cenB[0]
                params_B['yc'].value = cenB[1]
                params['psf_rot'].vary = False


                modelA = np.zeros_like(data)
                modelB = np.zeros_like(data)
                maskA = rr <= miri_psf_sigma_pix
                maskB = rr_B <= miri_psf_sigma_pix
                fitting_radius = 4*miri_psf_sigma_pix #pix
                mask_radius = 2 * miri_psf_sigma_pix
                flux_1sigma_A = np.nansum((data - modelB)[:, maskA], axis=1)
                flux_1sigma_B = np.nansum((data - modelA)[:, maskB], axis=1)
                save_model = True

            if 0:
                plt.subplots()
                plt.imshow(np.log10(np.abs(image)),origin='lower')
                plt.plot(cenA[1],cenA[0],'o',color='red')
                plt.plot(cenB[1],cenB[0],'o',color='blue')
                cen_max = np.argwhere(image == np.nanmax(image))[0]
                plt.plot(cen_max[1], cen_max[0], 'x', color='black')

                x = np.arange(maskA.shape[1])
                y = np.arange(maskA.shape[0])
                X, Y = np.meshgrid(x, y)
                plt.contour(X, Y, maskA.astype(float), levels=[0], colors='green', linewidths=1.5)
                plt.contour(X, Y, (rr< fitting_radius).astype(float), levels=[0], colors='black', linewidths=1.5)
                plt.contour(X, Y, maskB.astype(float), levels=[0], colors='green', linewidths=1.5)
                plt.contour(X, Y, (rr_B < fitting_radius).astype(float), levels=[0], colors='black', linewidths=1.5)

                plt.subplots()
                plt.plot(wavelength,flux_1sigma_A ,color='red')
                plt.plot(wavelength,flux_1sigma_B  ,color='blue')

                plt.show()
            # run calculations
            if 1:
                for it in range(5):
                    print('Iter', it, 'Step 1. subtract B and get model for spectrum A')
                    flux_1sigma_A = np.nansum((data-modelB)[:, maskA], axis=1)
                    print('Iter', it, 'Step 2. model A and subtract "model A"')
                    params['xc'].value= params_A['xc'].value
                    params['yc'].value =params_A['yc'].value
                    params['amp'].value = params_A['amp'].value
                    params['psf_rot'].value = params_B['psf_rot'].value
                    #set 2d mask for fitting region
                    mask_fitting = rr< fitting_radius
                    if 'cenB' in locals():
                        mask_fitting[rr_B<mask_radius] = False
                    if 'psf_center_C' in locals() and ch in ['1A_ch1-short', '1B_ch1-medium', '1C_ch1-long']:
                        # cenB = (psf_center_B[1],psf_center_B[0])
                        rr_C = np.hypot(xx - psf_center_C[0], yy - psf_center_C[1])
                        mask_fitting[rr_C<2*miri_psf_sigma_pix] = False
                        rr_D = np.hypot(xx - psf_center_D[0], yy - psf_center_D[1])
                        mask_fitting[rr_D<2*miri_psf_sigma_pix] = False

                    data_fit = data if it == 0 else data - modelB
                    modelA, params_A = subtract_psf(data = data_fit, psf_cube=psf, spectrum_model=flux_1sigma_A, params=params,
                             mask_fitting=mask_fitting,arcsec_pix_scale=1/pix_size)




                    #save model
                    if save_model:
                        plot_comparison(data_fit, params_A, psf, mask_fitting, flux_1sigma_A, params,
                                        filename=ch+'_subtr_A.pdf')
                        params_A.dump(open('paramsA.json', 'w'))
                        cube.data = modelA
                        # (optional but recommended) update history
                        cube.add_history_entry("Replaced data with PSF-convolved model")
                        # save new file
                        cube.save(fname.split('s3d.fits')[0] + 'modelA_based_HD-163466_s3d.fits')
                        # save psf subtracted cube
                        cube.data = data - modelA
                        # (optional but recommended) update history
                        cube.add_history_entry("Replaced data with PSF-subtracted data")
                        #cube.history.append("Replaced data with PSF-subtracted data")
                        # save new file
                        cube.save(fname.split('s3d.fits')[0]+  'subtr_A_based_HD-163466_s3d.fits')

                    print('Iter', it, 'Step 3. get model for spectrum B.')
                    flux_1sigma_B = np.nansum((data - modelA)[:, maskB], axis=1)
                    print('Iter', it, 'Step 4. model B and subtract "model B"')
                    mask_fitting = rr_B < fitting_radius
                    mask_fitting[rr < mask_radius] = False
                    params['xc'].value= params_B['xc'].value
                    params['yc'].value =params_B['yc'].value
                    params['amp'].value = params_B['amp'].value
                    params['psf_rot'].value =params_A['psf_rot'].value
                    data_fit = data if it == 0 else data - modelA
                    modelB, params_B = subtract_psf(data=data_fit, psf_cube=psf, spectrum_model=flux_1sigma_B, params=params,
                                                  mask_fitting=mask_fitting)
                    # save model
                    if save_model:
                        plot_comparison(data_fit, params_B, psf, mask_fitting, flux_1sigma_B, params,
                                        filename=ch+'_subtr_B.pdf')
                        params_B.dump(open('paramsB.json', 'w'))
                        cube.data = modelB
                        # (optional but recommended) update history
                        cube.add_history_entry("Replaced data with PSF-subtracted data")
                        # save new file
                        cube.save(fname.split('s3d.fits')[0]+ 'modelB_based_HD-163466_s3d.fits')
                        cube.data = data - modelB
                        # (optional but recommended) update history
                        cube.add_history_entry("Replaced data with PSF-subtracted data")
                        # save new file
                        cube.save(fname.split('s3d.fits')[0]+ 'subtr_B_based_HD-163466_s3d.fits')



