import webbpsf
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
#from chainconsumer import ChainConsumer
from multiprocessing import Pool


path_to_jwst_folder = '/home/slava/science/codes/python/jwst/'


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



if 0:
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

def rebin_2d(array, factor):
    array = np.array(array)
    # Ensure the array shape is divisible by the rebinning factor
    shape = (array.shape[0] // factor, factor, array.shape[1] // factor, factor)
    # Reshape and compute the mean along the rebinned axes
    return array.reshape(shape).mean(axis=(1, 3))*factor**2

def upsample_2d(array, factor):
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)

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


def model_img(params , img_shape, psf_image, overdist=False, debug=False, get_qso_pos=False):

    psf_center = (params['xc'].value, params['yc'].value)
    intensity = params['amp'].value
    model_img = np.zeros(shape=img_shape)

    def add_qso(img=model_img, center=psf_center, intensity=intensity, mode='smoothed'):
        img = np.array(img)
        if mode == 'single pixel':
            cen_int = [int(np.rint(c)) for c in center]
            img[cen_int[0], cen_int[1]] = intensity
        if mode == 'smoothed':
            cen_int = [int(np.rint(c)) for c in center]
            cen_delta = [center[i] - cen_int[i] for i in range(len(center))]
            # print('cen_int',cen_int)
            short_img = np.array(img[cen_int[0] - 1:cen_int[0] + 2, cen_int[1] - 1:cen_int[1] + 2])

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

            img[cen_int[0] - 1:cen_int[0] + 2, cen_int[1] - 1:cen_int[1] + 2] += short_img * intensity

        return img

    if overdist==False:
        model_img = add_qso(model_img)
        model_convolved = scipy.signal.convolve2d(model_img, psf_image, mode='same', boundary='fill',fillvalue=0)

        if debug:
            fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)
            ax[0].imshow(model_img, origin='lower')
            ax[0].set_title('Model')
            ax[1].imshow(np.log10(psf_image), origin='lower')
            ax[1].set_title('PSF(log scale)')
            ax[2].imshow(np.log10(model_convolved), origin='lower')
            ax[2].set_title('Convolved with PSF')

        if get_qso_pos:
            return model_img
        else:
            return model_convolved

    elif overdist==True:
        zoom_factor = 4
        filter_kernel = np.array(psf_image)

        cen_pix = int(np.rint(filter_kernel.shape[0]/2))
        filter_kernel = filter_kernel[cen_pix-50:cen_pix+50,cen_pix-50:cen_pix+50,]

        model_img_overdist =  upsample_2d(model_img, zoom_factor)
        psf_center_overdist = [p*zoom_factor for p in psf_center]

        model_img = add_qso(img=model_img_overdist,center=psf_center_overdist)

        model_convolved = scipy.signal.convolve2d(model_img, filter_kernel, mode='same', boundary='fill',
                                                      fillvalue=0)
        #rebin to normal resolution
        model_rebinned = rebin_2d(array=model_convolved, factor=4)

        if debug:
            fig, ax = plt.subplots(1, 6, sharey=True, sharex=True)
            ax[0].imshow(model_img, origin='lower')
            ax[0].set_title('Model')
            ax[1].imshow(np.log10(psf_image), origin='lower')
            ax[1].set_title('PSF(log scale)')
            ax[2].imshow(np.log10(model_convolved), origin='lower')
            ax[2].set_title('Convolved with PSF')

            ax[3].imshow(rebin_2d(array=model_img, factor=4), origin='lower')
            ax[3].set_title('Rebinned init')
            ax[4].imshow(np.log10(model_rebinned), origin='lower')
            ax[4].set_title('Rebinned model')
            tmp = scipy.signal.convolve2d(rebin_2d(array=model_img, factor=4), psf_image, mode='same', boundary='fill',
                                                      fillvalue=0)
            ax[5].imshow(np.log10(tmp),origin='lower')

            print(np.nansum(tmp),np.nansum(model_rebinned))
            #plt.show()

        return model_rebinned




def plot_comparsion(image, params,psf_image,mask_fitting, overdist=False):
    cmap = plt.cm.coolwarm
    cmap.set_bad('black')

    image = np.array(image)
    print('plot comarison, params:',params)
    model = model_img(params=params, img_shape=image.shape, psf_image=psf_image,
                    overdist=overdist, debug=False, get_qso_pos=False)

    fmax = np.nanmax(image)
    vmin,vmax = np.log10(fmax) - 3.5,np.log10(fmax)

    fig, ax = plt.subplots(1, 5, sharey=True, sharex=True)
    im0 = ax[0].imshow(np.log10(np.abs(image)), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0].set_title('Data')

    ax[1].imshow(np.log10(np.abs(model)), origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    ax[1].set_title('Model')

    im2 = ax[2].imshow(image - model, origin='lower', vmin=-0.1 * fmax, vmax=0.1 * fmax, cmap=cmap)
    ax[2].set_title('Data-Model\n linear')

    ax[3].imshow(np.log10(np.abs(image - model)), origin='lower', vmin=vmin,  vmax=vmax, cmap=cmap)
    ax[3].set_title('Data-Model\n log')

    source_model = model_img(params=params, img_shape=image.shape, psf_image=psf_image,
                    overdist=overdist, debug=False, get_qso_pos=True)
    ax[4].imshow(source_model, origin='lower')
    ax[4].set_title('S pos:' + str(params['xc'].value) + str(params['yc'].value))

    if 1:
        x = np.arange(mask_fitting.shape[1])
        y = np.arange(mask_fitting.shape[0])
        X, Y = np.meshgrid(x, y)
        ax[0].contour(X, Y, mask_fitting.astype(float), levels=[0], colors='green', linewidths=1.5)
        ax[2].contour(X, Y, mask_fitting.astype(float), levels=[0], colors='green', linewidths=1.5)

    if 1:
        fig.colorbar(im0, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=ax[2], orientation='vertical', fraction=0.046, pad=0.04)


    # plot profiles
    if 1:
        fig, ax = plt.subplots(1, 3, sharey=True, sharex=True)

        cen_int = [int(np.rint(c)) - 1 for c in [params['xc'].value,params['yc'].value]]
        y1 = cen_int[0]
        y2 = cen_int[0] + 2

        ax[0].plot(image[y1, :], label=str(y1))
        ax[0].plot(image[y2, :], label=str(y2))
        ax[0].set_title('Image')

        ax[1].plot(model[y1, :], label=str(y1))
        ax[1].plot(model[y2, :], label=str(y2))

        ax[1].plot(image[y1, :], ls='--')
        ax[1].plot(image[y2, :], ls='--')
        ax[1].set_title('profiles, model')

        ax[2].plot((image - model)[y1, :], label=str(y1))
        ax[2].plot((image - model)[y2, :], label=str(y2))
        ax[2].set_title('profiles, res:')

        for axs in ax[:]:
            axs.legend()
        plt.show()




def subtract_psf(image, mask_fitting, channel='1C', psf_type='custom_psf',overdist=False,
                 algorithm = 'mcmc',subtract=False):

    #settings
    subtract_background = False
    rad_background = 6  # pix - circle radius around the qso to calculate the average background
    debug = True

    #copy input
    image = np.array(image)

    #read psf for certain channel
    psf_image = read_psf(channel=channel, source=psf_type, overdist=overdist)

    #find init values for qso center
    init_pos = np.where(image==np.nanmax(image[mask_fitting]))
    (xc_init, yc_init) =init_pos[0][0],init_pos[1][0]

    # define parameters
    amp_init = np.nansum(image[mask_fitting])
    params = Parameters()
    names = ['xc', 'yc', 'amp']
    for name, value in zip(names, [xc_init, yc_init, amp_init]):
        params.add(name, value=value, min=0, max=np.inf)
    params['xc'].min=xc_init-5
    params['xc'].max = xc_init+5
    params['yc'].min = yc_init - 5
    params['yc'].max = yc_init + 5
    print('init values',params)


    # define radial coordinate system
    image_shape= image.shape
    image_ind = np.indices((image.shape[0], image.shape[1]))
    radial_coord = np.sqrt((image_ind[0] - params['xc'].value) ** 2 + (
            image_ind[1] - params['yc'].value) ** 2)

    # subtract background
    if subtract_background:
        mask_background = (radial_coord > rad_background)  # in pixels
        print('subtract median background:', np.nanmedian(image[mask_background]))
        image -= np.nanmedian(image[mask_background])

    if algorithm == 'fit_amplitude':
        debug = True
        kind = 'chi2'
        image_tmp = np.array(image)

        amp_0 = np.sum(image_tmp[mask_fitting])
        amp_range = np.linspace(0, 3 * amp_0, 10)

        stat_values = np.zeros(len(amp_range))
        for i, amp_i in enumerate(amp_range):
            print('iteration:', i)
            params['amp'].value = amp_i
            m_i = model_img(params=params , img_shape=image_shape, psf_image=psf_image,
                            overdist=overdist, debug=False, get_qso_pos=False)

            if kind == 'difference':
                d = (image_tmp - m_i)
                stat_values[i] = np.nansum(d[mask_fitting])
            elif kind == 'chi2':
                d = (image_tmp - m_i)[mask_fitting]
                w = np.ones_like(d)
                # if we need to reduce over-subtraction
                if 0:
                    w[d < 0] = 10
                stat_values[i] = np.nansum(np.power(d, 2) * w)

            print('chi',i, stat_values[i])

        if kind == 'difference':
            optimal_value = interp1d(stat_values, amp_range)
            print('optimal_value', optimal_value(0))
        elif kind == 'chi2':
            optimal_value = interp1d(amp_range, stat_values, kind='quadratic')
            x0 = np.linspace(amp_range[0], amp_range[-1], 1000)
            y0 = optimal_value(x0) - np.nanmin(optimal_value(x0))
            optimal_value = interp1d(y0, x0)
            print('optimal_value', optimal_value(0))

        if debug:
            plt.subplots()
            plt.plot(amp_range, stat_values)
            plt.plot(x0, y0)

        params['amp'].value = optimal_value(0)

    elif  algorithm == 'leastsq':
        image_tmp = np.array(image)

        def fcn2min(params):
            print(params)
            m_i = model_img(params=params , img_shape=image_shape, psf_image=psf_image,
                            overdist=overdist, debug=False, get_qso_pos=False)
            res = (image_tmp[mask_fitting] - m_i[mask_fitting])
            return res

        minner = Minimizer(fcn2min, params)
        result = minner.minimize(method='leastsq')

        for par in params:
            params[par].value = result.params[par].value

        del image_tmp

    elif algorithm == 'mcmc':

        image_tmp = np.array(image)

        def lnprior(params):
            xc, yc, amp = params
            if mask_fitting[int(xc), int(yc)] == True:
                return 0
            else:
                return -np.inf

        def log_probability(params, data):
            # over-subtraction
            reduce_over_subtraction = True
            chi_prior = lnprior(params)
            if ~np.isinf(chi_prior):
                pars_tmp = Parameters()
                names = ['xc', 'yc', 'amp']
                for name, value in zip(names, params):
                    pars_tmp.add(name, value=value, min=0, max=np.inf)
                m_i =model_img(params=pars_tmp , img_shape=image_shape, psf_image=psf_image,
                            overdist=overdist, debug=False, get_qso_pos=False)

                residuals = (data[mask_fitting] - m_i[mask_fitting])
                err = np.ones_like(residuals)
                w = np.ones_like(residuals)
                # if we need to reduce over-subtraction
                if reduce_over_subtraction:
                    w[residuals < 0] = 10
                return -0.5 * np.nansum(np.power(residuals / err, 2) * w)
            else:
                return chi_prior

        def fcn2min(params):
            m_i = model_img(params=params, img_shape=image_shape, psf_image=psf_image,
                            overdist=overdist, debug=False, get_qso_pos=False)
            res = (image_tmp[mask_fitting] - m_i[mask_fitting])
            return res

        minner = Minimizer(fcn2min, params)
        result = minner.minimize(method='leastsq')
        for par in params:
            params[par].value = result.params[par].value

        if 1:
            # set uncertanties for mcmc
            result.params['xc'].stderr = 1
            result.params['yc'].stderr = 1
            result.params['amp'].stderr = 500.0

        print('run emcee')
        ndim, nwalkers, nsteps = 3, 100, 200
        init = [result.params[par].value for par in  params]
        init_range = [result.params[par].stderr for par in  params]
        p0 = []
        for i in range(nwalkers):
            prob = -np.inf
            while prob == -np.inf:
                rndm = np.random.randn(ndim)
                wal_pos = init + init_range * rndm
                prob = log_probability(params=wal_pos,data=image_tmp)
            p0.append(wal_pos)
        p0 = np.array(p0)
        #p0 = np.asarray([result.params[par].value + np.random.randn(nwalkers) * result.params[par].stderr for par in
        #                 params]).transpose()
        if 1:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[image_tmp])
            sampler.run_mcmc(p0, nsteps, progress=True)
        else:
            #multiprocessing
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn=log_probability, args=[data_img], pool=pool)
                sampler.run_mcmc(p0, nsteps, progress=True)

        samples = sampler.chain[:, int(nsteps / 2):, :].reshape((-1, ndim))

        if debug:
            fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
            samples_image = sampler.get_chain()
            labels = ['xc', 'yc', 'amp']
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples_image[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples_image))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel("step number")
            plt.show()

        c = ChainConsumer()
        names, truth = ['xc', 'yc', 'amp'], [params['xc'].value, params['yc'].value,
                                             params['amp'].value]
        print('truth', truth)
        c.add_chain(samples, walkers=nwalkers, parameters=names)
        c.configure(smooth=True,
                    cloud=True,
                    sigmas=[0, 1, 2, 3],
                    )
        c.configure_truth(ls='--', lw=1., c='lightblue')  # c='darkorange')
        c.plotter.plot(figsize=(30, 30),
                       # filename="output/fit.png",
                       display=True,
                       truth=truth
                       )

        res_mcmc = c.analysis.get_summary(parameters=names)
        print('res_mcmc:', res_mcmc)
        params['xc'].value, params['yc'].value, params['amp'].value = (res_mcmc['xc'][1],
                                                                       res_mcmc['yc'][1],
                                                                       res_mcmc['amp'][1])

    plot_comparsion(image=image, params=params,psf_image=psf_image,mask_fitting=mask_fitting, overdist=False)

    if subtract:
        model = model_img(params=params , img_shape=image_shape, psf_image=psf_image,
                            overdist=overdist, debug=False, get_qso_pos=False)

        return model, params

if __name__ == '__main__':

    #read source
    if 1:
        # PKS1830
        if 1:
            fname = 'QSO-B1830-211-SIGHTLINEB_1C_ch1-long_s3d.fits'
            psf_center = (15.34, 9.44)  # sA
            psf_center_B = (19.75, 15.43)  # sA
        if 0:
            fname = 'QSO-B1830-211-SIGHTLINEB_2A_ch2-short_s3d.fits'
            psf_center = (13, 10)  # sA
            psf_center_B = (16, 15)  # sA
        # HD159222
        if 0:
            fname = 'HD-159222_1C_ch1-long_s3d.fits'
            psf_center = (25, 24)  # sA

        # HD163466
        if 0:
            fname = 'HD-163466_1C_ch1-long_BKG_SUBTR_s3d.fits'
            psf_center = (21, 23)  # sA

        # J1007
        if 0:
            fname = 'J1007+2853_dith=1_3A_ch3-short_s3d.fits'
            psf_center = (17.2, 18.5)  # 2  # sA

        # J0901
        if 0:
            fname = 'J0901+2044_dith=1_1A_ch1-short_s3d.fits'
            psf_center = (18.05, 13.3)  # sA

        # J0900
        if 0:
            fname = 'J0900+0214_dith=1_1C_ch1-long_s3d.fits'
            psf_center = (19.4, 12.8)  # sA
        # J1017
        if 0:
            fname = 'J1017+4749_dith=1_2A_ch2-short_s3d.fits'
            psf_center = (15.6, 13.5)  # sA


        cube = datamodels.open(path_to_jwst_folder + '/output/detector3/' + fname)
        # read wavelenght
        sstring = fname.split('_s3d')[0] + '_x1d.fits'
        hdu2 = fits.open(path_to_jwst_folder + '/output/detector3/' + sstring)
        wavelengths = hdu2['EXTRACT1D'].data['WAVELENGTH']
        hdu2.close()

        cube_shape = cube.data.shape  # Data cube: 5 slices, 60x60 pixels each
        image = np.nanmedian(cube.data, axis=0)  # [slice_numer]

    if 1:
        # define params
        (xc_init, yc_init) = psf_center
        amp_init = 1.0
        params = Parameters()
        names = ['xc', 'yc', 'amp']
        for name, value in zip(names, [xc_init, yc_init, amp_init]):
            params.add(name, value=value, min=0, max=np.inf)

        # define mask for fitting region
        rad_fitting_region = 5  # pix

        data_ind = np.indices((image.shape[0], image.shape[1]))
        radial_coord = np.sqrt((data_ind[0] - params['xc'].value) ** 2 + (
                data_ind[1] - params['yc'].value) ** 2)
        mask_fitting = (radial_coord < rad_fitting_region)
        #exclude sourceB
        if 0:
            mask_sB = np.sqrt((data_ind[0] - psf_center_B[0]) ** 2 + (
                    data_ind[1] - psf_center_B[1]) ** 2)<rad_fitting_region
            mask_fitting*=~mask_sB

        #[fit_amplitude,leastsq ,mcmc
        kind = 'leastsq'
        imageA,parsA = subtract_psf(image=image, mask_fitting=mask_fitting, channel='3A', psf_type='custom_psf', overdist=False,
        algorithm = kind, subtract=True)


        #subtrct source B
        if 1:
            params['xc'].value,params['yc'].value = psf_center_B
            data_ind = np.indices((image.shape[0], image.shape[1]))
            radial_coord = np.sqrt((data_ind[0] - params['xc'].value) ** 2 + (
                    data_ind[1] - params['yc'].value) ** 2)
            mask_fitting_2 = (radial_coord < rad_fitting_region)
            if 1:
                mask_sA = np.sqrt((data_ind[0] - psf_center[0]) ** 2 + (
                        data_ind[1] - psf_center[1]) ** 2) < rad_fitting_region
                mask_fitting_2 *= ~mask_sA

            imageB, parsB = subtract_psf(image=image-imageA, mask_fitting=mask_fitting_2, channel='2A', psf_type='custom_psf',
                                          overdist=False,
                                          algorithm=kind, subtract=True)

        if 1:
            def f_imshow(x):
                return np.log10(np.abs(x))

            cmap = plt.cm.coolwarm
            cmap.set_bad('black')
            fig,ax = plt.subplots(1,5,sharey=True,sharex=True)
            vmin,vmax = np.nanmax(f_imshow(image).flatten())-3,np.nanmax(f_imshow(image).flatten())#np.nanquantile(f_imshow(image).flatten(),0.1), np.nanquantile(f_imshow(image).flatten(),0.8)

            im1 = ax[0].imshow(f_imshow(image),origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
            ax[1].imshow(f_imshow(imageA),origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
            ax[2].imshow(f_imshow(imageB),origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
            ax[3].imshow(f_imshow(imageB+imageA), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            im4 = ax[4].imshow((image-(imageA+imageB)),origin='lower',cmap=cmap,vmin=-50,vmax=50)
            fig.colorbar(im1)
            fig.colorbar(im4)
        plt.show()

