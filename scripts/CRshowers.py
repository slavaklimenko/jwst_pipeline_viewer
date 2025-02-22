from scipy.fft import fft, fftfreq
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from jwst.datamodels import dqflags
import scipy
import os,glob
# Number of sample points
#int_file = './output/results/jw02441001001_04104_00001_mirifushort_rateints.fits'
#int_file = '/home/slava/science/codes/python/jwst/output/detector1/jw02441001001_04106_00002_mirifushort_rateints.fits'
#int_file = '/home/slava/science/codes/python/jwst/output/results/jw02441001001_04102_00001_mirifulong_rateints.fits'



def fit_hist(d=np.histogram([1,2,3]),mask_lim = 2,debug=False,return_full_range=False):
    xc = np.array([np.mean([d[1][ii], d[1][ii + 1]]) for ii in range(len(d[0]))])
    from scipy.optimize import leastsq
    fitfunc = lambda p, x: p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2)
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    # mask outliers
    mask = (d[0] > 0.1 * np.nanmax(d[0])) #*(np.abs(xc)>2*np.mean(np.diff(xc)))  # + (-mask_lim < xc) * (xc < mask_lim)
    xdata = xc[mask]
    ydata = d[0][mask]


    #print('# pixel (<0.01)', np.sum(xc))

    init = [np.nanmax(d[0]), 0.0, 0.1]

    out = leastsq(errfunc, init, args=(xdata, ydata))
    #print('params:',out[0])
    if debug:
        fig,ax = plt.subplots()
        ax.plot(xc,d[0],'o')
        ax.plot(xdata, ydata, 'o')
        ax.plot(xdata, fitfunc(out[0], xdata))
        plt.show()
    if return_full_range:
        xdata, ydata = xc,d[0]
    return out,fitfunc,xdata,ydata


def check_nearby_pix(data, mask, sigma_limit=3):
    xi, yi = np.arange(data.shape[0]), np.arange(data.shape[1])
    Xi, Yi = np.meshgrid(yi, xi)
    arg = np.argwhere(mask == True)
    mask2 = mask.copy()
    for i in range(arg.shape[0]):
        x, y = arg[i, 0], arg[i, 1]
        if x < data.shape[0] - 10 and x > 10 and y < data.shape[1] - 10 and y > 10:
            mask_loc = (Yi < x + 5) * (Yi > x - 5) * (Xi < y + 5) * (Xi > y - 5)
            mask_loc[(Yi < x + 2) * (Yi > x - 2) * (Xi < y + 2) * (Xi > y - 2)] = False
            data_loc = data[mask_loc].copy()
            mean, std = np.nanmean(data_loc), np.nanstd(data_loc)
            mask_loc = (Yi < x + 5) * (Yi > x - 5) * (Xi < y + 5) * (Xi > y - 5)
            arg2 = np.argwhere(mask_loc == True)
            for j in range(arg2.shape[0]):
                if np.abs(data[arg2[j, 0], arg2[j, 1]] - mean) > sigma_limit * std:
                    mask2[arg2[j, 0], arg2[j, 1]] = True
    mask = mask2
    return mask

def convlove2d(data=np.ones((3,3)), mask=np.ones((3,3)), radius=2, debug=False):
    # set kernel
    res = np.zeros_like(data)
    res[:, :] = np.nan
    filter_kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    for i in range(filter_kernel.shape[0]):
        for j in range(filter_kernel.shape[1]):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                filter_kernel[i, j] = 1
    for i in range(data.shape[0]):
        #print(i)
        for j in range(data.shape[1]):
            if i>radius and i<data.shape[0] - radius and j>radius and j<data.shape[1] - radius and mask[i,j]==1:
                d = data[i-radius:i+radius+1,j-radius:j+radius+1]
                m = mask[i-radius:i+radius+1,j-radius:j+radius+1]
                res[i,j] = np.nansum(d*m*filter_kernel)/np.sum(m*filter_kernel)
            elif i>radius and i<data.shape[0] - radius and j<radius:
                d = data[i - radius:i + radius + 1, :j + radius + 1]
                m = mask[i - radius:i + radius + 1, :j + radius + 1]
                r = filter_kernel[:,:j + radius + 1]
                res[i, j] = np.nansum(d * m * r) / np.sum(m * r)
            elif i > radius and i < data.shape[0] - radius and j>data.shape[1] - radius:
                d = data[i - radius:i + radius + 1, j - radius:]
                m = mask[i - radius:i + radius + 1, j - radius:]
                r = filter_kernel[:,:radius + data.shape[1] - j]
                res[i, j] = np.nansum(d * m * r) / np.sum(m * r)
    if debug:
        cmap = plt.cm.viridis
        cmap.set_bad('black')

        vmin, vmax = np.nanquantile(data.flatten(), 0.2), np.nanquantile(data.flatten(), 0.8)
        fig,ax = plt.subplots(1,3,sharex=True, sharey=True)
        ax[0].imshow(data, cmap=cmap,vmin=vmin,vmax=vmax)
        ax[1].imshow(res,cmap=cmap,vmin=vmin,vmax=vmax)
        ax[2].imshow(mask,cmap=cmap)
        plt.subplots()
        plt.hist(res[~np.isnan(res)],log=True,bins=np.linspace(-0.2,0.5,500))

        plt.show()
    return res

if 0:
    def calc_mean_rate_copy(images, sig_images, dqs, debug=True, radius=15, hot_pix_limit=3, skip_cr_events=True,
                       save_figure=False,level=0.2):
        n_int = len(images)
        im = np.array([images[i] for i in range(n_int)])
        sigim = np.array([sig_images[i] for i in range(n_int)])
        dqim = np.array([dqs[i] for i in range(n_int)])
        mask_im = np.ones((n_int, im.shape[1], im.shape[2]))
        mask_im_CR = np.ones((n_int, im.shape[1], im.shape[2]))

        if skip_cr_events:
            mask_cr = np.zeros_like(im)
            for i in range(n_int):
                mask_cr[i] = (np.bitwise_and(dqim[i], dqflags.pixel['JUMP_DET'])).astype(bool)
                mask_im_CR[i][mask_cr[i] == 1] = 0
            mask_cr_tot = np.sum(mask_cr, axis=0)
            for i in range(n_int):
                mask_im_CR[i][mask_cr_tot == n_int] = 1
                im[i][mask_im_CR[i] == 0] = np.nan

        mean_im = np.nanmedian(im, axis=0)

        #plot images
        if debug:
            cmap = plt.cm.viridis
            cmap.set_bad('black')
            fig, ax = plt.subplots(5, n_int + 1, sharex=True, sharey=True)
            fig_h, ah = plt.subplots(1,2, sharex=True, sharey=True)

            vmin, vmax = np.nanquantile(mean_im.flatten(), 0.2), np.nanquantile(mean_im.flatten(), 0.8)
            for i in range(n_int):
                ax[0, i].imshow(images[i], vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
                #ax[1, i].imshow(im[i], vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
                #ax[3, i].imshow(mask_cr[i],origin='lower')
                ax[0, i].set_title('Image ' + str(i))
            ax[0, n_int].imshow(mean_im, vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
            ax[0, n_int].set_title('Median')



        # set kernel
        filter_kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
        for i in range(filter_kernel.shape[0]):
            for j in range(filter_kernel.shape[1]):
                if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                    filter_kernel[i, j] = 1

        n_smooth_iters = 2

        mask_hot_pixels =np.ones((n_int, im.shape[1], im.shape[2]))

        for k in range(n_smooth_iters):
            print('iteration ',k)
            im = np.array([images[i] for i in range(n_int)])
            for i in range(n_int):
                x = np.array(im[i] / mean_im - 1)
                x[np.isnan(x)] = 0
                mask_hot_pixels[i][x > hot_pix_limit] = 0
                mask_hot_pixels[i][x < -hot_pix_limit] = 0
                x[x > hot_pix_limit] = 0
                x[x < -hot_pix_limit] = 0


                # smooth diff
                npix = scipy.signal.convolve2d(np.ones_like(x), filter_kernel, mode='same', boundary='fill', fillvalue=0)
                x_smoothed = scipy.signal.convolve2d(x, filter_kernel, mode='same', boundary='fill', fillvalue=0) / npix

                #mask ouliers
                mask_im[i][(x_smoothed > level) + (x_smoothed < -2*level)] = 0

                if k == n_smooth_iters - 1 and debug:
                    vmin_sm, vmax_sm = -0.3, 0.3
                    ax[1, i].imshow(x_smoothed, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap, origin='lower')
                    ax[1, i].set_title('Smoothed model' + str(i))
                    xi, yi = np.arange(x_smoothed.shape[1]), np.arange(x_smoothed.shape[0])
                    zi = x_smoothed
                    ax[1, i].contour(xi, yi, zi, levels=[-level, level], linewidths=0.5, colors='k')
                    ax[1, i].contour(xi, yi, zi, levels=[0.1], linewidths=0.5, colors='blue')
                    if 1:
                        fig3,ax3 = plt.subplots(1,3,sharex=True,sharey=True)
                        ax3[0].imshow(x_smoothed, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap, origin='lower')
                        ax3[0].contour(xi, yi, zi, levels=[-level, level], linewidths=0.5, colors='k',lw=2)
                        #ax3[0].contour(xi, yi, zi, levels=[0.1], linewidths=0.5, colors='blue')
                        #ax3[0].contour(xi, yi, zi, levels=[-0.1], linewidths=0.5, colors='red')
                        #plt.subplots()
                        ax3[1].imshow(x, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap, origin='lower')
                        ax3[2].imshow(im[i],vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
                        #plt.contour(xi, yi, zi, levels=[0.1], linewidths=0.5, colors='blue')
                        #plt.contour(xi, yi, zi, levels=[-0.1], linewidths=0.5, colors='red')
                        plt.subplots()
                        bins = np.linspace(-4, 4, 200)
                        plt.hist(x.flatten(), log=True, bins=bins, alpha=0.5, color='black')
                        d = np.histogram(x.flatten(), bins=bins)
                        print('std:x:', np.std(x.flatten()))
                        out, fitfunc, xdata, ydata = fit_hist(d,debug=True)
                        plt.plot(bins, fitfunc(out[0], bins))
                        #plt.show()
                    # plot hist
                    if 1:
                        bins = np.linspace(-0.3, 0.5, 200)
                        ah[i].hist(zi.flatten(), log=False, bins=bins, alpha=0.5, color='black')
                        d = np.histogram(zi.flatten(), bins=bins)
                        if i == 0:
                            d_tot = d
                        else:
                            d_tot[0].data = np.sum([d_tot[0], d[0]], axis=0)

                        out, fitfunc,xdata,ydata = fit_hist(d)
                        c = out[0]
                        xx = bins
                        ah[i].plot(xx, fitfunc(c, xx))
                        ah[i].plot(xdata,ydata,'o')
                        ah[i].set_yscale('log')
                        ah[i].axvline(c[1], ls=':', color='red')
                        ah[i].axvline(c[1] + c[2], ls='--', color='red')
                        # ah[i].axvline(c[1]+2*c[2], ls='--', color='red')
                        # ah[i].axvline(c[1]+3*c[2], ls='--', color='red')
                        ah[i].axvline(c[1] - c[2], ls='--', color='red')
                        # ah[i].axvline(c[1] - 2 * c[2], ls='--', color='red')
                        # ah[i].axvline(c[1] - 3 * c[2], ls='--', color='red')
                        ah[i].fill_betweenx(x1=c[1] - 1 * c[2], x2=c[1] + 1 * c[2], y=[0, 1e6], alpha=0.1, color='red')
                        ah[i].set_ylim(1,1e5)
                        plt.show()
                        print(i, 'fit pars:', c)

                        if i == n_int - 1:
                            out, fitfunc = fit_hist(d_tot, mask_lim=0.1)
                            c = out[0]
                            xx = np.linspace(-0.2, 0.2, 100)
                            ah[i + 1].plot(xx, fitfunc(c, xx))
                            ah[i + 1].axvline(c[1], ls=':', color='blue')
                            ah[i + 1].fill_betweenx(x1=c[1] - 1 * c[2], x2=c[1] + 1 * c[2], y=[0, 1e6], alpha=0.1,
                                                    color='blue')
                            for j in range(n_int):
                                ah[j].fill_betweenx(x1=c[1] - 1 * c[2], x2=c[1] + 1 * c[2], y=[0, 1e6], alpha=0.1,
                                                    color='blue')
                        # print(d)
                        # ah[1].hist(zi.flatten(), log=False, bins=20,alpha=0.5)

                if  k==1 and save_figure:
                    if i == 0:
                        bx[3].imshow(x_smoothed, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap, origin='lower')
                        bx[3].contour(xi, yi, zi, levels=[-level, level], linewidths=0.5, colors='k')

            #add to mask CR events
            if k == n_smooth_iters - 1:
                for i in range(n_int):
                    m = np.array(mask_im[i])
                    m_sum = np.zeros_like(m)
                    for j in range(n_int):
                        if j!=i:
                            m_sum += mask_im[j]
                    mask_im[i][(mask_im_CR[i] == 0)*(m_sum>0)] = 0
                    #if np.sum(mask_im,axis=0)

            # mask CR showers and Cr events
            for i in range(n_int):
                im[i][mask_im[i] == 0] = np.nan


    #        if k <2:
            mean_im = np.nanmedian(im, axis=0)

            if k == n_smooth_iters - 1 and debug:
                vmin_sm, vmax_sm = -0.3, 0.3
                for i in range(n_int):
                    ax[2, i].imshow(mask_im[i], cmap=cmap, origin='lower')
                    ax[2, i].set_title('Mask Im ' + str(i))
                    ax[3, i].imshow(mask_im_CR[i], cmap=cmap, origin='lower')
                    ax[3, i].set_title('Mask CR ' + str(i))
                    ax[4, i].imshow(mask_hot_pixels[i], cmap=cmap, origin='lower')
                    ax[4, i].set_title('Mask HotPix ' + str(i))


        if debug:
            vmin, vmax = np.nanquantile(mean_im.flatten(), 0.2), np.nanquantile(mean_im.flatten(), 0.8)
            ax[1, n_int].imshow(mean_im, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
            mask_n_sum = np.sum(mask_im,axis=0)
            mask_n_sum[mask_n_sum==0] = np.nan
            ax[2, n_int].imshow(mask_n_sum, vmin=0, vmax=n_int, cmap=cmap, origin='lower')
            print('zero exp pixels:', np.sum( np.sum(mask_im,axis=0).flatten()==0) )



        if skip_cr_events and 0:
            mask_cr = np.zeros_like(im)
            for i in range(n_int):
                mask_cr[i] = (np.bitwise_and(dqim[i], dqflags.pixel['JUMP_DET'])).astype(bool)
            mask_cr_shower = 1 - mask_im
            mask_cr_tot = mask_cr + mask_cr_shower

            mask_cr_tot = np.sum(mask_cr_tot.astype(bool), axis=0)

            for i in range(n_int):
                mask_im[i][(mask_cr[i] == 1) * (mask_cr_tot != n_int)] = 0
        if 0:
            for i in range(n_int):
                x = np.array(im[i] / mean_im - 1)
                x[np.isnan(x)] = 0
                x[x > hot_pix_limit] = 0
                x[x < -hot_pix_limit] = 0

                # smooth diff
                npix = scipy.signal.convolve2d(np.ones_like(x), filter_kernel, mode='same', boundary='fill',
                                               fillvalue=0)
                x_smoothed = scipy.signal.convolve2d(x, filter_kernel, mode='same', boundary='fill',
                                                     fillvalue=0) / npix

                mask_im[i][x_smoothed < 0.1] = 1
        #if debug:
        #    for i in range(n_int):
                #ax[3, i].imshow(mask_cr[i], cmap=cmap,origin='lower')
                #ax[3, i].set_title('Mask for CR ' + str(i))

        plt.subplots()
        plt.hist(zi.flatten(), log=True)

        #plt.show()

        imsig_inv = np.power(sigim, -2)
        # imtot = np.nansum(im * imsig_inv * mask_im, axis=0) / np.nansum(imsig_inv * mask_im, axis=0)
        for i in range(n_int):
            im[i][mask_im[i] == 0] = np.nan
        imtot = np.nanmedian(im, axis=0)
        imtotsig = np.power(np.nansum(imsig_inv * mask_im, axis=0), -0.5)

        if debug:
            # vmin, vmax = np.nanquantile(mean_im.flatten(), 0.05), np.nanquantile(mean_im.flatten(), 0.8)
            ax[3, n_int].imshow(imtot, vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
            ax[3, n_int].set_title('Final')
            # ax[2, n_int].imshow(imtot, vmin=vmin, vmax=vmax)

        if save_figure:
            im_b = bx[4].imshow(imtot, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
            cbar_bx = fig2.add_axes([0.91, 0.165, 0.01, 0.68])
            fig2.colorbar(im_b, cax=cbar_bx)
            # cbar_bx.set_yticklabels(fontsize=fontsize)
            # cbar_bx.set_label('Slope (DNs/Groups)')
            bx[4].text(1450, 300, 'Rate (DN/s)', rotation=90, fontsize=fontsize)
            for axs in bx[:]:
                axs.tick_params(which='both', width=1, direction='in',
                                labelsize=fontsize,
                                right='True',
                                top='True')
                axs.tick_params(which='major', length=5)
                axs.tick_params(which='minor', length=3)
                axs.xaxis.set_minor_locator(AutoMinorLocator(4))
                axs.xaxis.set_major_locator(MultipleLocator(200))
                axs.yaxis.set_minor_locator(AutoMinorLocator(4))
                axs.yaxis.set_major_locator(MultipleLocator(200))
                axs.set_xlabel('X coordinate', fontsize=fontsize)
            bx[0].set_ylabel('Y coordinate', fontsize=fontsize)
            bx[0].set_title('Backg.Image (Dither1)')
            bx[1].set_title('Backg.Image (Dither2)')
            bx[3].set_title('CR showers Mask')
            bx[2].set_title('Mean Image')
            bx[4].set_title('Mean CR corrected')

            # fig2.savefig('./output/detector2/bkgr_subtracted/bckgr_model.pdf', bbox_inches='tight',
            #            dpi=2000)
            # fig2.savefig('./output/detector2/bkgr_subtracted/bckgr_model_200.pdf', bbox_inches='tight',
            #             dpi=200)
            #fig2.savefig('./output/detector2/bkgr_subtracted/bckgr_model_300.pdf', bbox_inches='tight', dpi=300)
            #np.savetxt('./output/detector2/bkgr_subtracted/bckgr_model.dat', imtot)

        if debug:
            plt.show()

        return imtot, imtotsig
def calc_mean_rate(images, sig_images, dqs, photom_mask, debug=True, radius=10,  skip_cr_events=True,
                   save_figure=False,level=0.2,alpha=1,n_smooth_iters = 1 ):
    n_int = len(images)
    im = np.array([images[i] for i in range(n_int)])
    sigim = np.array([sig_images[i] for i in range(n_int)])
    dqim = np.array([dqs[i] for i in range(n_int)])
    mask_im = np.ones((n_int, im.shape[1], im.shape[2]))
    mask_im_CR = np.ones((n_int, im.shape[1], im.shape[2]))
    mask_hot_pixels =np.ones((n_int, im.shape[1], im.shape[2]))
    im_shift = np.zeros((n_int,n_smooth_iters+1))


    if skip_cr_events:
        mask_cr = np.zeros_like(im)
        for i in range(n_int):
            mask_cr[i] = (np.bitwise_and(dqim[i], dqflags.pixel['JUMP_DET'])).astype(bool)
            mask_im_CR[i][mask_cr[i] == 1] = 0
            #plt.subplots()
            #plt.imshow(mask_cr[i])
            #plt.title('mask_cr[i]')
            #plt.show()
        mask_cr_tot = np.sum(mask_cr, axis=0)
        for i in range(n_int):
            mask_im_CR[i][mask_cr_tot == n_int] = 1
            im[i][mask_im_CR[i] == 0] = np.nan

    mean_im = np.nanmedian(im, axis=0)


    #plot images
    if debug:
        cmap = plt.cm.viridis
        cmap.set_bad('black')
        fig, ax = plt.subplots(5, n_int + 1, sharex=True, sharey=True)
        fig_h, ah = plt.subplots(2,n_int + 1, sharex=True)

        vmin, vmax = np.nanquantile(mean_im.flatten(), 0.2), np.nanquantile(mean_im.flatten(), 0.8)
        for i in range(n_int):
            ax[0, i].imshow(images[i], vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
            ax[0, i].set_title('Image ' + str(i))
        ax[0, n_int].imshow(mean_im, vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
        ax[0, n_int].set_title('Median')
        #plt.show()



    # set kernel
    filter_kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    for i in range(filter_kernel.shape[0]):
        for j in range(filter_kernel.shape[1]):
            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                filter_kernel[i, j] = 1


    for k in range(n_smooth_iters):
        print('outliers search iteration: ',k)
        im = np.array([images[i] for i in range(n_int)])
        mask_im = np.ones((n_int, im.shape[1], im.shape[2]))
        for i in range(n_int):
            print('itegration:', i)
            # shift the median of image[i] distribution form zero
            im[i] -= im_shift[i,k]*mean_im
            x = np.array(im[i] / mean_im - 1)
            # set nan and outliers with abs(x)>10 as 0
            sigma = 10
            hot_pix_limit = sigma
            mask_hot_pixels[i][np.abs(x) > hot_pix_limit] = 0
            x[np.abs(x) > hot_pix_limit] = np.nan
            x[~photom_mask] = np.nan
            # set nan and outliers with abs(x)>10 as 0
            mean, sigma = np.nanmedian(x.flatten()), np.nanstd(x.flatten())
            print('mean,std of the ratio of images',i,': ', mean, sigma)
            x[np.abs(x) > 3*sigma] = np.nan
            #x[np.isnan(x)] = 0

            x_smoothed = convlove2d(data=x, mask=~np.isnan(x),radius=radius)

            # smooth diff (1 iter)
            #npix = scipy.signal.convolve2d(np.ones_like(x), filter_kernel, mode='same', boundary='fill', fillvalue=0)
            #x_smoothed = scipy.signal.convolve2d(x, filter_kernel, mode='same', boundary='fill', fillvalue=0) / npix


            # fit the distribution of smoothed relative signal
            if 1:
                # correct for the shift
                bins = np.linspace(-2 * sigma, 2 * sigma, 500)
                mask = x_smoothed!=0
                out, fitfunc, xdata, ydata = fit_hist(np.histogram(x_smoothed[mask].flatten(), bins=bins), debug=False)
                im_shift[i,k+1] = im_shift[i,k] + out[0][1]
                #print('shift image', i, ' by -', im_shift[i])
                x_smoothed -= out[0][1]
                #x -= im_shift[i,k+1]
                #npix = scipy.signal.convolve2d(np.ones_like(x), filter_kernel, mode='same', boundary='fill', fillvalue=0)
                #x_smoothed = scipy.signal.convolve2d(x, filter_kernel, mode='same', boundary='fill', fillvalue=0) / npix

            #fit hist and estimate values for borders
            if 1:
                mask = x_smoothed !=-out[0][1]
                out, fitfunc, xdata, ydata = fit_hist(np.histogram(x_smoothed[mask].flatten(), bins=bins),return_full_range=True, debug=False)
                # determine upper border of distribution
                mask_up = xdata > out[0][1]
                signal_up = np.log10(ydata[mask_up]) - np.log10(fitfunc(out[0], xdata[mask_up]))
                up_lim = len(xdata[mask_up]) - 1
                if len(np.where(signal_up < np.log10(alpha*np.sqrt(ydata[mask_up])))[0]) > 0:
                    up_lim = np.where(signal_up < np.log10(alpha*np.sqrt(ydata[mask_up])))[0][-1]
                x_up = xdata[mask_up][up_lim]
                #print(up_lim,xdata[mask_up][up_lim])
                #plt.show()

                mask_low = xdata < out[0][1]
                signal_low = np.log10(ydata[mask_low]) - np.log10(fitfunc(out[0], xdata[mask_low]))
                low_lim = 0
                if len(np.where(signal_low < np.log10(alpha*np.sqrt(ydata[mask_low])))[0]) > 0:
                    low_lim = np.where(signal_low < np.log10(alpha*np.sqrt(ydata[mask_low])))[0][0]
                x_low = xdata[mask_low][low_lim]

                print('contours:',x_low,x_up, ' cen shift:', im_shift[i,k],im_shift[i,k+1])


            #mask outliers
            mask_im[i][(x_smoothed > x_up) + (x_smoothed < x_low)] = 0

            #
            if debug and k == n_smooth_iters - 1:
                vmin_sm, vmax_sm = 2*x_low, 2*x_up
                ax[2, i].imshow(x_smoothed, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap, origin='lower')
                ax[2, i].set_title('Smoothed model' + str(i))
                xi, yi = np.arange(x_smoothed.shape[1]), np.arange(x_smoothed.shape[0])
                ax[2, i].contour(xi, yi,  x_smoothed, levels=[x_low, x_up], linewidths=0.5, colors='k')
                ax[1, i].imshow(x, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap, origin='lower')

                # plot fit to hist

                if 1:
                    ah[0,i].set_title('Image'+str(i)+' shift:'+str(round(im_shift[i,k],2)))
                    ah[0,i].plot(bins, fitfunc(out[0], bins))
                    ah[0,i].plot(xdata, ydata, 'o')
                    ah[0,i].axvline(xdata[mask_up][up_lim],c='red')
                    ah[0,i].axvline(xdata[mask_low][low_lim],c='blue')
                    ah[0,i].set_yscale('log')

                    ah[1,i].plot(xdata[mask_up], signal_up)
                    ah[1,i].plot(xdata[mask_up], np.log10(np.sqrt(ydata[mask_up])))
                    ah[1,i].plot(xdata[mask_low], signal_low)
                    ah[1,i].plot(xdata[mask_low], np.log10(np.sqrt(ydata[mask_low])))

                    #print(up_lim, xdata[mask_up][up_lim])
                    ah[1,i].axvline(xdata[mask_up][up_lim],c='red')
                    ah[1,i].axvline(xdata[mask_low][low_lim],c='blue')

        #add to mask CR events
        if k == n_smooth_iters - 1:
            #plt.show()

            for i in range(n_int):
                m = np.array(mask_im[i])
                m_sum = np.zeros_like(m)
                for j in range(n_int):
                    if j!=i:
                        m_sum += mask_im[j]
                mask_im[i][(mask_im_CR[i] == 0)*(m_sum>0)] = 0
                #if np.sum(mask_im,axis=0)

        # correct for pixels with zero sum mask
        mask_zero = (np.sum(mask_im,axis=0) == 0)
        print('#mask zero', np.sum(mask_zero))
        for i in range(n_int):
            mask_im[i][mask_zero] = 1

        for i in range(n_int):
            im[i][mask_im[i] == 0] = np.nan



#        if k <2:
        mean_im = np.nanmedian(im, axis=0)

        if k == n_smooth_iters - 1 and debug:
            vmin_sm, vmax_sm = -0.3, 0.3
            for i in range(n_int):
                ax[3, i].imshow(im[i], vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

                ax[4, i].imshow(mask_im[i], cmap=cmap, origin='lower')
                ax[4, i].set_title('Mask Im ' + str(i))
                #ax[4, i].imshow(mask_im_CR[i], cmap=cmap, origin='lower')
                #ax[4, i].set_title('Mask CR ' + str(i))
                #ax[4, i].imshow(mask_hot_pixels[i], cmap=cmap, origin='lower')
                #ax[4, i].set_title('Mask HotPix ' + str(i))


    if debug:
        vmin, vmax = np.nanquantile(mean_im.flatten(), 0.2), np.nanquantile(mean_im.flatten(), 0.8)
        ax[1, n_int].imshow(mean_im, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        mask_n_sum = np.sum(mask_im,axis=0)
        mask_n_sum[mask_n_sum==0] = np.nan
        #cmap_npix = plt.cm.viridis
        #cmap_npix.set_bad('red')
        ax[2, n_int].imshow(mask_n_sum, vmin=0, vmax=n_int, cmap=cmap, origin='lower')
        print('zero exp pixels:', np.sum( np.sum(mask_im,axis=0).flatten()==0) )





    if skip_cr_events and 0:
        mask_cr = np.zeros_like(im)
        for i in range(n_int):
            mask_cr[i] = (np.bitwise_and(dqim[i], dqflags.pixel['JUMP_DET'])).astype(bool)
        mask_cr_shower = 1 - mask_im
        mask_cr_tot = mask_cr + mask_cr_shower

        mask_cr_tot = np.sum(mask_cr_tot.astype(bool), axis=0)

        for i in range(n_int):
            mask_im[i][(mask_cr[i] == 1) * (mask_cr_tot != n_int)] = 0
    if 0:
        for i in range(n_int):
            x = np.array(im[i] / mean_im - 1)
            x[np.isnan(x)] = 0
            x[x > hot_pix_limit] = 0
            x[x < -hot_pix_limit] = 0

            # smooth diff
            npix = scipy.signal.convolve2d(np.ones_like(x), filter_kernel, mode='same', boundary='fill',
                                           fillvalue=0)
            x_smoothed = scipy.signal.convolve2d(x, filter_kernel, mode='same', boundary='fill',
                                                 fillvalue=0) / npix

            mask_im[i][x_smoothed < 0.1] = 1
    #if debug:
    #    for i in range(n_int):
            #ax[3, i].imshow(mask_cr[i], cmap=cmap,origin='lower')
            #ax[3, i].set_title('Mask for CR ' + str(i))

    #plt.subplots()
    #plt.hist(zi.flatten(), log=True)

    #plt.show()

    imsig_inv = np.power(sigim, -2)
    # imtot = np.nansum(im * imsig_inv * mask_im, axis=0) / np.nansum(imsig_inv * mask_im, axis=0)
    for i in range(n_int):
        im[i][mask_im[i] == 0] = np.nan
    imtot = np.nanmedian(im, axis=0)
    imtotsig = np.power(np.nansum(imsig_inv * mask_im, axis=0), -0.5)

    if debug:
        # vmin, vmax = np.nanquantile(mean_im.flatten(), 0.05), np.nanquantile(mean_im.flatten(), 0.8)
        fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
        ax[0].imshow(imtot, vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
        x = np.array(imtot)
        x[~photom_mask] = np.nan
        ax[1].imshow(x, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
        plt.title('Final')
        plt.subplots()
        plt.imshow(mask_im_CR[0])
        plt.title('mask_im_CR[0]')
        # ax[2, n_int].imshow(imtot, vmin=vmin, vmax=vmax)

    if debug:
        plt.show()

    return imtot, imtotsig


def calc_hot_pixels(images, sig_images, dqs, debug=True, radius=10,  skip_cr_events=False,
                   save_figure=False,level=0.2,alpha=1):
    n_int = len(images)
    im = np.array([images[i] for i in range(n_int)])
    mask_hot_pixels =np.zeros((n_int, im.shape[1], im.shape[2]))
    mask_im =np.zeros((n_int, im.shape[1], im.shape[2]))
    mask_diff = np.zeros_like(mask_im)

    for i in range(n_int):
        print(i)
        mask_hot_pixels[i] = (images[i]>3)+(images[i]<-1)
        mask_im[i] = check_nearby_pix(images[i],  mask_hot_pixels[i], sigma_limit=5)
        if i>0:
            mask_diff[i] = np.abs(mask_im[i]-mask_im[0])
            print(np.sum(mask_im[i]),np.sum(mask_diff[i]))

    fig,ax = plt.subplots(n_int,3, sharex=True, sharey=True)
    cmap = plt.cm.viridis
    cmap.set_bad('red')
    for i in range(n_int):
        ax[i,0].imshow(images[i],vmin=-1,vmax=4,cmap=cmap)
        ax[i,1].imshow(mask_im[i],cmap=cmap)
        ax[i,2].imshow(mask_diff[i],cmap=cmap)
        print(i,np.sum(mask_diff[i]))


    plt.show()



if __name__ == '__main__':

    if 1:
        int_file = '/home/slava/science/codes/python/jwst/output/results/jw02441001001_04102_00001_mirifulong_rateints.fits'
        hdulist = fits.open(int_file)
        int_slopes = hdulist['SCI'].data
        int_sig_slopes = hdulist['ERR'].data
        int_dq = hdulist['DQ'].data
        int_header = hdulist[0].header
        band, channel = int_header['BAND'], int_header['CHANNEl']
        hdulist.close()

        ref_file = '/home/slava/science/codes/python/jwst/output/results/jw02441001001_04102_00001_mirifulong_rateints.fits'
        hdulist = fits.open(ref_file)
        ref_slopes = hdulist['SCI'].data
        hdulist.close()

        # find path to photom mask
        photom_list = sorted(glob.glob('/home/slava/science/codes/python/jwst/data/references/jwst/miri/*photom*'))
        for f in photom_list:
            hdulist = fits.open(f)
            header = hdulist[0].header
            f_band, f_ch = header['BAND'], header['CHANNEl']
            if band == f_band and f_ch == channel:
                photom_file = f
                break
        from scripts.flat_field import get_mask

        photom_mask = get_mask(path=photom_file)

    if 0:
        calc_mean_rate(int_slopes, int_sig_slopes, int_dq, debug=True,  skip_cr_events=True,photom_mask=photom_mask)
    if 0:
        calc_hot_pixels(int_slopes, int_sig_slopes, int_dq, debug=True,  skip_cr_events=True)

    #run simulations
    if 1:
        num = 300
        mu = 0
        sigma = 2
        s = np.random.normal(mu, sigma, num*num)
        plt.subplots()
        plt.hist(s)
        d = s.reshape(num,num)
        #add shower
        num_sh = 50
        cen_sh = 20
        rad_sh=10
        pos_sh = 10
        amp_sh = sigma/1.5
        d[np.abs(d)>5*sigma] = 0

        d1 = np.zeros((num_sh,num_sh))
        for i in range(num_sh):
            for j in range(num_sh):
                d1[i,j] = amp_sh*np.exp(-((i-cen_sh)**2+(j-cen_sh)**2)/2/rad_sh**2)
        d[pos_sh:pos_sh+num_sh,pos_sh:pos_sh+num_sh]+=d1
        circle1 = plt.Circle((cen_sh + pos_sh, cen_sh + pos_sh), 2 * rad_sh, color='r', lw=0.5, ls='--', fill=False)

        # add negative shower
        num_sh_2 = 50
        cen_sh_2 = 20
        rad_sh_2 = 40
        pos_sh_2 = 100
        amp_sh_2 = sigma

        d1 = np.zeros((num_sh_2, num_sh_2))
        for i in range(num_sh_2):
            for j in range(num_sh_2):
                d1[i, j] = amp_sh_2 * np.exp(-((i - cen_sh_2) ** 2 + (j - cen_sh_2) ** 2) / 2 / rad_sh_2 ** 2)
        d[pos_sh_2:pos_sh_2 + num_sh_2, pos_sh_2:pos_sh_2 + num_sh_2] += d1
        circle2 = plt.Circle((cen_sh_2 + pos_sh_2, cen_sh_2 + pos_sh_2), 2 * rad_sh_2, color='r', lw=0.5, ls='--', fill=False)

        # add negative shower
        num_sh_3 = 50
        cen_sh_3 = 20
        rad_sh_3 = 30
        pos_sh_3 = 100
        amp_sh_3 = -sigma / 3

        d1 = np.zeros((num_sh_3, num_sh_3))
        for i in range(num_sh_3):
            for j in range(num_sh_3):
                d1[i, j] = amp_sh_3 * np.exp(-((i - cen_sh_3) ** 2 + (j - cen_sh_3) ** 2) / 2 / rad_sh_3 ** 2)
        d[2*pos_sh_3:2*pos_sh_3 + num_sh_3, pos_sh_3:pos_sh_3 + num_sh_3] += d1

        #add shift
        d+=0.15


        #plt.subplots()
        plt.hist(d1.flatten())
        print('std(d):', np.std(d))
        #plt.show()
        #
        fig,ax = plt.subplots(3,5)
        vmin,vmax = -sigma/2,sigma/2
        #ax[0].imshow(d,vmin=vmin,vmax=vmax,origin='lower')
        for k in range(5):
            m = d.copy()
            print(k)
            radius = 5+k*5
            # set kernel
            filter_kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
            for i in range(filter_kernel.shape[0]):
                for j in range(filter_kernel.shape[1]):
                    if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                        filter_kernel[i, j] = 1

            npix = scipy.signal.convolve2d(np.ones_like(m), filter_kernel, mode='same', boundary='fill', fillvalue=0)
            d_smoothed = scipy.signal.convolve2d(m, filter_kernel, mode='same', boundary='fill', fillvalue=0) / npix
            #ax[0,k].imshow(d_smoothed,vmin=vmin,vmax=vmax,origin='lower')
            #ax[0,k].set_title('rad='+str(radius))
            xi, yi = np.arange(d_smoothed.shape[1]), np.arange(d_smoothed.shape[0])
            zi = d_smoothed
            #correction for the shift
            if 1:
                bins = np.linspace(-2 * sigma, 2 * sigma, 500)
                out, fitfunc, xdata, ydata = fit_hist(np.histogram(d_smoothed.flatten(), bins=bins),debug=False)
                print('shift by ', out[0][1])
                m-=out[0][1]
                npix = scipy.signal.convolve2d(np.ones_like(m), filter_kernel, mode='same', boundary='fill', fillvalue=0)
                d_smoothed = scipy.signal.convolve2d(m, filter_kernel, mode='same', boundary='fill', fillvalue=0) / npix
                ax[0, k].imshow(d_smoothed, vmin=vmin, vmax=vmax, origin='lower')
                ax[0, k].set_title('rad=' + str(radius))
                xi, yi = np.arange(d_smoothed.shape[1]), np.arange(d_smoothed.shape[0])
                zi = d_smoothed


            #add circle
            circle1 = plt.Circle((cen_sh+pos_sh, cen_sh+pos_sh), rad_sh, color='r',lw=0.5,ls='--', fill=False)
            circle2 = plt.Circle((cen_sh_2 + pos_sh_2, cen_sh_2 + pos_sh_2),  rad_sh_2, color='b', lw=0.5, ls='--', fill=False)
            circle3 = plt.Circle((cen_sh_3 + pos_sh_3, cen_sh_3 + 2*pos_sh_3),  rad_sh_3, color='r', lw=0.5, ls='--',
                                 fill=False)
            ax[0, k].add_patch(circle1)
            ax[0, k].add_patch(circle2)
            ax[0, k].add_patch(circle3)

            #plot hist
            bins = np.linspace(-2*sigma,2*sigma,500)
            ax[1,k].hist(m.flatten(),bins=bins,alpha=0.5)
            #ax[1, k].hist(d.flatten(), bins=bins, alpha=0.5)
            ax[1,k].hist(d_smoothed.flatten(),bins=bins,alpha=0.5,log=True)
            ax[1,k].set_ylim(1,num*num)

            # fit hist
            out, fitfunc, xdata, ydata = fit_hist(np.histogram(d_smoothed.flatten(), bins=bins),return_full_range=True)
            xx = bins
            ax[1,k].plot(xx, fitfunc(out[0], xx))
            ax[1,k].plot(xdata, ydata, 'o')

            # upper border
            mask_up = xdata>out[0][1]
            signal = np.log10(ydata[mask_up])-np.log10(fitfunc(out[0], xdata[mask_up]))
            up_pos = len(xdata[mask_up])-1
            alpha = 0.3
            if len(np.where(signal < np.log10(alpha*np.sqrt(ydata[mask_up])))[0])>0:
                up_pos = np.where(signal < np.log10(alpha*np.sqrt(ydata[mask_up])))[0][-1]
            ax[2, k].plot(xdata[mask_up],signal)
            ax[2, k].plot(xdata[mask_up], np.log10(alpha*np.sqrt(ydata[mask_up])))
            # lower border
            mask_low = xdata < out[0][1]
            signal = np.log10(ydata[mask_low]) - np.log10(fitfunc(out[0], xdata[mask_low]))
            low_pos= 0
            if len(np.where(signal < np.log10(alpha*np.sqrt(ydata[mask_low])))[0])>0:
                low_pos = np.where(signal < np.log10(alpha*np.sqrt(ydata[mask_low])))[0][0]
            ax[2, k].plot(xdata[mask_low], signal)
            ax[2, k].plot(xdata[mask_low], np.log10(alpha*np.sqrt(ydata[mask_low])))

            #signal_mean = running_mean(signal)
            #signal_std = running_std(signal)
            #i_start = np.where(np.abs(signal)<(np.abs(signal_mean))+2*signal_std)[0]
            ax[1, k].axvline(xdata[mask_up][up_pos])
            ax[1, k].axvline(xdata[mask_low][low_pos])

            ax[2, k].axvline(xdata[mask_up][up_pos])
            ax[2, k].axvline(xdata[mask_low][low_pos])

            contour_values = [xdata[mask_low][low_pos],xdata[mask_up][up_pos]]
            ax[0, k].contour(xi, yi, zi, levels=contour_values, linewidths=0.5, colors='k')
            #plt.plot(signal_diff_std)


            #plt.show()
    plt.show()