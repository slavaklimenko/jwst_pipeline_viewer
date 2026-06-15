from scipy.fft import fft, fftfreq
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from jwst.datamodels import dqflags
import scipy
import os,glob
import pickle
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from IPython.core.pylabtools import figsize
from matplotlib import rcParams
from astropy.io import ascii, fits
rcParams['font.family'] = 'serif'


def fit_hist(d=np.histogram([1,2,3]),mask_lim = 2,debug=False,return_full_range=False):
    xc = np.array([np.mean([d[1][ii], d[1][ii + 1]]) for ii in range(len(d[0]))])
    from scipy.optimize import leastsq
    fitfunc = lambda p, x: p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2)
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    # mask outliers
    mask = (d[0] > 0.1 * np.nanmax(d[0])) #*(np.abs(xc)>2*np.mean(np.diff(xc)))  # + (-mask_lim < xc) * (xc < mask_lim)
    if debug:
        fig, ax = plt.subplots()
        ax.plot(xc, d[0], 'o')
        plt.show()
    xdata = xc[mask]
    ydata = d[0][mask]

    if ydata[0]>ydata[2]:
        mask = (d[0] > 0.01 * np.nanmax(
            d[0]))  # *(np.abs(xc)>2*np.mean(np.diff(xc)))  # + (-mask_lim < xc) * (xc < mask_lim)
        xdata = xc[mask]
        ydata = d[0][mask]


    #init = [np.nanmax(d[0]), 0.0, 0.1]
    init = [np.nanmax(d[0]), np.nanmean(xdata), np.nanstd(xdata)]

    out = leastsq(errfunc, init, args=(xdata, ydata))
    #print('params:',out[0])
    if debug:
        fig,ax = plt.subplots()
        ax.plot(xc,d[0],'o')
        ax.plot(xdata, ydata, 'o')
        ax.plot(xdata, fitfunc(out[0], xdata))
        ax.axvline(out[0][1])
        ax.axvline(out[0][1]+out[0][2],ls=':')
        ax.axvline(out[0][1]-out[0][2],ls=':')

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

def calc_mean_rate(images, sig_images, dqs, photom_mask, debug=False, radius=10,  skip_cr_events=False,
                   save_figure=False,level=0.2,alpha=10,n_smooth_iters = 1 ):
    n_int = len(images)
    im = np.array([images[i] for i in range(n_int)])
    sigim = np.array([sig_images[i] for i in range(n_int)])
    dqim = np.array([dqs[i] for i in range(n_int)])
    mask_im = np.ones((n_int, im.shape[1], im.shape[2]))
    mask_im_CR = np.ones((n_int, im.shape[1], im.shape[2]))
    im_shift = np.zeros((n_int,n_smooth_iters+1))

    if n_int==2:

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
        sigma = 10


        im = np.array([images[i] for i in range(n_int)])
        mask_im = np.ones((n_int, im.shape[1], im.shape[2]))
        x0 = np.array(im[0] / im[1] - 1)
        x1 = np.array(im[1] / im[0] - 1)
        for i,x in enumerate([x0,x1]):
            # set outliers as 0
            hot_pix_limit = sigma
            x[~photom_mask] = np.nan
            if save_figure:
                x[np.abs(x) > hot_pix_limit] = 0
            else:
                x[np.abs(x) > hot_pix_limit] = np.nan
            x =  convlove2d(data=im[i], mask=~np.isnan(x),radius=radius)
        y = np.array(x0- x1)
        y_m = np.zeros_like(y)
        y_m[:500, :] = np.nanmean(y[:500, :])
        y_m[500:, :] = np.nanmean(y[500:, :])
        y-=y_m
        std = np.nanstd(y)
        if debug:
            ax[1, 0].imshow(y, vmin=-5*std, vmax=5*std, cmap=cmap, origin='lower')
            xi, yi = np.arange(y.shape[1]), np.arange(y.shape[0])
            ax[1, 0].contour(xi, yi, y, levels=[3*std], linewidths=0.5, colors='k')
        mask_im[0][y>3*std] = 0

        y = -np.array(x0- x1)
        y_m = np.zeros_like(y)
        y_m[:500, :] = np.nanmean(y[:500, :])
        y_m[500:, :] = np.nanmean(y[500:, :])
        y-=y_m
        std = np.nanstd(y)
        if debug:
            ax[1, 1].imshow(y, vmin=-5*std, vmax=5*std, cmap=cmap, origin='lower')
            xi, yi = np.arange(y.shape[1]), np.arange(y.shape[0])
            ax[1, 1].contour(xi, yi, y, levels=[3*std], linewidths=0.5, colors='k')
        mask_im[1][y>3*std] = 0

        # correct for pixels with zero sum mask
        mask_zero = (np.sum(mask_im,axis=0) == 0)
        print('#mask zero', np.sum(mask_zero))
        for i in range(n_int):
            mask_im[i][mask_zero] = 1
        if debug:
            for i in range(n_int):
                im[i][mask_im[i] == 0] = np.nan
            imtot = np.nanmedian(im, axis=0)
            ax[1, n_int].imshow(imtot, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')



    elif n_int>2:
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
            im = np.array([images[i] for i in range(n_int)])
            mask_im = np.ones((n_int, im.shape[1], im.shape[2]))
            for i in range(n_int):
                print('itegration:', k, 'image/exposure', i)
                print('subtract (mean image)*', im_shift[i,k])
                # shift the median of image[i] distribution form zero
                im[i] -= im_shift[i,k]*mean_im
                x = np.array(im[i] / mean_im - 1)
                # set nan and outliers with abs(x)>10 as 0
                sigma = 10
                hot_pix_limit = sigma

                x[~photom_mask] = np.nan
                if save_figure:
                    x[np.abs(x) > hot_pix_limit] = 0
                else:
                    x[np.abs(x) > hot_pix_limit] = np.nan

                # set nan and outliers with abs(x)>10 as 0
                mean, sigma = np.nanmedian(x.flatten()), np.nanstd(x.flatten())
                print('mean,std of im',i,'/mean: ', mean, sigma)
                if save_figure:
                    x[np.abs(x) > 3 * sigma] = 0
                else:
                    x[np.abs(x) > 3*sigma] = np.nan

                x_smoothed = convlove2d(data=x, mask=~np.isnan(x),radius=radius)

                # fit the distribution of smoothed relative signal
                if 1:
                    # correct for the shift
                    bins = np.linspace(-2 * sigma, 2 * sigma, 500)
                    mask = x_smoothed!=0
                    out, fitfunc, xdata, ydata = fit_hist(np.histogram(x_smoothed[mask].flatten(), bins=bins), debug=True)
                    print('(define mean shift) out 1', out)
                    im_shift[i,k+1] = im_shift[i,k] + out[0][1]
                    print('shift smoothed image', i, ' by -', out[0][1])
                    x_smoothed -= out[0][1]

                #fit hist and estimate values for borders
                if 1:
                    #mask = x_smoothed !=-out[0][1]
                    out, fitfunc, xdata, ydata = fit_hist(np.histogram(x_smoothed[mask].flatten(), bins=bins),
                                                          return_full_range=True, debug=False)
                    print('(define xlow, xup) out 2', out)
                    # determine upper border of distribution

                    if 1:
                        mask_up = (xdata > out[0][1]) * (ydata > 100)
                        signal_up = np.abs(ydata[mask_up] - fitfunc(out[0], xdata[mask_up]))
                        up_lim = len(xdata[mask_up]) - 1
                        if len(np.where(signal_up < alpha * np.sqrt(ydata[mask_up]))[0]) > 0:
                            up_lim = np.where(signal_up < (alpha * np.sqrt(ydata[mask_up])))[0][-1]
                        x_up = xdata[mask_up][up_lim]
                        print('set 10sigma x_up',x_up)


                    if 1:
                        mask_low = (xdata < out[0][1])*(ydata > 100)
                        signal_low = np.abs(ydata[mask_low] - fitfunc(out[0], xdata[mask_low]))
                        low_lim = 0
                        if len(np.where(signal_low < alpha * np.sqrt(ydata[mask_low]))[0]) > 0:
                            low_lim = np.where(signal_low < alpha * np.sqrt(ydata[mask_low]))[0][0]
                        x_low = xdata[mask_low][low_lim]
                        print('set 10sigma x_low', x_low)
                    print('contours(x_low,x_up):',x_low,x_up, ' cen shift:', im_shift[i,k],im_shift[i,k+1])


                #mask outliers
                if n_int>2:
                    mask_im[i][(x_smoothed > x_up) + (x_smoothed < x_low)] = 0
                else:
                    mask_im[i][(x_smoothed > x_up)] = 0
                print('masked pixels:',np.sum((x_smoothed > x_up) + (x_smoothed < x_low)))

                #
                if debug and k == n_smooth_iters - 1:
                    vmin_sm, vmax_sm = 2*x_low, 2*x_up
                    ax[2, i].imshow(x_smoothed, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap, origin='lower')
                    ax[2, i].set_title('Smoothed model' + str(i))
                    xi, yi = np.arange(x_smoothed.shape[1]), np.arange(x_smoothed.shape[0])
                    ax[2, i].contour(xi, yi,  x_smoothed, levels=[x_low, x_up], linewidths=0.5, colors='k')
                    ax[1, i].imshow(x, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap, origin='lower')
                    if save_figure and i == 0:
                        np.savetxt('./output/detector2/bkgr_subtracted/x_smoothed.dat', x_smoothed)
                        with open('./output/detector2/bkgr_subtracted/x_smoothed_hist.pkl', 'wb') as f:
                            pickle.dump([ xdata, ydata, bins, fitfunc(out[0], bins),xdata[mask_up][up_lim],xdata[mask_low][low_lim]], f)

                    # plot fit to hist
                    if 1:
                        ah[0,i].set_title('Image'+str(i)+' shift:'+str(round(im_shift[i,k],2)))
                        ah[0,i].plot(bins, fitfunc(out[0], bins))
                        ah[0,i].plot(xdata, ydata, 'o')
                        ah[0,i].axvline(xdata[mask_up][up_lim],c='red')
                        ah[0,i].axvline(xdata[mask_low][low_lim],c='blue')
                        ah[0,i].set_yscale('log')

                        ah[1,i].plot(xdata[mask_up], signal_up)
                        #ah[1,i].plot(xdata[mask_up], np.log10(np.sqrt(ydata[mask_up])))
                        ah[1, i].plot(xdata[mask_up], alpha*(np.sqrt(ydata[mask_up])))
                        ah[1,i].plot(xdata[mask_low], signal_low)
                        #ah[1,i].plot(xdata[mask_low], np.log10(np.sqrt(ydata[mask_low])))
                        ah[1,i].plot(xdata[mask_low], alpha*(np.sqrt(ydata[mask_low])))

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
                for i in range(n_int):
                    ax[3, i].imshow(im[i], vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')

                    ax[4, i].imshow(mask_im[i], cmap=cmap, origin='lower')
                    ax[4, i].set_title('Mask Im ' + str(i))



        if debug:
            vmin, vmax = np.nanquantile(mean_im.flatten(), 0.2), np.nanquantile(mean_im.flatten(), 0.8)
            ax[1, n_int].imshow(mean_im, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
            ax[1, n_int].set_title('Mean final model')

            mask_n_sum = np.sum(mask_im,axis=0)
            mask_n_sum[mask_n_sum==0] = np.nan
            #cmap_npix = plt.cm.viridis
            #cmap_npix.set_bad('red')
            ax[2, n_int].imshow(mask_n_sum, vmin=0, vmax=n_int, cmap=cmap, origin='lower')
            ax[2, n_int].set_title('Sum_Mask')
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



    imsig_inv = np.power(sigim, -2)
    for i in range(n_int):
        im[i][mask_im[i] == 0] = np.nan
    imtot = np.nanmedian(im, axis=0)
    imtotsig = np.power(np.nansum(imsig_inv * mask_im, axis=0), -0.5)


    if debug:
        plt.show()

    if save_figure:
        fig_save, ax_save = plt.subplots(1, 5, figsize=(16.5, 3))
        fontsize = 10
        cmap2 = plt.cm.viridis
        cmap2.set_bad('black')
        #vmin, vmax = np.nanquantile(imtot.flatten(), 0.2), np.nanquantile(imtot.flatten(), 0.8)
        vmin,vmax = -0.08,0.25
        ax_save[0].imshow(images[0],vmin=vmin,vmax=vmax, cmap=cmap2,origin='lower')
        ax_save[1].imshow(np.nanmedian(np.array([images[i] for i in range(n_int)]), axis=0),vmin=vmin,vmax=vmax,
                          cmap=cmap2,origin='lower')
        #ax_save[1].plot(np.nanmedian(np.array([images[i] for i in range(n_int)]), axis=0))
        x_smoothed = np.loadtxt('./output/detector2/bkgr_subtracted/x_smoothed.dat')

        with open('./output/detector2/bkgr_subtracted/x_smoothed_hist.pkl', 'rb') as f:
            (xdata, ydata,bins, fitfunc,  x_up,x_low) = pickle.load(f)
        vmin_sm, vmax_sm = 2 * x_low, 2 * x_up
        ax_save[2].imshow(x_smoothed, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap2, origin='lower')
        xi, yi = np.arange(x_smoothed.shape[1]), np.arange(x_smoothed.shape[0])
        ax_save[2].contour(xi, yi, x_smoothed, levels=[x_up], linewidths=1, colors='red')
        #ax_save[2].contour(xi, yi, x_smoothed, levels=[x_low], linewidths=1, colors='blue')
        if 1:
            h=0.4
            add_ax = fig_save.add_axes([0.445+0.5*(0.579 - 0.445), 0.11+0.3*(0.579 - 0.445), (h)*(0.579 - 0.445), h*(0.88-0.11)])
            add_ax.imshow(x_smoothed, vmin=vmin_sm, vmax=vmax_sm, cmap=cmap2, origin='lower')
            add_ax.contour(xi, yi, x_smoothed, levels=[x_up], linewidths=1, colors='red')
            add_ax.set_xlim(140,340)
            add_ax.set_ylim(440,640)
            for el in ['bottom','left','right','top']:
                add_ax.spines[el].set_color('white')
                add_ax.spines[el].set_lw(3)
            add_ax.set_xticks([])
            add_ax.set_yticks([])

            ax_save[2].add_patch(plt.Rectangle((140,440), 200, 200, ls="--", ec="white", fc="none",lw=1.5))

        ax_save[3].step(xdata,ydata,where='mid')
        ax_save[3].fill_between(x=xdata, y1=ydata-alpha*np.sqrt(ydata), y2=ydata+alpha*np.sqrt(ydata), color='tab:blue',alpha=0.2,zorder=-10)
        ax_save[3].plot(bins, fitfunc)
        ax_save[3].axvline(x_up, c='red')
        ax_save[3].axvline(x_low, c='blue')
        ax_save[3].set_yscale('log')
        x = np.array(imtot)
        x[~photom_mask] = np.nan
        img = ax_save[4].imshow(x,vmin=vmin,vmax=vmax, cmap=cmap2,origin='lower')

        for axs in [ax_save[0],ax_save[1],ax_save[2],ax_save[4]]:
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
        if 1:
            axs =  ax_save[3]
            axs.set_xlim(-0.5,0.5)
            axs.set_ylim(10,1e5)
            axs.xaxis.set_minor_locator(AutoMinorLocator(4))
            axs.xaxis.set_major_locator(MultipleLocator(0.2))
            #axs.set_ylabel('Number of bins', fontsize=fontsize)
            axs.set_xlabel('Pixel Intensity', fontsize=fontsize)
        ax_save[0].set_ylabel('Y coordinate', fontsize=fontsize)
        ax_save[0].set_title('Background Image (Dither1)')
        ax_save[1].set_title('Median Image')
        ax_save[2].set_title('CR Showers Mask')
        ax_save[3].set_title('CR Showers Mask Hist')
        ax_save[4].set_title('Background Model')

        cbar = fig_save.add_axes([0.91, 0.13, 0.01, 0.72])
        fig_save.colorbar(img, cax=cbar)
        # cbar.set_yticklabels(fontsize=fontsize)
        ax_save[4].text(1450, 300, 'Rate (DN/s)', rotation=90, fontsize=fontsize)

        fig_save.savefig('/home/slava/science/codes/python/jwst/output/detector2/tmp.pdf', bbox_inches='tight',
                    dpi=200)
        np.savetxt('./output/detector2/bkgr_subtracted/bckgr_model.dat', imtot)

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
    cmap.set_bad('black')
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
