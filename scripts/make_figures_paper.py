import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import glob,os
from stdatamodels.jwst import datamodels

from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from IPython.core.pylabtools import figsize
from matplotlib import rcParams
from astropy.io import ascii, fits
rcParams['font.family'] = 'serif'

case = 'fig_backg_subtr'
#case = 'fig_artifact'
if case == 'fig_backg_subtr':
    fig,ax = plt.subplots(1, 5, figsize=(16.5,3))
    #fig2, bx = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(15, 1))
    fontsize = 10

    if 1:
        h  = 0.3
        ax1 = fig.add_axes([0.125, -h, 0.258-0.125, h])
        ax2 = fig.add_axes([0.285, -h, 0.418 - 0.285, h])
        ax3 = fig.add_axes([0.445, -h, 0.579 - 0.445, h])
        ax4 = fig.add_axes([0.606, -h, 0.739 - 0.606, h])
        ax5 = fig.add_axes([0.766, -h, 0.9 - 0.766, h])
        ax_bot = [ax1,ax2,ax3,ax4,ax5]

    path = '/home/slava/science/codes/python/jwst/output/detector2/residual_fringe/'
    file_list = sorted(glob.glob(path + '*_residual_fringe.fits'))
    band = 'MEDIUM'
    channel = '12'
    qname = 'J0901'
    filenames = []
    images = []
    for f in file_list:
        hdulist = fits.open(f)
        header = hdulist[0].header
        f_band, f_ch,f_name = header['BAND'], header['CHANNEl'], header['TARGPROP']
        print(f_band, f_ch,f_name)
        hdulist.close()
        if band == f_band and f_ch == channel and  qname in f_name: # and 'BACK' not in f_name:
            filenames.append(f)
            images.append(datamodels.open(f))
            print(f)
    print('exposure list')
    for f in filenames:
        print(f)



    #bkgr_tot = np.loadtxt('./../output/detector2/bkgr_subtracted/bckgr_model_J0901_MED12.dat')
    bkgr_tot = np.loadtxt('./../output/detector2/bkgr_subtracted/bckgr_model.dat')
    cmap = plt.cm.viridis
    cmap.set_bad('black')

    n_im = len(images)
    i_ref = 0
    if n_im>0:
        vmin,vmax = np.nanquantile(images[0].data.flatten(),0.05),np.nanquantile(images[0].data.flatten(),0.95)
        vmin, vmax = np.nanquantile(images[0].data.flatten(), 0.16), np.nanquantile(images[0].data.flatten(), 1-0.16)
        for i in range(2):
            ax[i].imshow(images[i_ref+i].data,vmin=vmin,vmax=vmax, cmap=cmap,origin='lower')
            #bx[i].plot(np.nanmedian(np.array(images[2+i].data), axis=0),lw=0.5)
            ax_bot[i].plot(np.nanmedian(np.array(images[i_ref+i].data), axis=0),lw=0.5,c='black')
            #ax[i].set_title(filenames[i].split('/')[-1])
        im = ax[2].imshow(bkgr_tot,vmin=vmin,vmax=vmax, cmap=cmap,origin='lower')
        #bx[2].plot(np.nanmedian(np.array(bkgr_tot), axis=0),lw=0.5)
        ax_bot[2].plot(np.nanmedian(np.array(bkgr_tot), axis=0),lw=0.5,c='black')

        #vmin, vmax = np.nanquantile((images[0].data-bkgr_tot).flatten(), 0.2), np.nanquantile((images[0].data-bkgr_tot).flatten(), 0.8)
        for i in range(2):
            ax[i+3].imshow(images[i_ref+i].data-bkgr_tot, vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
            #bx[i+3].plot(np.nanmedian(np.array(images[2 + i].data-bkgr_tot), axis=0),lw=0.5)
            ax_bot[i + 3].plot(np.nanmedian(np.array(images[i_ref + i].data - bkgr_tot), axis=0), lw=0.5,c='black')


        cbar = fig.add_axes([0.91, 0.165, 0.01, 0.68])
        fig.colorbar(im, cax=cbar)
        #cbar.set_yticklabels(fontsize=fontsize)
        ax[4].text(1450,300,'Rate (DN/s)',rotation=90,fontsize=fontsize)


        for axs in ax[:]:
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
            axs.set_xlabel('X coordinate',fontsize=fontsize)
        ax[0].set_ylabel('Y coordinate',fontsize=fontsize)
        ax[0].set_title('Target Image (Dither1)')
        ax[1].set_title('Target Image (Dither2)')
        ax[2].set_title('Background Model')
        ax[3].set_title('Target Image1 (Backg. subtr.)')
        ax[4].set_title('Target Image2 (Backg. subtr.)')


        for axs in ax_bot[:]:
            axs.tick_params(which='both', width=1, direction='in',
                            labelsize=fontsize,
                            right='True',
                            top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=3)
            print(ax[0].get_xlim())
            axs.set_xlim(ax[0].get_xlim())
            axs.set_ylim(-0.09, 0.29)
            axs.xaxis.set_minor_locator(AutoMinorLocator(4))
            axs.xaxis.set_major_locator(MultipleLocator(200))
            axs.yaxis.set_minor_locator(AutoMinorLocator(2))
            axs.yaxis.set_major_locator(MultipleLocator(0.1))
            axs.set_xlabel('X coordinate', fontsize=fontsize)
            axs.axhline(0, ls=':', c='red')
        ax_bot[0].set_ylabel('Mean-Y Rate', fontsize=fontsize)

        fig.savefig('./../output/detector2/bkgr_subtracted/procedure_example_dpi300.pdf', bbox_inches='tight', dpi=300)
        #fig2.savefig('./../output/detector2/bkgr_subtracted/procedure_example_2.pdf', bbox_inches='tight')

if case == 'fig_backg_subtr_short':
    fig,ax = plt.subplots(1, 4, figsize=(12,3))
    #fig2, bx = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(15, 1))
    fontsize = 10

    if 1:
        h  = 0.3
        ax1 = fig.add_axes([0.125, -h, 0.293-0.125, h])
        ax2 = fig.add_axes([0.327, -h, 0.495 - 0.327, h])
        ax3 = fig.add_axes([0.529, -h, 0.697 - 0.529, h])
        ax4 = fig.add_axes([0.731, -h, 0.90 - 0.731, h])
        ax_bot = [ax1,ax2,ax3,ax4]

    path = '/home/slava/science/codes/python/jwst/output/detector2/residual_fringe/'
    file_list = sorted(glob.glob(path + '*_residual_fringe.fits'))
    band = 'MEDIUM'
    channel = '12'
    qname = 'J0901'
    filenames = []
    images = []
    for f in file_list:
        hdulist = fits.open(f)
        header = hdulist[0].header
        f_band, f_ch,f_name = header['BAND'], header['CHANNEl'], header['TARGPROP']
        print(f_band, f_ch,f_name)
        hdulist.close()
        if band == f_band and f_ch == channel and  qname in f_name: # and 'BACK' not in f_name:
            filenames.append(f)
            images.append(datamodels.open(f))
            print(f)
    print('exposure list')
    for f in filenames:
        print(f)



    #bkgr_tot = np.loadtxt('./../output/detector2/bkgr_subtracted/bckgr_model_J0901_MED12.dat')
    bkgr_tot = np.loadtxt('./../output/detector2/bkgr_subtracted/bckgr_model.dat')
    cmap = plt.cm.viridis
    cmap.set_bad('black')

    n_im = len(images)
    i_ref = 0
    if n_im>0:
        vmin,vmax = np.nanquantile(images[0].data.flatten(),0.05),np.nanquantile(images[0].data.flatten(),0.95)
        vmin, vmax = np.nanquantile(images[0].data.flatten(), 0.16), np.nanquantile(images[0].data.flatten(), 1-0.16)
        for i in range(1):
            ax[i+1].imshow(images[i_ref+i].data,vmin=vmin,vmax=vmax, cmap=cmap,origin='lower')
            #bx[i].plot(np.nanmedian(np.array(images[2+i].data), axis=0),lw=0.5)
            ax_bot[i].plot(np.nanmedian(np.array(images[i_ref+i].data), axis=0),lw=0.5,c='black')
            #ax[i].set_title(filenames[i].split('/')[-1])
        im = ax[0].imshow(bkgr_tot,vmin=vmin,vmax=vmax, cmap=cmap,origin='lower')
        #bx[2].plot(np.nanmedian(np.array(bkgr_tot), axis=0),lw=0.5)
        ax_bot[0].plot(np.nanmedian(np.array(bkgr_tot), axis=0),lw=0.5,c='black')

        #vmin, vmax = np.nanquantile((images[0].data-bkgr_tot).flatten(), 0.2), np.nanquantile((images[0].data-bkgr_tot).flatten(), 0.8)
        for i in range(2):
            ax[i+3].imshow(images[i_ref+i].data-bkgr_tot, vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
            #bx[i+3].plot(np.nanmedian(np.array(images[2 + i].data-bkgr_tot), axis=0),lw=0.5)
            ax_bot[i + 3].plot(np.nanmedian(np.array(images[i_ref + i].data - bkgr_tot), axis=0), lw=0.5,c='black')


        cbar = fig.add_axes([0.91, 0.165, 0.01, 0.68])
        fig.colorbar(im, cax=cbar)
        #cbar.set_yticklabels(fontsize=fontsize)
        ax[4].text(1450,300,'Rate (DN/s)',rotation=90,fontsize=fontsize)


        for axs in ax[:]:
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
            axs.set_xlabel('X coordinate',fontsize=fontsize)
        ax[0].set_ylabel('Y coordinate',fontsize=fontsize)
        ax[0].set_title('Image (Dither1)')
        ax[1].set_title('Image (Dither2)')
        ax[2].set_title('Background Model')
        ax[3].set_title('Image1 (Backg. subtr.)')
        ax[4].set_title('Image2 (Backg. subtr.)')


        for axs in ax_bot[:]:
            axs.tick_params(which='both', width=1, direction='in',
                            labelsize=fontsize,
                            right='True',
                            top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=3)
            print(ax[0].get_xlim())
            axs.set_xlim(ax[0].get_xlim())
            axs.set_ylim(-0.09, 0.29)
            axs.xaxis.set_minor_locator(AutoMinorLocator(4))
            axs.xaxis.set_major_locator(MultipleLocator(200))
            axs.yaxis.set_minor_locator(AutoMinorLocator(2))
            axs.yaxis.set_major_locator(MultipleLocator(0.1))
            axs.set_xlabel('X coordinate', fontsize=fontsize)
            axs.axhline(0, ls=':', c='red')
        ax_bot[0].set_ylabel('Mean-Y Rate', fontsize=fontsize)

        #fig.savefig('./../output/detector2/bkgr_subtracted/procedure_example_dpi300.pdf', bbox_inches='tight', dpi=300)
        #fig2.savefig('./../output/detector2/bkgr_subtracted/procedure_example_2.pdf', bbox_inches='tight')

if case == 'fig_artifact':
    fig,ax = plt.subplots(1, 5, figsize=(15,3))
    fig2, bx = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(15, 1))
    fontsize = 10

    if 1:
        h  = 0.3
        ax1 = fig.add_axes([0.125, -h, 0.258-0.125, h])
        ax2 = fig.add_axes([0.285, -h, 0.418 - 0.285, h])
        ax3 = fig.add_axes([0.445, -h, 0.579 - 0.445, h])
        ax4 = fig.add_axes([0.606, -h, 0.739 - 0.606, h])
        ax5 = fig.add_axes([0.766, -h, 0.9 - 0.766, h])
        ax_bot = [ax1,ax2,ax3,ax4,ax5]

    path = '/home/slava/science/codes/python/jwst/output/detector2/residual_fringe/'
    file_list = sorted(glob.glob(path + '*_residual_fringe.fits'))
    band = 'SHORT'
    channel = '34'
    qname = 'J0901'
    filenames = []
    images = []
    for f in file_list:
        hdulist = fits.open(f)
        header = hdulist[0].header
        f_band, f_ch,f_name = header['BAND'], header['CHANNEl'], header['TARGPROP']
        print(f_band, f_ch,f_name)
        hdulist.close()
        if band == f_band and f_ch == channel and  qname in f_name: # and 'BACK' not in f_name:
            filenames.append(f)
            images.append(datamodels.open(f))
            print(f)
    print('exosure list')
    for f in filenames:
        print(f)

    bkgr_tot = np.loadtxt('./../output/detector2/bkgr_subtracted/bckgr_model.dat')
    cmap = plt.cm.viridis
    cmap.set_bad('black')

    n_im = len(images)
    if n_im>0:
        vmin,vmax = np.nanquantile(images[0].data.flatten(),0.05),np.nanquantile(images[0].data.flatten(),0.95)
        vmin, vmax = np.nanquantile(images[0].data.flatten(), 0.16), np.nanquantile(images[0].data.flatten(), 1-0.16)
        dith_0 =0
        for i in range(2):
            ax[i].imshow(images[dith_0+i].data,vmin=vmin,vmax=vmax, cmap=cmap,origin='lower')
            bx[i].plot(np.nanmedian(np.array(images[dith_0+i].data), axis=0),lw=0.5)
            ax_bot[i].plot(np.nanmedian(np.array(images[dith_0+i].data), axis=0),lw=0.5,c='black')
            #ax[i].set_title(filenames[i].split('/')[-1])
        im = ax[2].imshow(bkgr_tot,vmin=vmin,vmax=vmax, cmap=cmap,origin='lower')
        bx[2].plot(np.nanmedian(np.array(bkgr_tot), axis=0),lw=0.5)
        ax_bot[2].plot(np.nanmedian(np.array(bkgr_tot), axis=0),lw=0.5,c='black')

        #vmin, vmax = np.nanquantile((images[0].data-bkgr_tot).flatten(), 0.2), np.nanquantile((images[0].data-bkgr_tot).flatten(), 0.8)
        for i in range(2):
            ax[i+3].imshow(images[dith_0+i].data-bkgr_tot, vmin=vmin, vmax=vmax, cmap=cmap,origin='lower')
            bx[i+3].plot(np.nanmedian(np.array(images[dith_0 + i].data-bkgr_tot), axis=0),lw=0.5)
            ax_bot[i + 3].plot(np.nanmedian(np.array(images[dith_0 + i].data - bkgr_tot), axis=0), lw=0.5,c='black')


        cbar = fig.add_axes([0.91, 0.165, 0.01, 0.68])
        fig.colorbar(im, cax=cbar)
        #cbar.set_yticklabels(fontsize=fontsize)
        ax[4].text(1450,300,'Rate (DN/s)',rotation=90,fontsize=fontsize)


        for axs in ax[:]:
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
            axs.set_xlabel('X coordinate',fontsize=fontsize)
        ax[0].set_ylabel('Y coordinate',fontsize=fontsize)
        ax[0].set_title('Image (Dither1)')
        ax[1].set_title('Image (Dither2)')
        ax[2].set_title('Background Model')
        ax[3].set_title('Image1 (Backg. subtr.)')
        ax[4].set_title('Image2 (Backg. subtr.)')


        for axs in ax_bot[:]:
            axs.tick_params(which='both', width=1, direction='in',
                            labelsize=fontsize,
                            right='True',
                            top='True')
            axs.tick_params(which='major', length=5)
            axs.tick_params(which='minor', length=3)
            print(ax[0].get_xlim())
            axs.set_xlim(ax[0].get_xlim())
            axs.set_ylim(-0.3, 0.85)
            axs.xaxis.set_minor_locator(AutoMinorLocator(4))
            axs.xaxis.set_major_locator(MultipleLocator(200))
            axs.yaxis.set_minor_locator(AutoMinorLocator(4))
            axs.yaxis.set_major_locator(MultipleLocator(0.4))
            axs.set_xlabel('X coordinate', fontsize=fontsize)
            axs.axhline(0, ls=':', c='red')
        ax_bot[0].set_ylabel('Mean-Y Rate', fontsize=fontsize)
        for axs in [ax_bot[3],ax_bot[4]]:
            axs.set_ylim(-0.25, 0.25)
            axs.yaxis.set_minor_locator(AutoMinorLocator(2))
            axs.yaxis.set_major_locator(MultipleLocator(0.1))

        fig.savefig('./../output/detector2/bkgr_subtracted/procedure_artifact_dpi200_12_SHORT.pdf', bbox_inches='tight', dpi=200)
        #fig2.savefig('./../output/detector2/bkgr_subtracted/procedure_example_2.pdf', bbox_inches='tight')


plt.show()

