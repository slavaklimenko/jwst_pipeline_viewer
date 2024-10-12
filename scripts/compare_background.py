import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import glob,os
from stdatamodels.jwst import datamodels

path = '/home/slava/science/codes/python/jwst/output/detector2/residual_fringe/'
file_list = sorted(glob.glob(path + '*_residual_fringe.fits'))
band = 'SHORT'
channel = '34'
qname = 'J0900'
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

n_im = len(images)
if n_im>0:
    fig,ax = plt.subplots(1,n_im,sharex=True,sharey=True,figsize=(40,10))
    vmin,vmax = np.nanquantile(images[0].data.flatten(),0.05),np.nanquantile(images[0].data.flatten(),0.95)
    for i in range(n_im):
        ax[i].imshow(images[i].data,vmin=vmin,vmax=vmax)
        ax[i].set_title(filenames[i].split('/')[-1])
    plt.savefig('./../output/tmp/' + qname + '_' + band + '_' + channel + '_comparison.pdf', bbox_inches='tight', dpi=1000)

    fig,ax = plt.subplots(3,1,sharex=True,figsize=(20,10))
    diff = images[2].data-images[1].data
    #ax.plot(np.nanmedian(diff,axis=0),label='diff 1 and 0')
    for i in range(n_im):
        label = images[i].meta.target.proposer_name
        dith = images[i].meta.dither.position_number
        if 'BACK' in label:
            color='red'
            ax[0].plot(np.nanmedian(images[i].data,axis=0),label=label+' '+str(dith),c=color)
        else:
            ax[0].plot(np.nanmedian(images[i].data, axis=0), label=label + ' ' + str(dith))
    vmin,vmax = np.nanmin(np.nanmedian(images[i].data,axis=0)[200:800]),np.nanmax(np.nanmedian(images[i].data,axis=0)[200:800])
    ax[0].legend(loc='upper left')
    ax[0].set_ylim(2*vmin,2*vmax)

    tmp = []
    for i in range(n_im):
        label = images[i].meta.target.proposer_name
        if 'BACK' in label:
            tmp.append(np.nanmedian(images[i].data,axis=0))
    median_backgroud_profile = np.nanmedian(np.array(tmp),axis=0)
    tmp = []
    for i in range(n_im):
        label = images[i].meta.target.proposer_name
        if 'BACK' not in label:
            tmp.append(np.nanmedian(images[i].data, axis=0))
        median_qso_profile = np.nanmedian(np.array(tmp), axis=0)
    ax[1].plot(median_backgroud_profile,label='median background',color='red')
    ax[1].plot(median_qso_profile,label='median qso image',color='black')
    ax[1].set_ylim(2*vmin,2*vmax)

    ax[2].plot(median_qso_profile-median_backgroud_profile, label='diff of medians', color='black')
    ax[2].axhline(0,color='green', zorder=-10,ls=':')
    ax[2].legend()
    ax[2].set_xlabel('Coordinate across the traces')
    ax[2].set_ylabel('Flux, DN/s')
    ax[2].set_ylim(-0.1,0.1)

    ax[0].set_title(qname +' ' +band+' '+channel)
plt.savefig('./../output/tmp/'+qname +'_' +band+'_'+channel+'.pdf', bbox_inches='tight',dpi=1000)
plt.show()


if 0:
    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(40, 10))
    vmin, vmax = np.nanquantile(images[0].data.flatten(), 0.05), np.nanquantile(images[0].data.flatten(), 0.95)
    for i in range(4):
        ax[0,i].imshow(images[i].data, vmin=vmin, vmax=vmax)
        ax[0,i].set_title('SCIENCE EXP, DITH='+str(i))
    for i in range(4):
        ax[1,i].imshow(images[4+i].data, vmin=vmin, vmax=vmax)
        ax[1,i].set_title('BACKGR EXP, DITH='+str(i))

    plt.savefig('./../output/tmp/IMAGES_comparison.pdf', bbox_inches='tight',
                dpi=1000)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
    diff = images[2].data - images[1].data
    for i in range(n_im):
        label = images[i].meta.target.proposer_name
        dith = images[i].meta.dither.position_number
        if 'BACK' in label:
            color = 'red'
            ax[0].plot(np.nanmedian(images[i].data, axis=0), label='BACKGR DITH=' + str(dith), c=color)
        else:
            ax[0].plot(np.nanmedian(images[i].data, axis=0), label='SCIENCE DITH=' + ' ' + str(dith))
    vmin, vmax = np.nanmin(np.nanmedian(images[i].data, axis=0)[200:800]), np.nanmax(
        np.nanmedian(images[i].data, axis=0)[200:800])
    ax[0].legend(loc='upper left')
    ax[0].set_ylim(2 * vmin, 2 * vmax)

    tmp = []
    for i in range(n_im):
        label = images[i].meta.target.proposer_name
        if 'BACK' in label:
            tmp.append(np.nanmedian(images[i].data, axis=0))
    median_backgroud_profile = np.nanmedian(np.array(tmp), axis=0)
    tmp = []
    for i in range(n_im):
        label = images[i].meta.target.proposer_name
        if 'BACK' not in label:
            tmp.append(np.nanmedian(images[i].data, axis=0))
        median_qso_profile = np.nanmedian(np.array(tmp), axis=0)
    ax[1].plot(median_backgroud_profile, label='median background', color='red')
    ax[1].plot(median_qso_profile, label='median qso image', color='black')
    ax[1].set_ylim(2 * vmin, 2 * vmax)
    ax[1].legend()

    ax[2].plot(median_qso_profile - median_backgroud_profile, label='diff of medians', color='black')
    ax[2].axhline(0, color='green', zorder=-10, ls=':')
    ax[2].legend()
    ax[2].set_xlabel('Coordinate across the traces')
    ax[0].set_ylabel('Flux, DN/s')
    ax[1].set_ylabel('Flux, DN/s')
    ax[2].set_ylabel('Flux, DN/s')
    ax[2].set_ylim(-0.1, 0.1)

    ax[0].set_title('CHANNEL ' + band + ' ' + channel)
    plt.savefig('./../output/tmp/prifiles.pdf', bbox_inches='tight', dpi=1000)
    plt.show()