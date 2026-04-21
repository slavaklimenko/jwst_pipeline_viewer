# Basic system utilities for interacting with files
import glob
import sys
#Modify the path to a directory on your machine
import os
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
from IPython.core.pylabtools import figsize
from matplotlib import rcParams
from astropy.io import ascii, fits
rcParams['font.family'] = 'serif'

def read_settings(init_file='init.dat'):
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
                if values[0] == 'spec2_cachedir:':
                    init_settings['spec2_cachedir'] = values[1]
                if values[0] == 'output1_dir:':
                    init_settings['output1_dir'] = values[1]
                if values[0] == 'output2_dir:':
                    init_settings['output2_dir'] = values[1]
                if values[0] == 'CRDS_PATH:':
                    init_settings['CRDS_PATH'] = values[1]
                if values[0] == 'CRDS_SERVER_URL:':
                    init_settings['CRDS_SERVER_URL'] = values[1]
                if values[0] == 'CRDS_CONTEXT:':
                    init_settings['CRDS_CONTEXT'] = values[1]
    return init_settings
settings =  read_settings()
os.environ["CRDS_PATH"] = settings['CRDS_PATH']
os.environ["CRDS_SERVER_URL"] = settings['CRDS_SERVER_URL']
if 'CRDS_CONTEXT' in  settings.keys():
    os.environ["CRDS_CONTEXT"] = settings['CRDS_CONTEXT']
#os.environ["CRDS_PATH"] = "/home/slava/science/codes/python/jwst/data"
#os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
import time
import shutil
import warnings
import zipfile
import urllib.request

# Astropy utilities for opening FITS and ASCII files
from astropy.io import fits
from astropy.io import ascii
from astropy.utils.data import download_file
# Astropy utilities for making plots
from astropy.visualization import LinearStretch, LogStretch, ImageNormalize, ZScaleInterval

# Numpy for doing calculations
import numpy as np

# Matplotlib for making plots
import matplotlib.pyplot as plt
from matplotlib import rc
# Import the base JWST package and warn if not the expected version
import jwst

# JWST pipelines (encompassing many steps)
from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline

# Individual JWST pipeline steps
from jwst.assign_wcs import AssignWcsStep
from jwst.background import BackgroundStep
from jwst.flatfield import FlatFieldStep
from jwst.srctype import SourceTypeStep
from jwst.straylight import StraylightStep
from jwst.fringe import FringeStep
from jwst.photom import PhotomStep
from jwst.cube_build import CubeBuildStep
from jwst.extract_1d import Extract1dStep
from jwst.cube_skymatch import CubeSkyMatchStep
from jwst.master_background import MasterBackgroundStep
from jwst.outlier_detection import OutlierDetectionStep
from jwst.residual_fringe import ResidualFringeStep

# JWST pipeline utilities
from jwst import datamodels # JWST datamodels
from jwst.associations import asn_from_list as afl # Tools for creating association files
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase # Definition of a Lvl2 association file
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base # Definition of a Lvl3 association file
from stcal import dqflags # Utilities for working with the data quality (DQ) arrays
from jwst.datamodels import dqflags
import scipy

#define input/output
#output_dir = './output/detector2/'
#input_dir = './output/results'
#miri_uncal_file= 'jw02155001001_04102_00001_mirifulong_uncal.fits'
#input_file_base = os.path.basename(miri_uncal_file).replace('uncal.fits', '')
output_dir = settings['output2_dir'] #')./output/detector1/'
input_dir = settings['input2_dir'] #./input/detector1/'


class detector2():
    def __init__(self,miri_uncal_file='',path=None, output_dir=None,spec2_cachedir=None):
        self.name = os.path.basename(miri_uncal_file).replace('uncal.fits', '')
        self.path = path
        self.output_dir=output_dir
        self.data = None
        self.rate_file = None
        self.spec2_cachedir = spec2_cachedir
        self.read_ratefiles(det1_dir=self.path,input_file_base=self.name)
        if self.rate_file != None:
            self.init_rate_files(self.rate_file,mode='Init')
        #self.read_associated_bkg_ratefiles(det1_dir=self.path,input_file_base=self.name)


    def read_ratefiles(self,det1_dir =None, input_file_base = None, debug=1):
        if det1_dir != None:
            for (dirpath, dirname, filenames) in os.walk(det1_dir):
                for f in filenames:
                    if input_file_base in f  and '_rate.fits' in f:
                        self.rate_file = dirpath+'/'+f
                        if debug:
                            print('Read input file:', f)
    def read_associated_bkg_ratefiles(self,det1_dir =None, input_file_base = None, debug=1):
        if det1_dir != None:
            sstring = det1_dir + '/'+'*rate.fits'
            ratefiles = sorted(glob.glob(sstring))
            for f in ratefiles:
                if input_file_base in f:
                    self.rate_bkg_file = f
                    if debug:
                        print('Read input file:', f)


    def init_rate_files(self, input_file=None,mode=None):
        print('init_rate_file', input_file)
        with datamodels.open(input_file) as input_model:
            # If input type is not supported, log warning, set to 'skipped', exit
            if not (isinstance(input_model, datamodels.ImageModel) or
                    isinstance(input_model, datamodels.CubeModel) or
                    isinstance(input_model, datamodels.IFUImageModel)):
                print("Input dataset type is not supported.")
                print("assign_wcs expects ImageModel, IFUImageModel or CubeModel as input.")
                print("Skipping assign_wcs step.")
            else:
                result = input_model.copy()
                result.meta.cal_step.assign_wcs = 'SKIPPED'
                self.data =  result
            if mode=='Init':
                mask_DNU = (np.bitwise_and(self.data.dq, dqflags.pixel['DO_NOT_USE'])).astype(bool)
                self.data.data[mask_DNU] = np.nan
                print('set DNU pixels as nan')

    def update_data_file(self,input_file=None):
        with datamodels.open(input_file) as input_model:
            # If input type is not supported, log warning, set to 'skipped', exit
            if not (isinstance(input_model, datamodels.ImageModel) or
                    isinstance(input_model, datamodels.CubeModel) or
                    isinstance(input_model, datamodels.IFUImageModel)):
                print("Input dataset type is not supported.")
                print("assign_wcs expects ImageModel, IFUImageModel or CubeModel as input.")
                print("Skipping assign_wcs step.")
            else:
                result = input_model.copy()
                self.data = result

    def compare_maps(self, first_map_name='Initial',second_map_name='Initial'):
        read_first_map,read_sec_map= False,False
        map2mapfilename = {}
        map2mapfilename['Initial']='*assignwcsstep.fits'
        map2mapfilename['BkgrSub']='*backgroundstep.fits'
        map2mapfilename['Flatfield']='*flatfieldstep.fits'
        map2mapfilename['Straylight']='*straylightstep.fits'
        map2mapfilename['Fringe']='*fringestep.fits'
        map2mapfilename['Photom'] = '*cal.fits'
        map2mapfilename['ResFringe']='*residual_fringe.fits'

        sstring = self.output_dir +   map2mapfilename[first_map_name]
        filenames = sorted(glob.glob(sstring))
        newmapfile = None
        for f in filenames:
            if self.name in f:
                newmapfile = f
                break
        if newmapfile != None:
            hdulist = fits.open(newmapfile)
            firstmapimage = hdulist['SCI'].data[:, :]
            read_first_map = True

        sstring = self.output_dir +   map2mapfilename[second_map_name]
        filenames = sorted(glob.glob(sstring))
        newmapfile = None
        for f in filenames:
            if self.name in f:
                newmapfile = f
                break
        if newmapfile != None:
            hdulist = fits.open(newmapfile)
            secmapimage = hdulist['SCI'].data[:, :]
            read_sec_map = True


        if read_first_map== True and read_sec_map == True:
            rc('axes', linewidth=2)
            fig, (ax1, ax2, ax3,ax4) = plt.subplots(1, 4, figsize=(20, 4), dpi=100, sharey=True,sharex=True)

            vmin,vmax = np.nanquantile(firstmapimage.flatten(), 0.01), np.nanquantile(firstmapimage.flatten(), 0.99)

            # And plot the data
            c1 = ax1.imshow(firstmapimage, cmap='viridis', origin='lower',vmin=vmin,vmax=vmax)
            ax1.set_title(first_map_name)
            ax1.set_xlabel('X pixel')
            ax1.set_ylabel('Y pixel')

            c2 = ax2.imshow(secmapimage, cmap='viridis', origin='lower', vmin=vmin,vmax=vmax)
            ax2.set_title(second_map_name)
            ax2.set_xlabel('X pixel')

            new = secmapimage-firstmapimage
            vmin, vmax = np.nanquantile(new.flatten(), 0.01), np.nanquantile(new.flatten(), 0.99)
            if second_map_name == 'Flatfield':
                vmin, vmax = 0.8, 1.2
                c3 = ax3.imshow(firstmapimage/secmapimage, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
                flatfile = self.data.meta.ref_file.flat.name.split('//')[-1]
                hdu = fits.open(os.environ["CRDS_PATH"] +'/references/jwst/miri/'+flatfile)
                ax4.imshow(hdu['SCI'].data, vmin=vmin, vmax=vmax)
                hdu.close()
            else:
                c3 = ax3.imshow(secmapimage-firstmapimage, cmap='viridis', origin='lower', vmin=vmin,vmax=vmax)
            ax3.set_title(first_map_name +' - '+second_map_name)
            ax3.set_xlabel('X pixel')

            fig.colorbar(c1, ax=ax1)
            fig.colorbar(c2, ax=ax2)
            fig.colorbar(c3, ax=ax3)

            plt.show()
        else:
            print('wrong choice of the maps')


    def read_step_results(self,step_name = 'Initial'):
        map2mapfilename = {}
        map2mapfilename['Initial'] = '*assignwcsstep.fits'
        map2mapfilename['BkgrSub'] = '*backgroundstep.fits'
        map2mapfilename['Flatfield'] = '*flatfieldstep.fits'
        map2mapfilename['Straylight'] = '*straylightstep.fits'
        map2mapfilename['Fringe'] = '*fringestep.fits'
        map2mapfilename['FluxCalib'] = '*photomstep.fits'
        map2mapfilename['ResFringe'] = '*cal.fits'

        sstring = self.output_dir + map2mapfilename[step_name]
        filenames = sorted(glob.glob(sstring))
        newmapfile = None
        for f in filenames:
            if self.name in f:
                newmapfile = f
                break
        if newmapfile != None:
            print('load ',newmapfile)
            self.init_rate_files(input_file=newmapfile)
            if step_name == 'ResFringe':
                self.calfiles = self.data
    def read_step_results_version2(self,step_name = 'cal'):
        map2mapfilename = {}
        map2mapfilename['cal'] = '*_cal.fits'
        map2mapfilename['res_fringe'] = '*residual_fringe.fits'

        if step_name == 'res_fringe':
            sstring = self.output_dir+'residual_fringe/' + map2mapfilename[step_name]
        if step_name == 'cal':
            sstring = self.output_dir +  map2mapfilename[step_name]

        filenames = sorted(glob.glob(sstring))
        newmapfile = None
        for f in filenames:
            if self.name in f:
                newmapfile = f
                break
        if newmapfile != None:
            print('load ',newmapfile)
            self.init_rate_files(input_file=newmapfile)
            self.calfiles = self.data


    def create_mask_hot_pix_step(self, input_file=None, debug=True, output_dir = './output/results/',ref_file = None, dither_file=None,save_file=False):
        '''
        Mask hot pipxels as bad.
        '''

        filename = output_dir + input_file #self.rate_file
        ratefile = datamodels.open(filename)
        hdulist = fits.open(filename)
        rate_header = hdulist[0].header
        rate_name=rate_header['TARGPROP']+rate_header['CHANNEL']+rate_header['DETECTOR'] #+rate_header['TARGNAME']+str(rate_header['PATT_NUM'])
        print(rate_header['TARGPROP']+rate_header['CHANNEL']+rate_header['BAND'])


        filename = output_dir + ref_file  # self.rate_file
        reffile = datamodels.open(filename)
        hdulist = fits.open(filename)
        ref_header = hdulist[0].header
        ref_name = ref_header['TARGPROP'] + ref_header['CHANNEL'] + ref_header['DETECTOR'] #+ ref_header['TARGNAME'] +  str(ref_header['PATT_NUM'])
        print(ref_header['TARGPROP'] + ref_header['CHANNEL'] + ref_header['BAND'])

        filename = output_dir + dither_file  # self.rate_file
        dithfile = datamodels.open(filename)
        hdulist = fits.open(filename)
        dith_header = hdulist[0].header
        dith_name = dith_header['TARGPROP'] + dith_header['CHANNEL'] + dith_header['DETECTOR'] #+ dith_header['TARGNAME'] + str(dith_header['PATT_NUM'])
        print(dith_header['TARGPROP'] + dith_header['CHANNEL'] + dith_header['BAND'])

        filename = os.environ["CRDS_PATH"] +'/Hot_pixels/hot_pixels_12_SHORT.fits'
        #reference_file = datamodels.open(filename)
        reference_file = fits.open(filename)
        data_reference =  reference_file[0].data


        data = ratefile.data
        print(data.shape)
        dq = ratefile.dq
        data_ref = reffile.data
        dq_ref = reffile.dq
        data_dith = dithfile.data
        dq_dith = dithfile.dq
        fig,ax = plt.subplots(2,4,sharex=True, sharey=True)
        fig2, ax2 = plt.subplots()
        p = data.flatten()
        p_ref = data_ref.flatten()
        threshold_pix_val = 1
        plt.subplots()
        plt.hist(data.flatten(),log=True, bins=np.linspace(-10, 100, 101),alpha=0.2)
        plt.hist(data_dith.flatten(),log=True, bins=np.linspace(-10, 100, 101),alpha=0.2)

        def check_hot_pix(data,dq,limit):
            mask = data>limit
            arg = np.argwhere(data>limit)
            for i in range(arg.shape[0]):
                x,y = arg[i,0],arg[i,1]
                if x<data.shape[0]-6 and y < data.shape[1]-6 and x>6 and y>6:
                    #print(x,y)
                    f = data[x,y]
                    flux_mean_1,npix = 0,0
                    for j,k in zip([x-2,x-3,x-4,x-5,x-6,x+4,x+5,x+6,x+3,x+2],[y,y,y,y,y,y,y,y,y,y]):
                        if dq[j,k]==0:
                            flux_mean_1 += data[j,k]
                            npix +=1
                    if npix>0:
                        flux_mean_1/=npix


                        flux_mean2,npix,delta,ar = 0,0,[],[[],[]]
                        for j, k in zip([x-2,x-3,x-4,x-5,x-6,x+4,x+5,x+6,x+3,x+2],[y,y,y,y,y,y,y,y,y,y]):
                            if dq[j, k] == 0 and data[j, k]<2*flux_mean_1:
                                flux_mean2 += data[j, k]
                                npix += 1
                                delta.append(data[j, k])
                            ar[0].append(data[j, k])
                            ar[1].append(dq[j, k])
                        if npix > 0:
                            flux_mean2 /= npix

                        flux_mean=flux_mean2
                        f = data[x,y]
                        flux_disp = np.sqrt(np.sum(np.power(np.array(delta)-flux_mean,2))/npix)
                        if (np.abs(data[x,y]-flux_mean)<7*flux_disp)+(f<2*flux_mean):
                            mask[x,y] = False
                        elif mask[x,y]:
                            if f > 2 and x<497 and x>487 and y>763 and y<767:
                                print()
                        if np.bitwise_and(dq[x,y],1) and 0:
                            d = dq[x,y]
                            mask[x, y] = False
                        elif np.bitwise_and(dq[x,y],4) and 0:
                            d = dq[x,y]
                            mask[x, y] = False
                    else:
                        mask[x, y] = False

            return mask

        def check_hot_pix_negative(data,dq,limit):
            mask = data<limit
            arg = np.argwhere(data<limit)
            for i in range(arg.shape[0]):
                x,y = arg[i,0],arg[i,1]
                if x<data.shape[0]-1 and y < data.shape[1]-1 and x>1 and y>1:
                    f = data[x,y]
                    flux_mean,npix = 0,0
                    for j,k in zip([x,x,x-1,x+1],[y-1,y+1,y,y]):
                        if dq[j,k]==0:
                            flux_mean += data[j,k]
                            npix +=1
                    if npix>0:
                        flux_mean/=npix
                    if np.abs(data[x,y])<np.abs(2*flux_mean):
                        mask[x,y] = False
                    elif npix == 0:
                        mask[x, y] = False
                    else:
                        print(x, y)
                    if dq[x,y] != 0 and  dq[x,y] != 4:
                        mask[x, y] = False
            return mask

        hot_pixels_list = check_hot_pix(data, dq, threshold_pix_val) #(data > threshold_pix_val)*((dq==0) + (dq==4))
        hot_pixels_ref_list = check_hot_pix(data_ref, dq_ref, threshold_pix_val) #(data_ref > threshold_pix_val)*((dq_ref==0) + (dq_ref==4))
        hot_pixels_dith_list = check_hot_pix(data_dith, dq_dith, threshold_pix_val) #(data_dith > threshold_pix_val)*((dq_dith==0) + (dq_dith==4))







        ar = np.zeros_like(data)
        vmin,vmax = 0,threshold_pix_val
        ar[hot_pixels_list] = data[hot_pixels_list]
        ax[0,0].imshow(ar, vmin=vmin, vmax=vmax)
        ax[0,0].set_title(rate_name,fontsize=8)
        ax[1, 0].imshow(ratefile.data, vmin=-3, vmax=3)
        print('N of hot_pixels_list', np.sum(hot_pixels_list))
        #ax2.hist(ar[ar>threshold_pix_val].flatten(),bins=np.linspace(0,20,21),alpha=0.3)
        ar = np.zeros_like(data)
        ar[hot_pixels_ref_list] = data_ref[hot_pixels_ref_list]
        ax[0,1].imshow(ar, vmin=vmin, vmax=vmax)
        ax[0,1].set_title(ref_name,fontsize=8)
        ax[1, 1].imshow(reffile.data, vmin=-3, vmax=3)
        print('N of hot_pixels_ref_list', np.sum(hot_pixels_ref_list))
        #ax2.hist(ar[ar>threshold_pix_val].flatten(),bins=np.linspace(0,20,21),alpha=0.3)

        ar = np.zeros_like(data)
        ar[hot_pixels_dith_list] = data_dith[hot_pixels_dith_list]
        ax[0,2].imshow(ar, vmin=vmin, vmax=vmax)
        ax[0,2].set_title(dith_name,fontsize=8)
        ax[1,2].imshow(dithfile.data, vmin=-3, vmax=3)
        print('N of hot_pixels_ref_list', np.sum(hot_pixels_dith_list))

        ar = np.zeros_like(data)
        ar[hot_pixels_ref_list*hot_pixels_list*hot_pixels_dith_list] = data_dith[hot_pixels_ref_list*hot_pixels_list*hot_pixels_dith_list]
        ax[0,3].imshow(ar, vmin=vmin, vmax=vmax)
        #ax[0, 3].imshow(data_reference, vmin=vmin, vmax=vmax)
        ax[0,3].set_title('Mix')
        print('N of hot pix', np.sum(hot_pixels_ref_list*hot_pixels_list*hot_pixels_dith_list), ' of ',np.sum(hot_pixels_ref_list))
        ax2.hist(ar[ar>threshold_pix_val].flatten(),bins=np.linspace(0,20,101),alpha=0.3)
        ax2.hist(data_dith.flatten(), bins=np.linspace(-10, 20, 101), alpha=0.3)
        plt.show()
        #plt.hist(p[~np.isnan(p)],log=True,bins=np.linspace(-1,100,200))
        if save_file:
            CHAN,BAND,PROGRAMID =rate_header['CHANNEL'],rate_header['BAND'],rate_header['PROGRAM']
            mask = hot_pixels_ref_list*hot_pixels_list*hot_pixels_dith_list
            hdu = fits.PrimaryHDU(mask.astype(int))
            hdul = fits.HDUList([hdu])
            header = hdul[0].header
            header['TELESCOP'] = 'JWST'
            header['INSTRUME'] = 'MIRI'
            header['CHANNEL'] = CHAN
            header['BAND'] = BAND
            header['AUTHOR'] = 'V.KLIMENKO'
            header['COMMENT'] = 'MASK OF SINGLE HOT PIXELS IN MIRI DETECTORS'
            header['PROGRAM'] = PROGRAMID
            hdul.writeto(os.environ["CRDS_PATH"] +'/Hot_pixels/ID02441/hot_pixels_'+CHAN+'_'+BAND+ '.fits',overwrite=True)

        plt.show()
        p =1

    def show_hot_pix_maps(self,debug=True):
        output_dir = os.environ["CRDS_PATH"]

        hdulist = fits.open(output_dir+'hot_pixels_34_LONG.fits')
        mlong = hdulist[0].data[:, :]
        hdulist = fits.open(output_dir+'hot_pixels_34_MEDIUM.fits')
        mmed= hdulist[0].data[:, :]
        hdulist = fits.open(output_dir + 'hot_pixels_34_SHORT.fits')
        mshort = hdulist[0].data[:, :]
        if debug:
            fig,ax = plt.subplots(1,3,sharey=True,sharex=True)
            ax[0].imshow(mlong)
            ax[1].imshow(mmed)
            ax[2].imshow(mshort)
            plt.show()
    def select_hot_pix(self,output_dir='./data_local/Hot_pixels/',debug=False):
        band = self.data.meta.instrument.band
        channel = self.data.meta.instrument.channel
        if channel == '34':
            filename = 'hot_pixels_34_'+band+'.fits'
            hdulist = fits.open(output_dir + filename)
            mask_hot_pix = hdulist[0].data[:, :]
            hdulist.close()
        elif channel == '12':
            filename = 'hot_pixels_12_'+band+'.fits'
            hdulist = fits.open(output_dir + filename)
            mask_hot_pix = hdulist[0].data[:, :]
            hdulist.close()
        print('read hot_pix_map from',  filename)
        self.data.dq = np.bitwise_or(self.data.dq,mask_hot_pix)
        self.data.data[mask_hot_pix.astype(bool)] = np.nan
        if debug:
            fig,ax = plt.subplots(1,2,sharex=True,sharey=True)
            ax[0].imshow(mask_hot_pix)
            ax[1].imshow(self.data.data,vmin=0,vmax=4)
            plt.show()
    def select_hot_pix_version2(self,filelist='',debug=False,fast_mode=True):
        band = self.data.meta.instrument.band
        channel = self.data.meta.instrument.channel
        listnames_source = []
        listnames_backgr = []
        data = filelist.data
        for s in data:
            s_band = s[4]
            s_ch = s[5]
            if s_band == band and s_ch == channel:
                if 'BACK' in s[3]:
                    listnames_backgr.append(s)
                    if s[0].split('uncal')[0] == self.name:
                        ref_list = listnames_source
                else:
                    listnames_source.append(s)
                    if s[0].split('uncal')[0] == self.name:
                        ref_list = listnames_backgr

        ref_name = ref_list[0][0].replace('uncal', 'rate.fits')
        ref_file = os.path.join(self.path + '/' + ref_name)
        hdulist = fits.open(ref_file)
        ref_slope = hdulist['SCI'].data
        hdulist.close()

        # set mask of hot pixels
        bkgr_name = listnames_backgr[0][0].replace('uncal', 'rate.fits')
        bkgr_file = os.path.join(self.path + '/' + bkgr_name)
        hdulist = fits.open(bkgr_file)
        bkgr_slope = hdulist['SCI'].data
        hdulist.close()

        bkgr_mean = np.zeros_like(bkgr_slope)
        bkgr_std = np.zeros_like(bkgr_slope)

        bkgr_mean[500:,:500] = np.nanmean(bkgr_slope[500:,:500])
        bkgr_mean[:500, :500] = np.nanmean(bkgr_slope[:500, :500])

        bkgr_mean[:500, 500:] = np.nanmean(bkgr_slope[:500,500:])
        bkgr_mean[500:, 500:] = np.nanmean(bkgr_slope[500:, 500:])

        bkgr_std[:500, :500] = np.nanstd(bkgr_slope[:500,:500])
        bkgr_std[500:, :500] = np.nanstd(bkgr_slope[500:, :500])

        bkgr_std[:500, 500:] = np.nanstd(bkgr_slope[:500,500:])
        bkgr_std[500:, 500:] = np.nanstd(bkgr_slope[500:,500:])

        mask_hot_pix_loc = np.abs(ref_slope - bkgr_mean)>2*bkgr_std
        mask_hot_pix_ref = np.abs(self.data.data - bkgr_mean)>2*bkgr_std
        mask_hot = mask_hot_pix_loc*mask_hot_pix_ref

        def check_nearby_pix(data,mask,sigma_limit=3):
            xi, yi = np.arange(self.data.data.shape[0]), np.arange(self.data.data.shape[1])
            Xi, Yi = np.meshgrid(yi, xi)
            arg = np.argwhere(mask == True)
            mask2 = mask.copy()
            for i in range(arg.shape[0]):
                x, y = arg[i, 0], arg[i, 1]
                if x<data.shape[0]-10 and x>10 and y<data.shape[1]-10 and y>10:
                    mask_loc = (Yi<x+5)*(Yi>x-5)*(Xi<y+5)*(Xi>y-5)
                    mask_loc[(Yi<x+2)*(Yi>x-2)*(Xi<y+2)*(Xi>y-2)] = False
                    data_loc = data[mask_loc].copy()
                    mean, std = np.nanmean(data_loc),np.nanstd(data_loc)
                    mask_loc = (Yi<x+5)*(Yi>x-5)*(Xi<y+5)*(Xi>y-5)
                    arg2 = np.argwhere(mask_loc==True)
                    for j in range(arg2.shape[0]):
                        if np.abs(data[arg2[j, 0], arg2[j, 1]]- mean)>sigma_limit*std:
                            mask2[arg2[j, 0], arg2[j, 1]] = True
            mask= mask2
            return mask

        if not fast_mode:
            m1 = check_nearby_pix(ref_slope,mask_hot)
            m2 = check_nearby_pix(self.data.data,mask_hot)
            mask_hot = m1*m2

        print('number of hot pixels:',np.sum(mask_hot))

        if debug:
            fig,ax = plt.subplots(1,4,sharex=True,sharey=True)
            vmin, vmax = np.nanquantile(self.data.data.flatten(), 0.01), np.nanquantile(self.data.data.flatten(), 0.99)
            ax[0].imshow(self.data.data,vmin=vmin,vmax=vmax)
            ax[0].set_title(self.name)
            ax[1].imshow(ref_slope,vmin=vmin,vmax=vmax)
            ax[1].set_title(ref_name)
            ax[2].imshow(mask_hot,vmin=vmin,vmax=vmax)
            ax[3].imshow(m2, vmin=vmin, vmax=vmax)
            plt.show()

        #interpolation of flux in hot pixels

        def fix_hot_pix(data, mask, dq, err, debug=False):
            from scipy.interpolate import interp1d

            arg = np.argwhere(mask == True)
            sat_flag = dqflags.pixel["SATURATED"]
            dnu_flag = dqflags.pixel["DO_NOT_USE"]

            for i in range(arg.shape[0]):
                x, y = arg[i, 0], arg[i, 1]
                if x<data.shape[0]-10 and x>10:
                    y_loc = data[x - 10:x + 10, y].copy()
                    yerr_loc = err[x - 10:x + 10, y].copy()
                    dq_loc = dq[x - 10:x + 10, y].copy()
                    mask_loc = ~mask[x - 10:x + 10, y].copy()
                elif x<=10:
                    y_loc = data[0:x + 10, y].copy()
                    yerr_loc = err[0:x + 10, y].copy()
                    dq_loc = dq[0:x + 10, y].copy()
                    mask_loc = ~mask[0:x + 10, y].copy()
                elif (x>= data.shape[0]-10):
                    y_loc = data[x-10:, y].copy()
                    yerr_loc = err[x - 10:, y].copy()
                    dq_loc = dq[x - 10:, y].copy()
                    mask_loc = ~mask[x - 10:, y].copy()
                x_loc = np.arange(y_loc.shape[0])
                x_c = np.where(y_loc == data[x, y])[0][0]

                mask_loc *= (x_loc != x_c)  # *(x_loc!=4)*(x_loc!=6)
                mask_loc *= ~np.bitwise_and(dq_loc,sat_flag).astype(bool)
                mask_loc *= ~np.bitwise_and(dq_loc, dnu_flag).astype(bool)
                mask_loc *= ~np.isnan(y_loc)


                if np.sum(mask_loc)>2:
                    #mask_loc_bad_pix = np.zeros_like(mask_loc)
                    if 0:
                        mean_loc,std_loc = np.nanmean(y_loc[mask_loc]),np.nanstd(y_loc[mask_loc])
                        mask_loc_bad_pix = np.abs(y_loc - mean_loc)>5*std_loc
                        mask_loc *= ~mask_loc_bad_pix

                        if np.sum(mask_loc) > 2:
                            f_interp = interp1d(x_loc[mask_loc], y_loc[mask_loc], kind='quadratic',fill_value='extrapolate')
                            bad_pix_loc = np.where(mask_loc_bad_pix==True)[0]
                            for x_el in bad_pix_loc:
                                print(x+x_el-x_c, y, data[x+x_el-x_c, y], f_interp(x_el))
                                data[x+x_el-x_c, y] = f_interp(x_el)

                            print(x, y, y_loc)
                            print(f_interp(x_c))
                            if debug:
                                if len(bad_pix_loc)>1 and 0:
                                    xx = np.linspace(x_loc[mask_loc][0], x_loc[mask_loc][-1], 100)
                                    plt.subplots()
                                    plt.plot(x_loc, y_loc, 'o')
                                    plt.axhline(mean_loc)
                                    plt.axhline(mean_loc+5*std_loc, ls='--')
                                    plt.axhline(mean_loc-5*std_loc, ls='--')
                                    for x_el in bad_pix_loc:
                                        plt.plot(x_el, f_interp(x_el), 'D',color='orange')
                                    plt.plot(xx, f_interp(xx))
                                    plt.title(str(x)+str(y))
                                    plt.show()
                    if 1:
                        mean_loc, std_loc = np.nanmean(y_loc[mask_loc]), np.nanstd(y_loc[mask_loc])
                        mask_loc_bad_pix = np.abs(y_loc - mean_loc) > 5 * std_loc
                        mask_loc *= ~mask_loc_bad_pix

                        if np.sum(mask_loc) > 2:
                            f_interp = interp1d(x_loc[mask_loc], y_loc[mask_loc], kind='linear',
                                                fill_value='extrapolate')
                            bad_pix_loc = np.where(mask_loc_bad_pix == True)[0]
                            data[x, y] = f_interp(x_c)
                            if debug:
                                if f_interp(x_c)>mean_loc+2*std_loc:
                                    xx = np.linspace(x_loc[mask_loc][0], x_loc[mask_loc][-1], 100)
                                    plt.subplots()
                                    plt.plot(x_loc, y_loc, 'o')
                                    plt.errorbar(x=x_loc,y=y_loc,yerr=yerr_loc)
                                    plt.axhline(mean_loc)
                                    plt.axhline(mean_loc + 5 * std_loc, ls='--')
                                    plt.axhline(mean_loc - 5 * std_loc, ls='--')
                                    #for x_el in bad_pix_loc:
                                    #    plt.plot(x_el, f_interp(x_el), 'D', color='orange')
                                    plt.plot(x_c, f_interp(x_c), 'D', color='orange')
                                    plt.plot(xx, f_interp(xx))
                                    plt.title(str(x) + str(y))
                                    plt.show()

            return data

        self.data.data = fix_hot_pix(self.data.data, mask=mask_hot, dq=self.data.dq, err=self.data.err, debug=False)

        if debug:
            plt.subplots()
            vmin,vmax = np.nanquantile(self.data.data.flatten(),0.01),np.nanquantile(self.data.data.flatten(),0.99)
            plt.imshow(self.data.data,vmin=vmin,vmax=vmax)
            plt.show()

    def select_hot_pix_version3(self,filelist='',debug=False,fast_mode=True):
        band = self.data.meta.instrument.band
        channel = self.data.meta.instrument.channel
        listnames_source = []
        listnames_backgr = []
        data = filelist.data
        for s in data:
            s_band = s[4]
            s_ch = s[5]
            if s_band == band and s_ch == channel:
                print('name',s[3])
                if 'BACK' in s[3] and s[0].split('uncal')[0] != self.name:
                    listnames_backgr.append(s)
                elif 'BCK' in s[3] and s[0].split('uncal')[0] != self.name:
                    listnames_backgr.append(s)
                else:
                    listnames_source.append(s)
        ref_list = listnames_backgr

        ref_name = ref_list[0][0].replace('uncal', 'rate.fits')
        ref_file = os.path.join(self.path + '/' + ref_name)
        hdulist = fits.open(ref_file)
        ref_slope = hdulist['SCI'].data
        hdulist.close()

        # set mask of hot pixels
        bkgr_name = listnames_backgr[0][0].replace('uncal', 'rate.fits')
        bkgr_file = os.path.join(self.path + '/' + bkgr_name)
        hdulist = fits.open(bkgr_file)
        bkgr_slope = hdulist['SCI'].data
        hdulist.close()

        bkgr_mean = np.zeros_like(bkgr_slope)
        bkgr_std = np.zeros_like(bkgr_slope)

        bkgr_mean[500:,:500] = np.nanmean(bkgr_slope[500:,:500])
        bkgr_mean[:500, :500] = np.nanmean(bkgr_slope[:500, :500])
        print('bkgr_mean<500',np.nanmean(bkgr_slope[:500, :500]))

        bkgr_mean[:500, 500:] = np.nanmean(bkgr_slope[:500,500:])
        bkgr_mean[500:, 500:] = np.nanmean(bkgr_slope[500:, 500:])
        print('bkgr_mean>500',np.nanmean(bkgr_slope[500:, 500:]))

        bkgr_std[:500, :500] = np.nanstd(bkgr_slope[:500,:500])
        bkgr_std[500:, :500] = np.nanstd(bkgr_slope[500:, :500])
        print('bkgr_std<500', np.nanstd(bkgr_slope[500:, :500]))


        bkgr_std[:500, 500:] = np.nanstd(bkgr_slope[:500,500:])
        bkgr_std[500:, 500:] = np.nanstd(bkgr_slope[500:,500:])
        print('bkgr_std>500', np.nanstd(bkgr_slope[500:,500:]))

        mask_hot_pix_loc = np.abs(ref_slope - bkgr_mean)>1*bkgr_std
        mask_hot_pix_ref = np.abs(self.data.data - bkgr_mean)>1*bkgr_std
        mask_hot = mask_hot_pix_loc*mask_hot_pix_ref
        print('number of pixels to check:', np.sum(mask_hot))
        def check_nearby_pix(data,mask,sigma_limit=3):
            xi, yi = np.arange(self.data.data.shape[0]), np.arange(self.data.data.shape[1])
            Xi, Yi = np.meshgrid(yi, xi)
            arg = np.argwhere(mask == True)
            mask2 = mask.copy()
            for i in range(arg.shape[0]):
                x, y = arg[i, 0], arg[i, 1]
                if x<data.shape[0]-10 and x>10 and y<data.shape[1]-10 and y>10:
                    mask_loc = (Yi<x+5)*(Yi>x-5)*(Xi<y+5)*(Xi>y-5)
                    mask_loc[(Yi<x+2)*(Yi>x-2)*(Xi<y+2)*(Xi>y-2)] = False
                    data_loc = data[mask_loc].copy()
                    mean, std = np.nanmean(data_loc),np.nanstd(data_loc)
                    mask_loc = (Yi<x+5)*(Yi>x-5)*(Xi<y+5)*(Xi>y-5)
                    arg2 = np.argwhere(mask_loc==True)
                    for j in range(arg2.shape[0]):
                        if np.abs(data[arg2[j, 0], arg2[j, 1]]- mean)>sigma_limit*std:
                            mask2[arg2[j, 0], arg2[j, 1]] = True
            mask= mask2
            return mask

        if not fast_mode:
            m1 = check_nearby_pix(ref_slope,mask_hot)
            m2 = check_nearby_pix(self.data.data,mask_hot)
            mask_hot = m1*m2

        print('number of hot pixels:',np.sum(mask_hot))

        if debug:
            fig,ax = plt.subplots(1,5,sharex=True,sharey=True)
            vmin, vmax = np.nanquantile(self.data.data.flatten(), 0.01), np.nanquantile(self.data.data.flatten(), 0.99)
            ax[0].imshow(self.data.data,vmin=vmin,vmax=vmax)
            ax[0].set_title(self.name)
            ax[1].imshow(ref_slope,vmin=vmin,vmax=vmax)
            ax[1].set_title(ref_name)
            ax[2].imshow(mask_hot,vmin=vmin,vmax=vmax)
            ax[3].imshow(m2, vmin=vmin, vmax=vmax)


        #interpolation of flux in hot pixels

        def fix_hot_pix(data, mask, dq, err, debug=False):
            from scipy.interpolate import interp1d

            arg = np.argwhere(mask == True)
            sat_flag = dqflags.pixel["SATURATED"]
            dnu_flag = dqflags.pixel["DO_NOT_USE"]

            for i in range(arg.shape[0]):
                x, y = arg[i, 0], arg[i, 1]
                if x<data.shape[0]-10 and x>10:
                    y_loc = data[x - 10:x + 10, y].copy()
                    yerr_loc = err[x - 10:x + 10, y].copy()
                    dq_loc = dq[x - 10:x + 10, y].copy()
                    mask_loc = ~mask[x - 10:x + 10, y].copy()
                elif x<=10:
                    y_loc = data[0:x + 10, y].copy()
                    yerr_loc = err[0:x + 10, y].copy()
                    dq_loc = dq[0:x + 10, y].copy()
                    mask_loc = ~mask[0:x + 10, y].copy()
                elif (x>= data.shape[0]-10):
                    y_loc = data[x-10:, y].copy()
                    yerr_loc = err[x - 10:, y].copy()
                    dq_loc = dq[x - 10:, y].copy()
                    mask_loc = ~mask[x - 10:, y].copy()
                x_loc = np.arange(y_loc.shape[0])
                x_c = np.where(y_loc == data[x, y])[0][0]

                mask_loc *= (x_loc != x_c)  # *(x_loc!=4)*(x_loc!=6)
                mask_loc *= ~np.bitwise_and(dq_loc,sat_flag).astype(bool)
                mask_loc *= ~np.bitwise_and(dq_loc, dnu_flag).astype(bool)
                mask_loc *= ~np.isnan(y_loc)


                if np.sum(mask_loc)>2:
                    #mask_loc_bad_pix = np.zeros_like(mask_loc)
                    if 0:
                        mean_loc,std_loc = np.nanmean(y_loc[mask_loc]),np.nanstd(y_loc[mask_loc])
                        mask_loc_bad_pix = np.abs(y_loc - mean_loc)>5*std_loc
                        mask_loc *= ~mask_loc_bad_pix

                        if np.sum(mask_loc) > 2:
                            f_interp = interp1d(x_loc[mask_loc], y_loc[mask_loc], kind='quadratic',fill_value='extrapolate')
                            bad_pix_loc = np.where(mask_loc_bad_pix==True)[0]
                            for x_el in bad_pix_loc:
                                print(x+x_el-x_c, y, data[x+x_el-x_c, y], f_interp(x_el))
                                data[x+x_el-x_c, y] = f_interp(x_el)

                            print(x, y, y_loc)
                            print(f_interp(x_c))
                            if debug:
                                if len(bad_pix_loc)>1 and 0:
                                    xx = np.linspace(x_loc[mask_loc][0], x_loc[mask_loc][-1], 100)
                                    plt.subplots()
                                    plt.plot(x_loc, y_loc, 'o')
                                    plt.axhline(mean_loc)
                                    plt.axhline(mean_loc+5*std_loc, ls='--')
                                    plt.axhline(mean_loc-5*std_loc, ls='--')
                                    for x_el in bad_pix_loc:
                                        plt.plot(x_el, f_interp(x_el), 'D',color='orange')
                                    plt.plot(xx, f_interp(xx))
                                    plt.title(str(x)+str(y))
                                    plt.show()
                    if 1:
                        mean_loc, std_loc = np.nanmean(y_loc[mask_loc]), np.nanstd(y_loc[mask_loc])
                        mask_loc_bad_pix = np.abs(y_loc - mean_loc) > 5 * std_loc
                        mask_loc *= ~mask_loc_bad_pix

                        if np.sum(mask_loc) > 2:
                            f_interp = interp1d(x_loc[mask_loc], y_loc[mask_loc], kind='linear',
                                                fill_value='extrapolate')
                            bad_pix_loc = np.where(mask_loc_bad_pix == True)[0]
                            data[x, y] = f_interp(x_c)
                            if debug:
                                if f_interp(x_c)>mean_loc+2*std_loc:
                                    xx = np.linspace(x_loc[mask_loc][0], x_loc[mask_loc][-1], 100)
                                    plt.subplots()
                                    plt.plot(x_loc, y_loc, 'o')
                                    plt.errorbar(x=x_loc,y=y_loc,yerr=yerr_loc)
                                    plt.axhline(mean_loc)
                                    plt.axhline(mean_loc + 5 * std_loc, ls='--')
                                    plt.axhline(mean_loc - 5 * std_loc, ls='--')
                                    #for x_el in bad_pix_loc:
                                    #    plt.plot(x_el, f_interp(x_el), 'D', color='orange')
                                    plt.plot(x_c, f_interp(x_c), 'D', color='orange')
                                    plt.plot(xx, f_interp(xx))
                                    plt.title(str(x) + str(y))
                                    plt.show()

            return data

        self.data.data = fix_hot_pix(self.data.data, mask=mask_hot, dq=self.data.dq, err=self.data.err, debug=False)

        if debug:
            ax[4].imshow(self.data.data,vmin=vmin,vmax=vmax)
            plt.show()



    def create_background_model(self, filelist='', debug=False, smothing_rad=50,database='fringe_corr'):
        radius = 10
        band = self.data.meta.instrument.band
        channel = self.data.meta.instrument.channel
        listnames_source = []
        listnames_backgr = []
        data = filelist.data
        for s in data:
            s_band = s[4]
            s_ch = s[5]
            if s_band == band and s_ch == channel:
                if 'BACK' in s[3]:
                    listnames_backgr.append(s)
                elif 'BCK' in s[3]:
                    listnames_backgr.append(s)
                else:
                    listnames_source.append(s)

        print('listnames_backgr',listnames_backgr)
        print('listnames_source',listnames_source)

        bkgr_images = []
        bkgr_names = []
        bkgr_sigimages = []
        bkgr_dqs = []
        for el in listnames_backgr:
            if database == 'rate':
                bkgr_data = detector2(miri_uncal_file=el[0]+'.fits',path = self.path, output_dir=self.output_dir)
                bkgr_data.select_hot_pix_version2(filelist=filelist,debug=False)
            elif  database == 'fringe_corr':
                bkgr_data = detector2(miri_uncal_file=el[0]+'.fits',path = self.path, output_dir=self.output_dir)
                bkgr_data.read_res_fringes()
            elif  database == 'flat_fringe':
                bkgr_data = detector2(miri_uncal_file=el[0]+'.fits',path = self.path, output_dir=self.output_dir)
                bkgr_data.read_flat_fringes()
            bkgr_slope = bkgr_data.data.data
            bkgr_sig_slope = bkgr_data.data.err
            bkgr_dq = bkgr_data.data.dq
            bkgr_names.append(el[0].replace('uncal', ''))


            bkgr_images.append(bkgr_slope)
            bkgr_sigimages.append(bkgr_sig_slope)
            bkgr_dqs.append(bkgr_dq)
            del bkgr_dq,bkgr_slope,bkgr_sig_slope

        # find path to photom mask
        photom_list = sorted(glob.glob(os.environ["CRDS_PATH"] + '/references/jwst/miri/*photom*'))
        for f in photom_list:
            hdulist = fits.open(f)
            header = hdulist[0].header
            f_band, f_ch = header['BAND'], header['CHANNEl']
            if band == f_band and f_ch == channel:
                photom_file = f
                break
        from scripts.flat_field import get_mask
        photom_mask = get_mask(path=photom_file)

        from scripts.CRshowers import calc_mean_rate
        imtot, imtotsig = calc_mean_rate(images=bkgr_images, sig_images=bkgr_sigimages, dqs=bkgr_dqs, debug=debug,
                                         skip_cr_events=False, radius=radius,
                                         photom_mask=photom_mask, n_smooth_iters=3) #set to 1

        if debug:
            n_im = len(bkgr_images)
            fig, ax = plt.subplots(1, n_im+1, sharex=True, sharey=True)
            cmap = plt.cm.viridis
            cmap.set_bad('black')
            vmin,vmax = np.nanquantile(imtot.flatten(), 0.05), np.nanquantile(imtot.flatten(), 0.8)
            for i in range(n_im):
                ax[i].imshow(bkgr_images[i], vmin=vmin, vmax=vmax,origin='lower',cmap=cmap)
                ax[i].set_title(bkgr_names[i])
            ax[n_im].imshow(imtot, vmin=vmin, vmax=vmax,origin='lower',cmap=cmap)
            ax[n_im].set_title('MODEL')
            ax[n_im].imshow(imtot, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)

            plt.show()

        return imtot, imtotsig

    def subtract_bkgr_model(self, bkgr_model,bkgr_model_sig,savepdf=True):
        targ_name = self.data.meta.target.proposer_name
        print('targ name',targ_name)
        if 'BACKGROUND' in targ_name:
            self.data.data = bkgr_model
            self.data.err = bkgr_model_sig
            self.data.meta.background.subtracted = False
        elif 'BCK' in targ_name:
            self.data.data = bkgr_model
            self.data.err = bkgr_model_sig
            self.data.meta.background.subtracted = False
        else:
            print('subtract')
            data_tmp = np.array(self.data.data)
            nan_mask = ~np.isnan(bkgr_model)
            self.data.data[nan_mask] -= bkgr_model[nan_mask]
            self.data.err[nan_mask] = np.sqrt(np.power(self.data.err[nan_mask],2) + np.power(bkgr_model_sig[nan_mask],2))
            self.data.meta.background.method = 'STAGE2_PIX2PIX'
            self.data.meta.background.subtracted = True
            if savepdf:
                vmin, vmax = np.nanquantile(self.data.data.flatten(), 0.01), np.nanquantile(
                    self.data.data.flatten(), 0.99)
                fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
                ax[0].imshow(data_tmp, vmin=vmin, vmax=vmax)
                ax[0].set_title('Exposure')
                ax[1].imshow(bkgr_model, vmin=vmin, vmax=vmax)
                ax[1].set_title('Background Model')
                ax[2].imshow(self.data.data, vmin=vmin, vmax=vmax)
                ax[2].set_title('Bkgr subtracted')
                plt.savefig('./output/detector2/bkgr_subtracted/' + self.name + '.pdf', bbox_inches='tight',
                            dpi=2000)
            del(data_tmp)


    def mask_qso_traces(self, debug=False,q_ra= 135.3445,q_dec=20.746259,q_delta=3,save_pdf=True):
        targ_name = self.data.meta.target.proposer_name
        if 'BACKGROUND' not in targ_name:
            from JWST_cube_analyser import miri_psf_arcsec
            print('mask_qso_traces for', self.name)
            data = self.data.data
            wcs = self.data.meta.wcs
            cal_detector_to_world2 = wcs.get_transform('detector', 'world')

            #read photometry file
            if 1:
                band = self.data.meta.instrument.band
                channel = self.data.meta.instrument.channel

                photom_list = sorted(glob.glob(os.environ["CRDS_PATH"] + '/references/jwst/miri/*photom*'))
                for f in photom_list:
                    hdulist = fits.open(f)
                    header = hdulist[0].header
                    f_band, f_ch = header['BAND'], header['CHANNEl']
                    if band == f_band and f_ch == channel:
                        photom_file = f
                        break

                hdulist = fits.open(photom_file)
                hdu = hdulist[1].data
                hdulist.close()

                mask_flux = hdu.copy()
                mask_flux[np.isnan(mask_flux)] = 0
                mask_flux[mask_flux > 0] = 1
                del hdu

            # create mask for traces
            mask = np.zeros((data.shape[0], data.shape[1]))
            # define reference points for trace fitting
            tr_x = np.array([100, 250, 500, 750, 900])
            l1 = np.nanmean([cal_detector_to_world2(l, 500)[2] for l in np.arange(100, 400, 10)])
            miri_psf_fwhm1 = miri_psf_arcsec(l1)
            l2 = np.nanmean([cal_detector_to_world2(l, 500)[2] for l in np.arange(500, 800, 10)])
            miri_psf_fwhm2 = miri_psf_arcsec(l2)

            ntraces,itrace = 0,0
            for n,i in enumerate(tr_x):
                print(i)
                for j in np.arange(10, data.shape[0] - 10):
                    if mask_flux[i, j] > 0:
                        (x, y, l) = cal_detector_to_world2(j, i)
                        if ~np.isnan(x) and ~np.isnan(y):
                            if j < 500:
                                if np.sqrt((x - q_ra) ** 2 + (y - q_dec) ** 2) * 3600 < q_delta * miri_psf_fwhm1:
                                    mask[i, j] = 1
                            else:
                                if np.sqrt((x - q_ra) ** 2 + (y - q_dec) ** 2) * 3600 < q_delta * miri_psf_fwhm2:
                                    mask[i, j] = 1
                tmp = np.array([mask[i, e] - mask[i, e + 1] for e in np.arange(0, data.shape[1] - 1)])
                if np.sum(tmp == -1)>ntraces:
                    ntraces = np.sum(tmp == -1)
                    itrace = n
                print('detector traces:', ntraces, np.where(tmp == -1)[0])

            left_border = np.zeros((ntraces, len(tr_x)))
            right_border = np.zeros((ntraces, len(tr_x)))

            for k, i in enumerate(tr_x):
                tmp = np.array([mask[i, e] - mask[i, e + 1] for e in np.arange(0, data.shape[1] - 1)])
                s = np.where(tmp == -1)[0]
                print(s)
                tmp_traces = np.where(tmp == -1)[0]
                for ll,v in enumerate(tmp_traces):
                    left_border[ll, k] = v
                tmp_traces = np.where(tmp == 1)[0]
                for ll,v in enumerate(tmp_traces):
                    right_border[ll, k] = v

                #left_border[:, k] = np.where(tmp == -1)[0]
                #right_border[:, k] = np.where(tmp == 1)[0]

            for k in range(ntraces):
                mask_left = np.abs(left_border[k,:]-left_border[k,itrace])<20
                zl = np.polyfit(tr_x[mask_left], left_border[k, :][mask_left], 2)
                pl = np.poly1d(zl)

                mask_right = np.abs(right_border[k,:]-right_border[k,itrace])<20
                zr = np.polyfit(tr_x[mask_right], right_border[k, :][mask_right], 2)
                pr = np.poly1d(zr)
                for i in range(mask.shape[0]):
                    l, r = int(np.rint(pl(i))), int(np.rint(pr(i)))
                    mask[i, l:r] = 1

            if debug:
                data_copy = np.array(data)
            self.data.data[mask.astype(bool)] = -999

            if debug and save_pdf:
                fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
                ax[0].imshow(data_copy, vmin=-0.5, vmax=0.5)
                ax[1].imshow(data, vmin=-0.5, vmax=0.5)
                ax[0].set_title('Original')
                ax[1].set_title('with masked QSO pixels')
                plt.savefig('./output/detector2/masked_qso/'+self.name+'.pdf', bbox_inches='tight',dpi=1000)

    def compare_dither_images(self, debug=True,save_pdf=True):
        targ_name = self.data.meta.target.proposer_name
        if 'BACKGROUND' not in targ_name:
            band = self.data.meta.instrument.band
            channel = self.data.meta.instrument.channel
            dither_pos_number = self.data.meta.dither.position_number
            #filelist = sorted(glob.glob('./output/tmp/*__rate.fits'))
            filelist = sorted(glob.glob('./output/detector2/masked_qso/*.fits'))
            short_list = []
            for f in filelist:
                hdulist = fits.open(f)
                header = hdulist[0].header
                f_band, f_ch,f_name  = header['BAND'], header['CHANNEl'],header['TARGPROP']
                if band == f_band and f_ch == channel and 'BACK' not  in f_name and targ_name in f_name:
                    if 'jw02155003001_05101_00001_mirifulong' not in f:
                        short_list.append(f)
                hdulist.close()

            if len(short_list)>0:

                images = []
                names = []
                sigimages = []
                dqs = []
                print('list of exposures with the same detector settings')
                for k,el in enumerate(short_list):
                    print(k,el)
                    data_tmp = datamodels.open(el)
                    print(data_tmp.meta.target.proposer_name,
                          data_tmp.meta.instrument.channel,data_tmp.meta.instrument.band,data_tmp.meta.dither.position_number)
                    #mask traces
                    mask_traces = data_tmp.data == -999
                    data_tmp.data[mask_traces] = np.nan
                    data_tmp.err[mask_traces] = np.nan
                    data_tmp.dq[mask_traces] = 1
                    #add to the sample
                    images.append(data_tmp.data)
                    sigimages.append(data_tmp.err)
                    dqs.append(data_tmp.dq)
                    names.append(el.split('/')[-1])
                    if dither_pos_number == data_tmp.meta.dither.position_number:
                        dither_exposure_number = k
                    del data_tmp

                def calc_mean_rate(images, sig_images, dqs, debug=False, radius=10, hot_pix_limit=4, skip_cr_events=True):
                    n_int = len(images)
                    im = np.array([images[i] for i in range(n_int)])
                    sigim = np.array([sig_images[i] for i in range(n_int)])
                    dqim = np.array([dqs[i] for i in range(n_int)])
                    mask_im = np.ones((n_int, im.shape[1], im.shape[2]))

                    #mask pixels in traces
                    for i in range(n_int):
                        mask_im[i][np.isnan(im[i])] = 0

                    if skip_cr_events:
                        mask_cr = np.zeros_like(im)
                        for i in range(n_int):
                            mask_cr[i] = (np.bitwise_and(dqim[i], dqflags.pixel['JUMP_DET'])).astype(bool)
                            mask_im[i][mask_cr[i] == 1] = 0
                        for i in range(n_int):
                            mask_im[i][np.sum(mask_cr, axis=0) == n_int] = 1
                            im[i][mask_im[i]==0] = np.nan
                    #mean_im = np.nansum(im * mask_im, axis=0) / np.sum(mask_im, axis=0)
                    mean_im = np.nanmedian(im, axis=0)
                    #mask_im = np.ones((n_int, im.shape[1], im.shape[2]))

                    # set kernel
                    filter_kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
                    for i in range(filter_kernel.shape[0]):
                        for j in range(filter_kernel.shape[1]):
                            if (i - radius) ** 2 + (j - radius) ** 2 <= radius ** 2:
                                filter_kernel[i, j] = 1

                    if debug:
                        fig, ax = plt.subplots(4, n_int + 1, sharex=True, sharey=True)
                    for i in range(n_int):
                        x = np.array(im[i] / mean_im - 1)
                        x[np.isnan(x)] = 0
                        x[x > hot_pix_limit] = 0
                        x[x < -hot_pix_limit] = 0

                        # smooth diff
                        npix = scipy.signal.convolve2d(np.ones_like(x), filter_kernel, mode='same', boundary='fill',   fillvalue=0)
                        x_smoothed = scipy.signal.convolve2d(x, filter_kernel, mode='same', boundary='fill', fillvalue=0) / npix
                        #mask_cr_showers = x_smoothed > 0.1
                        #mask_im[i][np.sum(mask_im, axis=0) == 0] = 1
                        mask_im[i][x_smoothed > 0.1] = 0
                        if debug:
                            vmin, vmax = np.nanquantile(mean_im.flatten(), 0.05), np.nanquantile(mean_im.flatten(), 0.8)
                            ax[0, i].imshow(im[i], vmin=vmin, vmax=vmax)
                            ax[0, i].set_title('Image '+str(i))
                            ax[1, i].imshow(x_smoothed, vmin=-0.3, vmax=0.3)
                            ax[1, i].set_title('Smoothed model' + str(i))
                            xi, yi = np.arange(x_smoothed.shape[1]), np.arange(x_smoothed.shape[0])
                            zi = x_smoothed
                            ax[1, i].contour(xi, yi, zi, levels=[-0.1, 0.1], linewidths=0.5, colors='k')


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

                            #mask_im[i][x_smoothed < 0.1] *= 1
                    if debug:
                        for i in range(n_int):
                            ax[2, i].imshow(mask_im[i])
                            ax[2, i].set_title('Mask for images '+str(i))

                            ax[3, i].imshow(mask_cr[i])
                            ax[3, i].set_title('Mask for CR '+str(i))


                    imsig_inv = np.power(sigim, -2)
                    #imtot = np.nansum(im * imsig_inv * mask_im, axis=0) / np.nansum(imsig_inv * mask_im, axis=0)
                    for i in range(n_int):
                        im[i][mask_im[i]==0] = np.nan
                    imtot = np.nanmedian(im, axis=0)
                    imtotsig = np.power(np.nansum(imsig_inv * mask_im, axis=0), -0.5)

                    if debug:
                        ax[0, n_int].imshow(mean_im, vmin=vmin, vmax=vmax)
                        ax[0, n_int].set_title('Median')
                        ax[1, n_int].imshow(imtot, vmin=vmin, vmax=vmax)
                        ax[1, n_int].set_title('Final')
                        # ax[2, n_int].imshow(imtot, vmin=vmin, vmax=vmax)
                        plt.show()

                    return imtot, imtotsig

                smothing_rad = 10
                imtot, imtotsig = calc_mean_rate(images=images,sig_images=sigimages,dqs=dqs,debug=debug,radius=smothing_rad)

                if debug:
                    n_im = len(images)
                    fig, ax = plt.subplots(1, n_im + 1, sharex=True, sharey=True)
                    fig2, ax2 = plt.subplots(1, n_im + 1, sharex=True, sharey=True)
                    vmin, vmax = np.nanquantile(imtot.flatten(), 0.05), np.nanquantile(imtot.flatten(), 0.8)
                    for i in range(n_im):
                        ax[i].imshow(images[i], vmin=vmin, vmax=vmax)
                        ax2[i].plot(np.nanmean(images[i][:,10:(imtot.shape[1]-10)],axis=1))
                        ax2[i].plot(np.nanmedian(images[i][:,10:(imtot.shape[1]-10)], axis=1))
                        ax2[i].axhline(0, ls='--', color='red')
                        ax[i].set_title(names[i])
                    ax[n_im].imshow(imtot, vmin=vmin, vmax=vmax)
                    ax[n_im].set_title('MODEL')
                    ax2[n_im].plot(np.nanmean(imtot[:,10:(imtot.shape[1]-10)],axis=1))
                    ax2[n_im].plot(np.nanmedian(imtot[:,10:(imtot.shape[1]-10)],axis=1))
                    ax2[n_im].axhline(0,ls='--',color='red')
                    #ax[n_im].plot(100,300,'o',color='red',markersize=20)
                    fig,ax = plt.subplots()
                    for i in range(imtot.shape[0]):
                        s1 = np.nanmean(imtot[i,10:(imtot.shape[1]-10)])
                        s2 = np.nanmedian(imtot[i,10:(imtot.shape[1]-10)])
                        if np.abs(s1-s2)>0.05:
                            ax.plot(imtot[i,:],label=str(i))
                    ax.legend()
                    plt.show()

                for f in filelist:
                    if self.name in f:
                        break
                print('open datamodel: ', self.name, f)
                data_tmp = datamodels.open(f)
                mask_traces = data_tmp.data == -999
                mask_nan = np.isnan(imtot)
                del data_tmp
                data_tmp = np.array(self.data.data)

                self.data.data[~mask_traces*~mask_nan] = imtot[~mask_traces*~mask_nan]
                self.data.err[~mask_traces*~mask_nan] = imtotsig[~mask_traces*~mask_nan]

                if save_pdf:
                    fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
                    vmin, vmax = np.nanquantile(imtot.flatten(), 0.05), np.nanquantile(imtot.flatten(), 0.95)
                    ax[0].imshow(mask_traces)
                    ax[0].set_title('mask qso traces')
                    ax[1].imshow(data_tmp,vmin=vmin,vmax=vmax)
                    ax[1].set_title('orginal')
                    ax[2].imshow(self.data.data,vmin=vmin,vmax=vmax)
                    ax[2].set_title('CR cleaned')
                    plt.savefig('./output/detector2/cleaned_img/'+self.name+'.pdf',dpi=2000,bbox_inches='tight')
                    plt.close()



    def fix_hot_pix_step(self, debug=False):
        '''
        Mask hot pipxels as bad.
        '''

        data = self.data.data
        print(data.shape)
        dq = self.data.dq.copy()

        threshold_pix_val = 1


        def check_hot_pix(data,dq,limit):
            mask = data>limit
            arg = np.argwhere(data>limit)
            for i in range(arg.shape[0]):
                x,y = arg[i,0],arg[i,1]
                if x<data.shape[0]-6 and y < data.shape[1]-6 and x>6 and y>6:
                    #print(x,y)
                    f = data[x,y]
                    flux_mean_1,npix = 0,0
                    for j,k in zip([x-2,x-3,x-4,x-5,x-6,x+4,x+5,x+6,x+3,x+2],[y,y,y,y,y,y,y,y,y,y]):
                        if dq[j, k] in [0, 4] and ~np.isnan(data[j, k]):
                            flux_mean_1 += data[j,k]
                            npix +=1
                    if npix>0:
                        flux_mean_1/=npix


                        flux_mean2,npix,delta,ar = 0,0,[],[[],[]]
                        for j, k in zip([x-2,x-3,x-4,x-5,x-6,x+4,x+5,x+6,x+3,x+2],[y,y,y,y,y,y,y,y,y,y]):
                            if dq[j, k] in [0, 4] and data[j, k]<2*flux_mean_1:
                                flux_mean2 += data[j, k]
                                npix += 1
                                delta.append(data[j, k])
                            ar[0].append(data[j, k])
                            ar[1].append(dq[j, k])
                        if npix > 0:
                            flux_mean2 /= npix

                        flux_mean=flux_mean2
                        f = data[x,y]
                        flux_disp = np.sqrt(np.sum(np.power(np.array(delta)-flux_mean,2))/npix)
                        if (np.abs(data[x,y]-flux_mean)<7*flux_disp)+(f<2*flux_mean):
                            mask[x,y] = False
                        elif mask[x,y]:
                            if f > 2 and x<497 and x>487 and y>763 and y<767:
                                print()
                        if np.bitwise_and(dq[x,y],1) and 0:
                            d = dq[x,y]
                            mask[x, y] = False
                        elif np.bitwise_and(dq[x,y],4) and 0:
                            d = dq[x,y]
                            mask[x, y] = False
                    else:
                        mask[x, y] = False

            return mask
        mask_hot_pix = check_hot_pix(data,dq,threshold_pix_val)
        self.data.dq = np.bitwise_or(self.data.dq, mask_hot_pix)
        self.data.data[mask_hot_pix.astype(bool)] = np.nan

        if debug:
            fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
            vmin,vmax = 0,threshold_pix_val
            ax[1].imshow(mask_hot_pix.astype(int), vmin=0, vmax=1)
            ax[0].imshow(data, vmin=-3, vmax=3)
            ax[2].imshow(dq.astype(int), vmin=0, vmax=1)
            print('N of hot_pixels_list', np.sum(mask_hot_pix))
            plt.show()

    def fix_cold_pix_step(self, debug=False):
        '''
        Mask hot pipxels as bad.
        '''



        data = self.data.data.copy()
        print(data.shape)
        dq = self.data.dq.copy()

        threshold_pix_val = 1


        def check_hot_pix(data,dq,limit):
            mask = data<limit
            arg = np.argwhere(data<limit)
            for i in range(arg.shape[0]):
                x,y = arg[i,0],arg[i,1]
                if x<data.shape[0]-6 and y < data.shape[1]-6 and x>6 and y>6:
                    #print(x,y)
                    f = data[x,y]
                    flux_mean_1,npix = 0,0
                    for j,k in zip([x-2,x-3,x-4,x-5,x-6,x+4,x+5,x+6,x+3,x+2],[y,y,y,y,y,y,y,y,y,y]):
                        if dq[j,k] in [0,4] and ~np.isnan(data[j,k]):
                            flux_mean_1 += data[j,k]
                            npix +=1
                    if npix>0:
                        flux_mean_1/=npix


                        flux_mean2,npix,delta,ar = 0,0,[],[[],[]]
                        for j, k in zip([x-2,x-3,x-4,x-5,x-6,x+4,x+5,x+6,x+3,x+2],[y,y,y,y,y,y,y,y,y,y]):
                            if dq[j,k] in [0,4] and data[j, k]>2*flux_mean_1 and ~np.isnan(data[j,k]):
                                flux_mean2 += data[j, k]
                                npix += 1
                                delta.append(data[j, k])
                            ar[0].append(data[j, k])
                            ar[1].append(dq[j, k])
                        if npix > 0:
                            flux_mean2 /= npix

                        flux_mean=flux_mean2
                        f = data[x,y]
                        flux_disp = np.sqrt(np.sum(np.power(np.array(delta)-flux_mean,2))/npix)
                        if (np.abs(data[x,y]-flux_mean)<5*flux_disp)+(f>2*flux_mean):
                            mask[x,y] = False
                        elif mask[x,y]:
                            if f > 2 and x<945 and x>935 and y>590 and y<600:
                                print('test',x,y,ar)
                        if np.bitwise_and(dq[x,y],1) and 0:
                            d = dq[x,y]
                            mask[x, y] = False
                        elif np.bitwise_and(dq[x,y],4) and 0:
                            d = dq[x,y]
                            mask[x, y] = False
                    else:
                        mask[x, y] = False

            return mask

        threshold_pix_val = np.nanmean(data) - 5*np.nanstd(data)
        print('cold pix limit:',threshold_pix_val)
        mask_hot_pix = check_hot_pix(data,dq,threshold_pix_val)
        self.data.dq = np.bitwise_or(self.data.dq, mask_hot_pix)
        self.data.data[mask_hot_pix.astype(bool)] = np.nan

        if debug:
            fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
            vmin,vmax = 0,threshold_pix_val
            ax[1].imshow(mask_hot_pix.astype(int), vmin=0, vmax=1)
            ax[0].imshow(data, vmin=-3, vmax=3)
            ax[2].imshow(dq.astype(int), vmin=0, vmax=1)
            print('N of cold_pixels_list', np.sum(mask_hot_pix))
            plt.show()


    def assignwcsstep(self, input_file=None, debug=False, output_dir=None,save_results=False):
        '''
        This step populates the Data Quality (DQ) mask that is associated with the data file.
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.rate_file

        wcs_step = AssignWcsStep()
        wcs_step.output_dir = output_dir
        wcs_step.save_results = save_results

        # Call the run() method on the uncal file
        self.data = wcs_step.run(input_file)

        sstring = self.output_dir + '/' + '*assignwcsstep.fits'
        wcsfiles = sorted(glob.glob(sstring))
        self.wcs_files = []
        for f in wcsfiles:
            if self.name in f:
                self.wcs_files.append(f)

        print('WCS step done')

    # direct subtraction of bkg_exp from sci_exp
    def background_subtraction(self, input_file=None, debug=True, output_dir=None,save_results=False):
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data


        input_sci_exp_list = []
        input_bkg_exp_list = []

        sstring = output_dir + '/' + '*assignwcsstep.fits'
        files = sorted(glob.glob(sstring))
        for f in files:
            if self.name in f:
                input_sci_exp_list.append(f)

        bkgr_sbtr_step = BackgroundStep()
        bkgr_sbtr_step.output_dir = output_dir
        bkgr_sbtr_step.save_results = save_results

        bgfiles= bkgr_sbtr_step(input_sci_exp_list,input_bkg_exp_list)


    # This step divides the array data by the pixel flatfield reference file. (for MIRI MRS the reference flatfield is currently unity everywhere )
    def flat_field_step(self, input_file=None, debug=False, output_dir=None,save_results=False):
        '''
        This step populates the Data Quality (DQ) mask that is associated with the data file.
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data


        flat_field_step = FlatFieldStep()
        flat_field_step.output_dir = output_dir
        flat_field_step.save_results = save_results
        data_orig = self.data.data.copy()
        # Call the run() method on the uncal file
        self.data = flat_field_step.run(input_file)
        print('FLAT FIELD step done')
        if debug:
            vmin,vmax = np.nanquantile(data_orig.flatten(),0.01),np.nanquantile(data_orig.flatten(),0.99)
            fig,ax = plt.subplots(1,3,sharey=True,sharex=True)
            cmap = plt.cm.viridis
            cmap.set_bad('black')
            ax[0].imshow(data_orig,vmin=vmin,vmax=vmax,cmap=cmap,origin='lower')
            ax[1].imshow(self.data.data,vmin=vmin,vmax=vmax,cmap=cmap,origin='lower')
            flatfile = self.data.meta.ref_file.flat.name.split('//')[-1]
            hdu = fits.open(os.environ["CRDS_PATH"] +'/references/jwst/miri/' + flatfile)
            ax[2].imshow(hdu['SCI'].data, vmin=0.95, vmax=1.05,cmap=cmap,origin='lower')
            hdu.close()
            plt.show()
    def source_type_identification(self, input_file=None, debug=True, output_dir=None,save_results=False):
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        source_type_id_step = SourceTypeStep()
        source_type_id_step.output_dir = output_dir
        source_type_id_step.save_results = save_results

        # Call using the output from thSource_identificatione previously-run dq_init step
        self.data = source_type_id_step.run(input_file)
        if debug:
            print('Source type:', self.data.meta.target.source_type)
            print('Source identification step: Done.')

    def stray_light_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
        '''
        The MIRI MRS has been observed to have appreciable straylight at short wavelengths in ground-test data,
        and this step is therefore designed to model and subtract this component from the detector data
        using the small interstitial regions of detector pixels between the illuminated slices.

        As such, we usually recommend skipping this step when working with simulated data. However, we'll run it here just to see what it looks like.
        See https://jwst-pipeline.readthedocs.io/en/latest/jwst/straylight/index.html
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        stray_ligth_step = StraylightStep()
        stray_ligth_step.output_dir = output_dir
        stray_ligth_step.save_results = save_results

        if debug:
            data_orig = self.data.data.copy()
        # Call using the the output from the previously-run dq_init step
        self.data = stray_ligth_step.run(input_file)
        # Call the run() method on the uncal file
        print('STRAY LIGHT step done')
        if debug:
            vmin, vmax = np.nanquantile(data_orig.flatten(), 0.01), np.nanquantile(data_orig.flatten(), 0.99)
            fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
            ax[0].imshow(data_orig, vmin=vmin, vmax=vmax)
            ax[0].set_title('before')
            ax[1].imshow(self.data.data, vmin=vmin, vmax=vmax)
            ax[1].set_title('after')
            plt.show()

    def fringe_flat_step(self, input_file=None, debug=False, output_dir=None,save_results=False):
        '''
          This crucial step is the first pipeline correction for the strong periodic amplitude modulation (i.e., fringing)
          that occurs in the MIRI detectors due to internal reflections within the detectors.
          In this step, the pipeline simply divides by a reference fringe flatfield to make a first-order correction to the data.
          In detail, the fringing signal will depend on the geometry of sources within the scene, and thus there will be a residual
           fringe correction later in the pipeline too.  See https://jwst-pipeline.readthedocs.io/en/latest/jwst/fringe/index.html
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        if debug:
            data_orig = self.data.data.copy()

        fringe_flat_step = FringeStep()
        fringe_flat_step.output_dir = output_dir
        fringe_flat_step.save_results = save_results

        # Call using the the output from the previously-run dq_init step
        self.data = fringe_flat_step.run(input_file)
        if debug:
            print('FRINGE STEP: Done.')
            vmin, vmax = np.nanquantile(data_orig.flatten(), 0.01), np.nanquantile(data_orig.flatten(), 0.99)

            fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
            ax[0].imshow(data_orig, vmin=vmin, vmax=vmax)
            ax[0].set_title('before')
            ax[1].imshow(self.data.data, vmin=vmin, vmax=vmax)
            ax[1].set_title('after')
            plt.show()

    def res_fringe_step(self, input_file=None, debug=False, output_dir=None,save_results=False, rename=False):
        '''
          For spatially unresolved (point) sources or extended sources with structure, applying the fringe flat will undoubtedly leave
          residual fringes since these produce different fringe patterns on the detector than accounted for by the fringe flat.
          The second step for fringe removal is the residual_fringe_step. This step is part of the calwebb_spec2 pipeline,
          but currently it is skipped by default. To apply this step set the step parameter, --skip = False.
          This step is  applied after photom, but before cube_build
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data #calfiles

        res_fringe_flat_step = ResidualFringeStep()
        res_fringe_flat_step.skip = False
        #res_fringe_flat_step.ignore_region_min = []
        #res_fringe_flat_step.ignore_region_max = []
        res_fringe_flat_step.output_dir = output_dir+'residual_fringe/'
        res_fringe_flat_step.save_results = save_results

        if debug:
            data_orig = self.data.data.copy()


        # Call using the the output from the previously-run dq_init step
        self.data = res_fringe_flat_step.run(input_file)
        if debug:
            print('RES FRINGE STEP: Done.')
            vmin, vmax = np.nanquantile(data_orig.flatten(), 0.01), np.nanquantile(data_orig.flatten(), 0.99)

            fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
            ax[0].imshow(data_orig, vmin=vmin, vmax=vmax)
            ax[0].set_title('before')
            ax[1].imshow(self.data.data, vmin=vmin, vmax=vmax)
            ax[1].set_title('after (Res Fringe)')
            plt.show()



        if rename:
            # Rename residual_fringe to cal files
            # Look for our _residual_fringe.fits files produced by the photometric calibration step
            sstring = self.output_dir + self.name +  '*residual_fringe.fits'
            self.residual_fringe = sorted(glob.glob(sstring))
            # And print them out so that we can see them
            self.calfiles = self.residual_fringe.copy()
            for ii in range(0, len(self.residual_fringe)):
                self.calfiles[ii] = str.replace(self.residual_fringe[ii], 'residual_fringe', 'cal')
                os.renames(self.residual_fringe[ii], self.calfiles[ii])



        if debug:
            print('RES FRINGE STEP: Done.')

    def read_flat_fringes(self):
        updated = False
        input_dir = self.output_dir + '*fringestep.fits'
        obs_id = self.data.meta.observation.obs_id
        print('read obs_id', obs_id)
        filelist = sorted(glob.glob(input_dir))
        for f in filelist:
            hdulist = fits.open(f)
            header = hdulist[0].header
            f_obs_id = header['OBS_ID']
            if obs_id == f_obs_id and self.name in f:
                print('read from', f)
                self.update_data_file(input_file=f)
                updated = True
                break
        return updated

    def read_res_fringes(self):
        updated = False
        input_dir = self.output_dir + 'residual_fringe/*.fits'
        obs_id = self.data.meta.observation.obs_id

        filelist = sorted(glob.glob(input_dir))
        for f in filelist:
            hdulist = fits.open(f)
            header = hdulist[0].header
            f_obs_id = header['OBS_ID']
            if obs_id == f_obs_id and self.name in f:
                print('read from', f)
                self.update_data_file(input_file=f)
                updated = True
                break
        return updated

    def read_bkgr_subtracted(self):
        updated = False
        input_dir = self.output_dir + 'bkgr_subtracted/*.fits'
        obs_id = self.data.meta.observation.obs_id

        filelist = sorted(glob.glob(input_dir))
        for f in filelist:
            hdulist = fits.open(f)
            header = hdulist[0].header
            f_obs_id = header['OBS_ID']
            if obs_id == f_obs_id and self.name in f:
                print('filename', f, 'self.name ', self.data.meta.filename)
                self.update_data_file(input_file=f)
                updated = True
                break
        return updated


    def flux_calibration_step(self, input_file=None, debug=False, output_dir=None,save_results=False,rename=True):
        '''
        Correction of science data values for detector non-linearity.
        The correction is represented by an nth-order polynomial for each pixel in the detector (not selected as "NO_LIN_CORRECTION" or "SATURATED"),
        with n+1 arrays of coefficients read from the linearity reference file.
        Add  “NO_LIN_CORR” to PIXELDQ array for pixels with NAN in linear coefficients
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        photom_step = PhotomStep()
        photom_step.output_dir = output_dir
        photom_step.save_results = save_results

        # Call using the the output from the previously-run dq_init step
        self.data = photom_step.run(input_file)

        if rename:
            # Rename residual_fringe to cal files
            # Look for our _residual_fringe.fits files produced by the photometric calibration step
            sstring = self.output_dir + self.name + '*photomstep.fits'
            self.photom = sorted(glob.glob(sstring))
            # And print them out so that we can see them
            self.calfiles = self.photom.copy()
            for ii in range(0, len(self.photom)):
                self.calfiles[ii] = str.replace(self.photom[ii], 'photomstep', 'cal')
                os.renames(self.photom[ii], self.calfiles[ii])
                # example_file = fits.open(self.residual_fringe[ii])
                # example_file.writeto(self.calfiles[ii], overwrite=True)
                # example_file.close()

        if debug:
            print('PHOTOM step: Done.')

    def show_trace(self,trace_order=0,save_txt=True,trace_size=0):
        band = self.data.meta.instrument.band
        channel = self.data.meta.instrument.channel

        photom_list = sorted(glob.glob(os.environ["CRDS_PATH"] +'/references/jwst/miri/*photom*'))
        for f in photom_list:
            hdulist = fits.open(f)
            header = hdulist[0].header
            f_band,f_ch = header['BAND'],header['CHANNEl']
            if band == f_band and f_ch == channel:
                photom_file = f
                break
        from scripts.flat_field import get_trace_mask
        trace_mask = get_trace_mask(path=photom_file)

        data = self.data.data.copy()
        err = self.data.err.copy()

        fig, ax = plt.subplots()
        vmin,vmax = np.nanquantile(data.flatten(), 0.05), np.nanquantile(data.flatten(), 0.95)
        ax.imshow(data,vmin=vmin,vmax=vmax)


        fig2,ax2 = plt.subplots(2,1,sharex=True,figsize=(9,4))
        data[np.isinf(data)] = 0
        nrows = data.shape[0]
        trace = np.zeros(nrows)
        trace_err = np.zeros(nrows)
        mask = np.zeros_like(data)
        for i in range(nrows):
            j_min,j_max = int(trace_mask[trace_order,i,0]),int(trace_mask[trace_order,i,1])
            if trace_size != 0:
                jmean = int((j_min+j_max)/2)
                delta = int((trace_size-1)/2)
                j_min = jmean -delta
                j_max = jmean + delta
                print(jmean,j_min,j_max)
            trace[i] = np.nansum(data[i, j_min:j_max])
            trace_err[i] = np.sqrt(np.nansum(np.power(err[i, j_min:j_max], 2)))
            mask[i, j_min:j_max] = 1

        ax.plot(trace_mask[trace_order,:,0],np.arange(nrows),color='red',lw=2)
        ax.plot(trace_mask[trace_order,:,1],np.arange(nrows),color='red',lw=2)
        #for n in range(trace_mask.shape[0]):
        #    ax.plot(trace_mask[n, :, 0], np.arange(nrows), color='pink', lw=1)
        #    ax.plot(trace_mask[n, :, 1], np.arange(nrows), color='pink', lw=1)

        ax2[0].errorbar(y=trace,x=np.arange(nrows),yerr=trace_err)
        ax2[0].plot(np.arange(nrows),trace,color='black')

        mask = mask.astype(bool)
        data[~mask] = 0

        x = trace_mask[trace_order,:,0]
        min_val = int(np.min(x[x!=0]))
        x = trace_mask[trace_order, :, 1]
        max_val = int(np.max(x[x != 0]))
        ax2[1].imshow(np.transpose(data[:,min_val:
                                          max_val]),vmin=vmin,vmax=vmax)
        if save_txt:
            a = np.zeros((nrows,3))
            a[:,0]=np.arange(nrows)
            a[:,1] = trace
            a[:,2] = trace_err
            np.savetxt('./temp/trace.dat',a,fmt='%10.5e')

        plt.show()


    def show_Xtrace(self,trace_Xpos=100,delta=10):
        data = self.data.data.copy()

        yup,ylow = np.min([trace_Xpos+delta, data.shape[1]-10]),np.max([trace_Xpos-delta, 10])
        flux = np.nanmean(data[ylow:yup,:],axis=0)


        fig, ax = plt.subplots(1,2)
        vmin,vmax = np.nanquantile(data.flatten(), 0.05), np.nanquantile(data.flatten(), 0.95)
        ax[0].imshow(data[ylow:yup,:],vmin=vmin,vmax=vmax)
        ax[0].set_title(self.name)
        ax[1].plot(flux)
        plt.show()


    def cube_building_step(self, input_file=None, debug=True, output_dir=None,redolong=True):
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        if self.calfiles is not None:
            if (redolong is True):
                cube_build_step = CubeBuildStep()
                cube_build_step.output_dir = output_dir
                cube_build_step.save_results = True
                cube_build_step.output_type = 'multi'
                self.cube_data = cube_build_step.run(input_file)

            # Otherwise, just copy cached outputs into our output directory structure
            else:
                sstring = os.path.join(self.spec2_cachedir, 'det*s3d.fits')
                files = sorted(glob.glob(sstring))
                for file in files:
                    outfile = str.replace(file, self.spec2_cachedir, self.output_dir)
                    shutil.copy(file, outfile)

    def spec1d_extraction_step(self, input_file=None, debug=True, output_dir=None,redolong=True):
            if output_dir == None:
                output_dir = self.output_dir
            if input_file == None:
                input_file = self.data

            sstring = output_dir + self.name + '*s3d.fits'
            if len(sstring)>0:
                cubefiles = sorted(glob.glob(sstring))
                extract_spec_step =  Extract1dStep
                extract_spec_step.output_dir = output_dir
                extract_spec_step.save_results = True
                for file in cubefiles:
                    extract_spec_step.run(file)

    def show_dq(self, ax=None, flag = 'DO_NOT_USE'):
        self.pixdq_x =  np.zeros_like(self.data.dq)
        self.pixdq_y = np.zeros_like(self.data.dq)
        for i in range(self.data.dq.shape[1]):
            self.pixdq_x[:, i] = i
        for i in range(self.data.dq.shape[0]):
            self.pixdq_y[i, :] = i
        self.dq_types = [*dqflags.pixel]
        self.dq_colors = np.array(['white','blue','yellow','red','black', 'cyan','purple','tab:red','tab:blue','tab:green','tab:orange'])
        self.dq_bins_titles =[*dqflags.pixel.values()]

        if ax == None:
            fig, ax = plt.subplots(figsize=(20,20))
        jc =0
        tot_num = 0
        if flag == 'all':
            for k, val in enumerate(self.dq_bins_titles):
                if k>0 and self.dq_types[k] != 'UNRELIABLE_SLOPE' and self.dq_types[k] != 'DO_NOT_USE':
                    mask = np.bitwise_and(self.data.dq, val) == val
                    if np.sum(mask)>0:
                        jc+=1
                        tot_num+=np.sum(mask)
                        if jc> self.dq_colors.shape[0]:
                            break
                        ax.scatter(self.pixdq_x[mask], self.pixdq_y[mask], s=20, alpha=1, c=self.dq_colors[jc], label=self.dq_types[k])
                        print(self.dq_types[k], ':', np.sum(mask), self.dq_colors[jc])

        elif flag!= 'GOOD':
            mask = np.where(np.bitwise_and(self.data.dq, dqflags.pixel[flag]))
            if np.sum(mask)>0:
                tot_num+=np.size(mask[0])
                ax.scatter(self.pixdq_x[mask], self.pixdq_y[mask], s=20, alpha=1, c='black', label=flag)
        else:
            mask = self.data.dq == 0
            if np.sum(mask) > 0:
                tot_num += np.size(mask[0])
                ax.scatter(self.pixdq_x[mask], self.pixdq_y[mask], s=20, alpha=1, c='black', label=flag)
        print('The number of selected pixels:',tot_num) #, '  ',np.bitwise_and(self.data.pixeldq, 2).sum())
        ax.legend(fontsize=20,loc='lower left')
        ax.fill_between(x=[0,self.data.dq.shape[0]], y1 = [0,0],y2=[self.data.dq.shape[1],self.data.dq.shape[1]],color='tab:green',alpha=0.1)
        plt.show()



if __name__ == '__main__':
    print('Hi PyCharm')
    miri_uncal_file = 'jw02155001001_04102_00001_mirifulong_uncal.fits'
    input_dir = './output/tmp'
    spec2_cachedir = './temp/spec2/'
    exp1 = detector2(miri_uncal_file=miri_uncal_file, path = input_dir, output_dir=output_dir,spec2_cachedir=spec2_cachedir)
    #input_file =  'jw02155001001_04102_00001_mirifulong_rate.fits' #N of hot pix 983  of  1051
    input_file = 'jw02155001001_04102_00004_mirifulong_rate.fits'  # N of hot pix 983  of  1051
    exp1.create_mask_hot_pix_step(input_file='jw02155009001_02101_00001_mirifulong_rate.fits',ref_file = "jw02155016001_02103_00001_mirifulong_rate.fits",
                          dither_file='jw02155001001_04102_00002_mirifulong_rate.fits',save_file=False)
    #exp1.show_hot_pix_maps()
    exp1.compare_maps()
    exp1.fix_cold_pix_step(debug=True)
    plt.show()
    exp1.assignwcsstep()
    exp1.flat_field_step()
    exp1.source_type_identification()
    exp1.stray_light_step()
    exp1.fringe_flat_step()
    exp1.flux_calibration_step()
    exp1.cube_building_step()
    #exp1.spec1d_extraction_step()




