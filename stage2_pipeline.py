# Basic system utilities for interacting with files
import glob
import sys
#Modify the path to a directory on your machine
import os
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

#define input/output
#output_dir = './output/detector2/'
#input_dir = './output/results'
#miri_uncal_file= 'jw02155001001_04102_00001_mirifulong_uncal.fits'
#input_file_base = os.path.basename(miri_uncal_file).replace('uncal.fits', '')
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
    return init_settings
settings =  read_settings()
os.environ["CRDS_PATH"] = settings['CRDS_PATH']
os.environ["CRDS_SERVER_URL"] = settings['CRDS_SERVER_URL']
output_dir = settings['output2_dir'] #')./output/detector1/'
input_dir = settings['input2_dir'] #./input/detector1/'


class detector2():
    def __init__(self,miri_uncal_file=None,path=None, output_dir=None,spec2_cachedir=None):
        self.name = os.path.basename(miri_uncal_file).replace('uncal.fits', '')
        self.path = path
        self.output_dir=output_dir
        self.data = None
        self.rate_file = None
        self.spec2_cachedir = spec2_cachedir
        self.read_ratefiles(det1_dir=self.path,input_file_base=self.name)
        if self.rate_file != None:
            self.init_rate_files(self.rate_file)
        #self.read_associated_bkg_ratefiles(det1_dir=self.path,input_file_base=self.name)


    def read_ratefiles(self,det1_dir =None, input_file_base = None, debug=1):
        if det1_dir != None:
            sstring = det1_dir + '/'+'*rate.fits'
            ratefiles = sorted(glob.glob(sstring))
            for f in ratefiles:
                if input_file_base in f:
                    self.rate_file = f
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


    def init_rate_files(self, input_file=None):
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
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4), dpi=100, sharey=True,sharex=True)

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


    def fix_hot_pix_step(self, input_file=None, debug=True, output_dir = './output/results/',ref_file = None, dither_file=None):
        '''
        Mask hot pipxels as bad.
        '''

        filename = output_dir + input_file #self.rate_file
        ratefile = datamodels.open(filename)
        hdulist = fits.open(filename)
        rate_header = hdulist[0].header
        rate_name=rate_header['TARGPROP']+rate_header['CHANNEL']+rate_header['DETECTOR']+rate_header['TARGNAME']+str(rate_header['PATT_NUM'])
        print(rate_header['TARGPROP']+rate_header['CHANNEL']+rate_header['BAND'])


        filename = output_dir + ref_file  # self.rate_file
        reffile = datamodels.open(filename)
        hdulist = fits.open(filename)
        ref_header = hdulist[0].header
        ref_name = ref_header['TARGPROP'] + ref_header['CHANNEL'] + ref_header['DETECTOR'] + ref_header['TARGNAME'] + \
                    str(ref_header['PATT_NUM'])
        print(ref_header['TARGPROP'] + ref_header['CHANNEL'] + ref_header['BAND'])

        filename = output_dir + dither_file  # self.rate_file
        dithfile = datamodels.open(filename)
        hdulist = fits.open(filename)
        dith_header = hdulist[0].header
        dith_name = dith_header['TARGPROP'] + dith_header['CHANNEL'] + dith_header['DETECTOR'] + dith_header['TARGNAME'] + \
                   str(dith_header['PATT_NUM'])
        print(dith_header['TARGPROP'] + dith_header['CHANNEL'] + dith_header['BAND'])

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
                if x<data.shape[0]-4 and y < data.shape[1]-4 and x>4 and y>4:
                    #print(x,y)
                    f = data[x,y]
                    flux_mean,npix = 0,0
                    for j,k in zip([x-2,x-3,x-4,x+4,x+3,x+2],[y,y,y,y,y,y]):
                        if dq[j,k]==0:
                            flux_mean += data[j,k]
                            npix +=1
                    if npix>0:
                        flux_mean/=npix


                    flux_mean2,npix = 0,0
                    for j, k in zip([x-2,x-3,x-4,x+4,x+3,x+2],[y,y,y,y,y,y,]):
                        if dq[j, k] == 0 and data[j, k]<2*flux_mean:
                            flux_mean2 += data[j, k]
                            npix += 1
                    if npix > 0:
                        flux_mean2 /= npix

                    flux_mean=flux_mean2

                    if data[x,y]<5*flux_mean:
                        mask[x,y] = False
                    if np.bitwise_and(dq[x,y],1) and 0:
                        d = dq[x,y]
                        mask[x, y] = False
                    elif np.bitwise_and(dq[x,y],4) and 0:
                        d = dq[x,y]
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
        ax[0,0].set_title(rate_name)
        ax[1, 0].imshow(ratefile.data, vmin=-3, vmax=3)
        print('N of hot_pixels_list', np.sum(hot_pixels_list))
        #ax2.hist(ar[ar>threshold_pix_val].flatten(),bins=np.linspace(0,20,21),alpha=0.3)
        ar = np.zeros_like(data)
        ar[hot_pixels_ref_list] = data_ref[hot_pixels_ref_list]
        ax[0,1].imshow(ar, vmin=vmin, vmax=vmax)
        ax[0,1].set_title(ref_name)
        ax[1, 1].imshow(reffile.data, vmin=-3, vmax=3)
        print('N of hot_pixels_ref_list', np.sum(hot_pixels_ref_list))
        #ax2.hist(ar[ar>threshold_pix_val].flatten(),bins=np.linspace(0,20,21),alpha=0.3)

        ar = np.zeros_like(data)
        ar[hot_pixels_dith_list] = data_ref[hot_pixels_dith_list]
        ax[0,2].imshow(ar, vmin=vmin, vmax=vmax)
        ax[0,2].set_title(dith_name)
        print('N of hot_pixels_ref_list', np.sum(hot_pixels_dith_list))

        ar = np.zeros_like(data)
        ar[hot_pixels_ref_list*hot_pixels_list*hot_pixels_dith_list] = data_dith[hot_pixels_ref_list*hot_pixels_list*hot_pixels_dith_list]
        ax[0,3].imshow(ar, vmin=vmin, vmax=vmax)
        ax[0,3].set_title('Mix')
        ax[1,3].imshow(dithfile.data, vmin=-3, vmax=3)
        print('N of hot pix', np.sum(hot_pixels_ref_list*hot_pixels_list*hot_pixels_dith_list), ' of ',np.sum(hot_pixels_ref_list))
        ax2.hist(ar[ar>threshold_pix_val].flatten(),bins=np.linspace(0,20,101),alpha=0.3)
        ax2.hist(data_dith.flatten(), bins=np.linspace(-10, 20, 101), alpha=0.3)
        plt.show()
        #plt.hist(p[~np.isnan(p)],log=True,bins=np.linspace(-1,100,200))
        if 1:
            CHAN,BAND =rate_header['CHANNEL'],rate_header['BAND']
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
            hdul.writeto('./data/hot_pixels_'+CHAN+'_'+BAND+ '.fits',overwrite=True)

        plt.show()
        p =1

    def show_hot_pix_maps(self,debug=True):
        output_dir = './data/'

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
    def select_hot_pix(self,output_dir='./data/',debug=False):
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

        # Call the run() method on the uncal file
        self.data = flat_field_step.run(input_file)
        print('FLAT FIELD step done')

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

        # Call using the the output from the previously-run dq_init step
        self.data = stray_ligth_step.run(input_file)
        if debug:
            print('Stray light step: Done.')

    def fringe_flat_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
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

        fringe_flat_step = FringeStep()
        fringe_flat_step.output_dir = output_dir
        fringe_flat_step.save_results = save_results

        # Call using the the output from the previously-run dq_init step
        self.data = fringe_flat_step.run(input_file)
        if debug:
            print('FRINGE STEP: Done.')

    def res_fringe_step(self, input_file=None, debug=True, output_dir=None,save_results=True):
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
        res_fringe_flat_step.output_dir = output_dir
        res_fringe_flat_step.save_results = save_results

        # Call using the the output from the previously-run dq_init step
        self.data = res_fringe_flat_step.run(input_file)

        # Rename residual_fringe to cal files
        # Look for our _residual_fringe.fits files produced by the photometric calibration step
        sstring = self.output_dir + self.name +  '*residual_fringe.fits'
        self.residual_fringe = sorted(glob.glob(sstring))
        # And print them out so that we can see them
        self.calfiles = self.residual_fringe.copy()
        for ii in range(0, len(self.residual_fringe)):
            self.calfiles[ii] = str.replace(self.residual_fringe[ii], 'residual_fringe', 'cal')
            os.renames(self.residual_fringe[ii], self.calfiles[ii])
            #example_file = fits.open(self.residual_fringe[ii])
            #example_file.writeto(self.calfiles[ii], overwrite=True)
            #example_file.close()

        if debug:
            print('RES FRINGE STEP: Done.')
    def flux_calibration_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
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


        if debug:
            print('PHOTOM step: Done.')


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
    exp1.fix_hot_pix_step(input_file='jw02155004001_03106_00001_mirifushort_rate.fits',ref_file = "jw02155004001_03106_00004_mirifushort_rate.fits", dither_file='jw02155001001_04102_00003_mirifushort_rate.fits')
    exp1.show_hot_pix_maps()
    exp1.compare_maps()
    plt.show()
    exp1.assignwcsstep()
    exp1.flat_field_step()
    exp1.source_type_identification()
    exp1.stray_light_step()
    exp1.fringe_flat_step()
    exp1.flux_calibration_step()
    exp1.cube_building_step()
    #exp1.spec1d_extraction_step()




