# Basic system utilities for interacting with files
import glob
import sys
#Modify the path to a directory on your machine
import os
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

# JWST pipeline utilities
from jwst import datamodels # JWST datamodels
from jwst.associations import asn_from_list as afl # Tools for creating association files
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase # Definition of a Lvl2 association file
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base # Definition of a Lvl3 association file
from stcal import dqflags # Utilities for working with the data quality (DQ) arrays
from jwst.datamodels import dqflags
from stdatamodels.jwst import datamodels

#define input/output
#output_dir = './output/detector3/'
#input_dir = './output/detector2/'
#miri_uncal_file= 'jw02155001001_04102_00001_mirifulong_uncal.fits'
#input_file_base = os.path.basename(miri_uncal_file).replace('uncal.fits', '')

output_dir = settings['output3_dir'] #')./output/detector1/'
input_dir = settings['input3_dir'] #./input/detector1/'


class IFScube():
    def __init__(self):
        self.data = None
        self.err = None
        self.dq = None
        self.wmap = None
        self.hdtap = None
        self.asdf= None
        self.meta = None

class detector3():
    def __init__(self,cubename= None, path = None, obj_key_name = 'jw02155001001_04',bkgr_key_name = 'jw02155009001_02',output_dir=None,spec3_cachedir=None):
        self.cubename = cubename
        self.obj_key_name  = obj_key_name
        self.bkgr_key_name = bkgr_key_name
        self.path = path
        self.output_dir=output_dir
        self.data = None
        self.spec3_cachedir = spec3_cachedir
        self.flags = {}
        #self.read_ratefiles(det1_dir=self.path,input_file_base=self.name)
        #self.read_associated_bkg_ratefiles(det1_dir=self.path,input_file_base=self.name)

    def add_cube(self,cubename=None, debug = True):
        if cubename != None:
            self.__init__(cubename=cubename,path=self.path, output_dir=self.output_dir)
            if debug:
                print('Input file is updated to:', cubename)

    def init_cube(self, read_spectum=True):
        self.data = IFScube()
        if 1:
            data = datamodels.open(self.cubename)
            self.meta = data.meta
            del(data)
        hdu1 = fits.open(self.cubename)
        if isinstance(hdu1['WMAP'].data,(np.ndarray, np.generic)):
            self.data.data = hdu1['SCI'].data
            self.data.err = hdu1['ERR'].data
            self.data.dq = hdu1['DQ'].data
            self.data.wmap = hdu1['WMAP'].data
            #self.data.hdrtab = hdu1['HDRTAB'].data


            self.data.asdf = hdu1['ASDF'].data
            header = hdu1[0].header
            self.data.objname =  header['TARGPROP']
            if 'MIRI' in header['INSTRUME']:
                self.data.band =  header['BAND']
                self.data.channel =  header['CHANNEL']
            elif 'NIRSPEC' in header['INSTRUME']:
                self.data.band = header['FILTER']
                self.data.channel = header['GRATING']
            header = hdu1['SCI'].header
            wcs = {}
            wcs['CRPIX1'] = header['CRPIX1']
            wcs['CRPIX2'] = header['CRPIX2']
            wcs['CRPIX3'] = header['CRPIX3']
            wcs['CRVAL1'] = header['CRVAL1']
            wcs['CRVAL2'] = header['CRVAL2']
            wcs['CRVAL3'] = header['CRVAL3']
            wcs['CDELT1'] = header['CDELT1']
            wcs['CDELT2'] = header['CDELT2']
            wcs['CDELT3'] =header['CDELT3']
            wcs['NAXIS1'] = header['NAXIS1']
            wcs['NAXIS2'] = header['NAXIS2']
            wcs['NAXIS3'] = header['NAXIS3']
            self.flags['subtract_bkgr'] = False

            self.data.wcs = wcs
            hdu1.close()
        if read_spectum:
            sstring = self.cubename.split('_s3d')[0] +  '_x1d.fits'
            specfile = sorted(glob.glob(sstring))
            hdu2 = fits.open(specfile[0])
            self.data.wavelength = hdu2['EXTRACT1D'].data['WAVELENGTH']
            hdu2.close()
        else:
            self.data.wavelength = np.arange(self.data.data.shape[0])
    def conv_world_coord(self,t,x,y,mode='pipeline'):
        wcs1 = self.data.wcs
        #print('wcs1: pix size',wcs1['CDELT1'],wcs1['CDELT2'])
        if mode == 'cubevis':
            x_world = wcs1['CRVAL1'] - (x - wcs1['CRPIX1']+1) * wcs1['CDELT1']*1.043
            y_world = wcs1['CRVAL2'] + (y - wcs1['CRPIX2']+1) * wcs1['CDELT2']*1.043
            lam_world = wcs1['CRVAL3'] + wcs1['CDELT3']*(t - wcs1['CRPIX3']-wcs1['NAXIS3'])
        elif mode == 'normal':
            x_world = wcs1['CRVAL1'] - (x - wcs1['CRPIX1'] +1) * wcs1['CDELT1']
            y_world = wcs1['CRVAL2'] + (y - wcs1['CRPIX2'] +1) * wcs1['CDELT2']
            lam_world = wcs1['CRVAL3'] + (t - wcs1['CRPIX3'] + 1)* wcs1['CDELT3']
        elif  mode == 'pipeline':
            wcs = self.meta.wcs
            sky = wcs.pixel_to_world(x, y, t)
            x_world = sky[0].ra.deg
            y_world = sky[0].dec.deg
            lam_world = wcs1['CRVAL3'] + (t - wcs1['CRPIX3'] + 1) * wcs1['CDELT3']

        #print('world coord:', x_world, y_world, lam_world)
        #print(self.data.wcs((t,x,y)))
        return lam_world,x_world,y_world

    def conv_pix_2_world_coord(self, lam_world=None,x_world=5,y_world=5):
        wcs1 = self.data.wcs
        if lam_world != None:
            #x_world = wcs1['CRVAL1'] - (x - wcs1['CRPIX1'] + 1) * wcs1['CDELT1']
            x = (wcs1['CRVAL1'] - x_world)/ wcs1['CDELT1'] - 1 + wcs1['CRPIX1']
            #y_world = wcs1['CRVAL2'] + (y - wcs1['CRPIX2'] + 1) * wcs1['CDELT2']
            y = (y_world-wcs1['CRVAL2'])/wcs1['CDELT2'] -1 + wcs1['CRPIX2']
            #lam_world = wcs1['CRVAL3'] + (t - wcs1['CRPIX3'] + 1) * wcs1['CDELT3']
            t = (lam_world - wcs1['CRVAL3'])/wcs1['CDELT3'] -1 + wcs1['CRPIX3']
            print('world coord:', x_world, y_world, lam_world)
            # print(self.data.wcs((t,x,y)))
            return t,x,y
        else:
            x = (wcs1['CRVAL1'] - x_world) / wcs1['CDELT1'] - 1 + wcs1['CRPIX1']
            # y_world = wcs1['CRVAL2'] + (y - wcs1['CRPIX2'] + 1) * wcs1['CDELT2']
            y = (y_world - wcs1['CRVAL2']) / wcs1['CDELT2'] - 1 + wcs1['CRPIX2']
            # lam_world = wcs1['CRVAL3'] + (t - wcs1['CRPIX3'] + 1) * wcs1['CDELT3']
            return x, y

        #with datamodels.open(self.cubename) as input_models:
        #    if isinstance(input_models, datamodels.IFUImageModel):
        #        hdu1 =datamodels.open(self.cubename)


    def writel3asn(self,files, asnfile, prodname, **kwargs):
        '''
        The Spec3 pipeline is the first place where we really have to deal with many files at the same time and how they interact with each other
         (i.e., background observations, dithered observations, etc). As such, we need to create an 'Association File' describing these files and
         how they should be treated by the pipeline. These files are still under some development, and the recommended methods of using them will
         likely change before Cycle 1. We'll therefore define a function to create a very simple association file that will treat all exposures
         it is given as science exposures to be combined together.
        See https://jwst-pipeline.readthedocs.io/en/latest/jwst/associations/index.html
        '''
        # Define the basic association of science files
        asn = afl.asn_from_list(files, rule=DMS_Level3_Base, product_name=prodname)
        # Add any background files to the association
        if ('bg' in kwargs):
            for bgfile in kwargs['bg']:
                asn['products'][0]['members'].append({'expname': bgfile, 'exptype': 'background'})
        # Write the association to a json file
        _, serialized = asn.dump()
        with open(asnfile, 'w') as outfile:
            outfile.write(serialized)

    def load3asn(self, asnfile):
        '''
        Read asn file
        '''
        # Define the basic association of science files
        asn_exptypes = ['science', 'background']
        if 1:
            input_models = datamodels.open(asnfile, asn_exptypes=asn_exptypes)
            table = input_models.meta.instance['asn_table']
            ratefiles = table['products'][0]['members']
        else:
            d = {}
            d['expname'] = './output/detector2/jw02155001001_04102_00001_mirifushort_cal.fits'
            d['exptype'] = 'science'
            ratefiles = []
            ratefiles.append(d)
            d = {}
            d['expname'] = './output/detector2/jw02155001001_04102_00002_mirifushort_cal.fits'
            d['exptype'] = 'science'
            ratefiles.append(d)
            d = {}
            d['expname'] = './output/detector2/jw02155001001_04102_00003_mirifushort_cal.fits'
            d['exptype'] = 'science'
            ratefiles.append(d)
            d = {}
            d['expname'] = './output/detector2/jw02155001001_04102_00004_mirifushort_cal.fits'
            d['exptype'] = 'science'
            ratefiles.append(d)
        self.ratefiles = ratefiles
        f = 1

    def read_ratefiles(self,det1_dir =None, input_file_base = None, debug=1):
        if det1_dir != None:
            sstring = det1_dir + '/'+'*rate.fits'
            ratefiles = sorted(glob.glob(sstring))
            for f in ratefiles:
                if input_file_base in f:
                    self.rate_file = f
                    if debug:
                        print('Read input file:', f)

    def bkgr_subtraction_step(self, input_file=None, debug=False, output_dir=None):
        '''
        This step populates the Data Quality (DQ) mask that is associated with the data file.
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.rate_file


        bkgr_subtr_step = MasterBackgroundStep()
        bkgr_subtr_step.output_dir = output_dir
        bkgr_subtr_step.save_results = True

        # Call the run() method on the uncal file
        self.data = bkgr_subtr_step.run(input_file)


        print('BKGR SUBTR STEP 3 done')

    # direct subtraction of bkg_exp from sci_exp
    def residual_background_matching(self, input_dir=None, key_name = 'mirifushort_cal.fits', output_dir=None):
        '''
         Since the thermal background at MIRI wavelength is significant, there is the possibility that it could vary in undesirable ways between exposures.
         As such, a single background image would be insufficient to bring all exposures to the same level. The residual background matching step therefore
         ensures that the low-order background is consistent across all exposures.
        See https://jwst-pipeline.readthedocs.io/en/latest/jwst/mrs_imatch/index.html
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_dir == None:
            input_dir  = self.path

        sci_exp_list = []
        bkg_exp_list = []

        sstring = input_dir + '/' + '*' + key_name
        cal_files = sorted(glob.glob(sstring))
        for f in cal_files:
            if self.obj_key_name in f:
                sci_exp_list.append(f)
            elif self.bkgr_key_name in f:
                bkg_exp_list.append(f)

        self.writel3asn(cal_files, 'rbm.json', 'rbm')

        cb = CubeBuildStep()
        cb.call('rbm.json', channel='2', save_results=True, output_dir=self.output_dir, output_file='rbm_before')

    # This step divides the array data by the pixel flatfield reference file. (for MIRI MRS the reference flatfield is currently unity everywhere )
    def outlier_detection_step(self, input_file=None, debug=False, output_dir=None):
        '''
        In general, cosmic rays should have been flagged in the Detector1 pipeline as jumps in the readout ramp and corrected
        when determining the final slope image. However, sometimes fainter cosmic rays slip through Detector1, and other outliers
        (due to bad pixels, etc) can be present that cause artifacts in the final data cubes. The Outlier Detection routine
        is designed to identify and flag these artifacts.
        In brief, it builds a data cube from each individual exposure on a common data cube grid, and median combines the cubes
        to construct a cleaned cube. This cleaned cube is mapped back to the 2d calibrated detector data of each exposure,
        and outliers in individual frames with respect to this mapped-back data are flagged. When we next do cube building,
        it will use this updated flagging to construct our high-quality combined data cube.
        Since the IFUs are significantly undersampled however, we need to be careful not to accidentally flag point source
        as outliers since the fluxes can change significantly between different pointings. Note also that outlier detection
        cannot detect outliers when there are just 2 or fewer frames; therefore even in a four-point dither pattern
        it will only be able to work in the central regions where all four exposures overlap.
        See https://jwst-pipeline.readthedocs.io/en/latest/jwst/outlier_detection/index.html
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        sstring = output_dir + '*cal.fits'
        calfiles = sorted(glob.glob(sstring))

        testfiles = calfiles.copy()
        self.writel3asn(testfiles, 'od.json', 'od')

        cb = CubeBuildStep()
        cb.call('od.json', channel='2', save_results=True, output_dir=self.output_dir, output_file='od_before')

        # And for comparison we'll run this association through the Spec3 pipeline with just
        # Outlier Detection and Cube Build, saving our final cube as 'od_after'

        # This initial setup is just to make sure that we get the latest parameter reference files
        # pulled in for our files.  This is a temporary workaround to get around an issue with
        # how this pipeline calling method works.
        crds_config = Spec3Pipeline.get_config_from_reference('od.json')
        spec3 = Spec3Pipeline.from_config_section(crds_config)

        # Now make any modifications to the pipeline specification
        spec3.output_dir = self.output_dir
        spec3.save_results = True
        spec3.master_background.skip = True
        spec3.mrs_imatch.skip = True
        spec3.outlier_detection.save_intermediate_results = True  # We'll write out intermediate files to explore what they look like
        spec3.cube_build.channel = '2'
        spec3.cube_build.output_file = 'od_after'
        spec3.extract_1d.skip = True

        # If rerunning long pipeline steps, actually run the step
        spec3('od.json')
        # Otherwise, just copy cached outputs into our output directory structure

    def create_association(self, input_dir=None,source = 'Object',channel = '1', band ='SHORT',subfilename = '',dither=None):
        if band != 'ABC':
            exp_list = []
            bandname = {}
            bandname['SHORT'] = 'A'
            bandname['MEDIUM'] = 'B'
            bandname['LONG'] = 'C'

            #sstring = input_dir + '/' + ('*bkgr_sub_cal.fits')
            sstring = input_dir + '/' + ('*_cal.fits')
            cal_files = sorted(glob.glob(sstring))
            for f in cal_files:
                hdulist = fits.open(f)
                header = hdulist[0].header
                f_targ_name = header['TARGPROP']
                f_band = header['BAND']
                f_channel = header['CHANNEL']
                f_dit_pos = header['PATT_NUM']
                hdulist.close()
                if dither == None:
                    f_dit_pos = None
                if f_targ_name == source and channel in f_channel and f_band in band and f_dit_pos == dither:
                    exp_list.append(f.split('/')[-1])
            if subfilename == '':
                asn_name = input_dir + '/'+ source + '_'+channel + bandname[band] + '.json'
            else:
                asn_name = input_dir + '/' + source + subfilename + '_'+channel + bandname[band] +'.json'
            #exp_list = [exp_list[0],exp_list[1]]
            if len(exp_list)>0:
                print('ASN_FILE',asn_name, [el for el in exp_list])
                self.writel3asn(exp_list, asn_name, source + '_'+channel + bandname[band])
                self.local_asn_file = asn_name
                return asn_name



    def sort_calfiles(files):
        channel = []
        band = []

        for file in files:
            hdr = (fits.open(file))[0].header
            channel.append(hdr['CHANNEL'])
            band.append(hdr['BAND'])
        channel = np.array(channel)
        band = np.array(band)

        indx = np.where((channel == '12') & (band == 'SHORT'))
        files12A = files[indx]
        indx = np.where((channel == '12') & (band == 'MEDIUM'))
        files12B = files[indx]
        indx = np.where((channel == '12') & (band == 'LONG'))
        files12C = files[indx]
        indx = np.where((channel == '34') & (band == 'SHORT'))
        files34A = files[indx]
        indx = np.where((channel == '34') & (band == 'MEDIUM'))
        files34B = files[indx]
        indx = np.where((channel == '34') & (band == 'LONG'))
        files34C = files[indx]

        return files12A, files12B, files12C, files34A, files34B, files34C

    def build_cube(self, input_file='rbm.json', output_dir=None,channel = '1', band = 'A',master_bkgr_flag = 0,
        master_res_bkgr_flag = 0, master_outlier_flag = 0,  master_extract1d_flag = 1,master_resample_spec_flag=1):
        if output_dir == None:
            output_dir = self.output_dir

        spec3 = Spec3Pipeline()
        spec3.output_dir = output_dir
        spec3.save_results = True
        spec3.master_background.skip = 1 - master_bkgr_flag
        spec3.outlier_detection.skip = 1 - master_outlier_flag
        spec3.mrs_imatch.skip = 1 - master_res_bkgr_flag
        spec3.resample_spec.skip = 1 #-master_resample_spec_flag
        spec3.cube_build.channel = channel
        spec3.cube_build.output_file = (input_file.split('/')[-1]).split('.')[0]
        spec3.extract_1d.skip = 1 - master_extract1d_flag
        spec3.coord_system = 'ifualign'
        if band == 'ABC':
            spec3.cube_build.output_type = 'channel'

        print('master_background.skip', 1 - master_bkgr_flag)
        print('spec3.outlier_detection.skip', 1 - master_outlier_flag)
        print('mrs_imatch.skip',1 - master_res_bkgr_flag)
        print('mrs_resample_spec.skip', 1 - master_resample_spec_flag)
        print('spec3.extract_1d.skip', 1 - master_extract1d_flag)

        spec3(input_file)
        print('DONE!')

    def spec_extraction(self, cube_filename = 'sci_1short_ch1-short_s3d.fits'):
        '''
        The MIRI MRS has been observed to have appreciable straylight at short wavelengths in ground-test data,
        and this step is therefore designed to model and subtract this component from the detector data
        using the small interstitial regions of detector pixels between the illuminated slices.

        As such, we usually recommend skipping this step when working with simulated data. However, we'll run it here just to see what it looks like.
        See https://jwst-pipeline.readthedocs.io/en/latest/jwst/straylight/index.html
        '''

        Extract1dStep.call(cube_filename, save_results=True, output_dir=self.output_dir)

    def plot_cube(self,cube_filename = 'sci_1short_ch1-short_s3d.fits'):
        hdu1 = fits.open(cube_filename)
        flux1 = hdu1['SCI'].data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 7), dpi=100)
        norm = ImageNormalize(flux1, interval=ZScaleInterval(), stretch=LogStretch())

        # And plot the data.  Highlight a pixel in the bad column with a red X
        ax1.imshow(flux1[199, :, :], cmap='gray', origin='lower')
        ax1.set_title('SKYALIGN')

    def plot_spec(self,spec_file_name = 'sci_1short_ch1-short_extract1dstep.fits'):
        specfile = self.output_dir + spec_file_name

        # Let's look at one of them
        hdu = fits.open(specfile)
        spec = hdu['EXTRACT1D']

        fig,ax = plt.subplots()
        ax.plot(spec.data['WAVELENGTH'], spec.data['FLUX'])
        ax.set_xlabel('Wavelength (micron)')
        ax.set_ylabel('Flux (Jy)')

    def create_association_12pieces(self):

        # Find and sort all of the input files
        sstring = self.path + 'det*cal.fits'
        calfiles = np.array(sorted(glob.glob(sstring)))
        sortfiles = self.sort_calfiles(calfiles)  # Split them up into bands
        print('Found ' + str(len(calfiles)) + ' input files to process')

        asnlist = []
        names = ['12A', '12B', '12C', '34A', '34B', '34C']
        for ii in range(0, len(sortfiles)):
            thesefiles = sortfiles[ii]
            ninband = len(thesefiles)
            if (ninband > 0):
                filename = 'l3asn-' + names[ii] + '.json'
                asnlist.append(filename)
                self.writel3asn(thesefiles, filename, 'Level3')


if __name__ == '__main__':
    print('Hi PyCharm')
    miri_uncal_file = 'jw02155001001_04102_00001_mirifulong_uncal.fits'
    input_dir = './output/detector2'
    output_dir  = './output/detector3/'
    spec3_cachedir = './temp/spec3/'
    exposure = detector3(obj_key_name = 'jw02155001001_04102',bkgr_key_name = 'jw02155009001_02101', path = input_dir, output_dir=output_dir) #cubename='./output/detector3/sci_1SHORT(A)_ch1-short_s3d.fits')
    exposure.create_association(input_dir=input_dir, source='BACKGROUND-AO0235+164', channel='3', band='SHORT',
                                subfilename='_TEST_4EXP')
    exposure.build_cube(input_file=exposure.local_asn_file, channel='3', master_bkgr_flag=False,
               master_res_bkgr_flag=True, master_outlier_flag=True,
               master_resample_spec_flag=True, master_extract1d_flag=True)
    #exposure.init_cube()
    #exposure.load3asn(asnfile='sci_1short.json')
    #exposure.residual_background_matching()
    #exposure =  detector3(obj_key_name = 'jw02155001001_04102',bkgr_key_name = 'jw02155009001_02101', path = input_dir, output_dir=output_dir)
    #[sci, bkg] = exposure.create_association(input_dir=exposure.path, channel = '1', band ='short')
    #exposure.cube_creation(channel='1')
    #exposure.spec_extraction()
    #exposure.plot_cube(cube_filename='./output/detector3/sci_1SHORT(A)_ch1-short_s3d.fits')
    #exposure.plot_spec()

    plt.show()

