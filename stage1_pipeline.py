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
settings =  read_settings(init_file='init.dat')
os.environ["CRDS_PATH"] = settings['CRDS_PATH']
os.environ["CRDS_SERVER_URL"] = settings['CRDS_SERVER_URL']
if 'CRDS_CONTEXT' in  settings.keys():
    os.environ["CRDS_CONTEXT"] = settings['CRDS_CONTEXT']
#os.environ["CRDS_PATH"] = "/home/slava/science/codes/python/jwst/data"
#os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

# Packages that allow us to get information about objects:
import asdf
import copy
import shutil

# Numpy library:
import numpy as np

# For downloading data
import requests

# Astropy tools:
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch
#Import JWST pipeline-related modules
# List of possible data quality flags
from stdatamodels.jwst.datamodels import dqflags

# The entire calwebb_detector1 pipeline
from jwst.pipeline import calwebb_detector1

# Individual steps that make up calwebb_detector1
from jwst.dq_init import DQInitStep
from jwst.saturation import SaturationStep
from jwst.reset import ResetStep
from jwst.firstframe import FirstFrameStep
from jwst.lastframe import LastFrameStep
from jwst.linearity import LinearityStep
from jwst.rscd import RscdStep
from jwst.dark_current import DarkCurrentStep

from jwst.superbias import SuperBiasStep
from jwst.ipc import IPCStep
from jwst.refpix import RefPixStep
from jwst.linearity import LinearityStep
from jwst.persistence import PersistenceStep
from jwst.dark_current import DarkCurrentStep
from jwst.jump import JumpStep
from jwst.ramp_fitting import RampFitStep
from jwst import datamodels
import matplotlib.pyplot as plt
import matplotlib as mpl
from stdatamodels.jwst.datamodels import dqflags
from jwst.assign_wcs import AssignWcsStep
from jwst.superbias import SuperBiasStep


import jwst
print(jwst.__version__)


output_dir = settings['output1_dir'] #')./output/detector1/'
input_dir = settings['input1_dir'] #./input/detector1/'


#functions
def download_files(files, output_directory, force=False):
    """Given a tuple or list of tuples containing (URL, filename),
    download the given files into the current working directory.
    Downloading is done via astropy's download_file. A symbolic link
    is created in the specified output dirctory that points to the
    downloaded file.

    Parameters
    ----------
    files : tuple or list of tuples
        Each 2-tuple should contain (URL, filename), where
        URL is the URL from which to download the file, and
        filename will be the name of the symlink pointing to
        the downloaded file.

    output_directory : str
        Name of the directory in which to create the symbolic
        links to the downloaded files

    force : bool
        If True, the file will be downloaded regarless of whether
        it is already present or not.

    Returns
    -------
    filenames : list
        List of filenames corresponding to the symbolic links
        of the downloaded files
    """
    # In the case of a single input tuple, make it a
    # 1 element list, for consistency.
    filenames = []
    if isinstance(files, tuple):
        files = [files]

    for file in files:
        filenames.append(file[1])
        if force:
            print('Downloading {}...'.format(file[1]))
            demo_file = download_file(file[0], cache='update')
            # Make a symbolic link using a local name for convenience
            if not os.path.islink(os.path.join(output_directory, file[1])):
                os.symlink(demo_file, os.path.join(output_directory, file[1]))
        else:
            if not os.path.isfile(os.path.join(output_directory, file[1])):
                print('Downloading {}...'.format(file[1]))
                demo_file = download_file(file[0], cache=True)
                # Make a symbolic link using a local name for convenience
                os.symlink(demo_file, os.path.join(output_directory, file[1]))
            else:
                print('{} already exists, skipping download...'.format(file[1]))
                continue
    return filenames

def show_image(data_2d, vmin, vmax, xpixel=None, ypixel=None, title=None):
    """Function to generate a 2D, log-scaled image of the data,
    with an option to highlight a specific pixel (with a red dot).

    Parameters
    ----------
    data_2d : numpy.ndarray
        Image to be displayed

    vmin : float
        Minimum signal value to use for scaling

    vmax : float
        Maximum signal value to use for scaling

    xpixel : int
        X-coordinate of pixel to highlight

    ypixel : int
        Y-coordinate of pixel to highlight

    title : str
        String to use for the plot title
    """
    norm = ImageNormalize(data_2d, interval=ManualInterval(vmin=vmin, vmax=vmax),
                          stretch=LogStretch())
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(data_2d, origin='lower', norm=norm)

    if xpixel and ypixel:
        plt.plot(xpixel, ypixel, marker='o', color='red', label='Selected Pixel')

    fig.colorbar(im, label='DN')
    plt.xlabel('Pixel column')
    plt.ylabel('Pixel row')
    if title:
        plt.title(title)

class detector1():
    def __init__(self,input_file=None,path=None, output_dir=None):
        self.input_file = input_file
        self.name = input_file.split('/')[-1]
        self.path = path
        self.output_dir=output_dir
        self.data = None

    def add(self,name=None, debug=1):
        if name != None:
            self.__init__(input_file=name,path=self.path, output_dir=self.output_dir)
            if debug:
                print('Input file is updated to:', name)

    def dq_init_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
        '''
        This step populates the Data Quality (DQ) mask that is associated with the data file.
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.input_file

        print(' JWST VERSION:  ', jwst.__version__)

        dq_init_step = DQInitStep()
        dq_init_step.output_dir = output_dir
        dq_init_step.save_results = save_results

        # Call the run() method on the uncal file
        self.data = dq_init_step.run(self.path + '/'+ input_file)
        shape = self.data.data.shape
        self.nint = shape[0]
        self.ngroup = shape[1]
        self.nrows = shape[2]
        self.ncols = shape[3]
        #change to old version: ALL_MRS to ALL
        self.data.meta.dither.primary_channel = 'ALL'

        if debug:
            # Print some basic information on the number of flagged pixels
            idx_pixelDQ = np.where(self.data.pixeldq.flatten() == 0.)[0]
            num_flagged = self.data.pixeldq.size - len(idx_pixelDQ)
            print('Total pixels in PIXELDQ: {}'.format(self.data.pixeldq.size))
            print('{} pixels have no flags.'.format(len(idx_pixelDQ)))
            print('{} pixels ({:.2f}% of the detector) have flags.'.format(num_flagged, num_flagged / self.data.pixeldq.size))

    def saturation_step(self, input_file=None, debug=True, output_dir=None,n_pix_grow_sat=0,save_results=False):
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        saturation_step = SaturationStep()
        saturation_step.output_dir = output_dir
        saturation_step.save_results = save_results
        saturation_step.n_pix_grow_sat = n_pix_grow_sat

        # Call using the output from the previously-run dq_init step
        self.data = saturation_step.run(input_file)

        if debug and 0:
            saturation = self.data
            # Find indexes of saturated pixels
            saturated = np.where(saturation.groupdq & dqflags.pixel['SATURATED'] > 0)
            num_sat_flags = len(saturated[0])
            print(('Found {} saturated flags. This may include multiple saturated '
                   'groups within a given pixel'.format(num_sat_flags)))

            # Create a 4D boolean map of whether the saturation flag is present or not.
            saturated = (saturation.groupdq & dqflags.pixel['SATURATED'] > 0)
            # Collapse that down to a 2D map that lists the number of saturated groups
            # for each pixel.
            saturated_2d = np.sum(saturated[0, :, :, :], axis=0)
            # Get coordinates of pixels that are saturated in some, but not all, groups.
            partial_sat = np.where((saturated_2d > 0) & (saturated_2d < saturated.shape[1]))
            print("{} pixels are partially saturated.".format(len(partial_sat[0])))
            print("{} pixels are fully saturated.".format(len(np.where(saturated_2d == saturated.shape[1])[0])))

            sat_y, sat_x = partial_sat
            for i in range(len(sat_y)):
                print(i, sat_y[i], sat_x[i], np.sum(saturated[0, :, sat_y[i], sat_x[i]]),
                      saturated[0, :, sat_y[i], sat_x[i]], saturation.pixeldq[sat_y[i], sat_x[i]])
            fig, ax = plt.subplots()
            ax.plot(sat_x, sat_y, 'o', markersize=20)
            fig, ax = plt.subplots(2, 15, sharey=True)
            pix_index = np.random.randint(len(partial_sat[0]), size=len(partial_sat[0]))
            for axi in range(15):
                sat_index = pix_index[axi]
                y = sat_y[sat_index]
                x = sat_x[sat_index]
                grps = saturated[0, :, y, x]
                print('Saturation flags up the ramp (0 is not saturated, 2 is saturated): {}'.format(
                    saturation.groupdq[0, :, y, x]))
                print('Pixel signal values up the ramp: {}'.format(saturation.data[0, :, y, x]))
                # Get the science and DQ values for the pixel.
                groups = np.arange(saturation.data.shape[1])
                full_ramp = saturation.data[0, :, y, x]
                sat_dq = saturation.groupdq[0, :, y, x].astype(bool)

                # Make a copy of the science data and set all saturated groups to NaN
                saturated_points = copy.deepcopy(saturation.data[0, :, y, x])
                saturated_points[~sat_dq] = np.nan
                # Plot the pixel's values up the ramp and denote the saturated groups


            if 1:
                normal_pixels = np.where(saturated_2d == 0)

                pix_y, pix_x = normal_pixels
                pix_index = np.random.randint(1e5, size=40)
                for axi in range(15):
                    n = pix_index[axi]
                    y = pix_y[n]
                    x = pix_x[n]
                    full_ramp = saturation.data[0, :, y, x]
                    plot_ramp(groups, full_ramp, title='Normal pixel', xpixel=x, ypixel=y, ax=ax[1, axi])



    def reset_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        reset_step = ResetStep()
        reset_step.output_dir = output_dir
        reset_step.save_results = save_results

        # Call using the output from the previously-run dq_init step
        self.data = reset_step.run(input_file)
        if debug:
            print('Reset step: Done.')

    def first_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
        '''
        the first group in every integration is flagged as bad in  GROUPDQ array (if the number of groups >=3)
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        first_step = FirstFrameStep()
        first_step.output_dir = output_dir
        first_step.save_results = save_results

        # Call using the the output from the previously-run dq_init step
        self.data = first_step.run(input_file)
        if debug:
            print('First step: Done.')

    def last_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
        '''
        flags the final group in each integration as bad  (the “DO_NOT_USE” bit is set in the GROUPDQ flag array),
        but only if the total number of groups in each integration is greater than 2
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        last_step = LastFrameStep()
        last_step.output_dir = output_dir
        last_step.save_results = save_results

        # Call using the the output from the previously-run dq_init step
        self.data = last_step.run(input_file)
        if debug:
            print('LAST step: Done.')

    def exotic_drop_groups(self,input_file=None, mode = 'small_n_gr', output_dir=None,save_results=False):
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data

        custom_drop_groups = DropGroupsStep()
        if mode == 'small_n_gr':
            custom_drop_groups.drop_groups=[6]
        elif mode == 'large_n_gr':
            custom_drop_groups.drop_groups=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99]
        self.data = custom_drop_groups.run(input_file)

    def drop_first_steps(self,input_file=None, mode = 0, output_dir=None,):
        if input_file == None:
            input_file = self.data

        def do_correction(input_model, N_drop = 0):
            """
            Short Summary
            -------------
            The sole correction is to reset to DO_NOT_USE the GROUP data quality flags
            for the first 6 (or 12) groups, if the number of groups is higher than 12.

            Parameters
            ----------
            input_model: data model object
                science data to be corrected

            Returns
            -------
            output: data model object
                lastframe-corrected science data

            """

            # Save some data params for easy use later
            sci_ngroups = input_model.data.shape[1]

            # Create output as a copy of the input science data model
            output = input_model.copy()

            # Update the step status, and if ngroups > 2, set all of the GROUPDQ in
            # the first N group to 'DO_NOT_USE'
            if sci_ngroups < 20:
                N_drop = 2
            elif sci_ngroups >= 20 and sci_ngroups < 40:
                if N_drop < 6:
                    N_drop = 6
            else:
                if N_drop<12:
                    N_drop = 12

            self.drop_ngroups=N_drop
            for i in range(N_drop):
                output.groupdq[:, i, :, :] = \
                    np.bitwise_or(output.groupdq[:, i, :, :], dqflags.group['DO_NOT_USE'])
            print("LastFrame Sub: resetting GROUPDQ in last frame to DO_NOT_USE")
            output.meta.cal_step.drop_groups = 'COMPLETE'

            return output

        self.data = do_correction(input_file, N_drop=mode)

    def linear_step(self, input_file=None, debug=True, output_dir=None,save_results=False, override_linearity=False):
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

        linear_step = LinearityStep()
        linear_step.output_dir = output_dir
        linear_step.save_results = save_results
        linear_step.debug = debug
        if override_linearity:
            linear_step.override_linearity=self.linearity_model

        # Call using the output from the previously-run dq_init step
        self.data = linear_step.run(input_file)
        if debug:
            print('LINEAR step: Done.')

    def rscd_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
        '''
        This correction is currently only implemented for MIRI data and is only applied to integrations after the first integration.
        ( However, the reset FETS do not instantaneously reset the level, instead the exponential adjustment of the FET after a reset causes
         the initial frames in an integration to be offset from their expected values. )
        This step flags the N groups at the beginning of all 2nd and higher integrations as bad (the “DO_NOT_USE” bit is set in the GROUPDQ flag array),
        but only if the total number of groups in each integration is greater than N+3.
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data
        rscd_step = RscdStep()
        rscd_step.output_dir = output_dir
        rscd_step.save_results = save_results

        # Call using the the output from the previously-run dq_init step
        self.data = rscd_step.run(input_file)
        if debug:
            print('RSDF step: Done.')

    def dark_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
        '''
        subtrat dark model from reference files
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data
        dark_step = DarkCurrentStep()
        dark_step.output_dir = output_dir
        dark_step.save_results = save_results

        # Call using the output from the previously-run dq_init step
        self.data = dark_step.run(input_file)
        if debug:
            print('DARK subtract: Done.')

    def refpix_corr_step(self, input_file=None, debug=0, output_dir=None,save_results=False):
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data
        if debug:
            # Use the spec attribute to print available parameters
            print('RefPixStep:')
            print(RefPixStep.spec)
        # Instantiate and set parameters
        refpix_step = RefPixStep()
        refpix_step.output_dir = output_dir
        refpix_step.save_results = save_results

        # Call using the saturation instance from the previously-run
        # saturation step
        self.data = refpix_step.run(input_file)

        if debug:
            # Define the reference pixel step output filename
            show_image(self.data.data[0, 5, :, :], vmin=-1, vmax=10000, title="Difference with/without using side refpix")
            print('REF PIX correction: Done.')
            plt.show()

    def jump_corr_step(self, input_file=None, debug=False, output_dir=None,limit=5,flag_4_neighbors=True,RecalcMedian=True,
                       save_results=False, override_gain= False,find_showers=False):
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data
        if debug:
            print('Jump correction: params()')
            print(JumpStep.spec)

        # Using the run() method
        jump_step = JumpStep()
        jump_step.output_dir = output_dir
        jump_step.save_results = save_results
        #jump_step.rejection_threshold = limit
        #jump_step.three_group_rejection_threshold = limit
        #jump_step.four_group_rejection_threshold = limit
        #jump_step.min_jump_to_flag_neighbors = 15.
        #jump_step.debug=debug
        #jump_step.flag_4_neighbors = flag_4_neighbors
        #jump_step.expand_large_events = False
        jump_step.skip = False
        #jump_step.recalculate_median = RecalcMedian
        jump_step.maximum_cores = 'all' #'1', 'half', 'all'
        jump_step.find_showers = find_showers
        #if override_gain:
        #    jump_step.override_gain = self.gain_model
        print('jump_step.find_showers',jump_step.find_showers)

        # Call using the dark instance from the previously-run
        # dark current subtraction step
        # jump = jump_step.run(dark)
        self.data = jump_step.run(input_file)
        print('CR: update pdq')
        gdq = self.data.groupdq
        pdq = self.data.pixeldq
        for i in range(gdq.shape[0]):
            for j in range(gdq.shape[1]):
                slice = gdq[i,j,:,:]
                mask = np.where(np.bitwise_and(slice,dqflags.pixel['JUMP_DET']))
                pdq[mask] = np.bitwise_or(pdq[mask],dqflags.pixel['JUMP_DET'])
        self.data.pixeldq = pdq
        if 1:
            hdr = fits.Header()
            hdr['TELESCOP'] = 'JWST'
            hdr['INSTRUME'] = 'MIRI'
            hdr['AUTHOR'] = 'V.KLIMENKO'
            hdr['COMMENT'] = "This file contains groupdq data"
            empty_primary = fits.PrimaryHDU(header=hdr)
            #col1 = fits.Column(name='GROUPDQ', format='D', array=gdq)
            #col2 = fits.Column(name='PIXELDQ', format='D', array=pdq)
            #cols = fits.ColDefs([col1])
            #hdu1 = fits.BinTableHDU(data=gdq)
            #hdul = fits.HDUList([empty_primary, hdu1])
            hdul = fits.HDUList()
            hdul.append(empty_primary)
            #hdul.append(fits.PrimaryHDU())
            for img in gdq:
                hdul.append(fits.ImageHDU(data=img,name='GROUPDQ'))
            hdul.writeto(output_dir+self.name.split('uncal.fits')[0] + 'groupdq.fits', overwrite=True)


    def slope_fitting_step(self, input_file=None, debug=True, output_dir=None,
                           save_results=False,override_gain=False,algorithm='OLS_C'):
        '''

        :param input_file:
        :param debug:
            debug = 1 - show fir to slope in subgroups
        :param output_dir:
        :return:
        '''
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data
        if debug:
            print('Slope fitting: params()')
            print(RampFitStep.spec)

        # Using the run() method
        ramp_fit_step = RampFitStep()
        ramp_fit_step.output_dir = output_dir
        ramp_fit_step.save_results = save_results
        ramp_fit_step.debug = debug
        ramp_fit_step.algorithm = algorithm
        #ramp_fit_step.maximum_cores = 'all'
        #if override_gain:
        #    ramp_fit_step.override_gain = self.gain_model

        # Let's save the optional outputs, in order
        # to help with visualization later
        ramp_fit_step.save_opt = False

        # Call using the dark instance from the previously-run
        # jump step
        #self.ramp_fit,self.ramp_fit_info = ramp_fit_step.run(input_file)
        self.ramp_fit = ramp_fit_step.run(input_file) #out_model, int_model
        if ramp_fit_step.save_opt:
            self.ramp_fit_info = datamodels.open(ramp_fit_step.output_dir+
                                                 self.name.split('_uncal.fits')[0]+'_fitopt.fits')
            print()
            #jw05491003001_03104_00001_mirifulong_fitopt.fits


    def chiq_ramp_fit(self, input_file=None, debug=True, output_dir=None,save_results=False):
        if output_dir == None:
            output_dir = self.output_dir
        if input_file == None:
            input_file = self.data
        if debug:
            print('Slope chi1q fit')
        import fitramp
        import time
        niters = self.data.data.shape[0]
        ngroups = self.data.data.shape[1]
        nrows,ncols = self.data.data.shape[2],self.data.data.shape[3]
        readtimes = np.arange(1, ngroups+2)
        C = fitramp.Covar(readtimes)
        data = self.data.data.copy()
        err =  self.data.err.copy()
        pixdq = self.data.pixeldq
        groupdq = self.data.groupdq

        if 1:
            # Get the gain and readnoise reference files
            input_model= datamodels.RampModel(input_file)

            gain_filename = self.get_reference_file(input_model, 'gain')
            self.log.info('Using GAIN reference file: %s', gain_filename)
            gain_model = datamodels.GainModel(gain_filename)
            gain_2d = gain_model.data

            readnoise_filename = self.get_reference_file(input_model,'readnoise')
            self.log.info('Using READNOISE reference file: %s', readnoise_filename)
            readnoise_model = datamodels.ReadnoiseModel(readnoise_filename)
            readnoise_2d = readnoise_model.data

            data *= gain_2d
            err *= gain_2d
            readnoise_2d *= gain_2d
            rate = np.zeros([niters,data.shape[2],data.shape[3]])
            err_rate = np.zeros_like(rate)

        #t0 = time.time()

        for it in niters:
            im = data[it].copy()
            d = (im[1:] - im[:-1]) / C.delta_t[:, np.newaxis, np.newaxis]
            sig = readnoise_2d
            fit = np.zeros((d.shape[1], d.shape[2]))
            err_fit = np.zeros((d.shape[1], d.shape[2]))
            ped = np.zeros((d.shape[1], d.shape[2]))
            alljumps = np.zeros(d.shape)
            alljumpsigs = np.zeros(d.shape)

            for i in range(nrows):
                diffs2use, countrates = fitramp.mask_jumps(d[:, i], C, sig[i], threshold_oneomit=20.25, threshold_twoomit=23.8)
                ct = countrates * (countrates > 0)
                result = fitramp.fit_ramps(d[:, i], C, sig[i], diffs2use=diffs2use, detect_jumps=True, countrateguess=ct)

                alljumps[:, i] = result.jumpval_oneomit
                alljumpsigs[:, i] = result.jumpsig_oneomit
                fit[i, :] = result.countrate
                err_fit[i, :] = result.uncert

                ped[i, :] = (im[0,i,:]+im[1,i,:])/2
                for j in range(len(d)):
                    indx = diffs2use[j] == 0  # only need to redo these differences
                    if np.sum(indx) == 0:
                        continue
                    # each time we'll make sure that this difference isn't masked
                    mask = diffs2use[:, indx] * 1
                    mask[j] = 1
                    result = fitramp.fit_ramps(d[:, i, indx], C, sig[i, indx], diffs2use=mask,
                                               detect_jumps=True, countrateguess=ct[indx])
                    # Overwrite the jump value if it was previously masked.
                    alljumps[j, i, indx] = result.jumpval_oneomit[j]
                    alljumpsigs[j, i, indx] = result.jumpsig_oneomit[j]
                    pixdq[it,i,indx] = 4
                    groupdq[it,j,i,indx] = 4
            rate[it] = fit.copy()
            err_rate[it] = err_fit.copy()
        return rate,err_rate, alljumps, pixdq,groupdq


    def plot_image(self,ngroup = 1,vmin = 3000,vmax=5000):
        fig,ax = plt.subplots(1,2,figsize=(18,9))
        im = ax[0].imshow(self.data.data[0][ngroup],vmin=vmin,vmax=vmax)
        fig.colorbar(im, ax=ax, label='Interactive colorbar')
        ax[1].hist(self.data.data[0][ngroup].ravel(),bins=np.linspace(0,60000,60))
        ax[1].set_yscale('log')

        plt.show()

    def show_dq(self, ax=None, flag = 'DO_NOT_USE'):
        self.pixdq_x =  np.zeros_like(self.data.pixeldq)
        self.pixdq_y = np.zeros_like(self.data.pixeldq)
        for i in range(self.data.pixeldq.shape[1]):
            self.pixdq_x[:, i] = i
        for i in range(self.data.pixeldq.shape[0]):
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
                    mask = np.bitwise_and(self.data.pixeldq, val) == val
                    if np.sum(mask)>0:
                        jc+=1
                        tot_num+=np.sum(mask)
                        if jc> self.dq_colors.shape[0]:
                            break
                        ax.scatter(self.pixdq_x[mask], self.pixdq_y[mask], s=20, alpha=1, c=self.dq_colors[jc], label=self.dq_types[k])
                        print(self.dq_types[k], ':', np.sum(mask), self.dq_colors[jc])

        elif flag!= 'GOOD':
            mask = np.where(np.bitwise_and(self.data.pixeldq, dqflags.pixel[flag]))
            if np.sum(mask)>0:
                tot_num+=np.size(mask[0])
                ax.scatter(self.pixdq_x[mask], self.pixdq_y[mask], s=20, alpha=1, c='black', label=flag)
        else:
            mask = self.data.pixeldq == 0
            if np.sum(mask) > 0:
                tot_num += np.size(mask[0])
                ax.scatter(self.pixdq_x[mask], self.pixdq_y[mask], s=20, alpha=1, c='black', label=flag)
        print('The number of selected pixels:',tot_num) #, '  ',np.bitwise_and(self.data.pixeldq, 2).sum())
        ax.legend(fontsize=20,loc='lower left')
        ax.fill_between(x=[0,self.data.pixeldq.shape[0]], y1 = [0,0],y2=[self.data.pixeldq.shape[1],self.data.pixeldq.shape[1]],color='tab:green',alpha=0.1)
        plt.show()

    def show_pixel(self, ax=None,xcoord=850,ycoord =737,title='',slope=0):
        '''
        plot the counts profile in the pixel

        :param ax:
        :param xcoord: - x pixel coordinate
        :param ycoord: - y pixel coordinate
        :return:
        '''
        if ax == None:
            fig, ax = plt.subplots(figsize=(20,20))
        groups = np.arange(self.data.shape[1])
        ax.plot(groups, self.data.data[0, :, ycoord, xcoord])
        ax.set_title(title + ' ('+str([xcoord,ycoord])+')')
        #plt.show()

    def reset_dq(self,flagname='SATURATED'):
        pdq = self.data.pixeldq
        gdq = self.data.groupdq
        v = dqflags.pixel[flagname]
        mask = np.bitwise_and(pdq, dqflags.pixel[flagname])
        #print(np.sum(mask))
        #print(pdq[0][mask])
        for i in range(pdq.shape[0]):
            for j in range(pdq.shape[1]):
                #print(i,j,el)
                if mask[i,j]:
                    pdq[i, j] -=v
        #pdq[mask] = pdq[mask] - v
        #print('check')
        #mask = np.bitwise_and(self.data.pixeldq, dqflags.pixel[flagname])
        #print('n=0: = ', np.sum(mask))
        for i in range(gdq.shape[0]):
            for j in range(gdq.shape[1]):
                slice = gdq[i,j,:,:]
                mask = np.bitwise_and(slice, v)
                #print(np.sum(mask))
                for l in range(slice.shape[0]):
                    for m in range(slice.shape[1]):
                        #print(i,j,el)
                        if mask[l,m]:
                            slice[l, m] -=v
                #mask = np.bitwise_and(gdq[i,j,:,:], v)
                #print('check = 0? ', np.sum(mask))

    def get_readnoise(self):
        from stdatamodels.jwst import datamodels
        jump_step = JumpStep()
        with datamodels.RampModel(self.data) as input_model:
            readnoise_filename = jump_step.get_reference_file(input_model,'readnoise')
            jump_step.log.info('Using READNOISE reference file: %s',
                          readnoise_filename)
            readnoise_model = datamodels.ReadnoiseModel(readnoise_filename)
            readnoise_2d = readnoise_model.data

            gain_filename =  jump_step.get_reference_file(input_model, 'gain')
            gain_model = datamodels.GainModel(gain_filename)
            gain_2d = gain_model.data
            #readnoise_2d *= gain_2d
            self.readnoisearray = readnoise_2d
            return self.readnoisearray

    def run_detector1(self):
        from jwst.pipeline import Detector1Pipeline
        inpfile = self.path + '/'+ self.input_file
        crds_config = Detector1Pipeline.get_config_from_reference(inpfile)
        detector1 = Detector1Pipeline.from_config_section(crds_config)
        detector1.jump.maximum_cores = 'half'  # Set the jump step to use half of the available cores
        detector1.ramp_fit.maximum_cores = 'half'  # Set the ramp fitting step to use half of the available cores
        detector1.save_results = True  # Save results to disk
        detector1.input_data=inpfile
        result = detector1.run()

    #def calc_gain_model(self):
    #    '''
    #    make mddel for gain corection from https://exotic-miri.readthedocs.io/en
    #    :return: gain_model
    #    '''
    #    custom_set_gain = SetCustomGain()
    #    # Make custom gain datamodel (using the final segment).
    #    uncal_last = datamodels.RampModel(os.path.join(self.path, self.input_file))
    #    gain_model = custom_set_gain.call(uncal_last, gain_value=3.1)
    #    self.gain_model = gain_model
    #    del uncal_last

    #def calc_linearity_model(self):
    #    '''
    #    make mddel for gain corection from https://exotic-miri.readthedocs.io/en
    #    :return: gain_model
    #    '''
    #    custom_set_linearity = SetCustomLinearity()
    #    # Make custom gain datamodel (using the final segment).
    #    uncal_last = datamodels.RampModel(os.path.join(self.path, self.input_file))
    #    linearity_model = custom_set_linearity.call(uncal_last, group_idx_start_fit=10, group_idx_end_fit=28,
    #        group_idx_start_derive=10, group_idx_end_derive=28, row_idx_start_used=300, row_idx_end_used=380)
    #    del uncal_last
    #    self.linearity_model = linearity_model


if __name__ == '__main__':
    print('Hi PyCharm')
    miri_uncal_file = 'jw05491003001_03102_00001_mirifushort/jw05491003001_03102_00001_mirifushort_uncal.fits'

    exp1 =  detector1(input_file=miri_uncal_file, path = input_dir, output_dir=output_dir)
    exp1.dq_init_step()
    exp1.run_detector1()
    if 0:
        exp1.get_readnoise()
        exp1.show_dq()
        exp1.plot_image()
        #exp1.saturation_step()
        #exp1.reset_dq()
        exp1.reset_step()
        exp1.first_step()
        exp1.last_step()
        exp1.linear_step(debug=False)
        exp1.rscd_step()
        exp1.dark_step()
        exp1.refpix_corr_step()
        exp1.plot_image()
        exp1.jump_corr_step(debug=1,limit=5)
        exp1.slope_fitting_step(debug=True)


#    exp1.show_dq()


