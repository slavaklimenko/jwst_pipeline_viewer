#Modify the path to a directory on your machine
import os
os.environ["CRDS_PATH"] = "/home/slava/science/codes/python/jwst/data"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

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
from jwst.datamodels import dqflags

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

import jwst
print(jwst.__version__)

#define input/output
output_dir = './output/detector1/'
input_dir = './input/detector1/'
miri_uncal_file= 'jw02155001001_04102_00001_mirifulong_uncal.fits'
input_file_base = os.path.basename(miri_uncal_file).replace('uncal.fits', '')

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

def plot_jump(signal, jump_group, xpixel=None, ypixel=None, slope=None):
    """Function to plot the signal up the ramp for a
    pixel and show the location of flagged jumps.

    Parameters
    ----------
    signal : numpy.ndarray
        1D array of signal values

    jump_group : list
        List of boolean values whether a jump is present or
        not in each group

    slope : numpy.ndarray
        1D array of signal values constructed from the slope
    """
    groups = np.arange(len(signal))
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()

    plt.plot(groups, signal, marker='o', color='black')
    plt.plot(groups[jump_group], signal[jump_group], marker='o', color='red',
             label='Flagged Jump')

    if slope is not None:
        plt.plot(groups, slope, marker='o', color='blue', label='Data from slope')

    plt.legend(loc=2)

    plt.xlabel('Groups')
    plt.ylabel('Signal (DN)')
    fig.tight_layout()
    plt.subplots_adjust(top=0.95)

    if xpixel and ypixel:
        plt.title('Pixel (' + str(xpixel) + ',' + str(ypixel) + ')')

def plot_jumps(signals, jump_groups, pixel_loc, slopes=None):
    """Function to plot the ramp and show the jump location
    for several pixels. For simplicity, let's force the input
    number of pixels to be a square.

    Parameters
    ----------
    signals : numpy.ndarray
        2D array (groups x pix) of signal values

    jump_groups : numpy.ndarray
        2D array containing boolean entries for each group of
        each pixel, describing where the jumps were found

    pixel_loc : list
        List of 2-tuples containing the (x, y)
        location of the pixels with the jumps

    slopes : numpy.ndarray
        2D array (groups x pix) of linear signal values
        If not None, these will be overplotted onto the
        plots of signals
    """
    num_group, num_pix = signals.shape
    root = np.sqrt(num_pix)
    if int(root + 0.5) ** 2 != num_pix:
        raise ValueError('Number of pixels input should be a square.')

    root = int(root)
    groups = np.arange(num_group)
    fig, axs = plt.subplots(root, root, figsize=(10, 10))

    for index in range(len(pixel_loc)):
        i = int(index % root)
        j = int(index / root)
        axs[i, j].plot(groups, signals[:, index], marker='o', color='black')
        j_grp = jump_groups[:, index]
        axs[i, j].plot(groups[j_grp], signals[j_grp, index],
                       marker='o', color='red')

        if slopes is not None:
            axs[i, j].plot(groups, slopes[:, index], marker='o', color='blue')

        axs[i, j].set_title('Pixel ({}, {})'.format(pixel_loc[index][1], pixel_loc[index][0]))

    plt.xlabel('Groups')
    plt.ylabel('Signal (DN)')
    fig.tight_layout()

def plot_ramp(groups, signal, xpixel=None, ypixel=None, title=None, ax=None):
    """Function to plot the up the ramp signal for a pixel.

    Parameters
    ----------
    groups : numpy.ndarray
        1D array of group numbers. X-axis values.

    signal : numpy.ndarray
        1D array of pixel signal values.

    xpixel : int
        X-coordinate of the pixel being plotted. Used for legend only.

    ypixel : int
        Y-coordinate of the pixel being plotted. Used for legend only.

    title : str
        String to use for the plot title
    """
    if ax==None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot()
    if xpixel and ypixel:
        ax.plot(groups, signal, marker='o',
                 label='Pixel (' + str(xpixel) + ',' + str(ypixel) + ')')
        ax.legend(loc=2)

    else:
        ax.plot(groups, signal, marker='o')

    ax.set_xlabel('Groups')
    ax.set_ylabel('Signal (DN)')
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.15)

    if title:
        ax.set_title(title)

def plot_ramps(groups, signal1, signal2, label1=None, label2=None, title=None,ax=None):
    """Function to plot the up the ramp signal for two pixels
    on a single plot.

    Parameters
    ----------
    groups : numpy.ndarray
        1D array of group numbers. X-axis values.

    signal1 : numpy.ndarray
        1D array of signal values for first pixel

    signal2 : numpy.ndarray
        1D array of signal values for second pixel

    label1 : str
        Label to place in the legend for pixel1

    label2 : str
        Label to place in the legend for pixel2

    title : str
        String to place in the title of the plot
    """
    if ax == None:
        fig = plt.figure(figsize=(6, 6))
        ax = plt.subplot()
    if label1:
        ax.plot(groups, signal1, marker='o', color='black', label=label1)
    else:
        ax.plot(groups, signal1, marker='o', color='black')
    if label2:
        ax.plot(groups, signal2, marker='o', color='red', label=label2)
    else:
        ax.plot(groups, signal2, marker='o', color='red')
    if label1 or label2:
        ax.legend(loc=2)

    ax.set_xlabel('Groups')
    ax.set_ylabel('Signal (DN)')
    #fig.tight_layout()
    #plt.subplots_adjust(left=0.15)
    #plt.subplots_adjust(top=0.95)

    if title:
        ax.set_title(title)
    return ax

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

def side_by_side(data1, data2, vmin, vmax, title1=None, title2=None, title=None):
    """Show two images side by side for easy comparison. Optionally highlight
    a given pixel with a red dot.

    Parameters
    ----------
    data1 : numpy.ndarray
        First image to be displayed

    data2 : numpy.ndarray
        Second image to be displayed

    vmin : float
        Minimum signal value to use for scaling

    vmax : float
        Maximum signal value to use for scaling

    title1 : str
        Title to use for first (left) plot

    title2 : str
        Title to use for the second (right) plot

    title : str
        String to use for the plot title
    """
    norm = ImageNormalize(data1, interval=ManualInterval(vmin=vmin, vmax=vmax),
                          stretch=LogStretch())

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 8))
    im = axes[0].imshow(data1, origin='lower', norm=norm)
    im = axes[1].imshow(data2, origin='lower', norm=norm)

    axes[0].set_xlabel('Pixel column')
    axes[0].set_ylabel('Pixel row')
    axes[1].set_xlabel('Pixel column')

    if title1:
        axes[0].set_title(title1)
    if title2:
        axes[1].set_title(title2)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='DN')

    if title:
        fig.suptitle(title)

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


        dq_init_step = DQInitStep()
        dq_init_step.output_dir = output_dir
        dq_init_step.save_results = save_results

        # Call the run() method on the uncal file
        self.data = dq_init_step.run(self.path + '/'+ input_file)
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

        if debug and 1:
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
                plot_ramps(groups, full_ramp, saturated_points, label1='', label2='',
                           title='Pixel ({}, {})'.format(x, y), ax=ax[0, axi])

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

    def linear_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
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

        # Call using the the output from the previously-run dq_init step
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

        # Call using the the output from the previously-run dq_init step
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
            refpix_output_file = os.path.join(output_dir, '{}refpixstep.fits'.format(input_file_base))
            show_image(self.data.data[0, 5, :, :], vmin=-1, vmax=10000, title="Difference with/without using side refpix")
            print('REF PIX correction: Done.')
            plt.show()

    def jump_corr_step(self, input_file=None, debug=False, output_dir=None,limit=5,flag_4_neighbors=True,RecalcMedian=True,save_results=False):
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
        jump_step.rejection_threshold = limit
        jump_step.debug=debug
        jump_step.flag_4_neighbors = flag_4_neighbors
        jump_step.recalculate_median = RecalcMedian

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

        if 0:
            jump = self.data
            jump_output_file = os.path.join(output_dir, '{}jumpstep.fits'.format(input_file_base))
            # How many total jump flags were added? Note that some pixels
            # will have more than one group flagged with a jump.
            jump_flags = np.where(jump.groupdq & dqflags.pixel['JUMP_DET'] > 0)
            print('{} jump flags detected.'.format(len(jump_flags[0])))

            # Create a 4-dimensional map of the jump flags
            jump_map = (jump.groupdq & dqflags.pixel['JUMP_DET'] > 0)

            # Collapse down to a 2D map of the number of flagged jumps in each pixel
            jump_map_2d = np.sum(jump_map[0, :, :, :], axis=0)

            # Determine how many pixels have jump flags
            jump_map_indexes = np.where(jump_map_2d > 0)
            impacted_pix = np.sum(jump_map_2d > 0)
            total_pix = 2048 * 2048
            print(('{} pixels ({:.2f}% of the detector) have been flagged with '
                   'at least one jump.'.format(impacted_pix, 100. * impacted_pix / total_pix)))
            # The jump map is 4-dimensional, just like the science data
            print('jump_map.shape', jump_map.shape)

            # Create an array of group numbers to plot against
            group_indexes = np.arange(jump_map.shape[1]).astype(int)

            # Pick one pixel with a flagged jump, and find the group(s) with the jump flags
            j_index = 5112
            jumpy = jump_map_indexes[0][j_index]
            jumpx = jump_map_indexes[1][j_index]
            jump_grp = jump_map[0, :, jumpy, jumpx]
            print('Jump located in group(s) {} of pixel ({}, {})'.format(group_indexes[jump_grp], jumpx, jumpy))

            # Plot the signal up the ramp for this pixel
            plot_jump(jump.data[0, :, jumpy, jumpx], jump_grp, xpixel=jumpx, ypixel=jumpy)

            plt.show()

            indexes_to_plot = [200, 401, 600, 1202, 1400, 8112, 7888, 5555, 2222]
            jump_data = np.zeros((jump.shape[1], len(indexes_to_plot)))
            jump_grps = np.zeros((jump.shape[1], len(indexes_to_plot))).astype(bool)
            jump_locs = []
            for counter, idx in enumerate(indexes_to_plot):
                # integ, grp, y, x = jump_flags[idx]
                y = jump_map_indexes[0][idx]
                x = jump_map_indexes[1][idx]
                grp = jump_map[0, :, y, x]

                jump_data[:, counter] = jump.data[0, :, y, x]
                jump_grps[:, counter] = grp
                jump_locs.append((x, y))
            plot_jumps(jump_data, jump_grps, jump_locs)
            plt.show()
        #return jump

    def slope_fitting_step(self, input_file=None, debug=True, output_dir=None,save_results=False):
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

        # Let's save the optional outputs, in order
        # to help with visualization later
        ramp_fit_step.save_opt = save_results

        # Call using the dark instance from the previously-run
        # jump step
        self.ramp_fit = ramp_fit_step.run(input_file)
        if 0:
            rampfit_output_file = os.path.join(output_dir, '{}_0_rampfitstep.fits'.format(input_file_base))
            print(ramp_fit[0].shape, ramp_fit[1].shape)

            # Generate the name of the optional output file
            optional_file = os.path.join(output_dir, '{}fitopt.fits'.format(input_file_base))
            # Open the file and examine the extensions
            hdulist = fits.open(optional_file)
            print(hdulist.info())
            intercepts = hdulist['YINT'].data[0, 0, :, :]
            hdulist.close()

            # Get the exposure time associated with each group
            num_groups = ramp_fit[0].meta.exposure.ngroups
            group_time = ramp_fit[0].meta.exposure.group_time
            group_times = np.arange(num_groups) * group_time
            # Reconstruct linear ramps from the slope and intercept values
            indexes_to_plot = [200, 401, 600, 1202, 1400, 8112, 7888, 5555, 2222]
            jump = ramp_fit.data
            lin_ramps = np.zeros((jump.shape[1], len(indexes_to_plot)))
            for counter, idx in enumerate(indexes_to_plot):
                y = jump_map_indexes[0][idx]
                x = jump_map_indexes[1][idx]
                grp = jump_map[0, :, y, x]

                rate = ramp_fit[0].data[y, x]
                intercept = intercepts[y, x]
                lin_ramps[:, counter] = intercept + (rate * group_times)
            # Reconstruct the linear ramp for the single pixel we showed after the jump step
            lin_data = intercepts[jumpy, jumpx] + (ramp_fit[0].data[jumpy, jumpx] * group_times)
            # Plot again the single pixel from the jump step, along with its
            # reconstructed best-fit linear fit
            plot_jump(jump.data[0, :, jumpy, jumpx], jump_grp, xpixel=jumpx,
                      ypixel=jumpy, slope=lin_data)
            plt.show()
            show_image(ramp_fit[0].data, -1, 1)
            plt.show()

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

if __name__ == '__main__':
    print('Hi PyCharm')
    miri_uncal_file = 'jw02155001001_04102_00001_mirifulong_uncal.fits'

    exp1 =  detector1(input_file=miri_uncal_file, path = input_dir, output_dir=output_dir)
    exp1.dq_init_step()
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


