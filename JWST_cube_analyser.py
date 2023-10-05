import sys, os
import shutil
from astropy.io import ascii
import astropy.constants as ac
from astropy.cosmology import FlatLambdaCDM
from functools import partial
from io import StringIO
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import pickle
from PyQt5.QtCore import (Qt, )
from PyQt5.QtGui import (QFont, )
from PyQt5.QtWidgets import (QApplication, QMessageBox, QMainWindow, QSplitter, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QPushButton, QHeaderView, QCheckBox,
                             QRadioButton, QButtonGroup, QComboBox, QTableView, QLineEdit, QSlider)
import pyqtgraph as pg
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, Rbf
from scipy.interpolate import RBFInterpolator
from scipy.optimize import bisect
import sys
sys.path.append('/home/slava/science/codes/python')
from stage3_pipeline import *
from stage3_pipeline import detector3
from stage2_pipeline import *
from stage2_pipeline import detector2

#from spectro.stats import distr2d,distr1d
#from spectro.a_unc import a
#from spectro.pyratio import pyratio
import copy
from stdatamodels.jwst.datamodels import dqflags
from PyQt5.QtWidgets import (QApplication, QMessageBox, QMainWindow, QWidget,
                             QFileDialog, QTextEdit, QVBoxLayout,
                             QSplitter, QFrame, QLineEdit, QLabel, QPushButton, QCheckBox,
                             QGridLayout, QTabWidget, QFormLayout, QHBoxLayout, QRadioButton,
                             QTreeWidget, QComboBox, QTreeWidgetItem, QAbstractItemView,
                             QStatusBar, QMenu, QButtonGroup, QMessageBox, QToolButton, QColorDialog)
from pyqtgraph.Qt import QtCore, QtGui
from stdatamodels.jwst import datamodels
import csv
from PyQt5.QtWidgets import (QApplication)
from astropy import modeling
from scipy import signal
import scipy.signal

output_dir = './output/detector3/'
input_dir = './output/detector2/'
miri_uncal_file= 'jw02155001001_04102_00001_mirifulong_uncal.fits'
input_file_base = os.path.basename(miri_uncal_file).replace('uncal.fits', '')

class image():
    """
    class for working with images (2d spectra) inside Spectrum plotting
    """
    def __init__(self, x=None, y=None, z=None, err=None, mask=None):
        if any([v is not None for v in [x, y, z, err, mask]]):
            self.set_data(x=x, y=y, z=z, err=err, mask=mask)
        else:
            self.z = None

    def set_data(self, x=None, y=None, z=None, err=None, mask=None):
        for attr, val in zip(['z', 'err', 'mask'], [z, err, mask]):
            if val is not None:
                setattr(self, attr, np.asarray(val))
            else:
                setattr(self, attr, val)
        if x is not None:
            self.x = np.asarray(x)
        else:
            self.x = np.arange(z.shape[0])
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = np.arange(z.shape[1])

        self.pos = [self.x[0] - (self.x[1] - self.x[0]) / 2, self.y[0] - (self.y[1] - self.y[0]) / 2]
        self.scale = [(self.x[-1] - self.x[0]) / (self.x.shape[0]-1), (self.y[-1] - self.y[0]) / (self.y.shape[0]-1)]
        for attr in ['z', 'err']:
            self.getQuantile(attr=attr)
            self.setLevels(attr=attr)


    def getQuantile(self, quantile=0.997, attr='z'):
        if getattr(self, attr) is not None:
            x = np.sort(getattr(self, attr).flatten())
            x = x[~np.isnan(x)]
            setattr(self, attr+'_quantile', [x[int(len(x)*(1-quantile)/2)], x[int(len(x)*(1+quantile)/2)]])
        else:
            setattr(self, attr + '_quantile', [0, 1])

    def setLevels(self, bottom=None, top=None, attr='z'):
        quantile = getattr(self, attr+'_quantile')
        if bottom is None:
            bottom = quantile[0]
        if top is None:
            top = quantile[1]
        top, bottom = np.max([top, bottom]), np.min([top, bottom])
        if top - bottom < (quantile[1] - quantile[0]) / 100:
            top += ((quantile[1] - quantile[0]) / 100 - (top - bottom)) /2
            bottom -= ((quantile[1] - quantile[0]) / 100 - (top - bottom)) / 2
        setattr(self, attr+'_levels', [bottom, top])

    def find_nearest(self, x, y, attr='z'):
        z = getattr(self, attr)
        if len(z.shape) == 2:
            return z[np.min([z.shape[0]-1, (np.abs(self.y - y)).argmin()]), np.min([z.shape[1]-1, (np.abs(self.x - x)).argmin()])]
        else:
            return None



class plotCube(pg.ImageView): #(pg.PlotWidget):
    def __init__(self, parent,cube_name='A'):
        self.parent = parent
        self.cube_name = cube_name
        self.img = pg.ImageView.__init__(self, name='Cube image', view=pg.PlotItem(), discreteTimeLine=True)
        self.initstatus()
        self.vb = self.getView()
        self.roi = self.getRoiPlot()

        ## Set a custom color map
        cmap = pg.colormap.get('CET-D7') #pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        self.setColorMap(cmap)
        self.vb.invertY(False)
        self.cursorpos = pg.TextItem(anchor=(0, 1))
        self.vb.setLabel(axis='left', text='Y-axis')
        self.vb.setLabel(axis='bottom', text='X-axis')
        #self.setHistogramLabel("HiSt")
        self.label = QLabel("Pixel:", self.ui.graphicsView.viewport())
        self.label.move(25, 70)
        self.label2 = QLabel("World:", self.ui.graphicsView.viewport())
        self.label2.move(25, 35)
        self.label3 = QLabel("World:", self.ui.graphicsView.viewport())
        self.label3.move(25, 0)
        self.label_filename = QLabel("Filename:", self.ui.graphicsView.viewport())
        self.label_filename.move(1325, 15)

    def add_scale_stick(self):
        if self.cube_name == 'A':
            wcs1 = self.parent.CUBE_A.data.wcs
            #lam_world, x_world, y_world = self.parent.CUBE_A.conv_world_coord(t=self.lam, x=int(self.x), y=int(self.y))
        elif self.cube_name == 'B':
            wcs1 = self.parent.CUBE_B.data.wcs
        delta_x =  wcs1['CDELT1']
        delta_y = wcs1['CDELT2']
        scale = 1/3600
        stick = [[25,25+scale/delta_x],[10,10]]
        self.scale_bar = pg.PlotCurveItem(size=50, pen=pg.mkPen('white', width=5))
        self.scale_bar.setData(stick[0], stick[1])
        #self.scale_bar.setSymbol('o')
        self.vb.addItem(self.scale_bar)


    def initstatus(self):
        self.s_status = True
        self.selected_point = None
        self.selected_pixels_saturated = []
        self.selected_pixels_dnu = []
        self.selected_pixels_cr = []
        self.selected_pixels_cr_multi = []


    def add(self, name, add,mode=None,show_associated_galaxies =True):
        if add:
            if mode == None:
                print('add cube name:',name)
                self.init_name = name
                if self.cube_name == 'A':
                    filename = self.parent.Cubes_A.filelist[name]
                    self.parent.CUBE_A.add_cube(cubename=filename)
                    self.parent.CUBE_A.init_cube()
                    self.data = self.parent.CUBE_A.data.data
                    self.data_tot = self.parent.CUBE_A.data
                elif self.cube_name == 'B':
                    filename = self.parent.Cubes_B.filelist[name]
                    self.parent.CUBE_B.add_cube(cubename=filename)
                    self.parent.CUBE_B.init_cube()
                    self.data = self.parent.CUBE_B.data.data
                    self.data_tot = self.parent.CUBE_B.data
                self.label_filename.setText(filename.split('/')[-1].split('s3d')[0])
                self.label_filename.resize(600, 40)

                t,x, y = (0,2, 1)
                self.setImage(self.data, axes={'t': t, 'x': x, 'y': y, 'c': None}, levels=[-100,800]) #autoRange=True,
                self.vb.hoverEvent = self.imageHoverEvent
                hist = self.getHistogramWidget()
                hist.setHistogramRange(mn=-200, mx=1000)

                (self.t, self.time) = self.timeIndex(self.timeLine)


                if 1:
                    self.roi_list = []
                    if self.cube_name == 'A':
                        roi_colors = ['lightgreen','red','magenta']
                    elif self.cube_name == 'B':
                        roi_colors = ['purple','yellow','magenta']
                    self.roi_list.append(pg.EllipseROI([15, 15], [10, 10], pen=pg.mkPen(roi_colors[0], width=2)))
                    self.roi_list.append(pg.CircleROI([30, 30], [10, 10], pen=pg.mkPen(roi_colors[1], width=2), movable=True, resizable=True))
                    #self.roi_list.append(pg.RectROI([15, 15], [10, 10], pen=pg.mkPen(roi_colors[2], width=3)))
                    self.roi_list[-1].addRotateHandle([1, 0], [0.5, 0.5])
                    #self.roi_list = rois
                    #self.roi_mask = {}
                    #rois.append(pg.EllipseROI([20, 20], [12, 12], pen=(9, 2)))

                    def updateRoi(roi):
                        if roi is None:
                            return
                        arr1 = roi.getArrayRegion(arr=self.data, img=self.imageItem , axes=(2,1))
                        if 0:   #set_roi_mask
                            mask = np.array(np.zeros((self.data.shape[1],self.data.shape[2])), dtype='bool')
                            rows,cols = self.data.shape[1],self.data.shape[2]
                            m = np.mgrid[:rows, :cols]
                            possx = m[0, :, :]  # make the x pos array
                            #possx2 = (np.mgrid[:cols, :rows])[0,:,:]
                            possy = m[1, :, :]  # make the y pos array
                            possx.shape = rows,cols
                            possy.shape = rows,cols
                            mpossx = roi.getArrayRegion(arr=possx, img=self.imageItem , axes=(1,0)).astype(int)
                            mpossy = roi.getArrayRegion(arr=possy, img=self.imageItem, axes=(1, 0)).astype(int)
                            mpossx1 = mpossx[np.nonzero(mpossx*mpossy)]  # get the x pos from ROI
                            mpossy1 = mpossy[np.nonzero(mpossy*mpossx)]  # get the y pos from ROI
                            for i in range(mpossx.shape[0]):
                                print(mpossx[i,:])
                            print(np.nanmin(mpossx1),np.nanmax(mpossx1),np.size(mpossx1))
                            for i in range(mpossy.shape[0]):
                                print(mpossy[i,:])
                            print(np.nanmin(mpossy1), np.nanmax(mpossy1),np.size(mpossy1))
                            print(mask.shape)
                            mask[mpossx1, mpossy1] = True #self.data[0,mpossx, mpossy]>0
                            roi.roi_mask = mask
                        if 1:
                            rows, cols = self.data.shape[1], self.data.shape[2]
                            print('roi state',roi.state)
                            rc = roi.state['pos']
                            ra,rb =roi.state['size'][0]+1,roi.state['size'][1]
                            rt = roi.state['angle']/180*np.pi
                            rmask = np.zeros((rows,cols))
                            def ellipse_functiion(x=1,y=1,x0=0,y0=0,a=1,b=1):
                                f = (x+1-(x0+a/2))**2/(a/2)**2 + (y+1-(y0+b/2))**2/(b/2)**2
                                return f
                            for i in range(rmask.shape[0]):
                                for j in range(rmask.shape[1]):
                                    if ellipse_functiion(j,i,rc[0],rc[1],ra,rb)<1:
                                        rmask[i,j]= True
                            roi.roi_mask =rmask.astype(bool)
                        if 0:  # set_roi_mask
                            mask = np.array(np.zeros((self.data.shape[1], self.data.shape[2])), dtype='bool')
                            rows, cols = self.data.shape[1], self.data.shape[2]
                            m = np.mgrid[:rows, :cols]
                            possx = m[0, :, :]  # make the x pos array
                            # possx2 = (np.mgrid[:cols, :rows])[0,:,:]
                            possy = m[1, :, :]  # make the y pos array
                            possx.shape = rows, cols
                            possy.shape = rows, cols
                            mpossx = roi.getArrayRegion(arr=possx, img=self.imageItem, axes=(0, 1)).astype(int)
                            mpossx2 = roi.getArrayRegion(possx, self.imageItem).astype(float)
                            mpossx1 = mpossx[np.nonzero(mpossx)]  # get the x pos from ROI
                            mpossy = roi.getArrayRegion(arr=possy, img=self.imageItem, axes=(0, 1)).astype(int)
                            mpossy1 = mpossy[np.nonzero(mpossy)]  # get the y pos from ROI
                            mask[mpossy1, mpossx1] = True  # self.data[0,mpossx, mpossy]>0
                            roi.roi_mask = mask
                        updateRoiPlot(roi, arr1)



                    def updateRoiPlot(roi, data_roi=None):
                        if data_roi is None:
                            data_roi = roi.getArrayRegion(arr=self.data, img=self.imageItem , axes=(1,2))
                            #data = roi.getArrayRegion(im1.image, img=im1)
                        if data_roi is not None and 0:
                            #d= data_roi.mean(axis=(1,2))
                            #roi.curve.setData(data_roi.mean(axis=(1,2)))
                            roi.curve.setData(data_roi.mean(axis=(1, 2)))


                        if np.sum(roi.roi_mask)>0:
                            roi_selected_flux=np.zeros(self.data.shape[0])
                            roi_selected_flux_err = np.zeros(self.data.shape[0])
                            roi_mean_w_flux = np.zeros(self.data.shape[0])
                            roi_mean_w_flux_err = np.zeros(self.data.shape[0])
                            roi_max_flux = np.zeros(self.data.shape[0])
                            roi_max_flux_err = np.zeros(self.data.shape[0])
                            for i in range(self.data.shape[0]):
                                d = self.data[i,:,:]
                                d = d[roi.roi_mask]
                                d = d[~np.isnan(d)]
                                if np.sum([~np.isnan(d)])>0:
                                    derr = (self.data_tot.err[i, :, :])[roi.roi_mask]
                                    derr = derr[~np.isnan(derr)]
                                    w = np.power(derr, 2)
                                    num_pixels = np.size(d)
                                    roi_selected_flux[i] = np.nansum(d)
                                    roi_selected_flux_err[i] = np.power(np.sum(w),0.5)
                                    j = np.argwhere(d==np.max(d))[0]
                                    roi_max_flux[i] = d[j]
                                    roi_max_flux_err[i] = derr[j]
                                else:
                                    roi_selected_flux[i] = np.nan
                                    roi_selected_flux_err[i] = 1
                                    roi_max_flux[i] = np.nan
                                    roi_max_flux_err[i] = 1
                            roi.curve.setData(roi_selected_flux)
                            if self.parent.exp_commands.norm_flag_roi.currentText() == 'yes':
                                normalize = True
                            else:
                                normalize = False
                            if roi == self.roi_list[0]:
                                if self.cube_name == 'A':
                                    self.parent.plot_spectrum.plot_specA1(data=roi_selected_flux,err=roi_selected_flux_err, add=False,show_err_bar=True)
                                    self.parent.plot_spectrum.plot_specA1(data=roi_selected_flux,err=roi_selected_flux_err,  pen='gray',normalize=normalize,label='mean',show_err_bar=True,smoothing=False)
                                    self.parent.plot_spectrum.plot_specAmax(data=roi_mean_w_flux, add=False)
                                    self.parent.plot_spectrum.plot_specAmax(data=roi_max_flux,  err=roi_max_flux_err, normalize=normalize,label='max')
                                    self.parent.plot_hist1.plot_hist(data=self.data_tot, roi_mask=roi.roi_mask, timeind=self.time, add=False)
                                    self.parent.plot_hist1.plot_hist(data=self.data_tot, roi_mask=roi.roi_mask, timeind=self.time,pen=roi.pen)
                                if self.cube_name == 'B':
                                    self.parent.plot_spectrum.plot_specB1(data=roi_selected_flux,err=roi_selected_flux_err, add=False,show_err_bar=True)
                                    self.parent.plot_spectrum.plot_specB1(data=roi_selected_flux, err=roi_selected_flux_err,pen=roi.pen,normalize=normalize,show_err_bar=True,smoothing=False)
                                    self.parent.plot_spectrum.plot_specB_median(add=False)
                                    self.parent.plot_spectrum.plot_specB_median(pen=pg.mkPen('red', width=1.5),normalize=normalize,npixels = np.sum(roi.roi_mask),smoothing=False)
                                    self.parent.plot_hist3.plot_hist(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                     timeind=self.time, add=False)
                                    self.parent.plot_hist3.plot_hist(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                     timeind=self.time, pen=roi.pen,brush='purple')
                            elif roi == self.roi_list[1]:
                                if self.cube_name == 'A':
                                    self.parent.plot_spectrum.plot_specA2(data=roi_selected_flux, add=False,show_err_bar=True)
                                    self.parent.plot_spectrum.plot_specA2(data=roi_selected_flux,pen=roi.pen,err=roi_selected_flux_err,normalize=normalize,show_err_bar=True)
                                    self.parent.plot_hist2.plot_hist(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                     timeind=self.time, add=False)
                                    self.parent.plot_hist2.plot_hist(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                     timeind=self.time, pen=roi.pen)
                                    self.parent.plot_hist1.plot_hist2(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                     timeind=self.time, add=False)
                                    self.parent.plot_hist1.plot_hist2(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                     timeind=self.time, pen=roi.pen)
                                    self.parent.plot_hist4.plot_hist2(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                      timeind=self.time, add=False)
                                    self.parent.plot_hist4.plot_hist2(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                      timeind=self.time, pen=roi.pen)
                                    if self.parent.exp_commands.show_roi_1_minus_2.isChecked() == True:
                                        roi1,roi2 = self.roi_list[0],self.roi_list[1]
                                        mask_2m1 = roi2.roi_mask ^ (roi2.roi_mask*roi1.roi_mask)
                                        if np.sum(mask_2m1)>0 and np.sum(mask_2m1)!=np.sum(roi2.roi_mask):
                                            roi_diff_flux,roi_diff_flux_err = np.zeros(self.data.shape[0]),np.zeros(self.data.shape[0])
                                            for i in range(self.data.shape[0]):
                                                d =  self.data[i,:,:]
                                                d = d[mask_2m1]
                                                d = d[~np.isnan(d)]
                                                derr = (self.data_tot.err[i, :, :])[mask_2m1]
                                                derr = derr[~np.isnan(derr)]
                                                w = np.power(derr, 2)
                                                roi_diff_flux[i] = np.nansum(d)
                                                roi_diff_flux_err[i] = np.power(np.sum(w), 0.5)
                                            self.parent.plot_spectrum.plot_specA2m1(data=roi_diff_flux, add=False, show_err_bar=True)
                                            self.parent.plot_spectrum.plot_specA2m1(data=roi_diff_flux,err=roi_diff_flux_err,normalize=normalize,show_err_bar=True)


                                if self.cube_name == 'B':
                                    self.parent.plot_spectrum.plot_specB2(data=roi_selected_flux, add=False)
                                    self.parent.plot_spectrum.plot_specB2(data=roi_selected_flux, pen=roi.pen,normalize=normalize)
                                    self.parent.plot_hist4.plot_hist(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                     timeind=self.time, add=False)
                                    self.parent.plot_hist4.plot_hist(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                     timeind=self.time, pen=roi.pen,brush='yellow')
                                    self.parent.plot_hist3.plot_hist2(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                      timeind=self.time, add=False)
                                    self.parent.plot_hist3.plot_hist2(data=self.data_tot, roi_mask=roi.roi_mask,
                                                                      timeind=self.time, pen=roi.pen,brush='yellow')

                    ## Add each ROI to the scene and link its data to a plot curve with the same color
                    for r in self.roi_list:
                        self.vb.addItem(r)
                        c = self.roi.plot(pen=r.pen)
                        r.curve = c
                        r.sigRegionChanged.connect(updateRoi)








                    #self.timeLine.sigPositionChanged.connect(updatePlotSpec)

                # add galaxies
                if show_associated_galaxies:
                    gal1 = [(2+38/60 + 38.94/3600)/24*360,(16+36/60+57.3/3600)]
                    gal2 =[(2+38/60 + 39.01/3600)/24*360,(16+36/60+59.2/3600)]
                    if self.cube_name == 'A':
                        x1,y1 = self.parent.CUBE_A.conv_pix_2_world_coord(x_world=gal1[0],y_world=gal1[1])
                        x2, y2 = self.parent.CUBE_A.conv_pix_2_world_coord(x_world=gal2[0], y_world=gal2[1])
                    elif self.cube_name == 'B':
                        x1, y1 = self.parent.CUBE_B.conv_pix_2_world_coord(x_world=gal1[0], y_world=gal1[1])
                        x2, y2 = self.parent.CUBE_B.conv_pix_2_world_coord(x_world=gal2[0], y_world=gal2[1])
                    self.gal1 = pg.ScatterPlotItem(size=20, pen=pg.mkPen('red', width=3))
                    self.gal2 = pg.ScatterPlotItem(size=20, pen=pg.mkPen('red', width=3))
                    self.gal1.setData([x1], [y1])
                    self.gal1.setSymbol('star')
                    self.gal2.setData([x2], [y2])
                    self.gal2.setSymbol('star')
                    self.vb.addItem(self.gal1)
                    self.vb.addItem(self.gal2)
                    self.add_scale_stick()


        else:
            try:
                #self.roi.clearPoints()
                for r in self.roi_list:
                    self.vb.removeItem(r)
                    self.roi.removeItem(r.curve)
                self.vb.removeItem(self.gal1)
                self.vb.removeItem(self.gal2)
                self.roi.plot()
                del self.view[name]
                self.roi_list= []

                self.vb.removeItem(self.view[name])
                self.legend.removeItem(self.view[name])
                del self.view[name]
                del self.roi

            except:
                pass

    def add_Roi(self,add=False):
        if add:
            self.roi_add = pg.RectROI([25, 25], [10, 5], pen=pg.mkPen('blue', width=3))
            self.roi_add.addRotateHandle([1, 0], [0.5, 0.5])
            self.vb.addItem(self.roi_add)
            if 1:
                def updateRoi(roi):
                    if roi is None:
                        return
                    arr1 = roi.getArrayRegion(data=self.data, img=self.imageItem, axes=(2, 1))
                    if 1:  # set_roi_mask
                        (t, time) = self.timeIndex(self.timeLine)
                        mask = np.array(np.zeros((self.data.shape[1], self.data.shape[2])), dtype='bool')
                        rows, cols = self.data.shape[1], self.data.shape[2]
                        m = np.mgrid[:rows, :cols]
                        possx = m[0, :, :]  # make the x pos array
                        possy = m[1, :, :]  # make the y pos array
                        possx.shape = rows, cols
                        possy.shape = rows, cols
                        mpossx = roi.getArrayRegion(data=possx, img=self.imageItem, axes=(1, 0)).astype(int)
                        mpossy = roi.getArrayRegion(data=possy, img=self.imageItem, axes=(1, 0)).astype(int)
                        #mask_mposs = (mpossx > 0) * (mpossy > 0)
                        mask_mposs =~np.isnan(self.data[time, mpossx, mpossy])
                        mpossx1 = mpossx[mask_mposs] # get the x pos from ROI
                        mpossy1 = mpossy[mask_mposs]  # get the y pos from ROI
                        mask[mpossx1, mpossy1] = True  # self.data[0,mpossx, mpossy]>0
                        roi.roi_mask = mask
                        if 1:
                            bottom_line = [mpossy[:,0],mpossx[:,0]]
                            mask_bottom_line = ~np.isnan(self.data[time, bottom_line[1], bottom_line[0]])
                            bottom_line[0] = (bottom_line[0])[mask_bottom_line]
                            bottom_line[1] = (bottom_line[1])[mask_bottom_line]
                            if np.sum(mask_bottom_line)>0:
                                mask_mposs = (mpossx>0)*(mpossy>0)
                                mpossx1 = mpossx[mask_mposs].flatten()
                                mpossy1 = mpossy[mask_mposs].flatten()
                                posA,posB = [bottom_line[0][0],bottom_line[1][0]],[bottom_line[0][-1],bottom_line[1][-1]]
                                alpha = np.arcsin((posB[1]-posA[1])/np.power((posB[0]-posA[0])**2+(posB[1]-posA[1])**2,0.5))
                                print('alpha',alpha,alpha/3.14*180)

                                def conv_coord(x0,y0,xc,yc,theta):
                                    dx = x0-xc
                                    dy = y0-yc
                                    x1 = dx*np.cos(theta) + dy*np.sin(theta)
                                    y1 = -dx*np.sin(theta) + dy*np.cos(theta)
                                    return np.array(x1).astype(int), np.array(y1).astype(int)
                                #XX,YY = conv_coord(mpossy1,mpossx1,posA[0],posA[1],alpha)
                                XX, YY = conv_coord(mpossy1, mpossx1, mpossy1[0], mpossx1[0], alpha)
                                roi.align_coord = [XX,YY]
                                roi.slit_array = np.zeros((np.max(XX)+1,np.max(YY)+1))

                                #print('time slit',time)
                                for i in range(np.size(XX)):
                                    roi.slit_array[XX[i],YY[i]] = self.data[time, mpossx1[i], mpossy1[i]]
                                if 0:
                                    print(XX,YY)
                                    fig,ax = plt.subplots()
                                    ax.plot(XX,YY,'o')
                                    ax.plot(mpossy1,mpossx1,'o')
                                    ax.plot([posA[0],posB[0]], [posA[1],posB[1]], '-')
                                    ax.set_xlim(0,40)
                                    ax.set_ylim(0, 40)
                                    plt.show()

                    updateRoiPlot(roi)

                def updateRoiPlot(roi):
                    if np.nansum(roi.slit_array) > 0:
                        slit_profile = np.array([np.nansum(roi.slit_array[i,:])/np.sum(~np.isnan(roi.slit_array[i,:])) for i in range(roi.slit_array.shape[0])])
                        if self.cube_name == 'A':
                            self.parent.plot_slit.plot_slit(data=slit_profile, add=False)
                            self.parent.plot_slit.plot_slit(data=slit_profile, add=True)

                self.roi_add.sigRegionChanged.connect(updateRoi)
        else:
            self.vb.removeItem(self.roi_add)
    def add_from_file(self, name='Median',filename='test_cube', add=True):
        if add:
            print('add cube name:',filename)
            self.init_name = name
            self.cube = detector3()
            self.cube.add_cube(cubename=filename)
            self.cube.init_cube(read_spectum=False)
            self.data = self.cube.data.data
            self.data_tot = self.cube.data
            self.label_filename.setText(filename.split('/')[-1].split('s3d')[0])
            self.label_filename.resize(600, 40)

            t,x, y = (0,2, 1)
            self.setImage(self.data, axes={'t': t, 'x': x, 'y': y, 'c': None}, levels=[-100,800]) #autoRange=True,
            self.vb.hoverEvent = self.imageHoverEvent
            hist = self.getHistogramWidget()
            hist.setHistogramRange(mn=-200, mx=1000)

            (self.t, self.time) = self.timeIndex(self.timeLine)


            if 1:
                self.roi_list = []
                roi_colors = ['lightgreen','red']
                self.roi_list.append(pg.EllipseROI([15, 15], [10, 10], pen=pg.mkPen(roi_colors[0], width=3)))
                self.roi_list.append(pg.CircleROI([30, 30], [10, 10], pen=pg.mkPen(roi_colors[1], width=2), movable=True, resizable=True))

                def updateRoi(roi):
                    if roi is None:
                        return
                    arr1 = roi.getArrayRegion(arr=self.data, img=self.imageItem , axes=(2,1))
                    if 1:   #set_roi_mask
                        mask = np.array(np.zeros((self.data.shape[1],self.data.shape[2])), dtype='bool')
                        rows,cols = self.data.shape[1],self.data.shape[2]
                        m = np.mgrid[:rows, :cols]
                        possx = m[0, :, :]  # make the x pos array
                        #possx2 = (np.mgrid[:cols, :rows])[0,:,:]
                        possy = m[1, :, :]  # make the y pos array
                        possx.shape = rows,cols
                        possy.shape = rows,cols
                        mpossx = roi.getArrayRegion(arr=possx, img=self.imageItem , axes=(1,0)).astype(int)
                        mpossx2 = roi.getArrayRegion(arr=possx, img=self.imageItem , axes=(1,0)).astype(float)
                        print(mpossx2)
                        mpossx1 = mpossx[np.nonzero(mpossx)]  # get the x pos from ROI
                        mpossy = roi.getArrayRegion(arr=possy, img = self.imageItem , axes=(1,0)).astype(int)
                        mpossy1 = mpossy[np.nonzero(mpossy)]  # get the y pos from ROI
                        mask[mpossx1, mpossy1] = True #self.data[0,mpossx, mpossy]>0
                        roi.roi_mask = mask

                    updateRoiPlot(roi, arr1)



                def updateRoiPlot(roi, data_roi=None):
                    if data_roi is None:
                        data_roi = roi.getArrayRegion(arr=self.data, img=self.imageItem , axes=(1,2))
                    if data_roi is not None:
                        roi.curve.setData(data_roi.mean(axis=(1,2)))



                ## Add each ROI to the scene and link its data to a plot curve with the same color
                for r in self.roi_list:
                    self.vb.addItem(r)
                    c = self.roi.plot(pen=r.pen)
                    r.curve = c
                    r.sigRegionChanged.connect(updateRoi)





        else:
            try:
                for r in self.roi_list:
                    self.vb.removeItem(r)
                    self.roi.removeItem(r.curve)
                self.roi.plot()
                del self.view[name]
                self.roi_list= []

                self.vb.removeItem(self.view[name])
                self.legend.removeItem(self.view[name])
                del self.view[name]
                del self.roi
            except:
                pass

    def updateRoiRadius(self,roi):
        def updateRoi(roi):
            if roi is None:
                return
            arr1 = roi.getArrayRegion(arr=self.data, img=self.imageItem, axes=(2, 1))
            if 1:  # set_roi_mask
                mask = np.array(np.zeros((self.data.shape[1], self.data.shape[2])), dtype='bool')
                rows, cols = self.data.shape[1], self.data.shape[2]
                m = np.mgrid[:rows, :cols]
                possx = m[0, :, :]  # make the x pos array
                # possx2 = (np.mgrid[:cols, :rows])[0,:,:]
                possy = m[1, :, :]  # make the y pos array
                possx.shape = rows, cols
                possy.shape = rows, cols
                mpossx = roi.getArrayRegion(arr=possx, img=self.imageItem, axes=(1, 0)).astype(int)
                mpossx2 = roi.getArrayRegion(arr=possx, img=self.imageItem, axes=(1, 0)).astype(float)
                mpossx1 = mpossx[np.nonzero(mpossx)]  # get the x pos from ROI
                mpossy = roi.getArrayRegion(arr=possy, img=self.imageItem, axes=(1, 0)).astype(int)
                mpossy1 = mpossy[np.nonzero(mpossy)]  # get the y pos from ROI
                mask[mpossx1, mpossy1] = True  # self.data[0,mpossx, mpossy]>0
                roi.roi_mask = mask

        if roi is None:
            return
        if self.cube_name == 'A':
            self.data = self.parent.CUBE_A.data.data
        elif self.cube_name == 'B':
            self.data = self.parent.CUBE_B.data.data
        data_mean = np.nanmean(self.data,axis=0)
        pos = np.argwhere(data_mean == np.nanmax(data_mean))[0]
        size = 2*int(self.parent.exp_commands.roi_radius.text())
        s = roi.size()
        print('center roi (max):', pos, ' size=',size)
        roi.setSize(size=(size,size))
        roi.setPos(pos=(pos[1]-size/2+0.5,pos[0]-size/2+0.5))
        #updateRoi(roi)

    def add_contours(self,level=1-0.68,add=False,local=True):
        if add:
            #self.image_levels = pg.ScatterPlotItem(size=20, pen=pg.mkPen('white', width=3))
            #self.image_levels.setSymbol('o')
            if self.cube_name == 'A':
                image = self.parent.CUBE_A.data.data
            elif self.cube_name == 'B':
                image = self.parent.CUBE_B.data.data

            if local:
                (self.t, self.time) = self.timeIndex(self.timeLine)
                image_comb = image[self.time,:,:]
            else:
                image_comb = np.nanmean(image, axis=0)
            x,y = np.arange(image_comb.shape[0]),np.arange(image_comb.shape[1])
            if 0:
                from scipy import interpolate, integrate, optimize
                def func(level, conf=0.683, x=None, y=None, z=None):
                    zs = np.copy(z)
                    zs[zs < level] = 0
                    return integrate.simps(integrate.simps(zs, y, axis=0), x) - conf


                def level(conf=0.683,x=None, y=None,z=None):
                    """
                    Level of pdf at given confidence level
                    parameters:
                        - conf           :  confidence level

                    return: level
                        - level          :  pdf value above pdf contains conf level of probability
                    """
                    x1, y1 = np.linspace(np.min(x), np.max(x), 300), np.linspace(np.min(y), np.max(y),300)
                    inter = interpolate.interp2d(x, y, z, kind='cubic', fill_value=0)
                    z1 = inter(x1, y1)
                    if 1:
                        res = optimize.bisect(self.func, 0, self.zmax, args=(conf, x1, y1, z1), xtol=self.xtol,
                                              disp=self.debug)

                    if self.debug:
                        print('fsolve:', res)

                    return res

                l1 = level(x=x,y=y,z=image_comb)

            plt.subplots()
            plt.imshow(image_comb/np.nanmax(image_comb))
            if 0:
                from scipy import interpolate, integrate, optimize
                z=image_comb/np.nanmax(image_comb)
                z[np.isnan(z)] = 0
                x1, y1 = np.linspace(np.min(x), np.max(x), 300), np.linspace(np.min(y), np.max(y), 300)
                inter = interpolate.interp2d(y, x, z, kind='cubic', fill_value=0)
                z1 = inter(x1, y1)
                plt.contourf(image_comb/np.nanmax(z1), levels=5)
            if 0:
                im_hist = np.histogram(image_comb.flatten()[~np.isnan(image_comb.flatten())],bins=100)
                im_flux_dist = [0] + [im_hist[0][i]*im_hist[1][i] for i in range(np.size(im_hist[0]))]
                fig,ax = plt.subplots(1,2)
                ax[0].hist(image_comb.flatten()[~np.isnan(image_comb.flatten())])
                ax[1].plot(im_hist[1],np.cumsum(im_flux_dist))

            d = image_comb/np.nanmax(image_comb)


            d[np.isnan(d)]=0
            self.data_contours = pg.IsocurveItem(data=d, level=level, pen=pg.mkPen('black', width=2), axisOrder='row-major')

            self.vb.addItem(self.data_contours )
            plt.show()
        else:
            self.vb.removeItem(self.data_contours)


    def redraw(self):
        for i, v in enumerate(self.view.values()):
            v[1].setBrush(pg.mkBrush(cm.rainbow(0.01 + 0.98 * i / len(self.view), bytes=True)[:3] + (255,)))
        for i, v in enumerate(self.models.values()):
            v.setPen(pg.mkPen(cm.rainbow(0.01 + 0.98 * i / len(self.models), bytes=True)[:3] + (255,)))

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            return
        self.mousePoint = self.vb.vb.mapSceneToView(event.pos())
        self.x, self.y = self.mousePoint.x(), self.mousePoint.y()
        col, row = int(self.x), int(self.y)
        #row, col = int(self.x), int(self.y)
        (tind,time) = self.timeIndex(self.timeLine)
        self.lam = time
        #wcs = self.parent.stage2.data.meta.wcs
        #wcs = self.parent.CUBE.data.wcs
        print('mouse coord:',self.x, self.y)
        if 0:
            wcs1 = self.parent.CUBE.data.wcs
            self.x_world = wcs1['CRVAL1']- (self.x-wcs1['CRPIX1'])*  wcs1['CDELT1']
            self.y_world = wcs1['CRVAL2'] + (self.y - wcs1['CRPIX2']) * wcs1['CDELT2']
            self.lam_world = wcs1['CRVAL3'] + (self.lam - wcs1['CRPIX3']) * wcs1['CDELT3']
            print('world coord:',self.x_world,self.y_world,self.lam_world)
        else:
            if self.cube_name == 'A':
                lam_world,x_world,y_world = self.parent.CUBE_A.conv_world_coord(t=self.lam, x=int(self.x), y=int(self.y))
            elif self.cube_name == 'B':
                lam_world,x_world,y_world = self.parent.CUBE_B.conv_world_coord(t=self.lam, x=self.x, y=self.y)
        if row < self.data.shape[1] and row >= 0 and col < self.data.shape[2] and col > 0:
            val = self.data[tind,row,col]
        else:
            val = np.nan
        #text = "pixel: (x=%.1f, y=%.1f, l=%d), val: %.2f MJy/sr" % (self.x, self.y, time, val)
        text = "pixel: (row=%d, col=%d, l=%d), val: %.2f MJy/sr" % ( col, row,time,val)
        print(text)

        text_world_coord = "world: (ra=%.6f, dec=%.6f, l=%.6f)" % (x_world,y_world,lam_world)
        print(text_world_coord)

        rah = int(x_world/360*24)
        ramin = int((x_world/360*24-rah)*60)
        rasec = (x_world/360*24-rah - ramin/60)*3600
        decdeg = int(y_world)
        decmin = int((y_world-decdeg)*60)
        decsec = (y_world - decdeg - decmin/60)*3600

        text_world_coord_asec = "world: (ra=%d %d %.4f dec=%d %d %.2f)" % (rah, ramin, rasec,decdeg,decmin,decsec)
        print(text_world_coord_asec)
        #cal_world_to_detector = wcs.get_transform('world', 'detector')
        #print('world->detector', cal_world_to_detector(x_world,y_world, lam_world))

        self.label.setText(text)
        self.label.resize(600, 40)
        self.label2.setText(text_world_coord)
        self.label2.resize(600, 40)
        self.label3.setText(text_world_coord_asec)
        self.label3.resize(600, 40)


    def mousePressEvent(self, event, pos=None):
        print(pos)
        if pos is None:
            super(plotCube, self).mousePressEvent(event)
            if event.button() == Qt.LeftButton:
                self.mousePoint = self.vb.vb.mapSceneToView(event.pos())
                #self.mousePoint = self.vb.mapRectToView(event.pos())
                self.x, self.y = self.mousePoint.x(), self.mousePoint.y()
        else:
            self.x, self.y = pos
        print('Position on Image:', self.x, self.y)
        if self.s_status:
            #name = self.parent.name

            if 0:
                col,row =int(self.x),int(self.y)
                show_fit = self.parent.s3d.Cubes_A.table.flags['read_fit_slopes']
                self.parent.plot_pixel.plot_profile(row=row, col=col,add=False,show_fit=show_fit)
                self.parent.plot_pixel.plot_profile(row=row, col=col,show_fit=show_fit)
                self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col, add=False)
                self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col)

    #def paintEvent(self, x=250,y=250,rad=1):
    #    painter = QPainter(self)
    #    painter.setPen(QPen(QColor(0, 0, 255), 2, Qt.SolidLine))
    #    painter.setBrush(QColor(0, 0, 255, 50))
    #    painter.drawEllipse(x, y, rad, rad)

    def selectPixel(self,add=False,x=250,y=250,rad=1,color='r',type='saturated'):
        if add:
            #roi_circle=pg.CircleROI(pos=[x+0.5, y+0.5], size=[rad, rad], pen=pg.mkPen('r', width=2))
            item = pg.ScatterPlotItem(size=30, pen=pg.mkPen(color, width=2))
            item.setSymbol('s')
            item.setData([x+0.5],[y+0.5])
            #item = pg.PlotCurveItem( pen=pg.mkPen('r', width=2))
            #item.setData([x,x+1,x+1,x,x],[y,y,y+1,y+1,y])
            if type == 'saturated':
                self.selected_pixels_saturated.append(item)
            elif type == 'dnu':
                self.selected_pixels_dnu.append(item)
            elif type == 'cr':
                self.selected_pixels_cr.append(item)
            elif type == 'cr_multi':
                self.selected_pixels_cr_multi.append(item)
            self.vb.addItem(item)
            #self.vb.addItem(roi_circle)
        else:
            if type == 'saturated':
                sample = self.selected_pixels_saturated
            elif type == 'cr':
                sample = self.selected_pixels_cr
            elif type == 'cr_multi':
                sample = self.selected_pixels_cr_multi
            elif type == 'dnu':
                sample =self.selected_pixels_dnu
            for el in sample:
                self.vb.removeItem(el)

    def selectPixels(self,add=False,x=250,y=250,rad=1,color='r',type='saturated'):
        if add:
            #roi_circle=pg.CircleROI(pos=[x+0.5, y+0.5], size=[rad, rad], pen=pg.mkPen('r', width=2))
            item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(color, width=2))
            item.setSymbol('s')
            item.setData(x+0.5,y+0.5)
            #item = pg.PlotCurveItem( pen=pg.mkPen('r', width=2))
            #item.setData([x,x+1,x+1,x,x],[y,y,y+1,y+1,y])
            if type == 'saturated':
                self.selected_pixels_saturated.append(item)
            elif type == 'dnu':
                self.selected_pixels_dnu.append(item)
            elif type == 'cr':
                self.selected_pixels_cr.append(item)
            elif type == 'cr_multi':
                self.selected_pixels_cr_multi.append(item)
            self.vb.addItem(item)
            #self.vb.addItem(roi_circle)
        else:
            if type == 'saturated':
                sample = self.selected_pixels_saturated
            elif type == 'cr':
                sample = self.selected_pixels_cr
            elif type == 'cr_multi':
                sample = self.selected_pixels_cr_multi
            elif type == 'dnu':
                sample = self.selected_pixels_dnu

            for el in sample:
                self.vb.removeItem(el)

    #def keyPressEvent(self, event):
    #    super(plotGrid, self).keyPressEvent(event)
    #    key = event.key()

class plotImage(pg.ImageView): #(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        # Add Plot item to show axis labels
        # pg.setLabel(axis='left', text='Y-axis')
        # pg.setLabel(axis='bottom', text='X-axis')

        #pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'Row', 'bottom': 'Col'})
        #pg.ImageView.__init__(self,name='Exposure image',view=pg.ViewBox())
        self.img = pg.ImageView.__init__(self, name='Cube image', view=pg.PlotItem(), discreteTimeLine=True)
        #pg.setLabel(axis='left', text='Y-axis')
        #pg.setLabel(axis='bottom', text='X-axis')
        #imv = pg.ImageView()
        #imv.__init__(self, name='Exposure image')
        self.initstatus()
        #self.vb = self.getViewBox()
        self.vb = self.getView()
        self.roiplot = self.getRoiPlot()

        #self.roi = pg.EllipseROI(pos=(0, 0), size=(10, 10))
        #self.view = {}
        ## Set a custom color map
        cmap = pg.colormap.get('CET-D7') #pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        self.setColorMap(cmap)
        self.vb.invertY(False)
        self.cursorpos = pg.TextItem(anchor=(0, 1))
        self.vb.setLabel(axis='left', text='Y-axis')
        self.vb.setLabel(axis='bottom', text='X-axis')
        self.label = QLabel("this is a QLabel", self.ui.graphicsView.viewport())
        self.label.move(25, 25)
        self.Roilabel = QLabel("ROI:", self.ui.graphicsView.viewport())
        self.Roilabel.move(25, 60)
        self.Roilabel.resize(400, 40)


        #self.vb.addItem(pg.LabelItem("this is a nice label"))

        #self.vb.invertY(True)
        #self.models = {}
        #self.legend = pg.LegendItem(offset=(-70, 30))
        #self.legend.setParentItem(self.vb)
        #self.legend_model = pg.LegendItem(offset=(-70, -30))
        #self.legend_model.setParentItem(self.vb)

    def initstatus(self):
        self.s_status = True
        self.selected_point = None
        self.selected_pixels_saturated = []
        self.selected_pixels_dnu = []
        self.selected_pixels_cr = []
        self.selected_pixels_cr_multi = []

        #self.show_mouse_event()


    def add(self, name, add,mode=None,Nscreen=1):
        if add:
            if mode == None:
                nexp = {}
                nexp[1] = 0 #int(self.parent.exp_pars.nEXP1.text())
                nexp[2] = 1 #int(self.parent.exp_pars.nEXP2.text())
                nexp[3] = 2 #int(self.parent.exp_pars.nEXP3.text())
                nexp[4] = 3 #int(self.parent.exp_pars.nEXP4.text())
                self.Nscreen = Nscreen
                print('add cube name:',name)
                asn_filename = self.parent.Cubes_A.associtations_list[name] #.split('/')[-1]
                self.parent.CUBE_A.load3asn(asn_filename)

                rate_filename = self.parent.CUBE_A.ratefiles[nexp[Nscreen]]
                f = rate_filename['expname']
                path = self.parent.stage2[Nscreen-1].path
                exp_name = f.split('/')[-1].replace('cal.fits', '')
                output2_dir, spec2_cachedir = self.parent.stage2[Nscreen-1].output_dir, self.parent.stage2[Nscreen-1].spec2_cachedir
                self.parent.stage2[Nscreen-1].__init__(miri_uncal_file=exp_name, path=path, output_dir=output2_dir,
                                                   spec2_cachedir=spec2_cachedir)
                print('add stage2 name:', exp_name)
                f = rate_filename['expname']
                self.parent.stage2[Nscreen-1].read_step_results(step_name='ResFringe')
                print('Show image of' + self.parent.stage2[Nscreen-1].name)
                data = self.parent.stage2[Nscreen-1].data.data
                zmin, zmax = np.nanquantile(data.flatten(), 0.05), np.nanquantile(data.flatten(), 0.95)
                x, y = (1, 0)
                self.data = data
                self.setImage(data, autoRange=True,  # levels=[-1, 5],
                              axes={'t': None, 'x': x, 'y': y, 'c': None}, levels=[zmin, zmax])
                self.vb.hoverEvent = self.imageHoverEvent
                hist = self.getHistogramWidget()
                hist.setHistogramRange(mn=-100,mx=500)
                #hist.vb.state['viewRange'] = [[-1,1],[-200,400]]



        else:
            try:
                self.vb.removeItem(self.view[name])
                self.legend.removeItem(self.view[name])
                del self.view[name]
            except:
                pass

    def redraw(self):
        for i, v in enumerate(self.view.values()):
            v[1].setBrush(pg.mkBrush(cm.rainbow(0.01 + 0.98 * i / len(self.view), bytes=True)[:3] + (255,)))
        for i, v in enumerate(self.models.values()):
            v.setPen(pg.mkPen(cm.rainbow(0.01 + 0.98 * i / len(self.models), bytes=True)[:3] + (255,)))

    def imageHoverEvent(self, event):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isExit():
            # pos = event.pos()
            # i, j = pos.y(), pos.x()
            # print('exit')
            # self.label.setText('exit')
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        self.mousePoint = self.vb.vb.mapSceneToView(event.pos())
        # self.mousePoint = self.vb.mapRectToView(event.pos())
        self.x, self.y = self.mousePoint.x(), self.mousePoint.y()
        col, row, val = int(self.x), int(self.y), None
        expname = {}
        expname[1] = 'EXP0'
        expname[2] = 'EXP1'
        expname[3] = 'EXP2'
        expname[4] = 'EXP3'

        text = expname[self.Nscreen]+":(%d, %d), val: None" % (col, row)
        if row < self.data.shape[0] and row >= 0 and col < self.data.shape[1] and col >= 0:
            val = self.data[row, col]
            text = expname[self.Nscreen]+":(%d, %d), val: %.2f" % (col, row, val)
        # print('coord', text, val)
        self.label.setText(text)
        self.label.resize(400, 40)

    def mousePressEvent(self, event, pos=None):
        print(pos)
        if pos is None:
            super(plotImage, self).mousePressEvent(event)
            if event.button() == Qt.LeftButton:
                self.mousePoint = self.vb.vb.mapSceneToView(event.pos())
                # self.mousePoint = self.vb.mapRectToView(event.pos())
                self.x, self.y = self.mousePoint.x(), self.mousePoint.y()
        else:
            self.x, self.y = pos
        print('Position on Image:', self.x, self.y)
        if self.s_status and 1:
            # name = self.parent.name

            if 0:
                ln =  self.parent.stage2.data.data[pos]
                self.roiplot.plot(ln)
            if 0:
                rois = []
                rois.append(pg.EllipseROI([15, 15], [10, 10], pen=(3, 9)))

                # rois.append(pg.EllipseROI([20, 20], [12, 12], pen=(9, 2)))

                def updateRoi(roi):
                    if roi is None:
                        return
                    arr1 = roi.getArrayRegion(arr=self.data, img=self.imageItem, axes=(2, 1))
                    if 1:  # set_roi_mask
                        mask = np.array(np.zeros((self.data.shape[1], self.data.shape[2])), dtype='bool')
                        rows, cols = self.data.shape[1], self.data.shape[2]
                        m = np.mgrid[:rows, :cols]
                        possx = m[0, :, :]  # make the x pos array
                        possy = m[1, :, :]  # make the y pos array
                        possx.shape = rows, cols
                        possy.shape = rows, cols
                        mpossx = roi.getArrayRegion(possx, self.imageItem).astype(int)
                        mpossx = mpossx[np.nonzero(mpossx)]  # get the x pos from ROI
                        mpossy = roi.getArrayRegion(possy, self.imageItem).astype(int)
                        mpossy = mpossy[np.nonzero(mpossy)]  # get the y pos from ROI
                        mask[mpossy, mpossx] = True  # self.data[0,mpossx, mpossy]>0
                        self.roi_mask = mask
                    updateRoiPlot(roi, arr1)

                def updateRoiPlot(roi, data_roi=None):
                    if data_roi is None:
                        data_roi = roi.getArrayRegion(arr=self.data, img=self.imageItem, axes=(1, 2))
                        # data = roi.getArrayRegion(im1.image, img=im1)
                    if data_roi is not None:
                        d = data_roi.mean(axis=(1, 2))
                        roi.curve.setData(data_roi.mean(axis=(1, 2)))
                        self.parent.plot_spectrum.plot_spec(data=d, add=False)
                        self.parent.plot_spectrum.plot_spec(data=d)
                    if np.sum(self.roi_mask) > 0:
                        num_pixels = np.sum(self.roi_mask)
                        roi_selected_flux = np.zeros(self.data.shape[0])
                        for i in range(self.data.shape[0]):
                            d = self.data[i, :, :]
                            d = d[self.roi_mask]
                            roi_selected_flux[i] = np.sum(d) / num_pixels
                        self.parent.plot_spectrum.plot_spec_2(data=roi_selected_flux, add=False)
                        self.parent.plot_spectrum.plot_spec_2(data=roi_selected_flux)

                ## Add each ROI to the scene and link its data to a plot curve with the same color
                for r in rois:
                    self.vb.addItem(r)
                    c = self.roi.plot(pen=r.pen)
                    r.curve = c
                    r.sigRegionChanged.connect(updateRoi)

                # self.updateRoi = updateRoi(rois[0])

                def updatePlotSpec():
                    data_roi = self.roi.getArrayRegion(arr=self.data, img=self.imageItem, axes=(1, 2))
                    if data_roi is not None:
                        d = data_roi.mean(axis=(1, 2))
                        self.roi.curve.setData(data_roi.mean(axis=(1, 2)))
                        self.parent.plot_spectrum.plot_spec(data=d, add=False)
                        self.parent.plot_spectrum.plot_spec(data=d)

                # self.timeLine.sigPositionChanged.connect(updatePlotSpec)

            #xMin,xMax,yMin,yMax =
            #self.vb.setLimits(self.parent.plot_2dimage1)


    # def paintEvent(self, x=250,y=250,rad=1):
    #    painter = QPainter(self)
    #    painter.setPen(QPen(QColor(0, 0, 255), 2, Qt.SolidLine))
    #    painter.setBrush(QColor(0, 0, 255, 50))
    #    painter.drawEllipse(x, y, rad, rad)

    def selectPixel(self, add=False, x=250, y=250, rad=1, color='r', type='saturated'):
        if add:
            # roi_circle=pg.CircleROI(pos=[x+0.5, y+0.5], size=[rad, rad], pen=pg.mkPen('r', width=2))
            item = pg.ScatterPlotItem(size=30, pen=pg.mkPen(color, width=2))
            item.setSymbol('s')
            item.setData([x + 0.5], [y + 0.5])
            # item = pg.PlotCurveItem( pen=pg.mkPen('r', width=2))
            # item.setData([x,x+1,x+1,x,x],[y,y,y+1,y+1,y])
            if type == 'saturated':
                self.selected_pixels_saturated.append(item)
            elif type == 'dnu':
                self.selected_pixels_dnu.append(item)
            elif type == 'cr':
                self.selected_pixels_cr.append(item)
            elif type == 'cr_multi':
                self.selected_pixels_cr_multi.append(item)
            self.vb.addItem(item)
            # self.vb.addItem(roi_circle)
        else:
            if type == 'saturated':
                sample = self.selected_pixels_saturated
            elif type == 'cr':
                sample = self.selected_pixels_cr
            elif type == 'cr_multi':
                sample = self.selected_pixels_cr_multi
            elif type == 'dnu':
                sample = self.selected_pixels_dnu
            for el in sample:
                self.vb.removeItem(el)

    def selectPixels(self, add=False, x=250, y=250, rad=1, color='r', type='saturated'):
        if add:
            # roi_circle=pg.CircleROI(pos=[x+0.5, y+0.5], size=[rad, rad], pen=pg.mkPen('r', width=2))
            item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(color, width=2))
            item.setSymbol('s')
            item.setData(x + 0.5, y + 0.5)
            # item = pg.PlotCurveItem( pen=pg.mkPen('r', width=2))
            # item.setData([x,x+1,x+1,x,x],[y,y,y+1,y+1,y])
            if type == 'saturated':
                self.selected_pixels_saturated.append(item)
            elif type == 'dnu':
                self.selected_pixels_dnu.append(item)
            elif type == 'cr':
                self.selected_pixels_cr.append(item)
            elif type == 'cr_multi':
                self.selected_pixels_cr_multi.append(item)
            self.vb.addItem(item)
            # self.vb.addItem(roi_circle)
        else:
            if type == 'saturated':
                sample = self.selected_pixels_saturated
            elif type == 'cr':
                sample = self.selected_pixels_cr
            elif type == 'cr_multi':
                sample = self.selected_pixels_cr_multi
            elif type == 'dnu':
                sample = self.selected_pixels_dnu

            for el in sample:
                self.vb.removeItem(el)

class plotSpec(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'SB (MJy/sr)', 'bottom': 'Wavelength [micron]'})
        #self.initstatus()
        self.vb = self.getViewBox()
        self.image = None
        self.text = None
        self.grid = image()
        cdict = cm.get_cmap('viridis')
        cmap = np.array(cdict.colors)
        cmap[-1] = [1, 0.4, 0]
        map = pg.ColorMap(np.linspace(0, 1, cdict.N), cmap, mode='rgb')
        self.colormap = map.getLookupTable(0.0, 1.0, 256, alpha=False)
        self.legend = pg.LegendItem(offset=(-70, 30))
        self.legend.setParentItem(self.vb)
        self.legend_model = pg.LegendItem(offset=(-70, -30))
        self.legend_model.setParentItem(self.vb)
        self.setTitle("Pixel difference", color="olive", size="10pt")
        self.lines = self.listDataItems()




    def plot_specA1(self, data=None, err=None, add=True,pen=pg.mkPen(color='white', style=Qt.DashLine, width=1),normalize=False,show_err_bar=False,label='A1',smoothing=True):
        if add:
            wavel = self.parent.CUBE_A.data.wavelength
            if np.size(data) == np.size(wavel):
                if normalize:
                    norm = np.mean(data[10:40])
                    data = data / norm
                    err = err / norm
                if smoothing:
                    win = signal.windows.hann(10)
                    data = signal.convolve(data, win, mode='same') / sum(win)
                self.plot_lineA1 = pg.PlotCurveItem(wavel, data,pen='lightgreen')
                self.plot_errbarA1 = pg.ErrorBarItem(x=wavel,y=data,height=err,pen=pen, beam=1/6000)
                self.vb.addItem(self.plot_lineA1)
                self.legend_model.addItem(self.plot_lineA1, label)
                if show_err_bar:
                    self.vb.addItem(self.plot_errbarA1)
                pen = pg.mkPen(color='darkgray', style=Qt.DashLine, width=1)
                #self.zero_level = pg.PlotCurveItem([wavel[0]-2, wavel[-1] + 2], [0, 0], pen=pen)
                self.zero_level = pg.PlotCurveItem([0, 30], [0, 0], pen=pen)
                self.vb.addItem(self.zero_level)

                #NGC = np.loadtxt('/home/slava/science/codes/python/jwst/input/NGC19.txt',delimiter=',')
                #NGC = np.loadtxt('/home/slava/science/data/SPITZER/AO0235/cassis_yaaar_spcfw_15121152t.dat')
                NGC = np.loadtxt('/media/slava/14999070-ec17-4bcc-993d-c556030e9642/home/slava/science/data/SPITZER/AO0235/cassis_yaaar_spcfw_15121152t-copy-red_norm.dat')
                x,y = NGC[:,0], NGC[:,1]
                mask = (x>wavel[10])*(x<wavel[40])
                self.show_template = False
                if np.sum(mask)>0:
                    self.show_template = True
                    norm = 1/np.mean(y[mask])*np.mean(data[10:40])
                    self.plot_NGC = pg.PlotCurveItem(x,y*norm, pen='lightgreen')
                    self.vb.addItem(self.plot_NGC)
                    self.legend_model.addItem(self.plot_NGC, 'Template')

                self.lr = pg.LinearRegionItem(values=[5,5])
                self.lr.setZValue(-10)
                self.vb.addItem(self.lr)


                def update_lr():
                    (timeind, time) = self.parent.plot_3dcubeA.timeIndex(self.parent.plot_3dcubeA.timeLine)
                    (l,x,y) = self.parent.CUBE_A.conv_world_coord(t=time, x=10, y=10)
                    self.lr.setRegion(rgn=[l,l])

                update_lr()

                def update_roi_slicer(pos_ind=None):
                    if pos_ind is None:
                        return
                    if pos_ind != self.parent.plot_3dcubeA.currentIndex:
                        self.parent.plot_3dcubeA.currentIndex = pos_ind
                        #self.parent.plot_3dcube.updateRoi
                        #self.parent.plot_3dcube.updateImage()


                pos_ind = int(self.lr.getRegion()[0])


                #self.vb.autoRange()
                #self.vb.setLimits(xMin=wavel[0]*0.98, xMax=wavel[-1]*1.02)


        else:
            try:
                self.vb.removeItem(self.plot_lineA1)
                if show_err_bar:
                    self.vb.removeItem(self.plot_errbarA1)
                self.legend_model.removeItem(self.plot_lineA1)
                if self.show_template:
                    self.show_template = False
                    self.vb.removeItem(self.plot_NGC)
                    self.legend_model.removeItem(self.plot_NGC)



                self.vb.removeItem(self.zero_level)
                self.vb.removeItem(self.lr)



            except:
                pass


    def plot_specAmax(self, data=None, err=None,add=True,pen=pg.mkPen(color='royalblue', style=Qt.DashLine, width=1),normalize=False,show_err_bar=False,label='Amax'):
        if add:
            wavel = self.parent.CUBE_A.data.wavelength
            if np.size(data) == np.size(wavel):
                if normalize:
                    norm = np.mean(data[10:40])
                    data = data / norm
                    err = err / norm
                self.plot_lineAmax = pg.PlotCurveItem(wavel, data,pen=pen)
                self.plot_errbarAmax = pg.ErrorBarItem(x=wavel, y=data, height=err, pen=pen)
                self.vb.addItem(self.plot_lineAmax)
                self.legend_model.addItem(self.plot_lineAmax, label)
                if show_err_bar:
                    self.vb.addItem(self.plot_errbarAmax)

                self.lr = pg.LinearRegionItem(values=[5,5])
                self.lr.setZValue(-10)
                self.vb.addItem(self.lr)
                def update_lr():
                    (timeind, time) = self.parent.plot_3dcubeA.timeIndex(self.parent.plot_3dcubeA.timeLine)
                    (l,x,y) = self.parent.CUBE_A.conv_world_coord(t=time, x=10, y=10)
                    self.lr.setRegion(rgn=[l,l])

                update_lr()







        else:
            try:
                self.vb.removeItem(self.plot_lineAmax)
                self.legend_model.removeItem(self.plot_lineAmax)
                if show_err_bar:
                    self.vb.removeItem(self.plot_errbarAmax)

                self.vb.removeItem(self.lr)



            except:
                pass

    def plot_specA2(self, data=None, err=None, add=True, pen=pg.mkPen(color='royalblue', style=Qt.DashLine, width=1),
                     normalize=False, show_err_bar=False, label='A2',smoothing=True):
        if add:
            wavel = self.parent.CUBE_A.data.wavelength
            if np.size(data) == np.size(wavel):
                if normalize:
                    norm = np.mean(data[10:40])
                    data = data / norm
                    err = err / norm
                if smoothing:
                    win = signal.windows.hann(10)
                    data = signal.convolve(data, win, mode='same') / sum(win)
                self.plot_lineA2 = pg.PlotCurveItem(wavel, data, pen=pen)
                self.plot_errbarA2 = pg.ErrorBarItem(x=wavel, y=data, height=err, pen=pen)
                self.vb.addItem(self.plot_lineA2)
                self.legend_model.addItem(self.plot_lineA2, label)
                if show_err_bar:
                    self.vb.addItem(self.plot_errbarA2)

        else:
            try:
                self.vb.removeItem(self.plot_lineA2)
                self.legend_model.removeItem(self.plot_lineA2)
                if show_err_bar:
                    self.vb.removeItem(self.plot_errbarA2)
                #self.vb.removeItem(self.plot_lineA2m1)
                #self.vb.removeItem(self.plot_errbarA2m1)


            except:
                pass

    def plot_specA2m1(self, data=None, err=None, add=True, pen=pg.mkPen(color='royalblue', style=Qt.SolidLine, width=2),
                    normalize=False, show_err_bar=False, label='A2-A1'):
        if add:
            wavel = self.parent.CUBE_A.data.wavelength
            if np.size(data) == np.size(wavel):
                if normalize:
                    norm = np.mean(data[10:40])
                    data = data / norm
                    err = err / norm
                self.plot_lineA2m1 = pg.PlotCurveItem(wavel, data, pen=pen)
                self.plot_errbarA2m1 = pg.ErrorBarItem(x=wavel, y=data, height=err, pen=pen)
                self.vb.addItem(self.plot_lineA2m1)
                self.legend_model.addItem(self.plot_lineA2m1, label)
                if show_err_bar:
                    self.vb.addItem(self.plot_errbarA2m1)

        else:
            try:
                self.vb.removeItem(self.plot_lineA2m1)
                self.legend_model.removeItem(self.plot_lineA2m1)
                if show_err_bar:
                    self.vb.removeItem(self.plot_errbarA2m1)



            except:
                pass


    def plot_specB1(self, data=None, err=None, add=True, pen=pg.mkPen(color='white', style=Qt.DashLine,
                                                            width=1),normalize=True,show_err_bar=False,smoothing=True):  # pg.mkPen(color='gray', style=Qt.DashLine, width=3)
        if add:
            wavel = self.parent.CUBE_B.data.wavelength
            if np.size(data) == np.size(wavel):
                if normalize:
                    norm = np.mean(data[10:40])
                    data = data / norm
                    err = err/norm
                if smoothing:
                    win = signal.windows.hann(10)
                    data = signal.convolve(data, win, mode='same') / sum(win)
                self.plot_lineB1 = pg.PlotCurveItem(wavel, data, pen=pen)
                self.plot_errbarB1 = pg.ErrorBarItem(x=wavel, y=data, height=err, pen=pen)
                self.vb.addItem(self.plot_lineB1)
                if show_err_bar:
                    self.vb.addItem(self.plot_errbarB1)

                #self.vb.autoRange()
                #self.vb.setLimits(xMin=wavel[0] * 0.98, xMax=wavel[-1] * 1.02)
        else:
            try:
                self.vb.removeItem(self.plot_lineB1)
                if show_err_bar:
                    self.vb.removeItem(self.plot_errbarB1)


            except:
                pass

    def plot_specB2(self, data=None, add=True, pen=pg.mkPen(color='white', style=Qt.DashLine,
                                                            width=1),normalize=True,smoothing=True):  # pg.mkPen(color='gray', style=Qt.DashLine, width=3)
        if add:
            wavel = self.parent.CUBE_B.data.wavelength
            if np.size(data) == np.size(wavel):
                if normalize:
                    norm = np.mean(data[10:40])
                    data = data / norm
                if smoothing:
                    win = signal.windows.hann(10)
                    data = signal.convolve(data, win, mode='same') / sum(win)
                self.plot_lineB2 = pg.PlotCurveItem(wavel, data, pen=pen)
                self.vb.addItem(self.plot_lineB2)

                #self.vb.autoRange()
                #self.vb.setLimits(xMin=wavel[0] * 0.98, xMax=wavel[-1] * 1.02)
        else:
            try:
                self.vb.removeItem(self.plot_lineB2)

            except:
                pass


    def plot_specB_median(self, data=None, add=True, pen=pg.mkPen(color='white', style=Qt.DashLine,
                                                            width=1),normalize=True,smoothing=True,npixels=1):  # pg.mkPen(color='gray', style=Qt.DashLine, width=3)
        if add:
            wavel = self.parent.CUBE_B.data.wavelength
            flux = self.parent.CUBE_B.data.data
            mean_flux = np.nanmean(np.nanmean(flux,axis=1),axis=1)*npixels
            if np.size(mean_flux) == np.size(wavel):
                if normalize:
                    norm = np.mean(mean_flux[10:40])
                    data = mean_flux / norm
                else:
                    data = mean_flux
                if smoothing:
                    win = signal.windows.hann(10)
                    data = signal.convolve(data, win, mode='same') / sum(win)
                self.plot_lineB_median = pg.PlotCurveItem(wavel, data, pen=pen)
                self.vb.addItem(self.plot_lineB_median)

                #self.vb.autoRange()
                #self.vb.setLimits(xMin=wavel[0] * 0.98, xMax=wavel[-1] * 1.02)
        else:
            try:
                self.vb.removeItem(self.plot_lineB_median)

            except:
                pass
class plotHist(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'SB (MJy/sr)', 'bottom': 'Wavelength [micron]'})
        #self.initstatus()
        self.vb = self.getViewBox()
        self.image = None
        self.text = None
        self.grid = image()
        cdict = cm.get_cmap('viridis')
        cmap = np.array(cdict.colors)
        cmap[-1] = [1, 0.4, 0]
        map = pg.ColorMap(np.linspace(0, 1, cdict.N), cmap, mode='rgb')
        self.colormap = map.getLookupTable(0.0, 1.0, 256, alpha=False)
        self.legend = pg.LegendItem(offset=(-70, 30))
        self.legend.setParentItem(self.vb)
        self.legend_model = pg.LegendItem(offset=(-70, -30))
        self.legend_model.setParentItem(self.vb)
        self.setTitle("Roi Flux Hist", color="olive", size="10pt")
        self.lines = self.listDataItems()

    def plot_hist(self, data=None, roi_mask=[1], timeind=0, add=True,pen=pg.mkPen(color='white', style=Qt.DashLine, width=1),med_mode='median',brush='green'): #pg.mkPen(color='gray', style=Qt.DashLine, width=3)
        if add:
            if np.sum(roi_mask) > 0:
                num_pixels = np.sum(roi_mask)
                d = data.data[timeind, :, :]
                d = d[roi_mask]
                d = d[~np.isnan(d)]
                derr = data.err[timeind, :, :]
                derr = derr[roi_mask]
                derr = derr[~np.isnan(derr)]

                y, x = np.histogram(d[~np.isnan(d)])
                self.plot_hist_roi = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, pen=pen)
                self.vb.addItem(self.plot_hist_roi)

                self.zero_level = pg.PlotCurveItem([np.min(x),np.max(x)], [0,0], pen='white')
                self.vb.addItem(self.zero_level)
                if med_mode== 'median':
                    median = np.sum(d / derr ** 2) / np.sum(1 / derr ** 2)
                    median_error = np.sqrt( 1 / np.sum(1 / derr ** 2))
                    mean = np.sum(d) / np.size(d)
                    c2 =  pg.PlotCurveItem([median+ median_error,median+ median_error], [0,np.max(y)],  pen=pen)
                    c1 = pg.PlotCurveItem([median-median_error, median-median_error], [0, np.max(y)], pen=pen)
                    #c3 = pg.PlotCurveItem([median, median], [0, np.max(y)], pen='grey')
                    self.fill1 = pg.FillBetweenItem(c1, c2,brush=brush)
                    self.mean_line = pg.PlotCurveItem([mean, mean], [0, 1.5*np.max(y)], pen='red')
                    self.median_line = pg.PlotCurveItem([median, median], [0, 1.5*np.max(y)], pen='lightgreen')
                    self.vb.addItem(self.fill1)
                    self.vb.addItem(self.mean_line)
                    self.vb.addItem(self.median_line)


        else:
            try:
                self.vb.removeItem(self.plot_hist_roi)
                self.vb.removeItem(self.fill1)
                self.vb.removeItem(self.zero_level)
                self.vb.removeItem(self.mean_line)
                self.vb.removeItem(self.median_line)
            except:
                pass

    def plot_hist2(self, data=None, roi_mask=[1], timeind=0, add=True, pen=pg.mkPen(color='white', style=Qt.DashLine,
                                                                                   width=1),med_mode = 'median',brush='red'):  # pg.mkPen(color='gray', style=Qt.DashLine, width=3)
        if add:
            if np.sum(roi_mask) > 0:
                num_pixels = np.sum(roi_mask)
                d = data.data[timeind, :, :]
                d = d[roi_mask]
                derr = data.err[timeind, :, :]
                derr = derr[roi_mask]
                y, x = np.histogram(d[~np.isnan(d)])
                self.plot_hist_roi2 = pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, pen=pen)
                self.vb.addItem(self.plot_hist_roi2)
                if med_mode == 'median':
                    median = np.sum(d / derr ** 2) / np.sum(1 / derr ** 2)
                    median_error = 1 / np.sum(1 / derr ** 2)
                    c1 = pg.PlotCurveItem([median- median_error, median- median_error], [0, np.max(y)], pen=pen)
                    c2 = pg.PlotCurveItem([median + median_error, median + median_error], [0, np.max(y)], pen=pen)
                    #c3 = pg.PlotCurveItem([median, median], [0, np.max(y)], pen='grey')
                    self.median_line2 = pg.PlotCurveItem([median, median], [0, 1.2*np.max(y)], pen='grey')
                    self.fill2 = pg.FillBetweenItem(c1, c2, brush=brush)
                    self.vb.addItem(self.fill2)
                    self.vb.addItem(self.median_line2)
        else:
            try:
                self.vb.removeItem(self.plot_hist_roi2)
                self.vb.removeItem(self.fill2)
                self.vb.removeItem(self.median_line2)
            except:
                pass


class plotSlit(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'Flux (MJy/sr*Npix)', 'bottom': 'Spatial axis [pix]'})
        #self.initstatus()
        self.vb = self.getViewBox()
        self.image = None
        self.text = None
        self.grid = image()
        cdict = cm.get_cmap('viridis')
        cmap = np.array(cdict.colors)
        cmap[-1] = [1, 0.4, 0]
        map = pg.ColorMap(np.linspace(0, 1, cdict.N), cmap, mode='rgb')
        self.colormap = map.getLookupTable(0.0, 1.0, 256, alpha=False)
        self.legend = pg.LegendItem(offset=(-70, 30))
        self.legend.setParentItem(self.vb)
        self.legend_model = pg.LegendItem(offset=(-70, -30))
        self.legend_model.setParentItem(self.vb)
        self.setTitle("Slit profile", color="olive", size="10pt")
        self.lines = self.listDataItems()

    def plot_slit(self, data=None, add=True,pen=pg.mkPen(color='white', style=Qt.DashLine, width=1)): #pg.mkPen(color='gray', style=Qt.DashLine, width=3)
        if add:
            if np.sum(data) > 0:
                x = np.arange(data.shape[0])
                self.plot_slit_profile = pg.PlotCurveItem(x, data,  pen=pen)
                self.vb.addItem(self.plot_slit_profile)
                if 1:
                    from specutils.spectra import Spectrum1D
                    from specutils.fitting import fit_lines
                    from astropy import units as u
                    from astropy.modeling import models
                    from scipy.interpolate import interp1d
                    (timeind, time) = self.parent.plot_3dcubeA.timeIndex(self.parent.plot_3dcubeA.timeLine)
                    lambda_local = self.parent.CUBE_A.data.wavelength[timeind]
                    def miri_psf_pix(lam):
                        #interpolation of miri psf https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-performance/miri-point-spread-functions
                        f = np.loadtxt('./data/miri_psf_pix.dat')

                        print(f[:,0])
                        f1d = interp1d(f[:,0],f[:,1],fill_value='extrapolate')
                        print('miri psf =',f1d(lam))
                        return f1d(lam)
                    miri_psf_fwhm = miri_psf_pix(lambda_local)
                    miri_psf_stddev = miri_psf_fwhm/2.355
                    spectrum = Spectrum1D(flux=data * u.Jy, spectral_axis=x * u.um)
                    # Fit the spectrum and calculate the fitted flux values (``y_fit``)
                    g_init = models.Gaussian1D(amplitude=3. * u.Jy, mean=np.nanargmax(data) * u.um, stddev=miri_psf_stddev * u.um)
                    g_init.stddev.fixed = True #tied = tie_disp
                    g_init.mean.fixed = True #tied = tie_disp
                    g_fit = fit_lines(spectrum, g_init)
                    y_fit = g_fit(np.linspace(x[0],x[-1],100) * u.um)
                    self.plot_model2 = pg.PlotCurveItem(np.linspace(x[0],x[-1],100), y_fit, pen=pg.mkPen(color='yellow', style=Qt.DashLine, width=1))

                    fitter = modeling.fitting.LevMarLSQFitter()
                    model = modeling.models.Gaussian1D()  # depending on the data you need to give some initial values
                    fitted_model = fitter(model, x, data)
                    self.plot_model = pg.PlotCurveItem(x,fitted_model(x), pen='red')

                    self.vb.addItem(self.plot_model)
                    self.vb.addItem(self.plot_model2)
                self.zero_level = pg.PlotCurveItem([np.min(x),np.max(x)], [0,0], pen='white')
                self.vb.addItem(self.zero_level)

        else:
            try:
                self.vb.removeItem(self.plot_slit_profile)
                self.vb.removeItem(self.zero_level)
                self.vb.removeItem(self.plot_model)
                self.vb.removeItem(self.plot_model2)

            except:
                pass




class CUBElistTable(pg.TableWidget):
    def __init__(self, parent):
        super().__init__(editable=False, sortable=False)
        self.setStyleSheet(open('styles.ini').read())
        self.parent = parent
        self.format = None

        #self.contextMenu.addSeparator()
        #self.contextMenu.addAction('results').triggered.connect(self.show_results)

        self.resize(100, 1800)
        self.show()
        self.flags = {}
        if 1:
            self.flags['show_roi_detector'] = False
            self.flags['show_roi_disp'] = False
            self.flags['saturation_step'] = False
            self.flags['show_saturated'] = False
            self.flags['CR_step'] = False
            self.flags['show_CR'] = False
            self.flags['show_dnu'] = False
            self.flags['show_ROI'] = False
            self.flags['show_multi_CR'] = False
            self.flags['slope_fit_step'] = False
            self.flags['read_fit_slopes'] = False

    def setdata(self, data):
        self.data = data
        self.setData(data)
        if self.format is not None:
            for k, v in self.format.items():
                self.setFormat(v, self.columnIndex(k))
        self.resizeColumnsToContents()
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        #self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        w = 180 + self.verticalHeader().width() + self.autoScrollMargin()*1.5
        w += sum([self.columnWidth(c) for c in range(self.columnCount())])
        self.resize(int(w), self.size().height())
        self.setSortingEnabled(True)

    def update_cube_list(self):
        if 1:
            self.parent.parent.Cubes_A.__init__(self.parent.parent, closebutton=False)
        else:
            Cubes = self.parent.parent.Cubes
            filenames, fileparams, codenames = Cubes.readfolder(self.parent.parent.CUBE.output_dir)
            lst = []
            for s, pars in zip(filenames, fileparams):
                d = [s.split('/')[-1]]
                for p in pars:
                    d.append(p)
                lst.append(d)
                # filenamelst.append(d[0].split('/')[-1])
                Cubes.filelist[d[0].split('/')[-1]] = s
                Cubes.associtations_list[d[0].split('/')[-1]] = self.parent.parent.CUBE.output_dir + d[-1]
            lst = np.array([tuple(l) for l in lst], dtype=[('name', 'U400')] + [(p, 'U50') for p in codenames])
            data = lst
            #self.parent.parent.Cubes.table.setdata(data)
            self.setdata(data)

    def create_asn_file(self):
        source = self.parent.parent.exp_pars.asn_source.currentText()
        print('source',source)
        #break
        channel = self.parent.parent.exp_pars.asn_channel.currentText()
        band = self.parent.parent.exp_pars.asn_band.currentText()
        cube_filename = self.parent.parent.exp_pars.cube_filename.text()
        print('ASN file params:', source, channel, band)

        input_dir = self.parent.parent.CUBE_A.path
        files = self.parent.parent.CUBE_A.create_association(input_dir=input_dir,source = source,channel = channel, band =band, subfilename=cube_filename)
        print('Asn files for cube building:', files)
        #self.parent.parent.CUBE.writel3asn(files=files,asnfile=source+'_'+channel+'_'+band+)
        #sort_calfiles(files):

    def build_3dcube(self):
        master_bkgr_flag = int(self.parent.parent.exp_pars.master_bkgr_flag.currentIndex())
        master_res_bkgr_flag = int(self.parent.parent.exp_pars.master_res_bkgr_flag.currentIndex())
        master_outlier_flag = int(self.parent.parent.exp_pars.master_outlier_flag.currentIndex())
        master_resample_spec_flag = int(self.parent.parent.exp_pars.master_resample_spec_flag.currentIndex())
        master_extract1d_flag = int(self.parent.parent.exp_pars.master_extract1d_flag.currentIndex())

        channel = self.parent.parent.exp_pars.asn_channel.currentText()
        asn_file = self.parent.parent.CUBE_A.local_asn_file
        print('Built cube from asn files:')
        self.parent.parent.CUBE_A.build_cube(input_file=asn_file, channel = channel, master_bkgr_flag = master_bkgr_flag ,
                                           master_res_bkgr_flag = master_res_bkgr_flag, master_outlier_flag = master_outlier_flag,
                                           master_resample_spec_flag=master_resample_spec_flag, master_extract1d_flag = master_extract1d_flag)

    def build_3dcube_12_channels(self):
        source = self.parent.parent.exp_pars.asn_source.currentText()
        master_bkgr_flag = int(self.parent.parent.exp_pars.master_bkgr_flag.currentIndex())
        master_res_bkgr_flag = int(self.parent.parent.exp_pars.master_res_bkgr_flag.currentIndex())
        master_outlier_flag = int(self.parent.parent.exp_pars.master_outlier_flag.currentIndex())
        master_resample_spec_flag = int(self.parent.parent.exp_pars.master_resample_spec_flag.currentIndex())
        master_extract1d_flag = int(self.parent.parent.exp_pars.master_extract1d_flag.currentIndex())

        print('source', source)
        for channel in ['1','2','3','4']:
            for band in ['SHORT', 'MEDIUM','LONG']:
                cube_filename = self.parent.parent.exp_pars.cube_filename.text()
                print('ASN file params:', source, channel, band, cube_filename)
                input_dir = self.parent.parent.CUBE_A.path
                self.parent.parent.CUBE_A.create_association(input_dir=input_dir, source=source, channel=channel,
                                                                     band=band, subfilename=cube_filename)

                asn_file = self.parent.parent.CUBE_A.local_asn_file
                print('Build cube:')
                self.parent.parent.CUBE_A.build_cube(input_file=asn_file, channel=channel, master_bkgr_flag=master_bkgr_flag,
                                                     master_res_bkgr_flag=master_res_bkgr_flag,
                                                     master_outlier_flag=master_outlier_flag,
                                                     master_resample_spec_flag=master_resample_spec_flag,
                                                     master_extract1d_flag=master_extract1d_flag)

    def extract_roi(self, cube_name = '(A)'):
        if cube_name == '(A)':
            cube = self.parent.parent.plot_3dcubeA
            data = self.parent.parent.CUBE_A.data
            wavel = self.parent.parent.CUBE_A.data.wavelength
            name = self.parent.parent.CUBE_A.cubename.split('/')[-1].split('.')[0]
        elif cube_name == '(B)':
            cube = self.parent.parent.plot_3dcubeB
            data = self.parent.parent.CUBE_B.data
            wavel = self.parent.parent.CUBE_B.data.wavelength
            name = self.parent.parent.CUBE_B.cubename.split('/')[-1].split('.')[0]
        roi_name = ['sci','bkgr']
        for ir,roi in enumerate(cube.roi_list):
            if np.sum(roi.roi_mask) > 0:
                num_pixels = np.sum(roi.roi_mask)
                roi_mean_w_flux = np.zeros(cube.data.shape[0])
                roi_mean_w_f_error = np.zeros(cube.data.shape[0])
                roi_mean =  np.zeros(cube.data.shape[0])
                y = np.zeros(cube.data.shape[0])
                y_err = np.zeros(cube.data.shape[0])
                for i in range(cube.data.shape[0]):
                    d = data.data[i, :, :]
                    mask = (roi.roi_mask)*(~np.isnan(d))
                    d = d[mask]
                    derr = data.err[i, :, :]
                    derr = derr[mask]
                    w = np.power(derr,2)
                    y[i] = np.nansum(d, axis=0)
                    y_err[i] =  np.power(np.nansum(w, axis=0), 0.5)
                    roi_mean_w_flux[i] = np.nansum(d/derr**2) / np.nansum(1/derr**2)
                    roi_mean_w_f_error[i] = 1/np.nansum(1/derr**2)

                    roi_mean[i] = np.nansum(d) / np.size(d)
                if 1:
                    fig,ax = plt.subplots()
                    ax.errorbar(x=np.arange(np.size(y)),y=y,yerr=y_err,label='w_weighted',lw =2)
                    ax.errorbar(x=np.arange(np.size(y)),y=roi_mean_w_flux,yerr=roi_mean_w_f_error, label='mean weighted',ls = '--')
                    ax.plot(roi_mean, label = 'mean')
                    ax.legend()
                    plt.show()
                filename = './output/detector3/roi_spectra/'+name+'_'+cube_name+'_'+roi_name[ir]+'.spec1d'
                with open(filename, 'w') as fout:
                    #for x,y,e in zip(wavel,roi_mean_w_flux,roi_mean_w_f_error):
                    for x, y, e in zip(wavel, y, y_err):
                        fout.write('%.4e %.4e %.4e \n' %(x,y,e))
                fout.close()





    def show_roi(self,mode='A1'):
        print('show_ROI pixels:')
        if 1:
            if mode == 'A1':
                roi_mask = self.parent.parent.plot_3dcubeA.roi_list[0].roi_mask
                (timeind, time) = self.parent.parent.plot_3dcubeA.timeIndex(self.parent.parent.plot_3dcubeA.timeLine)
            elif mode == 'A2':
                roi_mask = self.parent.parent.plot_3dcubeA.roi_list[1].roi_mask
                (timeind, time) = self.parent.parent.plot_3dcubeA.timeIndex(self.parent.parent.plot_3dcubeA.timeLine)
            elif mode == 'B1':
                roi_mask = self.parent.parent.plot_3dcubeB.roi_list[0].roi_mask
                (timeind, time) = self.parent.parent.plot_3dcubeB.timeIndex(self.parent.parent.plot_3dcubeB.timeLine)
            elif mode == 'B2':
                roi_mask = self.parent.parent.plot_3dcubeB.roi_list[1].roi_mask
                (timeind, time) = self.parent.parent.plot_3dcubeB.timeIndex(self.parent.parent.plot_3dcubeB.timeLine)
            elif mode == 'S1':
                roi_mask = self.parent.parent.plot_3dcubeA.roi_add.roi_mask
                (timeind, time) = self.parent.parent.plot_3dcubeA.timeIndex(self.parent.parent.plot_3dcubeA.timeLine)

            x, y = np.where(roi_mask > 0)
            lam = time
            lam_array = np.zeros_like(x) + lam
            print('roi pixels at lambda=',lam)
            if mode in ['A1', 'A2','S1']:
                lam_world, x_world,y_world = self.parent.parent.CUBE_A.conv_world_coord(t=lam_array, x=y, y=x)
            elif mode in ['B1', 'B2']:
                lam_world, x_world, y_world = self.parent.parent.CUBE_B.conv_world_coord(t=lam_array, x=y, y=x)
            if self.flags['show_ROI'] == False:
                self.flags['show_ROI'] = True
                if mode in ['A1','A2','S1']:
                    self.parent.parent.plot_3dcubeA.selectPixels(add=self.flags['show_ROI'], x=y, y=x, color='m',type='cr_multi')
                elif mode in ['B1','B2']:
                    self.parent.parent.plot_3dcubeB.selectPixels(add=self.flags['show_ROI'], x=y, y=x, color='m',
                                                                 type='cr_multi')
                if self.flags['show_roi_detector'] == True:
                    wcs1 = self.parent.parent.stage2[0].data.meta.wcs
                    cal_world_to_detector1 = wcs1.get_transform('world', 'detector')
                    (x_det1, y_det1) = cal_world_to_detector1(x_world, y_world, lam_world)
                    mask = (x_det1>=0)*(x_det1<=self.parent.parent.stage2[0].data.data.shape[0])*(y_det1>-1)
                    x_det1,y_det1 = np.array(x_det1[mask],dtype=int),np.array(y_det1[mask],dtype=int)
                    text = 'ROI: %.2f' % (np.nansum(self.parent.parent.stage2[0].data.data[y_det1,x_det1])) #/np.size(x_det1))
                    print(text)
                    self.parent.parent.plot_2dimage1.Roilabel.setText(text)
                    #self.parent.parent.plot_2dimage1.resize(400, 40)

                    wcs2 = self.parent.parent.stage2[1].data.meta.wcs
                    cal_world_to_detector2 = wcs2.get_transform('world', 'detector')
                    (x_det2, y_det2) = cal_world_to_detector2(x_world, y_world, lam_world)
                    mask = (x_det2 >= 0) * (x_det2 <= self.parent.parent.stage2[1].data.shape[0])*(y_det2>-1)
                    x_det2,y_det2 = np.array(x_det2[mask],dtype=int),np.array(y_det2[mask],dtype=int)
                    text = 'ROI: %.2f' % (np.nansum(self.parent.parent.stage2[1].data.data[y_det2,x_det2])) #/ np.size(x_det2))
                    print(text)
                    self.parent.parent.plot_2dimage2.Roilabel.setText(text)

                    wcs3 = self.parent.parent.stage2[2].data.meta.wcs
                    cal_world_to_detector3 = wcs3.get_transform('world', 'detector')
                    (x_det3, y_det3) = cal_world_to_detector3(x_world, y_world, lam_world)
                    mask = (x_det3 >= 0) * (x_det3 <= self.parent.parent.stage2[2].data.shape[0])*(y_det3>0)
                    x_det3,y_det3 = np.array(x_det3[mask],dtype=int),np.array(y_det3[mask],dtype=int)
                    text = 'ROI: %.2f' % (np.nansum(self.parent.parent.stage2[2].data.data[y_det3, x_det3]) )#/ np.size(x_det3))
                    print(text)
                    self.parent.parent.plot_2dimage3.Roilabel.setText(text)

                    wcs4 = self.parent.parent.stage2[3].data.meta.wcs
                    cal_world_to_detector4 = wcs4.get_transform('world', 'detector')
                    (x_det4, y_det4) = cal_world_to_detector4(x_world, y_world, lam_world)
                    mask = (x_det4 >= 0) * (x_det4 <= self.parent.parent.stage2[3].data.shape[0])*(y_det4>0)
                    x_det4,y_det4 = np.array(x_det4[mask],dtype=int),np.array(y_det4[mask],dtype=int)
                    text = 'ROI: %.2f' % (np.nansum(self.parent.parent.stage2[3].data.data[y_det4, x_det4])) #/ np.size(x_det4))
                    print(text)
                    self.parent.parent.plot_2dimage4.Roilabel.setText(text)



                    #x,y = np.array(x_det),np.array(y_det)
                    self.parent.parent.plot_2dimage1.selectPixels(add=self.flags['show_ROI'], x=x_det1, y=y_det1,color='m',type='cr_multi')
                    print('x_det1')
                    print(x_det1)
                    #self.parent.parent.plot_2dimage1.vb.setLimits(xMin=np.min(x_det1), xMax=np.max(x_det1), yMin=np.min(y_det1), yMax=np.max(x_det1))
                    self.parent.parent.plot_2dimage1.vb.setRange(xRange=(np.min(x_det1)*0.95,np.max(x_det1)*1.05), yRange=(np.min(y_det1)*0.95,np.max(y_det1)*1.05))
                    self.parent.parent.plot_2dimage2.selectPixels(add=self.flags['show_ROI'], x=x_det2, y=y_det2,color='m',type='cr_multi')
                    self.parent.parent.plot_2dimage2.vb.setRange(xRange=(np.min(x_det2) * 0.95, np.max(x_det2) * 1.05),
                                                                 yRange=(np.min(y_det2) * 0.95, np.max(y_det2) * 1.05))
                    self.parent.parent.plot_2dimage3.selectPixels(add=self.flags['show_ROI'], x=x_det3, y=y_det3,color='m',type='cr_multi')
                    self.parent.parent.plot_2dimage3.vb.setRange(xRange=(np.min(x_det3) * 0.95, np.max(x_det3) * 1.05),
                                                                 yRange=(np.min(y_det3) * 0.95, np.max(y_det3) * 1.05))
                    self.parent.parent.plot_2dimage4.selectPixels(add=self.flags['show_ROI'], x=x_det4, y=y_det4,color='m',type='cr_multi')
                    self.parent.parent.plot_2dimage4.vb.setRange(xRange=(np.min(x_det4) * 0.95, np.max(x_det4) * 1.05),
                                                                 yRange=(np.min(y_det4) * 0.95, np.max(y_det4) * 1.05))



            else:
                self.flags['show_ROI'] = False
                if mode in ['A1', 'A2','S1']:
                    self.parent.parent.plot_3dcubeA.selectPixels(add=False, type='cr_multi')
                elif mode in ['B1', 'B2']:
                    self.parent.parent.plot_3dcubeB.selectPixels(add=False, type='cr_multi')
                if self.flags['show_roi_detector'] == True:
                    self.parent.parent.plot_2dimage1.selectPixels(add=False, type='cr_multi')
                    self.parent.parent.plot_2dimage2.selectPixels(add=False, type='cr_multi')
                    self.parent.parent.plot_2dimage3.selectPixels(add=False, type='cr_multi')
                    self.parent.parent.plot_2dimage4.selectPixels(add=False, type='cr_multi')



    def show_detector_roi(self,add=True,mode = 'A'):
        print('show_ROI pixels:')
        #plt = pg.plot()
        #plt1 =  QSplitter(Qt.Vertical)
        #plot_2dimage2 = plotImage(self)
        #plt1.addWidget(plot_2dimage2)
        if 1:
            if add:
                self.flags['show_roi_detector']=True
                self.parent.parent.spec_image1.show()
                if mode == 'A':
                    name = self.parent.parent.plot_3dcubeA.init_name
                elif mode == 'B':
                    name = self.parent.parent.plot_3dcubeB.init_name
                self.parent.parent.plot_2dimage1.add(name, add=add, Nscreen=1)
                self.parent.parent.plot_2dimage2.add(name, add=add, Nscreen=2)
                self.parent.parent.plot_2dimage3.add(name, add=add, Nscreen=3)
                self.parent.parent.plot_2dimage4.add(name, add=add, Nscreen=4)
            else:
                self.flags['show_roi_detector']=False

        if 0:
            app1 = QApplication([])
            show_gui = 0

            win = pg.GraphicsLayoutWidget(show=True)
            win.setWindowTitle('pyqtgraph example: Scrolling Plots')
            # 1) Simplest approach -- update data in the array such that plot appears to scroll
            #    In these examples, the array size is fixed.
            p1 = win.addPlot()
            p2 = win.addPlot()
            data1 = np.random.normal(size=300)
            curve1 = p1.plot(data1)
            curve2 = p2.plot(data1)
            ptr1 = 0

            def update1():
                global data1, ptr1
                data1[:-1] = data1[1:]  # shift data in the array one sample left
                # (see also: np.roll)
                data1[-1] = np.random.normal()
                curve1.setData(data1)

                ptr1 += 1
                curve2.setData(data1)
                curve2.setPos(ptr1, 0)

            # 2) Allow data to accumulate. In these examples, the array doubles in length
            #    whenever it is full.
            win.nextRow()
            p3 = win.addPlot()
            p4 = win.addPlot()
            # Use automatic downsampling and clipping to reduce the drawing load
            p3.setDownsampling(mode='peak')
            p4.setDownsampling(mode='peak')
            p3.setClipToView(True)
            p4.setClipToView(True)
            p3.setRange(xRange=[-100, 0])
            p3.setLimits(xMax=0)
            curve3 = p3.plot()
            curve4 = p4.plot()

            data3 = np.empty(100)
            ptr3 = 0

            def update2():
                global data3, ptr3
                data3[ptr3] = np.random.normal()
                ptr3 += 1
                if ptr3 >= data3.shape[0]:
                    tmp = data3
                    data3 = np.empty(data3.shape[0] * 2)
                    data3[:tmp.shape[0]] = tmp
                curve3.setData(data3[:ptr3])
                curve3.setPos(-ptr3, 0)
                curve4.setData(data3[:ptr3])

            # 3) Plot in chunks, adding one new plot curve for every 100 samples
            chunkSize = 100
            # Remove chunks after we have 10
            maxChunks = 10
            win.nextRow()
            p5 = win.addPlot(colspan=2)
            p5.setLabel('bottom', 'Time', 's')
            p5.setXRange(-10, 0)
            curves = []
            data5 = np.empty((chunkSize + 1, 2))
            ptr5 = 0

            app1.exec_()  # Start QApplication event loop ***

    def show_roi_dispersion(self,show=True):
        if show:
            self.parent.parent.roi_hist.show()
            self.flags['show_roi_disp'] = True
            if 0:
                self.parent.parent.roi_hist.show()
                for mode,ind in zip(['A1','A2','B1','B2'],[0,1,0,1]):
                    if 'A' in mode:
                        roi_mask = self.parent.parent.plot_3dcubeA.roi_list[ind].roi_mask
                        (timeind, time) = self.parent.parent.plot_3dcubeA.timeIndex(self.parent.parent.plot_3dcubeA.timeLine)
                        data = self.parent.parent.CUBE_A.data
                    elif 'B' in mode:
                        roi_mask = self.parent.parent.plot_3dcubeB.roi_list[ind].roi_mask
                        (timeind, time) = self.parent.parent.plot_3dcubeB.timeIndex(self.parent.parent.plot_3dcubeB.timeLine)
                        data = self.parent.parent.CUBE_B.data
                    if mode == 'A1':
                        self.parent.parent.plot_hist1.plot_hist(data=data, roi_mask=roi_mask, timeind=time, add=True)


    def show_roi_slit_command(self,add=True,rad=1):
        if add:
            self.parent.parent.roi_slit.show()
        self.parent.parent.plot_3dcubeA.add_Roi(add=add)

    def show_gradient_command(self, add=True,level=1,local=True):
        #if add:
        #    self.parent.parent.roi_slit.show()
        self.parent.parent.plot_3dcubeA.add_contours(add=add,level=level,local=local)


    def calc_median_cube(self,add=True,radius = 3,mode='mean-weighted',save_cube=True,debug =True, method = '3Dsmothing'):
        '''

        :param add:
        :param radius:
        :param mode:
        :param save_cube:
        :param debug:
        :param method = '3Dsmothing', '2Dsmothing','Average','Pix2pix'
        :return:
        '''
        if add:
            if method in ['3Dsmothing','2Dsmothing']:
                print('Method of construction of the Background model', method)
                print('kernel radius = ',radius)
                cube = self.parent.parent.CUBE_B
                data =cube.data.data
                err = cube.data.err
                dq = cube.data.dq
                mean_data = np.zeros_like(data)
                mean_err = np.zeros_like(err)
                filter_kernel = np.zeros((2 * radius+1, 2 * radius+1))
                for i in range(filter_kernel.shape[0]):
                    for j in range(filter_kernel.shape[1]):
                        if (i-radius)**2+(j-radius)**2 <= radius**2:
                            filter_kernel[i,j] = 1

                for i in range(data.shape[0]):
                    slice = data[i,:,:]
                    slice_err = err[i,:,:]
                    slice[np.isnan(slice)] = 0
                    slice_err[np.isnan(slice)] = 1e-10
                    slice_one_array = np.ones_like(slice)
                    slice_one_array[slice == 0] = 0

                    #calculate spatial smoothing
                    if 1:
                        npixels = scipy.signal.convolve2d(slice_one_array, filter_kernel, mode='same', boundary='fill', fillvalue=0)
                        mean_data[i,:,:] = scipy.signal.convolve2d(slice, filter_kernel, mode='same', boundary='fill', fillvalue=0)/npixels
                        mean_data[i, :, :][slice == 0] = 0
                        #mean_2_data[i, :, :] = scipy.signal.convolve2d(slice/slice_err**2, filter_kernel,
                        #                                             mode='same', boundary='fill', fillvalue=0) / scipy.signal.convolve2d(1/slice_err**2, filter_kernel,
                        #                                             mode='same', boundary='fill', fillvalue=0)
                        mean_err[i, :, :] = np.power(scipy.signal.convolve2d(slice_err**2,filter_kernel,mode='same',boundary='fill',fillvalue=0),0.5)/npixels

                        #calc_manually
                        if 0:
                            for j in range(data.shape[1]):
                                for k in range(data.shape[2]):
                                    mask = np.zeros_like(slice)
                                    for jj in range(data.shape[1]):
                                        for kk in range(data.shape[2]):
                                            r = np.sqrt((jj-j)**2 + (kk-k)**2)
                                            if r<=radius:
                                                mask[jj,kk] =  1*(slice_dq[jj,kk]==0)
                                    if np.sum(mask)>0:
                                        mask = np.array(mask, dtype=bool)
                                        d= slice[mask]
                                        derr = slice_err[mask]
                                        mean_data[i,j,k] = np.sum(d / derr ** 2) / np.sum(1 / derr ** 2)
                                        mean_err[i,j,k]= 1 / np.sum(1 / derr ** 2)
                                    else:
                                        mean_data[i,j, k] = np.nan
                                        mean_err[i,j, k] = np.nan

                    # calculate wavelength smoothing
                    if method == '3Dsmothing':
                        show_smoothed_spec = False
                        win = signal.windows.hann(20)
                        for i in range(data.shape[1]):
                            for j in range(data.shape[2]):
                                if i == 20 and j ==20 and show_smoothed_spec:
                                    fig2, ax2 = plt.subplots()
                                    ax2.plot(mean_data[:,i,j],label='init',lw=2)
                                mask = (np.arange(data.shape[0])>10 )*(np.arange(data.shape[0])<data.shape[0]-10 )
                                convloved = signal.convolve(mean_data[:,i,j], win, mode='same')/ sum(win)
                                mean_data[:, i, j][mask] = convloved[mask]
                                if i == 20 and j == 20 and show_smoothed_spec:
                                    ax2.plot(mean_data[:, i, j], label='smoothed',ls='--',lw=2)
                                    ax2.legend()
                                mean_err[:, i, j] = signal.convolve(mean_err[:, i, j], win, mode='same') / sum(win)
                if debug:
                    fig,ax = plt.subplots(1,4)
                    ax[0].imshow(data[0,:,:])
                    ax[1].imshow(mean_data[0, :, :])
                    ax[2].imshow(data[0,:,:]-mean_data[0,:,:])
                    ax[3].hist(data[0,:,:].flatten()-mean_data[0,:,:].flatten(),label='subtracted')
                    ax[3].hist(data[0,:,:].flatten(),label='initial',alpha=0.5)
                    ax[3].hist(mean_data[0, :, :].flatten(), label='model',alpha=0.5)
                    ax[3].legend()


                    if 0:
                        fig2, ax2 = plt.subplots(1, 4)
                        ax2[0].imshow(data[0, :, :])
                        ax2[1].imshow(mean_2_data[0, :, :])
                        ax2[2].imshow(data[0, :, :] - mean_2_data[0, :, :])
                        ax2[3].hist(data[0,:,:].flatten()-mean_2_data[0,:,:].flatten(),label='subtracted-weighted',alpha=0.5)
                        ax2[3].hist(data[0,:,:].flatten(),label='initial',alpha=0.5)
                        ax2[3].legend()

                    plt.show()

            elif method == 'Average':
                print('Method of construction of the Background model', method)
                cube = self.parent.parent.CUBE_B
                data = cube.data.data
                err = cube.data.err
                mean_spec = np.nanmean(np.nanmean(data,axis=1),axis=1)
                mean_spec_err = np.nanstd(np.nanstd(data,axis=1),axis=1)
                mean_data = np.zeros_like(data)
                mean_err = np.zeros_like(err)
                for i in range(data.shape[1]):
                    for j in range(data.shape[2]):
                        mean_data[:,i,j] = mean_spec
                        mean_err[:,i,j] = mean_spec_err
                if 1:
                    mean_err1 = np.zeros_like(mean_spec)
                    for i in range(err.shape[0]):
                        slice_err = err[i,:,:]
                        mean_err1[i] = np.sqrt(np.nansum(np.power(slice_err.flatten(),2)))/np.sum(~np.isnan(slice_err.flatten()))


                if debug:
                    fig, ax = plt.subplots(1, 4)
                    ax[0].imshow(data[0, :, :])
                    ax[1].imshow(mean_data[0, :, :])
                    ax[2].imshow(data[0, :, :] - mean_data[0, :, :])
                    ax[3].hist(data[0, :, :].flatten() - mean_data[0, :, :].flatten(), label='subtracted')
                    ax[3].hist(data[0, :, :].flatten(), label='initial', alpha=0.5)
                    ax[3].hist(mean_data[0, :, :].flatten(), label='model', alpha=0.5)
                    ax[3].legend()

                    for i in range(data.shape[1]):
                        for j in range(data.shape[2]):
                            if i == 20 and j == 20:
                                fig2, ax2 = plt.subplots()
                                ax2.plot(data[:, i, j], label='init', lw=2)
                                mask = (np.arange(data.shape[0]) > 10) * (np.arange(data.shape[0]) < data.shape[0] - 10)
                                ax2.plot(mean_data[:, i, j], label='smoothed', ls='--', lw=2)
                                ax2.legend()

                    plt.show()

            elif method == 'Pix2pix':
                print('Method of construction of the Background model', method)
                cube = self.parent.parent.CUBE_B
                data = cube.data.data
                err = cube.data.err
                mean_data = data.copy()
                mean_err = err.copy()

                if debug:
                    fig, ax = plt.subplots(1, 4)
                    ax[0].imshow(data[0, :, :])
                    ax[1].imshow(mean_data[0, :, :])
                    ax[2].imshow(data[0, :, :] - mean_data[0, :, :])
                    ax[3].hist(data[0, :, :].flatten() - mean_data[0, :, :].flatten(), label='subtracted')
                    ax[3].hist(data[0, :, :].flatten(), label='initial', alpha=0.5)
                    ax[3].hist(mean_data[0, :, :].flatten(), label='model', alpha=0.5)
                    ax[3].legend()
                    plt.show()

            if save_cube:
                filename = './output/detector3/cash/median_cube.fits'
                hdu1 = fits.open(cube.cubename)
                hdu1['SCI'].data = mean_data
                hdu1['ERR'].data = mean_err
                hdu1.writeto(filename,overwrite=True)
                self.parent.parent.plot_3dcube_median.add_from_file(self, filename=filename, add=True)


    def save_local_cube_code(self,cube_name = 'A'):
        if cube_name == 'A':
            cube = self.parent.parent.CUBE_A
            data = self.parent.parent.CUBE_A.data
            wavel = self.parent.parent.CUBE_A.data.wavelength
            name = self.parent.parent.CUBE_A.cubename.split('/')[-1].split('.')[0]
        elif cube_name == 'B':
            cube = self.parent.parent.CUBE_B
            data = self.parent.parent.CUBE_B.data
            wavel = self.parent.parent.CUBE_B.data.wavelength
            name = self.parent.parent.CUBE_B.cubename.split('/')[-1].split('.')[0]


        filename = cube.cubename.split('.fits')[0]+'_bkgr_subtr.fits'
        hdu1 = fits.open(cube.cubename)
        hdu1['SCI'].data = data.data
        hdu1['ERR'].data = data.err
        hdu1.writeto(filename,overwrite=True)
        print('new cube is seved in', filename)


class chooseExpWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent, closebutton=True,cube_choice='A'):
        super().__init__()
        self.parent = parent
        self.cube_choice=cube_choice
        #self.resize(700, 900)
        #self.move(400, 100)
        self.setStyleSheet(open('styles.ini').read())

        self.saved = []

        layout = QVBoxLayout()

        self.table = CUBElistTable(self)
        self.filelist = {}
        self.associtations_list = {}

        if 1:
            filenames,fileparams,codenames = self.readfolder(self.parent.CUBE_A.output_dir)
            lst = []
            if len(filenames)>0:
                for s,pars in zip(filenames,fileparams):
                    d = [s.split('/')[-1]]
                    for p in pars:
                        d.append(p)
                    lst.append(d)
                    #filenamelst.append(d[0].split('/')[-1])
                    self.filelist[d[0].split('/')[-1]]=s
                    self.associtations_list[d[0].split('/')[-1]]=self.parent.CUBE_A.path +'/'+ d[4]
                lst = np.array([tuple(l) for l in lst], dtype=[('name','U400')] + [(p,'U50') for p in codenames])
            data = lst
        if len(data)>0:
            self.table.setdata(data)
            self.buttons = {}
            for i, d in enumerate(data):
                wdg = QWidget()
                l = QVBoxLayout()
                l.addSpacing(3)
                button = QPushButton(d[0].split('_uncal')[0], self, checkable=True)
                #button.setFixedSize(600, 30)
                button.resize(600, 30)
                button.setChecked(False)
                button.clicked[bool].connect(partial(self.click, d[0]))
                #button.clicked.connect(partial(self.click, d[0]))
                self.buttons[d[0]] = button
                l.addWidget(button)
                l.addSpacing(3)
                l.setContentsMargins(0, 0, 0, 0)
                wdg.setLayout(l)
                self.table.setCellWidget(i, 0, wdg)

        layout.addWidget(self.table)

        self.scroll = None

        self.layout = QVBoxLayout()
        layout.addLayout(self.layout)

        if closebutton and 1:
            self.okButton = QPushButton("Close")
            #self.okButton.setFixedSize(110, 30)
            self.okButton.resize(110, 30)
            self.okButton.clicked[bool].connect(self.ok)
            hbox = QHBoxLayout()
            hbox.addWidget(self.okButton)
            hbox.addStretch()
            layout.addLayout(hbox)

        self.setLayout(layout)


    def readfolder(self, folder='', verbose=False):
        """
        Read list of models from the folder
        """
        self.tablefiles = {}
        self.association_files = {}
        if 1:
            lst = []
            params= []
            for (dirpath, dirname, filenames) in os.walk(folder):
                print(dirpath, dirname, filenames)
                for k,f in enumerate(filenames):
                    if f.endswith('_s3d.fits'):
                        print(k,'from', len(filenames))
                        #lst.append(dirpath.split('/')[-1]+'/'+f)
                        lst.append(dirpath+ f)
                        params.append(self.readfile(pathotofile=dirpath,filename=f))
                        self.tablefiles[f] = f
        codenames = ['TARGPROP','BAND','CHANNEL','ASNFILE','MRSMAT','OUTLIR','BKGSUB']

        return lst,params,codenames

    def readfile(self,pathotofile = '',filename = ''):
        hdulist = fits.open(pathotofile+'/'+filename)
        header = hdulist[0].header
        prog_id, obs_id, targ_name, miri_band, miri_channel = None,None,None,None,None
        asn_file, s_mrsmat, s_outlir, s_bkgsub= None,None,None,None
        if 'PROGRAM' in header.keys():
            prog_id = header['PROGRAM']
        if 'OBSERVTN' in header.keys():
            obs_id = header['OBSERVTN']
        if 'TARGPROP' in header.keys():
            targ_name = header['TARGPROP']
        if 'BAND' in header.keys():
            miri_band = header['BAND']
        if 'CHANNEL' in header.keys():
            miri_channel = header['CHANNEL']
        if 'ASNTABLE' in header.keys():
            asn_file = header['ASNTABLE']
        if 'S_MRSMAT' in header.keys():
            s_mrsmat = header['S_MRSMAT']
        if 'S_OUTLIR' in header.keys():
            s_outlir = header['S_OUTLIR']
        if 'BKGSUB' in header.keys():
            s_bkgsub = str(header['BKGSUB'])

        return [targ_name,miri_band,miri_channel,asn_file,s_mrsmat,s_outlir,s_bkgsub]




    def click(self, name):
        self.parent.current_name = name
        if self.cube_choice == 'A':
            self.parent.plot_3dcubeA.add(name, self.buttons[name].isChecked())
        elif self.cube_choice == 'B':
            self.parent.plot_3dcubeB.add(name, self.buttons[name].isChecked())
        print('self.buttons[name].isChecked()',self.buttons[name].isChecked())
        if self.buttons[name].isChecked()==True:
            flags = self.parent.Cubes_A.table.flags
            self.parent.Cubes_A.table.flags = flags.fromkeys(flags, False)
            print('added by click', self.parent.Cubes_A.table.flags)
        p = np.where(self.table.data['name'] == name)
        self.table.setCurrentCell(np.where(self.table.data['name'] == name)[0][0], 0)

    def ok(self):
        self.hide()
        self.parent.chooseFit = None
        self.deleteLater()

    def cancel(self):
        for par in self.parent.fit.list():
            par.fit = self.saved[str(par)]
        self.close()

class expParsWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        #self.resize(700, 900)
        #self.move(400, 100)
        #self.pars = {'n0': 'x', 'uv': 'y', 'Z': 'fixed', 'Av': 'disable', 'NCO': 'z'}
        #self.parent.H2.setgrid(pars=list(self.pars.keys()), show=False)
        #self.cols, self.x_, self.y_, self.z_, self.lnL_, self.mpars = None, None, None, None, None, None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Parameters:'))


        #l = QHBoxLayout(self)
        #l.addWidget(QLabel('Exp1:'))
        #%self.nEXP1 = QLineEdit()
        #%self.nEXP1.setText(str(0))
        #self.nEXP1.setFixedSize(60, 30)
        #l.addWidget(self.nEXP1)
        #l.addWidget(QLabel('Exp2:'))
        #self.nEXP2 = QLineEdit()
        #self.nEXP2.setText(str(1))
        #self.nEXP2.setFixedSize(60, 30)
        #l.addWidget(self.nEXP2)
        #l.addWidget(QLabel('Exp3:'))
        #self.nEXP3 = QLineEdit()
        #self.nEXP3.setText(str(2))
        #self.nEXP3.setFixedSize(60, 30)
        #l.addWidget(self.nEXP3)
        #l.addWidget(QLabel('Exp4:'))
        #self.nEXP4 = QLineEdit()
        #self.nEXP4.setText(str(3))
        #self.nEXP4.setFixedSize(60, 30)
        #l.addWidget(self.nEXP4)
        #l.addStretch(1)
        #layout.addLayout(l)

        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('MasterBkgr:'))
        self.master_bkgr_flag = QComboBox()
        self.master_bkgr_flag.addItems(['No', 'Yes'])
        self.master_bkgr_flag.setCurrentIndex(0)
        #self.master_bkgr_flag.setFixedSize(90, 30)
        self.master_bkgr_flag.resize(90, 30)
        horizontal_layout.addWidget(self.master_bkgr_flag)

        horizontal_layout.addWidget(QLabel('ResBkgrMatch:'))
        self.master_res_bkgr_flag = QComboBox()
        self.master_res_bkgr_flag.addItems(['No', 'Yes'])
        self.master_res_bkgr_flag.setCurrentIndex(1)
        #self.master_res_bkgr_flag.setFixedSize(90, 30)
        self.master_res_bkgr_flag.resize(90, 30)
        horizontal_layout.addWidget(self.master_res_bkgr_flag)

        horizontal_layout.addWidget(QLabel('OutlierDet:'))
        self.master_outlier_flag = QComboBox()
        self.master_outlier_flag.addItems(['No', 'Yes'])
        self.master_outlier_flag.setCurrentIndex(1)
        #self.master_outlier_flag.setFixedSize(90, 30)
        self.master_outlier_flag.resize(90, 30)
        horizontal_layout.addWidget(self.master_outlier_flag)
        horizontal_layout.addStretch(1)
        layout.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)

        horizontal_layout.addWidget(QLabel('ResampleSpec:'))
        self.master_resample_spec_flag = QComboBox()
        self.master_resample_spec_flag.addItems(['No', 'Yes'])
        self.master_resample_spec_flag.setCurrentIndex(1)
        #self.master_resample_spec_flag.setFixedSize(90, 30)
        self.master_resample_spec_flag.resize(90, 30)
        horizontal_layout.addWidget(self.master_resample_spec_flag)


        horizontal_layout.addWidget(QLabel('Extract1D:'))
        self.master_extract1d_flag = QComboBox()
        self.master_extract1d_flag.addItems(['No', 'Yes'])
        self.master_extract1d_flag.setCurrentIndex(1)
        #self.master_extract1d_flag.setFixedSize(90, 30)
        self.master_extract1d_flag.resize(90, 30)
        horizontal_layout.addWidget(self.master_extract1d_flag)

        horizontal_layout.addStretch(1)
        layout.addLayout(horizontal_layout)


        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('Source:'))
        self.asn_source = QComboBox()
        if 1:
            lst = self.readfolder(self.parent.CUBE_A.path)
        self.asn_source.addItems(lst) #['Object', 'Background','Both'])
        self.asn_source.setCurrentIndex(0)
        #p = self.asn_source.currentText()
        #print(self.asn_source.itemData[0])
        #self.asn_source.setFixedSize(90, 30)
        self.asn_source.resize(90, 30)
        horizontal_layout.addWidget(self.asn_source)

        horizontal_layout.addWidget(QLabel('Channel:'))
        self.asn_channel = QComboBox()
        self.asn_channel.addItems(['1', '2','3','4'])
        self.asn_channel.setCurrentIndex(0)
        #self.asn_channel.setFixedSize(90, 30)
        self.asn_channel.resize(90, 30)
        horizontal_layout.addWidget(self.asn_channel)

        horizontal_layout.addWidget(QLabel('Band:'))
        self.asn_band = QComboBox()
        self.asn_band.addItems(['SHORT', 'MEDIUM', 'LONG','ABC'])
        self.asn_band.setCurrentIndex(0)
        self.asn_band.resize(90, 30)
        horizontal_layout.addWidget(self.asn_band)

        horizontal_layout.addStretch(1)
        layout.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('Cubename:'))
        self.cube_filename = QLineEdit()
        self.cube_filename.setText('')
        self.cube_filename.resize(200, 30)
        horizontal_layout.addWidget(self.cube_filename)
        horizontal_layout.addStretch(1)
        layout.addLayout(horizontal_layout)


        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addStretch(1)
        layout.addLayout(horizontal_layout)


        layout.addStretch(1)
        self.setLayout(layout)

        self.setStyleSheet(open('styles.ini').read())

    def readfolder(self, folder='./'):
        """
        Read list of names of objects from the folder
        """
        lst = []
        for (dirpath, dirname, filenames) in os.walk(folder):
            for k, f in enumerate(filenames):
                if f.endswith('_cal.fits'):
                    hdulist = fits.open(dirpath+'/'+f)
                    header = hdulist[0].header
                    if 'TARGPROP' in header.keys():
                        obj_name = header['TARGPROP']
                        if obj_name not in lst:
                            lst.append(obj_name)
        return lst

class expRunWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        #self.resize(700, 900)
        #self.move(400, 100)
        #self.pars = {'n0': 'x', 'uv': 'y', 'Z': 'fixed', 'Av': 'disable', 'NCO': 'z'}
        #self.parent.H2.setgrid(pars=list(self.pars.keys()), show=False)
        #self.cols, self.x_, self.y_, self.z_, self.lnL_, self.mpars = None, None, None, None, None, None

        layout = QVBoxLayout(self)

        l = QVBoxLayout(self)
        l.addWidget(QLabel('Commands:'))

        horizontal_layout = QHBoxLayout(self)
        #self.create_asn = QPushButton('Create AssFile')
        #self.create_asn.clicked[bool].connect(partial(self.create_ASN_file))
        #self.create_asn.setFixedSize(200, 60)
        #horizontal_layout.addWidget(self.create_asn)
        self.build_cube = QPushButton('Build cube')
        self.build_cube.clicked[bool].connect(partial(self.call_build_3dCube))
        #self.build_cube.setFixedSize(200, 60)
        self.build_cube.resize(200, 60)
        horizontal_layout.addWidget(self.build_cube)
        self.build_cube_12ch = QPushButton('Build 12cubes')
        self.build_cube_12ch.clicked[bool].connect(partial(self.call_build_cube_12_ch))
        #self.build_cube_12ch.setFixedSize(200, 60)
        self.build_cube_12ch.resize(200, 60)
        horizontal_layout.addWidget(self.build_cube_12ch)
        self.update_cubes_list = QPushButton('Update_List')
        self.update_cubes_list.clicked[bool].connect(partial(self.update_CubeList))
        #self.update_cubes_list.setFixedSize(200, 60)
        self.update_cubes_list.resize(200, 60)
        horizontal_layout.addWidget(self.update_cubes_list)

        self.save_local_cube = QPushButton('Save Cube', self, checkable=True)
        self.save_local_cube.setChecked(False)
        self.save_local_cube.clicked[bool].connect(partial(self.SaveLocalCube))
        #self.save_local_cube.setFixedSize(200, 60)
        self.save_local_cube.resize(200, 60)
        horizontal_layout.addWidget(self.save_local_cube)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)
        layout.addLayout(l)

        l = QVBoxLayout(self)
        l.addWidget(QLabel('Background model:'))
        horizontal_layout = QHBoxLayout(self)
        l.addLayout(horizontal_layout)
        horizontal_layout = QHBoxLayout(self)
        self.calc_median_flux = QPushButton('CalcMedian', self, checkable=True)
        self.calc_median_flux.setChecked(False)
        self.calc_median_flux.clicked[bool].connect(partial(self.CalcMedCube))
        #self.calc_median_flux.setFixedSize(200, 60)
        self.calc_median_flux.resize(200, 60)
        horizontal_layout.addWidget(self.calc_median_flux)
        self.calc_median_mode = QComboBox()
        self.calc_median_mode.addItems(['3Dsmothing', '2Dsmothing','Average','Pix2pix'])
        self.calc_median_mode.setCurrentIndex(0)
        #self.calc_median_mode.setFixedSize(90, 30)
        self.calc_median_mode.resize(90, 30)
        horizontal_layout.addWidget(self.calc_median_mode)
        horizontal_layout.addWidget(QLabel('Rad:'))
        self.mean_kernel_rad = QLineEdit()
        self.mean_kernel_rad.setText(str(5))
        #self.mean_kernel_rad.setFixedSize(60, 30)
        self.mean_kernel_rad.resize(60, 30)
        horizontal_layout.addWidget(self.mean_kernel_rad)
        self.subtr_median_flux = QPushButton('SubtractMed', self, checkable=True)
        self.subtr_median_flux.setChecked(False)
        self.subtr_median_flux.clicked[bool].connect(partial(self.SubtractMedFlux))
        #self.subtr_median_flux.setFixedSize(200, 60)
        self.subtr_median_flux.resize(200, 60)
        horizontal_layout.addWidget(self.subtr_median_flux)
        self.name_cube_subtracted = QComboBox()
        self.name_cube_subtracted.addItems(['A', 'B'])
        self.name_cube_subtracted.setCurrentIndex(1)
        #self.name_cube_subtracted.setFixedSize(90, 30)
        self.name_cube_subtracted.resize(90, 30)
        horizontal_layout.addWidget(self.name_cube_subtracted)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)
        layout.addLayout(l)

        l = QVBoxLayout(self)
        l.addWidget(QLabel('Spectrum extraction:'))
        horizontal_layout = QHBoxLayout(self)
        l.addLayout(horizontal_layout)
        self.select_roi = QPushButton('Select ROI')
        #self.okButton = QPushButton("Close")
        #self.okButton.setFixedSize(110, 30)
        #self.okButton.clicked[bool].connect(self.ok)
        self.select_roi.clicked[bool].connect(self.ShowROI)
        #self.select_roi.setFixedSize(200, 60)
        self.select_roi.resize(200, 60)
        horizontal_layout.addWidget(self.select_roi)
        self.roi_type = QComboBox()
        self.roi_type.addItems(['green', 'red', 'purple', 'yellow', 'blue'])
        self.roi_type.setCurrentIndex(0)
        #self.roi_type.setFixedSize(90, 30)
        self.roi_type.resize(90, 30)
        horizontal_layout.addWidget(self.roi_type)
        self.show_roi = QPushButton('Show Detector', self, checkable=True)
        self.show_roi.setChecked(False)
        self.show_roi.clicked[bool].connect(partial(self.ShowDetectorROI))
        #self.show_roi.setFixedSize(200, 60)
        self.show_roi.resize(200, 60)
        horizontal_layout.addWidget(self.show_roi)
        horizontal_layout.addWidget(QLabel('NormView'))
        self.norm_flag_roi = QComboBox()
        self.norm_flag_roi.addItems(['no', 'yes'])
        self.norm_flag_roi.setCurrentIndex(0)
        #self.norm_flag_roi.setFixedSize(90, 30)
        self.norm_flag_roi.resize(90, 30)
        horizontal_layout.addWidget(self.norm_flag_roi)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        self.show_disp_roi = QPushButton('ShowHistROI', self, checkable=True)
        self.show_disp_roi.setChecked(False)
        self.show_disp_roi.clicked[bool].connect(partial(self.ShowROIDispersion))
        #self.show_disp_roi.setFixedSize(200, 60)
        self.show_disp_roi.resize(200, 60)
        horizontal_layout.addWidget(self.show_disp_roi)
        self.extract_1d_roi = QPushButton('Extract ROI')
        self.extract_1d_roi.clicked[bool].connect(partial(self.extract_Roi))
        self.extract_1d_roi.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.extract_1d_roi)
        self.extract_1d_roi_cube = QComboBox()
        self.extract_1d_roi_cube.addItems(['(A)', '(B)'])
        self.extract_1d_roi_cube.setCurrentIndex(0)
        #self.extract_1d_roi_cube.setFixedSize(100, 30)
        self.extract_1d_roi_cube.resize(100, 30)
        horizontal_layout.addWidget(self.extract_1d_roi_cube)

        self.show_roi_1_minus_2 = QPushButton('Show A1/A2', self, checkable=True)
        self.show_roi_1_minus_2.setChecked(False)
        self.show_roi_1_minus_2.clicked[bool]
        #self.show_roi_1_minus_2.setFixedSize(200, 60)
        self.show_roi_1_minus_2.resize(200, 60)
        horizontal_layout.addWidget(self.show_roi_1_minus_2)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        self.set_roi_radius = QPushButton('SetRoi_R', self, checkable=False)
        #self.set_roi_radius.setChecked(False)
        self.set_roi_radius.clicked[bool].connect(self.SetRoi_radius)
        #self.set_roi_radius.setFixedSize(200, 60)
        self.set_roi_radius.resize(200, 60)
        #self.select_roi.clicked[bool].connect(self.ShowROI)
        #self.select_roi.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.set_roi_radius)
        self.roi_radius = QLineEdit()
        self.roi_radius.setText(str(10))
        #self.roi_radius.setFixedSize(60, 30)
        self.roi_radius.resize(60, 30)
        horizontal_layout.addWidget(self.roi_radius)

        self.show_roi_slit = QPushButton('Show Slit', self, checkable=True)
        self.show_roi_slit.setChecked(False)
        self.show_roi_slit.clicked[bool].connect(partial(self.ShowSlit))
        #self.show_roi_slit.setFixedSize(200, 60)
        self.show_roi_slit.resize(200, 60)
        horizontal_layout.addWidget(self.show_roi_slit)

        self.show_gradient = QPushButton('Contours', self, checkable=True)
        self.show_gradient.setChecked(False)
        self.show_gradient.clicked[bool].connect(partial(self.ShowGrad))
        #self.show_gradient.setFixedSize(200, 60)
        self.show_gradient.resize(200, 60)
        horizontal_layout.addWidget(self.show_gradient)
        self.level_value = QLineEdit()
        self.level_value.setText(str(0.68))
        self.level_value.resize(60, 30)
        horizontal_layout.addWidget(self.level_value)
        self.level_type = QComboBox()
        self.level_type.addItems(['LOC', 'TOT'])
        self.level_type.setCurrentIndex(0)
        #self.level_type.setFixedSize(100, 30)
        self.level_type.resize(100, 30)
        horizontal_layout.addWidget(self.level_type)

        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        layout.addLayout(l)

        layout.addStretch(1)
        self.setLayout(layout)




        self.setStyleSheet(open('styles.ini').read())


    def create_ASN_file(self):
        self.parent.Cubes_A.table.create_asn_file()

    def call_build_3dCube(self):
        print('create_asn:')
        self.parent.Cubes_A.table.create_asn_file()
        print('build cube:')
        self.parent.Cubes_A.table.build_3dcube()

    def call_build_cube_12_ch(self):
        self.parent.Cubes_A.table.build_3dcube_12_channels()

    def update_CubeList(self):
        print('update list:')
        self.parent.Cubes_A.table.update_cube_list()

    def extract_Roi(self,mode=None):
        print('extract Roi:')
        cube = self.extract_1d_roi_cube.currentText()
        if cube == '(A)':
            self.parent.Cubes_A.table.extract_roi(cube_name=cube)
        elif cube == '(B)':
            self.parent.Cubes_B.table.extract_roi(cube_name=cube)

    def ShowSlit(self):
        self.parent.Cubes_A.table.show_roi_slit_command(add=self.parent.exp_commands.show_roi_slit.isChecked())

        if 0:
            def CalcMedCube(self):
                self.parent.Cubes_A.table.calc_median_cube(add=self.parent.exp_commands.calc_median_flux.isChecked(),
                                                           radius=int(self.parent.exp_commands.mean_kernel_rad.text()))

                filename = './output/detector3/cash/median_cube.fits'
                self.parent.plot_3dcube_median.show()

    def ShowGrad(self):
        level = float(self.level_value.text())
        if self.level_type.currentText() == 'LOC':
            local = True
        elif self.level_type.currentText() == 'TOT':
            local = False
        self.parent.Cubes_A.table.show_gradient_command(add=self.parent.exp_commands.show_gradient.isChecked(),level=1-level,local=local)


    def SetRoi_radius(self):
        print('')
        roi_type = self.roi_type.currentText()
        rois = {}
        if hasattr(self.parent.plot_3dcubeA,'roi_list'):
            rois['green'] = self.parent.plot_3dcubeA.roi_list[0]
            rois['red'] = self.parent.plot_3dcubeA.roi_list[1]
            cube = self.parent.plot_3dcubeA
        if hasattr(self.parent.plot_3dcubeB,'roi_list'):
            rois['purple'] = self.parent.plot_3dcubeB.roi_list[0]
            rois['yellow'] = self.parent.plot_3dcubeB.roi_list[1]
            cube = self.parent.plot_3dcubeB
        #rois['blue'] = None
        if roi_type in rois.keys():
            roi = rois[roi_type]
            cube.updateRoiRadius(roi=roi)


    def set_DQ_map(self, debug = False):
        print('set_DQ_map, debug:', debug)
        self.parent.Cubes_A.table.set_dq()

    def show_DQ_map(self, debug = False,group=None):
        print('set_DQ_map, debug:', debug)
        self.parent.Cubes_A.table.show_dq()


    def SaturationStep(self, debug = False):
        print('SaturationStep, debug:', debug)
        self.parent.Cubes_A.table.check_saturation()

    def show_SATpixels(self):
        print('Show Saturation pixels')
        self.parent.Cubes_A.table.show_saturation()

    def show_DNUpixels(self):
        print('Show Saturation pixels')
        self.parent.Cubes_A.table.show_DNU()

    def FirstLastStep(self, debug = False):
        self.parent.Cubes_A.table.first_group()
        self.parent.Cubes_A.table.last_group()

    def ResetStep(self, debug=False):
        self.parent.Cubes_A.table.reset_correction()


    def LinearStep(self, debug=False):
        self.parent.Cubes_A.table.linear_correction()



    def ShowROI(self, debug=False):
        roi_type = self.roi_type.currentText()
        rois = {}
        rois['green'] = 'A1'
        rois['red']=  'A2'
        rois['purple']=  'B1'
        rois['yellow'] =  'B2'
        rois['blue'] =  'S1'
        self.parent.Cubes_A.table.show_roi(mode = rois[roi_type])
    def ShowDetectorROI(self):
        roi_type = self.roi_type.currentText()
        rois = {}
        rois['green'] = 'A'
        rois['red'] = 'A'
        rois['purple'] = 'B'
        rois['yellow'] = 'B'
        self.parent.Cubes_A.table.show_detector_roi(add=self.parent.exp_commands.show_roi.isChecked(), mode = rois[roi_type])

    def ShowROIDispersion(self):
        self.parent.Cubes_A.table.show_roi_dispersion(show=self.parent.exp_commands.show_disp_roi.isChecked())

    def CalcMedCube(self):
        calc_median_mode = self.calc_median_mode.currentText()
        self.parent.Cubes_A.table.calc_median_cube(add=self.parent.exp_commands.calc_median_flux.isChecked(),radius=int(self.parent.exp_commands.mean_kernel_rad.text()), method =calc_median_mode)

        filename = './output/detector3/cash/median_cube.fits'
        self.parent.plot_3dcube_median.show()

    def SubtractMedFlux(self, apply=True, first_cube = 'B',debug=True):
        first_cube = self.name_cube_subtracted.currentText()
        median_cube = self.parent.plot_3dcube_median.cube
        if apply:
            if first_cube == 'B':
                tmp = np.array(self.parent.CUBE_B.data.data)
                if self.parent.CUBE_B.flags['subtract_bkgr'] == False:
                    self.parent.CUBE_B.flags['subtract_bkgr'] = True
                    self.parent.CUBE_B.data.data -= median_cube.data.data
                    if debug:
                        fig,ax = plt.subplots(1,2)
                        ax[0].imshow(tmp[10,:,:])
                        ax[1].imshow(self.parent.CUBE_B.data.data[10,:,:])
            elif first_cube == 'A':
                tmp = np.array(self.parent.CUBE_A.data.data)
                if self.parent.CUBE_A.flags['subtract_bkgr'] == False:
                    self.parent.CUBE_A.flags['subtract_bkgr'] = True

                    self.parent.CUBE_A.data.data[~np.isnan(self.parent.CUBE_A.data.data)] -= median_cube.data.data[~np.isnan(self.parent.CUBE_A.data.data)]
                    if debug:
                        fig, ax = plt.subplots(1, 2)
                        ax[0].imshow(tmp[10, :, :])
                        ax[1].imshow(self.parent.CUBE_A.data.data[10, :, :])

    def SaveLocalCube(self):
        first_cube = self.name_cube_subtracted.currentText()
        self.parent.Cubes_A.table.save_local_cube_code(cube_name=first_cube)



class JWST_spec_viewer(QMainWindow):

    def __init__(self):
        super().__init__()
        input_dir = './output/detector2'
        spec3_cachedir = './temp/spec3/'
        self.CUBE_A = detector3(obj_key_name = 'jw02155001001_04102',bkgr_key_name = 'jw02155009001_02101', path = input_dir, output_dir=output_dir)
        self.CUBE_B = detector3(obj_key_name='jw02155001001_04102', bkgr_key_name='jw02155009001_02101', path=input_dir,
                              output_dir=output_dir)

        input2_dir = './output/results/'
        spec2_cachedir = './temp/spec2/'
        output2_dir = './output/detector2/'
        self.stage2 = []
        self.stage2.append(detector2(miri_uncal_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=input2_dir, output_dir=output2_dir,spec2_cachedir=spec2_cachedir))
        self.stage2.append(detector2(miri_uncal_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=input2_dir, output_dir=output2_dir, spec2_cachedir=spec2_cachedir))
        self.stage2.append(detector2(miri_uncal_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=input2_dir, output_dir=output2_dir, spec2_cachedir=spec2_cachedir))
        self.stage2.append(detector2(miri_uncal_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=input2_dir, output_dir=output2_dir, spec2_cachedir=spec2_cachedir))

        #self.H2.readfolder()
        self.initStyles()
        print('me')
        self.initUI()

    def initStyles(self):
        self.setStyleSheet(open('styles.ini').read())

    def initUI(self):
        #dbg = pg.dbg()
        # self.specview sets the type of plot representation

        if 1:# >>> create panel for plotting spectra
            self.plot_3dcubeA = plotCube(self,cube_name='A')
            self.plot_3dcubeB = plotCube(self,cube_name='B')
            self.plot_3dcube_median = plotCube(self, cube_name='Median')
            if 1:
                self.plot_2dimage1 = plotImage(self)
                self.plot_2dimage2 = plotImage(self)
                self.plot_2dimage3 = plotImage(self)
                self.plot_2dimage4 = plotImage(self)
                self.plot_hist1 = plotHist(self)
                self.plot_hist2 = plotHist(self)
                self.plot_hist3 = plotHist(self)
                self.plot_hist4 = plotHist(self)
                self.plot_slit = plotSlit(self)

            self.plot_spectrum = plotSpec(self)
            self.Cubes_A = chooseExpWidget(self, closebutton=False,cube_choice='A')
            self.Cubes_B = chooseExpWidget(self, closebutton=False,cube_choice='B')
            self.exp_pars = expParsWidget(self)
            self.exp_commands = expRunWidget(self)
            # self.plot.setFrameShape(QFrame.StyledPanel)

            self.splitter = QSplitter(Qt.Vertical)
            self.splitter_image = QSplitter(Qt.Horizontal)
            self.splitter_image.addWidget(self.plot_3dcubeA)

            if 1:
                self.spec_image1 = QSplitter(Qt.Vertical)
                self.spec_image12 = QSplitter(Qt.Horizontal)
                self.spec_image12.addWidget(self.plot_2dimage1)
                self.spec_image12.addWidget(self.plot_2dimage2)
                self.spec_image1.addWidget(self.spec_image12)
                self.spec_image34 = QSplitter(Qt.Horizontal)
                self.spec_image34.addWidget(self.plot_2dimage3)
                self.spec_image34.addWidget(self.plot_2dimage4)
                self.spec_image1.addWidget(self.spec_image34)
            if 1:
                self.roi_hist = QSplitter(Qt.Horizontal)
                self.roi_hist.addWidget(self.plot_hist1)
                self.roi_hist.addWidget(self.plot_hist2)
                self.roi_hist.addWidget(self.plot_hist3)
                self.roi_hist.addWidget(self.plot_hist4)
            if 1:
                self.roi_slit = QSplitter(Qt.Horizontal)
                self.roi_slit.addWidget(self.plot_slit)




            #self.splitter_image.addWidget(self.spec_image1)
            self.splitter_image.addWidget(self.plot_3dcubeB)
            #self.splitter_image.addWidget(self.plot_2dimage)
            self.splitter.addWidget(self.splitter_image)

            self.splitter_plot = QSplitter(Qt.Vertical)
            self.splitter_plot.addWidget(self.plot_spectrum)
            #self.splitter_plot.addWidget(self.roi_hist)
            self.splitter.addWidget(self.splitter_plot)

            self.splitter_pars = QSplitter(Qt.Horizontal)
            self.splitter_pars_left_panel = QSplitter(Qt.Vertical)
            self.splitter_pars_left_panel.addWidget(self.exp_pars)
            self.splitter_pars_left_panel.addWidget(self.exp_commands)
            self.splitter_pars.addWidget(self.splitter_pars_left_panel)
            self.splitter_pars_right_panel = QSplitter(Qt.Vertical)
            self.splitter_pars_right_panel.addWidget(self.Cubes_A)
            self.splitter_pars_right_panel.addWidget(self.Cubes_B)
            self.splitter_pars.addWidget(self.splitter_pars_right_panel)
            self.splitter.addWidget(self.splitter_pars)

            self.splitter.setSizes([150, 300,100,100,100,100])
            #self.splitter.setStretchFactor(0, 10)
            #self.splitter.setStretchFactor(1, 10)
            #self.splitter.setStretchFactor(2, 10)
            #self.splitter.setStretchFactor(3, 10)

            self.setCentralWidget(self.splitter)

            # >>> create Menu
            #self.initMenu()

            # create toolbar
            # self.toolbar = self.addToolBar('B-spline')
            # self.toolbar.addAction(Bspline)


            #self.draw()
            self.showMaximized()
            self.show()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex2 = JWST_spec_viewer()
    sys.exit(app.exec_())