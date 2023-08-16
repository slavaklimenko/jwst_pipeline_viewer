import os
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
        self.label.move(25, 50)
        self.label2 = QLabel("World:", self.ui.graphicsView.viewport())
        self.label2.move(25, 15)




    def initstatus(self):
        self.s_status = True
        self.selected_point = None
        self.selected_pixels_saturated = []
        self.selected_pixels_dnu = []
        self.selected_pixels_cr = []
        self.selected_pixels_cr_multi = []



    def add(self, name, add,mode=None):
        if add:
            if mode == None:
                print('add cube name:',name)
                if self.cube_name == 'A':
                    filename = self.parent.Cubes_A.filelist[name]
                    self.parent.CUBE_A.add_cube(cubename=filename)
                    self.parent.CUBE_A.init_cube()
                    channel = self.parent.CUBE_A.data.channel
                    band = self.parent.CUBE_A.data.band
                    self.data = self.parent.CUBE_A.data.data
                elif self.cube_name == 'B':
                    filename = self.parent.Cubes_B.filelist[name]
                    self.parent.CUBE_B.add_cube(cubename=filename)
                    self.parent.CUBE_B.init_cube()
                    channel = self.parent.CUBE_B.data.channel
                    band = self.parent.CUBE_B.data.band
                    self.data = self.parent.CUBE_B.data.data

                t,x, y = (0,2, 1)
                self.setImage(self.data, axes={'t': t, 'x': x, 'y': y, 'c': None}, levels=[-100,800]) #autoRange=True,
                self.vb.hoverEvent = self.imageHoverEvent
                hist = self.getHistogramWidget()
                hist.setHistogramRange(mn=-200, mx=1000)
                if 1:
                    self.roi_list = []
                    if self.cube_name == 'A':
                        roi_colors = ['lightgreen','red']
                    elif self.cube_name == 'B':
                        roi_colors = ['purple','yellow']
                    self.roi_list.append(pg.EllipseROI([15, 15], [10, 10], pen=pg.mkPen(roi_colors[0], width=3)))
                    self.roi_list.append(pg.CircleROI([30, 30], [10, 10], pen=pg.mkPen(roi_colors[1], width=2), movable=True, resizable=True))
                    #self.roi_list = rois
                    #self.roi_mask = {}
                    #rois.append(pg.EllipseROI([20, 20], [12, 12], pen=(9, 2)))

                    def updateRoi(roi):
                        if roi is None:
                            return
                        arr1 = roi.getArrayRegion(arr=self.data, img=self.imageItem , axes=(2,1))
                        if 1:   #set_roi_mask
                            mask = np.array(np.zeros((self.data.shape[1],self.data.shape[2])), dtype='bool')
                            rows,cols = self.data.shape[1],self.data.shape[2]
                            m = np.mgrid[:rows, :cols]
                            possx = m[0, :, :]  # make the x pos array
                            possy = m[1, :, :]  # make the y pos array
                            possx.shape = rows,cols
                            possy.shape = rows,cols
                            mpossx = roi.getArrayRegion(possx, self.imageItem).astype(int)
                            mpossx = mpossx[np.nonzero(mpossx)]  # get the x pos from ROI
                            mpossy = roi.getArrayRegion(possy, self.imageItem).astype(int)
                            mpossy = mpossy[np.nonzero(mpossy)]  # get the y pos from ROI
                            mask[mpossy, mpossx] = True #self.data[0,mpossx, mpossy]>0
                            roi.roi_mask = mask
                        updateRoiPlot(roi, arr1)



                    def updateRoiPlot(roi, data_roi=None):
                        if data_roi is None:
                            data_roi = roi.getArrayRegion(arr=self.data, img=self.imageItem , axes=(1,2))
                            #data = roi.getArrayRegion(im1.image, img=im1)
                        if data_roi is not None:
                            d= data_roi.mean(axis=(1,2))
                            roi.curve.setData(data_roi.mean(axis=(1,2)))
                            #self.parent.plot_spectrum.plot_spec(data=d,add=False)
                            #self.parent.plot_spectrum.plot_spec(data=d,pen=roi.pen)

                        if np.sum(roi.roi_mask)>0:
                            num_pixels=np.sum(roi.roi_mask)
                            roi_selected_flux=np.zeros(self.data.shape[0])
                            for i in range(self.data.shape[0]):
                                d = self.data[i,:,:]
                                d = d[roi.roi_mask]
                                roi_selected_flux[i] = np.nansum(d)/num_pixels
                            if roi == self.roi_list[0]:
                                if self.cube_name == 'A':
                                    self.parent.plot_spectrum.plot_specA1(data=roi_selected_flux, add=False)
                                    self.parent.plot_spectrum.plot_specA1(data=roi_selected_flux, pen=roi.pen)
                                if self.cube_name == 'B':
                                    self.parent.plot_spectrum.plot_specB1(data=roi_selected_flux, add=False)
                                    self.parent.plot_spectrum.plot_specB1(data=roi_selected_flux, pen=roi.pen)
                            elif roi == self.roi_list[1]:
                                if self.cube_name == 'A':
                                    self.parent.plot_spectrum.plot_specA2(data=roi_selected_flux, add=False)
                                    self.parent.plot_spectrum.plot_specA2(data=roi_selected_flux,pen=roi.pen)
                                if self.cube_name == 'B':
                                    self.parent.plot_spectrum.plot_specB2(data=roi_selected_flux, add=False)
                                    self.parent.plot_spectrum.plot_specB2(data=roi_selected_flux, pen=roi.pen)

                    ## Add each ROI to the scene and link its data to a plot curve with the same color
                    for r in self.roi_list:
                        self.vb.addItem(r)
                        c = self.roi.plot(pen=r.pen)
                        r.curve = c
                        r.sigRegionChanged.connect(updateRoi)

                    #self.updateRoi = updateRoi(self.roi_list[0])

                    def updatePlotSpec():
                        data_roi = self.roi.getArrayRegion(arr=self.data, img=self.imageItem, axes=(1, 2))
                        if data_roi is not None:
                            d= data_roi.mean(axis=(1,2))
                            self.roi.curve.setData(data_roi.mean(axis=(1,2)))
                            self.parent.plot_spectrum.plot_spec(data=d,add=False)
                            self.parent.plot_spectrum.plot_spec(data=d)


                    #self.timeLine.sigPositionChanged.connect(updatePlotSpec)

        else:
            try:
                #self.roi.clearPoints()
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
        (self.t,time) = self.timeIndex(self.timeLine)
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
                lam_world,x_world,y_world = self.parent.CUBE_A.conv_world_coord(t=self.lam, x=self.x, y=self.y)
            elif self.cube_name == 'B':
                lam_world,x_world,y_world = self.parent.CUBE_B.conv_world_coord(t=self.lam, x=self.x, y=self.y)
        if row < self.data.shape[0] and row >= 0 and col < self.data.shape[1] and col > 0:
            val = self.data[self.t,row,col]
        else:
            val = -999
        text = "pixel: (row=%d, col=%d, l=%d), val: %.2f MJy/sr" % ( row, col,time,val)
        print(text)

        text_world_coord = "world: (ra=%.5f, dec=%.5f, l=%.4f)" % (x_world,y_world,lam_world)
        print(text_world_coord)
        #cal_world_to_detector = wcs.get_transform('world', 'detector')
        #print('world->detector', cal_world_to_detector(x_world,y_world, lam_world))

        self.label.setText(text)
        self.label.resize(600, 40)
        self.label2.setText(text_world_coord)
        self.label2.resize(600, 40)


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

            if self.parent.CUBE_A.data.data is not None and 0:
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

   #     if not event.isAutoRepeat():
   #         if event.key() == Qt.Key_S:
   #             self.s_status = True

    #def keyReleaseEvent(self, event):
    #    super(plotGrid, self).keyReleaseEvent(event)
    #    key = event.key()
#        if not event.isAutoRepeat():
#            if event.key() == Qt.Key_S:
#                self.s_status = True
#                self.parent.plot_exc.add_temp([], add=False)

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
                nexp[1] = int(self.parent.exp_pars.nEXP1.text())
                nexp[2] = int(self.parent.exp_pars.nEXP2.text())
                nexp[3] = int(self.parent.exp_pars.nEXP3.text())
                nexp[4] = int(self.parent.exp_pars.nEXP4.text())
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
                self.parent.stage2[Nscreen-1].read_step_results(step_name='FluxCalib')
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




    def plot_specA1(self, data=None, add=True,pen=pg.mkPen(color='white', style=Qt.DashLine, width=1)):
        if add:
            wavel = self.parent.CUBE_A.data.wavelength
            if np.size(data) == np.size(wavel):
                pen = pen
                self.plot_lineA1 = pg.PlotCurveItem(wavel, data,pen=pen)
                self.vb.addItem(self.plot_lineA1)
                pen = pg.mkPen(color='darkgray', style=Qt.DashLine, width=1)
                #self.zero_level = pg.PlotCurveItem([wavel[0]-2, wavel[-1] + 2], [0, 0], pen=pen)
                self.zero_level = pg.PlotCurveItem([0, 30], [0, 0], pen=pen)
                self.vb.addItem(self.zero_level)

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


                self.vb.autoRange()
                #self.vb.setLimits(xMin=wavel[0]*0.98, xMax=wavel[-1]*1.02)


        else:
            try:
                self.vb.removeItem(self.plot_lineA1)

                self.vb.removeItem(self.zero_level)
                self.vb.removeItem(self.lr)



            except:
                pass

    def plot_specA2(self, data=None, add=True,pen=pg.mkPen(color='white', style=Qt.DashLine, width=1)): #pg.mkPen(color='gray', style=Qt.DashLine, width=3)
        if add:
            wavel = self.parent.CUBE_A.data.wavelength
            if np.size(data) == np.size(wavel):

                pen = pen
                self.plot_lineA2 = pg.PlotCurveItem(wavel, data,pen=pen)
                self.vb.addItem(self.plot_lineA2)

                self.vb.autoRange()
                #self.vb.setLimits(xMin=wavel[0]*0.98, xMax=wavel[-1]*1.02)
        else:
            try:
                self.vb.removeItem(self.plot_lineA2)

            except:
                pass

    def plot_specB1(self, data=None, add=True, pen=pg.mkPen(color='white', style=Qt.DashLine,
                                                            width=1)):  # pg.mkPen(color='gray', style=Qt.DashLine, width=3)
        if add:
            wavel = self.parent.CUBE_B.data.wavelength
            if np.size(data) == np.size(wavel):
                pen = pen
                self.plot_lineB1 = pg.PlotCurveItem(wavel, data, pen=pen)
                self.vb.addItem(self.plot_lineB1)

                self.vb.autoRange()
                #self.vb.setLimits(xMin=wavel[0] * 0.98, xMax=wavel[-1] * 1.02)
        else:
            try:
                self.vb.removeItem(self.plot_lineB1)

            except:
                pass

    def plot_specB2(self, data=None, add=True, pen=pg.mkPen(color='white', style=Qt.DashLine,
                                                            width=1)):  # pg.mkPen(color='gray', style=Qt.DashLine, width=3)
        if add:
            wavel = self.parent.CUBE_B.data.wavelength
            if np.size(data) == np.size(wavel):
                pen = pen
                self.plot_lineB2 = pg.PlotCurveItem(wavel, data, pen=pen)
                self.vb.addItem(self.plot_lineB2)

                self.vb.autoRange()
                #self.vb.setLimits(xMin=wavel[0] * 0.98, xMax=wavel[-1] * 1.02)
        else:
            try:
                self.vb.removeItem(self.plot_lineB2)

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
        print('ASN file params:', source, channel, band)

        input_dir = self.parent.parent.CUBE_A.path
        files = self.parent.parent.CUBE_A.create_association(input_dir=input_dir,source = source,channel = channel, band =band)
        print('Asn files for cube building:', files)
        #self.parent.parent.CUBE.writel3asn(files=files,asnfile=source+'_'+channel+'_'+band+)
        #sort_calfiles(files):

    def build_3dcube(self):
        master_bkgr_flag = int(self.parent.parent.exp_pars.master_bkgr_flag.currentIndex())
        master_res_bkgr_flag = int(self.parent.parent.exp_pars.master_res_bkgr_flag.currentIndex())
        master_outlier_flag = int(self.parent.parent.exp_pars.master_outlier_flag.currentIndex())
        master_extract1d_flag = int(self.parent.parent.exp_pars.master_extract1d_flag.currentIndex())
        channel = self.parent.parent.exp_pars.asn_channel.currentText()
        asn_file = self.parent.parent.CUBE_A.tmp_asn_file
        print('Built cube from asn files:')
        self.parent.parent.CUBE_A.build_cube(input_file=asn_file, channel = channel, master_bkgr_flag = master_bkgr_flag ,
                                           master_res_bkgr_flag = master_res_bkgr_flag, master_outlier_flag = master_outlier_flag,
                                           master_extract1d_flag = master_extract1d_flag)


    def extract_roi(self, mode = 'pixels'):
        cube = self.parent.parent.plot_3dcubeA
        data = self.parent.parent.CUBE_A.data
        wavel = self.parent.parent.CUBE_A.data.wavelength
        name = self.parent.parent.CUBE_A.cubename.split('/')[-1].split('.')[0]
        roi_name = ['sci','bkgr']
        if mode == 'pixels':
            for ir,roi in enumerate(cube.roi_list):
                if np.sum(roi.roi_mask) > 0:
                    num_pixels = np.sum(roi.roi_mask)
                    roi_selected_flux = np.zeros(cube.data.shape[0])
                    roi_selected_flux_error = np.zeros(cube.data.shape[0])
                    for i in range(cube.data.shape[0]):
                        d = data.data[i, :, :]
                        d = d[roi.roi_mask]
                        derr = data.err[i, :, :]
                        derr = derr[roi.roi_mask]
                        roi_selected_flux[i] = np.sum(d/derr**2) / np.sum(1/derr**2)
                        roi_selected_flux_error[i] = 1/np.sum(1/derr**2)
                    filename = './output/detector3/roi_spectra/'+name+roi_name[ir]+'.spec1d'
                    with open(filename, 'w') as fout:
                        for x,y,e in zip(wavel,roi_selected_flux,roi_selected_flux_error):
                            fout.write('%.4e %.4e %.4e \n' %(x,y,e))
                    fout.close()





    def show_roi(self):
        print('show_ROI pixels:')
        if 1:
            if 1:
                roi_mask = self.parent.parent.plot_3dcubeA.roi_list[0].roi_mask
                x,y = np.where(roi_mask>0)
                (timeind, time) = self.parent.parent.plot_3dcubeA.timeIndex(self.parent.parent.plot_3dcubeA.timeLine)
                lam = time+2
                lam_array = np.zeros_like(x) + lam
                print('roi pixels at lambda=',lam)
                #for e,k in zip(x,y):
                #    print(e,k)
                #self.parent.CUBE.data.data
            if 0:
                wcs1 = self.parent.parent.CUBE.data.wcs
                x_world = wcs1['CRVAL1'] - (x - wcs1['CRPIX1']) * wcs1['CDELT1']
                y_world = wcs1['CRVAL2'] + (y - wcs1['CRPIX2']) * wcs1['CDELT2']
                lam_world = wcs1['CRVAL3'] + (lam - wcs1['CRPIX3']) * wcs1['CDELT3']
                lam_w_axis = np.zeros_like(x_world)
                lam_w_axis[:] = lam_world
                print('roi world pixels:')
                for e, k in zip(x_world, y_world):
                    print(e, k)
            else:
                lam_world, x_world,y_world = self.parent.parent.CUBE_A.conv_world_coord(t=lam_array, x=y, y=x)

            if 1:
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


            if self.flags['show_ROI'] == False:
                self.flags['show_ROI'] = True
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

                self.parent.parent.plot_3dcubeA.selectPixels(add=self.flags['show_ROI'], x=y, y=x,color='m',type='cr_multi')

            else:
                self.flags['show_ROI'] = False
                self.parent.parent.plot_2dimage1.selectPixels(add=False, type='cr_multi')
                self.parent.parent.plot_2dimage2.selectPixels(add=False, type='cr_multi')
                self.parent.parent.plot_2dimage3.selectPixels(add=False, type='cr_multi')
                self.parent.parent.plot_2dimage4.selectPixels(add=False, type='cr_multi')
                self.parent.parent.plot_3dcubeA.selectPixels(add=False, type='cr_multi')











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
        self.table.setdata(data)


        self.buttons = {}
        for i, d in enumerate(data):
            wdg = QWidget()
            l = QVBoxLayout()
            l.addSpacing(3)
            button = QPushButton(d[0].split('_uncal')[0], self, checkable=True)
            button.setFixedSize(600, 30)
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
            self.okButton.setFixedSize(110, 30)
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
            if 0:
                self.parent.plot_2dimage1.add(name, self.buttons[name].isChecked(), Nscreen=1)
                self.parent.plot_2dimage2.add(name, self.buttons[name].isChecked(), Nscreen=2)
                self.parent.plot_2dimage3.add(name, self.buttons[name].isChecked(), Nscreen=3)
                self.parent.plot_2dimage4.add(name, self.buttons[name].isChecked(), Nscreen=4)
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


        l = QHBoxLayout(self)
        l.addWidget(QLabel('Exp1:'))
        self.nEXP1 = QLineEdit()
        self.nEXP1.setText(str(0))
        self.nEXP1.setFixedSize(60, 30)
        l.addWidget(self.nEXP1)

        l.addWidget(QLabel('Exp2:'))
        self.nEXP2 = QLineEdit()
        self.nEXP2.setText(str(1))
        self.nEXP2.setFixedSize(60, 30)
        l.addWidget(self.nEXP2)

        l.addWidget(QLabel('Exp3:'))
        self.nEXP3 = QLineEdit()
        self.nEXP3.setText(str(2))
        self.nEXP3.setFixedSize(60, 30)
        l.addWidget(self.nEXP3)


        l.addWidget(QLabel('Exp4:'))
        self.nEXP4 = QLineEdit()
        self.nEXP4.setText(str(3))
        self.nEXP4.setFixedSize(60, 30)
        l.addWidget(self.nEXP4)

        l.addStretch(1)
        layout.addLayout(l)

        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('MasterBkgr:'))
        self.master_bkgr_flag = QComboBox()
        self.master_bkgr_flag.addItems(['No', 'Yes'])
        self.master_bkgr_flag.setCurrentIndex(0)
        self.master_bkgr_flag.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.master_bkgr_flag)

        horizontal_layout.addWidget(QLabel('ResBkgrMatch:'))
        self.master_res_bkgr_flag = QComboBox()
        self.master_res_bkgr_flag.addItems(['No', 'Yes'])
        self.master_res_bkgr_flag.setCurrentIndex(1)
        self.master_res_bkgr_flag.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.master_res_bkgr_flag)

        horizontal_layout.addWidget(QLabel('OutlierDet:'))
        self.master_outlier_flag = QComboBox()
        self.master_outlier_flag.addItems(['No', 'Yes'])
        self.master_outlier_flag.setCurrentIndex(1)
        self.master_outlier_flag.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.master_outlier_flag)

        horizontal_layout.addWidget(QLabel('Extract1D:'))
        self.master_extract1d_flag = QComboBox()
        self.master_extract1d_flag.addItems(['No', 'Yes'])
        self.master_extract1d_flag.setCurrentIndex(1)
        self.master_extract1d_flag.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.master_extract1d_flag)

        horizontal_layout.addStretch(1)
        layout.addLayout(horizontal_layout)


        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('Source:'))
        self.asn_source = QComboBox()
        self.asn_source.addItems(['Object', 'Background'])
        self.asn_source.setCurrentIndex(0)
        #p = self.asn_source.currentText()
        #print(self.asn_source.itemData[0])
        self.asn_source.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.asn_source)

        horizontal_layout.addWidget(QLabel('Channel:'))
        self.asn_channel = QComboBox()
        self.asn_channel.addItems(['1', '2','3','4'])
        self.asn_channel.setCurrentIndex(0)
        self.asn_channel.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.asn_channel)

        horizontal_layout.addWidget(QLabel('Band:'))
        self.asn_band = QComboBox()
        self.asn_band.addItems(['SHORT(A)', 'MEDIUM(B)', 'LONG(C)'])
        self.asn_band.setCurrentIndex(0)
        self.asn_band.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.asn_band)

        horizontal_layout.addStretch(1)
        layout.addLayout(horizontal_layout)



        layout.addStretch(1)
        self.setLayout(layout)

        self.setStyleSheet(open('styles.ini').read())

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
        self.create_asn = QPushButton('Create AssFile')
        self.create_asn.clicked[bool].connect(partial(self.create_ASN_file))
        self.create_asn.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.create_asn)
        self.build_cube = QPushButton('Build cube')
        self.build_cube.clicked[bool].connect(partial(self.call_build_3dCube))
        self.build_cube.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.build_cube)
        self.select_roi = QPushButton('Select ROI')
        self.select_roi.clicked[bool].connect(partial(self.ShowROI, 'slope'))
        self.select_roi.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.select_roi)

        self.extract_1d_roi = QPushButton('Extract ROI')
        self.extract_1d_roi.clicked[bool].connect(partial(self.extract_Roi, 'slope'))
        self.extract_1d_roi.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.extract_1d_roi)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        self.update_cubes_list = QPushButton('Update_List')
        self.update_cubes_list.clicked[bool].connect(partial(self.update_CubeList))
        self.update_cubes_list.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.update_cubes_list)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        layout.addLayout(l)
        layout.addStretch(1)
        self.setLayout(layout)

        self.setStyleSheet(open('styles.ini').read())


    def create_ASN_file(self):
        print('create_asn:')
        self.parent.Cubes_A.table.create_asn_file()

    def call_build_3dCube(self):
        print('build cube:')
        self.parent.Cubes_A.table.build_3dcube()

    def update_CubeList(self):
        print('update list:')
        self.parent.Cubes_A.table.update_cube_list()

    def extract_Roi(self,mode=None):
        print('extract Roi:')
        self.parent.Cubes_A.table.extract_roi()


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
        self.parent.Cubes_A.table.show_roi()
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
        self.stage2.append(detector2(miri_uncal_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=input2_dir,
                                     output_dir=output2_dir, spec2_cachedir=spec2_cachedir))
        self.stage2.append(detector2(miri_uncal_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=input2_dir,
                                     output_dir=output2_dir, spec2_cachedir=spec2_cachedir))
        self.stage2.append(detector2(miri_uncal_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=input2_dir,
                                     output_dir=output2_dir, spec2_cachedir=spec2_cachedir))

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
            self.plot_2dimage1 = plotImage(self)
            self.plot_2dimage2 = plotImage(self)
            self.plot_2dimage3 = plotImage(self)
            self.plot_2dimage4 = plotImage(self)
            self.plot_spectrum = plotSpec(self)
            self.Cubes_A = chooseExpWidget(self, closebutton=False,cube_choice='A')
            self.Cubes_B = chooseExpWidget(self, closebutton=False,cube_choice='B')
            self.exp_pars = expParsWidget(self)
            self.exp_commands = expRunWidget(self)
            # self.plot.setFrameShape(QFrame.StyledPanel)

            self.splitter = QSplitter(Qt.Vertical)
            self.splitter_image = QSplitter(Qt.Horizontal)
            self.splitter_image.addWidget(self.plot_3dcubeA)

            self.spec_image1 = QSplitter(Qt.Vertical)
            self.spec_image12 = QSplitter(Qt.Horizontal)
            self.spec_image12.addWidget(self.plot_2dimage1)
            self.spec_image12.addWidget(self.plot_2dimage2)
            self.spec_image1.addWidget(self.spec_image12)
            self.spec_image34 = QSplitter(Qt.Horizontal)
            self.spec_image34.addWidget(self.plot_2dimage3)
            self.spec_image34.addWidget(self.plot_2dimage4)
            self.spec_image1.addWidget(self.spec_image34)

            self.splitter_image.addWidget(self.spec_image1)
            self.splitter_image.addWidget(self.plot_3dcubeB)
            #self.splitter_image.addWidget(self.plot_2dimage)
            self.splitter.addWidget(self.splitter_image)

            self.splitter_plot = QSplitter(Qt.Vertical)
            self.splitter_plot.addWidget(self.plot_spectrum)
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