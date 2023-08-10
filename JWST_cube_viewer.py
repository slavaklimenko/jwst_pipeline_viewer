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
                             QRadioButton, QButtonGroup, QComboBox, QTableView, QLineEdit)
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
        self.roi = self.getRoiPlot()

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


    def add(self, name, add,mode=None):
        if add:
            if mode == None:
                nint = int(self.parent.exp_pars.nINT.text())
                ngroup =  int(self.parent.exp_pars.nGROUP.text())
                # number of group in integration
                print('add name:',name)
                filename = self.parent.Cubes.filelist[name]
                self.parent.CUBE.add_cube(cubename=filename)
                self.parent.CUBE.init_cube()
                channel = self.parent.CUBE.data.channel
                band = self.parent.CUBE.data.band
                data = self.parent.CUBE.data.data
                t,x, y = (0,1, 2)
                self.setImage(data, axes={'t': t, 'x': x, 'y': y, 'c': None},levels=[-500,1500]) #autoRange=True,
                self.vb.hoverEvent = self.imageHoverEvent
                if 1:
                    rois = []
                    rois.append(pg.EllipseROI([15, 15], [10, 10], pen=(3, 9)))

                    def updateRoi(roi):
                        if roi is None:
                            return
                        #arr1 = roi.getArrayRegion(self.vb.image, img=self.vb)
                        arr1 = roi.getArrayRegion(arr=data, img=self.imageItem , axes=(1,2))
                        #self.setImage(arr1)
                        updateRoiPlot(roi, arr1)

                    def updateRoiPlot(roi, data_roi=None):
                        if data_roi is None:
                            data_roi = roi.getArrayRegion(arr=data, img=self.imageItem , axes=(1,2))
                            #data = roi.getArrayRegion(im1.image, img=im1)
                        if data_roi is not None:
                            d= data_roi.mean(axis=(1,2))
                            roi.curve.setData(data_roi.mean(axis=(1,2)))
                            self.parent.plot_spectrum.plot_spec(data=d,add=False)
                            self.parent.plot_spectrum.plot_spec(data=d)

                    ## Add each ROI to the scene and link its data to a plot curve with the same color
                    for r in rois:
                        self.vb.addItem(r)
                        c = self.roi.plot(pen=r.pen)
                        r.curve = c
                        r.sigRegionChanged.connect(updateRoi)

                #self.vb.addItem(self.view[name])
                #self.legend.addItem(self.view[name], name)
                #self.redraw()
                #self.parent.CUBE.get_readnoise()
            elif mode == 'image':
                nint = int(self.parent.exp_pars.nINT.text())
                ngroup = int(self.parent.exp_pars.nGROUP.text())
                # number of group in integration
                nintmax = self.parent.CUBE.data.data.shape[0]
                ngrmax = self.parent.CUBE.data.data.shape[1]
                if nint > nintmax - 1:
                    print('Set correct Integration number, <=', nintmax - 1)
                    data = self.parent.CUBE.data.data[nintmax - 1][ngrmax - 1]
                    self.setImage(data * 0, autoRange=True, levels=[2000, 6000])
                elif ngroup > ngrmax - 1:
                    print('Set correct Group number, <=', ngrmax - 1)
                    data = self.parent.CUBE.data.data[nintmax - 1][ngrmax - 1]
                    self.setImage(data * 0, autoRange=True, levels=[2000, 6000])
                else:
                    data = self.parent.CUBE.data.data[nint][ngroup]
                x, y = (1, 0)
                self.setImage(data, autoRange=True, levels=[2000, 6000], axes={'t': None, 'x': x, 'y': y, 'c': None})
                self.vb.hoverEvent = self.imageHoverEvent
            elif mode == 'slope':
                if self.parent.Cubes.table.flags['read_fit_slopes']:
                    data = self.parent.CUBE.ramp_fit[0].data
                    x, y = (1, 0)
                    self.setImage(data, autoRange=True, levels=[-1, 5],
                                  axes={'t': None, 'x': x, 'y': y, 'c': None})
                    self.vb.hoverEvent = self.imageHoverEvent
                else:
                    print('There is no SLOPE ARRAY')
                    data = (self.parent.CUBE.data.data[0][0]).copy()*0
                    x, y = (1, 0)
                    self.setImage(data, autoRange=True, levels=[-1, 5],
                                  axes={'t': None, 'x': x, 'y': y, 'c': None})
                    self.vb.hoverEvent = self.imageHoverEvent


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
            #pos = event.pos()
            #i, j = pos.y(), pos.x()
            #print('exit')
            #self.label.setText('exit')
            return
        pos = event.pos()
        i, j = pos.y(), pos.x()
        self.mousePoint = self.vb.vb.mapSceneToView(event.pos())
        # self.mousePoint = self.vb.mapRectToView(event.pos())
        self.x, self.y = self.mousePoint.x(), self.mousePoint.y()
        row, col = int(self.x), int(self.y)
        wcs = self.parent.stage2.data.meta.wcs
        cal_detector_to_world = wcs.get_transform('detector', 'alpha_beta')
        text = "pixel: (%d, %d)" % ( row, col)
        #print(text)
        self.label.setText(text)
        self.label.resize(300, 40)


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

            if self.parent.CUBE.data.data is not None and 0:
                col,row =int(self.x),int(self.y)
                show_fit = self.parent.s3d.Cubes.table.flags['read_fit_slopes']
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
        self.roi = self.getRoiPlot()

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


    def add(self, name, add,mode=None):
        if add:
            if mode == None:
                nint = int(self.parent.exp_pars.nINT.text())
                ngroup =  int(self.parent.exp_pars.nGROUP.text())
                nexp = int(self.parent.exp_pars.nEXP.text())
                # number of group in integration
                print('add name:',name)
                asn_filename = self.parent.Cubes.associtations_list[name].split('/')[-1]
                self.parent.CUBE.load3asn(asn_filename)
                rate_filename = self.parent.CUBE.ratefiles[nexp]
                f = rate_filename['expname']
                #self.parent.CUBE.add_cube(cubename=filename)
                #self.parent.CUBE.init_cube()
                path = self.parent.stage2.path
                exp_name = f.split('/')[-1].replace('cal.fits', '')
                output2_dir, spec2_cachedir = self.parent.stage2.output_dir, self.parent.stage2.spec2_cachedir
                self.parent.stage2.__init__(miri_uncal_file=exp_name, path=path, output_dir=output2_dir,
                                                   spec2_cachedir=spec2_cachedir)
                f = rate_filename['expname']
                self.parent.stage2.read_step_results(step_name='FluxCalib')
                print('Show image of' + self.parent.stage2.name)
                data = self.parent.stage2.data.data
                zmin, zmax = np.nanquantile(data.flatten(), 0.01), np.nanquantile(data.flatten(), 0.99)
                x, y = (1, 0)
                self.data = data
                self.setImage(data, autoRange=True,  # levels=[-1, 5],
                              axes={'t': None, 'x': x, 'y': y, 'c': None}, levels=[zmin, zmax])
                self.vb.hoverEvent = self.imageHoverEvent


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
        text = "pixel: (%d, %d), val: None" % (col, row)
        if row < self.data.shape[0] and row >= 0 and col < self.data.shape[1] and col > 0:
            val = self.data[row, col]
            text = "pixel: (%d, %d), val: %.2f" % (col, row, val)
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
        if self.s_status:
            # name = self.parent.name

            if self.parent.EXP.data.data is not None:
                col, row = int(self.x), int(self.y)
                show_fit = self.parent.Exposures.table.flags['read_fit_slopes']
                self.parent.plot_pixel.plot_profile(row=row, col=col, add=False, show_fit=show_fit)
                self.parent.plot_pixel.plot_profile(row=row, col=col, show_fit=show_fit)
                if self.parent.Exposures.table.current_pipeline_stage == 'stage1':
                    self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col, add=False)
                    self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col)
                elif self.parent.Exposures.table.current_pipeline_stage == 'stage2':
                    self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col, add=False)
                    self.parent.plot_pixel_diffs.plot_verical_stripe(row=row, col=col, add=False)
                    self.parent.plot_pixel_diffs.plot_verical_stripe(row=row, col=col)

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
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'Diffs/Sigma', 'bottom': 'Groups'})
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




    def plot_spec(self, data=None, add=True):
        if add:
            wavel = self.parent.CUBE.data.wavelength
            if np.size(data) == np.size(wavel):

                self.plot_line = pg.PlotCurveItem(wavel, data)
                #self.plot_scatters = pg.ScatterPlotItem(x[mask], diffs2err[mask],symbol = 'o',size=20,brush='r')
                #self.plot_ebars = pg.ErrorBarItem(x=np.asarray(x)[mask], y=diffs2err[mask], top=1, bottom=1, beam=0.5)
                #mask_CR = dq_map == dqflags.pixel['JUMP_DET']
                #if np.sum(mask_CR>0):
                #self.plot_scatters_CRs = pg.ScatterPlotItem(x[mask_CR], diffs2err[mask_CR], symbol='o', size=20, brush='b')

                self.vb.addItem(self.plot_line)
                #self.vb.addItem(self.plot_scatters)
                #self.vb.addItem(self.plot_scatters_CRs)
                #self.vb.addItem(self.plot_ebars)
                #self.vb.setLimits(xMin=wavel[0]*0.98, xMax=wavel[-1]*1.01)
                pen = pg.mkPen(color='darkgray', style=Qt.DashLine, width=1)
                self.zero_level = pg.PlotCurveItem([wavel[0]-2, wavel[-1] + 2], [0, 0], pen=pen)
                self.vb.addItem(self.zero_level)

                #pen = pg.mkPen(color='gray', style=Qt.DashLine, width=3)
                #self.median_diff_level = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1], median_diffs/error[1]], pen=pen)
                #self.vb.addItem(self.median_diff_level)

                #sat_limit = float(self.parent.exp_pars.CRlimit.text())
                #print('pixel diffs:', (diffs-median_diffs)/error)
                #print('')
                #if np.sum((diffs-median_diffs)/error > 5):
                #    self.parent.show_cr_limit = True
                #    pen = pg.mkPen(color='b', style=Qt.DashLine, width=3)
                #    self.crp_limit = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1]+sat_limit, median_diffs/error[1]+sat_limit], pen=pen)
                #    self.crm_limit = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1]-sat_limit, median_diffs/error[1]-sat_limit], pen=pen)
                #    self.vb.addItem(self.crp_limit)
                #    self.vb.addItem(self.crm_limit)
                #    pen = pg.mkPen(color='g', style=Qt.DashLine, width=3)
                #    self.crp_limit_3cr = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1]+5,median_diffs/error[1]+5], pen=pen)
                #    self.vb.addItem(self.crp_limit_3cr)
                #    pen = pg.mkPen(color='r', style=Qt.DashLine, width=3)
                #    self.crp_limit_2cr = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1]+6,median_diffs/error[1]+6], pen=pen)
                #    self.vb.addItem(self.crp_limit_2cr)

                self.vb.autoRange()
                self.vb.setLimits(xMin=wavel[0]*0.98, xMax=wavel[-1]*1.02)

                #text = 'Pixel: '
                #if col is not None:
                #    text += ' {0:.0f} {1:.0f}'.format(col, row)
                #self.legend_model.addItem(self.plot_line, text)
        else:
            try:
                self.vb.removeItem(self.plot_line)
                #self.vb.removeItem(self.plot_scatters)
                #self.vb.removeItem(self.plot_scatters_CRs)
                #self.vb.removeItem(self.plot_ebars)
                #self.vb.removeItem(self.median_diff_level)
                self.vb.removeItem(self.zero_level)

                #self.legend_model.removeItem(self.plot_line)
                #if self.parent.show_cr_limit == True:
                #    self.vb.removeItem(self.crp_limit)
                #    self.vb.removeItem(self.crm_limit)
                #    self.parent.show_cr_limit = False
                #    self.vb.removeItem(self.crp_limit_3cr)
                #    self.vb.removeItem(self.crp_limit_2cr)

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

    def show_image(self,mode = 'image'):
        if mode == 'image':
            print('show current image, name',self.parent.parent.current_name)
            self.parent.parent.plot_image.add(name = self.parent.parent.current_name,add = False)
            self.parent.parent.plot_image.add(name=self.parent.parent.current_name,add=True,mode='image')
        if mode == 'slope':
            print('show slope, name', self.parent.parent.current_name)
            #self.parent.parent.plot_image.add(name=self.parent.parent.current_name, add=False)
            self.parent.parent.plot_image.add(name=self.parent.parent.current_name, add=True,mode='slope')

    def set_dq(self):
        print('set data quality map')
        self.parent.parent.CUBE.dq_init_step()
        self.parent.parent.CUBE.dq_init_step()

    def show_dq(self):
        print('set data quality map')
        flag = self.parent.parent.exp_commands.dq_categories.currentText()
        self.parent.parent.CUBE.show_dq(flag=flag)

    def check_saturation(self):
        if self.flags['saturation_step'] == False:
            flag = self.parent.parent.exp_pars.addneighbors.isChecked()
            debug = self.parent.parent.exp_pars.debug.isChecked()
            print('Run Saturation step: add_neighbors=',flag, ' show_debug = ',debug)
            self.parent.parent.CUBE.saturation_step(n_pix_grow_sat=flag,debug=debug)
            self.flags['saturation_step'] = True
            print('SATURATION STEP: DONE')
        else:
            flag = self.parent.parent.exp_pars.addneighbors.isChecked()
            debug = self.parent.parent.exp_pars.debug.isChecked()
            print('Run second saturation step: add_neighbors=', flag, ' show_debug = ', debug)
            self.parent.parent.CUBE.reset_dq(flagname='SATURATED')
            self.parent.parent.CUBE.saturation_step(n_pix_grow_sat=flag, debug=debug)
            self.flags['saturation_step'] = True
            print('SATURATION STEP: DONE')
        if debug:
            plt.show()

    def show_saturation(self):
        if self.flags['saturation_step'] == True:
            if self.flags['show_saturated'] == False:
                self.flags['show_saturated'] = True
                mask = np.where(np.bitwise_and(self.parent.parent.CUBE.data.pixeldq, dqflags.pixel['SATURATED']))
                self.parent.parent.plot_image.selectPixels(add=self.flags['show_saturated'], x=mask[1], y=mask[0],type='saturated')
                #for i, j in zip(mask[0], mask[1]):
                #    self.parent.parent.plot_image.selectPixel(add=self.flags['show_saturated'], x=j, y=i)
            else:
                self.flags['show_saturated'] = False
                self.parent.parent.plot_image.selectPixels(add=False,type='saturated')

    def show_DNU(self):
            if self.flags['show_dnu'] == False:
                self.flags['show_dnu'] = True
                mask = np.where(np.bitwise_and(self.parent.parent.CUBE.data.pixeldq, dqflags.pixel['DO_NOT_USE']))
                self.parent.parent.plot_image.selectPixels(add=self.flags['show_dnu'], x=mask[1], y=mask[0],color='orange',
                                                           type='dnu')
                # for i, j in zip(mask[0], mask[1]):
                #    self.parent.parent.plot_image.selectPixel(add=self.flags['show_saturated'], x=j, y=i)
            else:
                self.flags['show_dnu'] = False
                self.parent.parent.plot_image.selectPixels(add=False, type='dnu')

    def first_group(self):
        debug = self.parent.parent.exp_pars.debug.isChecked()
        self.parent.parent.CUBE.first_step(debug=debug)
        print('FIRST STEP: DONE')
    def last_group(self):
        debug = self.parent.parent.exp_pars.debug.isChecked()
        self.parent.parent.CUBE.last_step(debug=debug)
        print('LAST STEP: DONE')

    def reset_correction(self):
        debug = self.parent.parent.exp_pars.debug.isChecked()
        self.parent.parent.CUBE.reset_step(debug=debug)
        print('RESET STEP: DONE')

    def linear_correction(self):
        if self.parent.parent.CUBE.data.meta.cal_step.linearity == 'COMPLETE':
            print('Linearity correction was applied already')
        else:
            debug = self.parent.parent.exp_pars.debug.isChecked()
            self.parent.parent.CUBE.linear_step(debug=debug)
            print('LINEAR STEP: DONE')

    def show_linear_correction(self):
        if self.parent.parent.CUBE.data.meta.cal_step.linearity == 'COMPLETE':
            print('Linear correction was completed already')
        else:
            debug = self.parent.parent.exp_pars.debug.isChecked()
            print('show_linear_correction: not ready')

    def rscd_correction(self):
        if self.parent.parent.CUBE.data.meta.cal_step.rscd == 'COMPLETE':
            print('RSCD correction was applied already')
        else:
            debug = self.parent.parent.exp_pars.debug.isChecked()
            self.parent.parent.CUBE.rscd_step(debug=debug)
            print('RSCD STEP: DONE')

    def dark_correction(self):
        if self.parent.parent.EXP.data.meta.cal_step.dark_sub == 'COMPLETE':
            print('Dark subtraction was applied already')
        else:
            debug = self.parent.parent.exp_pars.debug.isChecked()
            self.parent.parent.EXP.dark_step(debug=debug)
            print('DARK STEP: DONE')

    def reference_pix_correction(self):
        if self.parent.parent.CUBE.data.meta.cal_step.refpix == 'COMPLETE':
            print('RefPix correction was applied already')
        else:
            debug = self.parent.parent.exp_pars.debug.isChecked()
            self.parent.parent.CUBE.refpix_corr_step(debug=debug)
            print('REF PIX STEP: DONE')

    def jump_detection(self):
        if self.flags['CR_step'] == False:
            CRlimit = float(self.parent.parent.exp_pars.CRlimit.text())
            RecalcMedian = int(self.parent.parent.exp_pars.CR_recalc_flag.currentIndex())
            flag = self.parent.parent.exp_pars.addCRneighbors.isChecked()
            debug = 3 #self.parent.parent.exp_pars.debug.isChecked()
            print('Run CR step: add_neighbors=', flag, ' show_debug = ', debug)
            self.parent.parent.CUBE.jump_corr_step(debug=debug, limit=CRlimit, flag_4_neighbors=flag,RecalcMedian=RecalcMedian)
            self.flags['CR_step'] = True
            print('CR STEP: DONE')
        else:
            CRlimit = float(self.parent.parent.exp_pars.CRlimit.text())
            flag = self.parent.parent.exp_pars.addCRneighbors.isChecked()
            debug = 3 #self.parent.parent.exp_pars.debug.isChecked()
            print('Run second CR step: add_neighbors=', flag, ' show_debug = ', debug)
            self.parent.parent.CUBE.reset_dq(flagname='JUMP_DET')
            self.parent.parent.CUBE.jump_corr_step(debug=debug, limit=CRlimit, flag_4_neighbors=flag)
            self.flags['CR_step'] = True
            print('CR STEP: DONE')
        if debug:
            plt.show()

        #debug = self.parent.parent.exp_pars.debug.isChecked()
        #CRlimit = float(self.parent.parent.exp_pars.CRlimit.text())
        #self.parent.parent.EXP.jump_corr_step(debug=debug,limit = CRlimit)
        #print('JUMP STEP (CR): DONE')

    def show_single_CR(self):
        if self.flags['CR_step'] == True:
            if self.flags['show_CR'] == False:
                self.flags['show_CR'] = True
                mask = np.where(np.bitwise_and(self.parent.parent.CUBE.data.pixeldq, dqflags.pixel['JUMP_DET']))
                #for i, j in zip(mask[0], mask[1]):
                #    self.parent.parent.plot_image.selectPixels(add=self.flags['show_CR'], x=j, y=i)
                self.parent.parent.plot_image.selectPixels(add=self.flags['show_CR'], x=mask[1], y=mask[0],color='g',type='cr')
            else:
                self.flags['show_CR'] = False
                self.parent.parent.plot_image.selectPixels(add=False,type='cr')

    def show_second_CR(self):
        print('show_muliple_CR:')
        if self.flags['CR_step'] == True:
            if self.flags['show_multi_CR'] == False:
                self.flags['show_multi_CR'] = True
                mask = np.where(np.bitwise_and(self.parent.parent.CUBE.data.pixeldq, dqflags.pixel['JUMP_DET']))
                x,y = [],[]
                for i in range(mask[0].shape[0]):
                    mask_tmp = np.bitwise_and(self.parent.parent.CUBE.data.groupdq[0,:,mask[0][i],mask[1][i]], dqflags.pixel['JUMP_DET'])
                    if np.sum(mask_tmp)>4:
                        x.append(mask[1][i])
                        y.append(mask[0][i])
                x,y = np.array(x),np.array(y)

                #for i, j in zip(mask[0], mask[1]):
                #    self.parent.parent.plot_image.selectPixels(add=self.flags['show_CR'], x=j, y=i)
                self.parent.parent.plot_image.selectPixels(add=self.flags['show_multi_CR'], x=x, y=y,color='m',type='cr_multi')
            else:
                self.flags['show_multi_CR'] = False
                self.parent.parent.plot_image.selectPixels(add=False,type='cr_multi')

    def slope_fit(self,denug=False):

        #CRlimit = float(self.parent.parent.exp_pars.CRlimit.text())
        #flag = self.parent.parent.exp_pars.addCRneighbors.isChecked()
        debug = self.parent.parent.exp_pars.debug.isChecked()
        print('Run Slope Fit step: show_debug = ', debug)
        self.parent.parent.CUBE.slope_fitting_step(debug=debug)
        self.flags['slope_fit_step'] = True
        print('Slope Fit STEP: DONE')
        if debug:
            plt.show()

    def read_slopes(self,input_file = None):
        #miri1 = calwebb_detector1.Detector1Pipeline()
        #miri_output = miri1.run(miri_uncal_file)
        if self.flags['slope_fit_step'] == True:
            print('Read Slope Fits from local files')
            ramp_fit = self.parent.parent.CUBE.ramp_fit
            if input_file == None:
                input_file = self.parent.parent.CUBE.input_file
            input_file_base = os.path.basename(input_file).replace('uncal.fits', '')
            output_dir = self.parent.parent.CUBE.output_dir
            rampfit_output_file = os.path.join(output_dir, '{}_0_rampfitstep.fits'.format(input_file_base))

            # Generate the name of the optional output file
            optional_file = os.path.join(output_dir, '{}fitopt.fits'.format(input_file_base))
            hdulist = fits.open(optional_file)
            intercepts = hdulist['YINT'].data[0, :, :, :]
            intercepts_err = hdulist['SIGYINT'].data[0, :, :, :]
            local_slopes = hdulist['SLOPE'].data[0, :, :, :]
            local_sig_slopes = hdulist['SIGSLOPE'].data[0, :, :, :]
            hdulist.close()

            num_groups = ramp_fit[0].meta.exposure.ngroups
            group_time = ramp_fit[0].meta.exposure.group_time
            print('Time per group:', group_time, ' in s')
            #group_times = np.arange(num_groups) * group_time


            self.parent.parent.CUBE.itercepts = intercepts
            self.parent.parent.CUBE.itercepts_err = intercepts_err
            self.parent.parent.CUBE.slopes = ramp_fit[0].data* group_time  #in DN/groups
            self.parent.parent.CUBE.slopes_err = ramp_fit[0].err* group_time #in DN/groups
            self.parent.parent.CUBE.local_slopes = local_slopes * group_time  # in DN/groups
            self.parent.parent.CUBE.local_slopes_err = local_sig_slopes * group_time  # in DN/groups
            self.flags['read_fit_slopes'] = True

        else:
            print('Read Slope Fits from Final files')
            if input_file == None:
                input_file = self.parent.parent.CUBE.input_file
            input_file_base = os.path.basename(input_file).replace('uncal.fits', '')
            output_dir = './output/results/' #self.parent.parent.EXP.output_dir

            files = os.listdir(output_dir)
            if np.sum(input_file_base in f for f in files):
                # Generate the name of the optional output file
                optional_file = os.path.join(output_dir, '{}fitopt.fits'.format(input_file_base))
                hdulist = fits.open(optional_file)
                intercepts = hdulist['YINT'].data[0, :, :, :]
                intercepts_err = hdulist['SIGYINT'].data[0, :, :, :]
                local_slopes = hdulist['SLOPE'].data[0, :, :, :]
                local_sig_slopes = hdulist['SIGSLOPE'].data[0, :, :, :]
                hdulist.close()

                optional_file = os.path.join(output_dir, '{}rate.fits'.format(input_file_base))
                hdulist = fits.open(optional_file)
                header = hdulist[0].header
                group_time =  hdulist[0].header['TGROUP']
                slopes = hdulist['SCI'].data[:,:]
                slope_errs = hdulist['ERR'].data[:, :]
                dq = hdulist['DQ'].data[:, :]
                print('Time per group:', group_time, ' in s')
                hdulist.close()
                ramp_fit_0  = datamodels.open(optional_file)
                ramp_fit_1 = datamodels.open(optional_file = os.path.join(output_dir, '{}rateints.fits'.format(input_file_base)))
                # group_times = np.arange(num_groups) * group_time

                self.parent.parent.CUBE.itercepts = intercepts
                self.parent.parent.CUBE.itercepts_err = intercepts_err
                self.parent.parent.CUBE.slopes = slopes * group_time  # in DN/groups
                self.parent.parent.CUBE.slopes_err = slope_errs * group_time  # in DN/groups
                self.parent.parent.CUBE.local_slopes = local_slopes * group_time  # in DN/groups
                self.parent.parent.CUBE.local_slopes_err = local_sig_slopes * group_time  # in DN/groups
                self.parent.parent.CUBE.data.pixeldq = dq
                self.parent.parent.CUBE.ramp_fit = ramp_fit_0,ramp_fit_1
                #self.parent.parent.EXP.ramp_fit[0].data = slopes
                #self.flags['read_fit_slopes'] = True
                self.flags=self.flags.fromkeys(self.flags, True)
                self.flags['show_saturated'] = False
                self.flags['show_CR'] = False
                self.flags['show_multi_CR'] = False
                self.flags['show_dnu'] = False
                print('flags after reading',self.flags)

                def val2log(val='1'):
                    val = val.replace("\n", "")
                    if val == '2':
                        return None
                    else:
                        return val
                cal_steps_file = output_dir + input_file_base+'cal_steps.csw'
                with open(cal_steps_file, 'r') as f:
                    for k, line in enumerate(f):
                        values = [s for s in line.split(',')]
                self.parent.parent.CUBE.data.meta.cal_step.linearity = val2log(values[0])
                self.parent.parent.CUBE.data.meta.cal_step.rscd = val2log(values[1])
                self.parent.parent.CUBE.data.meta.cal_step.dark_sub = val2log(values[2])
                self.parent.parent.CUBE.data.meta.cal_step.refpix = val2log(values[3])
                print('self.parent.parent.EXP.data.meta.cal_step.refpix',self.parent.parent.CUBE.data.meta.cal_step.refpix)

            else:
                print('There is no saved files')

    def show_slope_image(self):
        if self.flags['slope_fit_step'] == True:
            ramp_fit = self.parent.parent.CUBE.ramp_fit
            fig,ax = plt.subplots()
            print(np.min(ramp_fit[0].data),np.max(ramp_fit[0].data))
            im = ax.imshow(ramp_fit[0].data, vmin=-10, vmax=10)
            fontsize=12
            ax.set_title('Slope Image')
            ax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True', top='True')
            ax.tick_params(which='major', length=5)
            ax.tick_params(which='minor', length=3)
            fig.colorbar(im, ax=ax, label='Slope')

            plt.show()



    def save_slope_fit(self,output_dir=None,copy_add_data=True):
        if self.flags['slope_fit_step'] == True:
            from jwst.pipeline import calwebb_detector1
            miri1 = calwebb_detector1.Detector1Pipeline()
            if output_dir == 'local':
                miri1.output_dir = self.parent.parent.CUBE.output_dir
            elif output_dir == 'final':
                miri1.output_dir = './output/results/'
            else:
                output_dir = None
            if output_dir != None:
                miri1.output_file =  self.parent.parent.CUBE.name
                input, ints_model = self.parent.parent.CUBE.ramp_fit
                if ints_model is not None:
                    miri1.save_model(ints_model, 'rateints')
                if input is not None:
                    miri1.save_model(input, 'rate')
            if output_dir == 'final' and copy_add_data==True:
                # Providing the folder path
                origin = self.parent.parent.CUBE.output_dir
                target = miri1.output_dir

                # Fetching the list of all the files
                files = os.listdir(origin)

                # Fetching all the files to directory
                exp_name = self.parent.parent.CUBE.name.split('_uncal')[0]
                for file_name in files:
                    if exp_name in file_name:
                        shutil.copy(origin + file_name, target + file_name)

                s = self.parent.parent.CUBE.data.meta.cal_step
                lst = []
                for l in [s.linearity,s.rscd,s.dark_sub,s.refpix]:
                    if l != None:
                        lst.append(l)
                    else:
                        lst.append(2)
                print(lst)
                with open(target + exp_name+'_cal_steps.csw', 'w') as f:
                    csv_writer = csv.writer(f, delimiter=',')
                    csv_writer.writerows([lst])


                print("Fit slopes are saved to", target+exp_name)


class chooseExpWidget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """
    def __init__(self, parent, closebutton=True):
        super().__init__()
        self.parent = parent
        #self.resize(700, 900)
        #self.move(400, 100)
        self.setStyleSheet(open('styles.ini').read())

        self.saved = []

        layout = QVBoxLayout()

        self.table = CUBElistTable(self)
        self.filelist = {}
        self.associtations_list = {}

        if 1:
            filenames,fileparams,codenames = self.readfolder(self.parent.CUBE.output_dir)
            lst = []
            for s,pars in zip(filenames,fileparams):
                d = [s.split('/')[-1]]
                for p in pars:
                    d.append(p)
                lst.append(d)
                #filenamelst.append(d[0].split('/')[-1])
                self.filelist[d[0].split('/')[-1]]=s
                self.associtations_list[d[0].split('/')[-1]]=self.parent.CUBE.output_dir + d[-1]
            lst = np.array([tuple(l) for l in lst], dtype=[('name','U400')] + [(p,'U50') for p in codenames])
            data = lst
        self.table.setdata(data)
        #for l in lst:
        #    name = l[0]
        #    lst2 = []
        #    for l2 in lst:
        #        if l2 != l and (l2[i] in l[i] for i in range(len(l))):
        #            lst2.append(l2[0])
        #    if len(lst2)>0:
        #        self.associtations_list[name] = lst2
        #    else:
        #        self.associtations_list[name] = None

                #self.table.setSelectionBehavior(QTableView.SelectRows);

        #indexes = self.table.selectionModel().selectedRows()
        #for index in sorted(indexes):
        #    print('Row %d is selected' % index.row())
        self.buttons = {}
        for i, d in enumerate(data):
            wdg = QWidget()
            l = QVBoxLayout()
            l.addSpacing(3)
            button = QPushButton(d[0].split('_uncal')[0], self, checkable=True)
            button.setFixedSize(600, 30)
            #button.setChecked(False)
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
        codenames = ['TARGPROP','BAND','CHANNEL','ASNFILE']

        return lst,params,codenames

    def readfile(self,pathotofile = '',filename = ''):
        hdulist = fits.open(pathotofile+'/'+filename)
        header = hdulist[0].header
        prog_id, obs_id, targ_name, miri_band, miri_channel = None,None,None,None,None
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
        return [targ_name,miri_band,miri_channel,asn_file]




    def click(self, name):
        self.parent.current_name = name
        self.parent.plot_3dcube.add(name, self.buttons[name].isChecked())
        self.parent.plot_2dimage.add(name, self.buttons[name].isChecked())
        print('self.buttons[name].isChecked()',self.buttons[name].isChecked())
        if self.buttons[name].isChecked()==True:
            flags = self.parent.Cubes.table.flags
            self.parent.Cubes.table.flags = flags.fromkeys(flags, False)
            print('added by click', self.parent.Cubes.table.flags)
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
        l.addWidget(QLabel('Intergration:'))
        self.nINT = QLineEdit()
        self.nINT.setText(str(0))
        self.nINT.setFixedSize(90, 30)
        l.addWidget(self.nINT)

        l.addWidget(QLabel('Group:'))
        self.nGROUP = QLineEdit()
        self.nGROUP.setText(str(1))
        self.nGROUP.setFixedSize(90, 30)
        l.addWidget(self.nGROUP)

        l.addWidget(QLabel('Exp:'))
        self.nEXP = QLineEdit()
        self.nEXP.setText(str(0))
        self.nEXP.setFixedSize(90, 30)
        l.addWidget(self.nEXP)

        l.addStretch(1)
        layout.addLayout(l)

        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('SatPix:'))
        self.addneighbors = QCheckBox('4NeighPix')
        self.addneighbors.setChecked(True)
        horizontal_layout.addWidget(self.addneighbors)
        self.debug = QCheckBox('Show Debug')
        self.debug.setChecked(False)
        horizontal_layout.addWidget(self.debug)
        horizontal_layout.addStretch(1)
        layout.addLayout(horizontal_layout)

        #horizontal_layout.addWidget(QLabel('SatPixList:'))
        #self.satpixlist = QComboBox()
        #if self.parent.EXP.data == None:
        #    self.satpixlist.addItems(['None'])
        #elif self.parent.EXP.data.groupdq != None:
        #    list = []
        #    self.satpixlist.addItems(['total', 'one side', 'both sides'])
        #self.satpixlist.setCurrentIndex(2)
        #self.satpixlist.setFixedSize(90, 30)
        #horizontal_layout.addWidget(self.satpixlist)
        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('CR limit:'))
        self.CRlimit = QLineEdit()
        self.CRlimit.setText(str(5.0))
        self.CRlimit.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.CRlimit)
        self.addCRneighbors = QCheckBox('4NeighPix')
        self.addCRneighbors.setChecked(True)
        horizontal_layout.addWidget(self.addCRneighbors)
        horizontal_layout.addWidget(QLabel('VaryMed:'))
        self.CR_recalc_flag = QComboBox()
        #if self.parent.EXP.data == None:
        self.CR_recalc_flag.addItems(['No','Yes'])
        #    self.sides.setCurrentIndex(0)
        #elif self.parent.EXP.data.groupdq != None:
        #    list = []
        #    self.satpixlist.addItems(['total', 'one side', 'both sides'])
        self.CR_recalc_flag.setCurrentIndex(1)
        self.CR_recalc_flag.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.CR_recalc_flag)
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
        self.show_image = QPushButton('Show Image')
        self.show_image.clicked[bool].connect(partial(self.show_Image,'image'))
        self.show_image.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.show_image)
        self.show_slope = QPushButton('Show Slope')
        self.show_slope.clicked[bool].connect(partial(self.show_Image,'slope'))
        self.show_slope.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.show_slope)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        self.set_dq_map = QPushButton('Read DQ')
        self.set_dq_map.clicked[bool].connect(partial(self.set_DQ_map, False))
        self.set_dq_map.setFixedSize(150, 60)
        horizontal_layout.addWidget(self.set_dq_map)
        self.show_dq = QPushButton('Show DQ')
        self.show_dq.clicked[bool].connect(partial(self.show_DQ_map, False))
        self.show_dq.setFixedSize(150, 60)
        horizontal_layout.addWidget(self.show_dq)
        self.dq_categories = QComboBox()
        flags = [*dqflags.pixel]
        self.dq_categories.addItems(['all']+flags)
        self.dq_categories.setCurrentIndex(2)
        self.dq_categories.setFixedSize(90, 30)
        horizontal_layout.addWidget(self.dq_categories)
        horizontal_layout.addWidget(QLabel('Show:'))
        self.show_linear_step = QPushButton('DNU')
        self.show_linear_step.clicked[bool].connect(partial(self.show_DNUpixels))
        self.show_linear_step.setFixedSize(70, 60)
        horizontal_layout.addWidget(self.show_linear_step)
        self.show_sat_pixels = QPushButton('SatPix')
        self.show_sat_pixels.clicked[bool].connect(partial(self.show_SATpixels))
        self.show_sat_pixels.setFixedSize(100, 60)
        horizontal_layout.addWidget(self.show_sat_pixels)
        self.show_single_jumps = QPushButton('CR')
        self.show_single_jumps.clicked[bool].connect(partial(self.ShowFirstCR, False))
        self.show_single_jumps.setFixedSize(70, 60)
        horizontal_layout.addWidget(self.show_single_jumps)
        self.show_sec_jumps = QPushButton('MultiCR')
        self.show_sec_jumps.clicked[bool].connect(partial(self.ShowSecondCR, False))
        self.show_sec_jumps.setFixedSize(100, 60)
        horizontal_layout.addWidget(self.show_sec_jumps)

        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        self.saturation = QPushButton('1.Saturation')
        self.saturation.clicked[bool].connect(partial(self.SaturationStep, False))
        self.saturation.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.saturation)
        self.first_last_step = QPushButton('2.First&Last')
        self.first_last_step.clicked[bool].connect(partial(self.FirstLastStep, False))
        self.first_last_step.setFixedSize(150, 60)
        horizontal_layout.addWidget(self.first_last_step)
        self.reset_step = QPushButton('3.ResetAnomaly')
        self.reset_step.clicked[bool].connect(partial(self.ResetStep, False))
        self.reset_step.setFixedSize(250, 60)
        horizontal_layout.addWidget(self.reset_step)
        self.linear_step = QPushButton('4.LinearCorr')
        self.linear_step.clicked[bool].connect(partial(self.LinearStep, False))
        self.linear_step.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.linear_step)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)


        horizontal_layout = QHBoxLayout(self)
        self.rscd_step = QPushButton('5.RSCD')
        self.rscd_step.clicked[bool].connect(partial(self.RSCDStep, False))
        self.rscd_step.setFixedSize(150, 60)
        horizontal_layout.addWidget(self.rscd_step)
        self.dark_step = QPushButton('6.DarkSubtract')
        self.dark_step.clicked[bool].connect(partial(self.DarkStep, False))
        self.dark_step.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.dark_step)
        self.refpix_step = QPushButton('7.ReferencePix')
        self.refpix_step.clicked[bool].connect(partial(self.RefPixStep, False))
        self.refpix_step.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.refpix_step)
        self.jump_step = QPushButton('8. Jump Detection')
        self.jump_step.clicked[bool].connect(partial(self.JumpStep,False))
        self.jump_step.setFixedSize(250, 60)
        horizontal_layout.addWidget(self.jump_step)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)
        layout.addLayout(l)

        l = QVBoxLayout(self)
        l.addWidget(QLabel('Slope Fitting:'))
        horizontal_layout = QHBoxLayout(self)
        self.slope_fit_step = QPushButton('9. SlopeFit')
        self.slope_fit_step.clicked[bool].connect(partial(self.SlopeFitStep))
        self.slope_fit_step.setFixedSize(250, 60)
        horizontal_layout.addWidget(self.slope_fit_step)
        self.show_single_fit = QPushButton('SaveFit')
        self.show_single_fit.clicked[bool].connect(partial(self.SaveSlopeFit))
        self.show_single_fit.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.show_single_fit)
        self.show_jump_fit = QPushButton('ReadFit')
        self.show_jump_fit.clicked[bool].connect(partial(self.ReadSlopeFit, False))
        self.show_jump_fit.setFixedSize(200, 60)
        horizontal_layout.addWidget(self.show_jump_fit)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        layout.addLayout(l)
        layout.addStretch(1)
        self.setLayout(layout)

        self.setStyleSheet(open('styles.ini').read())


    def show_Image(self,mode=None):
        print('show Image:')
        self.parent.Cubes.table.show_image(mode=mode)


    def set_DQ_map(self, debug = False):
        print('set_DQ_map, debug:', debug)
        self.parent.Cubes.table.set_dq()

    def show_DQ_map(self, debug = False,group=None):
        print('set_DQ_map, debug:', debug)
        self.parent.Cubes.table.show_dq()


    def SaturationStep(self, debug = False):
        print('SaturationStep, debug:', debug)
        self.parent.Cubes.table.check_saturation()

    def show_SATpixels(self):
        print('Show Saturation pixels')
        self.parent.Cubes.table.show_saturation()

    def show_DNUpixels(self):
        print('Show Saturation pixels')
        self.parent.Cubes.table.show_DNU()

    def FirstLastStep(self, debug = False):
        self.parent.Cubes.table.first_group()
        self.parent.Cubes.table.last_group()

    def ResetStep(self, debug=False):
        self.parent.Cubes.table.reset_correction()


    def LinearStep(self, debug=False):
        self.parent.Cubes.table.linear_correction()

    def ShowLinearCorrection(self, debug=False):
        self.parent.Cubes.table.show_linear_correction()

    def RSCDStep(self, debug=False):
        self.parent.Cubes.table.rscd_correction()

    def DarkStep(self, debug=False):
        self.parent.Cubes.table.dark_correction()

    def RefPixStep(self, debug=False):
        self.parent.Cubes.table.reference_pix_correction()

    def JumpStep(self, debug=False):
        self.parent.Cubes.table.jump_detection()

    def ShowFirstCR(self, debug=False):
        self.parent.Cubes.table.show_single_CR()

    def ShowSecondCR(self, debug=False):
        self.parent.Cubes.table.show_second_CR()

    def SlopeFitStep(self, debug=False):
        self.parent.Cubes.table.slope_fit()
        self.parent.Cubes.table.read_slopes()
        self.parent.Cubes.table.save_slope_fit(output_dir='local')

    def ShowSlopeFit(self, debug=False):
        self.parent.Cubes.table.show_slope_image()

    def SaveSlopeFit(self, debug=False):
        self.parent.Cubes.table.save_slope_fit(output_dir='final')

    def ReadSlopeFit(self, debug=False):
        self.parent.Cubes.table.read_slopes()

    def RunDet1Pipeline(self,denug=False):
        print('Show Saturation pixels')
        self.parent.Cubes.table.show_saturation()

class JWST_spec_viewer(QMainWindow):

    def __init__(self):
        super().__init__()
        input_dir = './output/detector2'
        spec3_cachedir = './temp/spec3/'
        self.CUBE = detector3(obj_key_name = 'jw02155001001_04102',bkgr_key_name = 'jw02155009001_02101', path = input_dir, output_dir=output_dir)
        input2_dir = './output/results/'
        spec2_cachedir = './temp/spec2/'
        output2_dir = './output/detector2/'
        self.stage2 = detector2(miri_uncal_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=input2_dir, output_dir=output2_dir,spec2_cachedir=spec2_cachedir)
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
            self.plot_3dcube = plotCube(self)
            self.plot_2dimage = plotImage(self)
            self.plot_spectrum = plotSpec(self)
            self.Cubes = chooseExpWidget(self, closebutton=False)
            self.exp_pars = expParsWidget(self)
            self.exp_commands = expRunWidget(self)
            # self.plot.setFrameShape(QFrame.StyledPanel)

            self.splitter = QSplitter(Qt.Vertical)
            self.splitter_image = QSplitter(Qt.Horizontal)
            self.splitter_image.addWidget(self.plot_3dcube)
            self.splitter_image.addWidget(self.plot_2dimage)
            self.splitter.addWidget(self.splitter_image)

            self.splitter_plot = QSplitter(Qt.Vertical)
            self.splitter_plot.addWidget(self.plot_spectrum)
            self.splitter.addWidget(self.splitter_plot)

            self.splitter_pars = QSplitter(Qt.Horizontal)
            self.splitter_pars_left_panel = QSplitter(Qt.Vertical)
            self.splitter_pars_left_panel.addWidget(self.exp_pars)
            self.splitter_pars_left_panel.addWidget(self.exp_commands)
            self.splitter_pars.addWidget(self.splitter_pars_left_panel)
            self.splitter_pars.addWidget(self.Cubes)
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