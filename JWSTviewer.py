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
from PyQt6.QtCore import (Qt, )
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import (QFont, )
import pyqtgraph as pg
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline, Rbf
from scipy.interpolate import RBFInterpolator
from scipy.optimize import bisect
import sys
#sys.path.append('/home/slava/science/codes/python')
from stage1_pipeline import *
from stage1_pipeline import detector1
from stage2_pipeline import *
from stage2_pipeline import detector2
from scipy import signal
import scipy


#from spectro.stats import distr2d,distr1d
#from spectro.a_unc import a
#from spectro.pyratio import pyratio
import copy
from stdatamodels.jwst.datamodels import dqflags
from PyQt6.QtWidgets import (QApplication, QMessageBox, QMainWindow, QWidget,
                             QFileDialog, QTextEdit, QVBoxLayout,
                             QSplitter, QFrame, QLineEdit, QLabel, QPushButton, QCheckBox,QHeaderView,
                             QGridLayout, QTabWidget, QFormLayout, QHBoxLayout, QRadioButton,
                             QTreeWidget, QComboBox, QTreeWidgetItem, QAbstractItemView,
                             QStatusBar, QMenu, QButtonGroup, QMessageBox, QToolButton, QColorDialog)
#from pyqtgraph.Qt import QtCore, QtGui
from stdatamodels.jwst import datamodels
import csv
'''
#output_dir = './output/detector1/'
#output2_dir = './output/detector2/'
#input_dir = './input/detector1/'
#miri_uncal_file= 'jw02155001001_04102_00001_mirifulong_uncal.fits'
#input_file_base = os.path.basename(miri_uncal_file).replace('uncal.fits', '')

#class image():
#    """
#    class for working with images (2d spectra) inside Spectrum plotting
#    """
#    def __init__(self, x=None, y=None, z=None, err=None, mask=None):
#        if any([v is not None for v in [x, y, z, err, mask]]):
#            self.set_data(x=x, y=y, z=z, err=err, mask=mask)
#        else:
#            self.z = None

 #   def set_data(self, x=None, y=None, z=None, err=None, mask=None):
 #       for attr, val in zip(['z', 'err', 'mask'], [z, err, mask]):
 #           if val is not None:
 ##               setattr(self, attr, np.asarray(val))
  #          else:
  #              setattr(self, attr, val)
  #      if x is not None:
  #          self.x = np.asarray(x)
  #      else:
  #          self.x = np.arange(z.shape[0])
  #      if y is not None:
  #          self.y = np.asarray(y)
  #      else:
  #          self.y = np.arange(z.shape[1])

  #      self.pos = [self.x[0] - (self.x[1] - self.x[0]) / 2, self.y[0] - (self.y[1] - self.y[0]) / 2]
  ##      self.scale = [(self.x[-1] - self.x[0]) / (self.x.shape[0]-1), (self.y[-1] - self.y[0]) / (self.y.shape[0]-1)]
   #     for attr in ['z', 'err']:
  #          self.getQuantile(attr=attr)
  #          self.setLevels(attr=attr)#


 #   def getQuantile(self, quantile=0.997, attr='z'):
 #       if getattr(self, attr) is not None:
 #           x = np.sort(getattr(self, attr).flatten())
 #           x = x[~np.isnan(x)]
 #           setattr(self, attr+'_quantile', [x[int(len(x)*(1-quantile)/2)], x[int(len(x)*(1+quantile)/2)]])
 #       else:
 #           setattr(self, attr + '_quantile', [0, 1])
#
 #   def setLevels(self, bottom=None, top=None, attr='z'):
 #       quantile = getattr(self, attr+'_quantile')
 #       if bottom is None:
 #           bottom = quantile[0]
  #      if top is None:
 #           top = quantile[1]
  #      top, bottom = np.max([top, bottom]), np.min([top, bottom])
  #      if top - bottom < (quantile[1] - quantile[0]) / 100:
 #           top += ((quantile[1] - quantile[0]) / 100 - (top - bottom)) /2
 #           bottom -= ((quantile[1] - quantile[0]) / 100 - (top - bottom)) / 2
 #       setattr(self, attr+'_levels', [bottom, top])
#
 #   def find_nearest(self, x, y, attr='z'):
#        z = getattr(self, attr)
#        if len(z.shape) == 2:
#            return z[np.min([z.shape[0]-1, (np.abs(self.y - y)).argmin()]), np.min([z.shape[1]-1, (np.abs(self.x - x)).argmin()])]
#        else:
#            return None
'''


class plotImage(pg.ImageView): #(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        # Add Plot item to show axis labels
        # pg.setLabel(axis='left', text='Y-axis')
        # pg.setLabel(axis='bottom', text='X-axis')

        #pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'Row', 'bottom': 'Col'})
        #pg.ImageView.__init__(self,name='Exposure image',view=pg.ViewBox())
        pg.ImageView.__init__(self, name='Exposure image', view=pg.PlotItem())
        #pg.setLabel(axis='left', text='Y-axis')
        #pg.setLabel(axis='bottom', text='X-axis')
        #imv = pg.ImageView()
        #imv.__init__(self, name='Exposure image')
        self.initstatus()
        #self.vb = self.getViewBox()
        self.vb = self.getView()
        #self.view = {}
        ## Set a custom color map
        cmap = pg.colormap.get('CET-D7') #pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
        self.setColorMap(cmap)
        self.vb.invertY(False)
        self.cursorpos = pg.TextItem(anchor=(0, 1))
        self.vb.setLabel(axis='left', text='Y-axis (Wavelength)')
        self.vb.setLabel(axis='bottom', text='X-axis (Slit)')
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
                nint = int(self.parent.exp_pars.nINT.text())-1
                ngroup =  int(self.parent.exp_pars.nGROUP.text())
                # number of group in integration
                print('add name:',name)
                path = self.parent.Exposures.filelist[name]
                self.parent.EXP.add(name=path)
                self.parent.EXP.dq_init_step()
                nintmax = self.parent.EXP.data.data.shape[0]
                ngrmax = self.parent.EXP.data.data.shape[1]
                if nint > nintmax-1:
                    print('Set correct Integration number, <=', nintmax-1)
                    data = self.parent.EXP.data.data[nintmax-1][ngrmax-1]
                    self.setImage(data*0, autoRange=True, levels=[2000, 6000])
                elif ngroup > ngrmax - 1:
                    print('Set correct Group number, <=', ngrmax - 1)
                    data = self.parent.EXP.data.data[nintmax - 1][ngrmax - 1]
                    self.setImage(data * 0, autoRange=True, levels=[2000, 6000])
                else:
                    data = self.parent.EXP.data.data[nint][ngroup]
                    self.data = data
                    self.err = np.zeros_like(data)
                    #self.view[name] = pg.ImageView()
                    #self.view[name].setImage(img=data)
                    x, y = (1, 0)
                    self.setImage(data, autoRange=True,levels=[2000,6000],axes={'t': None, 'x': x, 'y': y, 'c': None})
                    self.vb.hoverEvent = self.imageHoverEvent
                    #self.vb.addItem(self.view[name])
                    #self.legend.addItem(self.view[name], name)
                    #self.redraw()
                self.parent.EXP.get_readnoise()
            elif mode == 'image':
                nint = int(self.parent.exp_pars.nINT.text())-1
                ngroup = int(self.parent.exp_pars.nGROUP.text())
                # number of group in integration
                nintmax = self.parent.EXP.data.data.shape[0]
                ngrmax = self.parent.EXP.data.data.shape[1]
                if nint > nintmax - 1:
                    print('Set correct Integration number, <=', nintmax - 1)
                    data = self.parent.EXP.data.data[nintmax - 1][ngrmax - 1]
                    err =self.parent.EXP.data.err[nintmax - 1][ngrmax - 1]
                    self.setImage(data * 0, autoRange=True, levels=[2000, 6000])
                elif ngroup > ngrmax - 1:
                    print('Set correct Group number, <=', ngrmax - 1)
                    data = self.parent.EXP.data.data[nintmax - 1][ngrmax - 1]
                    err = self.parent.EXP.data.err[nintmax - 1][ngrmax - 1]
                    self.setImage(data * 0, autoRange=True, levels=[2000, 6000])
                else:
                    data = self.parent.EXP.data.data[nint][ngroup]
                    err = self.parent.EXP.data.err[nint][ngroup]
                x, y = (1, 0)
                zmin, zmax = np.nanquantile(data.flatten(), 0.01), np.nanquantile(data.flatten(), 0.99)
                #self.setImage(data, autoRange=True, levels=[2000, 6000], axes={'t': None, 'x': x, 'y': y, 'c': None})
                self.data = data
                self.err = err
                self.setImage(data, autoRange=True, levels=[zmin, zmax], axes={'t': None, 'x': x, 'y': y, 'c': None})
                self.vb.hoverEvent = self.imageHoverEvent
            elif mode == 'slope':
                nint = int(self.parent.exp_pars.nINT.text())-1
                if self.parent.Exposures.table.flags['read_fit_slopes']:
                    #data = self.parent.EXP.ramp_fit[0].data
                    data = self.parent.EXP.int_slopes[nint,:,:]
                    zmin, zmax = np.nanquantile(data.flatten(), 0.01), np.nanquantile(data.flatten(), 0.99)
                    x, y = (1, 0)
                    self.data = data
                    self.err = self.parent.EXP.int_sigslopes[nint,:,:]
                    self.setImage(data, autoRange=True, levels=[zmin, zmax],
                                  axes={'t': None, 'x': x, 'y': y, 'c': None})
                    self.vb.hoverEvent = self.imageHoverEvent
                else:
                    print('There is no SLOPE ARRAY')
                    data = (self.parent.EXP.data.data[0][0]).copy()*0
                    x, y = (1, 0)
                    self.setImage(data, autoRange=True, levels=[-1, 5],
                                  axes={'t': None, 'x': x, 'y': y, 'c': None})
                    self.vb.hoverEvent = self.imageHoverEvent
            elif mode == 'rate':
                if self.parent.Exposures.table.flags['read_fit_slopes']:
                    data = self.parent.EXP.mean_slope
                    zmin, zmax = np.nanquantile(data.flatten(), 0.01), np.nanquantile(data.flatten(), 0.99)
                    x, y = (1, 0)
                    self.data = data
                    self.err = self.parent.EXP.mean_slope_sig

                    self.setImage(data, autoRange=True, levels=[zmin, zmax],
                                  axes={'t': None, 'x': x, 'y': y, 'c': None})
                    self.vb.hoverEvent = self.imageHoverEvent
                else:
                    print('There is no SLOPE ARRAY')
                    data = (self.parent.EXP.data.data[0][0]).copy() * 0
                    x, y = (1, 0)
                    self.setImage(data, autoRange=True, levels=[-1, 5],
                                  axes={'t': None, 'x': x, 'y': y, 'c': None})
                    self.vb.hoverEvent = self.imageHoverEvent

            elif mode == 'stage2':
                if self.parent.Exposures.table.current_pipeline_stage == 'stage2':
                    print('Show image of'+self.parent.stage2.name)
                    data = self.parent.stage2.data.data
                    zmin,zmax = np.nanquantile(data.flatten(),0.01),np.nanquantile(data.flatten(),0.99)
                    x, y = (1, 0)
                    self.data = data
                    self.err = self.parent.stage2.data.err
                    self.setImage(data, autoRange=True, #levels=[-1, 5],
                                  axes={'t': None, 'x': x, 'y': y, 'c': None}, levels=[zmin,zmax])
                    self.vb.hoverEvent = self.imageHoverEvent
                else:
                    print('There is no stage 2 data')
                    t = np.linspace(0, 1, 1024)*0
                    data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]
                    #data = (self.parent.EXP.data.data[0][0]).copy()*0
                    x, y = (1, 0)
                    self.setImage(data2d, autoRange=True, levels=[-1, 5],
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
        col,row,val = int(self.x), int(self.y), None
        text = "pixel: (%d, %d), val: None" % (col, row)
        if row < self.data.shape[0] and row >=0 and col <self.data.shape[1] and col > 0:
            val = self.data[row,col]
            err = self.err[row,col]
            text = "pixel: (%d, %d), val: %.2f,%.2f" % (col,row, val,err)
        #print('coord', text, val)
        self.label.setText(text)
        self.label.resize(500, 40)
        if self.parent.Exposures.table.current_pipeline_stage == 'stage2' and 0:
            wcs = self.parent.stage2.data.meta.wcs
            print(wcs.available_frames)
            cal_detector_to_alpha = wcs.get_transform('detector', 'alpha_beta')
            print('detector_to_alpha', cal_detector_to_alpha(col,row))
            cal_detector_to_v2v3 = wcs.get_transform('detector', 'v2v3')
            print('detector_to_v2v3', cal_detector_to_v2v3(col, row))
            #cal_detector_to_v2v3vacor = wcs.get_transform('detector', 'v2v3vacor')
            #print('detector_to_v2v3vacor', cal_detector_to_v2v3vacor(col, row))
            cal_detector_to_world = wcs.get_transform('detector', 'world')
            print('detector_to_world', cal_detector_to_world(col, row))
            print('')

    def mousePressEvent(self, event, pos=None):
        print(pos)
        if pos is None:
            super(plotImage, self).mousePressEvent(event)
            if  event.button() == Qt.MouseButton.LeftButton: #event.button() == Qt.LeftButton:
                #self.mousePoint = self.vb.vb.mapSceneToView(event.pos())
                self.mousePoint = self.vb.vb.mapSceneToView(QPointF(event.pos()))

                #self.mousePoint = self.vb.mapRectToView(event.pos())
                self.x, self.y = self.mousePoint.x(), self.mousePoint.y()
        else:
            self.x, self.y = pos
        print('Position on Image:', self.x, self.y)
        if self.s_status:
            #name = self.parent.name

            if self.parent.EXP.data.data is not None:
                col,row =int(self.x),int(self.y)
                show_fit = self.parent.Exposures.table.flags['read_fit_slopes']
                self.parent.plot_pixel.plot_profile(row=row, col=col,add=False,show_fit=show_fit)
                self.parent.plot_pixel.plot_profile(row=row, col=col,show_fit=show_fit)
                if self.parent.Exposures.table.current_pipeline_stage == 'stage1':
                    self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col, add=False)
                    self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col)
                elif self.parent.Exposures.table.current_pipeline_stage == 'stage2':
                    self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col, add=False)
                    self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col)
                elif self.parent.Exposures.table.current_pipeline_stage == 'stage2' and 0:
                    self.parent.plot_pixel_diffs.plot_pixel_diffs(row=row, col=col, add=False)
                    self.parent.plot_pixel_diffs.plot_verical_stripe(row=row, col=col, add=False)
                    self.parent.plot_pixel_diffs.plot_verical_stripe(row=row, col=col)

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

class plotPixProfile(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'DN', 'bottom': 'Groups'})
        #self.initstatus()
        self.vb = self.getViewBox()
        #self.image = None
        self.text = None
        #self.grid = image()
        cdict = cm.get_cmap('viridis')
        cmap = np.array(cdict.colors)
        cmap[-1] = [1, 0.4, 0]
        map = pg.ColorMap(np.linspace(0, 1, cdict.N), cmap, mode='rgb')
        self.colormap = map.getLookupTable(0.0, 1.0, 256, alpha=False)
        self.legend = pg.LegendItem(offset=(-70, 30))
        self.legend.setParentItem(self.vb)
        self.legend_model = pg.LegendItem(offset=(-70, -30))
        self.legend_model.setParentItem(self.vb)
        self.setTitle("Pixel profile", color="olive", size="10pt")



    def set_data(self, x=None, y=None, z=None, view='text'):
        if view == 'text':
            if self.text is not None:
                for t in self.text:
                    self.vb.removeItem(t)
            self.text = []
            for xi, yi, name in zip(x, y, [self.parent.H2.models[m].name for m in self.parent.H2.mask]):
                self.text.append(textLabel(self, '{:.1f}'.format(zi), x=np.log10(xi), y=np.log10(yi), name=name))
                #self.text.append(pg.TextItem(html='<div style="text-align: center"><span style="color: #FF0; font-size: 16pt;">' + '{:.1f}'.format(zi) + '</span></div>'))
                #self.text[-1].setPos(np.log10(xi), np.log10(yi))
                self.vb.addItem(self.text[-1])

        if view == 'image':
            if self.image is not None:
                self.vb.removeItem(self.image)
            self.image = pg.ImageItem()
            self.grid.set_data(x=x, y=y, z=z)
            self.image.translate(self.grid.pos[0], self.grid.pos[1])
            self.image.scale(self.grid.scale[0], self.grid.scale[1])
            self.image.setLookupTable(self.colormap)
            self.image.setLevels(self.grid.levels)
            self.vb.addItem(self.image)

    def plot_profile(self, row=None, col=None, add=True,show_fit=False,sat_limit = 55000,debug=False):
        if add:
            nint = int(self.parent.exp_pars.nINT.text())-1
            if row <= self.parent.EXP.data.data.shape[2] and row>=0 and col <= self.parent.EXP.data.data.shape[3] and col>=0 and nint <= self.parent.EXP.data.data.shape[0]:
                # number of group in integration
                data = self.parent.EXP.data.data[nint, :,row,col]
                error = np.zeros_like(data)
                diffs = np.append([0], np.diff(data))
                if 1:
                    sat_flag = dqflags.pixel["SATURATED"]
                    dnu_flag = dqflags.pixel["DO_NOT_USE"]
                    pixeldq = self.parent.EXP.data.groupdq[nint, :, row, col]
                    mask_diff = np.where(np.bitwise_and(pixeldq, dnu_flag) + np.bitwise_and(pixeldq, sat_flag))
                    median_diff = diffs.copy()
                    print('pixel group dq:', pixeldq)
                    print('pixel dq:', self.parent.EXP.data.pixeldq[row, col])
                    median_diffs = 0
                    if np.sum(pixeldq!=0) != np.size(pixeldq):
                        if debug:
                            print(' median_diff', median_diff)
                        median_diff[mask_diff] = np.nan
                        median_diff[np.nanargmax(np.abs(median_diff), axis=0)] = np.nan
                        median_diffs = np.nanmedian(median_diff, axis=0)


                read_noise = self.parent.EXP.readnoisearray[row, col]
                error += np.sqrt(np.abs(median_diffs) + read_noise ** 2)
                #print('Err',np.nanmax(self.parent.EXP.data.err))
                dq_map = self.parent.EXP.data.groupdq[nint, :,row,col]

                x= np.arange(data.shape[0])
                self.temp_model = pg.PlotCurveItem(x, data)
                mask = dq_map == 0
                self.good_groups = pg.ErrorBarItem(x=np.asarray(x), y=data, top=error, bottom=error, beam=2)
                self.plot_scatters = pg.ScatterPlotItem(x[mask], data[mask], symbol='o', size=20, brush='r')
                mask_CR = dq_map == dqflags.pixel['JUMP_DET']
                #if np.sum(mask_CR>0):
                self.plot_scatters_CRs = pg.ScatterPlotItem(x[mask_CR], data[mask_CR], symbol='o', size=20, brush='b')
                # add fit
                if show_fit:
                    algorithm = self.parent.exp_pars.FitAlgorithm_mode.currentText()
                    if algorithm == 'OLS_C':
                        YINT = np.mean(data[:2])
                        YSLP_MEAN = self.parent.EXP.mean_slope[row,col]
                        YSLP_INT = self.parent.EXP.int_slopes[nint, row, col]
                        print(row,col, 'YSLP',YSLP_MEAN,'YINT',YSLP_INT)
                        pen = pg.mkPen(color='r', style=Qt.PenStyle.DashLine, width=3)
                        if debug:
                            print('maskCR',mask_CR)
                            print('maskCR cumsum', np.cumsum(mask_CR))

                        self.slope_model,self.local_slope_model = [],[]
                        self.slope_model.append(pg.PlotCurveItem(x, YINT+YSLP_MEAN*x, pen=pen))
                        self.local_slope_model.append(pg.PlotCurveItem(x, YINT+YSLP_INT*x,
                                                                       pen=pg.mkPen(color='b',
                                                                        style=Qt.PenStyle.DashLine, width=3)))

                    elif algorithm == 'CHI2':
                        if 0 in dq_map:
                            i_gr = np.where(dq_map == 0)[0][0]
                            YINT = self.parent.EXP.data.data[nint,i_gr,row,col]
                            YSLP = self.parent.EXP.int_slopes[nint,0,row,col]
                            print(row,col, 'YSLP',YSLP, ' YINT',YINT)

                            YJUMP = self.parent.EXP.int_jumps[nint,:,row,col]
                            print('YJUMP',YJUMP)
                            y_fit = YINT + YSLP * (x-i_gr)
                            p = 0
                            for ii,s in enumerate(mask_CR):
                                if s:
                                    y_fit[ii:] += YJUMP[p]
                                    p+=1

                            pen = pg.mkPen(color='r', style=Qt.PenStyle.DashLine, width=3)
                            self.slope_model= []
                            self.slope_model.append(pg.PlotCurveItem(x, y_fit, pen=pen))



                self.vb.addItem(self.temp_model)
                self.vb.addItem(self.good_groups)
                self.vb.addItem(self.plot_scatters)
                self.vb.addItem(self.plot_scatters_CRs)
                self.vb.setLimits(xMin=-2,xMax=x[-1]+2)
                if show_fit:
                    if algorithm == 'OLS_C':
                        for m,l in zip(self.slope_model,self.local_slope_model):
                            self.vb.addItem(m)
                            self.vb.addItem(l)
                    else:
                        if 0 in dq_map:
                            self.vb.addItem(self.slope_model[0])

                if np.sum(data>sat_limit):
                    self.parent.show_sat_limit=True
                    pen = pg.mkPen(color='b', style=Qt.PenStyle.DashLine, width=3)
                    self.satur_limit = pg.PlotCurveItem([-2,x[-1]+2], [sat_limit,sat_limit], pen=pen)
                    self.vb.addItem(self.satur_limit)

                self.vb.autoRange()
                text = 'Pixel: '
                if col is not None:
                    text += ' {0:.0f} {1:.0f}'.format(col,row)
                self.legend_model.addItem(self.temp_model, str(text))

                if show_fit and algorithm == 'OLS_C':
                    self.legend_model.addItem(self.slope_model[0], 'mean')
                    self.legend_model.addItem(self.local_slope_model[0], f'int_{nint}')
            else:
                print('coords are out of limits of data', nint,row,col)
        else:
            try:
                algorithm = self.parent.exp_pars.FitAlgorithm_mode.currentText()
                self.legend_model.removeItem(self.temp_model)

                self.vb.removeItem(self.temp_model)
                self.vb.removeItem(self.good_groups)
                self.vb.removeItem(self.plot_scatters)
                self.vb.removeItem(self.plot_scatters_CRs)
                if show_fit:
                    if algorithm == 'OLS_C':
                        for m, l in zip(self.slope_model, self.local_slope_model):
                            self.vb.removeItem(m)
                            self.vb.removeItem(l)
                        self.legend_model.removeItem(self.local_slope_model[0])
                        self.legend_model.removeItem(self.slope_model[0])
                    if algorithm == 'CHI2':
                        self.vb.removeItem(self.slope_model[0])
                if self.parent.show_sat_limit:
                    self.vb.removeItem(self.satur_limit)
                    self.parent.show_sat_limit = False

            except:
                pass

    def plot_horizontal_stripe(self, row=None, col=None, add=True):
        if add:
            data = self.parent.stage2.data.data[row,:]
            error = np.array(self.parent.stage2.data.err[row,:])

            dq_map = self.parent.stage2.data.dq[ :, col]

            x= np.arange(data.shape[0])
            mask = (x>1)*(dq_map==0)
            self.plot_line = pg.PlotCurveItem(x, data,pen=pg.mkPen(color='lightpink', width=1))
            #self.plot_scatters = pg.ScatterPlotItem(x[mask], daiffs2err[mask],symbol = 'o',size=20,brush='r')
            self.plot_ebars = pg.ErrorBarItem(x=np.asarray(x)[mask], y=error[mask], top=1, bottom=1, beam=0.5)
            self.plot_col_lines = []
            colors= ['gray','gray','yellow','gray','blue','purple','pink']
            delta_y,delta_x = 0.,0
            for i,cols in enumerate(np.arange(col-2,col+2)):
                pen_spec = pg.mkPen(color=colors[i], width=1)
                data = self.parent.stage2.data.data[:, cols]
                win = signal.windows.hann(10)
                filtered = signal.convolve(data, win, mode='same') / sum(win)
                self.plot_col_lines.append(pg.PlotCurveItem(x+delta_x*(cols-col), filtered+ delta_y*(cols-col),pen=pen_spec))

            #mask_CR = dq_map == dqflags.pixel['JUMP_DET']
            #if np.sum(mask_CR>0):
            #self.plot_scatters_CRs = pg.ScatterPlotItem(x[mask_CR], diffs2err[mask_CR], symbol='o', size=20, brush='b')

            self.vb.addItem(self.plot_line)
            #self.vb.addItem(self.plot_scatters)
            #self.vb.addItem(self.plot_scatters_CRs)
            #self.vb.addItem(self.plot_ebars)
            self.vb.setLimits(xMin=-2, xMax=x[-1]+2)

            pen = pg.mkPen(color='darkgray', style=Qt.PenStyle.DashLine, width=1)
            self.zero_level = pg.PlotCurveItem([-2, x[-1] + 2], [0, 0], pen=pen)
            self.vb.addItem(self.zero_level)


            for l in self.plot_col_lines:
                self.vb.addItem(l)


            self.vb.autoRange()
            self.vb.setLimits(xMin=-2, xMax=x[-1] + 2)

            text = 'Col: '
            if col is not None:
                text += ' {0:.0f}'.format(col)
            self.legend_model.addItem(self.plot_line, text)
        else:
            try:
                self.vb.removeItem(self.plot_line)
                self.vb.removeItem(self.plot_ebars)
                self.vb.removeItem(self.zero_level)
                self.legend_model.removeItem(self.plot_line)
                for l in self.plot_col_lines:
                    self.vb.removeItem(l)


            except:
                pass

class plotPixDiffs(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'Diffs/Sigma', 'bottom': 'Groups'})
        #self.initstatus()
        self.vb = self.getViewBox()
        self.image = None
        self.text = None
        #self.grid = image()
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




    def plot_pixel_diffs(self, row=None, col=None, add=True):
        if add:
            if row <= self.parent.EXP.data.data.shape[2] and row >= 0 and col <= self.parent.EXP.data.data.shape[
                3] and col >= 0:
                nint = int(self.parent.exp_pars.nINT.text())-1
                # number of group in integration
                data = self.parent.EXP.data.data[nint, :,row, col]
                error = np.zeros_like(data)
                diffs = np.append([0],np.diff(data))
                if 1:
                    sat_flag = dqflags.pixel["SATURATED"]
                    dnu_flag = dqflags.pixel["DO_NOT_USE"]
                    pixeldq = self.parent.EXP.data.groupdq[nint, :, row, col]
                    mask_diff = np.where(np.bitwise_and(pixeldq, dnu_flag)+np.bitwise_and(pixeldq, sat_flag))
                    median_diff = diffs.copy()
                    median_diff[mask_diff] = np.nan
                    if np.sum(~np.isnan(median_diff))>0:
                        median_diff[np.nanargmax(np.abs(median_diff), axis=0)] = np.nan
                        median_diffs = np.nanmedian(median_diff, axis=0)
                    else:
                        median_diffs = 0

                read_noise = self.parent.EXP.readnoisearray[row,col]
                error+=np.sqrt(np.abs(median_diffs) + read_noise**2)
                #print('Err',np.nanmax(self.parent.EXP.data.err))
                dq_map = self.parent.EXP.data.groupdq[nint, :,row, col]
                x= np.arange(data.shape[0])
                mask = (x>1)*(dq_map==0)
                #diffs /=error
                diffs2err= diffs /error
                self.plot_line = pg.PlotCurveItem(x, diffs2err)
                self.plot_scatters = pg.ScatterPlotItem(x[mask], diffs2err[mask],symbol = 'o',size=20,brush='r')
                self.plot_ebars = pg.ErrorBarItem(x=np.asarray(x)[mask], y=diffs2err[mask], top=1, bottom=1, beam=0.5)
                mask_CR = dq_map == dqflags.pixel['JUMP_DET']
                #if np.sum(mask_CR>0):
                self.plot_scatters_CRs = pg.ScatterPlotItem(x[mask_CR], diffs2err[mask_CR], symbol='o', size=20, brush='b')

                self.vb.addItem(self.plot_line)
                self.vb.addItem(self.plot_scatters)
                self.vb.addItem(self.plot_scatters_CRs)
                self.vb.addItem(self.plot_ebars)
                self.vb.setLimits(xMin=-2, xMax=x[-1]+2)
                pen = pg.mkPen(color='darkgray', style=Qt.PenStyle.DashLine, width=1)
                self.zero_level = pg.PlotCurveItem([-2, x[-1] + 2], [0, 0], pen=pen)
                self.vb.addItem(self.zero_level)

                pen = pg.mkPen(color='gray', style=Qt.PenStyle.DashLine, width=3)
                self.median_diff_level = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1], median_diffs/error[1]], pen=pen)
                self.vb.addItem(self.median_diff_level)

                sat_limit = float(self.parent.exp_pars.CRlimit.text())
                print('pixel diffs:', (diffs-median_diffs)/error)
                print('')
                if np.sum((diffs-median_diffs)/error > 5):
                    self.parent.show_cr_limit = True
                    pen = pg.mkPen(color='b', style=Qt.PenStyle.DashLine, width=3)
                    self.crp_limit = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1]+sat_limit, median_diffs/error[1]+sat_limit], pen=pen)
                    self.crm_limit = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1]-sat_limit, median_diffs/error[1]-sat_limit], pen=pen)
                    self.vb.addItem(self.crp_limit)
                    self.vb.addItem(self.crm_limit)
                    pen = pg.mkPen(color='g', style=Qt.PenStyle.DashLine, width=3)
                    self.crp_limit_3cr = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1]+5,median_diffs/error[1]+5], pen=pen)
                    self.vb.addItem(self.crp_limit_3cr)
                    pen = pg.mkPen(color='r', style=Qt.PenStyle.DashLine, width=3)
                    self.crp_limit_2cr = pg.PlotCurveItem([-2, x[-1] + 2], [median_diffs/error[1]+6,median_diffs/error[1]+6], pen=pen)
                    self.vb.addItem(self.crp_limit_2cr)

                self.vb.autoRange()
                self.vb.setLimits(xMin=-2, xMax=x[-1] + 2)

                text = 'Pixel: '
                if col is not None:
                    text += ' {0:.0f} {1:.0f}'.format(col, row)
                self.legend_model.addItem(self.plot_line, text)
        else:
            try:
                self.vb.removeItem(self.plot_line)
                self.vb.removeItem(self.plot_scatters)
                self.vb.removeItem(self.plot_scatters_CRs)
                self.vb.removeItem(self.plot_ebars)
                self.vb.removeItem(self.median_diff_level)
                self.vb.removeItem(self.zero_level)

                self.legend_model.removeItem(self.plot_line)
                if self.parent.show_cr_limit == True:
                    self.vb.removeItem(self.crp_limit)
                    self.vb.removeItem(self.crm_limit)
                    self.parent.show_cr_limit = False
                    self.vb.removeItem(self.crp_limit_3cr)
                    self.vb.removeItem(self.crp_limit_2cr)

            except:
                pass

    def plot_verical_stripe(self, row=None, col=None, add=True):
        if add:
            data = self.parent.stage2.data.data[:, col]
            error = np.array(self.parent.stage2.data.err[:, col])

            dq_map = self.parent.stage2.data.dq[ :, col]
            if 0:
                wcs = self.parent.stage2.data.meta.wcs
                cal_detector_to_world = wcs.get_transform('detector', 'alpha_beta')
                print('cal_detector_to_world 376,589 ',cal_detector_to_world( 376, 589))
                print('cal_detector_to_world 377 589 ', cal_detector_to_world(377, 589))
                print('cal_detector_to_world 100 589', cal_detector_to_world(100, 589))
                run_detector_to_world = wcs.get_transform('detector', 'world')
                #data_coord_X,data_coord_Y = np.zeros_like(self.parent.stage2.data.data),np.zeros_like(self.parent.stage2.data.data)
                #for i in range(self.parent.stage2.data.data.shape[0]):
                #    data_coord_X[i,:] = i
                #for i in range(self.parent.stage2.data.data.shape[1]):
                #    data_coord_Y[:,i] = i
                ra, dec = np.zeros_like(self.parent.stage2.data.data), np.zeros_like(self.parent.stage2.data.data)
                for i in range(self.parent.stage2.data.data.shape[0]):
                    for j in range(self.parent.stage2.data.data.shape[1]):
                        #(r,d,w) = wcs(300+j,500+i)
                        (r,d,w) = cal_detector_to_world(j,i)
                        ra[i, j], dec[i, j] = r,d
                #(ra, dec, wave) = wcs.transform("detector", "world", 376, 589)

            x= np.arange(data.shape[0])
            mask = (x>1)*(dq_map==0)
            self.plot_line = pg.PlotCurveItem(x, data,pen=pg.mkPen(color='lightpink', width=1))
            #self.plot_scatters = pg.ScatterPlotItem(x[mask], daiffs2err[mask],symbol = 'o',size=20,brush='r')
            self.plot_ebars = pg.ErrorBarItem(x=np.asarray(x)[mask], y=error[mask], top=1, bottom=1, beam=0.5)
            self.plot_col_lines = []
            colors= ['gray','gray','yellow','gray','blue','purple','pink']
            delta_y,delta_x = 0.,0
            for i,cols in enumerate(np.arange(col-2,col+2)):
                pen_spec = pg.mkPen(color=colors[i], width=1)
                data = self.parent.stage2.data.data[:, cols]
                win = signal.windows.hann(10)
                filtered = signal.convolve(data, win, mode='same') / sum(win)
                self.plot_col_lines.append(pg.PlotCurveItem(x+delta_x*(cols-col), filtered+ delta_y*(cols-col),pen=pen_spec))

            #mask_CR = dq_map == dqflags.pixel['JUMP_DET']
            #if np.sum(mask_CR>0):
            #self.plot_scatters_CRs = pg.ScatterPlotItem(x[mask_CR], diffs2err[mask_CR], symbol='o', size=20, brush='b')

            self.vb.addItem(self.plot_line)
            #self.vb.addItem(self.plot_scatters)
            #self.vb.addItem(self.plot_scatters_CRs)
            #self.vb.addItem(self.plot_ebars)
            self.vb.setLimits(xMin=-2, xMax=x[-1]+2)

            pen = pg.mkPen(color='darkgray', style=Qt.PenStyle.DashLine, width=1)
            self.zero_level = pg.PlotCurveItem([-2, x[-1] + 2], [0, 0], pen=pen)
            self.vb.addItem(self.zero_level)


            for l in self.plot_col_lines:
                self.vb.addItem(l)

            self.vb.autoRange()
            self.vb.setLimits(xMin=-2, xMax=x[-1] + 2)

            text = 'Col: '
            if col is not None:
                text += ' {0:.0f}'.format(col)
            self.legend_model.addItem(self.plot_line, text)
        else:
            try:
                self.vb.removeItem(self.plot_line)
                self.vb.removeItem(self.plot_ebars)
                self.vb.removeItem(self.zero_level)
                self.legend_model.removeItem(self.plot_line)
                for l in self.plot_col_lines:
                    self.vb.removeItem(l)


            except:
                pass

class EXPlistTable(pg.TableWidget):
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
        self.current_pipeline_stage = 'stage1'
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
        #pyqt5
        #self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        #self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        #pyqt6
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.ResizeToContents)
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
        if mode == 'rate':
            print('show rate, name', self.parent.parent.current_name)
            # self.parent.parent.plot_image.add(name=self.parent.parent.current_name, add=False)
            self.parent.parent.plot_image.add(name=self.parent.parent.current_name, add=True, mode='rate')

        if mode == 'stage2':
            print('show rate (stage2), name', self.parent.parent.current_name)
            # self.parent.parent.plot_image.add(name=self.parent.parent.current_name, add=False)
            self.parent.parent.plot_image.add(name=self.parent.parent.current_name, add=True, mode='stage2')

    def save_image(self,mode='stage2',save=False):
        if save == True:
            if mode == 'stage2':
                input = self.parent.parent.stage2.data
                from jwst.pipeline import calwebb_spec2
                miri = calwebb_spec2.Spec2Pipeline()
                miri.output_dir = './output/tmp/'
                miri.output_file = self.parent.parent.stage2.name
                miri.save_model(input, 'rate',force=True)
            if mode == 'bkgr_sub':
                input = self.parent.parent.stage2.data
                from jwst.pipeline import calwebb_spec2
                miri = calwebb_spec2.Spec2Pipeline()
                miri.output_dir = './output/detector2/bkgr_subtracted/'
                miri.output_file = self.parent.parent.stage2.name
                miri.save_model(input, 'bkgr_sub',force=True)
            if mode == 'masked_qso':
                input = self.parent.parent.stage2.data
                from jwst.pipeline import calwebb_spec2
                miri = calwebb_spec2.Spec2Pipeline()
                miri.output_dir = './output/detector2/masked_qso/'
                miri.output_file = self.parent.parent.stage2.name
                miri.save_model(input, 'masked_qso', force=True)


    def calc_ExoTiC_models(self):
        print('calc gain model')
        self.parent.parent.EXP.calc_gain_model()
        #print('calc linearity model')
        #self.parent.parent.EXP.calc_linearity_model()

    def set_dq(self):
        print('set data quality map')
        save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        self.parent.parent.EXP.dq_init_step(save_results=bool(save_res_flag))
        #print('calc gain model (Bell 2023). Set gain = 3.1 instead of 5.5 el/Dn')
        #self.parent.parent.EXP.calc_gain_model()
        #self.parent.parent.EXP.dq_init_step()

    def show_dq(self):
        print('set data quality map')
        flag = self.parent.parent.exp_commands.dq_categories.currentText()
        if self.current_pipeline_stage == 'stage1':
            self.parent.parent.EXP.show_dq(flag=flag)
        elif  self.current_pipeline_stage == 'stage2':
            self.parent.parent.stage2.show_dq(flag=flag)
    def check_saturation(self):
        if self.flags['saturation_step'] == False:
            flag = self.parent.parent.exp_pars.addneighbors.isChecked()
            debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
            save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
            print('Run Saturation step: add_neighbors=',flag, ' show_debug = ',debug)
            self.parent.parent.EXP.saturation_step(n_pix_grow_sat=flag,debug=debug,save_results=bool(save_res_flag))
            self.flags['saturation_step'] = True
            print('SATURATION STEP: DONE')
        else:
            flag = self.parent.parent.exp_pars.addneighbors.isChecked()
            debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
            save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
            print('Run second saturation step: add_neighbors=', flag, ' show_debug = ', debug)
            self.parent.parent.EXP.reset_dq(flagname='SATURATED')
            self.parent.parent.EXP.saturation_step(n_pix_grow_sat=flag, debug=debug,save_results=bool(save_res_flag))
            self.flags['saturation_step'] = True
            print('SATURATION STEP: DONE')
        if debug:
            plt.show()



    def show_saturation(self):
        if self.flags['saturation_step'] == True:
            if self.flags['show_saturated'] == False:
                self.flags['show_saturated'] = True
                mask = np.where(np.bitwise_and(self.parent.parent.EXP.data.pixeldq, dqflags.pixel['SATURATED']))
                self.parent.parent.plot_image.selectPixels(add=self.flags['show_saturated'], x=mask[1], y=mask[0],type='saturated')
                #for i, j in zip(mask[0], mask[1]):
                #    self.parent.parent.plot_image.selectPixel(add=self.flags['show_saturated'], x=j, y=i)
            else:
                self.flags['show_saturated'] = False
                self.parent.parent.plot_image.selectPixels(add=False,type='saturated')

    def show_DNU(self):
            if self.flags['show_dnu'] == False:
                self.flags['show_dnu'] = True
                mask = np.where(np.bitwise_and(self.parent.parent.EXP.data.pixeldq, dqflags.pixel['DO_NOT_USE']))
                self.parent.parent.plot_image.selectPixels(add=self.flags['show_dnu'], x=mask[1], y=mask[0],color='orange',
                                                           type='dnu')
                # for i, j in zip(mask[0], mask[1]):
                #    self.parent.parent.plot_image.selectPixel(add=self.flags['show_saturated'], x=j, y=i)
            else:
                self.flags['show_dnu'] = False
                self.parent.parent.plot_image.selectPixels(add=False, type='dnu')


    def first_group(self):
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        #save_res_flag = False
        self.parent.parent.EXP.first_step(debug=debug,save_results=bool(save_res_flag))
        print('FIRST STEP: DONE')
    def last_group(self):
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        #save_res_flag = False
        self.parent.parent.EXP.last_step(debug=debug,save_results=bool(save_res_flag))
        print('LAST STEP: DONE')

    def exotic_drop_groups(self):
        self.parent.parent.EXP.drop_first_steps(mode = int(self.parent.parent.exp_pars.n_group_dropped.text()))
        print('DROP GROUPS STEP by ExoTiC pipeline')

    def reset_correction(self):
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        self.parent.parent.EXP.reset_step(debug=debug,save_results=bool(save_res_flag))
        print('RESET STEP: DONE')

    def linear_correction(self):
        if self.parent.parent.EXP.data.meta.cal_step.linearity == 'COMPLETE':
            print('Linearity correction was applied already')
        else:
            save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
            debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
            override_linearity_model = False #bool(int(self.parent.parent.exp_pars.ExoTiC_mode.currentIndex()))
            self.parent.parent.EXP.linear_step(debug=debug,save_results=bool(save_res_flag), override_linearity=override_linearity_model)
            if override_linearity_model:
                print('LINEAR STEP by ExoTiC pipeline: DONE')
            else:
                print('STANDARD LINEAR STEP: DONE')


    def show_linear_correction(self):
        if self.parent.parent.EXP.data.meta.cal_step.linearity == 'COMPLETE':
            print('Linear correction was completed already')
        else:
            print('show_linear_correction: not ready')

    def rscd_correction(self):
        if self.parent.parent.EXP.data.meta.cal_step.rscd == 'COMPLETE':
            print('RSCD correction was applied already')
        else:
            save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
            debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
            self.parent.parent.EXP.rscd_step(debug=debug,save_results=bool(save_res_flag))
            print('RSCD STEP: DONE')

    def dark_correction(self):
        if self.parent.parent.EXP.data.meta.cal_step.dark_sub == 'COMPLETE':
            print('Dark subtraction was applied already')
        else:
            save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
            debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
            self.parent.parent.EXP.dark_step(debug=debug,save_results=bool(save_res_flag))
            print('DARK STEP: DONE')

    def reference_pix_correction(self):
        if self.parent.parent.EXP.data.meta.cal_step.refpix == 'COMPLETE':
            print('RefPix correction was applied already')
        else:
            save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
            debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
            self.parent.parent.EXP.refpix_corr_step(debug=debug,save_results=bool(save_res_flag))
            print('REF PIX STEP: DONE')

    def jump_detection(self):

        if self.flags['CR_step'] == False:
            CRlimit = float(self.parent.parent.exp_pars.CRlimit.text())
            RecalcMedian = int(self.parent.parent.exp_pars.CR_recalc_flag.currentIndex())
            flag_4neighbors = bool(self.parent.parent.exp_pars.addCRneighbors.currentIndex())
            find_showers = bool(self.parent.parent.exp_pars.find_CRshowers.currentIndex())
            debug = False #self.parent.parent.exp_pars.debug.isChecked()
            save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
            override_gain = bool(int(self.parent.parent.exp_pars.ExoTiC_mode.currentIndex()))
            print('Run CR step: add_neighbors=', flag_4neighbors, ' show_debug = ', debug)
            self.parent.parent.EXP.jump_corr_step(debug=debug, limit=CRlimit, flag_4_neighbors=flag_4neighbors,RecalcMedian=RecalcMedian,
                                                  save_results=bool(save_res_flag), override_gain=override_gain,find_showers=find_showers)
            self.flags['CR_step'] = True
            print('CR STEP: DONE')
        else:
            CRlimit = float(self.parent.parent.exp_pars.CRlimit.text())
            flag_4neighbors = bool(self.parent.parent.exp_pars.addCRneighbors.currentIndex())
            find_showers = bool(self.parent.parent.exp_pars.find_CRshowers.currentIndex())
            debug = False #self.parent.parent.exp_pars.debug.isChecked()
            save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
            override_gain = bool(int(self.parent.parent.exp_pars.ExoTiC_mode.currentIndex()))
            print('Run second CR step: add_neighbors=', flag_4neighbors, ' show_debug = ', debug)
            self.parent.parent.EXP.reset_dq(flagname='JUMP_DET')
            self.parent.parent.EXP.jump_corr_step(debug=debug, limit=CRlimit, flag_4_neighbors=flag_4neighbors,save_results=bool(save_res_flag),
                                                  override_gain=override_gain,find_showers=find_showers)
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
                if 1:
                    nint = int(self.parent.parent.exp_pars.nINT.text())-1
                    if hasattr(self.parent.parent.EXP,'int_pixeldq'):
                        pdq =self.parent.parent.EXP.int_pixeldq[nint]
                    elif self.parent.parent.Exposures.table.current_pipeline_stage == 'stage2':
                        pdq =self.parent.parent.stage2.data.dq
                    else:
                        pdq = self.parent.parent.EXP.data.pixeldq
                    mask = np.where(np.bitwise_and(pdq, dqflags.pixel['JUMP_DET']))
                else:
                    nint = int(self.parent.parent.exp_pars.nINT.text())-1
                    mask = np.zeros((self.parent.parent.EXP.nrows,self.parent.parent.EXP.ncols)).astype(int)
                    for i_gr in range(self.parent.parent.EXP.ngroup):
                        mask = np.bitwise_or(self.parent.parent.EXP.data.groupdq[nint,i_gr],mask)
                    mask = np.where(np.bitwise_and(mask, dqflags.pixel['JUMP_DET']))
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
                nint = int(self.parent.parent.exp_pars.nINT.text())-1
                if hasattr(self.parent.parent.EXP, 'int_pixeldq'):
                    pdq = self.parent.parent.EXP.int_pixeldq[nint]
                elif self.parent.parent.Exposures.table.current_pipeline_stage == 'stage2':
                    pdq = self.parent.parent.stage2.data.dq
                else:
                    pdq = self.parent.parent.EXP.data.pixeldq
                mask = np.where(np.bitwise_and(pdq, dqflags.pixel['JUMP_DET']))
                x,y = [],[]
                for i in range(mask[0].shape[0]):
                    mask_tmp = np.bitwise_and(self.parent.parent.EXP.data.groupdq[0,:,mask[0][i],mask[1][i]], dqflags.pixel['JUMP_DET'])
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

    def slope_fit(self,debug=False,algorithm='CHI2'):
        algorithm = self.parent.parent.exp_pars.FitAlgorithm_mode.currentText()
        #CRlimit = float(self.parent.parent.exp_pars.CRlimit.text())
        #flag = self.parent.parent.exp_pars.addCRneighbors.isChecked()
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        override_gain = bool(int(self.parent.parent.exp_pars.ExoTiC_mode.currentIndex()))
        print('Run Slope Fit step: show_debug = ', debug, 'save result =  ', bool(save_res_flag))
        if algorithm == 'CHI2':
            save_res_flag = True
        self.parent.parent.EXP.slope_fitting_step(debug=debug,save_results=bool(save_res_flag),
                                                  override_gain=override_gain,algorithm=algorithm)
        self.flags['slope_fit_step'] = True
        if algorithm == 'CHI2':
            self.parent.parent.EXP.data.pixeldq = self.parent.parent.EXP.ramp_fit[0].dq
            self.flags['CR_step'] = True
            self.parent.parent.EXP.data.meta.cal_step.jump = 'COMPLETE'
        print('Slope Fit STEP: DONE')

        if debug:
            plt.show()

    def calc_mean_rate(self,debug=False,radius=10,hot_pix_limit=4):
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        algorithm = self.parent.parent.exp_pars.FitAlgorithm_mode.currentText()
        if algorithm == 'CHI2':
            im = self.parent.parent.EXP.int_slopes[:,0].copy()
            sigim = self.parent.parent.EXP.int_sigslopes[:,0].copy()
            #dqim = self.parent.parent.EXP.int_pixeldq[:,0].copy()
            dqim = self.parent.parent.EXP.int_pixeldq.copy()
            n_int = im.shape[0]
            mask_im = np.ones((n_int,im.shape[1],im.shape[2]))
            mean_im = np.mean(im,axis=0)

            #apply sigma clipping averaging for slope images in intgrations
            if n_int>1:
                if 1:
                    # find path to photom mask
                    from scripts.flat_field import get_mask
                    band = self.parent.parent.EXP.data.meta.instrument.band
                    channel = self.parent.parent.EXP.data.meta.instrument.channel
                    photom_list = sorted(glob.glob(os.environ["CRDS_PATH"] + '/references/jwst/miri/*photom*'))
                    for f in photom_list:
                        hdulist = fits.open(f)
                        header = hdulist[0].header
                        f_band, f_ch = header['BAND'], header['CHANNEl']
                        if band == f_band and f_ch == channel:
                            photom_file = f
                            break
                    photom_mask = get_mask(path=photom_file)


                from scripts.CRshowers import calc_mean_rate

                imtot, imtotsig = calc_mean_rate(images=im, sig_images=sigim, dqs=dqim,debug=debug,
                                                 skip_cr_events=True, radius=radius,
                                                 photom_mask=photom_mask,n_smooth_iters = 3)

                if debug:
                    n_im = len(im)
                    fig, ax = plt.subplots(1, n_im + 1, sharex=True, sharey=True)
                    vmin, vmax = np.nanquantile(imtot.flatten(), 0.05), np.nanquantile(imtot.flatten(), 0.8)
                    for i in range(n_im):
                        ax[i].imshow(im[i], vmin=vmin, vmax=vmax)
                        ax[i].set_title(str(i))
                    ax[n_im].imshow(imtot, vmin=vmin, vmax=vmax)
                    ax[n_im].set_title('MODEL')
                    plt.show()
            else:
                imtot = np.nanmedian(im, axis=0)
                imsig_inv = np.power(sigim, -2)
                imtotsig = np.power(np.nansum(imsig_inv, axis=0), -0.5)


            self.parent.parent.EXP.mean_slope = imtot
            self.parent.parent.EXP.mean_slope_sig = imtotsig

            ramp_fit = self.parent.parent.EXP.ramp_fit
            group_time = ramp_fit[0].meta.exposure.group_time
            ramp_fit[0].data = imtot/group_time
            ramp_fit[0].err = imtotsig/group_time

        #elif algorithm == 'OLS_C':
            #ramp_fit = self.parent.parent.EXP.ramp_fit
            #group_time = ramp_fit[0].meta.exposure.group_time
            #ramp_fit.data = ramp_fit[0].data
            #ramp_fit.err = ramp_fit[0].err

    def read_slopes(self,input_file = None,version='1.14'):
        print('Read Slopes:')
        group_time = self.parent.parent.EXP.data.meta.exposure.group_time
        algorithm = self.parent.parent.exp_pars.FitAlgorithm_mode.currentText()
        if self.flags['slope_fit_step'] == True and algorithm == 'OLS_C':
            ramp_fit = self.parent.parent.EXP.ramp_fit
            #ramp_info = self.parent.parent.EXP.ramp_fit_info
            data = self.parent.parent.EXP.data.data
            dataerr = np.zeros_like(data)
            #set derived slopes for integrations (for visual use only)
            self.parent.parent.EXP.int_slopes = ramp_fit[1].data  * group_time  # in DN/groups
            self.parent.parent.EXP.int_sigslopes = ramp_fit[1].err * group_time  # in DN/groups
            self.parent.parent.EXP.int_pixeldq = ramp_fit[1].dq
            #set derived slopes for ramp (rate files) (for visual use only)
            self.parent.parent.EXP.mean_slope = ramp_fit[0].data * group_time  # in DN/groups
            self.parent.parent.EXP.mean_slope_sig = ramp_fit[0].err * group_time  # in DN/groups
            #if 1:
            #    first_gr, last_gr = self.parent.parent.EXP.drop_ngroups, self.parent.parent.EXP.data.groupdq.shape[1] - 1
            #self.parent.parent.EXP.data.groupdq[:, first_gr:last_gr, :, :] = ramp_info[7]
            #self.parent.parent.EXP.int_jumps = np.zeros_like(ramp_info.slope)
            self.flags['read_fit_slopes'] = True

            if 0:
                gdq = ramp_info[7]
                nrows, ncols = ramp_fit[0].data.shape[0], ramp_fit[0].data.shape[1]
                self.parent.parent.EXP.itercepts = np.zeros((1, 1, nrows, ncols))
                self.parent.parent.EXP.itercepts_err = np.zeros((1, 1, nrows, ncols))
                for i in range(ramp_fit[0].data.shape[0]):
                    for j in range(ramp_fit[0].data.shape[1]):
                        # print(i,j,gdq[0, :, i, j])
                        if 0 in gdq[0, :, i, j]:
                            i_gr = np.where(gdq[0, :, i, j] == 0)[0][0]
                            self.parent.parent.EXP.itercepts[0, 0, i, j] = data[0, i_gr, i, j]
                            self.parent.parent.EXP.itercepts_err[0, 0, i, j] = dataerr[0, i_gr, i, j]
                        else:
                            self.parent.parent.EXP.itercepts[0, 0, i, j] = 0
                            self.parent.parent.EXP.itercepts_err[0, 0, i, j] = np.inf

        elif self.flags['slope_fit_step'] == True and algorithm == 'OLS_C' and 0:
            print('Read Slope Fits from local files:')
            ramp_fit = self.parent.parent.EXP.ramp_fit
            if input_file == None:
                input_file = self.parent.parent.EXP.input_file
            input_file_base = os.path.basename(input_file).replace('uncal.fits', '')
            output_dir = self.parent.parent.EXP.output_dir
            rampfit_output_file = os.path.join(output_dir, '{}_0_rampfitstep.fits'.format(input_file_base))

            # Generate the name of the optional output file
            optional_file = os.path.join(output_dir, '{}fitopt.fits'.format(input_file_base))
            print('optional_file',optional_file)
            hdulist = fits.open(optional_file)
            intercepts = hdulist['YINT'].data[:, :, :, :]
            intercepts_err = hdulist['SIGYINT'].data[:, :, :, :]
            local_slopes = hdulist['SLOPE'].data[:, :, :, :]
            local_sig_slopes = hdulist['SIGSLOPE'].data[:, :, :, :]
            hdulist.close()
            sci_file = os.path.join('./output/results/', '{}rate.fits'.format(input_file_base))
            print('sci_file',sci_file)
            #hdulist = fits.open(sci_file)
            #sci_arr =hdulist['SCI'].data[:, :]
            #hdulist.close()
            #num_groups = ramp_fit[0].meta.exposure.ngroups
            if version == '1.14':
                group_time = ramp_fit.meta.exposure.group_time
            else:
                group_time = ramp_fit[0].meta.exposure.group_time
            print('Time per group:', group_time, ' in s')
            #group_times = np.arange(num_groups) * group_time


            self.parent.parent.EXP.itercepts = intercepts
            self.parent.parent.EXP.itercepts_err = intercepts_err
            if version == '1.14':
                self.parent.parent.EXP.slopes = ramp_fit.data * group_time  # in DN/groups
                self.parent.parent.EXP.slopes_err = ramp_fit.err * group_time  # in DN/groups
            else:
                self.parent.parent.EXP.slopes = ramp_fit[0].data* group_time  #in DN/groups
                self.parent.parent.EXP.slopes_err = ramp_fit[0].err* group_time #in DN/groups
            self.parent.parent.EXP.local_slopes = local_slopes * group_time  # in DN/groups
            self.parent.parent.EXP.local_slopes_err = local_sig_slopes * group_time  # in DN/groups
            self.flags['read_fit_slopes'] = True
        elif self.flags['slope_fit_step'] == True and algorithm == 'CHI2':
            print('Read Slope Fits from local files:')

            ramp_fit = self.parent.parent.EXP.ramp_fit
            ramp_info = self.parent.parent.EXP.ramp_fit_info
            data = self.parent.parent.EXP.data.data
            dataerr = self.parent.parent.EXP.data.err
            self.parent.parent.EXP.int_slopes = ramp_info[0]*group_time
            self.parent.parent.EXP.int_sigslopes = ramp_info[1] * group_time
            print('read pixel dq from ramp fit')
            self.parent.parent.EXP.int_pixeldq = ramp_fit[1].dq
            if 1:
                first_gr,last_gr = self.parent.parent.EXP.drop_ngroups,self.parent.parent.EXP.data.groupdq.shape[1]-1
            self.parent.parent.EXP.data.groupdq[:,first_gr:last_gr,:,:] = ramp_info[7]
            self.parent.parent.EXP.int_jumps = ramp_info[8]
            self.parent.parent.EXP.slopes = ramp_fit[0].data * group_time  # in DN/groups
            self.parent.parent.EXP.slopes_err = ramp_fit[0].err * group_time  # in DN/groups

            self.flags['read_fit_slopes'] = True

            if 1:
                gdq = ramp_info[7]
                nrows,ncols = ramp_fit[0].data.shape[0],ramp_fit[0].data.shape[1]
                self.parent.parent.EXP.itercepts = np.zeros((1,1,nrows,ncols))
                self.parent.parent.EXP.itercepts_err = np.zeros((1,1,nrows,ncols))
                for i in range(ramp_fit[0].data.shape[0]):
                    for j in range(ramp_fit[0].data.shape[1]):
                        #print(i,j,gdq[0, :, i, j])
                        if 0 in gdq[0,:,i,j]:
                            i_gr = np.where(gdq[0,:,i,j] == 0)[0][0]
                            self.parent.parent.EXP.itercepts[0,0,i,j] = data[0,i_gr,i,j]
                            self.parent.parent.EXP.itercepts_err[0,0,i,j] = dataerr[0,i_gr,i,j]
                        else:
                            self.parent.parent.EXP.itercepts[0,0,i, j] = 0
                            self.parent.parent.EXP.itercepts_err[0,0,i, j] = np.inf
        else:
            print('Read Slope Fits from Final files')
            if input_file == None:
                input_file = self.parent.parent.EXP.input_file
            input_file_base = os.path.basename(input_file).replace('uncal.fits', '')
            output_dir = './output/results/' #self.parent.parent.EXP.output_dir
            #output_dir = './output/detector1/'  # self.parent.parent.EXP.output_dir

            files = os.listdir(output_dir)
            if np.sum(input_file_base in f for f in files)>0:
                # Generate the name of the optional output file
                int_file = os.path.join(output_dir, '{}rateints.fits'.format(input_file_base))
                hdulist = fits.open(int_file)
                int_slopes = hdulist['SCI'].data
                int_sig_slopes = hdulist['ERR'].data
                int_dq = hdulist['DQ'].data
                int_header = hdulist[0].header
                hdulist.close()


                data_shape = self.parent.parent.EXP.data.data.shape
                self.parent.parent.EXP.int_slopes = np.zeros((data_shape[0],1,data_shape[2],data_shape[3]))
                self.parent.parent.EXP.int_sigslopes= np.zeros((data_shape[0],1,data_shape[2],data_shape[3]))

                self.parent.parent.EXP.int_slopes[:,0,:,:] = int_slopes * group_time
                self.parent.parent.EXP.int_sigslopes[:,0,:,:] = int_sig_slopes * group_time
                self.parent.parent.EXP.int_pixeldq = int_dq
                self.parent.parent.EXP.int_jumps = np.zeros_like(self.parent.parent.EXP.data.data)

                if 1:
                    rate_file = os.path.join('./output/results/', '{}rate.fits'.format(input_file_base))
                    rate_ints_file = os.path.join('./output/results/', '{}rateints.fits'.format(input_file_base))

                    from stdatamodels.jwst import datamodels
                    miri_rate = datamodels.open(rate_file)
                    miri_rateints = datamodels.open(rate_ints_file)
                    self.parent.parent.EXP.ramp_fit = miri_rate,miri_rateints

                #rate_file = os.path.join('./output/results/', '{}rate.fits'.format(input_file_base))
                #hdulist = fits.open(rate_file)
                #rate = hdulist['SCI'].data
                #rate_err = hdulist['ERR'].data
                #input, ints_model = self.parent.parent.EXP.ramp_fit

                self.parent.parent.EXP.mean_slope = miri_rate.data
                self.parent.parent.EXP.mean_slope_sig = miri_rate.err
                print(int_header.keys())
                self.parent.parent.EXP.data.meta.cal_step.dark_sub = int_header['S_DARK']
                self.parent.parent.EXP.data.meta.cal_step.dq_init = int_header['S_DQINIT']
                if 'S_FRSTFR' in int_header.keys():
                    self.parent.parent.EXP.data.meta.cal_step.firstframe = int_header['S_FRSTFR']
                if 'S_JUMP' in int_header.keys():
                    self.parent.parent.EXP.data.meta.cal_step.jump = int_header['S_JUMP']
                else:
                    self.parent.parent.EXP.data.meta.cal_step.jump = 'None'
                self.parent.parent.EXP.data.meta.cal_step.lastframe = int_header['S_LASTFR']
                self.parent.parent.EXP.data.meta.cal_step.linearity = int_header['S_LINEAR']
                self.parent.parent.EXP.data.meta.cal_step.ramp_fit = int_header['S_RAMP']
                self.parent.parent.EXP.data.meta.cal_step.refpix = int_header['S_REFPIX']
                self.parent.parent.EXP.data.meta.cal_step.reset = int_header['S_RESET']
                self.parent.parent.EXP.data.meta.cal_step.rscd = int_header['S_RSCD']
                self.parent.parent.EXP.data.meta.cal_step.saturation = int_header['S_SATURA']

                self.flags=self.flags.fromkeys(self.flags, True)
                self.flags['show_saturated'] = False
                self.flags['show_CR'] = False
                self.flags['show_multi_CR'] = False
                self.flags['show_dnu'] = False
                flags_dict = {}
                flags_dict['COMPLETE'] = True
                flags_dict['SKIPPED'] = False
                flags_dict['None'] = False
                self.flags['saturation_step'] = flags_dict[self.parent.parent.EXP.data.meta.cal_step.saturation]
                self.flags['CR_step'] = True #flags_dict[self.parent.parent.EXP.data.meta.cal_step.jump]
                self.flags['slope_fit_step'] = flags_dict[self.parent.parent.EXP.data.meta.cal_step.ramp_fit]


            else:
                print('There is no saved files')

    def show_slope_image(self):
        if self.flags['slope_fit_step'] == True:
            ramp_fit = self.parent.parent.EXP.ramp_fit
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

    def save_slope_fit(self,output_dir=None,save_other_data=False):
        if self.flags['slope_fit_step'] == True:
            from jwst.pipeline import calwebb_detector1
            miri1 = calwebb_detector1.Detector1Pipeline()
            if output_dir == 'local':
                miri1.output_dir = self.parent.parent.EXP.output_dir
                print('save slopes for ramp and ints to local folder')
            elif output_dir == 'final':
                miri1.output_dir = './output/results/'
                print('save slopes for ramp and ints to _results_ folder')
            else:
                output_dir = None
            if output_dir != None:
                miri1.output_file =  self.parent.parent.EXP.name
                input, ints_model = self.parent.parent.EXP.ramp_fit
                if ints_model is not None:
                    miri1.save_model(ints_model, 'rateints')
                if input is not None:
                    miri1.save_model(input, 'rate')
            if 1:
                #save cal steps
                s = self.parent.parent.EXP.data.meta.cal_step
                lst = []
                for l in [s.linearity, s.rscd, s.dark_sub, s.refpix]:
                    if l != None:
                        lst.append(l)
                    else:
                        lst.append(2)
                print(lst)
                origin = self.parent.parent.EXP.output_dir
                target = miri1.output_dir
                # Fetching the list of all the files
                files = os.listdir(origin)
                # Fetching all the files to directory
                exp_name = self.parent.parent.EXP.name.split('_uncal')[0]
                with open(target + exp_name + '_cal_steps.csw', 'w') as f:
                    csv_writer = csv.writer(f, delimiter=',')
                    csv_writer.writerows([lst])
            if output_dir == 'final' and save_other_data==True:
                # Providing the folder path
                origin = self.parent.parent.EXP.output_dir
                target = miri1.output_dir

                # Fetching the list of all the files
                files = os.listdir(origin)

                # Fetching all the files to directory
                exp_name = self.parent.parent.EXP.name.split('_uncal')[0]
                for file_name in files:
                    if exp_name in file_name:
                        shutil.copy(origin + file_name, target + file_name)
                print("Fit slopes are saved to", target+exp_name)


    #********************************* Stage2 pipeline analysis part

    def init_stage2(self):
        print('Init_Stage2')

        path = self.parent.parent.stage2.path
        exp_name = self.parent.parent.EXP.name
        output2_dir,spec2_cachedir = self.parent.parent.stage2.output_dir, self.parent.parent.stage2.spec2_cachedir
        self.parent.parent.stage2.__init__(miri_uncal_file=exp_name, path=path, output_dir=output2_dir,spec2_cachedir=spec2_cachedir)
        if self.parent.parent.stage2.data != None:
            self.current_pipeline_stage = 'stage2'
        if 1:
            flags_dict = {}
            flags_dict['COMPLETE'] = True
            flags_dict['SKIPPED'] = False
            flags_dict['None'] = False
            cal_step = self.parent.parent.stage2.data.meta.cal_step.instance
            self.flags['saturation_step'] = flags_dict[cal_step['saturation']]
            #self.flags['CR_step'] = flags_dict[cal_step['jump']]
            self.flags['slope_fit_step'] = flags_dict[cal_step['ramp_fit']]
            if 'jump' not in cal_step.keys():
                self.flags['CR_step'] = True
            else:
                self.flags['CR_step'] = flags_dict[cal_step['jump']]

        instrument = self.parent.parent.stage2.data.meta.instrument.detector
        return instrument

    def stage2_read_steps(self):
        print('stage2: Read results')
        #step_name = self.parent.parent.stage2_commands.read_step_choice.currentText()
        self.parent.parent.stage2.read_step_results_version2(step_name='cal')

    def stage2_read_step_res_fringe(self):
        print('stage2: Read results')
        #step_name = self.parent.parent.stage2_commands.read_step_choice.currentText()
        self.parent.parent.stage2.read_step_results_version2(step_name='res_fringe')

    def stage2_fix_hot_pix(self):
        self.parent.parent.stage2.select_hot_pix(output_dir='./data_local/Hot_pixels/ID02441/')
        self.parent.parent.stage2.fix_hot_pix_step(debug=0)
        self.parent.parent.stage2.fix_cold_pix_step(debug=0)

    def stage2_fix_hot_pix_version2(self):
        table = self.parent.parent.Exposures.table
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        #self.parent.parent.stage2.select_hot_pix_version2(filelist=table,debug=debug,fast_mode=False)
        self.parent.parent.stage2.select_hot_pix_version3(filelist=table, debug=debug, fast_mode=False)

    def stage2_call_wcs(self):
        print('Call_WCS')
        save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        self.parent.parent.stage2.assignwcsstep()

    def stage2_background_model(self):
        #print('Background Subtraction - Stage 2 - skipped')
        table = self.parent.parent.Exposures.table
        smothing_rad = int(self.parent.parent.exp_pars.cr_shower_r_correction.text())
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        save_results = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        bkgr, bkgr_sig = self.parent.parent.stage2.create_background_model(filelist=table,smothing_rad=smothing_rad,debug=debug)
        print('Subtract model')
        self.parent.parent.stage2.subtract_bkgr_model(bkgr_model=bkgr,bkgr_model_sig=bkgr_sig,savepdf=save_results)

        #self.parent.parent.stage2.background_subtraction()

    def stage2_compare_dithers_sample(self):
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        save_results = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        self.parent.parent.stage2.compare_dither_images(debug=debug,save_pdf=save_results)

    def stage2_flat_field(self):
        print('Flat Field Step - Stage 2')
        save_res_flag = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        self.parent.parent.stage2.flat_field_step(save_results=False,debug=debug)

    def stage2_source_identification(self):
        print('Source ID Step - Stage 2')
        self.parent.parent.stage2.source_type_identification()

    def stage2_stray_light(self):
        print('Source ID Step - Stage 2')
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        self.parent.parent.stage2.stray_light_step(debug=debug)

    def stage2_fringe_flat_correction(self):
        print('Source ID Step - Stage 2')
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        save_results = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        self.parent.parent.stage2.fringe_flat_step(debug=debug)

    def stage2_residual_fringe_correction(self):
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        save_results = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        self.parent.parent.stage2.res_fringe_step(debug=debug,save_results=bool(save_results))

    def stage2_read_flat_fringes(self):
        return self.parent.parent.stage2.read_flat_fringes()

    def stage2_read_res_fringes(self):
        return self.parent.parent.stage2.read_res_fringes()

    def stage2_read_bkgr_subtracted(self):
        return self.parent.parent.stage2.read_bkgr_subtracted()

    def stage2_flux_calibration(self):
        print('Source ID Step - Stage 2')
        save_results = int(self.parent.parent.exp_pars.save_tmp_res.currentIndex())
        debug = bool(self.parent.parent.exp_pars.debug.currentIndex())
        self.parent.parent.stage2.flux_calibration_step(save_results=bool(save_results),debug=debug)

    def stage2_show_trace(self,trace_order=0,trace_size=0):
        self.parent.parent.stage2.show_trace(trace_order=trace_order,trace_size=trace_size)

    def stage2_show_Xtrace(self, trace_Xpos=100,delta=10):
        self.parent.parent.stage2.show_Xtrace(trace_Xpos=100,delta=10)

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

        self.table = EXPlistTable(self)
        self.filelist = {}
        if 1:
            filenames,fileparams,codenames = self.readfolder(self.parent.EXP.path)
            lst = []
            for s,pars in zip(filenames,fileparams):
                d = [(s.split('/')[-1]).split('.')[0]]
                for p in pars:
                    d.append(p)
                lst.append(d)
                self.filelist[d[0]]=s
            lst = np.array([tuple(l) for l in lst], dtype=[('name','U400')] + [(p,'U50') for p in codenames])
            data = lst
        self.table.setdata(data)
        # highilite rows
        #self.table.setSelectionBehavior(QTableView.SelectRows);

        self.buttons = {}
        for i, d in enumerate(data):
            wdg = QWidget()
            l = QVBoxLayout()
            l.addSpacing(3)
            button = QPushButton(d[0].split('_uncal')[0], self, checkable=True)
            button.resize(600, 30)
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
        print('load files from folder:',folder)
        self.tablefiles = {}
        if 1:
            lst = []
            params= []
            for (dirpath, dirname, filenames) in os.walk(folder):
                print(dirpath, dirname, filenames)
                for k,f in enumerate(filenames):
                    if f.endswith('_uncal.fits'):
                        str = dirpath+'/'+f
                        hdu = fits.open(dirpath+'/'+f)
                        if hdu[0].header['DETECTOR'] != 'MIRIMAGE':
                            lst.append(dirpath.split('/')[-1]+'/'+f)
                            params.append(self.readfile(pathotofile=dirpath,filename=f))
                            self.tablefiles[f] = f
                            print(f,params[-1])
                        else:
                            print('ERROR: you tried to load the IMAGE.fits file')
        codenames = ['PROGRAM','OBSERUN','TARGPROP','BAND','CHANNEL']
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
        return [prog_id,obs_id,targ_name,miri_band,miri_channel]




    def click(self, name):
        self.parent.current_name = name
        self.parent.plot_image.add(name, self.buttons[name].isChecked())
        print('self.buttons[name].isChecked()',self.buttons[name].isChecked())
        if self.buttons[name].isChecked()==True:
            flags = self.parent.Exposures.table.flags
            self.parent.Exposures.table.flags = flags.fromkeys(flags, False)
            self.parent.Exposures.table.current_pipeline_stage = 'stage1'
            print('added by click', self.parent.Exposures.table.flags)
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
        l.addWidget(QLabel('Integration:'))
        self.nINT = QLineEdit()
        self.nINT.setText(str(1))
        cb = self.nINT
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        #self.nINT.resize(90, 30)
        l.addWidget(self.nINT)
        nintmax = 'XX'
        if 1:
            if hasattr(self.parent.EXP.data,'data'):
                nintmax = self.parent.EXP.nint
            else:
                name = self.parent.Exposures.filelist.keys()
                for key in name:
                    name=key
                    break
                fname = self.parent.Exposures.filelist[name]
                hdu = fits.open(self.parent.EXP.path+'/'+fname)
                nintmax =hdu['SCI'].data.shape[0]
                hdu.close()
        l.addWidget(QLabel(' of '+str(nintmax)))

        l.addWidget(QLabel('Group:'))
        self.nGROUP = QLineEdit()
        self.nGROUP.setText(str(1))
        cb = self.nGROUP
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        #self.nGROUP.resize(90, 30)
        l.addWidget(self.nGROUP)
        l.addStretch(1)
        layout.addLayout(l)

        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('SatPix:'))
        self.addneighbors = QCheckBox('4NeighPix')
        self.addneighbors.setChecked(True)
        horizontal_layout.addWidget(self.addneighbors)
        horizontal_layout.addWidget(QLabel('Debug:'))
        self.debug = QComboBox()
        self.debug.addItems(['No', 'Yes'])
        self.debug.setCurrentIndex(0)
        cb = self.debug
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
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
        cb = self.CRlimit
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        #self.CRlimit.resize(90, 30)
        horizontal_layout.addWidget(self.CRlimit)
        #self.addCRneighbors = QCheckBox('4NeighPix', checkable=True)
        #self.addCRneighbors.setChecked(True)
        #horizontal_layout.addWidget(self.addCRneighbors)
        horizontal_layout.addWidget(QLabel('4neighbors:'))
        self.addCRneighbors = QComboBox()
        self.addCRneighbors.addItems(['No','Yes'])
        self.addCRneighbors.setCurrentIndex(1)
        cb = self.addCRneighbors
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        horizontal_layout.addWidget(self.addCRneighbors)
        horizontal_layout.addWidget(QLabel('showers:'))
        self.find_CRshowers = QComboBox()
        self.find_CRshowers.addItems(['No', 'Yes'])
        self.find_CRshowers.setCurrentIndex(1)
        cb = self.find_CRshowers
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        horizontal_layout.addWidget(self.find_CRshowers)



        horizontal_layout.addWidget(QLabel('VaryMed:'))
        self.CR_recalc_flag = QComboBox()
        self.CR_recalc_flag.addItems(['No','Yes'])
        self.CR_recalc_flag.setCurrentIndex(1)
        cb = self.CR_recalc_flag
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        #self.CR_recalc_flag.resize(90, 30)
        horizontal_layout.addWidget(self.CR_recalc_flag)
        horizontal_layout.addWidget(QLabel('SaveRes:'))
        self.save_tmp_res = QComboBox()
        self.save_tmp_res.addItems(['No','Yes'])
        self.save_tmp_res.setCurrentIndex(0)
        cb = self.save_tmp_res
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        #self.save_tmp_res.resize(90, 30)
        horizontal_layout.addWidget(self.save_tmp_res)

        horizontal_layout.addStretch(1)
        layout.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('Custom:'))
        horizontal_layout.addWidget(QLabel('ExoTiC:'))
        self.ExoTiC_mode = QComboBox()
        self.ExoTiC_mode.addItems(['No', 'Yes'])
        self.ExoTiC_mode.setCurrentIndex(0)
        cb = self.ExoTiC_mode
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        horizontal_layout.addWidget(self.ExoTiC_mode)
        horizontal_layout.addWidget(QLabel('Drop#gr:'))
        self.n_group_dropped = QLineEdit()
        self.n_group_dropped.setText(str(6))
        cb = self.n_group_dropped
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        horizontal_layout.addWidget(self.n_group_dropped)
        horizontal_layout.addWidget(QLabel('Fit:'))
        self.FitAlgorithm_mode = QComboBox()
        self.FitAlgorithm_mode.addItems(['OLS_C', 'LIKELY', 'CHI2'])
        self.FitAlgorithm_mode.setCurrentIndex(0)
        cb = self.FitAlgorithm_mode
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        horizontal_layout.addWidget(self.FitAlgorithm_mode)

        horizontal_layout.addWidget(QLabel('CRshoweR:'))
        self.cr_shower_r_correction = QLineEdit()
        self.cr_shower_r_correction.setText(str(10))
        cb = self.cr_shower_r_correction
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        horizontal_layout.addWidget(self.cr_shower_r_correction)

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
        cb = self.show_image
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        #self.show_image.resize(200, 60)
        horizontal_layout.addWidget(self.show_image)
        self.show_slope = QPushButton('Show Slope')
        self.show_slope.clicked[bool].connect(partial(self.show_Image,'slope'))
        cb = self.show_slope
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        #self.show_slope.resize(200, 60)
        horizontal_layout.addWidget(self.show_slope)

        self.show_rate = QPushButton('Show Rate')
        self.show_rate.clicked[bool].connect(partial(self.show_Image, 'rate'))
        cb = self.show_rate
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        # self.show_slope.resize(200, 60)
        horizontal_layout.addWidget(self.show_rate)

        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        self.set_dq_map = QPushButton('Init DQ')
        self.set_dq_map.clicked[bool].connect(partial(self.set_DQ_map, False))
        self.set_dq_map.resize(150, 60)
        horizontal_layout.addWidget(self.set_dq_map)
        self.show_dq = QPushButton('Show DQ')
        self.show_dq.clicked[bool].connect(partial(self.show_DQ_map, False))
        self.show_dq.resize(150, 60)
        horizontal_layout.addWidget(self.show_dq)
        self.dq_categories = QComboBox()
        flags = [*dqflags.pixel]
        self.dq_categories.addItems(['all']+flags)
        self.dq_categories.setCurrentIndex(2)
        cb = self.dq_categories
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        #self.dq_categories.resize(90, 30)
        horizontal_layout.addWidget(self.dq_categories)
        horizontal_layout.addWidget(QLabel('Show:'))
        self.show_linear_step = QPushButton('DNU')
        self.show_linear_step.clicked[bool].connect(partial(self.show_DNUpixels))
        self.show_linear_step.resize(70, 60)
        horizontal_layout.addWidget(self.show_linear_step)
        self.show_sat_pixels = QPushButton('SatPix')
        self.show_sat_pixels.clicked[bool].connect(partial(self.show_SATpixels))
        self.show_sat_pixels.resize(100, 60)
        horizontal_layout.addWidget(self.show_sat_pixels)
        self.show_single_jumps = QPushButton('CR')
        self.show_single_jumps.clicked[bool].connect(partial(self.ShowFirstCR, False))
        self.show_single_jumps.resize(70, 60)
        horizontal_layout.addWidget(self.show_single_jumps)
        self.show_sec_jumps = QPushButton('MultiCR')
        self.show_sec_jumps.clicked[bool].connect(partial(self.ShowSecondCR, False))
        self.show_sec_jumps.resize(100, 60)
        horizontal_layout.addWidget(self.show_sec_jumps)

        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        self.saturation = QPushButton('1.Saturation')
        self.saturation.clicked[bool].connect(partial(self.SaturationStep, False))
        self.saturation.resize(200, 60)
        horizontal_layout.addWidget(self.saturation)
        self.first_last_step = QPushButton('2.First&Last')
        self.first_last_step.clicked[bool].connect(partial(self.FirstLastStep, False))
        self.first_last_step.resize(150, 60)
        horizontal_layout.addWidget(self.first_last_step)
        self.reset_step = QPushButton('3.ResetAnomaly')
        self.reset_step.clicked[bool].connect(partial(self.ResetStep, False))
        self.reset_step.resize(250, 60)
        horizontal_layout.addWidget(self.reset_step)
        self.linear_step = QPushButton('4.LinearCorr')
        self.linear_step.clicked[bool].connect(partial(self.LinearStep, False))
        self.linear_step.resize(200, 60)
        horizontal_layout.addWidget(self.linear_step)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)


        horizontal_layout = QHBoxLayout(self)
        self.rscd_step = QPushButton('5.RSCD')
        self.rscd_step.clicked[bool].connect(partial(self.RSCDStep, False))
        self.rscd_step.resize(150, 60)
        horizontal_layout.addWidget(self.rscd_step)
        self.dark_step_win = QPushButton('6.DarkSubtract')
        self.dark_step_win.clicked[bool].connect(partial(self.DarkStep, False))
        self.dark_step_win.resize(200, 60)
        horizontal_layout.addWidget(self.dark_step_win)
        self.refpix_step = QPushButton('7.ReferencePix')
        self.refpix_step.clicked[bool].connect(partial(self.RefPixStep, False))
        self.refpix_step.resize(200, 60)
        horizontal_layout.addWidget(self.refpix_step)
        self.jump_step = QPushButton('8. Jump Detection')
        self.jump_step.clicked[bool].connect(partial(self.JumpStep,False))
        self.jump_step.resize(250, 60)
        horizontal_layout.addWidget(self.jump_step)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)
        layout.addLayout(l)

        horizontal_layout = QHBoxLayout(self)
        self.slope_fit_step = QPushButton('9. SlopeFit')
        self.slope_fit_step.clicked[bool].connect(partial(self.SlopeFitStep))
        self.slope_fit_step.resize(250, 60)
        horizontal_layout.addWidget(self.slope_fit_step)

        self.mean_rate_step = QPushButton('10. Rate')
        self.mean_rate_step.clicked[bool].connect(partial(self.GetMeanRate))
        self.mean_rate_step.resize(250, 60)
        horizontal_layout.addWidget(self.mean_rate_step)

        self.run_all_stage1 = QPushButton('Run stage(1)')
        self.run_all_stage1.clicked[bool].connect(partial(self.RunAllStage1, True))
        self.run_all_stage1.resize(250, 60)
        horizontal_layout.addWidget(self.run_all_stage1)

        self.run_stage1_obj_in_table = QPushButton('Run1 all')
        self.run_stage1_obj_in_table.clicked[bool].connect(partial(self.run_stage1_table))
        self.run_stage1_obj_in_table.resize(150, 60)
        horizontal_layout.addWidget(self.run_stage1_obj_in_table)

        self.show_single_fit = QPushButton('SaveFit')
        self.show_single_fit.clicked[bool].connect(partial(self.SaveSlopeFit))
        self.show_single_fit.resize(200, 60)
        horizontal_layout.addWidget(self.show_single_fit)
        self.show_jump_fit = QPushButton('ReadFit')
        self.show_jump_fit.clicked[bool].connect(partial(self.ReadSlopeFit, False))
        self.show_jump_fit.resize(200, 60)
        horizontal_layout.addWidget(self.show_jump_fit)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)
        layout.addLayout(l)
        layout.addStretch(1)



        self.setLayout(layout)

        self.setStyleSheet(open('styles.ini').read())


    def show_Image(self,mode=None):
        print('show Image:')
        self.parent.Exposures.table.show_image(mode=mode)


    def set_DQ_map(self, debug = False):
        #if int(self.parent.exp_pars.ExoTiC_mode.currentIndex()) == 1:
        #nt(self.parent.parent.exp_pars.CR_recalc_flag.currentIndex())
        #self.parent.Exposures.table.calc_ExoTiC_models()
        print('set_DQ_map, debug:', debug)
        self.parent.Exposures.table.set_dq()

    def show_DQ_map(self, debug = False,group=None):
        print('set_DQ_map, debug:', debug)
        self.parent.Exposures.table.show_dq()


    def SaturationStep(self, debug = False):
        print('SaturationStep, debug:', debug)
        self.parent.Exposures.table.check_saturation()

    def show_SATpixels(self):
        print('Show Saturation pixels')
        self.parent.Exposures.table.show_saturation()

    def show_DNUpixels(self):
        print('Show Saturation pixels')
        self.parent.Exposures.table.show_DNU()

    def FirstLastStep(self, debug = False):
        self.parent.Exposures.table.first_group()
        self.parent.Exposures.table.last_group()
        self.parent.Exposures.table.exotic_drop_groups()

    def ResetStep(self, debug=False):
        self.parent.Exposures.table.reset_correction()


    def LinearStep(self, debug=False):
        self.parent.Exposures.table.linear_correction()

    def LinearityModel(self):
        self.parent.Exposures.table.linearity_model()

    def ShowLinearCorrection(self, debug=False):
        self.parent.Exposures.table.show_linear_correction()

    def RSCDStep(self, debug=False):
        self.parent.Exposures.table.rscd_correction()

    def DarkStep(self, debug=False):
        self.parent.Exposures.table.dark_correction()

    def RefPixStep(self, debug=False):
        self.parent.Exposures.table.reference_pix_correction()

    def JumpStep(self, debug=False):
        self.parent.Exposures.table.jump_detection()

    def ShowFirstCR(self, debug=False):
        self.parent.Exposures.table.show_single_CR()

    def ShowSecondCR(self, debug=False):
        self.parent.Exposures.table.show_second_CR()

    def SlopeFitStep(self, debug=False):
        self.parent.Exposures.table.slope_fit()
        self.parent.Exposures.table.read_slopes()
        self.parent.Exposures.table.save_slope_fit(output_dir='local')

    def GetMeanRate(self, debug=False):
        self.parent.Exposures.table.calc_mean_rate(debug=True)

    def RunAllStage1(self,save_fit=False):
        print('Run all stage 1:')
        print('Init_Stage2:')
        self.parent.Exposures.table.set_dq()
        #self.parent.Exposures.table.show_image(mode='stage2')
        print('Saturated pixels')
        self.parent.Exposures.table.check_saturation()
        print('First Last')
        self.parent.Exposures.table.first_group()
        self.parent.Exposures.table.last_group()
        self.parent.Exposures.table.exotic_drop_groups()
        print('Reset correction')
        self.parent.Exposures.table.reset_correction()
        print('Linear corr')
        self.parent.Exposures.table.linear_correction()
        print('RSCD')
        self.parent.Exposures.table.rscd_correction()
        print('Dark Subtraction')
        self.parent.Exposures.table.dark_correction()
        print('Reference pixels')
        self.parent.Exposures.table.reference_pix_correction()

        algorithm = self.parent.exp_pars.FitAlgorithm_mode.currentText()
        if algorithm == 'OLS_C':
            print('Jump detection')
            self.parent.Exposures.table.jump_detection()
            print('Slope Fit')
            self.parent.Exposures.table.slope_fit()
        elif algorithm == 'CHI2':
            self.parent.Exposures.table.slope_fit()
        self.parent.Exposures.table.read_slopes()
        nint = self.parent.EXP.nint
        if nint>0 and algorithm == 'CHI2':
            print('Calc mean rate for ', nint,' integrations')
            self.parent.Exposures.table.calc_mean_rate(debug=False)

        #self.parent.Exposures.table.read_slopes()
        if save_fit and 0:
            print('Save Fit')
            self.parent.Exposures.table.save_slope_fit(output_dir='final', save_other_data=False)
            print('Run all: done.')

    def run_stage1_table(self):
        table = self.parent.Exposures.table
        for k, obj in enumerate(self.parent.Exposures.table.data):
            print(obj['name'], ' - ', k, ' from', np.size(table.data))
            name = obj['name']
            self.parent.plot_image.add(name, add=True)
            self.parent.Exposures.current_name = name

            self.RunAllStage1(save_fit=False)
            print('Save Fit')
            # set save_other_data to False to do not copy tmp data,
            self.parent.Exposures.table.save_slope_fit(output_dir='final', save_other_data=False)
            self.parent.plot_image.add(name, add=False)

    def ShowSlopeFit(self, debug=False):
        self.parent.Exposures.table.show_slope_image()

    def SaveSlopeFit(self, debug=False):
        self.parent.Exposures.table.save_slope_fit(output_dir='final')

    def ReadSlopeFit(self, debug=False):
        self.parent.Exposures.table.read_slopes()

    def RunDet1Pipeline(self,denug=False):
        print('Show Saturation pixels')
        self.parent.Exposures.table.show_saturation()

class expPipeline2Widget(QWidget):
    """
    Widget for choose fitting parameters during the fit.
    """

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        # self.resize(700, 900)
        # self.move(400, 100)
        # self.pars = {'n0': 'x', 'uv': 'y', 'Z': 'fixed', 'Av': 'disable', 'NCO': 'z'}
        # self.parent.H2.setgrid(pars=list(self.pars.keys()), show=False)
        # self.cols, self.x_, self.y_, self.z_, self.lnL_, self.mpars = None, None, None, None, None, None

        layout = QVBoxLayout(self)

        l = QVBoxLayout(self)
        l.addWidget(QLabel('Stage 2'))

        horizontal_layout = QHBoxLayout(self)
        self.init_rate = QPushButton('Init')
        self.init_rate.clicked[bool].connect(partial(self.Init_Stage2))
        self.init_rate.resize(200, 60)
        horizontal_layout.addWidget(self.init_rate)

        self.init_rate = QPushButton('Show')
        self.init_rate.clicked[bool].connect(partial(self.Show_Rate_stage2))
        self.init_rate.resize(200, 60)
        horizontal_layout.addWidget(self.init_rate)

        self.save_rate = QPushButton('Save')
        self.save_rate.clicked[bool].connect(partial(self.Save_Rate_stage2))
        self.save_rate.resize(200, 60)
        horizontal_layout.addWidget(self.save_rate)

        self.read_steps_stage2_redfringe = QPushButton('Read ResFringe')
        self.read_steps_stage2_redfringe.clicked[bool].connect(partial(self.Read_steps_stage2_Resfringe))
        self.read_steps_stage2_redfringe.resize(150, 60)
        horizontal_layout.addWidget(self.read_steps_stage2_redfringe)

        self.read_steps_stage2 = QPushButton('Read FluxCalib')
        self.read_steps_stage2.clicked[bool].connect(partial(self.Read_steps_stage2))
        self.read_steps_stage2.resize(150, 60)
        horizontal_layout.addWidget(self.read_steps_stage2)

        #self.show_comparison = QPushButton('Compare maps')
        #self.show_comparison.clicked[bool].connect(partial(self.Compare_maps))
        #self.show_comparison.resize(200, 60)
        #horizontal_layout.addWidget(self.show_comparison)
        #self.compare_init_map_choice = QComboBox()
        #flags = ['Initial','BkgrSub','Flatfield','Straylight','Fringe','Photom','ResFringe']
        #self.compare_init_map_choice.addItems(flags)
        #self.compare_init_map_choice.setCurrentIndex(0)
        #cb = self.compare_init_map_choice
        #width = cb.minimumSizeHint().width()
        #cb.setFixedWidth(width)
        #self.compare_init_map_choice.resize(150, 30)
        #horizontal_layout.addWidget(self.compare_init_map_choice)
        #self.compare_sec_map_choice = QComboBox()
        #flags = ['Initial', 'BkgrSub', 'Flatfield', 'Straylight', 'Fringe','Photom','ResFringe']
        #self.compare_sec_map_choice.addItems(flags)
        #self.compare_sec_map_choice.setCurrentIndex(0)
        #cb = self.compare_sec_map_choice
        #width = cb.minimumSizeHint().width()
        #cb.setFixedWidth(width)
        #self.compare_sec_map_choice.resize(140, 30)
        #horizontal_layout.addWidget(self.compare_sec_map_choice)

        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)

        self.run_wcs_step = QPushButton('1.AssignWCS')
        self.run_wcs_step.clicked[bool].connect(partial(self.AssignWCS))
        self.run_wcs_step.resize(200, 60)
        horizontal_layout.addWidget(self.run_wcs_step)

        self.select_jot_pix_step = QPushButton('2.FixHotPix')
        self.select_jot_pix_step.clicked[bool].connect(partial(self.FixHotPix))
        self.select_jot_pix_step.resize(200, 60)
        horizontal_layout.addWidget(self.select_jot_pix_step)

        self.run_flat_field = QPushButton('3.FlatField')
        self.run_flat_field.clicked[bool].connect(partial(self.Flat_Field ))
        self.run_flat_field.resize(150, 60)
        horizontal_layout.addWidget(self.run_flat_field)

        self.run_source_id = QPushButton('4.SourceID')
        self.run_source_id.clicked[bool].connect(partial(self.Source_identification))
        self.run_source_id.resize(150, 60)
        horizontal_layout.addWidget(self.run_source_id)

        self.run_stray_light = QPushButton('5.StrayLight')
        self.run_stray_light.clicked[bool].connect(partial(self.StrayLight))
        self.run_stray_light.resize(200, 60)
        horizontal_layout.addWidget(self.run_stray_light)

        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)


        self.run_fringe_corr = QPushButton('6.Fringe')
        self.run_fringe_corr.clicked[bool].connect(partial(self.Fringe_correction))
        self.run_fringe_corr.resize(200, 60)
        horizontal_layout.addWidget(self.run_fringe_corr)

        self.run_res_fringe_corr = QPushButton('7.ResFringe')
        self.run_res_fringe_corr.clicked[bool].connect(partial(self.Residual_Fringe_correction))
        self.run_res_fringe_corr.resize(200, 60)
        horizontal_layout.addWidget(self.run_res_fringe_corr)




        #self.run_all_stage2 = QPushButton('Sample Dithers')
        #self.run_all_stage2.clicked[bool].connect(partial(self.Prep_Dith))
        #self.run_all_stage2.resize(150, 60)
        #horizontal_layout.addWidget(self.run_all_stage2)



        self.run_flux_calib = QPushButton('8.Flux calib')
        self.run_flux_calib.clicked[bool].connect(partial(self.Flux_calibration))
        self.run_flux_calib.resize(150, 60)
        horizontal_layout.addWidget(self.run_flux_calib)

        self.run_bkgr_step = QPushButton('Bkgr Model')
        self.run_bkgr_step.clicked[bool].connect(partial(self.Run_stage2_bkgr_subtraction_only))
        #self.run_bkgr_step.clicked[bool].connect(partial(self.Run_stage2_bkgr_subtraction))
        self.run_bkgr_step.resize(200, 60)
        horizontal_layout.addWidget(self.run_bkgr_step)

        self.run_image_comparison = QPushButton('Mask_QSO')
        self.run_image_comparison.clicked[bool].connect(partial(self.Mask_QSO_traces))
        self.run_image_comparison.resize(150, 60)
        horizontal_layout.addWidget(self.run_image_comparison)


        self.run_image_comparison = QPushButton('CompareDithers')
        self.run_image_comparison.clicked[bool].connect(partial(self.Run_stage2_dither_comparison))
        self.run_image_comparison.resize(150, 60)
        horizontal_layout.addWidget(self.run_image_comparison)


        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)



        self.run_all_stage2 = QPushButton('Run Table (All)')
        self.run_all_stage2.clicked[bool].connect(partial(self.run_stage2_table_1_to_8))
        self.run_all_stage2.resize(150, 60)
        horizontal_layout.addWidget(self.run_all_stage2)

        self.run_stage2_obj_in_table = QPushButton('Run Table (1-7)')
        self.run_stage2_obj_in_table.clicked[bool].connect(partial(self.run_stage2_table_1_to_7))
        self.run_stage2_obj_in_table.resize(150, 60)
        horizontal_layout.addWidget(self.run_stage2_obj_in_table)

        self.run_stage2_obj_in_table = QPushButton('Table:Bkgr+FluxCalib')
        self.run_stage2_obj_in_table.clicked[bool].connect(partial(self.run_stage2_table_Bkgr_subtr))
        self.run_stage2_obj_in_table.resize(150, 60)
        horizontal_layout.addWidget(self.run_stage2_obj_in_table)

        self.run_cr_analyser = QPushButton('Table:CleanImage')
        #self.run_cr_analyser.clicked[bool].connect(partial(self.Compare_Dith))
        self.run_cr_analyser.clicked[bool].connect(partial(self.run_stage2_table_compare_Dith))
        self.run_cr_analyser.resize(200, 60)
        #horizontal_layout.addWidget(self.run_cr_analyser)


        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        self.show_trace_stage2 = QPushButton('Trace')
        self.show_trace_stage2.clicked[bool].connect(partial(self.Show_trace_stage2))
        self.show_trace_stage2.resize(150, 60)
        horizontal_layout.addWidget(self.show_trace_stage2)
        self.num_trace = QLineEdit()
        self.num_trace.setText(str(1))
        cb = self.num_trace
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        horizontal_layout.addWidget(self.num_trace)
        #self.trace_size_button = QPushButton('D:')
        #horizontal_layout.addWidget(self.trace_size_button)
        horizontal_layout.addWidget(QLabel('D:'))
        self.size_trace = QLineEdit()
        self.size_trace.setText(str(0))
        cb = self.size_trace
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        horizontal_layout.addWidget(self.size_trace)

        self.show_Xtrace_stage2 = QPushButton('XTrace')
        self.show_Xtrace_stage2.clicked[bool].connect(partial(self.Show_Xtrace_stage2))
        self.show_Xtrace_stage2.resize(150, 60)
        horizontal_layout.addWidget(self.show_Xtrace_stage2)
        self.Xtrace_coord = QLineEdit()
        self.Xtrace_coord.setText(str(1))
        cb = self.Xtrace_coord
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        horizontal_layout.addWidget(self.Xtrace_coord)
        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)

        horizontal_layout = QHBoxLayout(self)
        horizontal_layout.addWidget(QLabel('QSO_RA:'))
        self.qso_ra_coord = QLineEdit()
        #self.qso_ra_coord.setText(str(135.3445)) #J0901
        #self.qso_ra_coord.setText(str(151.807004)) #J1007
        self.qso_ra_coord.setText(str(39.662209)) #AO0235
        #self.qso_ra_coord.setText(str(154.463132))  # J1017
        cb = self.qso_ra_coord
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        #self.nINT.resize(90, 30)
        horizontal_layout.addWidget(self.qso_ra_coord)
        horizontal_layout.addWidget(QLabel('QSO_DEC:'))
        self.qso_dec_coord = QLineEdit()
        #self.qso_dec_coord.setText(str(20.746259)) #J0901
        #self.qso_dec_coord.setText(str(28.896786)) #J1007
        self.qso_dec_coord.setText(str(16.616465 	))  # AO0235
        #self.qso_dec_coord.setText(str(47.827824))  # J1017
        cb = self.qso_dec_coord
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        # self.nINT.resize(90, 30)
        horizontal_layout.addWidget(self.qso_dec_coord)

        horizontal_layout.addWidget(QLabel('Apert:'))
        self.qso_aperture = QLineEdit()
        #self.qso_dec_coord.setText(str(20.746259)) #J0901
        self.qso_aperture.setText(str(1.8)) #J1007
        cb = self.qso_aperture
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        # self.nINT.resize(90, 30)
        horizontal_layout.addWidget(self.qso_aperture)


        horizontal_layout.addStretch(1)
        l.addLayout(horizontal_layout)



        layout.addLayout(l)
        layout.addStretch(1)

        self.setLayout(layout)

        self.setStyleSheet(open('styles.ini').read())

    def Init_Stage2(self, mode='stage2'):
        print('Init_Stage2:')
        self.parent.Exposures.table.init_stage2()
        #self.parent.Exposures.table.read_slopes()
        self.parent.Exposures.table.show_image(mode='stage2')

    def Show_Rate_stage2(self, mode='stage2'):
        print('Show rate_Stage2:')
        self.parent.Exposures.table.show_image(mode='stage2')

    def Save_Rate_stage2(self, mode='stage2'):
        print('Show rate_Stage2:')
        self.parent.Exposures.table.save_image(mode='stage2')

    #def Compare_maps(self):
    #    self.parent.Exposures.table.stage2_compare_maps()

    def FixHotPix(self, debug=False):
        print('SelectHotPixels:')
        #self.parent.Exposures.table.stage2_fix_hot_pix()
        self.parent.Exposures.table.stage2_fix_hot_pix_version2()
    def AssignWCS(self):
        print('AssignWCS')
        self.parent.Exposures.table.stage2_call_wcs()

    def Bkgr_Model(self):
        print('Bkgr_Model')
        if 0:
            # default pipeline bkgr sub - pix by pix
            self.parent.Exposures.table.stage2_background_subtraction()
        if 1:
            self.parent.Exposures.table.stage2_background_model()

    def Mask_QSO_traces(self):
        save_res_flag = int(self.parent.exp_pars.save_tmp_res.currentIndex())
        qso_ra,qso_dec = float(self.parent.stage2_commands.qso_ra_coord.text()),float(self.parent.stage2_commands.qso_dec_coord.text())
        qso_aperture = float(self.parent.stage2_commands.qso_aperture.text())
        self.parent.stage2.mask_qso_traces(debug=True,q_ra= qso_ra,q_dec=qso_dec,q_delta=qso_aperture, save_pdf=save_res_flag)

    def Prep_Dith(self):
        table = self.parent.Exposures.table
        for k, obj in enumerate(table.data):
            print(obj['name'], ' - ', k, ' from', np.size(table.data))
            filename = obj['name']
            if 'BACK' not in obj[3]:
                print('Name:', obj[3], filename)
                self.parent.plot_image.add(filename, add=True)
                print('Init_Stage2:')
                self.parent.Exposures.table.init_stage2()
                self.parent.Exposures.current_name = filename
                print('AssignWCS')
                self.parent.Exposures.table.stage2_call_wcs()
                print('Fix hot Pix')
                self.FixHotPix()
                print('Bkgr_Subtraction')
                self.Bkgr_Model()  # parent.Exposures.table.stage2_background_model()
                print('Mask QSO traces')
                self.Mask_QSO_traces()  # parent.Exposures.table.stage2_background_model()
                print('Save file')
                self.Save_Rate_stage2()  # parent.Exposures.table.stage2_background_model()
                self.parent.plot_image.add(filename, add=False)

    def Compare_Dith(self):
        self.parent.Exposures.table.stage2_compare_dithers_sample()


    def Flat_Field(self):
        print('Flat_Field')
        self.parent.Exposures.table.stage2_flat_field()

    def Source_identification(self):
        print('Source_identification')
        self.parent.Exposures.table.stage2_source_identification()

    def StrayLight(self):
        print('StrayLight')
        self.parent.Exposures.table.stage2_stray_light()

    def Fringe_correction(self):
        print('Fringe Flat correction')
        self.parent.Exposures.table.stage2_fringe_flat_correction()

    def Residual_Fringe_correction(self):
        print('Residual Fringe correction')
        self.parent.Exposures.table.stage2_residual_fringe_correction()

    def Flux_calibration(self):
        print('Flux_calibration')
        self.parent.Exposures.table.stage2_flux_calibration()

    def Run2_allsteps(self):
        print('Run all steps')
        print('Init_Stage2:')
        detector = self.parent.Exposures.table.init_stage2()
        if detector != 'MIRIMAGE':
            print('AssignWCS')
            self.parent.Exposures.table.stage2_call_wcs()
            #print('Fix hot Pix')
            #self.FixHotPix()
            print('Flat_Field')
            self.parent.Exposures.table.stage2_flat_field()
            print('Source_identification')
            self.parent.Exposures.table.stage2_source_identification()
            print('StrayLight')
            self.parent.Exposures.table.stage2_stray_light()
            print('Fringe Flat correction')
            self.parent.Exposures.table.stage2_fringe_flat_correction()
            print('Residual flux correction')
            self.parent.Exposures.table.stage2_residual_fringe_correction()
            #print('Bkgr_Subtraction')
            #self.Bkgr_Model()  # parent.Exposures.table.stage2_background_model()
            print('Flux_calibration')
            self.parent.Exposures.table.stage2_flux_calibration()
            print('Run all: done.')
        else:
            print('Error: You try to run pipiline stage2 for IMAGE data')

    def Run2_steps_1_to_7(self):
        print('Run steps_1to_7')
        print('Init_Stage2:')
        detector = self.parent.Exposures.table.init_stage2()
        if detector != 'MIRIMAGE':
            #print('Init')
            #self.parent.Exposures.table.init_stage2()
            print('AssignWCS')
            self.parent.Exposures.table.stage2_call_wcs()
            print('Fix hot Pix')
            self.FixHotPix()
            print('Flat_Field')
            self.parent.Exposures.table.stage2_flat_field()
            print('Source_identification')
            self.parent.Exposures.table.stage2_source_identification()
            print('StrayLight')
            self.parent.Exposures.table.stage2_stray_light()
            print('Fringe Flat correction')
            self.parent.Exposures.table.stage2_fringe_flat_correction()
            print('Residual flux correction')
            self.parent.Exposures.table.stage2_residual_fringe_correction()
        else:
            print('Error: You try to run pipiline stage2 for IMAGE data')


    def Run_stage2_bkgr_subtraction(self):
        detector = self.parent.Exposures.table.init_stage2()
        print('read rate file')
        target_name = self.parent.stage2.data.meta.target.proposer_name
        print('target name',target_name)
        s = self.parent.Exposures.table.stage2_read_res_fringes()
        #s = self.parent.Exposures.table.stage2_read_flat_fringes()
        print('read fringe corrected file',s)
        if s and 'BACKGROUND' not in target_name:
            print('create corresponding background model for', target_name, s)
            self.Bkgr_Model()  # parent.Exposures.table.stage2_background_model()
            print('make bkgr model and subtract')
            self.parent.Exposures.table.stage2_flux_calibration()
            print('make flux calibration')
        elif s and 'BACKGROUND' in target_name:
            print('Ignore background exposure for', target_name)
        else:
            print('There is no the saved residual fringe file')

    def Run_stage2_bkgr_subtraction_only(self):
        print('Run steps8')
        detector = self.parent.Exposures.table.init_stage2()
        print('Update data file')
        s = self.parent.Exposures.table.stage2_read_res_fringes()
        save_res_flag = int(self.parent.exp_pars.save_tmp_res.currentIndex())
        if s:
            self.Bkgr_Model()  # parent.Exposures.table.stage2_background_model()
        else:
            print('There is no the saved residual fringe file')

    #def Run_stage2_mask_source_orders(self):
    #    s = self.parent.Exposures.table.stage2_read_res_fringes()
    #    if s:
    #        print('Bkgr_Subtraction')
    #        self.Bkgr_Model()  # parent.Exposures.table.stage2_background_model()
    #        self.parent.Exposures.table.save_image(mode='masked_qso')
    #    else:
    #        print('There is no the saved residual fringe file')

    def Run_stage2_dither_comparison(self):
        print('Run steps9-10')
        print('Init_Stage2:')
        detector = self.parent.Exposures.table.init_stage2()
        print('Update data file')
        s = self.parent.Exposures.table.stage2_read_bkgr_subtracted()
        if s:
            print('Compare dithered images')
            self.Compare_Dith()  # parent.Exposures.table.stage2_background_model()
            #self.parent.Exposures.table.save_image(mode='bkgr_sub')
        else:
            print('There is no the saved rate file')
        print('Done.')

    def Run_stage2_mask_qso(self):
        print('Mask QSO')
        print('Init_Stage2:')
        detector = self.parent.Exposures.table.init_stage2()
        s = self.parent.Exposures.table.stage2_read_bkgr_subtracted()
        if s:
            print('Compare dithered images')
            self.Compare_Dith()  # parent.Exposures.table.stage2_background_mod
        else:
            print('There is no the saved rate file')
        print('Done.')



    def Run_stage2_dither_comparison_and_fluxcalibration(self):
        print('Run steps9-10')
        print('Init_Stage2:')
        detector = self.parent.Exposures.table.init_stage2()
        print('Update data file')
        s = self.parent.Exposures.table.stage2_read_bkgr_subtracted()
        if s:
            print('Compare dithered images')
            self.Compare_Dith()  # parent.Exposures.table.stage2_background_model()
            #self.parent.Exposures.table.save_image(mode='bkgr_sub')
        else:
            print('There is no the saved rate file')
        print('Flux_calibration')
        self.parent.Exposures.table.stage2_flux_calibration()
        print('Done.')

    def Read_steps_stage2(self):
        print('Read step')
        self.parent.Exposures.table.stage2_read_steps()
        self.parent.Exposures.table.show_image(mode='stage2')

    def Read_steps_stage2_Resfringe(self):
        print('Read step')
        self.parent.Exposures.table.stage2_read_step_res_fringe()
        self.parent.Exposures.table.show_image(mode='stage2')

    def Show_trace_stage2(self):
        self.parent.Exposures.table.stage2_show_trace(trace_order=int(self.num_trace.text()),trace_size=int(self.size_trace.text()))

    def Show_Xtrace_stage2(self):
        self.parent.Exposures.table.stage2_show_Xtrace(trace_Xpos=int(self.Xtrace_coord.text()))
            #trace_order=int(self.num_trace.text()),trace_size=int(self.size_trace.text()))



    def run_stage2_table_1_to_8(self):
        table = self.parent.Exposures.table
        for k, obj in enumerate(self.parent.Exposures.table.data):
            print(obj['name'], ' - ', k,' from',np.size(table.data))
            name = obj['name']
            self.parent.plot_image.add(name, add=True)
            self.parent.Exposures.current_name = name
            self.Run2_allsteps()
            self.parent.plot_image.add(name, add=False)



    def run_stage2_table_1_to_7(self):
        table = self.parent.Exposures.table
        for k, obj in enumerate(self.parent.Exposures.table.data):
            print(obj['name'], ' - ', k,' from',np.size(table.data))
            name = obj['name']
            self.parent.plot_image.add(name, add=True)
            self.parent.Exposures.current_name = name
            self.Run2_steps_1_to_7()
            self.parent.plot_image.add(name, add=False)

    def run_stage2_table_Bkgr_subtr(self):
        table = self.parent.Exposures.table
        for k, obj in enumerate(self.parent.Exposures.table.data):
            print(obj['name'], ' - ', k,' from',np.size(table.data))
            name = obj['name']
            self.parent.plot_image.add(name, add=True)
            self.parent.Exposures.current_name = name
            self.Run_stage2_bkgr_subtraction()
            self.parent.plot_image.add(name, add=False)

    def run_stage2_table_compare_Dith(self):
        table = self.parent.Exposures.table
        for k, obj in enumerate(self.parent.Exposures.table.data):
            print(obj['name'], ' - ', k,' from',np.size(table.data))
            name = obj['name']
            self.parent.plot_image.add(name, add=True)
            self.parent.Exposures.current_name = name
            self.Run_stage2_dither_comparison_and_fluxcalibration()
            self.parent.plot_image.add(name, add=False)

class JWSTviewer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.read_settings()
        print('I use the JWST pipeline of VERSION:  ', jwst.__version__)

        self.EXP = detector1(input_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=self.init_settings['input1_dir'], output_dir=self.init_settings['output1_dir'])
        self.stage2 = detector2(miri_uncal_file='jw02155001001_04102_00001_mirifulong_uncal.fits', path=self.init_settings['input2_dir'], output_dir=self.init_settings['output2_dir'],
                                spec2_cachedir=self.init_settings['spec2_cachedir'])
        #self.H2.readfolder()
        self.initStyles()
        self.initUI()

    def read_settings(self,init_file='init.dat'):
        self.init_settings = {}
        with open(init_file) as f:
            for k, line in enumerate(f):
                values = [s for s in line.split()]
                if  line[0]!= '#':
                    if values[0] == 'input1_dir:':
                        input_dir  = values[1]
                        self.init_settings['input1_dir'] = values[1]
                    if values[0] == 'input2_dir:':
                        self.init_settings['input2_dir']=values[1]
                    if values[0] == 'spec2_cachedir:':
                        self.init_settings['spec2_cachedir'] = values[1]
                    if values[0] == 'output1_dir:':
                        self.init_settings['output1_dir'] = values[1]
                    if values[0] == 'output2_dir:':
                        self.init_settings['output2_dir'] = values[1]
    def initStyles(self):
        self.setStyleSheet(open('styles.ini').read())

    def initUI(self):
        #dbg = pg.dbg()
        # self.specview sets the type of plot representation

        if 1:# >>> create panel for plotting spectra
            self.plot_image = plotImage(self)
            self.plot_pixel = plotPixProfile(self)
            self.plot_pixel_diffs = plotPixDiffs(self)
            self.Exposures = chooseExpWidget(self, closebutton=False)
            self.exp_pars = expParsWidget(self)
            self.exp_commands = expRunWidget(self)
            self.stage2_commands = expPipeline2Widget(self)

            # self.plot.setFrameShape(QFrame.StyledPanel)

            #self.splitter = QSplitter(Qt.Horizontal) #pyqt5
            #self.splitter_plot = QSplitter(Qt.Vertical)
            self.splitter =  QSplitter(Qt.Orientation.Horizontal)
            self.splitter_plot = QSplitter(Qt.Orientation.Vertical)
            self.splitter_plot.addWidget(self.plot_image)
            self.splitter_plot.addWidget(self.plot_pixel)
            self.splitter_plot.addWidget(self.plot_pixel_diffs)
            self.splitter.addWidget(self.splitter_plot)

            self.splitter_pars = QSplitter(Qt.Orientation.Vertical) #QSplitter(Qt.Vertical)
            self.splitter_pars.addWidget(self.Exposures)
            self.splitter_pars.addWidget(self.exp_pars)
            self.splitter_pars.addWidget(self.exp_commands)
            self.splitter_pars.addWidget(self.stage2_commands)

            #self.splitter_pars.setSizes([10, 150])
            self.splitter.addWidget(self.splitter_pars)
            self.splitter.setSizes([1500, 250])


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
    ex = JWSTviewer()
    sys.exit(app.exec())

