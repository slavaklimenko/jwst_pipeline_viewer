import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit,QSizePolicy, QSlider, QSpacerItem, QVBoxLayout, QWidget,QComboBox, QPushButton
from PyQt5 import QtGui
import pyqtgraph as pg
import numpy as np
import sys, os
from scipy.interpolate import interp1d, UnivariateSpline
from functools import partial
import matplotlib.pyplot as plt

class spectrum():
    def __init__(self, x=None, y=None, err=None,name=None):
        if any([v is not None for v in [x, y, err]]):
            self.set_data(x=x, y=y, err=err,name=name)

    def set_data(self, x=None, y=None, err=None,name=None):
        if x is not None:
            self.x = np.asarray(x)
        if y is not None:
            self.y = np.asarray(y)
        if y is not None:
            self.y = np.asarray(y)
        if err is not None:
            self.err = np.asarray(err)
        if name is not None:
            self.name = name

    def append(self,s,mode = 'mean disp',debug=False):
        if not hasattr(self, 'x'):
            if hasattr(s,'x'):
                self.x = s.x.copy()
                self.y = s.y.copy()
                self.err = s.err.copy()
        else:
            if np.sum(s.x)>0:
                mask_intersection = s.x <= self.x[-1]
                mask_extension = ~mask_intersection
                if debug:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 2)
                    ax[0].plot(s.x,s.y/s.err,label='s')
                    ax[0].plot(self.x,self.y/self.err,label='self')
                    f = np.linspace(0.1,10,100)
                    ax[1].plot(f,(f+5)*5/(25 + f**2))
                    ax[0].legend()
                    plt.show()


                if np.sum(mask_intersection) > 0:
                    s_interp = interp1d(s.x, s.y, bounds_error=False, fill_value=np.NaN)
                    s_interp_err = interp1d(s.x, s.err, bounds_error=False, fill_value=np.NaN)
                    mask_selfx_intersection = self.x>=s.x[0]
                    comb = [self.y[mask_selfx_intersection],s_interp(self.x[mask_selfx_intersection])]
                    e_comb = [self.err[mask_selfx_intersection],s_interp_err(self.x[mask_selfx_intersection])]
                    if mode == 'mean weighted':
                        w = np.power(e_comb,-2)
                        f2 = np.nansum(comb * w, axis=0) / np.nansum(w, axis=0)
                        self.y[mask_selfx_intersection] = f2
                        self.err[mask_selfx_intersection] = np.power(np.nansum(w, axis=0), -0.5)
                    elif mode == 'mean':
                        self.y[mask_selfx_intersection] = np.nanmean(comb)
                        self.err[mask_selfx_intersection] = np.power(np.nansum(np.power(e_comb, -2), axis=0), -0.5)
                    elif mode == 'mean disp':
                        self.y[mask_selfx_intersection] = np.nansum(comb,axis=0)/2
                        f = comb-self.y[mask_selfx_intersection]
                        f1 = np.power(f,2)
                        self.err[mask_selfx_intersection] = np.power(np.nansum(f1,axis=0)/2, 0.5)

                self.x = np.append(self.x,s.x[mask_extension])
                self.y = np.append(self.y,s.y[mask_extension])
                self.err = np.append(self.err,s.err[mask_extension])


    def copy(self):
        return spectrum(np.array(self.x),np.array(self.y),np.array(self.err))



class Slider(QWidget):
    def __init__(self, minimum, maximum, parent=None,name='ch1A:',filenamelist=['file'],path ='',val=None):
        super(Slider, self).__init__(parent=parent)
        self.name = name
        self.spec_files_path = path
        self.verticalLayout = QVBoxLayout(self)

        self.horizontalLayout0 = QHBoxLayout(self)
        self.horizontalLayout0.addWidget(QLabel('f:'))
        self.filename_box = QComboBox(self)
        self.filename_box.addItems(filenamelist)
        self.filename_box.setCurrentIndex(0)
        self.filename_box.setFixedSize(200, 30)

        self.horizontalLayout0.addWidget(self.filename_box)
        self.verticalLayout.addLayout(self.horizontalLayout0)


        self.horizontalLayout1 = QHBoxLayout(self)
        self.horizontalLayout1.addWidget(QLabel(name))
        self.value_label = QLineEdit(self)
        self.value_label.setText(str(val))
        self.value_label.setFixedSize(60, 30)
        self.horizontalLayout1.addWidget(self.value_label)
        self.horizontalLayout1.addStretch(1)
        self.verticalLayout.addLayout(self.horizontalLayout1)

        self.up_label = QLineEdit(self)
        self.up_label.setText(str(maximum))
        self.up_label.setFixedSize(60, 30)
        self.verticalLayout.addWidget(self.up_label)
        self.horizontalLayout = QHBoxLayout(self)
        spacerItem = QSpacerItem(0, 100, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(40, 80, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.low_label = QLineEdit(self)
        self.low_label.setText(str(minimum))
        self.low_label.setFixedSize(60, 30)
        self.verticalLayout.addWidget(self.low_label)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = val
        self.setLabelValue(self.slider.value())


        self.read_spectrum()
        #self.filename_box.currentIndexChanged.connect(self.read_spectrum(f=f))

    def setLabelValue(self, value):
        self.setMinMaxLimits()
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (self.maximum - self.minimum)

        #self.label.setText("xmin:{0:.4g}".format(self.x))

    def setMinMaxLimits(self):
        self.maximum = float(self.up_label.text())
        self.minimum = float(self.low_label.text())


    def read_spectrum(self):
        f =  self.filename_box.currentText()
        if f not in ['None','']:
            d = np.loadtxt(fname=self.spec_files_path + f)
            self.data = spectrum(x=d[:,0],y=d[:,1],err=d[:,2],name=f)
        else:
            self.data = None


class plotSpec(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'SB (MJy/sr)', 'bottom': 'Wavelength [micron]'})
        self.vb = self.getViewBox()
        self.text = None

        self.legend = pg.LegendItem(offset=(-70, 30))
        self.legend.setParentItem(self.vb)
        self.legend_model = pg.LegendItem(offset=(-70, -30))
        self.legend_model.setParentItem(self.vb)
        self.setTitle("Pixel difference", color="olive", size="10pt")
        self.lines = self.listDataItems()
        self.spectra_list = {}
        self.spec_line_colors = {}

    def plot_spec(self, fname = None, fcolor='lightgreen',data=None, coef =1 ,add=True,show_err_bar=False):
        if fname != None:
            if add:
                x = data.x
                y = data.y.copy()
                err = data.err.copy()
                if coef!= 1:
                    y*=coef
                    err*=coef
                self.spec_line_colors[fname] = fcolor
                pen = pg.mkPen(color=self.spec_line_colors[fname], style=Qt.SolidLine, width=2)
                self.spectra_list[fname] = pg.PlotCurveItem(x, y,pen=pen)

                #self.plot_lineA1 = pg.PlotCurveItem(x, y,pen='lightgreen')
                #self.plot_errbarA1 = pg.ErrorBarItem(x=wavel,y=data,height=err,pen=pen, beam=1/6000)
                self.vb.addItem(self.spectra_list[fname])
                self.legend_model.addItem(self.spectra_list[fname], fname)
                if show_err_bar:
                    self.plot_errbar = pg.ErrorBarItem(x=x,y=y,height=err,pen=pen, beam=1/6000)
                    self.vb.addItem(self.plot_errbar)
                #    pen = pg.mkPen(color='darkgray', style=Qt.DashLine, width=1)
                    #self.zero_level = pg.PlotCurveItem([wavel[0]-2, wavel[-1] + 2], [0, 0], pen=pen)
                self.zero_level = pg.PlotCurveItem([0, 30], [0, 0], pen=pg.mkPen(color='darkgray', style=Qt.DashLine, width=1))
                self.vb.addItem(self.zero_level)
                self.show_template = False
                if 0:
                    NGC = np.loadtxt('/home/slava/science/codes/python/jwst/input/NGC19.txt',delimiter=',')
                    #NGC = np.loadtxt('/home/slava/science/data/SPITZER/AO0235/cassis_yaaar_spcfw_15121152t.dat')
                    x,y = NGC[:,0]/1e4, NGC[:,1]
                    mask = (x>wavel[10])*(x<wavel[40])
                    self.show_template = False
                    if np.sum(mask)>0:
                        self.show_template = True
                        norm = 1/np.mean(y[mask])*np.mean(data[10:40])
                        self.plot_NGC = pg.PlotCurveItem(x,y*norm, pen='lightgreen')
                        self.vb.addItem(self.plot_NGC)
                    #self.legend_model.addItem(self.plot_NGC, 'Template')


            else:
                try:
                    if show_err_bar:
                        self.vb.removeItem(self.plot_errbar)
                    self.vb.removeItem(self.spectra_list[fname])

                    self.legend_model.removeItem(self.spectra_list[fname])
                    #if self.show_template:
                    #    self.show_template = False
                    #    self.vb.removeItem(self.plot_NGC)
                    #    self.legend_model.removeItem(self.plot_NGC)



                    self.vb.removeItem(self.zero_level)
                   #self.vb.removeItem(self.lr)



                except:
                    pass






class Viewer(QWidget):
    def __init__(self, parent=None,spec_folder = './',filenamelist=['file']):
        super(Viewer, self).__init__(parent=parent)
        self.mainLayout = QVBoxLayout(self)
        self.spec_folder = spec_folder
        self.spec_tmp = '/home/slava/science/codes/python/jwst/output/detector3/tmp/'
        #self.win = pg.GraphicsLayoutWidget(title="Basic plotting examples")
        self.win = plotSpec(self)
        self.mainLayout.addWidget(self.win)

        self.horizontalLayout = QHBoxLayout(self)
        self.horizontalLayout.addWidget(QLabel('Obj name:'))
        objnamelist = self.readobjnameinfolder()

        self.objname_box = QComboBox(self)
        self.objname_box.addItems(objnamelist)
        self.objname_box.setCurrentIndex(0)
        self.objname_box.setFixedSize(200, 30)
        self.horizontalLayout.addWidget(self.objname_box)
        self.obj_name_win = QPushButton('ReadSpecList')
        self.obj_name_win.clicked[bool].connect(self.setObjName)
        self.obj_name_win.setFixedSize(200, 60)
        self.horizontalLayout.addWidget(self.obj_name_win)
        self.comb_dithers_win = QPushButton('CombineDithers')
        self.comb_dithers_win.clicked.connect(self.comb_dithers)
        self.comb_dithers_win.setFixedSize(200, 60)
        self.horizontalLayout.addWidget(self.comb_dithers_win)
        self.save_data_win = QPushButton('SaveSpec')
        #self.build_cube.clicked[bool].connect(partial(self.call_build_3dCube))
        self.save_data_win.clicked[bool].connect(partial(self.saveObj))
        self.save_data_win.setFixedSize(200, 60)
        self.horizontalLayout.addWidget(self.save_data_win)
        self.save_data_filename = QLineEdit()
        self.save_data_filename.setText('filename')
        self.save_data_filename.setFixedSize(150, 30)
        self.horizontalLayout.addWidget(self.save_data_filename)
        self.combine_win = QPushButton('Combine')
        # self.build_cube.clicked[bool].connect(partial(self.call_build_3dCube))
        self.combine_win.clicked[bool].connect(partial(self.combineChunks))
        self.combine_win.setFixedSize(200, 60)
        self.horizontalLayout.addWidget(self.combine_win)
        self.rebin_win = QPushButton('Rebin')
        # self.build_cube.clicked[bool].connect(partial(self.call_build_3dCube))
        self.rebin_win.clicked[bool].connect(partial(self.RebinIt))
        self.rebin_win.setFixedSize(200, 60)
        self.horizontalLayout.addWidget(self.rebin_win)
        self.rebin_n_pix = QLineEdit()
        self.rebin_n_pix.setText('1 pix')
        self.rebin_n_pix.setFixedSize(100, 30)
        self.horizontalLayout.addWidget(self.rebin_n_pix)
        self.recalc_errorbar =  QPushButton('RecalcStd')
        self.recalc_errorbar.clicked[bool].connect(partial(self.RecalcStd))
        self.recalc_errorbar.setFixedSize(200, 60)
        self.horizontalLayout.addWidget(self.recalc_errorbar)

        self.horizontalLayout.addStretch(1)
        self.mainLayout.addLayout(self.horizontalLayout)



        self.horizontalLayout = QHBoxLayout(self)
        if 1:
            filenamelist = ['None']+self.readfolder(obj_name=self.objname_box.currentText())
            self.w1 = Slider(0, 3, name='ch1A:', filenamelist=filenamelist, path=self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w1)

            self.w2 = Slider(0, 3,name='ch1B:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w2)

            self.w3 = Slider(0, 3,name='ch1C:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w3)

            self.w4 = Slider(0, 3,name='ch2A:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w4)

            self.w5 = Slider(0, 3, name='ch2B:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w5)

            self.w6 = Slider(0, 3, name='ch2C:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w6)

            self.w7 = Slider(0, 3, name='ch3A:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w7)

            self.w8 = Slider(0, 3, name='ch3B:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w8)

            self.w9 = Slider(0, 3, name='ch3C:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w9)

            self.w10 = Slider(0, 3, name='ch4A:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w10)

            self.w11 = Slider(0, 3, name='ch4B:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w11)

            self.w12 = Slider(0, 3, name='ch4C:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w12)

            self.w13 = Slider(0, 3, name='tmplate:', filenamelist=['cassis_yaaar_spcfw_15121152t-copy-red_norm.dat'], path=self.spec_tmp,val=1)
            self.horizontalLayout.addWidget(self.w13)
        self.mainLayout.addLayout(self.horizontalLayout)
        #self.p6 = self.win.addPlot(title="My Plot")
        #self.curve = self.p6.plot(pen='r')
        #self.update_plot()

        if 1:
             self.w1.filename_box.currentIndexChanged.connect(self.w1.read_spectrum)
             self.w2.filename_box.currentIndexChanged.connect(self.w2.read_spectrum)
             self.w3.filename_box.currentIndexChanged.connect(self.w3.read_spectrum)
             self.w4.filename_box.currentIndexChanged.connect(self.w4.read_spectrum)
             self.w5.filename_box.currentIndexChanged.connect(self.w5.read_spectrum)
             self.w6.filename_box.currentIndexChanged.connect(self.w6.read_spectrum)
             self.w7.filename_box.currentIndexChanged.connect(self.w7.read_spectrum)
             self.w8.filename_box.currentIndexChanged.connect(self.w8.read_spectrum)
             self.w9.filename_box.currentIndexChanged.connect(self.w9.read_spectrum)
             self.w10.filename_box.currentIndexChanged.connect(self.w10.read_spectrum)
             self.w11.filename_box.currentIndexChanged.connect(self.w11.read_spectrum)
             self.w12.filename_box.currentIndexChanged.connect(self.w12.read_spectrum)
             self.w13.filename_box.currentIndexChanged.connect(self.w13.read_spectrum)
             self.update_plot()

        if 1:
            self.w1.slider.valueChanged.connect(self.update_slider)
            self.w1.value_label.textChanged.connect(self.update_plot)


            self.w2.slider.valueChanged.connect(self.update_slider)
            self.w3.slider.valueChanged.connect(self.update_plot)
            self.w4.slider.valueChanged.connect(self.update_plot)
            self.w5.slider.valueChanged.connect(self.update_plot)
            self.w6.slider.valueChanged.connect(self.update_plot)
            self.w7.slider.valueChanged.connect(self.update_plot)
            self.w8.slider.valueChanged.connect(self.update_plot)
            self.w9.slider.valueChanged.connect(self.update_plot)
            self.w10.slider.valueChanged.connect(self.update_plot)
            self.w11.slider.valueChanged.connect(self.update_plot)
            self.w12.slider.valueChanged.connect(self.update_plot)
            self.w13.slider.valueChanged.connect(self.update_plot)

    def update_slider(self):
        self.update_plot()
        self.update_val_labels()
    def update_plot(self,update_val_label=False):


        if self.w1.data != None:
            self.win.plot_spec(fname='spec1', add=False, show_err_bar=True)
            self.win.plot_spec(fname='spec1',fcolor='blue',data=self.w1.data, coef=self.w1.x,   show_err_bar=True)
        if self.w2.data != None:
            self.win.plot_spec(fname='spec2', add=False,show_err_bar=True)
            self.win.plot_spec(fname='spec2',fcolor='orange',data=self.w2.data, coef=self.w2.x,   show_err_bar=True)
        if self.w3.data != None:
            self.win.plot_spec(fname='spec3', add=False,show_err_bar=False)
            self.win.plot_spec(fname='spec3',fcolor='green',data=self.w3.data, coef=self.w3.x,  show_err_bar=False)
        if self.w4.data != None:
            self.win.plot_spec(fname='spec4', add=False, show_err_bar=False)
            self.win.plot_spec(fname='spec4', fcolor='red', data=self.w4.data, coef=self.w4.x, show_err_bar=False)
        if self.w5.data != None:
            self.win.plot_spec(fname='spec5', add=False, show_err_bar=False)
            self.win.plot_spec(fname='spec5', fcolor='purple', data=self.w5.data, coef=self.w5.x,  show_err_bar=False)
        if self.w6.data != None:
            self.win.plot_spec(fname='spec6', add=False, show_err_bar=False)
            self.win.plot_spec(fname='spec6', fcolor='brown', data=self.w6.data, coef=self.w6.x, show_err_bar=False)
        if self.w7.data != None:
            self.win.plot_spec(fname='spec7', add=False, show_err_bar=False)
            self.win.plot_spec(fname='spec7', fcolor='pink', data=self.w7.data, coef=self.w7.x,  show_err_bar=False)
        if self.w8.data != None:
            self.win.plot_spec(fname='spec8', add=False, show_err_bar=False)
            self.win.plot_spec(fname='spec8', fcolor='gray', data=self.w8.data, coef=self.w8.x, show_err_bar=False)
        if self.w9.data != None:
            self.win.plot_spec(fname='spec9', add=False, show_err_bar=False)
            self.win.plot_spec(fname='spec9', fcolor='olive', data=self.w9.data, coef=self.w9.x,  show_err_bar=False)
        if self.w10.data != None:
            self.win.plot_spec(fname='spec10', add=False, show_err_bar=False)
            self.win.plot_spec(fname='spec10', fcolor='cyan', data=self.w10.data, coef=self.w10.x, show_err_bar=False)
        if self.w11.data != None:
            self.win.plot_spec(fname='spec11', add=False, show_err_bar=False)
            self.win.plot_spec(fname='spec11', fcolor='magenta', data=self.w11.data, coef=self.w11.x,  show_err_bar=False)

        if self.w12.data != None:
            self.win.plot_spec(fname='spec12', add=False, show_err_bar=False)
            self.win.plot_spec(fname='spec12', fcolor='yellow', data=self.w12.data, coef=self.w12.x,  show_err_bar=False)
        if self.w13.data != None:
            self.win.plot_spec(fname='template', add=False, show_err_bar=False)
            self.win.plot_spec(fname='template', fcolor='white', data=self.w13.data, coef=self.w13.x, show_err_bar=False)

        #x = np.linspace(0, 10, 100)
        #data = a + np.cos(x + c * np.pi / 180) * np.exp(-b * x) * d
        if update_val_label:
            self.w1.value_label.setText("%.1f" % (self.w1.x))
            self.w2.value_label.setText("%.1f" % (self.w2.x))
            self.w3.value_label.setText("%.1f" % (self.w3.x))
            self.w4.value_label.setText("%.1f" % (self.w4.x))
            self.w5.value_label.setText("%.1f" % (self.w5.x))
            self.w6.value_label.setText("%.1f" % (self.w6.x))
            self.w7.value_label.setText("%.1f" % (self.w7.x))
            self.w8.value_label.setText("%.1f" % (self.w8.x))
            self.w9.value_label.setText("%.1f" % (self.w9.x))
            self.w10.value_label.setText("%.1f" % (self.w10.x))
            self.w11.value_label.setText("%.1f" % (self.w11.x))
            self.w12.value_label.setText("%.1f" % (self.w12.x))

        #self.curve.setData(data)

    def update_val_labels(self, update_val_label=True):
        self.w1.value_label.setText("%.1f" % (self.w1.x))
        self.w2.value_label.setText("%.1f" % (self.w2.x))
        self.w3.value_label.setText("%.1f" % (self.w3.x))
        self.w4.value_label.setText("%.1f" % (self.w4.x))
        self.w5.value_label.setText("%.1f" % (self.w5.x))
        self.w6.value_label.setText("%.1f" % (self.w6.x))
        self.w7.value_label.setText("%.1f" % (self.w7.x))
        self.w8.value_label.setText("%.1f" % (self.w8.x))
        self.w9.value_label.setText("%.1f" % (self.w9.x))
        self.w10.value_label.setText("%.1f" % (self.w10.x))
        self.w11.value_label.setText("%.1f" % (self.w11.x))
        self.w12.value_label.setText("%.1f" % (self.w12.x))

    def setObjName(self,click=1,secret=''):
        filenamelist = ['None'] + self.readfolder(obj_name=self.objname_box.currentText())

        self.w1.filename_box.clear()
        self.w1.filename_box.addItems(filenamelist)
        self.w2.filename_box.clear()
        self.w2.filename_box.addItems(filenamelist)
        self.w3.filename_box.clear()
        self.w3.filename_box.addItems(filenamelist)
        self.w4.filename_box.clear()
        self.w4.filename_box.addItems(filenamelist)
        self.w5.filename_box.clear()
        self.w5.filename_box.addItems(filenamelist)
        self.w6.filename_box.clear()
        self.w6.filename_box.addItems(filenamelist)
        self.w7.filename_box.clear()
        self.w7.filename_box.addItems(filenamelist)
        self.w8.filename_box.clear()
        self.w8.filename_box.addItems(filenamelist)
        self.w9.filename_box.clear()
        self.w9.filename_box.addItems(filenamelist)
        self.w10.filename_box.clear()
        self.w10.filename_box.addItems(filenamelist)
        self.w11.filename_box.clear()
        self.w11.filename_box.addItems(filenamelist)
        self.w12.filename_box.clear()
        self.w12.filename_box.addItems(filenamelist)

        for f in filenamelist:
            if '1A' in f or 'ch1-short' in f and secret in f:
                self.w1.filename_box.setCurrentText(f)
            if '1B' in f or 'ch1-medium' in f and secret in f:
                self.w2.filename_box.setCurrentText(f)
            if '1C' in f or 'ch1-long' in f and secret in f:
                self.w3.filename_box.setCurrentText(f)
            if '2A' in f or 'ch2-short' in f and secret in f:
                self.w4.filename_box.setCurrentText(f)
            if '2B' in f or 'ch2-medium' in f and secret in f:
                self.w5.filename_box.setCurrentText(f)
            if '2C' in f or 'ch2-long' in f and secret in f:
                self.w6.filename_box.setCurrentText(f)
            if '3A' in f or 'ch3-short' in f and secret in f:
                self.w7.filename_box.setCurrentText(f)
            if '3B' in f or 'ch3-medium' in f and secret in f:
                self.w8.filename_box.setCurrentText(f)
            if '3C' in f or 'ch3-long' in f and secret in f:
                self.w9.filename_box.setCurrentText(f)
            if '4A' in f  or 'ch4-short' in f and secret in f:
                self.w10.filename_box.setCurrentText(f)
            if '4B' in f  or 'ch4-medium' in f and secret in f:
                self.w11.filename_box.setCurrentText(f)
            if '4C' in f  or 'ch4-long' in f and secret in f:
                self.w12.filename_box.setCurrentText(f)

    def comb_dithers(self, click=False, debug = True,   sigma_clip_level = 3):
        filenamelist = self.readfolder(obj_name=self.objname_box.currentText(),dith=True)

        for ch in ['ch1','ch2','ch3','ch4']:
            for band in ['short','medium','long']:

                exp_list_names = []
                if len(filenamelist)>0:
                    for f in filenamelist:
                        if 'dith' in f and ch in f and band in f:
                            exp_list_names.append(f)
                    print(exp_list_names)
                    exp_list = []
                    for f in exp_list_names:
                        d = np.loadtxt(fname=self.spec_folder + f)
                        exp_list.append(spectrum(x=d[:, 0], y=d[:, 1], err=d[:, 2], name=f))

                    if len(exp_list)>0:
                        # scale exposures to one level
                        from lmfit import Model
                        def func(x, scale_factor):
                            return x*scale_factor
                        fmodel = Model(func)
                        for i, s in enumerate(exp_list):
                            if i>0:
                                scale_factor = 1
                                result = fmodel.fit(exp_list[0].y, x=s.y, scale_factor=scale_factor)
                                print(result.fit_report())
                                s_f = result.best_values['scale_factor']
                                s.y *=s_f
                                s.err *=s_f


                        comb = exp_list[0].copy()
                        f = np.array([s.y for s in exp_list])
                        comb.y = np.nanmedian(f,axis=0)
                        comb.std = np.nanstd(np.array([s.y - comb.y for s in exp_list]),axis=0)
                        comb.std_mean = np.nanstd(np.array([s.y-comb.y for s in exp_list]))

                        # mask good pixels by selecting outliers with a threshold of sigma_clip_level sigma
                        mask_exp_good_pixels = np.zeros((len(exp_list),comb.x.shape[0]))
                        for i,s in enumerate(exp_list):
                            mask = np.abs(s.y-comb.y)/comb.std_mean<sigma_clip_level
                            mask_exp_good_pixels[i,:] = mask.copy()

                        if debug:
                            fig,ax = plt.subplots(2,1,sharex=True,sharey=True)
                            for i, s in enumerate(exp_list):
                                ax[0].step(s.x,s.y,ls='-',where='mid')
                                mask = mask_exp_good_pixels[i,:].astype(bool)
                                ax[0].plot(s.x[~mask], s.y[~mask],marker='*',markersize=10)
                            ax[0].step(s.x,comb.y,color='red',lw=2,ls='-',where='mid')
                            ax[0].fill_between(s.x,comb.y-sigma_clip_level*comb.std_mean,comb.y+sigma_clip_level*comb.std_mean,color='red',alpha=0.2)
                            plt.show()

                        f = np.array([s.y for s in exp_list])
                        err = np.array([s.err for s in exp_list])
                        inv = np.power(err,-2)
                        comb.y = (np.nansum(f*inv*mask_exp_good_pixels,axis=0))/np.nansum(inv*mask_exp_good_pixels,axis=0)
                        comb.err = np.power(np.nansum(inv*mask_exp_good_pixels,axis=0),-0.5)

                        mask_bad_pixels = np.sum(mask_exp_good_pixels,axis=0) == 0
                        comb.y[mask_bad_pixels] = 0.0
                        comb.err[mask_bad_pixels] = 1.0
                        if debug:
                            for s in exp_list:
                                ax[1].errorbar(s.x,s.y,yerr=s.err,label=s.name)
                            ax[1].errorbar(comb.x,comb.y,yerr=comb.err,color='black',lw=2)
                            ax[1].set_title(ch+band)
                            plt.legend()
                            plt.show()

                        filename = self.spec_folder + self.objname_box.currentText()+'_combined_'+ch+'-'+band+'_sci.spec1d'
                        with open(filename, 'w') as fout:
                            # for x,y,e in zip(wavel,roi_mean_w_flux,roi_mean_w_f_error):
                            for x, y, e in zip(comb.x, comb.y, comb.err):
                                fout.write('%.4e %.4e %.4e \n' % (x, y, e))
                        fout.close()
        self.setObjName(secret='combined')

    def saveObj(self):
        if hasattr(self,'combined_spec'):
            s = self.combined_spec
            label = s.label
            if 1:
                def savefits(filename='test', wave=[999], flux=[999], err=[999],objname='None',channels='None'):
                    from astropy.io import fits
                    hdr = fits.Header()
                    hdr['TELESCOP'] = 'JWST'
                    hdr['INSTRUME'] = 'MIRI'
                    hdr['AUTHOR'] = 'V.KLIMENKO'
                    hdr['OBJECT'] = objname
                    hdr['CHNNELS'] = channels
                    hdr['COMMENT'] = "This file was created by Spectro"
                    empty_primary = fits.PrimaryHDU(header=hdr)
                    col1 = fits.Column(name='WAVELENGTH', format='D', array=wave)
                    col2 = fits.Column(name='FLUX    ', format='E', array=flux)
                    col3 = fits.Column(name='ERROR    ', format='E', array=err)
                    cols = fits.ColDefs([col1, col2, col3])
                    hdu1 = fits.BinTableHDU.from_columns(cols)
                    hdul = fits.HDUList([empty_primary, hdu1])
                    hdul.writeto(filename + '.fits', overwrite=True)
            #normalization to f at 5 micron
            if 0:
                mask_N = np.abs(s.x-5)<0.1
                factor_N = np.mean(s.y[mask_N])
                s.y /=factor_N
                s.err /= factor_N
            if 1:
                f = './output/specviewer/' + self.save_data_filename.text()
                savefits(f,wave=s.x,flux=s.y,err=s.err,objname=self.objname_box.currentText(),channels=label)
            print("Combined spectrum is saved to ", f)
        else:
            print("Error: Can't save file. There is no combined spectrum")

    def combineChunks(self):
        s = spectrum()
        label = ''
        if self.w1.data != None and self.w1.x != 0:
            s1 = spectrum(x=self.w1.data.x,y=self.w1.data.y*self.w1.x,err=self.w1.data.err*self.w1.x)
            s.append(s1)
            label='CH1'
        if self.w2.data != None and self.w2.x != 0:
            s2 = spectrum(x=self.w2.data.x, y=self.w2.data.y * self.w2.x, err=self.w2.data.err * self.w2.x)
            s.append(s2)
            label += '+CH2'
        if self.w3.data != None and self.w3.x != 0:
            s3 = spectrum(x=self.w3.data.x, y=self.w3.data.y * self.w3.x, err=self.w3.data.err * self.w3.x)
            s.append(s3)
            label += '+CH3'
        if self.w4.data != None and self.w4.x != 0:
            s4 = spectrum(x=self.w4.data.x, y=self.w4.data.y * self.w4.x, err=self.w4.data.err * self.w4.x)
            s.append(s4)
            label += '+CH4'
        if self.w5.data != None and self.w5.x != 0:
            s5 = spectrum(x=self.w5.data.x, y=self.w5.data.y * self.w5.x, err=self.w5.data.err * self.w5.x)
            s.append(s5)
            label += '+CH5'
        if self.w6.data != None and self.w6.x != 0:
            s6 = spectrum(x=self.w6.data.x, y=self.w6.data.y * self.w6.x, err=self.w6.data.err * self.w6.x)
            s.append(s6)
            label += '+CH6'
        if self.w7.data != None and self.w7.x != 0:
            s7 = spectrum(x=self.w7.data.x, y=self.w7.data.y * self.w7.x, err=self.w7.data.err * self.w7.x)
            s.append(s7)
            label += '+CH7'
        if self.w8.data != None and self.w8.x != 0:
            s8 = spectrum(x=self.w8.data.x, y=self.w8.data.y * self.w8.x, err=self.w8.data.err * self.w8.x)
            s.append(s8)
            label += '+CH8'
        if self.w9.data != None and self.w9.x != 0:
            s9 = spectrum(x=self.w9.data.x, y=self.w9.data.y * self.w9.x, err=self.w9.data.err * self.w9.x)
            s.append(s9)
            label += '+CH9'
        if self.w10.data != None and self.w10.x != 0:
            s10 = spectrum(x=self.w10.data.x, y=self.w10.data.y * self.w10.x, err=self.w10.data.err * self.w10.x)
            s.append(s10)
            label += '+CH10'
        if self.w11.data != None and self.w11.x != 0:
            s11 = spectrum(x=self.w11.data.x, y=self.w11.data.y * self.w11.x, err=self.w11.data.err * self.w11.x)
            s.append(s11)
            label += '+CH11'
        if self.w12.data != None and self.w12.x != 0:
            s12 = spectrum(x=self.w12.data.x, y=self.w12.data.y * self.w12.x, err=self.w12.data.err * self.w12.x)
            s.append(s12)
            label += '+CH12'

        self.combined_spec = s.copy()
        self.combined_spec.name = 'Combined'
        self.combined_spec.dq = self.combined_spec.y != 0
        self.combined_spec.label = label

        self.win.plot_spec(fname='Combined', add=False, show_err_bar=True)
        self.win.plot_spec(fname='Combined', fcolor='green', data=self.combined_spec, coef=1, show_err_bar=True)

    def RebinIt(self):
        self.spec_factor = int(self.rebin_n_pix.text())
        self.spec_save = self.combined_spec.copy()
        cumsum = np.cumsum(np.r_[np.zeros((int(self.spec_factor / 2),)), self.spec_save.y, np.zeros(int(self.spec_factor / 2))])
        y = (cumsum[self.spec_factor:] - cumsum[:-self.spec_factor]) / float(self.spec_factor)
        cumsum = np.cumsum(np.r_[np.zeros((int(self.spec_factor / 2),)), self.spec_save.err, np.zeros(int(self.spec_factor / 2))])
        err = (cumsum[self.spec_factor:] - cumsum[:-self.spec_factor]) / float(self.spec_factor) / np.sqrt(float(self.spec_factor))
        self.rebinned_spec = spectrum(self.spec_save.x, y,err,'rebinned')
        self.win.plot_spec(fname='Rebinned', add=False, show_err_bar=False)
        self.win.plot_spec(fname='Rebinned', fcolor='red', data=self.rebinned_spec, coef=1, show_err_bar=False)

    def RecalcStd(self):
        debug=True
        spec_tmp = self.combined_spec.copy()
        npix =spec_tmp.x.shape[0]

        from scipy import signal
        win_size = 50
        win = signal.windows.hann(win_size)
        filtered = signal.convolve(spec_tmp.y, win, mode='same') / sum(win)
        s = np.arange(npix)
        mask = (s > win_size / 2) * (s < len(spec_tmp.x) - win_size / 2)
        spec_tmp.y[mask] = filtered[mask]
        spec_tmp.y[s <= win_size / 2] = np.mean(spec_tmp.y[s < win_size / 2])
        spec_tmp.y[s >= len(spec_tmp.x) - win_size / 2] = np.mean(spec_tmp.y[s >= len(spec_tmp.x) - win_size / 2])

        if debug:
            plt.subplots()
            plt.plot(self.combined_spec.x,self.combined_spec.y,label='combined')
            plt.plot(spec_tmp.x,spec_tmp.y,ls='--',label='convolved')
            plt.show()

        spec_tmp.y =  self.combined_spec.y-spec_tmp.y

        #select outliers:
        outlier_limit = 3*np.std(spec_tmp.y)
        mask_outliers = np.abs(spec_tmp.y)>outlier_limit

        #calc pixels std
        spec_std = np.zeros(npix)
        win = 40
        s = np.arange(npix)
        for i in range(npix):
            mask = (s<=i+win/2)*(s>=i-win/2)*self.combined_spec.dq*(~mask_outliers)
            spec_std[i] =np.nanstd(spec_tmp.y[mask])
        self.combined_spec.err = spec_std

        if debug:
            plt.subplots()
            plt.plot(spec_tmp.x, spec_tmp.y, ls='-')
            plt.plot(spec_tmp.x, spec_std, ls='-',color='red')
            plt.axhline(np.std(spec_tmp.y),color='red',ls='--')
            plt.show()

            plt.subplots()
            plt.hist(spec_tmp.y,bins=np.linspace(-2000,2000,100))
            plt.axvline(np.std(spec_tmp.y))

            if 0:
                from specutils.spectra import Spectrum1D
                from specutils.fitting import fit_lines
                from astropy import units as u
                from astropy.modeling import models
                data,x = np.histogram(spec_tmp.y,bins=np.linspace(-2000,2000,100))
                x = np.delete(x,0,0)
                data = np.array(data+1e-5)
                from astropy import modeling

                fitter = modeling.fitting.LevMarLSQFitter()
                model = modeling.models.Gaussian1D()  # depending on the data you need to give some initial values
                fitted_model = fitter(model, x, data)
                y = fitted_model(x)
                plt.subplots()
                plt.plot(x,data)
                plt.plot(x,y)

            plt.show()

        #self.win.plot_spec(fname='Combined', add=False, show_err_bar=False)
        #self.win.plot_spec(fname='Combined', fcolor='green', data=self.combined_spec, coef=1, show_err_bar=False)

    def readfolder(self,path=None,obj_name='',dith=True):
        if path==None:
           path = self.spec_folder

        lst = []
        for (dirpath, dirname, filenames) in os.walk(path):
            for k, f in enumerate(filenames):
                if dith == False:
                    if f.endswith('_sci.spec1d') and obj_name in f and 'dith' not in f:
                #if f.endswith('_s3d.dat') and obj_name in f:
                        lst.append(f)
                elif dith == True:
                    if f.endswith('_sci.spec1d') and obj_name in f:
                        lst.append(f)
        return sorted(lst)

    def readobjnameinfolder(self,path=None):
        if path==None:
           path = self.spec_folder

        lst = []
        for (dirpath, dirname, filenames) in os.walk(path):
            for k, f in enumerate(filenames):
                if f.endswith('.spec1d'):
                #if f.endswith('_s3d.dat'):
                    objname = f.split('_')[0]
                    if objname not in lst:
                        lst.append(objname)
        return lst

if __name__ == '__main__':

    input_dir ='./output/detector3/roi_spectra/'

    app = QApplication(sys.argv)
    v = Viewer(spec_folder=input_dir)
    v.show()
    sys.exit(app.exec_())