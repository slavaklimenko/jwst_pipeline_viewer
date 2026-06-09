import sys
import glob
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit,QSizePolicy, QSlider, QSpacerItem, QVBoxLayout, QWidget,QComboBox, QPushButton
from PyQt6 import QtGui
from PyQt6.QtGui import QGuiApplication
import pyqtgraph as pg
import numpy as np
import sys, os
from scipy.interpolate import interp1d, UnivariateSpline
from functools import partial
import matplotlib.pyplot as plt
from lmfit import Model

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

    def append(self,s,mode = 'second',debug=False):
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
                    s_interp = interp1d(s.x, s.y, bounds_error=False, fill_value=np.nan)
                    s_interp_err = interp1d(s.x, s.err, bounds_error=False, fill_value=np.nan)
                    mask_selfx_intersection = self.x>=s.x[0]
                    comb = [self.y[mask_selfx_intersection],s_interp(self.x[mask_selfx_intersection])]
                    e_comb = [self.err[mask_selfx_intersection],s_interp_err(self.x[mask_selfx_intersection])]
                    if mode == 'mean weighted':
                        w = np.power(e_comb,-2)
                        f2 = np.nansum(comb * w, axis=0) / np.nansum(w, axis=0)
                        self.y[mask_selfx_intersection] = f2
                        self.err[mask_selfx_intersection] = np.power(np.nansum(w, axis=0), -0.5)
                    elif mode == 'mean':
                        self.y[mask_selfx_intersection] = np.nanmean(comb,axis=0)
                        self.err[mask_selfx_intersection] = np.sqrt(np.nansum(np.power(e_comb, 2), axis=0))
                    elif mode == 'mean disp':
                        self.y[mask_selfx_intersection] = np.nansum(comb,axis=0)/2
                        f = comb-self.y[mask_selfx_intersection]
                        f1 = np.power(f,2)
                        self.err[mask_selfx_intersection] = np.power(np.nansum(f1,axis=0)/2, 0.5)
                    elif mode == 'second':
                        self.y[mask_selfx_intersection] = s_interp(self.x[mask_selfx_intersection])
                        self.err[mask_selfx_intersection] = np.sqrt(np.nansum(np.power(e_comb, 2), axis=0))

                self.x = np.append(self.x,s.x[mask_extension])
                self.y = np.append(self.y,s.y[mask_extension])
                self.err = np.append(self.err,s.err[mask_extension])


    def copy(self):
        return spectrum(np.array(self.x),np.array(self.y),np.array(self.err))



class Slider(QWidget):
    def __init__(self, minimum, maximum, parent=None,name='ch1A:',filenamelist=['file'],path ='',val=1):
        super(Slider, self).__init__(parent=parent)
        self.name = name
        self.spec_files_path = path
        self.verticalLayout = QVBoxLayout(self)

        self.horizontalLayout0 = QHBoxLayout(self)
        self.horizontalLayout0.addWidget(QLabel('f:'))
        self.filename_box = QComboBox(self)
        self.filename_box.addItems(filenamelist)
        self.filename_box.setCurrentIndex(0)
        #self.filename_box.setFixedSize(200, 30)
        self.filename_box.resize(200, 30)

        self.horizontalLayout0.addWidget(self.filename_box)
        self.verticalLayout.addLayout(self.horizontalLayout0)


        self.horizontalLayout1 = QHBoxLayout(self)
        self.horizontalLayout1.addWidget(QLabel(name))
        self.value_label = QLineEdit(self)
        self.value_label.setText(str(val))
        #self.value_label.setFixedSize(60, 30)
        self.value_label.resize(60, 30)
        self.horizontalLayout1.addWidget(self.value_label)
        #self.value_suggested_label = QLineEdit(self)
        #self.value_suggested_label.setText(str(val))
        #self.value_suggested_label.setFixedSize(60, 30)
        #self.horizontalLayout1.addWidget(self.value_suggested_label)
        self.horizontalLayout1.addStretch(1)
        self.verticalLayout.addLayout(self.horizontalLayout1)


        self.horizontalLayout1 = QHBoxLayout(self)
        self.horizontalLayout1.addWidget(QLabel('mod:'))
        self.value_suggested_label = QLineEdit(self)
        self.value_suggested_label.setText(str(val))
        #self.value_suggested_label.setFixedSize(90, 30)
        self.value_suggested_label.resize(90, 30)
        self.horizontalLayout1.addWidget(self.value_suggested_label)
        self.horizontalLayout1.addStretch(1)
        self.verticalLayout.addLayout(self.horizontalLayout1)


        self.up_label = QLineEdit(self)
        self.up_label.setText(str(maximum))
        #self.up_label.setFixedSize(60, 30)
        self.up_label.resize(60, 30)
        self.verticalLayout.addWidget(self.up_label)
        self.horizontalLayout = QHBoxLayout(self)
        #spacerItem = QSpacerItem(0, 100, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacerItem = QSpacerItem(
            0,
            100,
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        #self.slider.setOrientation(Qt.Vertical)
        self.slider.setOrientation(Qt.Orientation.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        #spacerItem1 = QSpacerItem(40, 80, QSizePolicy.Expanding, QSizePolicy.Minimum)
        spacerItem1 = QSpacerItem(
            40,
            80,
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.low_label = QLineEdit(self)
        self.low_label.setText(str(minimum))
        #self.low_label.setFixedSize(60, 30)
        self.low_label.resize(60, 30)
        self.verticalLayout.addWidget(self.low_label)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = val
        self.setLabelValue(self.slider.value())


        self.read_spectrum()
        #self.filename_box.currentIndexChanged.connect(self.read_spectrum(f=f))

    def helperSetSliderIntValue(self, x):
        self.slider.tracking = True
        self.slider.value = x
        self.slider.sliderPosition = x
        self.slider.update()
        self.slider.repaint()

    def update_suggested_value(self,x):
        self.value_suggested_label.setText(" %.2f" % (x))

    def setLabelValue(self, value):
        self.setMinMaxLimits()
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (self.maximum - self.minimum)

        #self.label.setText("xmin:{0:.4g}".format(self.x))

    def setMinMaxLimits(self):
        self.maximum = float(self.up_label.text())
        self.minimum = float(self.low_label.text())


    def read_spectrum(self,flag=None,debug=True):
        f =  self.filename_box.currentText()
        if f not in ['None','']:
            print('read file',self.spec_files_path + f)
            d = np.loadtxt(fname=self.spec_files_path + f)
            self.data = spectrum(x=np.array(d[:,0]),y=np.array(d[:,1]),err=np.array(d[:,2]),name=f)
        else:
            self.data = None


class plotSpec(pg.PlotWidget):
    def __init__(self, parent):
        self.parent = parent
        pg.PlotWidget.__init__(self, background=(29, 29, 29), labels={'left': 'SB (Jy)', 'bottom': 'Wavelength [micron]'})
        self.vb = self.getViewBox()
        self.text = None

        self.legend = pg.LegendItem(offset=(-70, 30))
        self.legend.setParentItem(self.vb)
        self.legend_model = pg.LegendItem(offset=(-70, -30))
        self.legend_model.setParentItem(self.vb)
        self.setTitle("Pixel difference", color="olive", size="10pt")
        self.lines = self.listDataItems()
        self.spectra_list = {}
        self.plot_errbar = {}
        self.spec_line_colors = {}

    def plot_spec(self, fname = None, fcolor='lightgreen',data=None, coef =1 ,add=True,show_err_bar=False):
        if fname != None:
            if add:
                scale_mode = self.parent.scale_mode_win_option.currentText()
                x = data.x
                y = data.y.copy()
                err = data.err.copy()
                if coef!= 1:
                    if scale_mode== 'multiply':
                        y*=coef
                        err*=coef
                    elif scale_mode == 'add':
                        y+=coef*y[0]

                self.spec_line_colors[fname] = fcolor
                pen = pg.mkPen(color=self.spec_line_colors[fname],  style=Qt.PenStyle.SolidLine, width=2)
                self.spectra_list[fname] = pg.PlotCurveItem(x, y,pen=pen)

                #self.plot_lineA1 = pg.PlotCurveItem(x, y,pen='lightgreen')
                #self.plot_errbarA1 = pg.ErrorBarItem(x=wavel,y=data,height=err,pen=pen, beam=1/6000)
                self.vb.addItem(self.spectra_list[fname])
                self.legend_model.addItem(self.spectra_list[fname], fname)
                if show_err_bar:
                    self.plot_errbar[fname] = pg.ErrorBarItem(x=x,y=y,height=err,pen=pen, beam=1/6000)
                    self.vb.addItem(self.plot_errbar[fname])
                #    pen = pg.mkPen(color='darkgray', style=Qt.DashLine, width=1)
                    #self.zero_level = pg.PlotCurveItem([wavel[0]-2, wavel[-1] + 2], [0, 0], pen=pen)
                #self.zero_level = pg.PlotCurveItem([0, 30], [0, 0], pen=pg.mkPen(color='darkgray', style=Qt.DashLine, width=1))
                self.zero_level = pg.PlotCurveItem(
                    [0, 30],
                    [0, 0],
                    pen=pg.mkPen(
                        color='darkgray',
                        style=Qt.PenStyle.DashLine,
                        width=1
                    )
                )
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
                        self.vb.removeItem(self.plot_errbar[fname])
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
        #self.objname_box.setFixedSize(200, 30)
        self.objname_box.resize(200, 30)
        self.horizontalLayout.addWidget(self.objname_box)
        self.obj_name_win = QPushButton('ReadSpecList')
        self.obj_name_win.clicked[bool].connect(self.setObjName)
        #self.obj_name_win.setFixedSize(200, 60)
        self.obj_name_win.resize(200, 60)
        self.horizontalLayout.addWidget(self.obj_name_win)
        #self.read_win_option = QComboBox()
        #self.read_win_option.addItems(['subtract_bkgr', 'add_to_err','None'])
        #self.read_win_option.setCurrentIndex(2)
        #self.horizontalLayout.addWidget(self.read_win_option)
        self.comb_dithers_win = QPushButton('CombineDithers')
        self.comb_dithers_win.clicked.connect(self.comb_dithers)
        #self.comb_dithers_win.setFixedSize(250, 60)
        self.comb_dithers_win.resize(250, 60)
        self.horizontalLayout.addWidget(self.comb_dithers_win)
        self.combine_dithers_win_option = QComboBox()
        self.combine_dithers_win_option.addItems(['green','red'])
        self.combine_dithers_win_option.setCurrentIndex(0)
        cb = self.combine_dithers_win_option
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        self.horizontalLayout.addWidget(self.combine_dithers_win_option)
        self.calc_scaling_win = QPushButton('ScaleCh')
        #self.build_cube.clicked[bool].connect(partial(self.call_build_3dCube))
        self.calc_scaling_win.clicked[bool].connect(partial(self.calcChunkCoeffs))
        #self.calc_scaling_win.setFixedSize(200, 60)
        self.calc_scaling_win.resize(200, 60)
        self.horizontalLayout.addWidget(self.calc_scaling_win)
        self.scale_mode_win_option = QComboBox()
        self.scale_mode_win_option.addItems(['multiply', 'add'])
        self.scale_mode_win_option.setCurrentIndex(0)
        cb = self.scale_mode_win_option
        width = cb.minimumSizeHint().width()
        cb.setFixedWidth(width)
        self.horizontalLayout.addWidget(self.scale_mode_win_option)

        self.save_data_win = QPushButton('SaveSpec')
        #self.build_cube.clicked[bool].connect(partial(self.call_build_3dCube))
        self.save_data_win.clicked[bool].connect(partial(self.saveObj))
        #self.save_data_win.setFixedSize(200, 60)
        self.save_data_win.resize(200, 60)
        self.horizontalLayout.addWidget(self.save_data_win)
        self.save_data_filename = QLineEdit()
        self.save_data_filename.setText('filename')
        #self.save_data_filename.setFixedSize(150, 30)
        self.save_data_filename.resize(150, 30)
        self.horizontalLayout.addWidget(self.save_data_filename)
        self.combine_win = QPushButton('Combine')
        self.combine_win.clicked[bool].connect(partial(self.combineChunks))
        #self.combine_win.setFixedSize(200, 60)
        self.combine_win.resize(200, 60)
        self.horizontalLayout.addWidget(self.combine_win)
        self.rebin_win = QPushButton('Rebin')
        # self.build_cube.clicked[bool].connect(partial(self.call_build_3dCube))
        self.rebin_win.clicked[bool].connect(partial(self.RebinIt))
        #self.rebin_win.setFixedSize(200, 60)
        self.rebin_win.resize(200, 60)
        self.horizontalLayout.addWidget(self.rebin_win)
        self.rebin_n_pix = QLineEdit()
        self.rebin_n_pix.setText('4')
        #self.rebin_n_pix.setFixedSize(100, 30)
        self.rebin_n_pix.resize(30, 30)
        self.horizontalLayout.addWidget(self.rebin_n_pix)
        self.horizontalLayout.addWidget(QLabel('Smooth:'))
        self.smooth_n_pix = QLineEdit()
        self.smooth_n_pix.setText('-1')
        #self.smooth_n_pix.setFixedSize(100, 30)
        self.smooth_n_pix.resize(100, 30)
        self.horizontalLayout.addWidget(self.smooth_n_pix)
        self.recalc_errorbar =  QPushButton('RecalcStd')
        self.recalc_errorbar.clicked[bool].connect(partial(self.RecalcStd))
        #self.recalc_errorbar.setFixedSize(200, 60)
        self.recalc_errorbar.resize(200, 60)
        self.horizontalLayout.addWidget(self.recalc_errorbar)
        self.leak_correction_button = QPushButton('Leak_corr')
        self.leak_correction_button.clicked[bool].connect(partial(self.ApplyLeakCorr))
        self.leak_correction_button.resize(200, 60)
        self.horizontalLayout.addWidget(self.leak_correction_button)
        self.leak_coeff = QLineEdit()
        self.leak_coeff.setText('1')
        self.leak_coeff.resize(10, 30)
        self.horizontalLayout.addWidget(self.leak_coeff)
        self.horizontalLayout.addStretch(1)
        self.mainLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout = QHBoxLayout(self)
        if 1:
            filenamelist = ['None']+self.readfolder(obj_name=self.objname_box.currentText())
            #bkgr_option = self.read_win_option.currentText()
            self.w1 = Slider(0, 1, name='ch1A:', filenamelist=filenamelist, path=self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w1)

            self.w2 = Slider(0, 1,name='ch1B:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w2)

            self.w3 = Slider(0, 1,name='ch1C:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w3)

            self.w4 = Slider(0, 1,name='ch2A:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w4)

            self.w5 = Slider(0, 1, name='ch2B:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w5)

            self.w6 = Slider(0, 1, name='ch2C:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w6)

            self.w7 = Slider(0, 1, name='ch3A:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w7)

            self.w8 = Slider(0, 1, name='ch3B:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w8)

            self.w9 = Slider(0, 1, name='ch3C:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w9)

            self.w10 = Slider(0, 1, name='ch4A:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w10)

            self.w11 = Slider(0, 1, name='ch4B:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w11)

            self.w12 = Slider(0, 1, name='ch4C:',filenamelist=filenamelist,path =  self.spec_folder,val=1)
            self.horizontalLayout.addWidget(self.w12)


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


    def update_slider(self):
        self.update_plot()
        self.update_val_labels()
    def update_plot(self,update_val_label=False):
        if self.w1.data != None:
            self.win.plot_spec(fname='CH1A', add=False, show_err_bar=True)
            self.win.plot_spec(fname='CH1A',fcolor='blue',data=self.w1.data, coef=1,   show_err_bar=True)
        if self.w2.data != None:
            self.win.plot_spec(fname='CH1B', add=False,show_err_bar=True)
            self.win.plot_spec(fname='CH1B',fcolor='orange',data=self.w2.data, coef=self.w2.x,   show_err_bar=True)
        if self.w3.data != None:
            self.win.plot_spec(fname='CH1C', add=False,show_err_bar=False)
            self.win.plot_spec(fname='CH1C',fcolor='green',data=self.w3.data, coef=self.w3.x,  show_err_bar=False)
        if self.w4.data != None:
            self.win.plot_spec(fname='CH2A', add=False, show_err_bar=False)
            self.win.plot_spec(fname='CH2A', fcolor='red', data=self.w4.data, coef=self.w4.x, show_err_bar=False)
        if self.w5.data != None:
            self.win.plot_spec(fname='CH2B', add=False, show_err_bar=False)
            self.win.plot_spec(fname='CH2B', fcolor='purple', data=self.w5.data, coef=self.w5.x,  show_err_bar=False)
        if self.w6.data != None:
            self.win.plot_spec(fname='CH2C', add=False, show_err_bar=False)
            self.win.plot_spec(fname='CH2C', fcolor='brown', data=self.w6.data, coef=self.w6.x, show_err_bar=False)
        if self.w7.data != None:
            self.win.plot_spec(fname='CH3A', add=False, show_err_bar=False)
            self.win.plot_spec(fname='CH3A', fcolor='pink', data=self.w7.data, coef=self.w7.x,  show_err_bar=False)
        if self.w8.data != None:
            self.win.plot_spec(fname='CH3B', add=False, show_err_bar=False)
            self.win.plot_spec(fname='CH3B', fcolor='gray', data=self.w8.data, coef=self.w8.x, show_err_bar=False)
        if self.w9.data != None:
            self.win.plot_spec(fname='CH3C', add=False, show_err_bar=False)
            self.win.plot_spec(fname='CH3C', fcolor='olive', data=self.w9.data, coef=self.w9.x,  show_err_bar=False)
        if self.w10.data != None:
            self.win.plot_spec(fname='CH4A', add=False, show_err_bar=False)
            self.win.plot_spec(fname='CH4A', fcolor='cyan', data=self.w10.data, coef=self.w10.x, show_err_bar=False)
        if self.w11.data != None:
            self.win.plot_spec(fname='CH4B', add=False, show_err_bar=False)
            self.win.plot_spec(fname='CH4B', fcolor='magenta', data=self.w11.data, coef=self.w11.x,  show_err_bar=False)

        if self.w12.data != None:
            self.win.plot_spec(fname='CH4C', add=False, show_err_bar=False)
            self.win.plot_spec(fname='CH4C', fcolor='yellow', data=self.w12.data, coef=self.w12.x,  show_err_bar=False)

        #x = np.linspace(0, 10, 100)
        #data = a + np.cos(x + c * np.pi / 180) * np.exp(-b * x) * d
        if update_val_label:
            self.w1.value_label.setText("%.2f" % (self.w1.x))
            self.w2.value_label.setText("%.2f" % (self.w2.x))
            self.w3.value_label.setText("%.2f" % (self.w3.x))
            self.w4.value_label.setText("%.2f" % (self.w4.x))
            self.w5.value_label.setText("%.2f" % (self.w5.x))
            self.w6.value_label.setText("%.2f" % (self.w6.x))
            self.w7.value_label.setText("%.2f" % (self.w7.x))
            self.w8.value_label.setText("%.2f" % (self.w8.x))
            self.w9.value_label.setText("%.2f" % (self.w9.x))
            self.w10.value_label.setText("%.2f" % (self.w10.x))
            self.w11.value_label.setText("%.2f" % (self.w11.x))
            self.w12.value_label.setText("%.2f" % (self.w12.x))

        #self.curve.setData(data)

    def update_val_labels(self, update_val_label=True):
        self.w1.value_label.setText("%.2f" % (self.w1.x))
        self.w2.value_label.setText("%.2f" % (self.w2.x))
        self.w3.value_label.setText("%.2f" % (self.w3.x))
        self.w4.value_label.setText("%.2f" % (self.w4.x))
        self.w5.value_label.setText("%.2f" % (self.w5.x))
        self.w6.value_label.setText("%.2f" % (self.w6.x))
        self.w7.value_label.setText("%.2f" % (self.w7.x))
        self.w8.value_label.setText("%.2f" % (self.w8.x))
        self.w9.value_label.setText("%.2f" % (self.w9.x))
        self.w10.value_label.setText("%.2f" % (self.w10.x))
        self.w11.value_label.setText("%.2f" % (self.w11.x))
        self.w12.value_label.setText("%.2f" % (self.w12.x))

    def setObjName(self,click=1,secret=''):
        read_mode = self.combine_dithers_win_option.currentText()
        if read_mode == 'green':
            keyname = '_green.spec1d'
        if read_mode == 'red':
            keyname = '_red.spec1d'

        filenamelist = ['None'] + self.readfolder(obj_name=self.objname_box.currentText(),keyname=keyname)

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
            if secret in f:
                if '1A' in f or 'ch1-short' in f:
                    self.w1.filename_box.setCurrentText(f)
                if '1B' in f or 'ch1-medium' in f:
                    self.w2.filename_box.setCurrentText(f)
                if '1C' in f or 'ch1-long' in f:
                    self.w3.filename_box.setCurrentText(f)
                if '2A' in f or 'ch2-short' in f:
                    self.w4.filename_box.setCurrentText(f)
                if '2B' in f or 'ch2-medium' in f:
                    self.w5.filename_box.setCurrentText(f)
                if '2C' in f or 'ch2-long' in f:
                    self.w6.filename_box.setCurrentText(f)
                if '3A' in f or 'ch3-short' in f:
                    self.w7.filename_box.setCurrentText(f)
                if '3B' in f or 'ch3-medium' in f:
                    self.w8.filename_box.setCurrentText(f)
                if '3C' in f or 'ch3-long' in f:
                    self.w9.filename_box.setCurrentText(f)
                if '4A' in f  or 'ch4-short' in f:
                    self.w10.filename_box.setCurrentText(f)
                if '4B' in f  or 'ch4-medium' in f:
                    self.w11.filename_box.setCurrentText(f)
                if '4C' in f  or 'ch4-long' in f:
                    self.w12.filename_box.setCurrentText(f)

    def comb_dithers(self, click=False, debug = True,   sigma_clip_level = 3, keyname='',method = 'mean'):
        read_mode = self.combine_dithers_win_option.currentText()
        if read_mode == 'green':
            keyname = '_green.spec1d'
        if read_mode == 'red':
            keyname = '_red.spec1d'


        filenamelist = self.readfolder(obj_name=self.objname_box.currentText(),dith=True,keyname=keyname)



        for ch in ['ch1','ch2','ch3','ch4']:
            for band in ['short','medium','long']:

                exp_list_names = []
                if len(filenamelist)>0:
                    for f in filenamelist:
                        if 'dith' in f and ch in f and band in f:
                            exp_list_names.append(f)
                    print('exp_list_names',exp_list_names)
                    exp_list = []
                    for f in exp_list_names:
                        d = np.loadtxt(fname=self.spec_folder + f)
                        s1 = spectrum(x=d[:, 0], y=d[:, 1], err=d[:, 2], name=f)
                        exp_list.append(s1)

                    if len(exp_list)>0:
                        # scale exposures
                        flux = np.array([s.y for s in exp_list])
                        mean_flux = np.nanmean(flux,axis=0)
                        npix = mean_flux.shape[0]

                        def func(x, scale_factor):
                            if scale_factor<0:
                                scale_factor = 1
                            return x*scale_factor
                        fmodel = Model(func)
                        for i, s in enumerate(exp_list):
                            if 1:
                                scale_factor = 1
                                mask_nan = np.isnan(s.y) + np.isnan(mean_flux) + (np.arange(npix)>0.5*npix)
                                result = fmodel.fit(mean_flux[~mask_nan], x=s.y[~mask_nan], scale_factor=scale_factor)
                                s_f = result.best_values['scale_factor']
                                if s_f<0.5:
                                    s_f = 1
                                s.y *=s_f
                                s.err *=s_f
                                print(i,s.name,s_f)


                        comb = exp_list[0].copy()
                        flux = np.array([s.y for s in exp_list])
                        comb.y = np.nanmean(flux,axis=0)
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
                                ax[0].step(s.x,s.y,ls='-',where='mid',label=s.name)
                                mask = mask_exp_good_pixels[i,:].astype(bool)
                                ax[0].plot(s.x[~mask], s.y[~mask],marker='*',markersize=10)
                            ax[0].step(s.x,comb.y,color='black',lw=2,ls='-',where='mid',label='combined')
                            ax[0].fill_between(s.x,comb.y-sigma_clip_level*comb.std_mean,comb.y+sigma_clip_level*comb.std_mean,color='red',alpha=0.2)
                            ax[0].legend()
                            plt.show()

                        f = np.array([s.y for s in exp_list])
                        err = np.array([s.err for s in exp_list])
                        inv = np.power(err,-2)
                        if method == 'mean':
                            comb.y = (np.nansum(f*mask_exp_good_pixels,axis=0))/np.nansum(mask_exp_good_pixels,axis=0)
                            comb.err = np.sqrt(np.nansum(np.power(err,2)*mask_exp_good_pixels,axis=0))
                        elif method == 'mean weighted':
                            comb.y =  (np.nansum(f*inv*mask_exp_good_pixels,axis=0))/np.nansum(inv*mask_exp_good_pixels,axis=0)
                            comb.err = np.power(np.nansum(inv * mask_exp_good_pixels, axis=0), -0.5)

                        mask_area_without_good_pixels = np.sum(mask_exp_good_pixels,axis=0) == 0
                        if 1:
                            #exclude these points
                            comb.x = np.delete(comb.x,mask_area_without_good_pixels)
                            comb.y = np.delete(comb.y,mask_area_without_good_pixels)
                            comb.err = np.delete(comb.err,mask_area_without_good_pixels)
                        else:
                            #set as zero
                            comb.y[mask_area_without_good_pixels] = 0.0
                            comb.err[mask_area_without_good_pixels] = 1.0
                        if debug:
                            for s in exp_list:
                                ax[1].errorbar(s.x,s.y,yerr=s.err,label=s.name,capsize=10,capthick=4)
                            ax[1].errorbar(comb.x,comb.y,yerr=comb.err,color='black',lw=2,label='combined')
                            ax[1].set_title(ch+band)
                            ax[1].legend()
                            plt.show()

                        filename = self.spec_folder + self.objname_box.currentText()+'_combined_'+ch+'-'+band+keyname
                        with open(filename, 'w') as fout:
                            # for x,y,e in zip(wavel,roi_mean_w_flux,roi_mean_w_f_error):
                            for x, y, e in zip(comb.x, comb.y, comb.err):
                                fout.write('%.4e %.4e %.4e \n' % (x, y, e))
                        fout.close()
        self.setObjName(secret='combined')



    def saveObj(self):
        if hasattr(self,'combined_spec'):
            s = self.combined_spec
            s_orig = self.combined_spec
            chunks = []
            scaling_coeffs = []
            if hasattr(self, 'rebinned_spec'):
                s = self.rebinned_spec
            if 1:
                if self.w1.data != None:
                    chunks.append(self.w1.data)
                    scaling_coeffs.append(self.w1.x)
                if self.w2.data != None:
                    chunks.append(self.w2.data)
                    scaling_coeffs.append(self.w2.x)
                if self.w3.data != None:
                    chunks.append(self.w3.data)
                    scaling_coeffs.append(self.w3.x)
                if self.w4.data != None:
                    chunks.append(self.w4.data)
                    scaling_coeffs.append(self.w4.x)
                if self.w5.data != None:
                    chunks.append(self.w5.data)
                    scaling_coeffs.append(self.w5.x)
                if self.w6.data != None:
                    chunks.append(self.w6.data)
                    scaling_coeffs.append(self.w6.x)
                if self.w7.data != None:
                    chunks.append(self.w7.data)
                    scaling_coeffs.append(self.w7.x)
                if self.w8.data != None:
                    chunks.append(self.w8.data)
                    scaling_coeffs.append(self.w8.x)
                if self.w9.data != None:
                    chunks.append(self.w9.data)
                    scaling_coeffs.append(self.w9.x)
                if self.w10.data != None:
                    chunks.append(self.w10.data)
                    scaling_coeffs.append(self.w10.x)
                if self.w11.data != None:
                    chunks.append(self.w11.data)
                    scaling_coeffs.append(self.w11.x)
                if self.w12.data != None:
                    chunks.append(self.w12.data)
                    scaling_coeffs.append(self.w12.x)

            label = s_orig.label
            if 1:

                def savefits(filename='test', wave=[999], flux=[999], err=[999],wave_full = [999],flux_full = [999],err_full = [999],
                             objname='None',channels='None',spec=s,write_chunks=True,scailing='None'):
                    from astropy.io import fits
                    hdr = fits.Header()
                    hdr['TELESCOP'] = 'JWST'
                    hdr['INSTRUME'] = 'MIRI'
                    hdr['AUTHOR'] = 'V.KLIMENKO'
                    hdr['OBJECT'] = objname
                    hdr['CHNNELS'] = channels
                    hdr['SCALING'] = scailing
                    empty_primary = fits.PrimaryHDU(header=hdr)
                    col1 = fits.Column(name='WAVELENGTH', format='D', array=wave)
                    col2 = fits.Column(name='FLUX    ', format='E', array=flux)
                    col3 = fits.Column(name='ERROR    ', format='E', array=err)
                    col4 = fits.Column(name='SCALE OF MIRI CHANNELS', format='E', array=spec.scale_parameter_list)
                    col5 = fits.Column(name='COEFFS', format='E', array=scaling_coeffs)
                    cols = fits.ColDefs([col1, col2, col3,col4,col5])
                    hdu1 = fits.BinTableHDU.from_columns(cols,name='SCI')
                    col6 = fits.Column(name='WAVELENGTH_ORIGBINNING', format='D', array=wave_full)
                    col7 = fits.Column(name='FLUX_ORIGBINNING', format='E', array=flux_full)
                    col8 = fits.Column(name='ERROR_ORIGBINNING', format='E', array=err_full)

                    cols_orig = fits.ColDefs([col6, col7, col8])
                    hdu2 = fits.BinTableHDU.from_columns(cols_orig,name='SCI_ORIG_BINNING')

                    lst = [empty_primary, hdu1, hdu2]
                    if write_chunks==True:
                        for i,labeli in zip(np.arange(12),['CH1A','CH1B','CH1C','CH2A','CH2B','CH2C','CH3A','CH3B','CH3C','CH4A','CH4B','CH4C']):
                            ch1ax = fits.Column(name='WAVELENGTH', format='D', array=chunks[i].x)
                            ch1ay = fits.Column(name='FLUX', format='E', array=chunks[i].y)
                            ch1aerr = fits.Column(name='ERROR', format='E', array=chunks[i].err)
                            cols = fits.ColDefs([ch1ax, ch1ay, ch1aerr])
                            hdu3 = fits.BinTableHDU.from_columns(cols,name=labeli)
                            lst.append(hdu3)
                    hdul = fits.HDUList(lst)
                    hdul.writeto(filename + '.fits', overwrite=True)
            #normalization to f at 5 micron
            if 1:
                f = './output/specviewer/' + self.save_data_filename.text()
                scailing = self.scale_mode_win_option.currentText()
                savefits(f,wave=s.x,flux=s.y,err=s.err,wave_full=s_orig.x,flux_full=s_orig.y,err_full=s_orig.err,objname=self.objname_box.currentText(),channels=label,spec=s_orig,scailing=scailing)
            print("Combined spectrum is saved to ", f)
        else:
            print("Error: Can't save file. There is no combined spectrum")

    def combineChunks(self):
        s = spectrum()
        label = ''
        scale_parameter_list= []

        scaling_mode = self.scale_mode_win_option.currentText()
        if scaling_mode == 'multiply':
            def calc_y(x, scale_factor):
                return x * scale_factor
            def calc_err(x, scale_factor):
                return x * scale_factor

        elif scaling_mode == 'add':
            def calc_y(x, add_factor):
                return x + add_factor*x[0]
            def calc_err(x, scale_factor):
                return x


        if self.w1.data != None:
            s1 = spectrum(x=self.w1.data.x,y=calc_y(self.w1.data.y,self.w1.x),err=calc_err(self.w1.data.err,self.w1.x))
            s.append(s1)
            scale_parameter_list.append(self.w1.x)
            label='CH1'
        if self.w2.data != None:
            s2 = spectrum(x=self.w2.data.x, y=calc_y(self.w2.data.y, self.w2.x), err=calc_err(self.w2.data.err, self.w2.x))
            s.append(s2)
            scale_parameter_list.append(self.w2.x)
            label += '+CH2'
        if self.w3.data != None and self.w3.x:
            s3 = spectrum(x=self.w3.data.x, y=calc_y(self.w3.data.y , self.w3.x), err=calc_err(self.w3.data.err, self.w3.x))
            s.append(s3)
            scale_parameter_list.append(self.w3.x)
            label += '+CH3'
        if self.w4.data != None and self.w4.x:
            s4 = spectrum(x=self.w4.data.x, y=calc_y(self.w4.data.y, self.w4.x), err=calc_err(self.w4.data.err, self.w4.x))
            s.append(s4)
            scale_parameter_list.append(self.w4.x)
            label += '+CH4'
        if self.w5.data != None and self.w5.x:
            s5 = spectrum(x=self.w5.data.x, y=calc_y(self.w5.data.y, self.w5.x), err=calc_err(self.w5.data.err, self.w5.x))
            s.append(s5)
            scale_parameter_list.append(self.w5.x)
            label += '+CH5'
        if self.w6.data != None and self.w6.x:
            s6 = spectrum(x=self.w6.data.x, y=calc_y(self.w6.data.y, self.w6.x), err=calc_err(self.w6.data.err, self.w6.x))
            s.append(s6)
            scale_parameter_list.append(self.w6.x)
            label += '+CH6'
        if self.w7.data != None and self.w7.x:
            s7 = spectrum(x=self.w7.data.x, y=calc_y(self.w7.data.y, self.w7.x), err=calc_err(self.w7.data.err,  self.w7.x))
            s.append(s7)
            scale_parameter_list.append(self.w7.x)
            label += '+CH7'
        if self.w8.data != None and self.w8.x:
            s8 = spectrum(x=self.w8.data.x, y=calc_y(self.w8.data.y, self.w8.x), err=calc_err(self.w8.data.err,  self.w8.x))
            s.append(s8)
            scale_parameter_list.append(self.w8.x)
            label += '+CH8'
        if self.w9.data != None and self.w9.x:
            s9 = spectrum(x=self.w9.data.x, y=calc_y(self.w9.data.y, self.w9.x), err=calc_err(self.w9.data.err, self.w9.x))
            s.append(s9)
            scale_parameter_list.append(self.w9.x)
            label += '+CH9'
        if self.w10.data != None and self.w10.x:
            s10 = spectrum(x=self.w10.data.x, y=calc_y(self.w10.data.y, self.w10.x), err=calc_err(self.w10.data.err, self.w10.x))
            s.append(s10)
            scale_parameter_list.append(self.w10.x)
            label += '+CH10'
        if self.w11.data != None and self.w11.x:
            s11 = spectrum(x=self.w11.data.x, y=calc_y(self.w11.data.y, self.w11.x), err=calc_err(self.w11.data.err, self.w11.x))
            s.append(s11)
            scale_parameter_list.append(self.w11.x)
            label += '+CH11'
        if self.w12.data != None and self.w12.x:
            s12 = spectrum(x=self.w12.data.x, y=calc_y(self.w12.data.y, self.w12.x), err=calc_err(self.w12.data.err, self.w12.x))
            s.append(s12)
            scale_parameter_list.append(self.w12.x)
            label += '+CH12'

        self.combined_spec = s.copy()
        self.combined_spec.name = 'Combined'
        #self.combined_spec.dq = self.combined_spec.y != 0
        self.combined_spec.label = label
        self.combined_spec.scale_parameter_list = scale_parameter_list

        self.win.plot_spec(fname='Combined', add=False, show_err_bar=True)
        self.win.plot_spec(fname='Combined', fcolor='green', data=self.combined_spec, coef=1, show_err_bar=True)

    def calcChunkCoeffs(self,flag,debug=True):
        s = spectrum()
        label = ''
        scale_parameter_list= []
        if self.w1.data != None:
            s1 = spectrum(x=self.w1.data.x,y=self.w1.data.y,err=self.w1.data.err)
        if self.w2.data != None:
            s2 = spectrum(x=self.w2.data.x, y=self.w2.data.y, err=self.w2.data.err)
        if self.w3.data != None:
            s3 = spectrum(x=self.w3.data.x, y=self.w3.data.y, err=self.w3.data.err)
        if self.w4.data != None:
            s4 = spectrum(x=self.w4.data.x, y=self.w4.data.y, err=self.w4.data.err)
        if self.w5.data != None:
            s5 = spectrum(x=self.w5.data.x, y=self.w5.data.y, err=self.w5.data.err)
        if self.w6.data != None:
            s6 = spectrum(x=self.w6.data.x, y=self.w6.data.y, err=self.w6.data.err)
        if self.w7.data != None:
            s7 = spectrum(x=self.w7.data.x, y=self.w7.data.y, err=self.w7.data.err)
        if self.w8.data != None:
            s8 = spectrum(x=self.w8.data.x, y=self.w8.data.y, err=self.w8.data.err)
        if self.w9.data != None:
            s9 = spectrum(x=self.w9.data.x, y=self.w9.data.y, err=self.w9.data.err)
        if self.w10.data != None:
            s10 = spectrum(x=self.w10.data.x, y=self.w10.data.y, err=self.w10.data.err)
        if self.w11.data != None:
            s11 = spectrum(x=self.w11.data.x, y=self.w11.data.y, err=self.w11.data.err)
        if self.w12.data != None:
            s12 = spectrum(x=self.w12.data.x, y=self.w12.data.y, err=self.w12.data.err)

        def calc_coeff(sa=s1, sb=s2,scaling_mode = self.scale_mode_win_option.currentText(),debug=False):
            mask_sa = (sa.x > sb.x[0]) * (~np.isnan(sa.y)) * (sa.y != 0)*(np.abs(sa.y-np.nanmean(sa.y))<4*np.nanstd(sa.y))
            mask_sb = (sb.x < sa.x[-1]) * (~np.isnan(sb.y)) * (sb.y != 0)*(np.abs(sb.y-np.nanmean(sb.y))<4*np.nanstd(sb.y))
            fa = interp1d(sa.x[mask_sa],sa.y[mask_sa],fill_value='extrapolate')
            fb = interp1d(sb.x[mask_sb],sb.y[mask_sb],fill_value='extrapolate')
            #z = np.polyfit(sa.x[sa.x<sb.x[0]], sa.y[sa.x<sb.x[0]], 3)
            #p = np.poly1d(z)
            #fa_cont = interp1d(sa.x[sa.x<sb.x[0]],p(sa.x[sa.x<sb.x[0]]),fill_value='extrapolate')
            xcommonrange = sa.x[mask_sa]
            #plt.subplots()
            #plt.plot(sa.x,sa.y,label='speca',ls='--')
            #plt.plot(sb.x, sb.y, label='specb',ls='--')
            #plt.plot(xcommonrange,fa(xcommonrange),label='fa')
            #plt.plot(xcommonrange,fa_cont(xcommonrange),label='cont')
            #plt.legend()
            #plt.show()

            from lmfit import Model
            def func_multi(x, scale_factor):
                return x * scale_factor
            def func_add(x, add_factor):
                return x+add_factor*x[0]

            if scaling_mode == 'multiply':
                fmodel = Model(func_multi)
                if debug:
                    fig,ax = plt.subplots(1,3)
                    ax[0].plot(fa(xcommonrange), label='fa')
                    ax[0].plot(fb(xcommonrange), ls='-', label='fb')
                    ax[0].legend()

                result = fmodel.fit(fa(xcommonrange), x=fb(xcommonrange), scale_factor=1)
                scale_btoa = result.best_values['scale_factor']

                #second iteration
                fit = func_multi(fb(xcommonrange),scale_btoa)
                chi2 = np.power(fa(xcommonrange)-fit,2)/np.std(fa(xcommonrange))**2
                mask_chi2 = chi2>5
                if np.sum(mask_chi2)>0:
                    result = fmodel.fit(fa(xcommonrange)[~mask_chi2], x=fb(xcommonrange)[~mask_chi2], scale_factor=1)
                    scale_btoa = result.best_values['scale_factor']

                    fit = func_add(fb(xcommonrange), scale_btoa)
                    chi2_new = np.power(fa(xcommonrange) - fit, 2) / np.std(fa(xcommonrange)) ** 2
                    print('chi2',np.sum(chi2),np.sum(chi2_new))

                if debug:
                    ax[2].plot(chi2, label='chi2')
                    if np.sum(mask_chi2) > 0:
                        ax[2].plot(chi2_new, ls='--',label='chi2_n')

                    ax[1].plot(fa(xcommonrange),label='fa')
                    ax[1].plot(fb(xcommonrange),ls='--',label='fb')
                    ax[1].plot(func_add(fb(xcommonrange),scale_btoa),label='fit')
                    ax[1].legend()
                    plt.show()
                    print()


            elif scaling_mode == 'add':
                fmodel = Model(func_add)
                if debug:
                    fig,ax = plt.subplots(1,3)
                    ax[0].plot(fa(xcommonrange), label='fa')
                    ax[0].plot(fb(xcommonrange), ls='-', label='fb')
                    ax[0].legend()

                result = fmodel.fit(fa(xcommonrange), x=fb(xcommonrange), add_factor=1)
                scale_btoa = result.best_values['add_factor']

                #second iteration
                fit = func_add(fb(xcommonrange),scale_btoa)
                chi2 = np.power(fa(xcommonrange)-fit,2)/np.std(fa(xcommonrange))**2
                mask_chi2 = chi2>5
                if np.sum(mask_chi2)>0:
                    result = fmodel.fit(fa(xcommonrange)[~mask_chi2], x=fb(xcommonrange)[~mask_chi2], add_factor=1)
                    scale_btoa = result.best_values['add_factor']
                    fit = func_add(fb(xcommonrange), scale_btoa)
                    chi2_new = np.power(fa(xcommonrange) - fit, 2) / np.std(fa(xcommonrange)) ** 2
                    print('chi2',np.sum(chi2),np.sum(chi2_new))

                if debug:
                    ax[2].plot(chi2, label='chi2')
                    if np.sum(mask_chi2) > 0:
                        ax[2].plot(chi2_new, ls='--',label='chi2_n')

                    ax[1].plot(fa(xcommonrange),label='fa')
                    ax[1].plot(fb(xcommonrange),ls='--',label='fb')
                    ax[1].plot(func_add(fb(xcommonrange),scale_btoa),label='fit')
                    ax[1].legend()
                    plt.show()
                    print()


            return scale_btoa

        scaling_mode = self.scale_mode_win_option.currentText()
        if scaling_mode == 'multiply':
            def calc_y(x, scale_factor):
                return x * scale_factor
            def calc_err(x, scale_factor):
                return x * scale_factor
            w1 = 1
        elif scaling_mode == 'add':
            def calc_y(x, add_factor):
                return x + add_factor*x[0]
            def calc_err(x, scale_factor):
                return x
            w1 = 0
        if hasattr(s1,'x') and hasattr(s2,'x'):
            w2 = calc_coeff(sa=s1,sb=s2)
            s2 = spectrum(x=self.w2.data.x, y=calc_y(self.w2.data.y,w2), err=calc_err(self.w2.data.err,w2))
            if hasattr(s3,'x'):
                w3 = calc_coeff(sa=s2,sb=s3)
                s3 = spectrum(x=self.w3.data.x, y=calc_y(self.w3.data.y,w3), err=calc_err(self.w3.data.err,w3))
                if hasattr(s4, 'x'):
                    w4 = calc_coeff(sa=s3, sb=s4)
                    s4 = spectrum(x=self.w4.data.x,y=calc_y(self.w4.data.y,w4), err=calc_err(self.w4.data.err,w4))
                    if hasattr(s5, 'x'):
                        w5 = calc_coeff(sa=s4, sb=s5)
                        s5 = spectrum(x=self.w5.data.x, y=calc_y(self.w5.data.y,w5), err=calc_err(self.w5.data.err,w5))
                        if hasattr(s6, 'x'):
                            w6 = calc_coeff(sa=s5, sb=s6)
                            s6 = spectrum(x=self.w6.data.x, y=calc_y(self.w6.data.y,w6), err=calc_err(self.w6.data.err,w6))
                            if hasattr(s7, 'x'):
                                w7 = calc_coeff(sa=s6, sb=s7)
                                s7 = spectrum(x=self.w7.data.x, y=calc_y(self.w7.data.y,w7), err=calc_err(self.w7.data.err,w7))
                                if hasattr(s8, 'x'):
                                    w8 = calc_coeff(sa=s7, sb=s8)
                                    s8 = spectrum(x=self.w8.data.x, y=calc_y(self.w8.data.y,w8), err=calc_err(self.w8.data.err,w8))
                                    if hasattr(s9, 'x'):
                                        w9 = calc_coeff(sa=s8, sb=s9)
                                        s9 = spectrum(x=self.w9.data.x, y=calc_y(self.w9.data.y,w9), err=calc_err(self.w9.data.err,w9))
                                        if hasattr(s10, 'x'):
                                            w10 = calc_coeff(sa=s9, sb=s10)
                                            s10 = spectrum(x=self.w10.data.x,y=calc_y(self.w10.data.y,w10), err=calc_err(self.w10.data.err,w10))
                                            if hasattr(s11, 'x'):
                                                w11 = calc_coeff(sa=s10, sb=s11)
                                                s11 = spectrum(x=self.w11.data.x, y=calc_y(self.w11.data.y,w11), err=calc_err(self.w11.data.err,w11))
                                                if hasattr(s12, 'x'):
                                                    w12 = calc_coeff(sa=s11, sb=s12)
                                                    s12 = spectrum(x=self.w12.data.x,y=calc_y(self.w12.data.y,w12), err=calc_err(self.w12.data.err,w12))
                                                    if debug:
                                                        plt.subplots()
                                                        plt.plot(s1.x,s1.y)
                                                        plt.plot(s2.x,s2.y)
                                                        plt.plot(s3.x,s3.y)
                                                        plt.plot(s4.x,s4.y)
                                                        plt.plot(s5.x,s5.y)
                                                        plt.plot(s6.x,s6.y)
                                                        plt.plot(s7.x,s7.y)
                                                        plt.plot(s8.x,s8.y)
                                                        plt.plot(s9.x,s9.y)
                                                        plt.plot(s10.x,s10.y)
                                                        plt.plot(s11.x,s11.y)
                                                        plt.plot(s12.x,s12.y)
                                                        plt.show()

        if 1:
            self.w1.x=w1
            self.w2.x = w2
            self.w3.x = w3
            self.w4.x = w4
            self.w5.x = w5
            self.w6.x = w6
            self.w7.x = w7
            self.w8.x = w8
            self.w9.x = w9
            self.w10.x = w10
            self.w11.x = w11
            self.w12.x = w12
            if 1:
                self.w1.update_suggested_value(w1)
                self.w2.update_suggested_value(w2)
                self.w3.update_suggested_value(w3)
                self.w4.update_suggested_value(w4)
                self.w5.update_suggested_value(w5)
                self.w6.update_suggested_value(w6)
                self.w7.update_suggested_value(w7)
                self.w8.update_suggested_value(w8)
                self.w9.update_suggested_value(w9)
                self.w10.update_suggested_value(w10)
                self.w11.update_suggested_value(w11)
                self.w12.update_suggested_value(w12)
            self.update_plot()
            self.update_val_labels()
    def RebinIt(self,click =False, smooth = True, double_binning =True):
        def rebin_arr(a, factor):
            n = a.shape[0] // factor
            return a[:n * factor].reshape(a.shape[0] // factor, factor).sum(1) / factor

        def rebin_weight_mean(y, err, factor):
            w = np.array(np.power(err, -2))
            a = np.array(y)
            n = a.shape[0] // factor
            a *= w
            a = a[:n * factor].reshape(n, factor)
            w = w[:n * factor].reshape(n, factor)
            a = a.sum(1)
            w = w.sum(1)
            return a / w, np.power(w, -0.5)

        n = int(self.rebin_n_pix.text())
        n_smooth = int(self.smooth_n_pix.text())

        #mask nan and zero (bad) flux data
        mask_nan = ~np.isnan(self.combined_spec.y)*(self.combined_spec.y!=0)
        x = np.array(self.combined_spec.x[mask_nan])
        y = np.array(self.combined_spec.y[mask_nan])
        err = np.array(self.combined_spec.err[mask_nan])
        if smooth and  n_smooth>0:
            from scipy.signal import savgol_filter
            y = savgol_filter(y, n_smooth, 3)

        if not double_binning:
            x_new = rebin_arr(x, n)
            y_new, err_new = rebin_weight_mean(y, err, n)
        else:
            mask_ch1_ch2 = x < 11.6 #11.6 border between ch2 and ch3
            x_ch12 =  rebin_arr(x[mask_ch1_ch2], 2*n)
            y_ch12, err_ch12 = rebin_weight_mean(y[mask_ch1_ch2], err[mask_ch1_ch2], 2*n)

            x_ch34 =  rebin_arr(x[~mask_ch1_ch2], n)
            y_ch34, err_ch34 = rebin_weight_mean(y[~mask_ch1_ch2], err[~mask_ch1_ch2], n)

            x_new = np.append(x_ch12,x_ch34)
            y_new = np.append(y_ch12,y_ch34)
            err_new = np.append(err_ch12,err_ch34)
        self.rebinned_spec = spectrum(x_new, y_new,err_new,'rebinned')
        # recalc errorbars
        self.RecalcStd(mode='Rebinned')
        self.win.plot_spec(fname='Rebinned', add=False, show_err_bar=True)
        self.win.plot_spec(fname='Rebinned', fcolor='red', data=self.rebinned_spec, coef=1, show_err_bar=True)

        #self.win.plot_spec(fname='Combined', add=False, show_err_bar=True)
        #self.win.plot_spec(fname='Combined', fcolor='green', data=self.combined_spec, coef=1, show_err_bar=True)


    def RecalcStd(self,click=False,mode = 'Combined'):
        debug=False
        if mode == 'Combined':
            y_orig = np.array(self.combined_spec.y)
            spec_tmp = self.combined_spec.copy()
        if mode == 'Rebinned':
            y_orig = np.array(self.rebinned_spec.y)
            spec_tmp = self.rebinned_spec.copy()

        npix =spec_tmp.x.shape[0]


        from scipy.signal import savgol_filter
        spec_tmp.y = savgol_filter(spec_tmp.y, 50, 3)
        if debug:
            plt.subplots()
            plt.plot(self.combined_spec.x,self.combined_spec.y,label='combined')
            plt.plot(spec_tmp.x,spec_tmp.y,ls='--',label='convolved')
            plt.show()

        spec_tmp.y =  y_orig-spec_tmp.y

        #select outliers:
        #outlier_limit = 3*np.std(spec_tmp.y)
        #mask_outliers = np.abs(spec_tmp.y)>outlier_limit

        #calc pixels std
        spec_std = np.zeros(npix)
        win = 50
        s = np.arange(npix)
        for i in range(npix):
            if i<npix/2:
                mask = (s<=i+win/2)*(s>=i-win/2)*(y_orig != 0) #*(~mask_outliers)
            else:
                mask = (s <= i ) * (s >= i - win) * (y_orig != 0)
            spec_std[i] =np.nanstd(spec_tmp.y[mask])
        if mode == 'Combined':
            self.combined_spec.err = spec_std
        elif mode == 'Rebinned':
            self.rebinned_spec.err = spec_std

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


        if mode == 'Combined':
            self.win.plot_spec(fname='Combined', add=False, show_err_bar=True)
            self.win.plot_spec(fname='Combined', fcolor='green', data=self.combined_spec, coef=1, show_err_bar=True)

    def ApplyLeakCorr(self, click=False, mode='Combined'):
        debug = False
        from scripts.leak_correction import correct_miri_mrs_spectral_leak
        leakfilename = '/home/slava/science/codes/python/jwst/data_local/leak/MRS_spectral_leak_fractional.fits'
        if mode == 'Combined':
            x = np.array(self.combined_spec.x)
            y = np.array(self.combined_spec.y)
            spec_tmp = self.combined_spec.copy()

            leak_coeff = float(self.leak_coeff.text())

            (y_corr, leak) = correct_miri_mrs_spectral_leak(ch3spec=(x, y),
                                                           ch1spec=(x, y*leak_coeff),
                                                           leakreffile=leakfilename)
            self.combined_spec.y  = y_corr
        #if mode == 'Combined':
        #    self.combined_spec.err = spec_std


        if mode == 'Combined':
            self.win.plot_spec(fname='Combined', add=False, show_err_bar=True)
            self.win.plot_spec(fname='Combined', fcolor='green', data=self.combined_spec, coef=1, show_err_bar=True)


    def readfolder(self,path=None,obj_name='',dith=True, keyname='_sci.spec1d'):

        if path==None:
           path = self.spec_folder

        lst = []
        for (dirpath, dirname, filenames) in os.walk(path):
            for k, f in enumerate(filenames):
                if dith == False:
                    if f.endswith(keyname) and obj_name in f and 'dith' not in f:
                #if f.endswith('_s3d.dat') and obj_name in f:
                        lst.append(f)
                elif dith == True:
                    if f.endswith(keyname) and obj_name in f:
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

    if 0:
        app = QApplication(sys.argv)
        v = Viewer(spec_folder=input_dir)
        #v.show()
        v.showMaximized()
        sys.exit(app.exec())

    from PyQt6.QtGui import QGuiApplication

    app = QApplication(sys.argv)

    screen = QGuiApplication.primaryScreen()
    geometry = screen.availableGeometry()

    v = Viewer(spec_folder=input_dir)
    v.resize(geometry.width(), geometry.height())
    v.show()

    sys.exit(app.exec())

# button.setFixedSize(600, 30)
#button.resize(600, 30)