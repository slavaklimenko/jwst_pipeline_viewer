import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit,QSizePolicy, QSlider, QSpacerItem, QVBoxLayout, QWidget,QComboBox, QPushButton
from PyQt5 import QtGui
import pyqtgraph as pg
import numpy as np
import sys, os
from scipy.interpolate import interp1d, UnivariateSpline
from functools import partial

from astropy.io import fits

folder = './'
for (dirpath, dirname, filenames) in os.walk(folder):
    print(dirpath, dirname, filenames)
    for k, f in enumerate(filenames):
        if f.endswith('fits'):
            hdu = fits.open(folder+f)
            data = hdu[1].data
            col1 = data['WAVELENGTH']
            col2 = data['FLUX']
            col3 = data['ERROR']
            spec = np.zeros((int(np.size(col1)),3))
            spec[:,0] = col1
            spec[:,1] = col2
            spec[:,2] = col3
            specname = f.split('fits')[0]+'txt'
            np.savetxt('./ascii/'+specname,spec)