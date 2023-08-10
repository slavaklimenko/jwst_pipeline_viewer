# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:36:40 2016
@author: Slava
"""
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__))[:-16])
#sys.path.append('C:/science/python')
#sys.path.append('/media/serj/3078FE3678FDFB04/science/python')
import JWSTviewer
import JWST_cube_viewer

from PyQt5.QtWidgets import (QApplication)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = JWSTviewer.JWSTviewer()
    ex2 = JWST_cube_viewer.JWST_spec_viewer()
    sys.exit(app.exec_())