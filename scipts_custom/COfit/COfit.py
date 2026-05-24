#!/usr/bin/env python

#from adjustText import adjust_text
from bisect import bisect_left
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
#from mendeleev import element
import numpy as np
from pathlib import Path
from scipy import interpolate
import sys
sys.path.append('/home/slava/science/codes/python/spectro/')
sys.path.append('/home/slava/anaconda3')
sys.path.append('/home/slava/science/codes/python/spectro/sviewer/')
#from spectro.sviewer.utils import roman
#from spectro.atomic import atomicData
from spectro.profiles import voigt, tau,convolveflux
from spectro.atomic import line
from astropy import constants as const
import os
import pickle
from scipy.interpolate import interp1d
from matplotlib import rcParams
from astropy.io import ascii, fits
rcParams['font.family'] = 'serif'
import emcee

def gauss(x, s):
    return 1/np.sqrt(2*np.pi)/s * np.exp(-.5*(x/s)**2)

class element():
    def __init__(self, sp_name = '',logN=15,Texc=10,redshift=0,b_doppler=50):
        self.name = sp_name
        if self.name is not None:
            self.set_levels()
            self.logN = logN
            self.Texc = Texc
            self.z=redshift
            self.b = b_doppler
    def set_levels(self):
        if self.name == 'H2':
            H2_energy = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/energy_X_H2.dat',
                                      dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                                      unpack=True, skip_header=3, comments='#')
            H2energy = np.zeros([max(H2_energy[0]) + 1, max(H2_energy[1]) + 1])
            for k, e in enumerate(H2_energy[0]):
                H2energy[e, H2_energy[1][k]] = H2_energy[2][k]
            stat_H2 = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]
            self.energy = H2energy[0,:np.size(stat_H2)]
            self.stat = stat_H2
        if self.name == '12CO':
            COenergy = np.array(
                [0.0,5.5321, 16.5962,33.1917,55.3180,82.9744,116.1597,154.8727,199.1120,248.8756,304.1619,364.9688,431.2938,503.1343,580.4879,663.3513,751.7215,
                 845.5952,944.9686])  # in cm-1
            self.energy = COenergy
            stat_CO = np.array([(2 * i + 1) for i in range(np.size(self.energy))])
            self.stat = stat_CO

        if self.name == '13CO':
            COenergy = np.array(
                [0.0, 5.288, 15.8662, 31.7319, 52.8852, 79.3253, 111.0515, 148.0622,190.3565])  # in K
            self.energy = COenergy
            stat_CO = np.array([(2 * i + 1) for i in range(np.size(self.energy))])
            self.stat = stat_CO

        if self.name == 'C18O':
            COenergy = np.array(
                [0.0, 5.288, 15.8662, 31.7319, 52.8852, 79.3253, 111.0515, 148.0622,190.3565])  # in K
            self.energy = COenergy
            stat_CO = np.array([(2 * i + 1) for i in range(np.size(self.energy))])
            self.stat = stat_CO
    def calc_cols(self):
        logNj = self.logN + np.log10(self.stat * np.exp(-self.energy / self.Texc))
        self.cols = logNj

class parameters():
    def __init__(self, name=None, val=None, errp=None, errm=None, var=False, vrange=[], disp=None, prior=0):
        if name is not None:
            self.name = name
        if val is not None:
            self.val = val
        if errp is not None:
            self.errp = errp
        if errm is not None:
            self.errm = errm
        if disp is not None:
            self.disp = disp
        if prior is not None:
            self.prior = prior
        self.var = var
        self.vrange = vrange

    def set_prior(self, cen, p, m):
        self.prior = 1
        self.cen = cen
        self.p = p
        self.m = m



#define species
species = {}
H2  = element(sp_name='H2',logN=21)
H2.calc_cols()
species['H2'] = H2
CO_12  = element(sp_name='12CO',logN=17)
CO_12.calc_cols()
species['12CO'] = CO_12
CO_13  = element(sp_name='13CO',logN=16)
CO_13.calc_cols()
species['13CO'] = CO_13
CO_18  = element(sp_name='C18O',logN=15)
CO_18.calc_cols()
species['C18O'] = CO_18


ISFresolution = 3000
micAA = 1e4 #micron to AA
mec_8_pi2_e2 = const.m_e.cgs.value * const.c.cgs.value/8/np.pi**2/const.e.gauss.value ** 2


#define lines
lines = []
if 1:
    #H2 rotational lines - Togi Smith 2016, Rueff A&A 630, A58 (2019)
    if 0:
        H2_energy = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/../energy_X_H2.dat',
                                  dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                                  unpack=True, skip_header=3, comments='#')
        H2energy = np.zeros([max(H2_energy[0]) + 1, max(H2_energy[1]) + 1])
        for k, e in enumerate(H2_energy[0]):
            H2energy[e, H2_energy[1][k]] = H2_energy[2][k]
        stat_H2 = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]

        #H2S(0) J=2-0
        name, nu, ju, nl, jl = 'H2S(0)', 0, 2, 0, 0
        lam = 28.219*micAA
        gk,gi = stat_H2[ju],stat_H2[jl]
        Aki = 2.95e-11
        fik = mec_8_pi2_e2*(lam*1e-8)**2*gk/gi*Aki
        #tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl],Aki=Aki,z=redshift))

        #H2S(1) J=3-1
        name, nu, ju, nl, jl = 'H2S(1)', 0, 3, 0, 1
        lam = 17.035*micAA
        gk,gi = stat_H2[ju],stat_H2[jl]
        Aki = 47.6e-11
        fik = mec_8_pi2_e2*(lam*1e-8)**2*gk/gi*Aki
        lines.append(line(name='H2S(1)', l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl],Aki=Aki,z=redshift))


        #H2S(2) J=4-2
        name, nu, ju, nl, jl = 'H2S(2)', 0, 4, 0, 2
        lam = 12.279*micAA
        gk,gi = stat_H2[ju],stat_H2[jl]
        Aki = 275e-11
        fik = mec_8_pi2_e2*(lam*1e-8)**2*gk/gi*Aki
        lines.append(line(name='H2S(2)', l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))

        #H2S(3) J=5-3
        name, nu, ju, nl, jl = 'H2S(3)', 0, 5, 0, 3
        lam = 9.665*micAA
        gk,gi = stat_H2[ju],stat_H2[jl]
        Aki = 980e-11
        fik = mec_8_pi2_e2*(lam*1e-8)**2*gk/gi*Aki
        lines.append(line(name='H2S(3)', l=lam, f=fik, g=1e9,nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl],Aki=Aki,z=redshift))

        #H2S(4) J=6-4
        name, nu, ju, nl, jl = 'H2S(4)', 0, 6, 0, 4
        lam = 8.025*micAA
        gk,gi = stat_H2[ju],stat_H2[jl]
        Aki = 2640e-11
        fik = mec_8_pi2_e2*(lam*1e-8)**2*gk/gi*Aki
        lines.append(line(name='H2S(4)', l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl],Aki=Aki,z=redshift))

        #H2S(5) J=7-5
        name, nu, ju, nl, jl = 'H2S(5)', 0, 7, 0, 5
        lam = 6.910*micAA
        gk,gi = stat_H2[ju],stat_H2[jl]
        Aki = 5880e-11
        fik = mec_8_pi2_e2*(lam*1e-8)**2*gk/gi*Aki
        lines.append(line(name='H2S(5)', l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl],Aki=Aki,z=redshift))

        #H2S(6) J=8-6
        name, nu, ju, nl, jl = 'H2S(6)', 0, 8, 0, 6
        lam = 6.109*micAA
        gk,gi = stat_H2[ju],stat_H2[jl]
        Aki = 11400e-11
        fik = mec_8_pi2_e2*(lam*1e-8)**2*gk/gi*Aki
        lines.append(line(name='H2S(6)', l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl],Aki=Aki,z=redshift))

        #H2S(7) J=9-7
        name, nu, ju, nl, jl = 'H2S(7)', 0, 9, 0, 7
        lam = 5.511*micAA
        gk,gi = stat_H2[ju],stat_H2[jl]
        Aki = 20000e-11
        fik = mec_8_pi2_e2*(lam*1e-8)**2*gk/gi*Aki
        lines.append(line(name='H2S(7)', l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl],Aki=Aki,z=redshift))

        # H2S(8) J=10-8
        name, nu, ju, nl, jl = 'H2S(8)', 0, 10, 0, 8
        lam = 5.053 * micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki = 3.236e-7
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        lines.append(
            line(name='H2S(8)', l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))
    #H2 rovibrational lines - Bialy 2022
    if 0:
        H2_energy = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/../energy_X_H2.dat',
                                  dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                                  unpack=True, skip_header=3, comments='#')
        H2energy = np.zeros([max(H2_energy[0]) + 1, max(H2_energy[1]) + 1])
        for k, e in enumerate(H2_energy[0]):
            H2energy[e, H2_energy[1][k]] = H2_energy[2][k]
        stat_H2 = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]

        name,nu,ju,nl,jl = 'H2(1-0)O(2)', 1,0,0,2
        lam = 2.626883054 * micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki = 8.532E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))

        name,nu,ju,nl,jl = 'H2(1-0)Q(1)', 1,1,0,1
        lam = 2.406591889 * micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki = 8.516E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))

        name,nu,ju,nl,jl = 'H2(1-0)O(4)', 1,2,0,4
        lam = 3.003868090* micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki =8.516E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))

        name,nu,ju,nl,jl = 'H2(1-0)Q(2)', 1,2,0,2
        lam = 2.413438823* micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki =8.472E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))



        name,nu,ju,nl,jl = 'H2(1-0)S(0)', 1,2,0,0
        lam = 2.223290181* micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki =8.472E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))

        name, nu, ju, nl, jl = 'H2(1-0)O(5)', 1, 3, 0, 5
        lam =  3.234987632* micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki = 8.380E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))

        name, nu, ju, nl, jl = 'H2(1-0)Q(3)', 1, 3, 0, 3
        lam = 2.423729703 * micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki = 8.380E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))

        name, nu, ju, nl, jl = 'H2(1-0)S(1)', 1, 3, 0, 1
        lam = 2.121833725* micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki = 8.380E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNH2[nl,jl], Aki=Aki,z=redshift))
    if 0:
        name, nu, ju, nl, jl = 'H2(1-0)O(6)', 1, 4, 0, 6
        lam = 3.500809170 * micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki = 8.226E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=0, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2, Aki=Aki))

        name, nu, ju, nl, jl = 'H2(1-0)Q(4)', 1, 4, 0, 4
        lam = 2.437489361* micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki = 8.226E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=0, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2, Aki=Aki))

        name, nu, ju, nl, jl = 'H2(1-0)Q(4)', 1, 4, 0, 2
        lam = 2.033757812 * micAA
        gk, gi = stat_H2[ju], stat_H2[jl]
        Aki = 8.226E-07
        fik = mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk / gi * Aki
        # tmpline = line(name='H2S(0)', l=lam, f=fik, g=1e9, nu_u=1, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=0, j_u=2, nu_l=0, j_l=0, b=bline, logN=logNH2, Aki=Aki))
    #CO rovibrational lines - Shirahata 2013PASJ...65....5S
    #Onishi The Astrophysical Journal, 921:141 (20pp), 2021
    if 1:
        COenergy = np.array(
                    [0.0,5.5321, 16.5962,33.1917,55.3180,82.9744,116.1597,154.8727,199.1120,248.8756,304.1619,364.9688,431.2938,503.1343,580.4879,663.3513,751.7215,
                     845.5952,944.9686])  # in cm-1
        bline = 100 #km/s
        redshift=0

        stat_CO = np.array([(2 * i + 1) for i in range(np.size(COenergy))])

        logNCO = np.zeros(20)
        name, nl, jl, nu, ju = '12CO(0-1)R(9)', 0, 9, 1, 10
        lam = 4.5876 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 6.2459 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)R(8)', 0, 8, 1, 9
        lam = 4.5950 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 6.2690 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)R(7)', 0, 7, 1, 8
        lam = 4.6024 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 6.3056 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))


        name, nl, jl, nu, ju = '12CO(0-1)R(6)', 0, 6, 1, 7
        lam = 4.6100 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 6.3526 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)R(5)', 0, 5, 1, 6
        lam = 4.6177 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 6.4224 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)R(4)', 0, 4, 1, 5
        lam = 4.6254 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 6.5272 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)R(3)', 0, 3, 1, 4
        lam = 4.6333 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 6.7034 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name,nl,jl,nu,ju = '12CO(0-1)R(2)', 0,2,1,3
        lam = 4.6412 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 7.0260*1e-6
        Aki = gi * fik/(mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name,nl,jl,nu,ju = '12CO(0-1)R(1)', 0,1,1,2
        lam = 4.6493 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik =  7.7884*1e-6
        Aki = gi * fik/(mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)R(0)', 0, 0, 1, 1
        lam = 4.6575 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 11.6587 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(1)', 0, 1, 1, 0
        lam = 4.6742 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 3.8715 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(2)', 0, 2, 1, 1
        lam = 4.6826 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 4.6371 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(3)', 0, 3, 1, 2
        lam = 4.6912 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 4.9585 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(4)', 0, 4, 1,3
        lam = 4.7002 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.1308 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))


        name, nl, jl, nu, ju = '12CO(0-1)P(5)', 0, 5, 1,4
        lam = 4.7088 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.2382 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(6)', 0, 6, 1, 5
        lam = 4.7177 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.3079 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(7)', 0, 7, 1, 6
        lam = 4.7267 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.3558 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(8)', 0, 8, 1, 7
        lam = 4.7359 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.3908 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(9)', 0, 9, 1, 8
        lam = 4.7451 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4154 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(10)', 0, 10, 1, 9
        lam = 4.7545 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4303 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(11)', 0, 11, 1, 10
        lam = 4.7640 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4428 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(12)', 0, 12, 1, 11
        lam = 4.7736 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4530 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(13)', 0, 13, 1, 12
        lam = 4.7833 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4596 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(14)', 0, 14, 1, 13
        lam = 4.7931 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4610 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(15)', 0, 15, 1, 14
        lam = 4.8031 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4646 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(16)', 0, 16, 1, 15
        lam = 4.8131 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4616 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(17)', 0, 17, 1, 16
        lam = 4.8233 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4622 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '12CO(0-1)P(18)', 0, 18, 1, 17
        lam = 4.8336 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.4604 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        #name, nl, jl, nu, ju = 'CO(0-1)P(19)', 0, 19, 1, 18
        #lam = 4.8440 * micAA
        #gk, gi = stat_CO[ju], stat_CO[jl]
        #fik = 5.4567 * 1e-6
        #Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        #lines.append(
        #    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
        #         z=redshift))

        #name, nl, jl, nu, ju = 'CO(0-1)P(20)', 0, 20, 1, 18
        #lam = 4.8544 * micAA
        #gk, gi = stat_CO[ju], stat_CO[jl]
        #fik = 5.3432 * 1e-6
        #Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        #lines.append(
        #    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
        #         z=redshift))

        #name, nl, jl, nu, ju = 'CO(0-1)P(10)', 0, 10, 1, 9
        #lam = 4.7359 * micAA
        #gk, gi = stat_CO[ju], stat_CO[jl]
        #fik = 5.3190 * 1e-6
        #Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        #lines.append(
        #line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki))
    #13CO  rovibrational lines - Ohyama 2023 arXiv:2305.09959v2
    if 1:
        COenergy = np.array(
            [0.0, 5.2888, 15.8662, 31.7319, 52.8852, 79.3253, 111.0515, 148.0622, 190.3565])  # in K
        stat_CO = np.array([(2 * i + 1) for i in range(np.size(COenergy))])
        logNCO = np.zeros(20)
        name, nl, jl, nu, ju = '13CO(0-1)R(0)', 0, 0, 1, 1
        lam = 4.7626 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 10.9188 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)R(1)', 0, 1, 1, 2
        lam = 4.7544 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 7.2983 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)R(2)', 0, 2, 1, 3
        lam = 4.7463 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 6.5775 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)R(3)', 0, 3, 1, 4
        lam = 4.7383 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 6.2755 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)R(4)', 0, 4, 1, 5
        lam = 4.7305 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik =  6.1138 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)R(5)', 0, 5, 1, 6
        lam = 4.7227  * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik =  6.0147  * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)R(6)', 0, 6, 1, 7
        lam = 4.7150  * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik =  5.9455  * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)R(7)', 0, 7, 1, 8
        lam = 4.7075  * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik =  5.9039   * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

    ####################

        name, nl, jl, nu, ju = '13CO(0-1)P(1)', 0, 1, 1, 0
        lam = 4.7792 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 3.6252 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)P(2)', 0, 2, 1, 1
        lam = 4.7877 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 4.3416 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)P(3)', 0, 3, 1, 2
        lam = 4.7963 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 4.6431 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)P(4)', 0, 4, 1, 3
        lam = 4.8050 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 4.8078 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)P(5)', 0, 5, 1, 4
        lam = 4.8138 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 4.9057 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)P(6)', 0, 6, 1, 5
        lam = 4.8227 * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 4.9713 * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = '13CO(0-1)P(7)', 0, 7, 1, 6
        lam = 4.8317  * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        fik = 5.0169  * 1e-6
        Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))
    #C18O  rovibrational lines - HITRAN
    if 1:
        COenergy = np.array(
            [0.0, 5.288, 15.8662, 31.7319, 52.8852, 79.3253, 111.0515, 148.0622, 190.3565])  # in K
        stat_CO = np.array([(2 * i + 1) for i in range(np.size(COenergy))])
        logNCO = np.zeros(20)
        name, nl, jl, nu, ju = 'C18O(0-1)R(0)', 0, 0, 1, 1
        nu = 2095.750994
        lam = 1e4/nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 10.63
        fik = Aki/gi*(mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        #fik = 10.9188 * 1e-6
        #Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = 'C18O(0-1)R(1)', 0, 1, 1, 2
        nu = 2099.34774
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 12.83
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = 'C18O(0-1)R(2)', 0, 2, 1, 3
        nu = 2102.91169
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 13.82
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = 'C18O(0-1)R(3)', 0, 3, 1, 4
        nu = 2106.442709
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 14.4
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = 'C18O(0-1)R(4)', 0, 4, 1, 5
        nu = 2109.940667
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 14.81
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))
        name, nl, jl, nu, ju = 'C18O(0-1)R(5)', 0, 5, 1, 6
        nu = 2113.405428
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 15.11
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = 'C18O(0-1)R(6)', 0, 7, 1, 8
        nu = 2116.83686
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 15.36
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = 'C18O(0-1)P(1)', 0, 1, 1, 0
        nu = 2088.459646
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 31.55
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = 'C18O(0-1)P(2)', 0, 2, 1, 1
        nu = 2084.76531
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 20.91
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))


        name, nl, jl, nu, ju = 'C18O(0-1)P(3)', 0, 3, 1, 2
        nu = 2081.03871
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 18.72
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))

        name, nl, jl, nu, ju = 'C18O(0-1)P(4)', 0, 4, 1, 3
        nu = 2077.27998
        lam = 1e4 / nu * micAA
        gk, gi = stat_CO[ju], stat_CO[jl]
        Aki = 17.73
        fik = Aki / gi * (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        # fik = 10.9188 * 1e-6
        # Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
        lines.append(
            line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
                 z=redshift))


class spectrum_model():
    def __init__(self, wave_min=1e4, wave_max = 30e4, num_res = 100000,lines=lines,species =None):
        self.wave_min = wave_min
        self.wave_max = wave_max
        self.num = int((wave_max-wave_min)/(wave_max+wave_min)*2*num_res)
        self.wavel = np.linspace(self.wave_min, self.wave_max, self.num)
        self.lines = lines
        self.sp = species
        self.Tau = np.zeros_like(self.wavel)

    def calc_spec(self, convolve=True,ISFresolution=3000,species=None):
        if species == None:
            species = self.sp
        for tmpline in self.lines:
            sp = None
            if '12CO' in tmpline.name:
                sp = '12CO'
            if '13CO' in tmpline.name:
                sp = '13CO'
            if 'C18O' in tmpline.name:
                sp = 'C18O'
            if 'H2' in tmpline.name:
                sp = 'H2'
            if sp != None and sp in species.keys():
                Jlevel = tmpline.j_l
                logNJ = species[sp].cols[Jlevel]
                tmpline.logN = logNJ
                tmpline.z = species[sp].z
                tmpline.b = species[sp].b
                linetau = tau(line=tmpline, resolution=50000)
                self.Tau += linetau.calctau(x=self.wavel, vel=False, debug=False, verbose=False, convolve=None, tlim=0.0001)

        #plt.subplots(       )
        #plt.plot(self.wavel/1e4,self.Tau)
        #plt.show()
        self.absorptionflux = np.exp(-self.Tau)
        self.observedflux = self.absorptionflux

        # add CONVOLUTION
        if convolve:
            self.resolution = ISFresolution
            self.observedflux = convolveflux(self.wavel, self.observedflux, res=self.resolution, kind='direct')

    def plot_spec(self,redshift=0.0,ax=None,label='$^{12}$CO',color='red'):
        plot_spec_flag = False
        if ax == None:
            fig,ax = plt.subplots()
            plot_spec_flag = True
        mask_c12 = self.wavel / 1e4 / (1 + redshift) < 4.725
        ax.plot(self.wavel/ 1e4 / (1 + redshift), self.observedflux, color=color, alpha=0.7,
                label=label)
        #ax.plot(self.wavel[mask_c12] / 1e4 / (1 + redshift), self.observedflux[mask_c12], color='red', alpha=0.7,label='$^{12}$CO')
        #ax.plot(self.wavel[~mask_c12] / 1e4 / (1 + redshift), self.observedflux[~mask_c12], color='tab:blue', alpha=0.7,label='$^{13}$CO')
        #ax.plot(self.wavel[~mask_c12] / 1e4 / (1 + redshift), self.observedflux[~mask_c12], color='tab:blue', alpha=0.7,
        #        label='$^{18}$CO')

        if plot_spec_flag:
            plt.show()
        return ax


#spec = spectrum(species=species)
#spec.calc_spec()
#spec.plot_spec()
#plt.show()

if __name__ == '__main__':

    #spec_file = np.loadtxt('/home/slava/science/research/kulkarni/JWST-DLAs/ID2441/previus_cubes/PKS1830B_CO_lines.dat')
    spec_file = np.loadtxt('/home/slava/science/research/kulkarni/JWST-DLAs/ID2441/CO_lines/spaxels/2AB_weighted_mean_spec_norm.dat')
    wave,flux,error = spec_file[:,0],spec_file[:,1],spec_file[:,2]
    #
    fig,ax = plt.subplots()
    z = 0.88582
    ax.plot(wave/(1+z),flux)
    for l in lines:
        if '12CO' in l.name:
            print(l.name)
            if l.j_u <= 10:
                ax.axvline(l.wavelength[0]/1e4,color='red',ls='--')
        if  '13CO' in l.name:
            print(l.name)
            if l.j_u <= 5:
                ax.axvline(l.wavelength[0]/1e4,color='blue',ls='--')
        if  'C18O' in l.name:
            print(l.name)
            ax.axvline(l.wavelength[0]/1e4,color='green',ls='--')

    plt.show()


    
    pars = {}
    num_12CO_lev = 6
    num_13CO_lev = 4
    num_C18O_lev = 3
    for i in range(num_12CO_lev):
        pars['12COJ'+str(i)] = parameters('12COJ'+str(i),val=18,var=1,disp=0.2, vrange=[0, np.inf])
    for i in range(num_13CO_lev):
        pars['13COJ'+str(i)] = parameters('13COJ'+str(i),val=18,var=1,disp=0.2, vrange=[0, np.inf])
    for i in range(num_C18O_lev):
        pars['C18OJ'+str(i)] = parameters('C18OJ'+str(i),val=18,var=1,disp=0.2, vrange=[0, np.inf])
    pars['z']  = parameters('z',val=0.88582,var=1,disp=1+100/3e5, vrange=[0, np.inf])
    pars['b'] = parameters('b',val=20,var=0,disp=10, vrange=[5, np.inf])
    pars['y0'] = parameters('y0',val=0,var=0,disp=0.1, vrange=[0, 1])
    pars['logNCO12'] = parameters('logNCO12', val=18, var=0, disp=2, vrange=[10, np.inf])
    pars['Texc12'] = parameters('Texc12', val=10, var=0, disp=2, vrange=[3, np.inf])
    pars['Texc13'] = parameters('Texc13', val=10, var=0, disp=2, vrange=[3, np.inf])
    pars['Texc18'] = parameters('Texc18', val=10, var=0, disp=2, vrange=[3, np.inf])
    pars['C13/C12'] = parameters('C13/C12', val=10, var=0, disp=5, vrange=[0, np.inf])
    pars['C18O/12CO'] = parameters('C18O/12CO', val=50, var=0, disp=5, vrange=[0, np.inf])


    def calc_excitation(pars):
            #12CO
            z12 = np.sum(species['12CO'].stat*np.exp(-species['12CO'].energy/pars['Texc12'].val))
            N0 = pars['logNCO12'].val - np.log10(z12)
            for i in range(num_12CO_lev):
                si,ei = species['12CO'].stat[i],species['12CO'].energy[i]
                pars['12COJ'+str(i)] = N0 + np.log10(si*np.exp(-ei/pars['Texc12'].val))
            # 13CO
            z13 = np.sum(species['13CO'].stat * np.exp(-species['13CO'].energy / pars['Texc13'].val))
            N0 = pars['logNCO12'].val - np.log10(pars['C13/C12'].val) - np.log10(z13)
            for i in range(num_13CO_lev):
                si, ei = species['13CO'].stat[i], species['13CO'].energy[i]
                pars['13COJ' + str(i)] = N0 + np.log10(si * np.exp(-ei / pars['Texc13'].val))
            # C18O
            z18 = np.sum(species['C18O'].stat * np.exp(-species['C18O'].energy / pars['Texc18'].val))
            N0 = pars['logNCO12'].val - np.log10(pars['C18O/12CO'].val) - np.log10(z18)
            for i in range(num_C18O_lev):
                si, ei = species['C18O'].stat[i], species['C18O'].energy[i]
                pars['C18OJ' + str(i)] = N0 + np.log10(si * np.exp(-ei / pars['Texc18'].val))



    def calc_model(pars,w_min=1e4,w_max=30e4):
        if pars['logNCO12'].var:
            calc_excitation(pars)
        species = {}


        if show_sp[1]:
            CO = element(sp_name='CO', redshift=params['z'], b_doppler=params['b'],
                         logNJ=[params['COJ0'],params['COJ1'],params['COJ2'],params['COJ3'],params['COJ4'],params['COJ5']])
            CO.calc_cols(mode=mode)
            species['CO'] = CO
        if show_sp[2]:
            CO_13 = element(sp_name='13CO', logN=params['logN13CO'], fcold=params['fCOcold'],
                            Texc_cold=params['TexcCO_cold'],
                            Texc_hot=params['TexcCO_hot'], redshift=params['z'], b_doppler=params['b'],
                            logNJ=[params['13COJ0'],params['13COJ1'],params['13COJ2']])
            CO_13.calc_cols(mode=mode)
            species['13CO'] = CO_13
        if show_sp[3]:
            CO_18 = element(sp_name='C18O', logN=params['logNC18O'], fcold=params['fCOcold'],
                            Texc_cold=params['TexcCO_cold'],
                            Texc_hot=params['TexcCO_hot'], redshift=params['z'], b_doppler=params['b'],
                            logNJ=[params['C18OJ0'],params['C18OJ1'],params['C18OJ2']])
            CO_18.calc_cols(mode=mode)
            species['C18O'] = CO_18


        spec = spectrum_model(species=species,wave_min=w_min,wave_max=w_max)
        spec.calc_spec()

        if 0:
            plt.subplot()
            plt.plot(species['H2'].energy, species['H2'].cols - np.log10(species['H2'].stat), 'o')
            plt.plot(species['CO'].energy, species['CO'].cols - np.log10(species['CO'].stat), 'o')
            plt.plot(species['13CO'].energy, species['13CO'].cols - np.log10(species['13CO'].stat), 'o')
            plt.plot(species['C18O'].energy, species['C18O'].cols - np.log10(species['C18O'].stat), 'o',color='black')
        return spec

    spec = calc_model(params)
    params1=params.copy()
    params['logNCOtot'],params['logN13CO'],params['logNC18O'] = params1['logNCOtot'],10,10
    specC12 = calc_model(params)
    params['logNCOtot'], params['logN13CO'], params['logNC18O'] = 10,params1['logN13CO'], 10
    specC13 = calc_model(params)
    params['logNCOtot'], params['logN13CO'], params['logNC18O'] = 10, 10,params1['logNC18O']
    specC18 = calc_model(params)
    params = params1

    fig,ax = plt.subplots()
    spec.plot_spec(ax=ax,redshift=0.88582)
    specC12.plot_spec(ax=ax, redshift=0.88582,color='red')
    specC13.plot_spec(ax=ax, redshift=0.88582,color='blue')
    specC18.plot_spec(ax=ax, redshift=0.88582,color='purple')
    ax.plot(wave/(1+ 0.88582),flux,color='black')
    plt.show()

    if 1:
        def log_likelihood(theta, params=params,spec_file=spec_file, redshift=0.88582,mode ='Texc'):
            if mode == 'Texc':
                NCO, TexcCO_cold, logN13CO, Texc13CO_cold, z, b = theta
                params['logNCOtot'] = NCO
                params['TexcCO_cold'] = TexcCO_cold
                params['logN13CO'] = logN13CO
                params['Texc13CO_cold'] = TexcCO_cold #Texc13CO_cold
                params['z'] = z
                params['b'] = b

                wave, flux, error = spec_file[:, 0].copy(), spec_file[:, 1].copy(), spec_file[:, 2].copy()+0.02
                wave /= (1 + redshift)
                spec = calc_model(params,w_min=4.5*(1 + redshift)*1e4,w_max=4.9*(1 + redshift)*1e4)
            elif mode == 'all_J':
                COJ0, COJ1, COJ2, COJ3, COJ4, COJ5,CO13J0, CO13J1, CO13J2,C18OJ0, C18OJ1, C18OJ2, z, b = theta
                params['COJ0'] = COJ0
                params['COJ1'] = COJ1
                params['COJ2'] = COJ2
                params['COJ3'] = COJ3
                params['COJ4'] = COJ4
                params['COJ5'] = COJ5
                params['13COJ0'] = CO13J0
                params['13COJ1'] = CO13J1
                params['13COJ2'] = CO13J2
                params['C18OJ0'] = C18OJ0
                params['C18OJ1'] = C18OJ1
                params['C18OJ2'] = C18OJ2
                params['z'] = z
                params['b'] = b
                wave, flux, error = spec_file[:, 0].copy(), spec_file[:, 1].copy(), spec_file[:, 2].copy() + 0.02
                wave /= (1 + redshift)
                spec = calc_model(params, w_min=4.5 * (1 + redshift) * 1e4, w_max=4.9 * (1 + redshift) * 1e4,mode =mode)
            # mask = spec.observedflux<0.098
            w = spec.wavel / 1e4 / (1 + redshift)
            f_interp = interp1d(w, spec.observedflux,fill_value='extrapolate')
            model = f_interp(wave)
            mask = (wave > 4.6) * (wave < 4.8)

            chiq = -0.5 * np.nansum(np.power(flux[mask] - model[mask], 2) / np.power(error[mask], 2))
            if 0:
                print(params)
                plt.subplots()
                plt.plot(wave,flux)
                plt.plot(wave, model)
                plt.plot(wave[mask], model[mask])
                plt.plot(w,spec.observedflux)
                plt.show()

            return chiq

        def log_prior(theta,mode = 'Texc'):
            '''
            set prior on params:
            params['logNH2tot'] = 22
            params['fH2cold'] = 1-1e-05
            params['TexcH2_cold'] = 50
            params['TexcH2_hot'] = 500
            params['logNCOtot'] = 17.5
            params['fCOcold'] = 1-1e-05
            params['TexcCO_cold'] = 30
            params['TexcCO_hot'] = 100
            params['logN13CO'] = 17
            params['Texc13CO_cold'] = 30
            params['Texc13CO_hot'] = 100
            params['z']  = 0.88582*(1+50/3e5)
            params['b'] = 20
            '''
            if mode == 'Texc':
                NCO, TexcCO_cold, logN13CO, Texc13CO_cold, z, b = theta
                if 16 < NCO < 20 and 3 < TexcCO_cold < 100 and 14 < logN13CO < 20 and 3 < Texc13CO_cold < 100 and 0.86 < z<0.9  and 1 < b < 100:
                    return 0.0
                return -np.inf
            elif mode == 'all_J':
                COJ0, COJ1, COJ2, COJ3, COJ4, COJ5, CO13J0, CO13J1, CO13J2, C18OJ0, C18OJ1, C18OJ2, z, b = theta
                logN = [COJ0, COJ1, COJ2, COJ3, COJ4,COJ5, CO13J0, CO13J1, CO13J2, C18OJ0, C18OJ1, C18OJ2]
                chi = 0
                for NJ in logN:
                    if 13 < NJ < 18:
                       chi+=0
                    else:
                       chi+=-np.inf 
                    if 0.86 < z<0.9  and 1 < b < 100:
                        chi+=0
                    else:
                        chi+=-np.inf
                if chi == 0:
                    return 0.0
                else:              
                    return -np.inf


        if 1:
            nwalkers = 500
            nsteps = 1000


            def log_probability(theta,x=None,y=None,mode ='all_J'):
                lp = log_prior(theta,mode=mode)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_likelihood(theta,mode=mode)


            # run sampler
            mode = 'all_J'
            ndim=14
            if 0:
                # set start position
                
                if mode == 'Texc':
                    # NCO, TexcCO_cold, logN13CO, Texc13CO_cold, z, b = theta
                    ndim = 6
                    init = [17, 30, 17,30,0.88582,15]
                    init_range = [1, 20, 1,20,0.00001,5]
                    pos2 = []
                elif mode == 'all_J':
                    #COJ0, COJ1, COJ2, COJ3, COJ4, CO13J0, CO13J1, CO13J2, C18OJ0, C18OJ1, C18OJ2, z, b = theta
                    ndim = 14
                    init = [18, 17.5,17.5,17,16,15,16,16,16,16,16,16,0.88594,10]
                    init_range = [1, 1,1,1,1,1,1,1,1,1,1,1,0.0001,5]
                    pos2 = []

                for i in range(nwalkers):
                    prob = -np.inf
                    while prob == -np.inf:
                        rndm = np.random.randn(ndim)
                        wal_pos = init + init_range * rndm
                        prob = log_probability(theta=wal_pos,mode=mode)
                    pos2.append(wal_pos)
                    if 0:
                        print('prob',prob)
                        NCO, TexcCO_cold, logN13CO, Texc13CO_cold, z, b = wal_pos
                        params['logNCOtot'] = NCO
                        params['TexcCO_cold'] = TexcCO_cold
                        params['logN13CO'] = logN13CO
                        params['Texc13CO_cold'] = Texc13CO_cold
                        params['z'] = z
                        params['b'] = b
                        spec = calc_model(params)
                        fig, ax = plt.subplots()
                        spec.plot_spec(ax=ax, redshift=0.88582)
                        ax.plot(wave / (1 + 0.88582), flux)
                        plt.show()

                pos = [init + init_range * np.random.randn(ndim) for i in range(nwalkers)]
                x,y=1,1

                if 0:
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y))

                    for i, result in enumerate(sampler.sample(pos2, iterations=nsteps)):
                        if i % int(nsteps / 10) == 0:
                            print("{0:5.1%}".format(float(i) / nsteps))
                elif 1:
                    from multiprocessing import Pool

                    pool = Pool(8)
                    # with Pool(processes=2) as pool:
                    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y),pool=pool)
                    if 1:
                        sampler.run_mcmc(pos2, nsteps, progress=True)

                        if 1:
                            samples = sampler.chain[:, :, :]
                            with open('data/chain_allJ.pkl', 'wb') as f:
                                pickle.dump(samples, f)
            if 1:
                with open('data/chain_allJ.pkl', 'rb') as f:
                    samples = pickle.load(f)

            if 1:
                #samples = sampler.chain[:, :, :]
                means = np.zeros((ndim, nsteps))
                vars = np.zeros((ndim, nsteps))
                single = np.zeros((ndim, nsteps))
                for i in range(nsteps):
                    for j in range(ndim):
                        means[j, i] = np.mean(samples[:, i, j])
                        vars[j, i] = np.std(samples[:, i, j])
                        single[j, i] = samples[5, i, j]

                # chain stats
                print('chain stats')
                fig, ax = plt.subplots(nrows=1, ncols=ndim)
                if ndim > 1:
                    i = 0
                    for col in ax:
                        print(i)
                        if i < ndim:
                            col.errorbar(np.arange(nsteps), means[i, :], yerr=vars[i, :],
                                         fmt='-', color='black',
                                         markeredgecolor='black', markeredgewidth=2, capsize=2,
                                         ecolor='royalblue', alpha=0.7)
                            col.plot(np.arange(nsteps), single[i, :], color='red')
                        i += 1
                else:
                    i = 0
                    ax[0].errorbar(np.arange(nsteps), means[i, :], yerr=vars[i, :],
                                   fmt='-', color='black',
                                   markeredgecolor='black', markeredgewidth=2, capsize=2,
                                   ecolor='royalblue', alpha=0.7)
                    ax[0].plot(np.arange(nsteps), single[i, :], color='red')

                #chain = sampler.chain[:, int(nsteps * 0.9):, :].reshape((-1, ndim))
                chain = samples[:, int(nsteps * 0.9):, :].reshape((-1, ndim))

            N12CO = np.log10(np.sum(np.power(10, chain[:, i]) for i in range(6)))
            # analyse the chain using chainconsumer
            from chainconsumer import ChainConsumer

            c = ChainConsumer()

            if mode =='Texc':
                par_names = ["NCO", "TexcCO_cold", "logN13CO", "Texc13CO_cold", "z", "b"]
                c.add_chain(chain, parameters=par_names)
                c.plotter.plot(filename="example.png", figsize="column")
                res = c.analysis.get_summary(parameters=par_names)
                print(res)

                # NCO, TexcCO_cold, logN13CO, Texc13CO_cold, z, b = theta
                params['logNCOtot'] = res['NCO'][1]
                params['TexcCO_cold'] = res['TexcCO_cold'][1]
                params['logN13CO'] = res['logN13CO'][1]
                params['Texc13CO_cold'] = res['Texc13CO_cold'][1]
                params['z'] = res['z'][1]
                params['b'] = res['b'][1]

                spec = calc_model(params)

                fig, ax = plt.subplots()
                spec.plot_spec(ax=ax, redshift=0.88582)

                ax.plot(wave / (1 + 0.88582), flux)
                plt.show()
            else:
                par_names = ["COJ0", "COJ1", "COJ2", "COJ3", "COJ4","COJ5", "CO13J0", "CO13J1", "CO13J2", "C18OJ0", "C18OJ1", "C18OJ2", "z", "b"]
                c.add_chain(chain, parameters=par_names)
                c.plotter.plot(filename="example.png", figsize="column")
                res = c.analysis.get_summary(parameters=par_names)
                print(res)

                #COJ0, COJ1, COJ2, COJ3, COJ4, CO13J0, CO13J1, CO13J2, C18OJ0, C18OJ1, C18OJ2, z, b = theta
                params['COJ0'] = res['COJ0'][1]
                params['COJ1'] = res['COJ1'][1]
                params['COJ2'] = res['COJ2'][1]
                params['COJ3'] = res['COJ3'][1]-0.4
                params['COJ4'] = res['COJ4'][1]
                params['COJ5'] = res['COJ5'][1]
                params['13COJ0'] = res['CO13J0'][1]
                params['13COJ1'] = res['CO13J1'][1]
                params['13COJ2'] = res['CO13J2'][1]
                params['C18OJ0'] = res['C18OJ0'][1]
                params['C18OJ1'] = res['C18OJ1'][1]
                params['C18OJ2'] = 0*res['C18OJ2'][1]
                params['z'] = res['z'][1]
                params['b'] = res['b'][1]



                
                if 1:
                    spec = calc_model(params,mode='all_J')
                    specC12 = calc_model(params,show_sp = [0,1,0,0],mode='all_J')
                    specC13 = calc_model(params,show_sp = [0,0,1,0],mode='all_J')
                    specC18 = calc_model(params,show_sp = [0,0,0,1],mode='all_J')

                    fig, ax = plt.subplots()
                    spec.plot_spec(ax=ax, redshift=0.88582)
                    specC12.plot_spec(ax=ax, redshift=0.88582, color='red')
                    specC13.plot_spec(ax=ax, redshift=0.88582, color='blue')
                    specC18.plot_spec(ax=ax, redshift=0.88582, color='tab:green')
                    ax.plot(wave / (1 + 0.88582), flux, color='black')
                    
                    #set labels
                    if 1:
                        for tmpline in spec.lines:
                            c=None
                            if '13CO' in tmpline.name:
                                c = 'blue'
                            elif 'C18O' in tmpline.name:
                                c = 'green'
                            elif 'CO' in tmpline.name:
                                c = 'red'
                            if c!=None:
                                w = tmpline.wavelength
                                n = tmpline.name
                                ax.text(w[0]*(1+params['z'])/(1+0.88582)/1e4,1.1,n,rotation=90,color=c)

                    #save spectra
                    if 1:
                        tmp=np.zeros((len(specC12.wavel),2))
                        tmp[:,0] = specC12.wavel/1e4
                        tmp[:, 1] = specC12.observedflux
                        np.savetxt('./CO_profiles/C12.dat',tmp)
                        tmp=np.zeros((len(specC12.wavel),2))
                        tmp[:,0] = specC13.wavel/1e4
                        tmp[:, 1] = specC13.observedflux
                        np.savetxt('./CO_profiles/C13.dat',tmp)
                        tmp=np.zeros((len(specC18.wavel),2))
                        tmp[:,0] = specC18.wavel/1e4
                        tmp[:, 1] = specC18.observedflux
                        np.savetxt('./CO_profiles/C18.dat',tmp)
                        del tmp
                    plt.show()




            if 0:
                fontsize=9
                fig, ax = plt.subplots(1,3,figsize=(9,3))
                ax[0].plot(wave / (1 + 0.88582), flux, color='black', label='MRS data', zorder=-100)

                specC12.plot_spec(ax=ax[0], redshift=0.88582, color='red')
                specC13.plot_spec(ax=ax[0], redshift=0.88582, color='blue')
                specC18.plot_spec(ax=ax[0], redshift=0.88582, color='tab:green')

                ax[0].legend()
                ax[0].set_title('CO lines at $z\\sim1$ in Archival JWST MRS Data',fontsize=fontsize)

                species = spec.sp
                #plt.plot(species['H2'].energy, species['H2'].cols - np.log10(species['H2'].stat), 'o')
                ax[1].plot(species['CO'].energy[:5], species['CO'].cols[:5] - np.log10(species['CO'].stat[:5]), 'o',color='red',label='$^{12}$CO')
                ax[1].plot(species['13CO'].energy[:4], species['13CO'].cols[:4] - np.log10(species['13CO'].stat[:4]), 'o',color='blue',label='$^{13}$CO')
                ax[1].plot(species['C18O'].energy[:4], species['C18O'].cols[:4] - np.log10(species['C18O'].stat[:4]),
                           'o', color='green', label='C$^{18}$O')

                ax[1].legend()

                N12CO = np.log10(np.sum(np.power(10,chain[:,i]) for i in range(6)))
                N13CO = np.log10(np.sum(np.power(10,chain[:,6+i]) for i in range(3)))
                NC18O = np.log10(np.sum(np.power(10,chain[:,9+i]) for i in range(3)))

                r1 = np.power(10,N12CO-N13CO)
                r2 = np.power(10, N12CO - NC18O)

                print(np.mean(r1),np.mean(r2))
                ax[2].hist(np.log10(r1),color='tab:blue',label = '$^{12}$C/$^{13}$C',density=True)
                ax[2].hist(np.log10(r2),color='green',label = '$^{16}$O/$^{18}$O',density=True)
                ax[2].legend()

                ax[0].set_xlim(4.59,4.86)
                ax[0].set_ylim(0.6,1.1)
                for secax in [ax[0]]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(4))
                    secax.xaxis.set_major_locator(MultipleLocator(0.1))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(4))
                    secax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax[0].set_xlabel('Restframe wavelength ($\mu$m)',fontsize=fontsize)
                ax[0].set_ylabel('Normalized flux',fontsize=fontsize,labelpad=2)

                ax[1].set_xlim(-10, 60)
                ax[1].set_ylim(14.5, 18)
                for secax in [ax[1]]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    secax.xaxis.set_major_locator(MultipleLocator(20))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    secax.yaxis.set_major_locator(MultipleLocator(1))
                ax[1].set_xlabel('Energy of J levels (K)', fontsize=fontsize)
                ax[1].set_ylabel('$\\log $N(J)/g(J)', fontsize=fontsize,labelpad=2)

                ax[2].set_xlim(0, 3.5)
                #ax[2].set_ylim(0,0.3)
                for secax in [ax[2]]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    #secax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    #secax.xaxis.set_major_locator(MultipleLocator(10))
                    #secax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    #secax.yaxis.set_major_locator(MultipleLocator(0.1))
                #ax[2].set_yticks([0, 0.2])
                ax[2].set_xlabel('$\\log r=^{12}$C/$^{13}$C ($^{16}$O/$^{18}$O)', fontsize=fontsize)
                ax[2].set_ylabel('PDF', fontsize=fontsize,labelpad=2)


                plt.show()

                str = 'J' + case + 'Miri_Av=' + str(Av) + '_H2' + '.pdf'
                f_name = '12CO13CO.pdf'
                fig.savefig(f_name, bbox_inches='tight')

            if 1:
                fontsize=9
                fig1, ax1 = plt.subplots(2,1,figsize=(3,3))
                fig2, ax2 = plt.subplots(1,2,figsize=(6,3))
                ax1[0].errorbar(wave / (1 + 0.88582), flux, color='black', zorder=-100,ds='steps-mid')
                ax1[1].errorbar(wave / (1 + 0.88582), flux, color='black',  zorder=-100,ds='steps-mid')

                if 1:
                    for tmpline in spec.lines:
                        c = None
                        if '13CO' in tmpline.name:
                            c = 'blue'
                        elif 'C18O' in tmpline.name:
                            c = 'green'
                        elif 'CO' in tmpline.name:
                            c = 'red'
                        if c == 'red':
                            w = tmpline.wavelength
                            J = tmpline.j_l
                            n = tmpline.name.split('CO(0-1)')[-1]
                            if J<7:
                                ax1[0].text(w[0] * (1 + params['z']) / (1 + 0.88582) / 1e4/1.0002, 1.05, n, rotation=90, color=c,fontsize=fontsize-2)
                        if c in ['blue']:
                            w = tmpline.wavelength
                            J = tmpline.j_l
                            n = tmpline.name.split('(0-1)')[-1]
                            if J<3:
                                ax1[1].text(w[0] * (1 + params['z']) / (1 + 0.88582) / 1e4/1.0002, 1.05, n, rotation=90, color=c,fontsize=fontsize-2)
                        if c in ['green']:
                            w = tmpline.wavelength
                            J = tmpline.j_l
                            n = tmpline.name.split('(0-1)')[-1]
                            if J<2:
                                ax1[1].text(w[0] * (1 + params['z']) / (1 + 0.88582) / 1e4, 1.05, n, rotation=90, color=c,fontsize=fontsize-2)

                specC12.plot_spec(ax=ax1[0], redshift=0.88582, color='red')
                specC13.plot_spec(ax=ax1[1], redshift=0.88582, color='blue',label='$^{13}$CO')
                specC18.plot_spec(ax=ax1[1], redshift=0.88582, color='tab:green',label='C$^{18}$O')

                ax1[0].legend(loc='lower left',fontsize=fontsize)
                ax1[1].legend(loc='lower left',fontsize=fontsize)
                ax1[0].set_title('CO lines at $z\\sim1$ in Archival JWST MRS Data',fontsize=fontsize)

                species = spec.sp
                #plt.plot(species['H2'].energy, species['H2'].cols - np.log10(species['H2'].stat), 'o')
                yerr = [res['COJ' + str(i)][1] - res['COJ' + str(i)][0] for i in range(5)]
                ax2[0].errorbar(species['CO'].energy[:5], species['CO'].cols[:5] - np.log10(species['CO'].stat[:5]),yerr=yerr, fmt='o', markerfacecolor='red', markeredgecolor='black',
                               ecolor='black',label='$^{12}$CO')
                yerr = [res['CO13J' + str(i)][1] - res['CO13J' + str(i)][0] for i in range(3)]
                ax2[0].errorbar(species['13CO'].energy[:3], species['13CO'].cols[:3] - np.log10(species['13CO'].stat[:3]),
                                yerr=yerr, fmt='o', markerfacecolor='blue', markeredgecolor='black',
                                ecolor='black', label='$^{13}$CO')
                yerr = [res['C18OJ' + str(i)][1] - res['C18OJ' + str(i)][0] for i in range(2)]
                ax2[0].errorbar(species['C18O'].energy[:2], species['C18O'].cols[:2] - np.log10(species['C18O'].stat[:2]),
                                yerr=yerr, fmt='o', markerfacecolor='green', markeredgecolor='black',
                                ecolor='black', label='C$^{18}$O')

                ax2[0].legend(loc='lower right',fontsize=fontsize)

                N12CO = np.log10(np.sum(np.power(10,chain[:,i]) for i in range(6)))
                N13CO = np.log10(np.sum(np.power(10,chain[:,6+i]) for i in range(3)))
                NC18O = np.log10(np.sum(np.power(10,chain[:,9+i]) for i in range(3)))

                r1 = np.power(10,N12CO-N13CO)
                r2 = np.power(10, N12CO - NC18O)

                print(np.mean(r1),np.mean(r2))
                ax2[1].hist(np.log10(r1),color='tab:blue',label = '$^{12}$C/$^{13}$C',density=True)
                ax2[1].hist(np.log10(r2),color='green',label = '$^{16}$O/$^{18}$O',density=True)
                ax2[1].legend()

                ax1[0].set_xlim(4.605,4.725)
                ax1[0].set_ylim(0.5,1.2)
                ax1[1].set_xlim(4.72,4.8)
                ax1[1].set_ylim(0.5,1.2)

                for secax in ax1[:]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.xaxis.set_major_locator(MultipleLocator(0.025))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(4))
                    secax.yaxis.set_major_locator(MultipleLocator(0.2))
                    secax.set_xlabel('Restframe wavelength ($\mu$m)',fontsize=fontsize)
                    secax.set_ylabel('Normalized flux',fontsize=fontsize,labelpad=2)

                ax2[0].set_xlim(-10, 60)
                ax2[0].set_ylim(14.5, 18)
                for secax in [ax2[0]]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    secax.xaxis.set_major_locator(MultipleLocator(20))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    secax.yaxis.set_major_locator(MultipleLocator(1))
                ax2[0].set_xlabel('Energy of J levels (K)', fontsize=fontsize)
                ax2[0].set_ylabel('$\\log $N(J)/g(J)', fontsize=fontsize,labelpad=2)

                ax2[1].set_xlim(0, 3.5)
                #ax[2].set_ylim(0,0.3)
                for secax in [ax2[1]]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    #secax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    #secax.xaxis.set_major_locator(MultipleLocator(10))
                    #secax.yaxis.set_minor_locator(AutoMinorLocator(2))
                    #secax.yaxis.set_major_locator(MultipleLocator(0.1))
                #ax[2].set_yticks([0, 0.2])
                ax2[1].set_xlabel('$\\log r=^{12}$C/$^{13}$C ($^{16}$O/$^{18}$O)', fontsize=fontsize)
                ax2[1].set_ylabel('PDF', fontsize=fontsize,labelpad=2)


                plt.show()

                str = 'J' + case + 'Miri_Av=' + str(Av) + '_H2' + '.pdf'
                f_name = '12CO13CO_1.pdf'
                fig1.savefig(f_name, bbox_inches='tight')
                f_name = '12CO13CO_2.pdf'
                fig2.savefig(f_name, bbox_inches='tight')



