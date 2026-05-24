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
    def __init__(self, sp_name = '',logN=20,fcold=1,Texc_cold=20,Texc_hot=300,redshift=0,b_doppler=50,logNJ=[0,1]):
        self.name = sp_name
        if self.name is not None:
            self.set_levels()
            self.logN = logN
            self.fcold=fcold
            self.Texc_cold = Texc_cold
            self.Texc_hot = Texc_hot
            self.z=redshift
            self.b = b_doppler
            self.logNJ = logNJ
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
        if self.name == 'CO':
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
    def calc_cols(self,mode='Texc'):
        if mode == 'Texc':
            logNcold = self.logN + np.log10(self.fcold)
            logNhot = self.logN + np.log10(1 - self.fcold)
            Zcold = np.sum(self.stat * np.exp(-self.energy / self.Texc_cold))
            Zhot = np.sum(self.stat * np.exp(-self.energy / self.Texc_hot))
            logN_c = logNcold + np.log10(self.stat * np.exp(-self.energy / self.Texc_cold)) - np.log10(Zcold)
            logN_h = logNhot + np.log10(self.stat * np.exp(-self.energy /  self.Texc_hot)) - np.log10(Zhot)
            self.cols = np.log10(10 ** logN_c + 10 ** logN_h)
        else:
            logN = np.zeros_like(self.stat).astype(float)
            self.cols = logN
            for i in range(np.size(self.logNJ)):
                self.cols[i] = self.logNJ[i]



logNH2tot = 21

case = 'J1830'
if case == 'J1830':
    bline = 100
    redshift = 0.0
    solidangle = np.pi*(2*np.pi/180/3600)**2
    mirires = 1
    ISFresolution = 3000
    SNRmiri = 100
    t9_7peak = 0.49
    t3_4 = 0.06
    Av = 5.8
    logNH2tot = 22.5
    logNCOtot = logNH2tot - 4.5


species = {}
H2  = element(sp_name='H2',logN=logNH2tot,fcold=1-1e-05,Texc_cold=20,Texc_hot=300)
H2.calc_cols()
species['H2'] = H2
CO  = element(sp_name='CO',logN=logNCOtot,fcold=1-1e-05,Texc_cold=20,Texc_hot=300)
CO.calc_cols()
species['CO'] = CO

CO_13  = element(sp_name='13CO',logN=logNCOtot + np.log10(1/60),fcold=1-1e-05,Texc_cold=20,Texc_hot=300)
CO_13.calc_cols()
species['13CO'] = CO_13

CO_18  = element(sp_name='C18O',logN=logNCOtot + np.log10(1/60),fcold=1-1e-05,Texc_cold=20,Texc_hot=300)
CO_18.calc_cols()
species['C18O'] = CO_18

if 0:
    plt.subplot()
    plt.plot(species['H2'].energy, species['H2'].cols - np.log10(species['H2'].stat) , 'o')
    plt.plot(species['CO'].energy, species['CO'].cols - np.log10(species['CO'].stat) , 'o')
    plt.plot(species['13CO'].energy, species['13CO'].cols - np.log10(species['13CO'].stat) , 'o')
#plt.show()
if 0:

    #H2 = H2list.Malec(0)
    H2_energy = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + r'/../energy_X_H2.dat', dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                              unpack=True, skip_header=3, comments='#')
    H2energy = np.zeros([max(H2_energy[0]) + 1, max(H2_energy[1]) + 1])
    for k,e in enumerate(H2_energy[0]):
        H2energy[e, H2_energy[1][k]] = H2_energy[2][k]
    stat_H2 = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]
    logNH2 = np.zeros_like(H2energy) - 10
    for i in range(2):
        for j in range(10):
            e = H2energy[i,j]
            if e<1500:
                logNH2[i,j] = np.log10(stat_H2[j]*np.exp(-e*1.44/(120)))
            else:
                logNH2[i, j] = -5.3 + np.log10(stat_H2[j] * np.exp(-e*1.44 / (1200)))
    logNH2+=logNH2tot

    plt.subplot()
    plt.plot(H2energy[0,:10],logNH2[0,:10]-np.log10(stat_H2[0:10]),'o')
    plt.plot(H2energy[1,:10],logNH2[1,:10]-np.log10(stat_H2[0:10]),'o')


    stat_CO = np.array([(2 * i + 1) for i in range(21)])
    logNCO = np.zeros(21)
    fcold = 0.
    logNCOcold = logNCOtot + np.log10(fcold)
    logNCOhot = logNCOtot + np.log10(1-fcold)
    Tcold,Thot = 20, 200
    COenergy = np.array([0.0, 3.84, 11.53, 23.06, 38.44, 57.67, 80.73, 107.64, 138.39, 172.97,211,40,253.66,299.76,349.69,403.46,461.05,522.47,587.72,656.78,729.67]) # in cm-1
    Zcold = np.sum(stat_CO*np.exp(-COenergy*1.44/Tcold))
    Zhot = np.sum(stat_CO*np.exp(-COenergy*1.44/Thot))
    logNCO1 = logNCOcold  + np.log10(stat_CO*np.exp(-COenergy*1.44/Tcold)) - np.log10(Zcold)
    logNCO2 = logNCOhot  + np.log10(stat_CO*np.exp(-COenergy*1.44/Thot)) - np.log10(Zhot)
    logNCO = np.log10(10**logNCO1 + 10**logNCO2)

    plt.subplot()
    plt.plot(COenergy,logNCO-np.log10(stat_CO),'o')
    plt.show()



micAA = 1e4 #micron to AA
mec_8_pi2_e2 = const.m_e.cgs.value * const.c.cgs.value/8/np.pi**2/const.e.gauss.value ** 2


lines = []
if 0:
    #HI k=1->i=0
    lam = 1215.6701
    gk,gi = 4,1
    Aki = 4.7e8
    
    
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

    stat_CO = np.array([(2 * i + 1) for i in range(np.size(COenergy))])

    logNCO = np.zeros(20)
    name, nl, jl, nu, ju = 'CO(0-1)R(9)', 0, 9, 1, 10
    lam = 4.5876 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 6.2459 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)R(8)', 0, 8, 1, 9
    lam = 4.5950 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 6.2690 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)R(7)', 0, 7, 1, 8
    lam = 4.6024 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 6.3056 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))


    name, nl, jl, nu, ju = 'CO(0-1)R(6)', 0, 6, 1, 7
    lam = 4.6100 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 6.3526 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)R(5)', 0, 5, 1, 6
    lam = 4.6177 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 6.4224 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)R(4)', 0, 4, 1, 5
    lam = 4.6254 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 6.5272 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)R(3)', 0, 3, 1, 4
    lam = 4.6333 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 6.7034 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name,nl,jl,nu,ju = 'CO(0-1)R(2)', 0,2,1,3
    lam = 4.6412 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 7.0260*1e-6
    Aki = gi * fik/(mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name,nl,jl,nu,ju = 'CO(0-1)R(1)', 0,1,1,2
    lam = 4.6493 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik =  7.7884*1e-6
    Aki = gi * fik/(mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)R(0)', 0, 0, 1, 1
    lam = 4.6575 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 11.6587 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(1)', 0, 1, 1, 0
    lam = 4.6742 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 3.8715 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(2)', 0, 2, 1, 1
    lam = 4.6826 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 4.6371 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(3)', 0, 3, 1, 2
    lam = 4.6912 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 4.9585 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(4)', 0, 4, 1,3
    lam = 4.7002 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.1308 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))


    name, nl, jl, nu, ju = 'CO(0-1)P(5)', 0, 5, 1,4
    lam = 4.7088 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.2382 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(6)', 0, 6, 1, 5
    lam = 4.7177 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.3079 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(7)', 0, 7, 1, 6
    lam = 4.7267 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.3558 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(8)', 0, 8, 1, 7
    lam = 4.7359 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.3908 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(9)', 0, 9, 1, 8
    lam = 4.7451 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.4154 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
    line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(10)', 0, 10, 1, 9
    lam = 4.7545 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.4303 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(11)', 0, 11, 1, 10
    lam = 4.7640 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.4428 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(12)', 0, 12, 1, 11
    lam = 4.7736 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.4530 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(13)', 0, 13, 1, 12
    lam = 4.7833 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.4596 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(14)', 0, 14, 1, 13
    lam = 4.7931 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.4610 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(15)', 0, 15, 1, 14
    lam = 4.8031 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.4646 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(16)', 0, 16, 1, 15
    lam = 4.8131 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.4616 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(17)', 0, 17, 1, 16
    lam = 4.8233 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 5.4622 * 1e-6
    Aki = gi * fik / (mec_8_pi2_e2 * (lam * 1e-8) ** 2 * gk)
    lines.append(
        line(name=name, l=lam, f=fik, g=1e9, nu_u=nu, j_u=ju, nu_l=nl, j_l=jl, b=bline, logN=logNCO[jl], Aki=Aki,
             z=redshift))

    name, nl, jl, nu, ju = 'CO(0-1)P(18)', 0, 18, 1, 17
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
        [0.0, 5.288, 15.8662, 31.7319, 52.8852, 79.3253, 111.0515, 148.0622, 190.3565])  # in K
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

    name, nl, jl, nu, ju = '13CO(0-1)P(3)', 0, 3, 1, 2
    lam = 4.8050 * micAA
    gk, gi = stat_CO[ju], stat_CO[jl]
    fik = 4.8078 * 1e-6
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


class spectrum():
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
            if 'CO' in tmpline.name:
                sp = 'CO'
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

        if convolve:
            self.resolution = ISFresolution
            # add CONVOLUTION
            self.observedflux = self.absorptionflux
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


    params = {}
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
    params['logNC18O'] = 16.5
    if 1:
        params['COJ0'] = 18
        params['COJ1'] = 18
        params['COJ2'] = 18
        params['COJ3'] = 18
        params['COJ4'] = 18
        params['COJ5'] = 18
        params['13COJ0'] = 18
        params['13COJ1'] = 18
        params['13COJ2'] = 18
        params['C18OJ0'] = 18
        params['C18OJ1'] = 18
        params['C18OJ2'] = 18
    params['z']  = 0.88582*(1+50/3e5)
    params['b'] = 20

    def calc_model(params,w_min=1e4,w_max=30e4,mode ='Texc',show_sp = [1,1,1,1]):
        species = {}
        if mode == 'Texc':
            H2 = element(sp_name='H2', logN=params['logNH2tot'], fcold=params['fH2cold'], Texc_cold=params['TexcH2_cold'],
                         Texc_hot=params['TexcH2_hot'], redshift=params['z'], b_doppler=params['b'])
            H2.calc_cols()
            species['H2'] = H2
            CO = element(sp_name='CO', logN=params['logNCOtot'], fcold=params['fCOcold'], Texc_cold=params['TexcCO_cold'],
                         Texc_hot=params['TexcCO_hot'], redshift=params['z'], b_doppler=params['b'],
                         logNJ=[params['COJ0'],params['COJ1'],params['COJ2'],params['COJ3'],params['COJ4'],params['COJ5']])
            CO.calc_cols(mode=mode)
            species['CO'] = CO
            CO_13 = element(sp_name='13CO', logN=params['logN13CO'], fcold=params['fCOcold'],
                            Texc_cold=params['TexcCO_cold'],
                            Texc_hot=params['TexcCO_hot'], redshift=params['z'], b_doppler=params['b'],
                            logNJ=[params['13COJ0'],params['13COJ1'],params['13COJ2']])
            CO_13.calc_cols(mode=mode)
            species['13CO'] = CO_13

            CO_18 = element(sp_name='C18O', logN=params['logNC18O'], fcold=params['fCOcold'],
                            Texc_cold=params['TexcCO_cold'],
                            Texc_hot=params['TexcCO_hot'], redshift=params['z'], b_doppler=params['b'],
                            logNJ=[params['C18OJ0'],params['C18OJ1'],params['C18OJ2']])
            CO_18.calc_cols(mode=mode)
            species['C18O'] = CO_18
        elif mode == 'all_J':
            if show_sp[1]:
                CO = element(sp_name='CO', logN=params['logNCOtot'], fcold=params['fCOcold'], Texc_cold=params['TexcCO_cold'],
                             Texc_hot=params['TexcCO_hot'], redshift=params['z'], b_doppler=params['b'],
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


        spec = spectrum(species=species,wave_min=w_min,wave_max=w_max)
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



if 0:
    calc_spec = 1
    if calc_spec:
        #ADD ABSORPTION LINES H2, CO
        for tmpline in lines:
            linetau = tau(line=tmpline,resolution=50000)
            Tau+=linetau.calctau(x=wavel, vel=False, debug=False, verbose=False, convolve=None, tlim=0.0001)
    
    
        #add DUST absorption
        t3_1 = 0
        if Av>3.3:
            t3_1=0.093*(Av-3.3) #0.04*t9_7peak
        dust_w, dust_tau, dust_b = [3.1,3.4,9.7,18],[t3_1,t3_4,t9_7peak,t9_7peak*0.42],[0.05,0.05,0.5,3]
        for j,w_dust in enumerate(dust_w):
            #dust_tau[j]/=10
            for k, lam in enumerate(wavel):
                b = 3e5*dust_b[j]/dust_w[j]
                x = (lam / w_dust/1e4 / (1 + redshift) - 1) * 3e5 / b
                if np.abs(x) < 3:
                    tau1 =  np.log(1 + gauss(x, 1) *dust_tau[j]/gauss(0, 1))
                    #print(gauss(0, 1))
                    Tau[k]+= tau1
        Absorptionflux = np.exp(-Tau)
    
    
    
        #ADD EMISSSION lines
    
        emission_flux = np.zeros_like(wavel)
        for tmpline in lines:
            if 'H2' in tmpline.name:
                hc = const.h.cgs.value*const.c.cgs.value
                Intensity =  hc/(tmpline.wavelength[0]*1e-8)*tmpline.Aki*10**logNH2[tmpline.nu_u,tmpline.j_u]/4/np.pi
                print(tmpline.name,tmpline.nu_u,tmpline.j_u,Intensity,logNH2[tmpline.nu_u,tmpline.j_u])
                tmpline.brgt = Intensity/1e-23/(const.c.cgs.value/(tmpline.wavelength[0]*1e-8))
                #1e-23 * const.c.cgs.value / (lam * 1e-4)
                for k,lam in enumerate(wavel):
                    x = (lam/tmpline.wavelength[0]/(1+redshift) - 1)*3e5/tmpline.b
                    if np.abs(x)<3:
                        emission_flux[k] += gauss(x, 1)*tmpline.brgt*solidangle
                if tmpline.name == 'H2S(1)' and 0:     # add PAH
                    dust_w, dust_b,dust_s = [6.2, 7.7,8.6,11.3,12.7], [0.1, 0.25,0.1,0.15,0.1],[2/5,1.5/5,1,1,1]
                    for j, w_dust in enumerate(dust_w):
                        for k, lam in enumerate(wavel):
                            b = 3e5 * dust_b[j] / dust_w[j]
                            x = (lam / w_dust/1e4 / (1 + redshift) - 1) * 3e5 / b
                            if np.abs(x) < 3:
                                emission_flux[k] += gauss(x, 1) * tmpline.brgt * solidangle
    
        if 1: # add PAH
            def Dij(lamij=5.0, gij=0.03, x=3):
                D = gij * lamij / ((x / lamij - lamij / x) ** 2 + gij ** 2)
                xint = np.linspace(lamij-3, lamij+3, 300)
                C = np.trapz(gij * lamij / ((xint / lamij - lamij / xint) ** 2 + gij ** 2), xint)
                return D / C
            if 1:
                F = np.zeros([4, np.size(wavel)])
                #6.2 micron
                lamij = [5.25, 5.70, 6.22, 6.69]
                gij = [0.03, 0.04, 0.0284, 0.07]
                bij = [0.227, 0.114, 0.536, 0.123]
    
                for i in range(4):
                    F[0,:]+= bij[i]*Dij(lamij[i],gij[i],wavel/1e4)
    
                # 7.6 micron
                lamij = [7.417, 7.598,7.85]
                gij = [0.126, 0.044, 0.053]
                bij = [0.24,0.389,0.371]
                for i in range(3):
                    F[1, :] += bij[i] * Dij(lamij[i], gij[i], wavel / 1e4)
    
                # 7.6 micron
                lamij = [8.33, 8.61]
                gij = [0.052, 0.039]
                bij = [0.189, 0.811]
                for i in range(2):
                    F[2, :] += bij[i] * Dij(lamij[i], gij[i], wavel / 1e4)
    
    
                # 11.3 micron
                lamij = [11.23, 11.30,11.99,12.61,13.6,14.19]
                gij = [0.01,0.029,0.05,0.0435,0.02,0.025]
                bij = [0.0732,0.335,0.349,0.234,0.0044,0.0048]
                for i in range(5):
                    F[3, :] += bij[i] * Dij(lamij[i], gij[i], wavel / 1e4)
    
    
            #TXS0218+357
            PAHbrgt = np.array([Av*2/2.5, Av*3/2,(Av-0.5)/2*1.5, (Av-0.5)*4/2])*1e6 #in Jy/sr
            lam = np.array([6.2,7.6,8.3,11.3])
            #brgt *= 1e-23*const.c.cgs.value/(lam*1e-4) # in erg/cm2/s/sr
            #print('brgt',brgt)
            #solidangle = np.pi*(3*np.pi/180/3600)**2
            pah_emission_flux = np.zeros_like(wavel)
            for i in range(4):
                pah_emission_flux += F[i,:] * PAHbrgt[i] * solidangle
            pah_emission_flux_interp = interp1d(wavel/1e4,pah_emission_flux, fill_value='extrapolate')
    
            emission_flux+=pah_emission_flux_interp(wavel/1e4/(1+redshift))
            plt.subplot()
            plt.plot(wavel/1e4,emission_flux)
            plt.show()
    
    
    
    
        # add MODEL SOURCE
    
        wisew = np.array([3.36,4.61,12.08,22.1]) #mkm
        wisefr = np.array([8.94e13,6.51e13, 2.59e13,1.36e13]) #Hz
    
    
    
        #wiseflux = np.array([5.9e-04, 0.00137, 0.00916,0.0199]) #Jy
        if case == '0218':
            wiseflux = np.array([3e-3, 5e-3, 13e-3, 20e-3])  # Jy #0218+357
        elif case == '0852':
            wisemag = np.array([14.23,12.97,9.79,7.531]) #0852+3435
            wiseflux = np.array([309.5,171.8,31.6,8.36])*10**(-wisemag/2.5)
        elif case == '1211':
            wisemag = np.array([14.98,13.99,10.62,8.011]) #1211
            wiseflux = np.array([309.5,171.8,31.6,8.36])*10**(-wisemag/2.5)
        elif case == 'J0006':
            wisemag = np.array([14.29, 12.74, 8.8, 6.5])  # 000
            wiseflux = np.array([309.5, 171.8, 31.6, 8.36]) * 10 ** (-wisemag / 2.5)
    
        wiseflux_interp = interp1d(wisew, np.log10(wiseflux), fill_value='extrapolate')
    
        fig,ax = plt.subplots()
        ax.plot(wavel/1e4, (Absorptionflux*10**wiseflux_interp(wavel/1e4) + emission_flux))
        ax.plot(wavel/1e4, emission_flux)
        ax.plot(wavel/1e4, (10**wiseflux_interp(wavel/1e4)))
        ax.plot(wavel / 1e4, (Absorptionflux * 10 ** wiseflux_interp(wavel / 1e4)))
    
        ax.set_title('Source and emission line flux')
    
        #add CONVOLUTION
        observedflux = (Absorptionflux*10**wiseflux_interp(wavel/1e4) + emission_flux)/10**wiseflux_interp(wavel/1e4)
        observedflux  = convolveflux(wavel, observedflux, res=ISFresolution,kind = 'direct')
    
    
    
        with open('model_'+case+str(logNH2tot)+'.pkl', 'wb') as f:
            pickle.dump([wavel, observedflux], f)
    
    else:
    
        if 0:
            with open('model_1211.pkl', 'rb') as f:
                w, wavel, observedflux, Flux_binned = pickle.load(f)
            fig3,ax3 = plt.subplots(2,3,figsize=(9,4))
            fontsize=8
            for axs in ax3[0,:]:
                axs.plot(wavel / 1e4/(1+redshift), observedflux, color='red', alpha=0.7)
                axs.step(w / 1e4/(1+redshift), Flux_binned, color='black', where='mid', zorder=-10)
                axs.set_ylim([-0.1, 1.1])
                axs.axhline(y=0,ls=':',lw=1,color='black')
            for axs in ax3[1, :]:
                axs.plot(wavel / 1e4/(1+redshift), observedflux, color='red', alpha=0.7)
                axs.step(w / 1e4/(1+redshift), Flux_binned, color='black', where='mid', zorder=-10)
                axs.set_ylim([-0.1, 1.1])
    
    
        if case == '0218' and 0:
            xlow, xup = 2.7, 3.7
            ax3[0,0].set_xlim(xlow,xup)
            ax3[0,0].set_ylim(-0.1,1.2)
            ax3[0,0].text(xlow+(xup-xlow)*0.05,0.1,'H$_2$O $(3.1{\\mu}m)$ and C-H $(3.4{\\mu}m)$',fontsize=fontsize)
            ax3[0, 0].text(xlow + (xup - xlow) * 0.6, 0.5, 'S/N = 100', fontsize=fontsize)
            ax3[0, 0].text(xlow + (xup - xlow) * 0.6, 0.37, 'MIRI MRS', fontsize=fontsize)
            ax3[0, 0].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            #ax3[0, 0].set_ylabel('Normalized flux, (${\\AA}$)', fontsize=fontsize)
            ax3[0, 0].plot([3.1,3.4],[1.05,1.05],'|',color='red')
    
            xlow, xup =4.6, 4.8
            ax3[0,1].set_xlim(xlow, xup)
            ax3[0,1].set_ylim(-0.1, 1.2)
            ax3[0, 1].text(xlow+(xup-xlow) * 0.05, 0.1, 'CO $\\nu\\to0$: J=0-10 $(4.6\\mu)$', fontsize=fontsize)
            ax3[0, 1].text(xlow + (xup - xlow) * 0.6, 0.5, 'S/N = 100', fontsize=fontsize)
            ax3[0, 1].text(xlow + (xup - xlow) * 0.6, 0.37, 'MIRI MRS', fontsize=fontsize)
            ax3[0, 1].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            for tmpline in lines:
                if 'CO' in tmpline.name:
                    ax3[0, 1].plot(tmpline.wavelength[0]/1e4, 1.05, '|', color='red')
    
            xlow, xup = 8,12
            ax3[0,2].set_xlim(xlow, xup)
            ax3[0,2].set_ylim(-0.1, 1.3)
            ax3[0, 2].text(xlow+(xup-xlow) * 0.05, 0.1, 'Si-O $(9.7{\\mu}m)$', fontsize=fontsize)
            ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.5, 'S/N ='+str(SNRmiri), fontsize=fontsize)
            ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.37, 'MIRI MRS', fontsize=fontsize)
            ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            ax3[0, 2].plot(9.7, 1.05, '|', color='red')
    
            xlow, xup = 4.8,6.5
            ax3[1, 0].set_xlim(xlow, xup)
            ax3[1, 0].set_ylim(-0.1, 1.3)
            ax3[1, 0].text(xlow+(xup-xlow) * 0.05, 0.05, 'PAH $(6.2{\\mu}m)$', fontsize=fontsize)
            ax3[1,0].text(xlow + (xup - xlow) * 0.6, 0.50, 'S/N ='+str(SNRmiri), fontsize=fontsize)
            ax3[1,0].text(xlow + (xup - xlow) * 0.6, 0.37, 'MIRI MRS', fontsize=fontsize)
            ax3[1,0].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            lamij = [5.25, 5.70, 6.22, 6.69]
            for l in lamij:
                ax3[1,0].plot(l, 0.95, '|', color='red')
    
            xlow, xup = 7,9
            ax3[1, 1].set_xlim(xlow, xup)
            ax3[1, 1].set_ylim(-0.1, 1.3)
            ax3[1, 1].text(xlow+(xup-xlow) * 0.05, 0.05, 'PAH $(7.6{\\mu}m$ and $8.6{\\mu}m)$', fontsize=fontsize)
            ax3[1, 1].text(xlow + (xup - xlow) * 0.6, 0.50, 'S/N = 100', fontsize=fontsize)
            ax3[1, 1].text(xlow + (xup - xlow) * 0.6, 0.37, 'MIRI MRS', fontsize=fontsize)
            ax3[1, 1].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            lamij = [7.417, 7.598, 7.85, 8.33, 8.61, 11.23, 11.30, 11.99, 12.61, 13.6, 14.19]
            for l in lamij:
                ax3[1,1].plot(l, 0.95, '|', color='red')
    
            xlow, xup = 10.5,12
            ax3[1, 2].set_xlim(xlow, xup)
            ax3[1, 2].set_ylim(-0.1, 1.3)
            ax3[1, 2].text(xlow+(xup-xlow) * 0.05, 0.05, 'PAH $(11.3\\mu)$', fontsize=fontsize)
            ax3[1, 2].text(xlow + (xup - xlow) * 0.6, 0.50, 'S/N = 100', fontsize=fontsize)
            ax3[1, 2].text(xlow + (xup - xlow) * 0.6, 0.37, 'MIRI MRS', fontsize=fontsize)
            ax3[1, 2].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            for l in lamij:
                ax3[1,2].plot(l, 0.95, '|', color='red')
    
    
            for secax in ax3[0,:]:
                secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                  top='True')
                secax.tick_params(which='major', length=5)
                secax.tick_params(which='minor', length=3)
                secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                secax.xaxis.set_major_locator(MultipleLocator(0.5))
                secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                secax.yaxis.set_major_locator(MultipleLocator(0.5))
    
    
            ax3[0,1].xaxis.set_minor_locator(AutoMinorLocator(4))
            ax3[0,1].xaxis.set_major_locator(MultipleLocator(0.1))
            ax3[0,2].xaxis.set_minor_locator(AutoMinorLocator(4))
            ax3[0,2].xaxis.set_major_locator(MultipleLocator(1))
    
            for secax in ax3[1,:]:
                secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                  top='True')
                secax.tick_params(which='major', length=5)
                secax.tick_params(which='minor', length=3)
                secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                secax.xaxis.set_major_locator(MultipleLocator(0.5))
                secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                secax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax3[1,1].set_xlabel('Rest-frame wavelength at $z_{\\rm abs}=0.685$, (${\\mu}m$)',fontsize=fontsize)
            ax3[0, 0].text(2.5,-1,'Flux normalized to continuum level', fontsize=fontsize,rotation=90)
            ax3[0, 0].text(2.7,1.3,'Quasar J 0218+357',fontsize=fontsize)
        elif case == '0218':
            with open('model_' + case + '.pkl', 'rb') as f:
                wavel, observedflux = pickle.load(f)
    
                # Rebinning
            if 1:
                w_new = [wavel[0]]
    
    
                def mirires_app(l=3, miri=True):
                    x = [5, 9, 12, 20, 25]
                    if miri:
                        y = [3800, 3500, 3000, 2000, 1500]
                    # y = [100,100,100,100,100]
                    res = interp1d(x, y, fill_value='extrapolate')
                    return res(l)
    
    
                for k in range(10000000):
                    w_new.append(w_new[k] + w_new[k] / mirires_app(w_new[k] / 1e4))
                    if w_new[k + 1] > 28 * 1e4:
                        break
                w_new = np.array(w_new)
    
                Flux_binned = np.zeros_like(w_new)
                for k, x in enumerate(w_new):
                    if k > 0 and k < np.size(w_new) - 1:
                        mask = (wavel < w_new[k + 1]) * (wavel > w_new[k - 1])
                        Flux_binned[k] = np.mean(observedflux[mask])
    
                # add S/N
                Flux_err = np.random.normal(loc=0.0, scale=1.0, size=len(w_new))
                Flux_binned = Flux_binned * (1 + 1 / SNRmiri * Flux_err)
    
            fig3, ax3 = plt.subplots(2, 3, figsize=(9, 4))
            fontsize = 8
            for axs in ax3[0, :]:
                axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                axs.set_ylim([-0.1, 1.1])
                axs.axhline(y=0, ls=':', lw=1, color='black')
            for axs in ax3[1, :]:
                axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                axs.set_ylim([-0.1, 1.1])
    
            xlow, xup = 2.7, 3.7
            ylow, yup = -0.1, 1.2
            ax3[0, 0].set_xlim(xlow, xup)
            ax3[0, 0].set_ylim(ylow, yup)
            ax3[0, 0].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1,
                           'H$_2$O $(3.1{\\mu}m)$ and C-H $(3.4{\\mu}m)$',
                           fontsize=fontsize)
            # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N ='+str(SNRmiri), fontsize=fontsize)
            # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.27, '$A_V$=1.1', fontsize=fontsize)
            # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated)', fontsize=fontsize)
            # ax3[0, 0].set_ylabel('Normalized flux, (${\\AA}$)', fontsize=fontsize)
            ax3[0, 0].plot([3.1, 3.4], [1.05, 1.05], '|', color='red')
    
            xlow, xup = 4.6, 4.8
            ax3[0, 1].set_xlim(xlow, xup)
            ax3[0, 1].set_ylim(ylow, yup)
            ax3[0, 1].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'CO $\\nu\\to0$: J=0-10 $(4.6\\mu)$',
                           fontsize=fontsize)
            # ax3[0, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N = 100', fontsize=fontsize)
            ax3[0, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup - ylow) * 0.27, '$N(CO)=10^{18}$', fontsize=fontsize)
            # ax3[0, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated', fontsize=fontsize)
            for tmpline in lines:
                if 'CO' in tmpline.name:
                    ax3[0, 1].plot(tmpline.wavelength[0] / 1e4, 1.05, '|', color='red')
    
            xlow, xup = 8, 12
            ax3[0, 2].set_xlim(xlow, xup)
            ax3[0, 2].set_ylim(-0.1, 1.3)
            ax3[0, 2].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'Si-O $(9.7{\\mu}m)$', fontsize=fontsize)
            # ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.5, 'S/N =' + str(SNRmiri), fontsize=fontsize)
            ax3[0, 2].text(xlow + (xup - xlow) * 0.6, ylow + (yup - ylow) * 0.27, '$\\tau_{9.7}=0.49$', fontsize=fontsize)
            # ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            ax3[0, 2].plot(9.7, 1.05, '|', color='red')
    
            xlow, xup = 4.8, 6.5
            ylow, yup = 0.5, 1.3
            ax3[1, 0].set_xlim(xlow, xup)
            ax3[1, 0].set_ylim(ylow, yup)
            ax3[1, 0].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'PAH $(6.2{\\mu}m)$', fontsize=fontsize)
            # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.25, 'S/N =' + str(SNRmiri), fontsize=fontsize)
            # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.17, '$A_V$=1.1', fontsize=fontsize)
            # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.1, '(simulated)', fontsize=fontsize)
            lamij = [5.25, 5.70, 6.22, 6.69]
            for l in lamij:
                ax3[1, 0].plot(l, 0.95, '|', color='red')
    
            xlow, xup = 7, 9
            ax3[1, 1].set_xlim(xlow, xup)
            ax3[1, 1].set_ylim(ylow, yup)
            ax3[1, 1].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'PAH $(7.6{\\mu}m$ and $8.6{\\mu}m)$',
                           fontsize=fontsize)
            # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.75, 'S/N = 50', fontsize=fontsize)
            # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.65, '$A_V$=1.1', fontsize=fontsize)
            # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            lamij = [7.417, 7.598, 7.85, 8.33, 8.61, 11.23, 11.30, 11.99, 12.61, 13.6, 14.19]
            for l in lamij:
                ax3[1, 1].plot(l, 0.95, '|', color='red')
    
            xlow, xup = 10.5, 12
            ax3[1, 2].set_xlim(xlow, xup)
            ax3[1, 2].set_ylim(ylow, yup)
            ax3[1, 2].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'PAH $(11.3\\mu)$', fontsize=fontsize)
            # ax3[1, 2].text(xlow + (xup - xlow) * 0.6, 0.50, 'S/N = 50', fontsize=fontsize)
            # ax3[1, 2].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.17, '$A_V$=1.1', fontsize=fontsize)
            # ax3[1, 2].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            for l in lamij:
                ax3[1, 2].plot(l, 0.95, '|', color='red')
    
            for secax in ax3[0, :]:
                secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                  top='True')
                secax.tick_params(which='major', length=5)
                secax.tick_params(which='minor', length=3)
                secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                secax.xaxis.set_major_locator(MultipleLocator(0.5))
                secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                secax.yaxis.set_major_locator(MultipleLocator(0.5))
    
            ax3[0, 1].xaxis.set_minor_locator(AutoMinorLocator(4))
            ax3[0, 1].xaxis.set_major_locator(MultipleLocator(0.1))
            ax3[0, 2].xaxis.set_minor_locator(AutoMinorLocator(4))
            ax3[0, 2].xaxis.set_major_locator(MultipleLocator(1))
    
            for secax in ax3[1, :]:
                secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                  top='True')
                secax.tick_params(which='major', length=5)
                secax.tick_params(which='minor', length=3)
                secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                secax.xaxis.set_major_locator(MultipleLocator(0.5))
                secax.yaxis.set_minor_locator(AutoMinorLocator(4))
                secax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax3[1, 1].set_xlabel('Rest-frame wavelength, (${\\mu}m$)', fontsize=fontsize)
            ax3[0, 0].text(2.5, -1, 'Flux normalized to the quasar continuum', fontsize=fontsize, rotation=90)
            #ax3[0, 0].text(2.7, 1.25, 'Quasar J 0218+357 MIRI MRS (Simulated spectrum), S/N = ' + str(
            #    SNRmiri) + ', $A_{\\rm V}$=5.8, $z_{\\rm abs}=0.685$', fontsize=fontsize)
            ax3[0, 0].text(2.7, 1.25, 'Quasar J 0218+357 MIRI MRS (Simulated spectrum), $A_{\\rm V}$=5.8, $z_{\\rm abs}=0.685$', fontsize=fontsize)
        elif case == '0852':
            with open('model_'+case+'.pkl', 'rb') as f:
                wavel, observedflux = pickle.load(f)
    
                # Rebinning
            if 1:
                w_new = [wavel[0]]
    
    
                def mirires_app(l=3, miri=True):
                    x = [5, 9, 12, 20, 25]
                    if miri:
                        y = [3800, 3500, 3000, 2000, 1500]
                    # y = [100,100,100,100,100]
                    res = interp1d(x, y, fill_value='extrapolate')
                    return res(l)
    
    
                for k in range(10000000):
                    w_new.append(w_new[k] + w_new[k] / mirires_app(w_new[k] / 1e4))
                    if w_new[k + 1] > 28 * 1e4:
                        break
                w_new = np.array(w_new)
    
                Flux_binned = np.zeros_like(w_new)
                for k, x in enumerate(w_new):
                    if k > 0 and k < np.size(w_new) - 1:
                        mask = (wavel < w_new[k + 1]) * (wavel > w_new[k - 1])
                        Flux_binned[k] = np.mean(observedflux[mask])
    
                # add S/N
                def SNR_appr(lam,debug=0):
                    x = [9,20,24,26,28]
                    y = [98,96,30,15,3]
                    f = interp1d(x,y,fill_value='extrapolate')
                    if debug:
                        fig, ax = plt.subplots()
                        ax.plot(x,y,'o')
                        ax.plot(np.arange(30),f(np.arange(30)))
                        plt.show()
                    return f(lam)
                SNR_appr(5)
    
                Flux_err = np.random.normal(loc=0.0, scale=1.0, size=len(w_new))
                #Flux_binned = Flux_binned * (1 + 1 / SNRmiri * Flux_err)
                Flux_binned = Flux_binned * (1 + 1 / SNR_appr(w_new/1e4) * Flux_err)
    
            mode = 'short'
            if mode == 'standard':
                fig3, ax3 = plt.subplots(2, 3, figsize=(9, 4))
                fontsize = 8
                for axs in ax3[0, :]:
                    axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                    axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                    axs.set_ylim([-0.1, 1.1])
                    axs.axhline(y=0, ls=':', lw=1, color='black')
                for axs in ax3[1, :]:
                    axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                    axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                    axs.set_ylim([-0.1, 1.1])
    
    
                xlow, xup = 2.7, 3.7
                ylow,yup = 0.7,1.1
                ax3[0, 0].set_xlim(xlow, xup)
                ax3[0, 0].set_ylim(ylow,yup)
                ax3[0, 0].text(xlow + (xup - xlow) * 0.05, ylow + (yup-ylow)*0.1, 'H$_2$O $(3.1{\\mu}m)$ and C-H $(3.4{\\mu}m)$',
                               fontsize=fontsize)
                #ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N ='+str(SNRmiri), fontsize=fontsize)
                #ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.27, '$A_V$=1.1', fontsize=fontsize)
                #ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated)', fontsize=fontsize)
                # ax3[0, 0].set_ylabel('Normalized flux, (${\\AA}$)', fontsize=fontsize)
                ax3[0, 0].plot([3.1, 3.4], [1.05, 1.05], '|', color='red')
    
                xlow, xup = 4.6, 4.8
                ax3[0, 1].set_xlim(xlow, xup)
                ax3[0, 1].set_ylim(ylow,yup)
                ax3[0, 1].text(xlow + (xup - xlow) * 0.05,  ylow + (yup-ylow)*0.1, 'CO $\\nu\\to0$: J=0-10 $(4.6\\mu)$', fontsize=fontsize)
                #ax3[0, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N = 100', fontsize=fontsize)
                ax3[0, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.27, '$N(CO)=10^{17}$', fontsize=fontsize)
                #ax3[0, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated', fontsize=fontsize)
                for tmpline in lines:
                    if 'CO' in tmpline.name:
                        ax3[0, 1].plot(tmpline.wavelength[0] / 1e4, 1.05, '|', color='red')
    
                xlow, xup = 8, 12
                ax3[0, 2].set_xlim(xlow, xup)
                ax3[0, 2].set_ylim(-0.1, 1.3)
                ax3[0, 2].text(xlow + (xup - xlow) * 0.05, 0.1, 'Si-O $(9.7{\\mu}m)$', fontsize=fontsize)
                #ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.5, 'S/N =' + str(SNRmiri), fontsize=fontsize)
                ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.37, '$\\tau_{9.7}=0.49$', fontsize=fontsize)
                #ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
                ax3[0, 2].plot(9.7, 1.05, '|', color='red')
    
                xlow, xup = 4.8, 6.5
                ylow, yup = 0.5, 1.3
                ax3[1, 0].set_xlim(xlow, xup)
                ax3[1, 0].set_ylim(ylow, yup )
                ax3[1, 0].text(xlow + (xup - xlow) * 0.05, ylow + (yup-ylow)*0.1, 'PAH $(6.2{\\mu}m)$', fontsize=fontsize)
                #ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.25, 'S/N =' + str(SNRmiri), fontsize=fontsize)
                #ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.17, '$A_V$=1.1', fontsize=fontsize)
                #ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.1, '(simulated)', fontsize=fontsize)
                lamij = [5.25, 5.70, 6.22, 6.69]
                for l in lamij:
                    ax3[1, 0].plot(l, 0.95, '|', color='red')
    
                xlow, xup = 7, 9
                ax3[1, 1].set_xlim(xlow, xup)
                ax3[1, 1].set_ylim(ylow, yup)
                ax3[1, 1].text(xlow + (xup - xlow) * 0.05, ylow + (yup-ylow)*0.1, 'PAH $(7.6{\\mu}m$ and $8.6{\\mu}m)$', fontsize=fontsize)
                #ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.75, 'S/N = 50', fontsize=fontsize)
                #ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.65, '$A_V$=1.1', fontsize=fontsize)
                #ax3[1, 1].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
                lamij = [7.417, 7.598, 7.85, 8.33, 8.61, 11.23, 11.30, 11.99, 12.61, 13.6, 14.19]
                for l in lamij:
                    ax3[1, 1].plot(l, 0.95, '|', color='red')
    
                xlow, xup = 10.5, 12
                ax3[1, 2].set_xlim(xlow, xup)
                ax3[1, 2].set_ylim(ylow, yup)
                ax3[1, 2].text(xlow + (xup - xlow) * 0.05, ylow + (yup-ylow)*0.1, 'PAH $(11.3\\mu)$', fontsize=fontsize)
                #ax3[1, 2].text(xlow + (xup - xlow) * 0.6, 0.50, 'S/N = 50', fontsize=fontsize)
                #ax3[1, 2].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.17, '$A_V$=1.1', fontsize=fontsize)
                #ax3[1, 2].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
                for l in lamij:
                    ax3[1, 2].plot(l, 0.95, '|', color='red')
    
                for secax in ax3[0, :]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.xaxis.set_major_locator(MultipleLocator(0.5))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.yaxis.set_major_locator(MultipleLocator(0.1))
    
                ax3[0, 1].xaxis.set_minor_locator(AutoMinorLocator(4))
                ax3[0, 1].xaxis.set_major_locator(MultipleLocator(0.1))
                ax3[0, 2].xaxis.set_minor_locator(AutoMinorLocator(4))
                ax3[0, 2].xaxis.set_major_locator(MultipleLocator(1))
                ax3[0, 2].yaxis.set_minor_locator(AutoMinorLocator(5))
                ax3[0, 2].yaxis.set_major_locator(MultipleLocator(0.5))
    
                for secax in ax3[1, :]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.xaxis.set_major_locator(MultipleLocator(0.5))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(4))
                    secax.yaxis.set_major_locator(MultipleLocator(0.2))
                ax3[1, 1].set_xlabel('Rest-frame wavelength, (${\\mu}m$)', fontsize=fontsize)
                ax3[0, 0].text(2.5, 0.5, 'Flux normalized to quasar continuum level', fontsize=fontsize, rotation=90)
                #ax3[0, 0].text(2.7, 1.12, 'Quasar J 0852+3435 MIRI MRS (Simulated spectrum), S/N = '+str(SNRmiri)+', $A_{\\rm V}$=1.1, $z_{\\rm abs}=1.309$', fontsize=fontsize)
                ax3[0, 0].text(2.7, 1.12, 'Quasar J 0852+3435 MIRI MRS (Simulated spectrum), S/N = 5-100, $A_{\\rm V}$=1.1, $z_{\\rm abs}=1.309$', fontsize=fontsize)
            elif mode == 'short':
                fig3, ax3 = plt.subplots(1, 5, figsize=(15, 3))
                fontsize = 8
                for axs in ax3[:]:
                    axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                    axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                    axs.set_ylim([-0.1, 1.1])
                    axs.axhline(y=0, ls=':', lw=1, color='black')
    
                xlow, xup = 2.7, 3.7
                ylow, yup = 0.7, 1.15
                ax3[0].set_xlim(xlow, xup)
                ax3[0].set_ylim(ylow, yup)
                ax3[0].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1,
                               'H$_2$O $(3.1{\\mu}m)$ and C-H $(3.4{\\mu}m)$',
                               fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N ='+str(SNRmiri), fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.27, '$A_V$=1.1', fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated)', fontsize=fontsize)
                # ax3[0, 0].set_ylabel('Normalized flux, (${\\AA}$)', fontsize=fontsize)
                ax3[0].plot([3.1, 3.4], [1.05, 1.05], '|', color='red')
    
                xlow, xup = 4.6, 4.8
                ax3[1].set_xlim(xlow, xup)
                ax3[1].set_ylim(ylow, yup)
                ax3[1].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'CO $\\nu\\to0$: J=0-10 $(4.6\\mu)$',
                               fontsize=fontsize)
                # ax3[0, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N = 100', fontsize=fontsize)
                ax3[1].text(xlow + (xup - xlow) * 0.6, ylow + (yup - ylow) * 0.27, '$N(CO)=10^{17}$', fontsize=fontsize)
                # ax3[0, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated', fontsize=fontsize)
                for tmpline in lines:
                    if 'CO' in tmpline.name:
                        ax3[1].plot(tmpline.wavelength[0] / 1e4, 1.05, '|', color='red')
    
                xlow, xup = 8, 12
                ax3[2].set_xlim(xlow, xup)
                ax3[2].set_ylim(ylow, yup)
                ax3[2].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'Si-O $(9.7{\\mu}m)$', fontsize=fontsize)
                # ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.5, 'S/N =' + str(SNRmiri), fontsize=fontsize)
                ax3[2].text(xlow + (xup - xlow) * 0.6, ylow + (yup - ylow) * 0.27, '$\\tau_{9.7}=0.49$', fontsize=fontsize)
                # ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
                ax3[2].plot(9.7, 1.05, '|', color='red')
    
                xlow, xup = 4.8, 6.5
                ax3[3].set_xlim(xlow, xup)
                ax3[3].set_ylim(ylow, yup)
                ax3[3].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'PAH $(6.2{\\mu}m)$',
                               fontsize=fontsize)
                # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.25, 'S/N =' + str(SNRmiri), fontsize=fontsize)
                # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.17, '$A_V$=1.1', fontsize=fontsize)
                # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.1, '(simulated)', fontsize=fontsize)
                lamij = [5.25, 5.70, 6.22, 6.69]
                for l in lamij:
                    ax3[3].plot(l, 0.95, '|', color='red')
    
                xlow, xup = 7, 9
                ax3[4].set_xlim(xlow, xup)
                ax3[4].set_ylim(ylow, yup)
                ax3[4].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'PAH $(7.6{\\mu}m$ and $8.6{\\mu}m)$',
                               fontsize=fontsize)
                # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.75, 'S/N = 50', fontsize=fontsize)
                # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.65, '$A_V$=1.1', fontsize=fontsize)
                # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
                lamij = [7.417, 7.598, 7.85, 8.33, 8.61, 11.23, 11.30, 11.99, 12.61, 13.6, 14.19]
                for l in lamij:
                    ax3[4].plot(l, 0.95, '|', color='red')
    
    
                for secax in ax3[:]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.xaxis.set_major_locator(MultipleLocator(0.5))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.yaxis.set_major_locator(MultipleLocator(0.1))
                    secax.set_xlabel('Rest-frame wavelength, (${\\mu}m$)', fontsize=fontsize)
    
                ax3[1].xaxis.set_minor_locator(AutoMinorLocator(4))
                ax3[1].xaxis.set_major_locator(MultipleLocator(0.1))
                ax3[2].xaxis.set_minor_locator(AutoMinorLocator(4))
                ax3[2].xaxis.set_major_locator(MultipleLocator(1))
                #ax3[2].yaxis.set_minor_locator(AutoMinorLocator(5))
                #ax3[2].yaxis.set_major_locator(MultipleLocator(0.5))
    
    
                ax3[0].set_ylabel( 'Flux normalized to the quasar continuum', fontsize=fontsize)
                # ax3[0, 0].text(2.7, 1.12, 'Quasar J 0852+3435 MIRI MRS (Simulated spectrum), S/N = '+str(SNRmiri)+', $A_{\\rm V}$=1.1, $z_{\\rm abs}=1.309$', fontsize=fontsize)
                ax3[0].text(2.7, 1.17,
                               'Quasar J 0852+3435 MIRI MRS (Simulated spectrum), $A_{\\rm V}$=1.1, $A_{\\rm 2175}=0.35$, $z_{\\rm abs}=1.309$',
                               fontsize=fontsize)
        elif case == '1211':
            with open('model_1211.pkl', 'rb') as f:
                wavel, observedflux = pickle.load(f)
    
                # Rebinning
            if 1:
                w_new = [wavel[0]]
    
    
                def mirires_app(l=3, miri=True):
                    x = [5, 9, 12, 20, 25]
                    if miri:
                        y = [3800, 3500, 3000, 2000, 1500]
                    # y = [100,100,100,100,100]
                    res = interp1d(x, y, fill_value='extrapolate')
                    return res(l)
    
    
                for k in range(10000000):
                    w_new.append(w_new[k] + w_new[k] / mirires_app(w_new[k] / 1e4))
                    if w_new[k + 1] > 28 * 1e4:
                        break
                w_new = np.array(w_new)
    
                Flux_binned = np.zeros_like(w_new)
                for k, x in enumerate(w_new):
                    if k > 0 and k < np.size(w_new) - 1:
                        mask = (wavel < w_new[k + 1]) * (wavel > w_new[k - 1])
                        Flux_binned[k] = np.mean(observedflux[mask])
    
                # add S/N
                def SNR_appr(lam,debug=0):
                    x = [3,16,17,18,24,26,28]
                    y = [110,110,100,85,18,8,2]
                    f = interp1d(x,y,fill_value='extrapolate')
                    if debug:
                        fig, ax = plt.subplots()
                        ax.plot(x,y,'o')
                        ax.plot(np.arange(30),f(np.arange(30)))
                        plt.show()
                    return f(lam)
                SNR_appr(5)
    
                Flux_err = np.random.normal(loc=0.0, scale=1.0, size=len(w_new))
                #Flux_binned = Flux_binned * (1 + 1 / SNRmiri * Flux_err)
                Flux_binned = Flux_binned * (1 + 1 / SNR_appr(w_new/1e4) * Flux_err)
            mode = 'PAH6.2'
            if mode == '3 panels':
                fig3, ax3 = plt.subplots(1, 3, figsize=(9, 2))
                fontsize = 8
                for axs in ax3[:]:
                    axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                    axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                    axs.set_ylim([-0.1, 1.1])
                    axs.axhline(y=0, ls=':', lw=1, color='black')
    
    
                xlow, xup = 2.7, 3.7
                ylow,yup = 0.8,1.2
                ax3[0].set_xlim(xlow, xup)
                ax3[0].set_ylim(ylow,yup)
                ax3[0].text(xlow + (xup - xlow) * 0.05, ylow + (yup-ylow)*0.1, 'H$_2$O $(3.1{\\mu}m)$ and C-H $(3.4{\\mu}m)$',
                               fontsize=fontsize)
                #ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N ='+str(SNRmiri), fontsize=fontsize)
                #ax3[0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '$A_V$=1.1', fontsize=fontsize)
                #ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated)', fontsize=fontsize)
                # ax3[0, 0].set_ylabel('Normalized flux, (${\\AA}$)', fontsize=fontsize)
                ax3[0].plot([3.1, 3.4], [1.05, 1.05], '|', color='red')
    
    
                xlow, xup = 4.8, 6.5
                ylow, yup = 0.8, 1.2
                ax3[1].set_xlim(xlow, xup)
                ax3[1].set_ylim(ylow, yup )
                ax3[1].text(xlow + (xup - xlow) * 0.05, ylow + (yup-ylow)*0.1, 'PAH $(6.2{\\mu}m)$', fontsize=fontsize)
                #ax31, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.25, 'S/N =' + str(SNRmiri), fontsize=fontsize)
                #ax3[1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '$A_V$=1.1', fontsize=fontsize)
                #ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.1, '(simulated)', fontsize=fontsize)
                lamij = [5.25, 5.70, 6.22, 6.69]
                for l in lamij:
                    ax3[1].plot(l, 0.95, '|', color='red')
    
                xlow, xup = 7, 9
                ax3[2].set_xlim(xlow, xup)
                ax3[2].set_ylim(ylow, yup)
                ax3[2].text(xlow + (xup - xlow) * 0.05, ylow + (yup-ylow)*0.1, 'PAH $(7.6{\\mu}m$ and $8.6{\\mu}m)$', fontsize=fontsize)
                #ax3[2].text(xlow + (xup - xlow) * 0.6, ylow + (yup - ylow) * 0.2, '$A_V$=1.1', fontsize=fontsize)
                #ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.75, 'S/N = 50', fontsize=fontsize)
                #ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.65, '$A_V$=1.1', fontsize=fontsize)
                #ax3[1, 1].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
                lamij = [7.417, 7.598, 7.85, 8.33, 8.61, 11.23, 11.30, 11.99, 12.61, 13.6, 14.19]
                for l in lamij:
                    ax3[2].plot(l, 0.95, '|', color='red')
    
    
    
                for secax in ax3[:]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.xaxis.set_major_locator(MultipleLocator(0.5))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.yaxis.set_major_locator(MultipleLocator(0.1))
    
    
    
                ax3[1].set_xlabel('Rest-frame wavelength, (${\\mu}m$)', fontsize=fontsize)
                ax3[0].text(2.5, 0.9, 'Normalized flux', fontsize=fontsize, rotation=90)
                #ax3[0].text(2.7, 1.22, 'Quasar J 1211+0833 MIRI MRS (Simulated spectrum), S/N = '+str(SNRmiri)+', $A_{\\rm V}=0.8$, $z_{\\rm abs}=2.117$', fontsize=fontsize)
                ax3[0].text(2.7, 1.22, 'Quasar J 1211+0833 MIRI MRS (Simulated spectrum), $A_{\\rm V}=0.8$, $z_{\\rm abs}=2.117$', fontsize=fontsize)
            elif mode == 'PAH6.2':
                fig3, ax3 = plt.subplots(figsize=(3, 2))
                fontsize = 8
                for axs in [ax3]:
                    axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                    axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                    axs.set_ylim([-0.1, 1.1])
                    axs.axhline(y=0, ls=':', lw=1, color='black')
    
    
                xlow, xup = 4.8, 6.5
                ylow, yup = 0.5, 1.2
                ax3.set_xlim(xlow, xup)
                ax3.set_ylim(ylow, yup)
                ax3.text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'PAH $(6.2{\\mu}m)$', fontsize=fontsize)
                # ax31, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.25, 'S/N =' + str(SNRmiri), fontsize=fontsize)
                # ax3[1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '$A_V$=1.1', fontsize=fontsize)
                # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.1, '(simulated)', fontsize=fontsize)
                lamij = [5.25, 5.70, 6.22, 6.69]
                for l in lamij:
                    ax3.plot(l, 0.95, '|', color='red')
    
    
                for secax in [ax3]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.xaxis.set_major_locator(MultipleLocator(0.5))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(4))
                    secax.yaxis.set_major_locator(MultipleLocator(0.2))
    
                ax3.set_xlabel('Rest-frame wavelength, (${\\mu}m$)', fontsize=fontsize)
                ax3.set_ylabel('Normalized flux', fontsize=fontsize)
                # ax3[0].text(2.7, 1.22, 'Quasar J 1211+0833 MIRI MRS (Simulated spectrum), S/N = '+str(SNRmiri)+', $A_{\\rm V}=0.8$, $z_{\\rm abs}=2.117$', fontsize=fontsize)
                ax3.set_title('Quasar J 1211+0833, $A_{\\rm V}=0.8$, $z_{\\rm abs}=2.117$', fontsize=fontsize)
                ax3.text(5.8,0.7,'MIRI MRS', fontsize=fontsize)
                ax3.text(5.8,0.63,'(simulated)', fontsize=fontsize)
        elif case == 'J0006':
            with open('model_'+case+str(logNH2tot)+'.pkl', 'rb') as f:
                wavel, observedflux = pickle.load(f)
    
                # Rebinning
            if 1:
                w_new = [wavel[0]]
    
    
                def mirires_app(l=3, miri=True):
                    x = [5, 9, 12, 20, 25]
                    if miri:
                        y = [3800, 3500, 3000, 2000, 1500]
                    # y = [100,100,100,100,100]
                    res = interp1d(x, y, fill_value='extrapolate')
                    return res(l)
    
    
                for k in range(10000000):
                    w_new.append(w_new[k] + w_new[k] / mirires_app(w_new[k] / 1e4))
                    if w_new[k + 1] > 28 * 1e4:
                        break
                w_new = np.array(w_new)
    
                Flux_binned = np.zeros_like(w_new)
                for k, x in enumerate(w_new):
                    if k > 0 and k < np.size(w_new) - 1:
                        mask = (wavel < w_new[k + 1]) * (wavel > w_new[k - 1])
                        Flux_binned[k] = np.mean(observedflux[mask])
    
                # add S/N
                def SNR_appr(lam,debug=0):
                    x = [1,7,10,15,17,18,22,26,29]
                    y = [10,10,45,96,88,84,53,30,26]
                    f = interp1d(x,y,fill_value='extrapolate')
                    if debug:
                        fig, ax = plt.subplots()
                        ax.plot(x,y,'o')
                        ax.plot(np.arange(30),f(np.arange(30)))
                        plt.show()
                    return f(lam)
                SNR_appr(5)
    
                Flux_err = np.random.normal(loc=0.0, scale=1.0, size=len(w_new))
                #Flux_binned = Flux_binned * (1 + 1 / SNRmiri * Flux_err)
                Flux_binned = Flux_binned * (1 + 1 / SNR_appr(w_new/1e4) * Flux_err)
            mode = 'CO'
            if mode == 'H2':
                fig3, ax3 = plt.subplots(2, 2, figsize=(9, 4))
                fontsize = 8
                for axs in ax3[0, :]:
                    axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                    axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                    axs.set_ylim([-0.1, 1.1])
                    axs.axhline(y=0, ls=':', lw=1, color='black')
                for axs in ax3[1, :]:
                    axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                    axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                    axs.set_ylim([-0.1, 1.1])
    
                xlow, xup = 2., 3.5
                ylow, yup = 0.9, 1.4
                ax3[0, 0].set_xlim(xlow, xup)
                ax3[0, 0].set_ylim(ylow, yup)
                ax3[0, 0].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.84,
                               'H$_2$ $\\nu = 1 \\to0$: rovibrational lines $(2.3\\mu)$',
                               fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N ='+str(SNRmiri), fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.27, '$A_V$=1.1', fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated)', fontsize=fontsize)
                # ax3[0, 0].set_ylabel('Normalized flux, (${\\AA}$)', fontsize=fontsize)
                #ax3[0, 0].plot([3.1, 3.4], [1.1, 1.1], '|', color='red')
    
                for tmpline in lines:
                    if 'H2(1-0)' in tmpline.name and 'H2(1-0)Q(2)' not in tmpline.name and 'H2(1-0)Q(3)' not in tmpline.name:
                        if (tmpline.wavelength[0] / micAA+0.02 <xup)*(tmpline.wavelength[0] / micAA+0.02>xlow):
                            ax3[0,0].text(tmpline.wavelength[0] / micAA+0.02, 1.2, (tmpline.name).split('1-0)')[-1], rotation=90,
                                     fontsize=fontsize - 1, color='red')
    
                xlow, xup = 5.2, 7.1
                ylow, yup = 0.9, 1.1
                ax3[0, 1].set_xlim(xlow, xup)
                ax3[0, 1].set_ylim(ylow, yup)
                ax3[0, 1].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.2,
                               'H$_2$ rotational lines',
                               fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N ='+str(SNRmiri), fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.27, '$A_V$=1.1', fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated)', fontsize=fontsize)
                # ax3[0, 0].set_ylabel('Normalized flux, (${\\AA}$)', fontsize=fontsize)
                # ax3[0, 0].plot([3.1, 3.4], [1.1, 1.1], '|', color='red')
    
                for tmpline in lines:
                    if 'H2S' in tmpline.name:
                        if (tmpline.wavelength[0] / micAA+0.02 <xup)*(tmpline.wavelength[0] / micAA+0.02>xlow):
                            ax3[0, 1].text(tmpline.wavelength[0] / micAA + 0.02, 1.05, tmpline.name,
                                           rotation=90,
                                           fontsize=fontsize - 1, color='red')
    
                xlow, xup = 2.7, 3.7
                ylow, yup = 0.9, 1.1
                ax3[1, 0].set_xlim(xlow, xup)
                ax3[1, 0].set_ylim(ylow, yup)
                ax3[1, 0].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.85,
                               'H$_2$O $(3.1{\\mu}m)$ and C-H $(3.4{\\mu}m)$',
                               fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N ='+str(SNRmiri), fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.27, '$A_V$=1.1', fontsize=fontsize)
                # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated)', fontsize=fontsize)
                # ax3[0, 0].set_ylabel('Normalized flux, (${\\AA}$)', fontsize=fontsize)
                ax3[1, 0].plot([3.1, 3.4], [1.05, 1.05], '|', color='red')
    
    
    
                xlow, xup = 5.6, 8.5
                ylow, yup = 0.9, 1.2
                ax3[1, 1].set_xlim(xlow, xup)
                ax3[1, 1].set_ylim(ylow, yup)
                ax3[1, 1].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.85, 'PAH $(6.2{\\mu}m$ and $7.6{\\mu}m)$',
                               fontsize=fontsize)
                # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.75, 'S/N = 50', fontsize=fontsize)
                # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.65, '$A_V$=1.1', fontsize=fontsize)
                # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
                lamij = [5.25, 5.70, 6.22, 6.69,7.417, 7.598, 7.85, 8.33, 8.61, 11.23, 11.30, 11.99, 12.61, 13.6, 14.19]
                for l in lamij:
                    ax3[1, 1].plot(l, 0.95, '|', color='red')
    
    
    
                for secax in ax3[0, :]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.xaxis.set_major_locator(MultipleLocator(0.5))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.yaxis.set_major_locator(MultipleLocator(0.1))
    
                ax3[0, 1].yaxis.set_minor_locator(AutoMinorLocator(4))
                ax3[0, 1].yaxis.set_major_locator(MultipleLocator(0.1))
    
                for secax in ax3[1, :]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.xaxis.set_major_locator(MultipleLocator(0.5))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.yaxis.set_major_locator(MultipleLocator(0.1))
                ax3[1, 1].yaxis.set_minor_locator(AutoMinorLocator(4))
                ax3[1, 1].yaxis.set_major_locator(MultipleLocator(0.1))
    
                ax3[1, 1].set_xlabel('Rest-frame wavelength, (${\\mu}m$)', fontsize=fontsize)
                ax3[1, 0].set_xlabel('Rest-frame wavelength, (${\\mu}m$)', fontsize=fontsize)
                ax3[0, 0].set_ylabel('Normalized flux', fontsize=fontsize)
                ax3[1, 0].set_ylabel('Normalized flux', fontsize=fontsize)
    
    
                ax3[0, 0].text(2.1, 1.45, 'Quasar J0006+1215 MIRI MRS (Simulated spectrum), $A_{\\rm V}$=3.5, $N({\\rm H_2})=10^{23.5}\\,{\\rm cm^{-2}}$ $z_{\\rm abs}=2.31$',
                               fontsize=fontsize)
            elif mode == 'CO':
                fig3, ax3 = plt.subplots(figsize=(9, 3))
                fontsize = 8
                for axs in [ax3]:
                    axs.plot(wavel / 1e4 / (1 + redshift), observedflux, color='red', alpha=0.7)
                    axs.step(w_new / 1e4 / (1 + redshift), Flux_binned, color='black', where='mid', zorder=-10)
                    axs.set_ylim([-0.1, 1.1])
                    axs.axhline(y=0, ls=':', lw=1, color='black')
    
                for tmpline in lines:
                    if 'CO' in tmpline.name:
                        ax3.text(tmpline.wavelength[0]/micAA, 1.1,(tmpline.name).split('0-1)')[-1],rotation=90,fontsize=fontsize-1,color='red')
    
    
                xlow, xup = 4.63, 4.9
                ylow, yup = -0.1, 1.4
                ax3.set_xlim(xlow, xup)
                ax3.set_ylim(ylow, yup)
                ax3.text(xlow + (xup - xlow) * 0.03, ylow + (yup - ylow) * 0.93, 'CO $\\nu\\to0$: J=0-10 $(4.6\\mu)$ rovibrational lines', fontsize=fontsize)
                # ax31, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.25, 'S/N =' + str(SNRmiri), fontsize=fontsize)
                # ax3[1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '$A_V$=1.1', fontsize=fontsize)
                # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.1, '(simulated)', fontsize=fontsize)
                lamij = [5.25, 5.70, 6.22, 6.69]
                for l in lamij:
                    ax3.plot(l, 0.95, '|', color='red')
    
    
                for secax in [ax3]:
                    secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                      top='True')
                    secax.tick_params(which='major', length=5)
                    secax.tick_params(which='minor', length=3)
                    secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.xaxis.set_major_locator(MultipleLocator(0.1))
                    secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    secax.yaxis.set_major_locator(MultipleLocator(0.5))
    
                ax3.set_xlabel('Rest-frame wavelength, (${\\mu}m$)', fontsize=fontsize)
                ax3.set_ylabel('Normalized flux', fontsize=fontsize)
                # ax3[0].text(2.7, 1.22, 'Quasar J 1211+0833 MIRI MRS (Simulated spectrum), S/N = '+str(SNRmiri)+', $A_{\\rm V}=0.8$, $z_{\\rm abs}=2.117$', fontsize=fontsize)
                ax3.text(4.63, 1.42,'Quasar J0006+1215 MIRI MRS (Simulated spectrum), $A_{\\rm V}$=3.5, $N({\\rm CO})=10^{19.5}\\,{\\rm cm^{-2}}$ $z_{\\rm abs}=2.31$', fontsize=fontsize)
                #ax3.text(5.8,0.7,'MIRI MRS', fontsize=fontsize)
               #ax3.text(5.8,0.63,'(simulated)', fontsize=fontsize)
        elif case == 'J1830':
            #with open('model_' + case + '.pkl', 'rb') as f:
            #    wavel, observedflux = pickle.load(f)
    
            # load spectrum
            if 1:
                redshift =  0.886
                hdu = fits.open('/home/slava/science/research/kulkarni/JWST-DLAs/ID2441/PKS1830_B.fits')
                prihdr = hdu[1].data
                wavel = hdu[1].data['WAVELENGTH']
                flux = hdu[1].data['FLUX']
                hdu.close()
                hdu = fits.open('/home/slava/science/research/kulkarni/JWST-DLAs/ID2441/PKS1830_B_normilized_tot.fits')
                prihdr = hdu[1].data
                wavel_norm = hdu[1].data['WAVELENGTH']
                flux_norm = hdu[1].data['FLUX']
    
            fig3, ax3 = plt.subplots(2, 3, figsize=(9, 4))
            fontsize = 8
            for axs in [ax3[0, 0],ax3[0,2]]:
                axs.step(wavel/ (1 + redshift), flux, color='black', where='mid', zorder=-10)
                axs.set_ylim([-0.1, 1.1])
                axs.axhline(y=0, ls=':', lw=1, color='black')
            for axs in [ax3[0, 1],ax3[1,0],ax3[1,1],ax3[1,2]]:
                axs.step(wavel_norm/ (1 + redshift), flux_norm, color='black', where='mid', zorder=-10)
                axs.set_ylim([-0.1, 1.1])
                axs.axhline(y=0, ls=':', lw=1, color='black')
            #for axs in ax3[1, :]:
            #    axs.step(wavel/ (1 + redshift), flux, color='black', where='mid', zorder=-10)
            #    axs.set_ylim([-0.1, 1.1])
    
            xlow, xup = 2.6, 4.1
            ylow, yup = -0.1, 1.4
            ax3[0, 0].set_xlim(xlow, xup)
            ax3[0, 0].set_ylim(ylow, yup)
            ax3[0, 0].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1,
                           'H$_2$O $(3.1{\\mu}m)$ and C-H $(3.4{\\mu}m)$',
                           fontsize=fontsize)
            # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.35, 'S/N ='+str(SNRmiri), fontsize=fontsize)
            # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.27, '$A_V$=1.1', fontsize=fontsize)
            # ax3[0, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.2, '(simulated)', fontsize=fontsize)
            # ax3[0, 0].set_ylabel('Normalized flux, (${\\AA}$)', fontsize=fontsize)
            ax3[0, 0].plot([3.1, 3.4], [1.15, 1.15], '|', color='red')
    
            xlow, xup = 4.6, 4.8
            ax3[0, 1].set_xlim(xlow, xup)
            ax3[0, 1].set_ylim(ylow, yup)
            ax3[0, 1].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'CO $\\nu\\to0$: J=0-10 $(4.6\\mu)$',
                           fontsize=fontsize)
            #ax3[0, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup - ylow) * 0.27, '$N(CO)=10^{18}$', fontsize=fontsize)
            for tmpline in lines:
                if 'CO' in tmpline.name:
                    ax3[0, 1].plot(tmpline.wavelength[0] / 1e4, 1.15, '|', color='red')
    
            xlow, xup = 7.5, 14
            ax3[0, 2].set_xlim(xlow, xup)
            ax3[0, 2].set_ylim(-0.1, 1.3)
            ax3[0, 2].text(xlow + (xup - xlow) * 0.55, ylow + (yup - ylow) * 0.1, 'Si-O $(9.7{\\mu}m)$', fontsize=fontsize)
            # ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.5, 'S/N =' + str(SNRmiri), fontsize=fontsize)
            #ax3[0, 2].text(xlow + (xup - xlow) * 0.6, ylow + (yup - ylow) * 0.27, '$\\tau_{9.7}=0.49$', fontsize=fontsize)
            # ax3[0, 2].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            ax3[0, 2].plot(9.7, 1.15, '|', color='red')
    
            xlow, xup = 4.8, 6.5
            ylow, yup = 0.5, 1.5
            ax3[1, 0].set_xlim(xlow, xup)
            ax3[1, 0].set_ylim(ylow, yup)
            ax3[1, 0].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'PAH $(6.2{\\mu}m)$', fontsize=fontsize)
            # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.25, 'S/N =' + str(SNRmiri), fontsize=fontsize)
            # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.17, '$A_V$=1.1', fontsize=fontsize)
            # ax3[1, 0].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.1, '(simulated)', fontsize=fontsize)
            lamij = [5.25, 5.70, 6.22, 6.69]
            for l in lamij:
                ax3[1, 0].plot(l, 0.8, '|', color='red')
    
            xlow, xup = 7, 9
            ax3[1, 1].set_xlim(xlow, xup)
            ax3[1, 1].set_ylim(ylow, yup)
            ax3[1, 1].text(xlow + (xup - xlow) * 0.05, ylow + (yup - ylow) * 0.1, 'PAH $(7.6{\\mu}m$ and $8.6{\\mu}m)$',
                           fontsize=fontsize)
            # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.75, 'S/N = 50', fontsize=fontsize)
            # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.65, '$A_V$=1.1', fontsize=fontsize)
            # ax3[1, 1].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            lamij = [7.417, 7.598, 7.85, 8.33, 8.61, 11.23, 11.30, 11.99, 12.61, 13.6, 14.19]
            for l in lamij:
                ax3[1, 1].plot(l, 0.8, '|', color='red')
    
            xlow, xup = 10.5, 12
            ax3[1, 2].set_xlim(xlow, xup)
            ax3[1, 2].set_ylim(ylow, yup)
            ax3[1, 2].text(xlow + (xup - xlow) * 0.55, ylow + (yup - ylow) * 0.1, 'PAH $(11.3\\mu)$', fontsize=fontsize)
            # ax3[1, 2].text(xlow + (xup - xlow) * 0.6, 0.50, 'S/N = 50', fontsize=fontsize)
            # ax3[1, 2].text(xlow + (xup - xlow) * 0.6, ylow + (yup-ylow)*0.17, '$A_V$=1.1', fontsize=fontsize)
            # ax3[1, 2].text(xlow + (xup - xlow) * 0.6, 0.25, '(simulated)', fontsize=fontsize)
            for l in lamij:
                ax3[1, 2].plot(l, 0.8, '|', color='red')
    
            for secax in ax3[0, :]:
                secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                  top='True')
                secax.tick_params(which='major', length=5)
                secax.tick_params(which='minor', length=3)
                secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                secax.xaxis.set_major_locator(MultipleLocator(0.5))
                secax.yaxis.set_minor_locator(AutoMinorLocator(5))
                secax.yaxis.set_major_locator(MultipleLocator(0.5))
    
            ax3[0, 1].xaxis.set_minor_locator(AutoMinorLocator(4))
            ax3[0, 1].xaxis.set_major_locator(MultipleLocator(0.1))
            ax3[0, 2].xaxis.set_minor_locator(AutoMinorLocator(4))
            ax3[0, 2].xaxis.set_major_locator(MultipleLocator(1))
    
            for secax in ax3[1, :]:
                secax.tick_params(which='both', width=1, direction='in', labelsize=fontsize, right='True',
                                  top='True')
                secax.tick_params(which='major', length=5)
                secax.tick_params(which='minor', length=3)
                secax.xaxis.set_minor_locator(AutoMinorLocator(5))
                secax.xaxis.set_major_locator(MultipleLocator(0.5))
                secax.yaxis.set_minor_locator(AutoMinorLocator(4))
                secax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax3[1, 1].set_xlabel('Rest-frame wavelength, (${\\mu}m$)', fontsize=fontsize)
            ax3[0, 0].text(2.3, -1.5, 'Flux normalized to the quasar continuum', fontsize=fontsize, rotation=90)
            # ax3[0, 0].text(2.7, 1.25, 'Quasar J 0218+357 MIRI MRS (Simulated spectrum), S/N = ' + str(
            #    SNRmiri) + ', $A_{\\rm V}$=5.8, $z_{\\rm abs}=0.685$', fontsize=fontsize)
            ax3[0, 0].text(2.7, 1.5,
                           'Molecular cloud, $z_{\\rm abs}=0.9$ (archival data)',
                           fontsize=fontsize)
    
        if 1:
            #str = 'J'+case+'Miri_Av='+str(Av)+'_SNR='+str(SNRmiri)+'.pdf'
            str = 'J' + case + 'Miri_Av=' + str(Av) + '_H2' + '.pdf'
            f_name = 'PKS1830-211_6_panels.pdf'
            fig3.savefig(f_name, bbox_inches='tight')



