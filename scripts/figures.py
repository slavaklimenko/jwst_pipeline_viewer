import sys
sys.path.append('home/slava/science/codes/python/')
sys.path.append('/home/toksovogo/science/codes/python')
sys.path.append('/home/slava/science/codes/python/spectro/')
from spectro.profiles import voigt, tau,convolveflux
import numpy as np
from scipy.interpolate import Rbf
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d,interp1d
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from os import listdir
from os.path import isfile, join
from scipy import interpolate,integrate, optimize
import matplotlib.gridspec as gridspec
from scipy.stats import moment

from scipy.interpolate import interp2d, RectBivariateSpline, Rbf
import scipy
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
if 1:
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['text.latex.unicode'] = True
    #matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['axes.titlesize'] = 10


labelsize = 16
case = 'monique_conf'
if case == 'preliminar':
    fig, ax = plt.subplots(1,2,figsize=(12, 6))
    filepath = '/home/slava/science/codes/JWST/tmp-data/'
    spec1 = np.loadtxt(filepath+'channel1_spec.specrtro')
    spec2 = np.loadtxt(filepath + 'channel2_spec.specrtro')
    spec3 = np.loadtxt(filepath + 'channel3_spec.specrtro')
    spec4 = np.loadtxt(filepath + 'channel4_spec.specrtro')
    spec_spitzer = np.loadtxt('/home/slava/science/data/SPITZER/AO0235/' + 'cassis_yaaar_spcfw_15121152t-copy-red.dat')
    spec_spitzer[:,0] /= 1e4
    np.savetxt('/home/slava/science/data/SPITZER/AO0235/' + 'cassis_yaaar_spcfw_15121152t-copy-red_norm.dat',spec_spitzer)

    for el in [spec1[:,0],spec2[:,0],spec3[:,0],spec4[:,0],spec_spitzer[:,0]]:
        el*=1/1e4/1.52

    coeffs = np.array([1,14/8,22/10,28/5])/14000
    for k,el in enumerate([spec1[:, 1], spec2[:, 1], spec3[:, 1], spec4[:, 1]]):
        el *=coeffs[k]
    for k,el in enumerate([spec1[:, 2], spec2[:, 2], spec3[:, 2], spec4[:, 2]]):
        el *=coeffs[k]
    ax[0].plot(spec1[:,0],spec1[:,1],label='MIRI Channel1')
    ax[0].plot(spec2[:, 0], spec2[:, 1],label='MIRI Channel2')
    ax[0].plot(spec3[:, 0], spec3[:, 1],label='MIRI Channel3')
    ax[0].plot(spec4[:, 0], spec4[:, 1],label='MIRI Channel4')

    x=np.linspace(0.1,30,100)
    corr = 1/14000
    y1 = 9100*(x/5)**1.11*corr
    cont1 = interp1d(np.array(x/1.52),y1,kind='linear',fill_value='extrapolate')
    y2 = 11000 * (x / 5) ** 0.9*corr
    cont2 = interp1d(np.array(x/1.52),y2,kind='linear',fill_value='extrapolate')
    y3 = np.zeros_like(x)
    y3[x<8.6*1.52] = y1[x<8.6*1.52]
    y3[x >= 8.6*1.52] = y1[x >= 8.6*1.52]
    cont3 = interp1d(np.array(x/1.52),y3,kind='linear',fill_value='extrapolate')
    #ax[0].plot(x/(1.52),9000*(x/5)**1.11*corr)
    #ax[0].plot(x / (1.52), 9000 * (x / 5) ** 1.05*corr)
    ax[0].plot(x,cont1(x),ls='-',color='blue',lw=1.5)
    ax[0].plot(x, cont3(x),ls='--',color='blue',lw=1.5)

    spec_spitzer[:, 1] /=2.5e-2
    spec_spitzer[:, 2] /= 2.5e-2
    ax[0].plot(spec_spitzer[:,0],spec_spitzer[:,1],color='black',lw=3,label='Spitzer/IRS')

    ax[1].errorbar(x=spec_spitzer[:, 0], y=spec_spitzer[:, 1]/cont1(spec_spitzer[:, 0]),yerr=spec_spitzer[:, 2]/cont1(spec_spitzer[:, 0]), color='black', lw=1.5,zorder=10,ds='steps-mid',label='Spitzer/IRS')
    ax[1].errorbar(x=spec1[:, 0], y=spec1[:, 1] / cont3(spec1[:, 0]),yerr=spec1[:, 2] / cont3(spec1[:, 0]),lw=1,ds='steps-mid')
    ax[1].errorbar(x=spec2[:, 0], y=spec2[:, 1] / cont3(spec2[:, 0]),yerr=spec2[:, 2] / cont3(spec2[:, 0]),lw=1,ds='steps-mid')
    ax[1].errorbar(x=spec3[:, 0], y=spec3[:, 1] / cont3(spec3[:, 0]),yerr=spec3[:, 2] / cont3(spec3[:, 0]),lw=1,ds='steps-mid',label='MIRI Channel3')
    ax[1].errorbar(x=spec4[:, 0], y= spec4[:, 1] / cont3(spec4[:, 0]),yerr= spec4[:, 2] / cont3(spec4[:, 0]),lw=1,ds='steps-mid')

    if 1:
        def gauss(x, s):
            return 1 / np.sqrt(2 * np.pi) / s * np.exp(-.5 * (x / s) ** 2)

        if 0:
            t9_7peak = 0.08
            dust_w, dust_tau, dust_b = 9.7, t9_7peak,  0.5
            w_dust = dust_w
            wavel = np.array(x)
            Tau = np.zeros_like(x)
            for k, lam in enumerate(wavel):
                b = 3e5 * dust_b / dust_w
                x = (lam / w_dust  - 1) * 3e5 / b
                if np.abs(x) < 3:
                    tau = np.log(1 + gauss(x, 1) * dust_tau / gauss(0, 1))
                    # print(gauss(0, 1))
                    Tau[k] += tau
            Absorptionflux = np.exp(-Tau)

            dust_w, dust_tau, dust_b = 11, t9_7peak, 1.5
            w_dust = dust_w
            #wavel = np.array(x)
            Tau = np.zeros_like(wavel)
            for k, lam in enumerate(wavel):
                b = 3e5 * dust_b / dust_w
                x = (lam / w_dust - 1) * 3e5 / b
                if np.abs(x) < 3:
                    tau = np.log(1 + gauss(x, 1) * dust_tau / gauss(0, 1))
                    # print(gauss(0, 1))
                    Tau[k] += tau
            Absorptionflux *= np.exp(-Tau)
            ax[1].plot(wavel,Absorptionflux,color='red',zorder=20,ls='--',label='$\\tau(9.7\\mu m)=0.08$')
    if 1:
        ax[0].legend(fontsize=labelsize,loc='upper left',frameon=False)
        ax[0].text(12,0.65,'AO0235+164',fontsize=labelsize+2)
        ax[0].text(12, 0.35, 'z$_{abs}$=0.524',fontsize=labelsize+2)
        ax[0].set_xlabel('Rest Wavelength [$\\mu$m]', fontsize=labelsize)
        ax[0].set_ylabel('Flux, a.u.', fontsize=labelsize)
        ax[0].set_xlim(3, 19)
        ax[0].set_ylim(0, 4.5)
        ax[1].set_xlabel('Rest Wavelength [$\\mu$m]', fontsize=labelsize)
        ax[1].set_ylabel('Normalized Flux', fontsize=labelsize)
        ax[1].set_xlim(6.5, 16)
        ax[1].set_ylim(0.3, 1.5)
        #ax[0].text(0.9,1.6,'$T_{\\rm 01} \\propto n_{\\rm H} ^{-0.16}$',fontsize=labelsize,color='black')
        #ax[1].text(0.9,5,'$P_{\\rm th} \\propto n_{\\rm H} ^{0.84}$',fontsize=labelsize, color='black')

        for col in ax[:]:
            col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True',
                            top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=3)
        ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].xaxis.set_major_locator(MultipleLocator(5))
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].yaxis.set_major_locator(MultipleLocator(1))
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1].xaxis.set_major_locator(MultipleLocator(2))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1].yaxis.set_major_locator(MultipleLocator(0.2))

        ax[1].axhline(1,ls='--',color='blue',lw=1.5)
        ax[1].plot(9.7,1.3,'|',color='black',markersize=20,markeredgewidth=2)
        ax[1].text(8.5, 1.37, 'Si-O $(9.7{\\mu}m)$', fontsize=labelsize+2,color='black')
        #ax[1].legend(fontsize=labelsize - 2, loc='lower left')


    if 1:
        handles, labels = ax[1].get_legend_handles_labels()
        # ax.legend(loc ='lower right',fontsize=labelsize)
        print(labels)
        order2 = [1, 0]
        ax[1].legend([handles[idx] for idx in order2], [labels[idx] for idx in order2],
                  loc='lower left', frameon=False, fontsize=labelsize , markerscale=0.8)
if case == 'smoothed':
    filepath = '/home/slava/science/codes/JWST/tmp-data/'
    spec1 = np.loadtxt(filepath + 'channel1_spec.specrtro')
    spec2 = np.loadtxt(filepath + 'channel2_spec.specrtro')
    spec3 = np.loadtxt(filepath + 'channel3_spec.specrtro')
    spec4 = np.loadtxt(filepath + 'channel4_spec.specrtro')
    spec_spitzer = np.loadtxt('/home/slava/science/data/SPITZER/AO0235/' + 'cassis_yaaar_spcfw_15121152t-copy-red.dat')

    for el in [spec1[:, 0], spec2[:, 0], spec3[:, 0], spec4[:, 0], spec_spitzer[:, 0]]:
        el *= 1 / 1e4 / 1.52

    coeffs = np.array([1, 14 / 8, 22 / 10, 28 / 5]) / 14000
    for k, el in enumerate([spec1[:, 1], spec2[:, 1], spec3[:, 1], spec4[:, 1]]):
        el *= coeffs[k]
    for k, el in enumerate([spec1[:, 2], spec2[:, 2], spec3[:, 2], spec4[:, 2]]):
        el *= coeffs[k]
    spec_jwst = np.append(spec1,spec2,axis=0)
    spec_jwst = np.append(spec_jwst,spec3,axis=0)
    spec_jwst = np.append(spec_jwst,spec4,axis=0)
    spec = np.zeros((spec_jwst.shape[0],3))
    spec[:,0] = spec_jwst[:,0]
    spec[:, 1] = spec_jwst[:, 1]
    spec[:, 2] = spec_jwst[:, 2]

    spec_spitzer[:, 1] /= 2.5e-2
    spec_spitzer[:, 2] /= 2.5e-2

    # convolve with filter
    if 1:
        spec_jwst = spec[spec[:,0].argsort()]
        resolution = 100
        convolved_to_IRS_flux = convolveflux(spec_jwst[:,0], spec_jwst[:,1], res=resolution, kind='direct')
        convolved_to_JWST_flux = convolveflux(spec_jwst[:,0], spec_jwst[:,1], res=1000, kind='direct')

    # derive continuum
    if 1:
        x = np.linspace(0.1, 30, 100)
        corr = 1 / 14000
        y1 = 9100 * (x / 5) ** 1.11 * corr
        cont1 = interp1d(np.array(x / 1.52), y1, kind='linear', fill_value='extrapolate')
        y2 = 11000 * (x / 5) ** 0.9 * corr
        cont2 = interp1d(np.array(x / 1.52), y2, kind='linear', fill_value='extrapolate')
        y3 = np.zeros_like(x)
        y3[x < 8.6 * 1.52] = y1[x < 8.6 * 1.52]
        y3[x >= 8.6 * 1.52] = y2[x >= 8.6 * 1.52]
        cont3 = interp1d(np.array(x / 1.52), y3, kind='linear', fill_value='extrapolate')

    # rebinning spectrum
    if 1:
        y,R = convolved_to_IRS_flux,100
        model = interp1d(spec_jwst[:,0], y,fill_value='extrapolate')
        err_model = interp1d(spec_jwst[:,0], spec_jwst[:,2],fill_value='extrapolate')
        w_new = [3]
        rebinned_flux  = [model(3)]
        x = 3
        while x<20:
            x+=x/R/3
            w_new.append(x)
            rebinned_flux.append(model(x))

    if 0:
        plt.subplots()
        plt.plot(spec_jwst[:,0],spec_jwst[:,1])
        plt.plot(spec_jwst[:,0], convolved_to_IRS_flux)
        plt.plot(spec_spitzer[:, 0], spec_spitzer[:, 1],ds='steps-mid')
        plt.show()
    #plt.plot(spec[:, 0], spec[:, 1])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    x = np.linspace(0.1, 30, 100)
    ax[0].plot(x, cont1(x), ls='-', color='blue', lw=1.5)
    ax[0].plot(x, cont3(x), ls='--', color='blue', lw=1.5)

    ax[0].plot(spec_spitzer[:, 0], spec_spitzer[:, 1], color='black', lw=3, label='Spitzer/IRS')
    ax[1].errorbar(x=spec_spitzer[:, 0], y=spec_spitzer[:, 1] / cont1(spec_spitzer[:, 0]),
                   yerr=spec_spitzer[:, 2] / cont1(spec_spitzer[:, 0]), color='black', lw=1.5, zorder=10,
                   ds='steps-mid', label='Spitzer/IRS')

    y,ylabel =convolved_to_IRS_flux,'MIRI (R=100)'
    ax[0].plot(spec_jwst[:,0], y, color='tab:green', label=ylabel)
    #ax[1].errorbar(x=spec_jwst[:,0], y=y / cont3(spec_jwst[:,0]),
    #               yerr=spec_jwst[:,2] / cont1(spec_jwst[:,0]), color='tab:green', lw=1, zorder=10,
    #               ds='steps-mid', label=ylabel,markeredgewidth=0.5,elinewidth=0.5)
    ax[1].errorbar(x=w_new,y=rebinned_flux/ cont3(w_new),yerr=err_model(w_new)/np.sqrt(3000/R),zorder=-1, color='tab:green', lw=1,
                   ds='steps-mid', label=ylabel,markeredgewidth=0.5,elinewidth=1)

    if 1:
        ax[0].legend(fontsize=labelsize, loc='upper left', frameon=False)
        ax[0].text(12, 0.65, 'AO0235+164', fontsize=labelsize + 2)
        ax[0].text(12, 0.35, 'z$_{abs}$=0.524', fontsize=labelsize + 2)
        ax[0].set_xlabel('Rest Wavelength [$\\mu$m]', fontsize=labelsize)
        ax[0].set_ylabel('Flux, a.u.', fontsize=labelsize)
        ax[0].set_xlim(5, 25)
        ax[0].set_ylim(0, 4.5)
        ax[1].set_xlabel('Rest Wavelength [$\\mu$m]', fontsize=labelsize)
        ax[1].set_ylabel('Normalized Flux', fontsize=labelsize)
        ax[1].set_xlim(6.5, 16)
        ax[1].set_ylim(0.3, 1.5)
        # ax[0].text(0.9,1.6,'$T_{\\rm 01} \\propto n_{\\rm H} ^{-0.16}$',fontsize=labelsize,color='black')
        # ax[1].text(0.9,5,'$P_{\\rm th} \\propto n_{\\rm H} ^{0.84}$',fontsize=labelsize, color='black')

        for col in ax[:]:
            col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True',
                            top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=3)
        ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].xaxis.set_major_locator(MultipleLocator(5))
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].yaxis.set_major_locator(MultipleLocator(1))
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1].xaxis.set_major_locator(MultipleLocator(2))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1].yaxis.set_major_locator(MultipleLocator(0.2))

        ax[1].axhline(1, ls='--', color='blue', lw=1.5)
        ax[1].plot(9.7, 1.3, '|', color='black', markersize=20, markeredgewidth=2)
        ax[1].text(8.5, 1.37, 'Si-O $(9.7{\\mu}m)$', fontsize=labelsize + 2, color='black')
        # ax[1].legend(fontsize=labelsize - 2, loc='lower left')

    if 1:
        handles, labels = ax[1].get_legend_handles_labels()
        # ax.legend(loc ='lower right',fontsize=labelsize)
        print(labels)
        order2 = [1, 0]
        ax[1].legend([handles[idx] for idx in order2], [labels[idx] for idx in order2],
                     loc='lower left', frameon=False, fontsize=labelsize, markerscale=0.8)
if case == 'ch34_mathing':
    fig, ax = plt.subplots(2,1,figsize=(12, 18))
    filepath = '/home/slava/science/codes/python/jwst/output/detector3/roi_spectra/'
    #spec1 = np.loadtxt(filepath+'sci_3SHORT(A)_ch3-short_s3d_(A)_sci.spec1d')
    #spec2 = np.loadtxt(filepath + 'sci_3MEDIUM(B)_ch3-medium_s3d_(A)_sci.spec1d')
    #spec3 = np.loadtxt(filepath + 'sci_3LONG(C)_ch3-long_s3d_(A)_sci.spec1d')
    #spec4 = np.loadtxt(filepath + 'sci_4SHORT(A)_ch4-short_s3d_(A)_sci.spec1d')
    #spec5 = np.loadtxt(filepath + 'sci_4MEDIUM(B)_ch4-medium_s3d_(A)_sci.spec1d')
    #spec6 = np.loadtxt(filepath + 'sci_4LONG(C)_ch4-long_s3d_(A)_sci.spec1d')
    spec_spitzer = np.loadtxt('/home/slava/science/data/SPITZER/AO0235/' + 'cassis_yaaar_spcfw_15121152t-copy-red.dat')

    spec_spitzer[:,0] /= 1e4
    spec_spitzer[:,1]=spec_spitzer[:,1]/2.550415899999999847e-02*8000
    np.savetxt('/home/slava/science/data/SPITZER/AO0235/' + 'cassis_yaaar_spcfw_15121152t-copy-red_norm.dat',spec_spitzer)

    z = 0.
    for el in [spec1[:,0],spec2[:,0],spec3[:,0],spec4[:,0],spec5[:,0],spec6[:,0]]:
        el*=1/(1+z)
    for el in [spec_spitzer[:,0]]:
        el*=1/1e4/(1+z)

    coeffs = np.array([1,425/342,521/485,635/307,670/484,670/484*770/566])
    for k,el in enumerate([spec1[:, 1], spec2[:, 1], spec3[:, 1], spec4[:, 1], spec5[:, 1], spec6[:, 1]]):
        el *=coeffs[k]
    for k,el in enumerate([spec1[:, 2], spec2[:, 2], spec3[:, 2], spec4[:, 2], spec5[:, 2], spec6[:, 2]]):
        el *=coeffs[k]
    ax[0].plot(spec1[:,0],spec1[:,1],label='MIRI 3A')
    ax[0].plot(spec2[:, 0], spec2[:, 1],label='MIRI 3B')
    ax[0].plot(spec3[:, 0], spec3[:, 1],label='MIRI 3C')
    ax[0].plot(spec4[:, 0], spec4[:, 1],label='MIRI 4A')
    ax[0].plot(spec5[:, 0], spec5[:, 1], label='MIRI 4B')
    ax[0].plot(spec6[:, 0], spec6[:, 1], label='MIRI 4C')

    x=np.linspace(0.1,30,100)
    corr =1
    y1 = 217/254*100*(x/5*(1+z))**1.11*corr
    cont1 = interp1d(np.array(x/1.52),y1,kind='linear',fill_value='extrapolate')
    y2 = 180/254*100 * (x / 5*(1+z)) ** 1.3*corr
    cont2 = interp1d(np.array(x/1.52),y2,kind='linear',fill_value='extrapolate')

    ax[0].plot(x,cont1(x),ls='-',color='blue',lw=1.5)
    ax[0].plot(x, cont2(x),ls='--',color='blue',lw=1.5)

    #spec_spitzer[:, 1] /=2.5e-2
    #spec_spitzer[:, 2] /= 2.5e-2
    spitz_coeff = 1e4*352/414
    ax[0].plot(spec_spitzer[:,0],spec_spitzer[:,1]*spitz_coeff,color='black',lw=3,label='Spitzer/IRS')

    ax[1].errorbar(x=spec_spitzer[:, 0], y=spec_spitzer[:, 1]*spitz_coeff/cont1(spec_spitzer[:, 0]),yerr=spec_spitzer[:, 2]*spitz_coeff/cont1(spec_spitzer[:, 0]), color='black', lw=1.5,zorder=10,ds='steps-mid',label='Spitzer/IRS')
    ax[1].errorbar(x=spec1[:, 0], y=spec1[:, 1] / cont2(spec1[:, 0]),yerr=spec1[:, 2] / cont2(spec1[:, 0]),lw=1,ds='steps-mid')
    ax[1].errorbar(x=spec2[:, 0], y=spec2[:, 1] / cont2(spec2[:, 0]),yerr=spec2[:, 2] / cont2(spec2[:, 0]),lw=1,ds='steps-mid')
    ax[1].errorbar(x=spec3[:, 0], y=spec3[:, 1] / cont2(spec3[:, 0]),yerr=spec3[:, 2] / cont2(spec3[:, 0]),lw=1,ds='steps-mid',label='MIRI Channel3')
    ax[1].errorbar(x=spec4[:, 0], y= spec4[:, 1] / cont2(spec4[:, 0]),yerr= spec4[:, 2] / cont2(spec4[:, 0]),lw=1,ds='steps-mid')
    ax[1].errorbar(x=spec5[:, 0], y=spec5[:, 1] / cont2(spec5[:, 0]), yerr=spec5[:, 2] / cont2(spec5[:, 0]), lw=1,
                   ds='steps-mid', label='MIRI Channel3')
    ax[1].errorbar(x=spec6[:, 0], y=spec6[:, 1] / cont2(spec6[:, 0]), yerr=spec6[:, 2] / cont2(spec6[:, 0]), lw=1,
                   ds='steps-mid')

    if 1:
        def gauss(x, s):
            return 1 / np.sqrt(2 * np.pi) / s * np.exp(-.5 * (x / s) ** 2)

        if 0:
            t9_7peak = 0.08
            dust_w, dust_tau, dust_b = 9.7, t9_7peak,  0.5
            w_dust = dust_w
            wavel = np.array(x)
            Tau = np.zeros_like(x)
            for k, lam in enumerate(wavel):
                b = 3e5 * dust_b / dust_w
                x = (lam / w_dust  - 1) * 3e5 / b
                if np.abs(x) < 3:
                    tau = np.log(1 + gauss(x, 1) * dust_tau / gauss(0, 1))
                    # print(gauss(0, 1))
                    Tau[k] += tau
            Absorptionflux = np.exp(-Tau)

            dust_w, dust_tau, dust_b = 11, t9_7peak, 1.5
            w_dust = dust_w
            #wavel = np.array(x)
            Tau = np.zeros_like(wavel)
            for k, lam in enumerate(wavel):
                b = 3e5 * dust_b / dust_w
                x = (lam / w_dust - 1) * 3e5 / b
                if np.abs(x) < 3:
                    tau = np.log(1 + gauss(x, 1) * dust_tau / gauss(0, 1))
                    # print(gauss(0, 1))
                    Tau[k] += tau
            Absorptionflux *= np.exp(-Tau)
            ax[1].plot(wavel,Absorptionflux,color='red',zorder=20,ls='--',label='$\\tau(9.7\\mu m)=0.08$')
    if 1:
        ax[0].legend(fontsize=labelsize,loc='upper left',frameon=False)
        #ax[0].text(12,0.65,'AO0235+164',fontsize=labelsize+2)
        #ax[0].text(12, 0.35, 'z$_{abs}$=0.524',fontsize=labelsize+2)
        ax[0].set_xlabel('Rest Wavelength [$\\mu$m]', fontsize=labelsize)
        ax[0].set_ylabel('Flux, a.u.', fontsize=labelsize)
        ax[0].set_xlim(5, 20)
        ax[0].set_ylim(0, 1600)
        ax[1].set_xlabel('Rest Wavelength [$\\mu$m]', fontsize=labelsize)
        ax[1].set_ylabel('Normalized Flux', fontsize=labelsize)
        ax[1].set_xlim(5, 20)
        ax[1].set_ylim(0.3, 1.5)
        #ax[0].text(0.9,1.6,'$T_{\\rm 01} \\propto n_{\\rm H} ^{-0.16}$',fontsize=labelsize,color='black')
        #ax[1].text(0.9,5,'$P_{\\rm th} \\propto n_{\\rm H} ^{0.84}$',fontsize=labelsize, color='black')

        for col in ax[:]:
            col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True',
                            top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=3)
        ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].xaxis.set_major_locator(MultipleLocator(5))
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].yaxis.set_major_locator(MultipleLocator(500))
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(4))
        ax[1].xaxis.set_major_locator(MultipleLocator(2))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1].yaxis.set_major_locator(MultipleLocator(0.2))

        ax[1].axhline(1,ls='--',color='blue',lw=1.5)
        ax[1].plot(9.7,1.3,'|',color='black',markersize=20,markeredgewidth=2)
        ax[1].text(8.5, 1.37, 'Si-O $(9.7{\\mu}m)$', fontsize=labelsize+2,color='black')
        ax[1].legend(fontsize=labelsize - 2, loc='lower left')


    if 1:
        handles, labels = ax[1].get_legend_handles_labels()
        # ax.legend(loc ='lower right',fontsize=labelsize)
        print(labels)
        order2 = [1, 0]
        ax[1].legend([handles[idx] for idx in order2], [labels[idx] for idx in order2],
                  loc='lower left', frameon=False, fontsize=labelsize , markerscale=0.8)
if case == 'monique_conf':
    filepath = '/home/slava/science/codes/python/jwst/'
    fig, ax = plt.subplots(2,1,figsize=(9, 18))
    jwst = np.loadtxt(filepath +'Figures/data/jwst.spec')
    jwst_cont = np.loadtxt(filepath +'Figures/data/jwst_cont.dat')
    jwst_norm = np.loadtxt(filepath +'Figures/data/jwst_norm.dat')
    jwst_rebinned = np.loadtxt(filepath +'Figures/data/jwst_rebinned.dat')
    spitzer = np.loadtxt(filepath +'Figures/data/spitzer.spec')
    spitzer_cont = np.loadtxt(filepath +'Figures/data/spitzer_cont.dat')
    spitzer_norm = np.loadtxt(filepath +'Figures/data/spitzer_norm.dat')

    coef_z = 1/(1+0.524)

    from scipy import signal

    win = signal.windows.hann(20)
    filtered = signal.convolve(jwst[:, 1], win, mode='same') / sum(win)
    ax[0].plot(jwst[10:,0]*coef_z,filtered[10:],label='JWST/MIRI(2023)',color='tab:red')
    ax[0].plot(jwst_cont[:,0]*coef_z,jwst_cont[:,1],color='black',lw=1,ls='--')

    ax[0].plot(spitzer[:,0]*coef_z,spitzer[:,1],color='blue',lw=1,label='Spitzer/IRS(2006)')
    ax[0].plot(spitzer_cont[:,0]*coef_z,spitzer_cont[:,1],color='black',lw=1,ls='--')

    ax[1].errorbar(x=spitzer_norm[:, 0]*coef_z, y=spitzer_norm[:, 1],yerr=spitzer_norm[:, 2], color='blue', lw=1.5,
                   zorder=10,ds='steps-mid',label='Spitzer/IRS')
    #ax[1].errorbar(x=jwst_norm[:, 0]*coef_z, y=jwst_norm[:, 1],yerr=jwst_norm[:, 2] ,lw=1,ds='steps-mid',color='tab:red',label='JWST/MIRI')
    cont_interp = interp1d(jwst_cont[:,0]*coef_z,jwst_cont[:,1],fill_value='extrapolate')
    jwst_reb_norm = jwst_rebinned.copy()
    jwst_reb_norm[:,1] = jwst_rebinned[:,1]/cont_interp(jwst_rebinned[:,0]*coef_z)
    jwst_reb_norm[:, 2] = jwst_rebinned[:, 2] / np.abs(cont_interp(jwst_rebinned[:, 0]*coef_z))

    from scipy import signal
    win = signal.windows.hann(30)
    filtered = signal.convolve(jwst_norm[:, 1], win, mode='same') / sum(win)
    ax[1].errorbar(x=jwst_norm[:, 0] * coef_z, y=filtered, yerr=jwst_norm[:, 2], lw=1, ds='steps-mid',
                   color='tab:red', label='JWST/MIRI')
    #ax[1].errorbar(x=jwst_reb_norm[:, 0] * coef_z, y=jwst_reb_norm[:, 1], yerr=jwst_reb_norm[:, 2], lw=1, ds='steps-mid',
    #               color='tab:red', label='JWST/MIRI')

    if 1:
        def gauss(x, s):
            return 1 / np.sqrt(2 * np.pi) / s * np.exp(-.5 * (x / s) ** 2)

        if 1:
            t9_7peak = 0.05
            dust_w, dust_tau, dust_b = 9.7, t9_7peak,  0.7
            w_dust = dust_w
            x = np.linspace(0.1, 30, 100)
            wavel = np.array(x)
            Tau = np.zeros_like(x)
            for k, lam in enumerate(wavel):
                b = 3e5 * dust_b / dust_w
                x = (lam / w_dust  - 1) * 3e5 / b
                if np.abs(x) < 3:
                    tau = np.log(1 + gauss(x, 1) * dust_tau / gauss(0, 1))
                    # print(gauss(0, 1))
                    Tau[k] += tau
            Absorptionflux = np.exp(-Tau)

            dust_w, dust_tau, dust_b = 11, t9_7peak*0.4, 1.5
            w_dust = dust_w
            #wavel = np.array(x)
            Tau = np.zeros_like(wavel)
            for k, lam in enumerate(wavel):
                b = 3e5 * dust_b / dust_w
                x = (lam / w_dust - 1) * 3e5 / b
                if np.abs(x) < 3:
                    tau = np.log(1 + gauss(x, 1) * dust_tau / gauss(0, 1))
                    # print(gauss(0, 1))
                    Tau[k] += tau
            Absorptionflux *= np.exp(-Tau)
            ax[1].plot(wavel,Absorptionflux,color='black',zorder=20,ls='--',label='Fit to $\\tau(9.7\\mu m)=0.08$')
    if 1:
        ax[0].legend(fontsize=labelsize,loc='upper left',frameon=False)
        #ax[0].text(12,0.65,'AO0235+164',fontsize=labelsize+2)
        #ax[0].text(12, 0.35, 'z$_{abs}$=0.524',fontsize=labelsize+2)
        ax[0].set_xlabel('Rest Wavelength [$\\mu$m]', fontsize=labelsize)
        ax[0].set_ylabel('Flux, a.u.', fontsize=labelsize)
        ax[0].set_xlim(2, 18)
        ax[0].set_ylim(0, 3.5)
        ax[1].set_xlabel('Rest Wavelength [$\\mu$m]', fontsize=labelsize)
        ax[1].set_ylabel('Normalized Flux', fontsize=labelsize)
        ax[1].set_xlim(6, 14)
        ax[1].set_ylim(0.8, 1.1)
        #ax[0].text(0.9,1.6,'$T_{\\rm 01} \\propto n_{\\rm H} ^{-0.16}$',fontsize=labelsize,color='black')
        #ax[1].text(0.9,5,'$P_{\\rm th} \\propto n_{\\rm H} ^{0.84}$',fontsize=labelsize, color='black')

        for col in ax[:]:
            col.tick_params(which='both', width=1, direction='in', labelsize=labelsize, right='True',
                            top='True')
            col.tick_params(which='major', length=5)
            col.tick_params(which='minor', length=3)
        ax[0].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].xaxis.set_major_locator(MultipleLocator(5))
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[0].yaxis.set_major_locator(MultipleLocator(1))
        ax[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1].xaxis.set_major_locator(MultipleLocator(1))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(5))
        ax[1].yaxis.set_major_locator(MultipleLocator(0.1))

        ax[1].axhline(1,ls='--',color='black',lw=1)
        ax[1].plot(9.7,1.3,'|',color='black',markersize=20,markeredgewidth=2)
        ax[1].text(9, 1.05, 'Si-O $(9.7{\\mu}m)$', fontsize=labelsize+2,color='black')
        ax[1].legend(fontsize=labelsize - 2, loc='lower left')


    if 0:
        handles, labels = ax[1].get_legend_handles_labels()
        # ax.legend(loc ='lower right',fontsize=labelsize)
        print(labels)
        order2 = [1, 0]
        ax[1].legend([handles[idx] for idx in order2], [labels[idx] for idx in order2],
                  loc='lower left', frameon=False, fontsize=labelsize , markerscale=0.8)



save = 0
if save:
    figname='Figures/fig_ch34_rest.pdf'
    fig.savefig("".join((figname)), bbox_inches='tight')

plt.show()