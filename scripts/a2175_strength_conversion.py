import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import numpy as np

def Drude(x,x0,gamma):
    return x**2/((x**2 - x0**2)**2+x**2*gamma**2)

def Ext_curve(l,c1,c2,c3,x0,gamma):
    x =1e4/l
    f = c1+c2*x+c3*Drude(x,x0,gamma)
    return f



fig,ax = plt.subplots(1,2)
for g in [0.6,1,1.8]:
    for c3 in np.linspace(0,4,20):
        l = np.linspace(3,5,1000)
        l = np.power(10,l)
        if 0:
            plt.subplots()
            plt.plot(1e4/l,Ext_curve(l,0.09,0.47,c3,g,1.11))
            plt.plot(1e4/l,Ext_curve(l,0.09,0.47,0,g,1.11))
            plt.xlabel('x=Wavelength^-1')

            plt.subplots()
            plt.plot(l,Ext_curve(l,0.09,0.47,c3,4.66,1.11))
            plt.plot(l,Ext_curve(l,0.09,0.47,0,4.66,1.11))
            plt.xlabel('Wavelength')


        c1,c2,x0 = 0.09,0.47,4.66
        y = Ext_curve(l,c1,c2,c3,x0,g) - Ext_curve(l,c1,c2,0,x0,g)
        if 0:
            plt.subplots()
            plt.plot(l,np.power(10,-y/2.5))
            plt.xlim(1e3,3e3)
            plt.axhline(0,ls=':')
            plt.axhline(1,ls='--')
            plt.title('c3='+str(c3))
        lmax = 5e3
        lmin = 1e3
        mask = (l < lmax) * (l > lmin)
        flux_2175 = np.power(10, -y / 2.5) - 1
        Int = np.trapz(flux_2175[mask], l[mask]) / (lmax - lmin)
        print('c3:',c3, ' Int:',Int)
        print('Abump:',np.pi*c3/2/g)

        ax[0].plot(c3,Int,'o')
        ax[0].plot(c3,np.pi*c3/2/g,'s')
        ax[1].plot(np.pi*c3/2/g,Int,'o')
        ax[1].set_xlabel('Abump')
        ax[1].set_ylabel('strength')
plt.show()



fig,ax = plt.subplots(1,2)
delta_f, Abump = [],[]
for g in [1.44]:
    for c3 in np.linspace(0,4,20):
        l = np.linspace(3,5,1000)
        l = np.power(10,l)
        if 0:
            plt.subplots()
            plt.plot(1e4/l,Ext_curve(l,0.09,0.47,c3,g,1.11))
            plt.plot(1e4/l,Ext_curve(l,0.09,0.47,0,g,1.11))
            plt.xlabel('x=Wavelength^-1')

            plt.subplots()
            plt.plot(l,Ext_curve(l,0.09,0.47,c3,4.66,1.11))
            plt.plot(l,Ext_curve(l,0.09,0.47,0,4.66,1.11))
            plt.xlabel('Wavelength')


        c1,c2,x0 = -0.55,0.62,4.66
        y = Ext_curve(l,c1,c2,c3,x0,g) - Ext_curve(l,c1,c2,0,x0,g)
        if 0:
            plt.subplots()
            plt.plot(l,np.power(10,-y/2.5)-1)
            plt.xlim(1e3,3e3)
            plt.axhline(0,ls=':')
            plt.axhline(1,ls='--')
            plt.title('c3='+str(c3))
            plt.show()
        lmax = 5e3
        lmin = 1e3
        mask = (l<lmax)*(l>lmin)
        flux_2175 = np.power(10,-y/2.5) - 1
        Int = np.trapz(flux_2175[mask],l[mask])/(lmax -lmin)
        print('c3:',c3, ' Int:',Int)
        print('Abump:',np.pi*c3/2/g)

        ax[0].plot(c3,Int,'o')
        ax[0].plot(c3,np.pi*c3/2/g,'s')
        ax[1].plot(np.pi*c3/2/g,Int,'o')
        ax[1].set_xlabel('Abump')
        ax[1].set_ylabel('strength')
        delta_f.append(Int)
        Abump.append(np.pi*c3/2/g)
plt.show()

from scipy.interpolate import interp1d
conv = interp1d(delta_f,Abump)
print('0235+164','delta_f=',-0.227,' A2175=', conv(-0.227))
print('0852','delta_f=',-0.152,' A2175=', conv(-0.152))
print('0927','delta_f=',-0.050,' A2175=', conv(-0.050))
print('1203','delta_f=',-0.138,' A2175=', conv(-0.138))
