import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import glob,os

flist = sorted(glob.glob('/home/slava/science/codes/python/jwst/temp/*'))
nrows = len(flist)
fig,ax = plt.subplots(2,1,sharex=True)
for i,f in enumerate(flist):
    data = np.loadtxt(f)
    if 'hotpix' in f or 'bkgr' in f:
        if 'bkgr' in f:
            f_b = data[:,1]
        else:
            f_h = data[:,1]
        ax[0].errorbar(data[:,0],data[:,1],yerr=data[:,2],label=f.split('/')[-1])
    if 'flat' in f or 'res_fringe' in f:
        ax[1].errorbar(data[:,0],data[:,1],yerr=data[:,2],label=f.split('/')[-1])
        if 'flat' in f:
            f_f = data[:,1]
ax[0].plot(data[:,0],f_h-f_b,label='bkgr')
ax[1].plot(data[:,0],f_b/f_f,label='flat')
ax[0].legend()
ax[1].legend()

plt.show()



for i,f in enumerate(flist):
    if 'flux' in f:
        data = np.loadtxt(f)
        x,y = data[:,0],data[:,1]

plt.subplots()

#y *= 1+0.5*np.sin(2.0*np.pi*x/100)
plt.plot(x,y)

x=x[550:850]
y=y[550:850]
from scipy.fft import fft, fftfreq
plt.subplots()
plt.plot(x,y)
yf = fft(y)
N=x.shape[0]
T=1/1000
xf = fftfreq(N, T)[:N//2]

plt.subplots()
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

plt.plot(xf,  np.abs(yf[0:N//2]))

plt.grid()

plt.show()