import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
path = '/home/slava/science/codes/python/jwst/data/references/jwst/miri/jwst_miri_flat_0813.fits'
from scipy.interpolate import interp1d
from scipy import signal


def get_trace_mask(path,debug=False):
    hdulist = fits.open(path)
    hdu = hdulist[1].data
    hdulist.close()

    data = hdu.copy()
    data[np.isnan(data)] = 0


    win = signal.windows.hann(20)
    l = np.nansum(data,axis=0)/data.shape[0]
    ls = signal.convolve(l,win)/np.sum(win)
    if debug:
        plt.plot(win)
        plt.plot(ls)
        plt.plot(l)
        plt.show()

    # calc number of traces
    nrows,ncols =data.shape[0],data.shape[1]
    norders = 0
    for i in range(ls.shape[0]-2):
        if i>2 and ls[i-1]<ls[i] and ls[i]>ls[i+1]:
            norders+=1


    data[data>1] = 1
    win = signal.windows.hann(4)
    #l = np.nansum(data,axis=0)/data.shape[0]
    #ls = signal.convolve(l,win)/ sum(win)
    orders_size = np.zeros((norders,nrows,2))
#    n = 0
    left_border,right_border = 0,0
    for i in range(nrows):
        line = data[i,:]
        line[line>0] = 1
        if np.sum(line)>0:
            if i<nrows/2:
                l = np.nansum(data[i:i+30,:], axis=0)/30
            else:
                l = np.nansum(data[i-30:i, :], axis=0) / 30
            ls = signal.convolve(l, win)/ sum(win)

            ls[ls>0] = 1

            f_interp = interp1d(np.arange(ls.shape[0]),ls)
            f = f_interp(np.arange(l.shape[0]))

            if debug:
                fig, ax = plt.subplots(2, 1,sharex=True)

                ax[0].plot(line)
                ax[0].plot(ls)
                ax[0].set_title(str(i))
                ax[1].imshow(data[i - 3:i + 30, :])
                ax[0].plot(f,color='red',ls=':')
                plt.show()

            num_ord = -1
            for j in range(f.shape[0]-1):
                if f[j]==0 and f[j+1]==1:
                    num_ord += 1
                    left_border = 0
                    for k in np.arange(j-10,j+10):
                        if k>=0 and k<=line.shape[0]-1:
                            if line[k] == 0 and line[k + 1] == 1:
                                left_border = k+1
                                orders_size[num_ord, i, 0] = left_border

                if f[j] == 1 and f[j + 1] == 0 and left_border>0:
                    right_border = 0
                    for k in np.arange(j,j-20,-1):
                        if line[k] == 1 and line[k + 1] == 0:
                            right_border = k
                            orders_size[num_ord, i, 1] = right_border
                    #print(num_ord,left_border,right_border)

    return orders_size

get_trace_mask(path=path)