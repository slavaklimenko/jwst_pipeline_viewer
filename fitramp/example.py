import numpy as np
import fitramp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time



Npix = 100                          # pixels per row
nrows = 100                         # number of rows of pixels
countrate = 5*np.ones((nrows, Npix)) # count rate for each pixel
sig = 30*np.ones((nrows, Npix))      # uncertainty for each pixel
ijump = 12
jump_dy = 5*15
xx,yy = 45,50

#readtimes = [1, 2, 3, [4, 5], [6, 7, 8], [10, 11, 13], [15, 18], [21, 22], 23, [25, 26]]
# Use the code below to do the same checks on a ramp of 30 individual reads.
readtimes = np.arange(1, 31)

C = fitramp.Covar(readtimes)

im = np.empty((len(readtimes), nrows, Npix))
for i in range(nrows):
    im[:, i] = fitramp.getramps(countrate[i], sig[i], readtimes, nramps=Npix)
d = (im[1:] - im[:-1])/C.delta_t[:, np.newaxis, np.newaxis]
groupdq = np.zeros((1,len(readtimes), nrows, Npix))
groupdq[0,:10] = 1
groupdq[0,-1] = 1

plt.subplots()
plt.plot(im[:,10,10],'o')
plt.plot(d[:,10,10],'-')

########################
if 0:
    t0 = time.time()
    debias = True

    res = np.zeros_like(countrate)

    for i in range(nrows):
        result = fitramp.fit_ramps(d[:, i], C, sig[i])

        if debias:
            countrateguess = result.countrate * (result.countrate > 0)
            result = fitramp.fit_ramps(d[:, i], C, sig[i],
                                       countrateguess=countrateguess)
        res[i] = result.countrate
        # The count rates, chi squared values, and uncertainties for this row are:
        # result.countrate
        # result.chisq
        # result.uncert
        #
        # You can save these to another structure if you want, or you can
        # reorganize the structure for result used here.

        # Note: if we want the weights, they are available, as follows.

        # Weights on the scaled resultant differences:

        # weights_differences = result.weights

        # Or the equivalent weights on the resultants themselves:

        # weights_resultants = np.zeros(d[:, i].shape)
        # weights_resultants[1:] = result.weights/C.delta_t[:, np.newaxis]
        # weights_resultants[:-1] -= result.weights/C.delta_t[:, np.newaxis]

    plt.subplots()
    plt.hist(res)

    plt.subplots()
    ima = plt.imshow(res)
    cbar = plt.colorbar(ima, extend='both', shrink=0.9)
    plt.show()
    print("\nTime per H4RG: %.3g seconds" %
          ((time.time() - t0) * 4096 ** 2 / (Npix * nrows)))

    print("Time per 1e8 pixel-resultants: %.3g seconds" %
          ((time.time() - t0) * 1e8 / np.prod(d.shape)))

########################
#add  the pedestal to fit parameters
if 0:
    C_wped = fitramp.Covar(readtimes, pedestal=True)
    d_wped = np.zeros(im.shape)
    d_wped[0] = im[0]/C.mean_t[0]
    d_wped[1:] = (im[1:] - im[:-1])/C.delta_t[:, np.newaxis, np.newaxis]

    for i in range(nrows):
        result = fitramp.fit_ramps(d_wped[:, i], C_wped, sig[i], resetval=0, resetsig=np.inf)

        # The count rates, chi squared values, pedestals, and uncertainties are then:
        #
        # result.countrate
        # result.chisq
        # result.uncert
        # result.pedestal
        # result.uncert_pedestal
        # result.covar_countrate_pedestal

########################
#calc the dependence between count rate and bias ??
if 0:
    sig_bias = 10
    countrates = 10**(np.linspace(-1, 4, 100))
    cvec = np.ones(d.shape[0])
    bias_constant_c = C.calc_bias(countrates, sig_bias, cvec)
    plt.figure(figsize=(6, 4))
    rcParams['font.size'] = 13.5
    plt.semilogx(countrates, bias_constant_c, linewidth=3)
    plt.xlabel("Count Rate ($e^-{\\rm s}^{-1}$)")
    plt.ylabel("Bias ($e^-{\\rm s}^{-1}$)")
    plt.show()

########################
#add jump detecton to the fit parameters
t0 = time.time()
########################
res = np.zeros((d.shape[1],d.shape[2]))
ped = np.zeros((d.shape[1],d.shape[2]))
for i in range(nrows):
    diffs2use, countrates = fitramp.mask_jumps(d[:, i], C, sig[i], threshold_oneomit=20.25,    threshold_twoomit=23.8)
    #result = fitramp.fit_ramps(d[:, i], C, sig[i], diffs2use=diffs2use, countrateguess=countrates * (countrates > 0))
    C_wped = fitramp.Covar(readtimes, pedestal=True)
    d_wped = np.zeros(im.shape)
    d_wped[0] = im[0] / C.mean_t[0]
    d_wped[1:] = (im[1:] - im[:-1]) / C.delta_t[:, np.newaxis, np.newaxis]
    diffs2use_wped = np.ones((diffs2use.shape[0]+1,diffs2use.shape[1]))
    diffs2use_wped[1:]  = diffs2use
    #diffs2use, countrates = fitramp.mask_jumps(d_wped[:, i], C_wped, sig[i], threshold_oneomit=20.25,    threshold_twoomit=23.8)
    result = fitramp.fit_ramps(d_wped[:, i], C_wped, sig[i], resetval=0, resetsig=np.inf, diffs2use=diffs2use_wped, countrateguess=countrates * (countrates > 0))
    res[i, :] = result.countrate
    ped[i,:] = result.pedestal
print("\nTime per H4RG with jump detection & debiasing: %.3g seconds" %
      ((time.time() - t0) * 4096 ** 2 / (nrows * Npix)))

print("Time per 1e8 pixel-resultants with jump detection & debiasing: %.3g seconds" %
      ((time.time() - t0) * 1e8 / np.prod(d.shape)))

plt.plot(ped[xx,yy]+res[xx,yy]*np.arange(np.size(readtimes)))
plt.plot(ped[xx,yy]+5*np.arange(np.size(readtimes)),c='red',ls=':')
plt.show()

########################

x, y = np.meshgrid(np.arange(d.shape[2]), np.arange(d.shape[1]))
d = (im[1:] - im[:-1])/C.delta_t[:, np.newaxis, np.newaxis]
d[ijump] += jump_dy*np.exp(-((x - x.mean())**2 + (y - y.mean())**2)/(2*5**2))
im_jumped = im.copy()
for i in range(im.shape[0]-1):
    im_jumped[i+1] = im_jumped[i]+d[i]
print(x.mean(),y.mean())
plt.subplots()
plt.plot(im_jumped[:,xx,yy],'o')
plt.plot(d[:,xx,yy])


alljumps = np.zeros(d.shape)
alljumpsigs = np.zeros(d.shape)

res = np.zeros((d.shape[1],d.shape[2]))
ped = np.zeros((d.shape[1],d.shape[2]))
res2 = np.zeros((d.shape[1],d.shape[2]))

for i in range(nrows):
    diffs2use, countrates = fitramp.mask_jumps(d[:, i], C, sig[i], threshold_oneomit=20.25,
                                               threshold_twoomit=23.8)
    ct = countrates * (countrates > 0)
    indx = groupdq[0,1:,i,:] != 0
    diffs2use[indx] = 0
    #result = fitramp.fit_ramps(d[:, i], C, sig[i], diffs2use=diffs2use,  detect_jumps=True, countrateguess=ct)
    #alljumps[:, i] = result.jumpval_oneomit
    #alljumpsigs[:, i] = result.jumpsig_oneomit
    #res[i,:] = result.countrate
    #ped[i, :] = result.pedestal
    #res2[i,:] = result.countrate
    if 1:
        #C_wped = fitramp.Covar(readtimes, pedestal=True)
        #d_wped = np.zeros(im.shape)
        #d_wped[0] = im[0] / C.mean_t[0]
        #d_wped[1:] = (im[1:] - im[:-1]) / C.delta_t[:, np.newaxis, np.newaxis]
        #diffs2use_wped = np.ones((diffs2use.shape[0] + 1, diffs2use.shape[1]))
        #diffs2use_wped[1:] = diffs2use
        result = fitramp.fit_ramps(d[:, i], C, sig[i], diffs2use=diffs2use,  detect_jumps=True, countrateguess=ct)
        alljumps[:, i] = result.jumpval_oneomit
        alljumpsigs[:, i] = result.jumpsig_oneomit
        res[i,:] = result.countrate
        ped[i, :] = (im[0,i,:]+im[1,i,:])/2
        #res2[i,:] = result.countrate

    for j in range(len(d)):
        indx = diffs2use[j] == 0  # only need to redo these differences
        if np.sum(indx) == 0:
            continue
        # each time we'll make sure that this difference isn't masked
        mask = diffs2use[:, indx] * 1
        mask[j] = 1
        dd = d[:, i, indx]
        result = fitramp.fit_ramps(d[:, i, indx], C, sig[i, indx], diffs2use=mask,
                                   detect_jumps=True, countrateguess=ct[indx])

        # Overwrite the jump value if it was previously masked.
        alljumps[j, i, indx] = result.jumpval_oneomit[j]
        alljumpsigs[j, i, indx] = result.jumpsig_oneomit[j]
        groupdq[0,j,i,indx] = 4
        #res2[i, indx] = result.countrate
        #if res2[i, indx] != res[i, indx]:
        #print(i,np.where(indx!=False),res2[i, indx],res[i, indx])

plt.plot(ped[xx,yy]+5*np.arange(np.size(readtimes)),c='red',ls=':')
plt.plot(ped[xx,yy]+res[xx,yy]*np.arange(np.size(readtimes)))
plt.plot(ped[xx,yy]+alljumps[ijump,xx,yy]+res[xx,yy]*np.arange(np.size(readtimes)),'--')
#plt.plot(0+res2[xx,yy]*np.arange(np.size(readtimes)))

plt.show()


print(x.mean(),y.mean())
plt.subplots()
plt.plot(d[:,55,48],'o')
plt.axhline(res[55,48])
plt.axhline(res2[55,48])
plt.show()

fig,ax = plt.subplots(1,3)
ax[0].imshow(res,vmin=4,vmax=6)
ax[0].set_title('res')
ax[1].imshow(res2,vmin=4,vmax=6)
ax[1].set_title('res2')
ax[2].imshow(res2-res,vmin=-1,vmax=1)

fig,ax = plt.subplots(figsize=(12, 4))
maxval = np.amax(np.abs(alljumps[ijump]))*0.5
ax.imshow(alljumps[ijump], origin='lower', cmap='seismic')
ax.set_title("Jump Value (from $\chi^2$ analysis)")
#plt.clf()
plt.figure(figsize=(12, 4))
plt.imshow(d[ijump] - np.median(d, axis=0), origin='lower', cmap='seismic')
#           vmin=-maxval, vmax=maxval)
plt.title("Jump Value (from single difference)")
#plt.show()
plt.figure(figsize=(12, 4))
plt.imshow(d[ijump], origin='lower', cmap='seismic')
#           vmin=-maxval, vmax=maxval)
plt.title("Jump Value (from single difference)")
plt.show()
