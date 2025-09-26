import numpy as np
from scipy.interpolate import splrep, BSpline
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal

from jwst import datamodels # JWST datamodels
from jwst.associations import asn_from_list as afl # Tools for creating association files
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase # Definition of a Lvl2 association file
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base # Definition of a Lvl3 association file
from stcal import dqflags # Utilities for working with the data quality (DQ) arrays
from jwst.datamodels import dqflags
from stdatamodels.jwst import datamodels

from specutils.spectra import Spectrum1D
from specutils.fitting import fit_lines
from astropy import units as u
from astropy.modeling import models
from scipy.interpolate import interp1d
from astropy import modeling

l_name,l_central = '[ArII]' ,6.985 #12.81 #)
cube_name = './../output/detector3/QSO-B1830-211-SIGHTLINEB_3A_ch3-short_ArII_map_s3d.fits'
#cube_name = './../output/detector3/QSO-B1830-211-SIGHTLINEB_4B_ch4-medium__NeII_s3d.fits'
#l_name,l_central = '[NeIII]G2' ,15.555
#l_name,l_central = 'PAH6.25' ,6.22
z_gal = 0.886
#z_gal = 0.1926
detection_limit = 3 #sigma

#cube_name = './../output/detector3/QSO-B1830-211-SIGHTLINEB_4A_ch4-short_G2_NeIII_s3d.fits'


data = datamodels.open(cube_name)
(npix, nraw, ncol) = data.shape

wave_name = cube_name.split('_s3d')[0]+'_x1d.fits'
hdu2 = fits.open(wave_name)
data.wavelength = hdu2['EXTRACT1D'].data['WAVELENGTH']
hdu2.close()
# set wcs and quasars coords
if 1:
    def conv_world_coord(t,x,y,cube=data,mode='pipeline_world_to_pix'):
        if mode == 'pipeline_world_to_pix':
            wcs = cube.meta.wcs
            sky = wcs.world_to_pixel(x, y, t)
            x_world = sky[0]
            y_world = sky[1]
            lam_world = t
        return lam_world, x_world, y_world

    #for PKS1830
    l, raq, deq = np.nanmean(data.wavelength), 278.416405, -21.061055
    # l, raq, deq = np.nanmean(wave), 278.416310, -21.0611197
    asec = 1 / 3600.
    qA_pos_pix = conv_world_coord(t=l, x=raq, y=deq,
                                       mode='pipeline_world_to_pix')
    qB_pos_pix = conv_world_coord(t=l, x=raq - 0.653 * asec, y=deq - 0.721 * asec,
                                       mode='pipeline_world_to_pix')

    wcsinfo = data.meta.wcsinfo
    pix_solid_angle = np.array(wcsinfo.cdelt1 * wcsinfo.cdelt2 * (np.pi / 180) ** 2) * 1e6*1e3 #in mJy


from astropy.constants import c
c.to('km/s')
data.velocity = 3e5*(data.wavelength /l_central/(1+z_gal) - 1)

mask_to_line = (np.abs(data.velocity)<2000)
mom0 = np.zeros((nraw, ncol))
mom1 = np.zeros((nraw, ncol))
mom2 = np.zeros((nraw, ncol))
debug=False

image = np.nansum(data.data[np.abs(data.velocity)<1000],axis=0)
image_std = np.nanstd(data.data[np.abs(data.velocity)<1000],axis=0)

fig,ax = plt.subplots(1,2)
ax[0].imshow(image,origin='lower')
ax[0].contour(image,levels=[np.nanmax(image)*0.1,np.nanmax(image)*0.5],colors='black',origin='lower')
ax[0].set_title('Integral image')
ax[0].plot(qA_pos_pix[1],qA_pos_pix[2],'*',markersize=10,color='magenta')
ax[0].plot(qB_pos_pix[1],qB_pos_pix[2],'*',markersize=10,color='orange')
ax[1].imshow(image_std,origin='lower')
ax[1].contour(image_std,levels=[0.1*np.nanmax(image_std)],colors='black',origin='lower',linestyles='dashed')
ax[1].plot(qA_pos_pix[1],qA_pos_pix[2],'*',markersize=10,color='magenta')
ax[1].plot(qB_pos_pix[1],qB_pos_pix[2],'*',markersize=10,color='orange')
ax[1].set_title('STD image')

plt.show()
mask_flux = (image>np.nanmax(image)*0.1) #(image_std>0.1*np.nanmax(image_std))

case = 'calc_moments'

if case ==  'calc_flux_map':
    print('')
elif case == 'calc_moments':
    def gaussian(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi))*np.exp(-0.5 * ((x - mu) / sigma)**2)


    for i in range(nraw):
        for j in range(ncol):
            y = data.data[:,i,j][mask_to_line]
            v = data.velocity[mask_to_line]
            #mask nan values
            mask_nan = ~np.isnan(y)
            y = y[mask_nan]
            v = v[mask_nan]

            if np.sum(mask_nan)>0.5*np.sum(mask_to_line) and mask_flux[i,j]:
                def func(x, y0,k,a, cen, sig):
                    return y0+k*x + a*gaussian(x,cen,sig)


                from lmfit import Model

                init = [0, 0, 100, 0, 1000]
                fmodel = Model(func)
                pars = fmodel.make_params()
                pars['a'].min,pars['a'].max = 0,np.inf
                pars['sig'].min,pars['sig'].max = 30,3000
                pars['cen'].min, pars['cen'].max = -500, 500

                #for PAH
                #pars['sig'].value=1500
                #pars['sig'].vary = False

                result = fmodel.fit(y,  params=pars,x=v,y0=init[0],k=init[1],a=init[2],cen=init[3],sig=init[4])
                pars = result.params
                #res = [result.best_values['y0'], result.best_values['a'], result.best_values['m'], result.best_values['sig']]




                mask_out_of_line = np.abs(v-pars['cen'].value)>5*pars['sig'].value
                y_std = np.nanstd(y[mask_out_of_line])
                y_mean = np.nanmean(y[mask_out_of_line])
                line_amplitude =  func(x=pars['cen'].value,y0=pars['y0'].value,k=pars['k'].value,a=pars['a'].value,cen=pars['cen'].value,
                                       sig=pars['sig'].value)

                if line_amplitude>y_mean+detection_limit*y_std:
                    if debug:
                        print(pars)
                        print([el.value for el in pars.values()])
                        print(line_amplitude,y_mean+y_std)
                        plt.subplots()
                        plt.plot(v, y, '-')
                        v0 = np.linspace(-5000,5000,10000)
                        plt.plot(v0, func(v0, pars['y0'].value,pars['k'].value, pars['a'].value, pars['cen'].value, pars['sig'].value))
                        plt.axhline(2*y_std + y_mean, ls='--', color='gray')
                        plt.axhline(3 * y_std + y_mean, ls='--', color='gray')
                        plt.axhline(y_std+y_mean,ls='--',color='gray')
                        plt.axhline(y_mean, ls=':', color='gray')
                        plt.plot(data.velocity,data.data[:,i,j],zorder=-10)
                        plt.show()

                    mask_in_line = np.abs(v - pars['cen'].value) <= 5 * pars['sig'].value
                    mom0[i, j] =  np.trapz(y[mask_in_line]-pars['y0'].value, v[mask_in_line])
                    mom1[i, j] =  pars['cen'].value
                    mom2[i, j] =  pars['sig'].value

    #
    mask = mom0==0
    mom0[mask] = np.nan
    mom1[mask] = np.nan
    mom2[mask] = np.nan


    from matplotlib.colors import LogNorm
    fig,ax = plt.subplots(1,3,figsize=(12,3))
    fig.subplots_adjust(wspace=0.5)
    fontsize=12
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='black')
    im0=ax[0].imshow(mom0*pix_solid_angle,origin='lower',cmap=cmap) #,norm=LogNorm(vmin=1e3, vmax=1e4))
    cmap = plt.get_cmap('coolwarm').copy()
    cmap.set_bad(color='black')
    im1=ax[1].imshow(mom1,origin='lower',vmin=-150,vmax = 0,cmap=cmap)
    im2=ax[2].imshow(mom2,origin='lower',vmin =0, vmax = 100,cmap=cmap)
    for axs in ax[:]:
        axs.plot(qA_pos_pix[1], qA_pos_pix[2], '*', markersize=10, color='magenta')
        axs.plot(qB_pos_pix[1], qB_pos_pix[2], '*', markersize=10, color='orange')

    for k,im in enumerate([im0,im1,im2]):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax[k])
        cax = divider.append_axes("right", size="5%", pad=0.05)  # adjust size and padding
        cbar = plt.colorbar(im, cax=cax)
    ax[0].set_title('Flux, mJy km/s')
    ax[1].set_title('V, km/s')
    ax[2].set_title('$\Delta V$, km/s')

    fig.savefig("./../output/scripts/"+l_name+'.pdf', bbox_inches='tight')
    plt.show()
    print()
