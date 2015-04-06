# coding: utf-8
import os
import glob
import math
import numpy as np
import numpy.fft as fft
import pylab as py
import array
import pyspeckit
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.ndimage.filters import median_filter

from kdist import *
from NH3FirstGuess import *
from parse_coords import *

"""
Variables:  data* = spectral data extracted from .fits file
	    A* = array used to calculate nu*
	    nu* = frequency
	    v* = doppler shift velocity

	    guess = "guesses" of fit paramaters to pass through pyspeckit; derived via cross-correlation
	    spec* = Spectrum object created from pyspeckit
	    spectrum = dictionary to pass through NH3 fitting from pyspeckit
	    
	    Note: * is the value of the transition; ie. 1 = (1,1), 2 = (2,2), etc.
	  
Output:  nh3dict = Dictionary of the entire spectrum
	 Histogram of fit parameters

	 t_par = table of fit parameters
	 t_int = table of integrated intensities
	 t_W11 = table of W11; with errors
	 t_errs = table of errors of fit parameters

	 creates fitted spectra plots and saves it all into a directory
	 Bivariate plots with fit parameters
	
	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻ 	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻
"""

# Whole set of data
fileNames = glob.glob('./nh3_all/*fits')

# Small section of data
#fileNames = glob.glob('./nh3_all/G049*.fits')


a = np.arange(len(fileNames))
objects = [((os.path.basename(fileNames[name])))[0:-9] for name in range(max(a))]
objects = sorted(set(objects))

# Tables; Note: We can extract an array from the astropy.Tables that are generated; ex. t_pars['TKIN']
t_int = Table(names=('FILENAME','W11','W22','W33','W44'),dtype=('S20','f5','f5','f5','f5'))
t_w11 = Table(names=('FILENAME','W11OBS','W11EMP','RMSOBS','RMSEMP','DIFF','PERCERR'),dtype=('S20','f5','f5','f5','f5','f5','f5'))
t_pars = Table(names=('FILENAME','TKIN','TEX','N','SIGV','V','F'),dtype=('S20','f5','f5','f5','f5','f5','f1'))
t_errs = Table(names=('FILENAME','TKINERR','TEXERR','NERR','SIGVERR','VERR','FERR'),dtype=('S20','f5','f5','f5','f5','f5','f1'))
t_dist = Table(names=('FILENAME','DIST','RGAL','G.LONG','G.LAT'),dtype=('S20','f5','f5','f5','f5'))

c = 2.99792458e8

voff_lines = np.array([19.8513, 
                  19.3159, 
                  7.88669, 
                  7.46967, 
                  7.35132, 
                  0.460409, 
                  0.322042, 
                  -0.0751680, 
                  -0.213003,  
                  0.311034, 
                  0.192266, 
                  -0.132382, 
                  -0.250923, 
                  -7.23349, 
                  -7.37280, 
                  -7.81526, 
                  -19.4117, 
                  -19.5500])

tau_wts = np.array([0.0740740, 
              0.148148, 
              0.0925930, 
              0.166667, 
              0.0185190, 
              0.0370370, 
              0.0185190, 
              0.0185190, 
              0.0925930, 
              0.0333330, 
              0.300000, 
              0.466667, 
              0.0333330, 
              0.0925930, 
              0.0185190, 
              0.166667, 
              0.0740740, 
              0.148148])

deltanu = -1*voff_lines/((c/1000)*23.6944955e9)

vmin = -250
vmax = 250

for thisObject in objects: 
    spectrum = {}
    fnameT = './nh3_figures/'+thisObject+'.png'
    fnameT2 = './nh3_figures2/'+thisObject+'.png'
    fnameT3 = './nh3_figures3/'+thisObject+'.png'

    if os.path.exists('./nh3_all/'+thisObject+'.n11.fits'):
       data1 = fits.getdata('./nh3_all/'+thisObject+'.n11.fits')
       A1 = np.arange(len(data1['DATA'].T))
       nu1 = data1['CDELT1']*(A1-data1['CRPIX1']+1)+data1['CRVAL1']
       v1 = c*(1-nu1/data1['RESTFREQ'])

       spec1 = pyspeckit.Spectrum(data=data1['DATA'].T.squeeze(),unit='K',xarr=v1,xarrkwargs={'unit':'m/s','refX':data1['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spectrum['oneone'] = spec1

       # Cross-correlation method to get first guess
       linewidth = 0.5
       chanwidth = (spec1.xarr[1]-spec1.xarr[0])/1e3
       ftdata = fft.fft(spec1.data.filled(0))
       tvals = fft.fftfreq(len(spec1.data))/chanwidth
       deltafcns = np.zeros(spec1.data.shape,dtype=np.complex)
       for idx, dv in enumerate(voff_lines):
          deltafcns += tau_wts[idx]*(np.cos(2*np.pi*dv*tvals)+
                               1j*np.sin(2*np.pi*dv*tvals))*\
                               np.exp(-tvals**2*(linewidth/chanwidth)**2/(2))
       ccor = np.real((fft.ifft(np.conj(ftdata)*deltafcns))[::-1])

       vaxis = np.array(spec1.xarr.as_unit('km/s'))
       subsetidx = (vaxis>vmin)*(vaxis<vmax)*np.isfinite(np.array(spec1.data))
       peakIndex = np.argmax(ccor[subsetidx])

       #pull out a 6 km/s slice around the peak
       deltachan = np.abs(3.0 / chanwidth)
       t = ((spec1.data.filled(0))[subsetidx])[(peakIndex-deltachan):(peakIndex+deltachan)]
       v = (vaxis[subsetidx])[(peakIndex-deltachan):(peakIndex+deltachan)]
       # Calculate line width.
       sigv = np.sqrt(abs(np.sum(t*v**2)/np.sum(t)-(np.sum(t*v)/np.sum(t))**2))

       if (np.isnan(sigv)):
          sigv = 0.85840189 # mean of sigv from first set of data from ./nh3_all/

       # Peak of cross correlation is the brightness.

       v0 = np.float((vaxis[subsetidx])[peakIndex])
       # Set the excitation temperature to be between CMB and 20 K
       # and equal to the peak brightness + 2.73 if valid.
       tex = np.min([np.max([spec1.data.filled(0)[peakIndex],0])+2.73,20])


       guess = [20, # 20 K kinetic temperature
           tex,  #
           15, # Log NH3
           sigv, # velocity dispersion
           v0,
           0.5]

    if os.path.exists('./nh3_all/'+thisObject+'.n22.fits'):
       data2 = fits.getdata('./nh3_all/'+thisObject+'.n22.fits')
       A2 = np.arange(len(data2['DATA'].T))
       nu2 = data2['CDELT1']*(A2-data2['CRPIX1']+1)+data2['CRVAL1']
       v2 = c*(1-nu2/data2['RESTFREQ'])
       spec2 = pyspeckit.Spectrum(data=data2['DATA'].T.squeeze(),unit='K',xarr=v2,xarrkwargs={'unit':'m/s','refX':data2['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spectrum['twotwo'] = spec2

    if os.path.exists('./nh3_all/'+thisObject+'.n33.fits'):
       data3 = fits.getdata('./nh3_all/'+thisObject+'.n33.fits')
       A3 = np.arange(len(data3['DATA'].T))
       nu3 = data3['CDELT1']*(A3-data3['CRPIX1']+1)+data3['CRVAL1']
       v3 = c*(1-nu3/data3['RESTFREQ'])
       spec3 = pyspeckit.Spectrum(data=data3['DATA'].T.squeeze(),unit='K',xarr=v3,xarrkwargs={'unit':'m/s','refX':data3['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spectrum['threethree'] = spec3

    if os.path.exists('./nh3_all/'+thisObject+'.n44.fits'):
       data4 = fits.getdata('./nh3_all/'+thisObject+'.n44.fits')
       A4 = np.arange(len(data4['DATA'].T))
       nu4 = data4['CDELT1']*(A4-data4['CRPIX1']+1)+data4['CRVAL1']
       v4 = c*(1-nu4/data4['RESTFREQ'])
       spec4 = pyspeckit.Spectrum(data=data4['DATA'].T.squeeze(),unit='K',xarr=v4,xarrkwargs={'unit':'m/s','refX':data4['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spectrum['fourfour'] = spec4
						
    spdict1,spectra1 = pyspeckit.wrappers.fitnh3.fitnh3tkin(spectrum,dobaseline=False,guesses=guess,fixed=[False,False,False,False,False,True])

    glong, glat = parse_coords(data1)

    # Filters out good and bad fits
    if -150 < spectra1.specfit.modelpars[4] < 150:

       # Further filtering out bad fits with Tk < 8 and Tex < 3

       if spectra1.specfit.modelpars[0] < 7.5:
          plt.savefig(fnameT2.format(thisObject), dpi = 100, format='png')
          plt.close()        
  
       elif spectra1.specfit.modelpars[1] < 2.9:
          plt.savefig(fnameT2.format(thisObject), dpi = 100, format='png')
          plt.close()

       else:  
          # The good fits are stored in tables
          spec_pars = [thisObject,spectra1.specfit.modelpars[0],spectra1.specfit.modelpars[1],spectra1.specfit.modelpars[2],spectra1.specfit.modelpars[3],spectra1.specfit.modelpars[4],spectra1.specfit.modelpars[5]]    	                	
			        
          spec_errs = spectra1.specfit.modelerrs    	        
          spec_errs.insert(0,thisObject)                 	

          # Distances and galactic coordinates
          
          distance, rgal = kdist(np.atleast_1d(glong), np.atleast_1d(glat), np.atleast_1d(spectra1.specfit.modelpars[4]), rrgal = True)
          d_row = [thisObject,distance,rgal,glong,glat]

          # Error calculation for W11 between observational and empirical
          W11_oarr = spec1.specfit.model
          W11_obs = np.sum(W11_oarr)*(v1.max()-v1.min())/(len(v1)*1000)
          W11_index = np.where(W11_oarr > 1e-6)
          W11_emp = np.sum(spec1.data[W11_index])*(v1.max()-v1.min())/(len(v1)*1000)
          W11_diff = W11_obs - W11_emp
          W11_perc = abs(((W11_obs - W11_emp)*100)/W11_obs)
          W11_oerr = np.nanstd(W11_oarr)
          NoSignal = np.where(W11_oarr < 1e-6)
          W11_eerr = np.nanstd(spec1.data[NoSignal])
          W11_row = [thisObject,W11_obs,W11_emp,W11_oerr,W11_eerr,W11_diff,W11_perc]
    
          # Integrated intensities for all transitions
          w_int = [thisObject,np.sum(spec1.specfit.model)*(v1.max()-v1.min())/(len(v1)*1000),np.sum(spec2.specfit.model)*(v2.max()-v2.min())/(len(v2)*1000),None,None]
          if os.path.exists('./nh3/'+thisObject+'.n33.fits'):
             w_int[3] = np.sum(spec3.specfit.model)*(v3.max()-v3.min())/(len(v3)*1000)
       
          if os.path.exists('./nh3/'+thisObject+'.n44.fits'):
             w_int[4] = np.sum(spec4.specfit.model)*(v4.max()-v4.min())/(len(v4)*1000)
    
          t_pars.add_row(spec_pars) 
          t_errs.add_row(spec_errs) 
          t_w11.add_row(W11_row)   
          t_int.add_row(w_int)
          t_dist.add_row(d_row)

          plt.savefig(fnameT.format(thisObject), dpi = 100, format='png')
          plt.close()

    else:
       plt.savefig(fnameT2.format(thisObject), dpi = 100, format='png')
       plt.close()

# Save tables after loop is done; note we get errors as we can't overwrite it

t_pars.write('./nh3_tables/nh3_pars.fits',format='fits')
t_errs.write('./nh3_tables/nh3_errs.fits',format='fits')
t_w11.write('./nh3_tables/nh3_w11.fits',format='fits')
t_int.write('./nh3_tables/nh3_int.fits',format='fits')
t_dist.write('./nh3_tables/nh3_dist.fits',format='fits')

"""
#plotting 
pars = fits.getdata('./nh3_tables/final/nh3_pars.fits')
errs = fits.getdata('./nh3_tables/final/nh3_errs.fits')
dist = fits.getdata('./nh3_tables/final/nh3_dist.fits')
w11  = fits.getdata('./nh3_tables/final/nh3_w11.fits')
wint = fits.getdata('./nh3_tables/final/nh3_int.fits')


f, axarr = plt.subplots(2,sharex=True)
axarr[0].scatter(dist['RGAL']/1000,pars['N'])
axarr[0].set_ylabel('Column Density (log(N))')
#axarr[0].set_ylim([0,55])
axarr[0].set_xlim([0,15])
axarr[1].scatter(dist['RGAL']/1000,pars['TEX'])
axarr[1].set_ylabel('Excitation Temperature (K)')
axarr[1].set_xlabel('Galactocentric Distance (kpc)')
#axarr[1].set_ylim([-65,140])
axarr[1].set_xlim([0,15])


# Fit parameter histograms
plt.clf()            
py.hist(t_pars['TKIN'],bins=100)
plt.xlabel('Kinetic Temperature (K)')
plt.ylabel('Numbers')
plt.title('Histogram of Kinetic Temperatures ($T_k$)')
plt.savefig('./ammonia_plots/histogram_tkin.png', format='png')
plt.close()

plt.clf()            
py.hist(t_pars['TEX'],bins=100)
plt.xlabel('Excitation Temperature (K)')
plt.ylabel('Numbers')
plt.title('Histogram of Excitation Temperatures ($T_{ex}$)')
plt.savefig('./ammonia_plots/histogram_tex.png', format='png')
plt.close()

plt.clf()            
py.hist(t_pars['N'],bins=100)
plt.xlabel('Column Density')
plt.ylabel('Numbers')
plt.title('Histogram of Column Density ($log(N)$)')
plt.savefig('./ammonia_plots/histogram_N.png', format='png')
plt.close()

plt.clf()            
py.hist(t_pars['SIGV'],bins=100)
plt.xlabel('Line Width ($cm^{-2}$)')
plt.ylabel('Numbers')
plt.title('Histogram of Line Width ($\sigma$)')
plt.savefig('./ammonia_plots/histogram_sigma.png', format='png')
plt.close()

plt.clf()            
py.hist(t_pars['V'],bins=100)
plt.xlabel('Line-of-Sight Velocity (km/s)')
plt.ylabel('Numbers')
plt.title('Histogram of Line-of-Sight Velocity ($v$)')
plt.savefig('./ammonia_plots/histogram_vlos.png', format='png')
plt.close()


# Scatter plots with fit parameters
plt.clf()            
plt.scatter(t_pars['TKIN'],t_pars['SIGV'])
plt.xlabel('Kinetic Temperature (K)')
plt.ylabel('Line Width ($cm^{-2}$)')
plt.title('Kinetic Temperature ($T_k$) and Line Width ($\sigma$)')
plt.savefig('./ammonia_plots/tkin_vs_sigma.png', format='png')
plt.close()

plt.clf()            
plt.scatter(t_pars['TKIN'],t_pars['TEX'])
plt.xlabel('Kinetic Temperature (K)')
plt.ylabel('Excitation Temperature (K)')
plt.title('Kinetic Temperature ($T_k$) vs Excitation Temperature ($T_{ex}$)')
plt.savefig('./ammonia_plots/tkin_tex.png', format='png')
plt.close()

plt.clf()            
plt.scatter(t_pars['N'],t_pars['TKIN'])
plt.ylabel('Kinetic Temperature (K)')
plt.xlabel('Column Density (log(N))')
plt.title('Column Density ($log(N)$) vs Kinetic Temperature ($T_k$)')
plt.savefig('./ammonia_plots/N_vs_tkin.png', format='png')
plt.close()

# assume gamma = 1 cause its isothermal
kb = 1.3806488E-23
m = 2.82E-26
c_s = np.zeros(len(t_pars['TKIN']),dtype = np.float64)
Ma = np.zeros(len(t_pars['TKIN']),dtype = np.float64)
for i in range(0,len(t_pars['TKIN'])):
   c_s[i] = math.sqrt(kb*t_pars['TKIN'][i]/m)
   Ma[i] = t_pars['SIGV'][i]/(np.float(c_s[i])/1000)

plt.clf()            
plt.scatter(Ma,t_pars['TKIN'])
plt.ylabel('Kinetic Temperature (K)')
plt.xlabel('Mach Number')
plt.title('Mach Number ($Ma$) vs Kinetic Temperature ($T_k$)')
plt.savefig('./ammonia_plots/Ma_vs_tkin.png', format='png')
plt.close()
"""
