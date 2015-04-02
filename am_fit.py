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

#fileNames = glob.glob('./nh3_all/*fits')
fileNames = glob.glob('./nh3_all/B*.fits')
#fileNames = glob.glob('./nh3_all/GSerpBolo*.fits')

a = np.arange(len(fileNames))
objects = [((os.path.basename(fileNames[name])))[0:-9] for name in range(max(a))]
objects = sorted(set(objects))

# Tables; Note: We can extract an array from the astropy.Tables that are generated; ex. t_pars['TKIN']
t_int = Table(names=('FILENAME','W11','W22','W33','W44'),dtype=('S20','f5','f5','f5','f5'))
t_w11 = Table(names=('FILENAME','W11OBS','W11EMP','RMSOBS','RMSEMP','DIFF','PERCERR'),dtype=('S20','f5','f5','f5','f5','f5','f5'))
t_pars = Table(names=('FILENAME','TKIN','TEX','N','SIGV','V','F'),dtype=('S20','f5','f5','f5','f5','f5','f1'))
t_errs = Table(names=('FILENAME','TKIN','TEX','N','SIGV','V','F'),dtype=('S20','f5','f5','f5','f5','f5','f1'))
t_dist = Table(names=('FILENAME','DIST','RGAL','G.LONG','G.LAT'),dtype=('S20','f5','f5','f5','f5'))

c = 2.99792458e8

for thisObject in objects: 
    spectrum = {}
    fnameT = './nh3_figures/'+thisObject+'.png'
    fnameT2 = './nh3_figures2/'+thisObject+'.png'

    if os.path.exists('./nh3_all/'+thisObject+'.n11.fits'):
       data1 = fits.getdata('./nh3_all/'+thisObject+'.n11.fits')
       A1 = np.arange(len(data1['DATA'].T))
       nu1 = data1['CDELT1']*(A1-data1['CRPIX1']+1)+data1['CRVAL1']
       v1 = c*(1-nu1/data1['RESTFREQ'])

       spec1 = pyspeckit.Spectrum(data=data1['DATA'].T.squeeze(),unit='K',xarr=v1,xarrkwargs={'unit':'m/s','refX':data1['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spectrum['oneone'] = spec1

       # Cross-correlation method to get first guess
       guess = NH3FirstGuess(spec1,vmin=-250.0,vmax=250)

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
          spec_pars = spectra1.specfit.modelpars    	        
          spec_pars.insert(0,thisObject)                 	
			        
          spec_errs = spectra1.specfit.modelerrs    	        
          spec_errs.insert(0,thisObject)                 	

          # Distances and galactic coordinates
          glong, glat = parse_coords(data1)
          distance, rgal = kdist(glong, glat, spectra1.specfit.modelpars[4], rrgal = True)
          d_row = [thisObject,distance,rgal,glong, glat]

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
  

print t_pars
print t_errs
print t_w11
print t_int
print t_dist

    
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

