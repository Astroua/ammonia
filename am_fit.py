# coding: utf-8
import os
import glob
import math
import numpy as np
import pylab as py
import array
import pyspeckit as psk
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.ndimage.filters import median_filter

"""
Variables:  data* = spectral data extracted from .fits file
	    A* = array used to calculate nu*
	    nu* = frequency
	    v* = doppler shift velocity

	    v_medf = median filter velocity for smoothing
	    max_index = index of the maximum value of the spectrum

	    guess = "guesses" of fit paramaters to pass through pyspeckit
	    spec* = Spectrum object created from pyspeckit
	    spectrum = dictionary to pass through NH3 fitting from pyspeckit
	    
	    Note: * is the value of the transition; ie. 1 = (1,1), 2 = (2,2), etc.
	  

Output:  nh3dict = Dictionary of the entire spectrum
	 Histogram of fit parameters
	 t_par = table of fit parameters
	 t_int = table of integrated intensities
	 t_W11 = table of W11; with errors
	 creates fitted spectra plots and saves it all into a directory
	 Bivariate plots with fit parameters
	
	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻ 	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻
"""


fileNames = glob.glob('./nh3/*fits')
#fileNames = glob.glob('./nh3/GSerpBolo2*.n*.fits')
#fileNames = glob.glob('./nh3/G010*.n*.fits')

a = np.arange(len(fileNames))
objects = [((os.path.basename(fileNames[name])))[0:-9] for name in range(max(a))]
objects = sorted(set(objects))

# Tables; Note: We can extract an array from the astropy.Tables that are generated; ex. t_pars['TKIN']
t_int = Table(names=('FILENAME','W11','W22','W33','W44'),dtype=('S20','f5','f5','f5','f5'))
t_w11 = Table(names=('FILENAME','W11_obs','W11_emp','RMS-error; obs','RMS-error; emp','W11_obs - W11_emp','%-error'),dtype=('S20','f5','f5','f5','f5','f5','f5'))
t_pars = Table(names=('FILENAME','TKIN','TEX','N(0)','SIGMA(0)','V(0)','F_0(0)'),dtype=('S20','f5','f5','f5','f5','f5','f1'))

# This creates the dictionary to then pass to pyspeckit to create the fit
# value is the pixel size for filtering with median_filter
value = 17
c = 2.99792458e8

# One big thing to note is that the guess was determined with a bias towords the GSerpBolo .fits files
for thisObject in objects: 
    spectrum = {}
    fnameT = './nh3_figures/'+thisObject+'.png'
    fnameT2 = './nh3_figures2/'+thisObject+'.png'
    guess = [15, 7, 15, 2, 30, 0]
    w_row = []

    if os.path.exists('./nh3/'+thisObject+'.n11.fits'):
       data1 = fits.getdata('./nh3/'+thisObject+'.n11.fits')
       A1 = np.arange(len(data1['DATA'].T))
       nu1 = data1['CDELT1']*(A1-data1['CRPIX1']+1)+data1['CRVAL1']
       v1 = c*(nu1/data1['RESTFREQ']-1)

       # Median filter used for smoothing
       max_index = np.nanargmax(median_filter(data1['DATA'].T,size=value))
       v_medf = median_filter(v1,size=value)
       guess[4] = v_medf[max_index]/1000

       spec1 = psk.Spectrum(data=data1['DATA'].T.squeeze(),unit='K',xarr=v1,xarrkwargs={'unit':'m/s','refX':data1['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spectrum['oneone'] = spec1

    if os.path.exists('./nh3/'+thisObject+'.n22.fits'):
       data2 = fits.getdata('./nh3/'+thisObject+'.n22.fits')
       A2 = np.arange(len(data2['DATA'].T))
       nu2 = data2['CDELT1']*(A2-data2['CRPIX1']+1)+data2['CRVAL1']
       v2 = c*(nu2/data2['RESTFREQ']-1)
       spec2 = psk.Spectrum(data=data2['DATA'].T.squeeze(),unit='K',xarr=v2,xarrkwargs={'unit':'m/s','refX':data2['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spectrum['twotwo'] = spec2

    if os.path.exists('./nh3/'+thisObject+'.n33.fits'):
       data3 = fits.getdata('./nh3/'+thisObject+'.n33.fits')
       A3 = np.arange(len(data3['DATA'].T))
       nu3 = data3['CDELT1']*(A3-data3['CRPIX1']+1)+data3['CRVAL1']
       v3 = c*(nu3/data3['RESTFREQ']-1)
       spec3 = psk.Spectrum(data=data3['DATA'].T.squeeze(),unit='K',xarr=v3,xarrkwargs={'unit':'m/s','refX':data3['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spectrum['threethree'] = spec3

    if os.path.exists('./nh3/'+thisObject+'.n44.fits'):
       data4 = fits.getdata('./nh3/'+thisObject+'.n44.fits')
       A4 = np.arange(len(data4['DATA'].T))
       nu4 = data4['CDELT1']*(A4-data4['CRPIX1']+1)+data4['CRVAL1']
       v4 = c*(nu4/data4['RESTFREQ']-1)
       spec4 = psk.Spectrum(data=data4['DATA'].T.squeeze(),unit='K',xarr=v4,xarrkwargs={'unit':'m/s','refX':data4['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spectrum['fourfour'] = spec4
						
    spdict1,spectra1 = psk.wrappers.fitnh3.fitnh3tkin(spectrum,dobaseline=False,guesses=guess)
    
    # Filters out good and bad fits
    if -150 < spectra1.specfit.modelpars[4] < 150:       
       spec_row = spectra1.specfit.modelpars    	        
       spec_row.insert(0,thisObject)                 	
       t_pars.add_row(spec_row) 
			        
      # Calculate W11, W22, W33, W44
      # Error calculation for W11
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
       t_w11.add_row(W11_row)

       if os.path.exists('./nh3/'+thisObject+'.n44.fits'):
          w_row = [thisObject,np.sum(spec1.specfit.model)*(np.float(4096-0)/np.float(4096)),np.sum(spec2.specfit.model)*(np.float(4096-0)/np.float(4096)),np.sum(spec3.specfit.model)*(np.float(4096-0)/np.float(4096)),np.sum(spec4.specfit.model)*(np.float(4096-0)/np.float(4096))]
          t_int.add_row(w_row)

       # If W44 = 666, then it means N/A cause astropy.table can accept either float/string but not both
       else: 
          w_row = [thisObject,np.sum(spec1.specfit.model)*(np.float(4096-0)/np.float(4096)),np.sum(spec2.specfit.model)*(np.float(4096-0)/np.float(4096)),np.sum(spec3.specfit.model)*(np.float(4096-0)/np.float(4096)),666]
          t_int.add_row(w_row)
          
       plt.savefig(fnameT.format(thisObject), format='png')
       plt.close()

    else:
       plt.savefig(fnameT2.format(thisObject), format='png')
       plt.close()

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
py.hist(t_pars['N(0)'],bins=100)
plt.xlabel('Column Density')
plt.ylabel('Numbers')
plt.title('Histogram of Column Density ($log(N)$)')
plt.savefig('./ammonia_plots/histogram_N.png', format='png')
plt.close()

plt.clf()            
py.hist(t_pars['SIGMA(0)'],bins=100)
plt.xlabel('Line Width ($cm^{-2}$)')
plt.ylabel('Numbers')
plt.title('Histogram of Line Width ($\sigma$)')
plt.savefig('./ammonia_plots/histogram_sigma.png', format='png')
plt.close()

plt.clf()            
py.hist(t_pars['V(0)'],bins=100)
plt.xlabel('Line-of-Sight Velocity (km/s)')
plt.ylabel('Numbers')
plt.title('Histogram of Line-of-Sight Velocity ($v$)')
plt.savefig('./ammonia_plots/histogram_vlos.png', format='png')
plt.close()


# Scatter plots with fit parameters
plt.clf()            
plt.scatter(t_pars['TKIN'],t_pars['SIGMA(0)'])
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
plt.scatter(t_pars['N(0)'],t_pars['TKIN'])
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
   Ma[i] = sigma[i]/(np.float(c_s[i])/1000)

plt.clf()            
plt.scatter(Ma,t_pars['TKIN'])
plt.ylabel('Kinetic Temperature (K)')
plt.xlabel('Mach Number')
plt.title('Mach Number ($Ma$) vs Kinetic Temperature ($T_k$)')
plt.savefig('./ammonia_plots/Ma_vs_tkin.png', format='png')
plt.close()

