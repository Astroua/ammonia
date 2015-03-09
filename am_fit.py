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
	  

Output:  Arrays of the fit parameters:
		tkin = Kinetic temperature
		tex = Excitation temperature
		N = Column density
		sigma = Line width/Velocity dispersion
		v_los = Apparent line of sight velocity

	 nh3dict = Dictionary of the entire spectrum
	 Histogram of kinetic temperatures from tkin
	 t = table of fit parameters
	 creates fitted spectra plots and saves it all into a directory

	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻ 	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻
"""

"""
fileNames = glob.glob('./nh3/*fits')
#fileNames = glob.glob('./nh3/GSerpBolo2*.n*.fits')
#fileNames = glob.glob('./nh3/G015*.n*.fits')

a = np.arange(len(fileNames))
objects = [((os.path.basename(fileNames[name])))[0:-9] for name in range(max(a))]
objects = sorted(set(objects))

nh3dict = {}

t = Table(names=('FILENAME','TKIN','TEX','N(0)','SIGMA(0)','V(0)','F_0(0)'),dtype=('S20','f5','f5','f5','f5','f5','f1'))

# Fit parameters
tkin = []
tex = []
N = []
sigma = []
v_los = []

# This creates the dictionary to then pass to pyspeckit to create the fit
# value is the pixel size for filtering with median_filter
value = 17
c = 3E8

for thisObject in objects: 
    spectrum = {}
    fnameT = './nh3_figures/'+thisObject+'.png'
    fnameT2 = './nh3_figures2/'+thisObject+'.png'
    guess = [15, 7, 15, 2, 30, 0]

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
			
    nh3dict[thisObject] = spectrum			
    spdict1,spectra1 = psk.wrappers.fitnh3.fitnh3tkin(spectrum,dobaseline=False,guesses=guess)
    # Filters out good and bad fits
    if -200 < spectra1.specfit.modelpars[4] < 200:
       tkin.append(spectra1.specfit.modelpars[0])
       tex.append(spectra1.specfit.modelpars[1])
       N.append(spectra1.specfit.modelpars[2])
       sigma.append(spectra1.specfit.modelpars[3])
       v_los.append(spectra1.specfit.modelpars[4])
       
       spec_row = spectra1.specfit.modelpars    	        
       spec_row.insert(0,thisObject)                 	
       t.add_row(spec_row) 			        
                                
       plt.savefig(fnameT.format(thisObject), format='png')
       plt.close()
    else:
       plt.savefig(fnameT2.format(thisObject), format='png')
       plt.close()
"""

# Fit parameter histograms
plt.clf()            
py.hist(tkin,bins=100)
plt.xlabel('Kinetic Temperature (K)')
plt.ylabel('Numbers')
plt.title('Histogram of Kinetic Temperatures ($T_k$)')
plt.savefig('./ammonia_plots/histogram_tkin.png', format='png')
plt.close()

plt.clf()            
py.hist(tex,bins=100)
plt.xlabel('Excitation Temperature (K)')
plt.ylabel('Numbers')
plt.title('Histogram of Excitation Temperatures ($T_{ex}$)')
plt.savefig('./ammonia_plots/histogram_tex.png', format='png')
plt.close()

plt.clf()            
py.hist(N,bins=100)
plt.xlabel('Column Density')
plt.ylabel('Numbers')
plt.title('Histogram of Column Density ($log(N)$)')
plt.savefig('./ammonia_plots/histogram_N.png', format='png')
plt.close()

plt.clf()            
py.hist(sigma,bins=100)
plt.xlabel('Line Width ($cm^{-2}$)')
plt.ylabel('Numbers')
plt.title('Histogram of Line Width ($\sigma$)')
plt.savefig('./ammonia_plots/histogram_sigma.png', format='png')
plt.close()

plt.clf()            
py.hist(v_los,bins=100)
plt.xlabel('Line-of-Sight Velocity (km/s)')
plt.ylabel('Numbers')
plt.title('Histogram of Line-of-Sight Velocity ($v$)')
plt.savefig('./ammonia_plots/histogram_vlos.png', format='png')
plt.close()

# Scatter plots with fit parameters
plt.clf()            
plt.scatter(tkin,sigma)
plt.xlabel('Kinetic Temperature (K)')
plt.ylabel('Line Width ($cm^{-2}$)')
plt.title('Kinetic Temperature ($T_k$) and Line Width ($\sigma$)')
plt.savefig('./ammonia_plots/tkin_vs_sigma.png', format='png')
plt.close()

plt.clf()            
plt.scatter(tkin,tex)
plt.xlabel('Kinetic Temperature (K)')
plt.ylabel('Excitation Temperature (K)')
plt.title('Kinetic Temperature ($T_k$) vs Excitation Temperature ($T_{ex}$)')
plt.savefig('./ammonia_plots/tkin_tex.png', format='png')
plt.close()

plt.clf()            
plt.scatter(N,tkin)
plt.ylabel('Kinetic Temperature (K)')
plt.xlabel('Column Density (log(N))')
plt.title('Column Density ($log(N)$) vs Kinetic Temperature ($T_k$)')
plt.savefig('./ammonia_plots/N_vs_tkin.png', format='png')
plt.close()

# assume gamma = 1 cause its isothermal
kb = 1.38E-23
m = 2.82E-26
cs = np.arange(len(tkin),dtype = np.float64)
Ma = np.arange(len(tkin),dtype = np.float64)
for i in range(0,len(tkin)):
   cs[i] = math.sqrt(kb*tkin[i]/m)
   Ma[i] = sigma[i]/(np.float(cs[i])/1000)

plt.clf()            
plt.scatter(Ma,tkin)
plt.ylabel('Kinetic Temperature (K)')
plt.xlabel('Mach Number')
plt.title('Mach Number ($Ma$) vs Kinetic Temperature ($T_k$)')
plt.savefig('./ammonia_plots/Ma_vs_tkin.png', format='png')
plt.close()

