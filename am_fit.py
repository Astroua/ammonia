# coding: utf-8
import os
import glob
import numpy as np
import pylab as py
import pyspeckit as psk
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.ndimage.filters import median_filter

# Using glob we find all of the .fits file within a dicrectory
#fileNames = glob.glob('./nh3/*fits')
fileNames = glob.glob('./nh3/GSerpBolo3*.n*.fits')
#fileNames = glob.glob('./nh3/G010*.n*.fits')
a = np.arange(len(fileNames))

objects = [((os.path.basename(fileNames[name])))[0:-9] for name in range(max(a))]
objects = sorted(set(objects))

c = 3E8
# This creates an empty dictionary to store all of the spectra in the for loop
nh3dict = {}

# This creates a am empty table to insert the fit parameters created from the for loop
t = Table(names=('FILENAME','TKIN','TEX','N(0)','SIGMA(0)','V(0)','F_0(0)'),dtype=('S20','f5','f5','f5','f5','f5','f1'))

# Creates an empty array to insert kinetic temperatures for histogram
htkin = []
fnameh = './hist_figs/histogram_tkin.png'


# value adjusts the size for median_filter
value = 17

# uses median_filter to fit on top of spectra
"""
for thisObject in objects: 
    spect2 = {}
    fnameT = './nh3_figures2/'+thisObject+'.png'
    if os.path.exists('./nh3/'+thisObject+'.n11.fits'):
       data1 = fits.getdata('./nh3/'+thisObject+'.n11.fits')
       A1 = np.arange(len(data1['DATA'].T))
       nu1 = data1['CDELT1']*(A1-data1['CRPIX1']+1)+data1['CRVAL1']
       v1 = c*(nu1/data1['RESTFREQ']-1)
       t11 = median_filter(data1['DATA'].T,size=value)
       spec11 = psk.Spectrum(data=(data1['DATA'].T).squeeze(),unit='K',xarr=v1,xarrkwargs={'unit':'m/s','refX':data1['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['oneone'] = spec11
    if os.path.exists('./nh3/'+thisObject+'.n22.fits'):
       data2 = fits.getdata('./nh3/'+thisObject+'.n22.fits')
       A2 = np.arange(len(data2['DATA'].T))
       nu2 = data2['CDELT1']*(A2-data2['CRPIX1']+1)+data2['CRVAL1']
       v2 = c*(nu2/data2['RESTFREQ']-1)
       t22 = median_filter(data2['DATA'].T,size=value)
       spec22 = psk.Spectrum(data=(data2['DATA'].T).squeeze(),unit='K',xarr=v2,xarrkwargs={'unit':'m/s','refX':data2['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})      
       spect2['twotwo'] = spec22
    if os.path.exists('./nh3/'+thisObject+'.n33.fits'):
       data3 = fits.getdata('./nh3/'+thisObject+'.n33.fits')
       A3 = np.arange(len(data3['DATA'].T))
       nu3 = data3['CDELT1']*(A3-data3['CRPIX1']+1)+data3['CRVAL1']
       v3 = c*(nu3/data3['RESTFREQ']-1)  
       t33 = median_filter(data3['DATA'].T,size=value)
       spec33 = psk.Spectrum(data=(data3['DATA'].T).squeeze(),unit='K',xarr=v3,xarrkwargs={'unit':'m/s','refX':data3['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['threethree'] = spec33
    if os.path.exists('./nh3/'+thisObject+'.n44.fits'):
       data4 = fits.getdata('./nh3/'+thisObject+'.n44.fits')
       A4 = np.arange(len(data4['DATA'].T))
       nu4 = data4['CDELT1']*(A4-data4['CRPIX1']+1)+data4['CRVAL1']
       v4 = c*(nu4/data4['RESTFREQ']-1)
       t44 = median_filter(data4['DATA'].T,size=value)
       spec44 = psk.Spectrum(data=(data4['DATA'].T).squeeze(),unit='K',xarr=v4,xarrkwargs={'unit':'m/s','refX':data4['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['fourfour'] = spec44
    if len(spect2) == 3:
       plt.figure(1)
       plt.subplot(211)
       plt.plot(v1,data1['DATA'].T,'k',v1,t11,'r')
       plt.subplot(223)
       plt.plot(v2,data2['DATA'].T,'k',v2,t22,'r')
       plt.subplot(224)
       plt.plot(v3,data3['DATA'].T,'k',v3,t33,'r')
    elif len(spect2) == 4:
       plt.figure(1)
       plt.subplot(221)
       plt.plot(v1,data1['DATA'].T,'k',v1,t11,'r')
       plt.subplot(222)
       plt.plot(v2,data2['DATA'].T,'k',v2,t22,'r')
       plt.subplot(223)
       plt.plot(v3,data3['DATA'].T,'k',v3,t33,'r')
       plt.subplot(224)
       plt.plot(v4,data4['DATA'].T,'k',v4,t44,'r')
    plt.savefig(fnameT.format(thisObject), format='png')
    plt.close()   

"""


# attempt to use median_filter with pyspeckit
for thisObject in objects: 
    spect2 = {}
    fnameT = './nh3_figures/'+thisObject+'.png'
    if os.path.exists('./nh3/'+thisObject+'.n11.fits'):
       data1 = fits.getdata('./nh3/'+thisObject+'.n11.fits')
       A1 = np.arange(len(data1['DATA'].T))
       nu1 = data1['CDELT1']*(A1-data1['CRPIX1']+1)+data1['CRVAL1']
       v1 = c*(nu1/data1['RESTFREQ']-1)
       t11 = median_filter(data1['DATA'].T,size=value)
       v11 = median_filter(v1,size=value)
       spec11 = psk.Spectrum(data=data1['DATA'].T.squeeze(),unit='K',xarr=v11,xarrkwargs={'unit':'m/s','refX':data1['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['oneone'] = spec11
    if os.path.exists('./nh3/'+thisObject+'.n22.fits'):
       data2 = fits.getdata('./nh3/'+thisObject+'.n22.fits')
       A2 = np.arange(len(data2['DATA'].T))
       nu2 = data2['CDELT1']*(A2-data2['CRPIX1']+1)+data2['CRVAL1']
       v2 = c*(nu2/data2['RESTFREQ']-1)
       t22 = median_filter(data2['DATA'].T,size=value)
       v22 = median_filter(v2,size=value)
       spec22 = psk.Spectrum(data=data2['DATA'].T.squeeze(),unit='K',xarr=v22,xarrkwargs={'unit':'m/s','refX':data2['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['twotwo'] = spec22
    if os.path.exists('./nh3/'+thisObject+'.n33.fits'):
       data3 = fits.getdata('./nh3/'+thisObject+'.n33.fits')
       A3 = np.arange(len(data3['DATA'].T))
       nu3 = data3['CDELT1']*(A3-data3['CRPIX1']+1)+data3['CRVAL1']
       v3 = c*(nu3/data3['RESTFREQ']-1)
       t33 = median_filter(data3['DATA'].T,size=value)
       v33 = median_filter(v3,size=value)
       spec33 = psk.Spectrum(data=data3['DATA'].T.squeeze(),unit='K',xarr=v33,xarrkwargs={'unit':'m/s','refX':data3['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['threethree'] = spec33
    if os.path.exists('./nh3/'+thisObject+'.n44.fits'):
       data4 = fits.getdata('./nh3/'+thisObject+'.n44.fits')
       A4 = np.arange(len(data4['DATA'].T))
       nu4 = data4['CDELT1']*(A4-data4['CRPIX1']+1)+data4['CRVAL1']
       v4 = c*(nu4/data4['RESTFREQ']-1)
       t44 = median_filter(data4['DATA'].T,size=value)
       v44 = median_filter(v4,size=value)
       spec44 = psk.Spectrum(data=data4['DATA'].T.squeeze(),unit='K',xarr=v44,xarrkwargs={'unit':'m/s','refX':data4['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['fourfour'] = spec44
    nh3dict[thisObject] = spect2
    spdict1,spectra1 = psk.wrappers.fitnh3.fitnh3tkin(spect2,dobaseline=False)
    spec_row = spectra1.specfit.modelpars    	        # modelpars grabs the fit parameters
    spec_row.insert(0,thisObject)                 	# inserts the string name into spec_row
    t.add_row(spec_row) 			        # adds the whole row into t
    htkin.append(spec_row[1])                          
    plt.savefig(fnameT.format(thisObject), format='png')
    plt.close()

    


# this creates a histogram and saves it
plt.clf()            
py.hist(htkin,bins=100)
plt.xlabel('Kinetic Temperature (K)')
plt.ylabel('Numbers')
plt.title('Histogram of Kinetic Temperatures (T_k) of the Spectral Data')
plt.savefig(fnameh, format='png')
plt.close()


