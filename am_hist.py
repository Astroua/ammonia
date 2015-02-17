# coding: utf-8
import os
import glob
import numpy as np
import pylab as py
import pyspeckit as psk
from astropy.io import fits
import matplotlib.pyplot as plt

#fileNames = glob.glob('./nh3/*fits')
#fileNames = glob.glob('./nh3/GSerpBolo3*.n*.fits')
fileNames = glob.glob('./nh3/G010*.n*.fits')
a = np.arange(len(fileNames))

objects = [((os.path.basename(fileNames[name])))[0:-9] for name in range(max(a))]
objects = sorted(set(objects))

c = 3E8    
fnameT = './hist_figs/histogram_tkin.png'
# creates an empty array, to store tkin values into for histogram
htkin = []

for thisObject in objects: 
    spect2 = {}
    if os.path.exists('./nh3/'+thisObject+'.n11.fits'):
       data1 = fits.getdata('./nh3/'+thisObject+'.n11.fits')
       A1 = np.arange(len(data1['DATA'].T))
       nu1 = data1['CDELT1']*(A1-data1['CRPIX1']+1)+data1['CRVAL1']
       v1 = c*(nu1/data1['RESTFREQ']-1)
       spec11 = psk.Spectrum(data=(data1['DATA'].T).squeeze(),xarr=v1,xarrkwargs={'unit':'m/s','refX':data1['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['oneone'] = spec11
    if os.path.exists('./nh3/'+thisObject+'.n22.fits'):
       data2 = fits.getdata('./nh3/'+thisObject+'.n22.fits')
       A2 = np.arange(len(data2['DATA'].T))
       nu2 = data2['CDELT1']*(A2 - data2['CRPIX1'] + 1) + data2['CRVAL1']
       v2 = c*(nu2/data2['RESTFREQ']-1)
       spec22 = psk.Spectrum(data=(data2['DATA'].T).squeeze(),xarr=v2,xarrkwargs={'unit':'m/s','refX':data2['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['twotwo'] = spec22
    if os.path.exists('./nh3/'+thisObject+'.n33.fits'):
       data3 = fits.getdata('./nh3/'+thisObject+'.n33.fits')
       A3 = np.arange(len(data3['DATA'].T))
       nu3 = data3['CDELT1']*(A3 - data3['CRPIX1'] + 1) + data3['CRVAL1']
       v3 = c*(nu3/data3['RESTFREQ']-1)
       spec33 = psk.Spectrum(data=(data3['DATA'].T).squeeze(),xarr=v3,xarrkwargs={'unit':'m/s','refX':data3['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['threethree'] = spec33
    if os.path.exists('./nh3/'+thisObject+'.n44.fits'):
       data4 = fits.getdata('./nh3/'+thisObject+'.n44.fits')
       A4 = np.arange(len(data4['DATA'].T))
       nu4 = data4['CDELT1']*(a - data4['CRPIX1'] + 1) + data4['CRVAL1']
       v4 = c*(nu4/data4['RESTFREQ']-1)
       spec44 = psk.Spectrum(data=(data4['DATA'].T).squeeze(),xarr=v4,xarrkwargs={'unit':'m/s','refX':data4['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['fourfour'] = spec44
    spdict1,spectra1 = psk.wrappers.fitnh3.fitnh3tkin(spect2,dobaseline=False)
    fitp = spectra1.specfit.modelpars
    # the first value in the array is tkin, takes it and stores it into htkin
    htkin.append(fitp[0]) 


# this creates the histogram
plt.clf()
py.hist(htkin,bins=30)
plt.xlabel('Kinetic Temperature (K)')
plt.title('Histogram of T_k of all .fits Files')
plt.savefig(fnameT, format='png')
plt.close()


