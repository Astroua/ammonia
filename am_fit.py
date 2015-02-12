# coding: utf-8
import matplotlib.pyplot as plt

import os
import glob
import numpy as np
#fileNames = glob.glob('./nh3/*fits')
#fileNames = glob.glob('./nh3/GSerpBolo3*.n*.fits')
fileNames = glob.glob('./nh3/G010*.n*.fits')
a = np.arange(len(fileNames))

# This line finds the filenames in the path and drops the last 9 characters from each (the .n??.fits)
objects = [((os.path.basename(fileNames[name])))[0:-9] for name in range(max(a))]

# This line replaces the objects list with a sorted set of the unique items in that list.
objects = sorted(set(objects))
 
# nh3dict creates one large dictionary where each entry corresponds to the spectrum of the (1,1), (2,2), (3,3)
import pyspeckit as psk
from astropy.io import fits
c = 3E8
nh3dict = {}
for thisObject in objects: 
    spect2 = {}
    fnameT = './nh3_figures/'+thisObject+'.png'
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
    nh3dict[thisObject] = spect2
    spdict1,spectra1 = psk.wrappers.fitnh3.fitnh3tkin(spect2,dobaseline=False)
    plt.savefig(fnameT.format(thisObject), format='png')
    plt.close()
    

