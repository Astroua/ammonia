# coding: utf-8
from astropy.table import Table
from astropy.io import fits
import numpy as np
import glob
import os
import pyspeckit as psk
from astropy.io import fits
import matplotlib.pyplot as plt

fileNames = glob.glob('./nh3/G010*.n*.fits')
#fileNames = glob.glob('./nh3/GSerpBolo3*.n*.fits')
a = np.arange(len(fileNames))
objects = [((os.path.basename(fileNames[name])))[0:-9] for name in range(max(a))]
objects = sorted(set(objects))

# Similar to am_fit.py use a for loop to create each dictionary then make the row to add to the table

c = 3E8
# this creates the table with all the labeled columns
t = Table(names=('FILENAME','TKIN','TEX','N(0)','SIGMA(0)','V(0)','F_0(0)'),dtype=('S20','f5','f5','f5','f5','f5','f1'))

for thisFile in objects: 
    spect2 = {}
    if os.path.exists('./nh3/'+thisFile+'.n11.fits'):
       data1 = fits.getdata('./nh3/'+thisFile+'.n11.fits')
       A1 = np.arange(len(data1['DATA'].T))
       nu1 = data1['CDELT1']*(A1-data1['CRPIX1']+1)+data1['CRVAL1']
       v1 = c*(nu1/data1['RESTFREQ']-1)
       spec11 = psk.Spectrum(data=(data1['DATA'].T).squeeze(),xarr=v1,xarrkwargs={'unit':'m/s','refX':data1['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['oneone'] = spec11
    if os.path.exists('./nh3/'+thisFile+'.n22.fits'):
       data2 = fits.getdata('./nh3/'+thisFile+'.n22.fits')
       A2 = np.arange(len(data2['DATA'].T))
       nu2 = data2['CDELT1']*(A2 - data2['CRPIX1'] + 1) + data2['CRVAL1']
       v2 = c*(nu2/data2['RESTFREQ']-1)
       spec22 = psk.Spectrum(data=(data2['DATA'].T).squeeze(),xarr=v2,xarrkwargs={'unit':'m/s','refX':data2['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['twotwo'] = spec22
    if os.path.exists('./nh3/'+thisFile+'.n33.fits'):
       data3 = fits.getdata('./nh3/'+thisFile+'.n33.fits')
       A3 = np.arange(len(data3['DATA'].T))
       nu3 = data3['CDELT1']*(A3 - data3['CRPIX1'] + 1) + data3['CRVAL1']
       v3 = c*(nu3/data3['RESTFREQ']-1)
       spec33 = psk.Spectrum(data=(data3['DATA'].T).squeeze(),xarr=v3,xarrkwargs={'unit':'m/s','refX':data3['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['threethree'] = spec33
    if os.path.exists('./nh3/'+thisFile+'.n44.fits'):
       data4 = fits.getdata('./nh3/'+thisFile+'.n44.fits')
       A4 = np.arange(len(data4['DATA'].T))
       nu4 = data4['CDELT1']*(a - data4['CRPIX1'] + 1) + data4['CRVAL1']
       v4 = c*(nu4/data4['RESTFREQ']-1)
       spec44 = psk.Spectrum(data=(data4['DATA'].T).squeeze(),xarr=v4,xarrkwargs={'unit':'m/s','refX':data4['RESTFREQ']/1E6,'refX_units':'MHz','xtype':'VLSR-RAD'})
       spect2['fourfour'] = spec44
    spdict1,spectra1 = psk.wrappers.fitnh3.fitnh3tkin(spect2,dobaseline=False)
    sprow = spectra1.specfit.modelpars
    # insert takes 'thisFile' and inserts it into sprow, to which we can use add_row
    sprow.insert(0,thisFile)
    t.add_row(sprow)



