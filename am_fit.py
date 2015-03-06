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
	  

Output:  dictionary of the entire spectrum we are working with from nh3dict
	 histogram of kinetic temperatures from htkin
	 table (t) of fit parameters which comes from pyspeckit
	 creates fitted spectra plots and saves it all into a directory

	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻ 	(╯°□°）╯︵ ┻━┻	(╯°□°）╯︵ ┻━┻
"""

#fileNames = glob.glob('./nh3/*fits')
fileNames = glob.glob('./nh3/GSerpBolo3*.n*.fits')
#fileNames = glob.glob('./nh3/G010*.n*.fits')

a = np.arange(len(fileNames))
objects = [((os.path.basename(fileNames[name])))[0:-9] for name in range(max(a))]
objects = sorted(set(objects))

nh3dict = {}

t = Table(names=('FILENAME','TKIN','TEX','N(0)','SIGMA(0)','V(0)','F_0(0)'),dtype=('S20','f5','f5','f5','f5','f5','f1'))

h_tkin = []
fnameh = './hist_figs/histogram_tkin.png'

# This creates the dictionary to then pass to pyspeckit to create the fit
# value is the pixel size for filtering with median_filter
value = 17
c = 3E8
for thisObject in objects: 
    spectrum = {}
    fnameT = './nh3_figures/'+thisObject+'.png'
    guess = [15, 7, 15, 2, 30, 0]

    if os.path.exists('./nh3/'+thisObject+'.n11.fits'):
       data1 = fits.getdata('./nh3/'+thisObject+'.n11.fits')
       A1 = np.arange(len(data1['DATA'].T))
       nu1 = data1['CDELT1']*(A1-data1['CRPIX1']+1)+data1['CRVAL1']
       v1 = c*(nu1/data1['RESTFREQ']-1)

       # Median filter used for smoothing
       t_medf = median_filter(data1['DATA'].T,size=value)[~np.isnan(median_filter(data1['DATA'].T,size=value))]
       a_medf = np.arange(len(t_medf))
       nu_med = data1['CDELT1']*(a_medf-data1['CRPIX1']+1)+data1['CRVAL1']
       v_dmed = c*(nu_med/data1['RESTFREQ']-1)
       max_index = np.argmax(t_medf)    
       v_medf = median_filter(v_dmed,size=value)
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
    # modelpars extracts the fit parameters created from pyspeckit where: [Tk Tex N sigma v F]
    spec_row = spectra1.specfit.modelpars    	        
    spec_row.insert(0,thisObject)                 	
    t.add_row(spec_row) 			        
    h_tkin.append(spec_row[1])                          
    plt.savefig(fnameT.format(thisObject), format='png')
    plt.close()

# to filter out bad fits if needed to in the future? [tentative]:
# if spectra1.specfit.modelpars[4] > spectra1.specfit.modelerrs[4]:
#    if spectra1.specfit.modelerrs[4] ~== 0:
#      ... [ make plots as per usual ] 
    

# Creates the histogram
plt.clf()            
py.hist(h_tkin,bins=100)
plt.xlabel('Kinetic Temperature (K)')
plt.ylabel('Numbers')
plt.title('Histogram of Kinetic Temperatures (T_k) of the Spectral Data')
plt.savefig(fnameh, format='png')
plt.close()

