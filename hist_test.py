# coding: utf-8
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import pyspeckit as psk
import pylab as py

# use glob to pull .fits files and sort them
#specg = glob.glob('./nh3/G02*.n*.fits')
specg = glob.glob('./nh3/GSerpBolo3*.n*.fits')
specg.sort()

a = len(specg)

# then throw it into a dictionary, which is now sorted
globd = {i : specg[i] for i in range(a)}

spect = {i: fits.getdata(globd[i])['DATA'].T[~np.isnan(fits.getdata(globd[i])['DATA'].T)] for i in range(a)}

A = np.arange(a)
fnameT = "./hist_figs/plot{0:07d}.png"
for i in range(min(A),max(A)):
	py.hist(spect[i],bins=1000)
	plt.xlabel('Antennae Temperature (K)')
	plt.title('Histogram of ".fits"')
	plt.savefig(fnameT.format(i), format='png')
	plt.close()




