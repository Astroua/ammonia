import numpy.fft as fft
import numpy as np
def NH3FirstGuess(pskobj):
  
  ckms = 2.99792458e5
  
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

  deltanu = -1*voff_lines/ckms*23.6944955e9

  #guess at typical line width for NH3 line
  linewidth = 0.5
  chanwidth = (pskobj.xarr[1]-pskobj.xarr[0])/1e3
  ftdata = fft.fft(pskobj.data.filled(0))
  tvals = fft.fftfreq(len(pskobj.data))/chanwidth
  deltafcns = np.zeros(pskobj.data.shape,dtype=np.complex)
  for idx, dv in enumerate(voff_lines):
    deltafcns += tau_wts[idx]*(np.cos(2*np.pi*dv*tvals)+
                               1j*np.sin(2*np.pi*dv*tvals))*\
                               np.exp(-tvals**2*(linewidth/chanwidth)**2/(2))
  ccor = np.real((fft.ifft(np.conj(ftdata)*deltafcns))[::-1])

  peakIndex = np.argmax(ccor)

  #pull out a 6 km/s slice around the peak
  deltachan = 6.0 / chanwidth
  t = (pskobj.data.filled(0))[(peakIndex-deltachan):(peakIndex+deltachan)]
  v = np.array(pskobj.xarr)[(peakIndex-deltachan):(peakIndex+deltachan)]
  # Calculate line widht.
  sigv = np.sqrt(np.sum(t*v**2)/np.sum(t)-(np.sum(t*v)/np.sum(t))**2)/1e3


  # Peak of cross correlation is the brightness.

  v0 = np.float(pskobj.xarr[peakIndex])/1e3
  # Set the excitation temperature to be between CMB and 20 K
  # and equal to the peak brightness + 2.73 if valid.
  tex = np.min([np.max([pskobj.data[peakIndex],0])+2.73,20])

  guess = [20, # 20 K kinetic temperature
           tex,  #
           15, # Log NH3
           sigv, # velocity dispersion
           v0,
           0.5]
  return(guess)
    
