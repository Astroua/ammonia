import numpy.fft as fft
import numpy as np
def NH3FirstGuess(pskobj,vmin=-250.0,vmax=250):
  
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

  vaxis = np.array(pskobj.xarr.as_unit('km/s'))
  subsetidx = (vaxis>vmin)*(vaxis<vmax)*np.isfinite(np.array(pskobj.data))
  peakIndex = np.argmax(ccor[subsetidx])
  
  #pull out a 6 km/s slice around the peak
  deltachan = np.abs(3.0 / chanwidth)
  t = ((pskobj.data.filled(0))[subsetidx])[(peakIndex-deltachan):(peakIndex+deltachan)]
  v = (vaxis[subsetidx])[(peakIndex-deltachan):(peakIndex+deltachan)]
  # Calculate line width.
  sigv = np.sqrt(abs(np.sum(t*v**2)/np.sum(t)-(np.sum(t*v)/np.sum(t))**2))

  if (np.isnan(sigv)):
    sigv = 0.85840189 # mean of sigv from first set of data from ./nh3_all/

  # Peak of cross correlation is the brightness.

  v0 = np.float((vaxis[subsetidx])[peakIndex])
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
    
