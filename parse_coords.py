from astropy.coordinates import SkyCoord

def parse_coords(s):
    if 'RA' in s['CTYPE2']:
        frame = 'fk5'
    if 'GLON' in s['CTYPE2']:
        frame = 'galactic'
    skyc = SkyCoord(s['CRVAL2'],s['CRVAL3'],unit='deg',frame=frame)
    glon = (skyc.galactic.l.degree)[0]
    glat = (skyc.galactic.b.degree)[0]
    return(glon,glat)
