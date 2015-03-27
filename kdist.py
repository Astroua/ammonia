from astropy import coordinates as coords
import astropy.units as u
from numpy import sqrt, abs, pi, cos, sin, max, ones, array
from warnings import warn
import numpy as np

def kdist(l, b, vin, near=True,r0=8.4e3,v0=2.54e2,
          dynamical=False, kinematic=True, regular=False, rrgal=False,
          verbose=False, inverse=False, silent=False, returnvtan=False):
    """
    Return the distance to an object given its Galactic longitude, latitude,
    and LSR velocity assuming a uniform rotation curve
    
    Parameters
    ----------
    l, b: float,float
        Galactic Longitude and Latitude (decimal degrees)
        (can also input astropy degrees and vectors)
    v: float
        Velocity w.r.t. LSR in km/s
    near: bool
        Return the near kinematic distance if set, otherwise return the far
        kinematic distance.  The "Kinematic Distance Ambiguity" (i.e., near/far
        for one velocity) only exists for quadrants 1 & 4 (-90<l<90)
    rO, vO: float,float
        Values for galactocentric distance for sun and velocity of the LSR
        around the GC.  Default to 8.4 kpc and 254 km/s (Reid et al., 2009)
    rrgal: bool
        return galactocentric distance in addition to distance from us
    dynamical: bool
        Use the dynamical definition of the LSR
    kinematic: bool
        Use the kinematic definition of the LSR (default)
    regular: bool
        Do not apply the rotation correction for High mass star forming
        regions.
    inverse: bool
        If set, pass DISTANCE instead of velocity, and output is velocity
    returnvtan: bool
        if set, return the tanent velocity and ignore the input velocity
    
    Returns
    -------
    The kinematic distance in the same units as R0 (defaults to pc).  However,
    the boolean parameters inverse and returnvtan and rrgal all change the
    return values.
    """

    if regular:
        vs = 0.0 
    else:
        vs=15.0

    if kinematic or not(dynamical):
        solarmotion_ra = ((18+03/6e1+50.29/3.6e3)*15)
        solarmotion_dec = (30+0/6e1+16.8/3.6e3)
        solarmotion_mag = 20.0
    else:
        solarmotion_ra = ((17+49/6e1+58.667/3.6e3)*15)
        solarmotion_dec = (28+7/6e1+3.96/3.6e3)
        solarmotion_mag = 16.55294

    if not hasattr(l,'unit') or not hasattr(b,'unit'):
        l = np.array(l)*u.deg
        b = np.array(b)*u.deg
    cg = coords.Galactic(l, b)
    solarmotion = coords.ICRS(solarmotion_ra*u.deg,solarmotion_dec*u.deg)
    #  ra,dec = cg.j2000()
    #  gcirc, 2, solarmotion_ra, solarmotion_dec, ra, dec, theta
    theta = solarmotion.separation(cg).to(u.radian).value

    vhelio = vin-solarmotion_mag*cos(theta)

    # UVW from Dehnen and Binney
    bigu = 10.0
    bigv = 5.23
    bigw = 7.17

    lrad,brad = l.to(u.radian).value, b.to(u.radian).value
    ldeg,bdeg = l.to(u.degree).value, b.to(u.degree).value

    v = vhelio+(bigu*cos(lrad)+bigv*sin(lrad))*cos(brad)+bigw*sin(brad)

    # Compute tangent distance and velocity
    rtan = r0*(cos(lrad))/(cos(brad))
    vTEMP = (1/sin(lrad) - v0/(v0-vs)) * ((v0-vs)*sin(lrad)*cos(brad))
    vhelioTEMP = vTEMP - ((bigu*cos(lrad)+bigv*sin(lrad))*cos(brad)+bigw*sin(brad))
    vtan = vhelioTEMP+solarmotion_mag*cos(theta)
    if returnvtan:
        return vtan

    # This is r/r0
    null = (v0/(v0-vs)+v/((v0-vs)*sin(lrad)*cos(brad)))**(-1)

    if inverse:
        radical = cos(lrad) - cos(brad) * vin / r0 
        null = sqrt(1 - cos(lrad)**2 + radical**2)
        v = (1/null - v0/(v0-vs)) * ((v0-vs)*sin(lrad)*cos(brad))
        vhelio = v - ((bigu*cos(lrad)+bigv*sin(lrad))*cos(brad)+bigw*sin(brad))
        vlsr = vhelio+solarmotion_mag*cos(theta)
        return vlsr
    else:
        #  The > 0 traps things near the tangent point and sets them to the
        #  tangent distance.  So quietly.  Perhaps this should pitch a flag?
        radical = sqrt(((cos(lrad))**2-(1-null**2)))
        if np.isscalar(radical):
            radical = max(radical,0)
        else:
            radical[radical<0] = 0

        fardist = r0*(cos(lrad)+radical)/(cos(brad))

        neardist = r0*(cos(lrad)-radical)/(cos(brad))

    rgal = null*r0
    ind = (abs(ldeg-180) < 90)
    if ind.sum() > 1:
        neardist[ind] = fardist[ind]
    elif np.isscalar(ind) and ind==True:
        neardist = fardist

    if not(near):
        dist = fardist
    else:
        dist = neardist

    if np.any(vin > vtan):
        if not silent:
            warn("Velocity is greater than tangent velocity.  Returning tangent distance.")
        if np.isscalar(dist):
            dist = rtan
        else:
            dist[vin>vtan] = rtan[vin>vtan]

    if verbose:
        print "radical: %f  null: %f  vin: %f  v: %f  vhelio: %f rgal: %f  neardist: %f  fardist: %f" % (radical,null,vin,v,vhelio,rgal,neardist,fardist)

    if rrgal:
        if np.any(vin>vtan):
            dist[vin>vtan] = rtan[vin>vtan]
            rgal[vin>vtan] = (null*r0)[vin>vtan]
        return abs(dist),abs(rgal)
    return abs(dist)

def vector_kdist(x,y,z,**kwargs):
    """ obsolete """

    if type(z)==type(1) or type(z)==type(1.0):
        z = z*ones(len(x))
    v = []
    for i,j,k in array([x,y,z]).T:
        v.append( kdist(i,j,k,**kwargs) )
    return array(v)

def threekpcarm(longitude,radius=3.0,center_distance=8.5):
    return sqrt(radius**2+center_distance**2-2*radius*center_distance*cos( (90-3*longitude) / 180. * pi ))
