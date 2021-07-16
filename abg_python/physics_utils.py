import numpy as np 
from scipy.spatial.distance import cdist as cdist

def getSpeedOfSound(U_code):
    """U_code = snapdict['InternalEnergy'] INTERNAL ENERGY_code = VELOCITY_code^2 = (params.txt default = (km/s)^2)
        mu = mean molecular weight in this context
        c_s = sqrt(gamma kB T / mu)

        n kB T = (gamma - 1) U rho
        (gamma -1 ) U = kB T/mu = c_s^2/gamma
        c_s = sqrt(5/3 * 2/3 * U)
        """
    return np.sqrt(10/9.*U_code)

def getTemperature(
    U_code,
    helium_mass_fraction=None,
    ElectronAbundance=None,
    mu = None):
    """U_code = snapdict['InternalEnergy'] INTERNAL ENERGY_code = VELOCITY_code^2 = (params.txt default = (km/s)^2)
    helium_mass_fraction = snapdict['Metallicity'][:,1]
    ElectronAbundance= snapdict['ElectronAbundance']"""
    U_cgs = U_code*1e10 ## to convert from (km/s)^2 -> (cm/s)^2
    gamma=5/3.
    kB=1.38e-16 #erg /K
    m_proton=1.67e-24 # g
    if mu is None:
        ## not provided from chimes, hopefully you sent me helium_mass_fraction and
        ##  electron abundance!
        try: 
            assert helium_mass_fraction is not None
            assert ElectronAbundance is not None
        except AssertionError:
            raise ValueError(
                "You need to either provide mu or send helium mass fractions and electron abundances to calculate it!")
        y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction)) ## is this really Y -> Y/4X
        mu = (1.0 + 4*y_helium) / (1+y_helium+ElectronAbundance) 
    mean_molecular_weight=mu*m_proton
    return mean_molecular_weight * (gamma-1) * U_cgs / kB

def get_IMass(age,mass,apply_factor=False):
    ## age must be in Gyr
    ## based off mass loss in gizmo plot, averaged over 68 stars
    b = 0.587
    a = -8.26e-2
    factors = b*age**a 
    factors[factors > 1]=1
    factors[factors < 0.76]=0.76
    if not apply_factor:
        return mass/factors
    else:
        return mass*factors

def getBolometricLuminosity(ageGyrs,masses):
    ## convert to Myr
    ageMyrs = ageGyrs*1000 

    ## initialize luminosity array
    lums = np.zeros(ageMyrs.size)
    lums[ageMyrs<3.5] = 1136.59 ## Lsun/Msun

    x = np.log10(ageMyrs[ageMyrs>=3.5]/3.5)
    lums[ageMyrs>=3.5] = 1500*np.exp(-4.145*x + 0.691*x**2 - 0.0576*x**3) ## Lsun/Msun
    ##  3.83e33 erg/s / 2e33 g =  1.915 cm^2 /s^2
    lums *= 1.915 ## erg/s / g = cm^2/s^2 

    return lums*masses ## erg/s code_mass/g, typically, or erg/s if masses is in g

def getLuminosityBands(ageGyrs,masses):
    """
    Then the bolometric Ψbol = 1136.59 for tMyr < 3.5, and 
    Ψbol = 1500 exp[−4.145x+0.691x2 −0.0576x3] 
    with x ≡ log10(tMyr/3.5) for tMyr > 3.5. 
    For the bands used in our radiation hydrodynamics, we have the following 
    intrinsic (before attenuation) bolometric corrections. 

    In the mid/far IR, ΨIR = 0. 

    In optical/NIR, Ψopt = fopt Ψbol with fopt = 0.09 for tMyr < 2.5; 
    fOpt = 0.09(1 + [(tMyr − 2.5)/4]2) for 2.5 < tMyr < 6; 
    fOpt = 1 − 0.841/(1 + [(tMyr − 6)/300]) for tMyr > 6. 

    For the photo-electric FUV band ΨFUV = 271[1+(tMyr/3.4)2] for tMyr < 3.4; 
    ΨFUV = 572(tMyr/3.4)−1.5 for tMyr > 3.4. 
    
    For the ionizing band Ψion = 500 for tMyr < 3.5; 
    Ψion = 60(tMyr/3.5)−3.6 + 470(tMyr/3.5)0.045−1.82 ln tMyr
    for 3.5 < tMyr < 25; Ψion = 0 for tMyr > 25. 
    
    The remaining UV luminosity, Ψbol − (ΨIR +Ψopt + ΨFUV + Ψion) 
    is assigned to the NUV band ΨNUV.
    """
    bolo_lums = getBolometricLuminosity(ageGyrs,masses)

def calculateKappa(vcs,rs):
    """calculate the epicyclic frequency"""
    
    dvcdr = (vcs[1:] - vcs[:-1])/(rs[1:]-rs[:-1])
    mid_rs = (rs[1:]+rs[:-1])/2.
    mid_vcs = (vcs[1:]+vcs[:-1])/2.
    kappas = np.sqrt(4*mid_vcs**2/mid_rs**2+mid_rs**2*dvcdr)
    kappa_fn = interp1d(mid_rs,kappas,fill_value="extrapolate",kind='linear')
    return kappa_fn(rs)

## USEFUL PHYSICS 
def calculateSigma1D(vels,masses):
    vcom = np.sum(vels*masses[:,None],axis=0)/np.sum(masses)
    vels = vels - vcom # if this has already been done, then subtracting out 0 doesn't matter
    v_avg_2 = (np.sum(vels*masses[:,None],axis=0)/np.sum(masses))**2
    v2_avg = (np.sum(vels**2*masses[:,None],axis=0)/np.sum(masses))
    return (np.sum(v2_avg-v_avg_2)/3)**0.5

def ff_timeToDen(ff_time):
    """ff_time must be in yr"""
    Gcgs = 6.67e-8 # cm^3 /g /s^2
    den = 3*np.pi/(32*Gcgs)/(ff_time * 3.15e7)**2 # g/cc
    return den 

def denToff_time(den):
    """den must be in g/cc"""
    Gcgs = 6.67e-8 # cm^3 /g /s^2
    ff_time = (
        3*np.pi/(32*Gcgs) /
        den  # g/cc
        )**0.5 # s

    ff_time /=3.15e7 # yr
    return ff_time

def getVcom(masses,velocities):
    assert np.sum(masses) > 0 
    return np.sum(masses[:,None]*velocities,axis=0)/np.sum(masses)

def iterativeCoM(coords,masses,n=4,r0=np.array([0,0,0])):
    rcom = r0
    radius = 1e10
    for i in range(n):
        mask = extractSphericalVolumeIndices(coords,rcom,radius)
        rcom = np.sum(coords[mask]*masses[mask][:,None],axis=0)/np.sum(masses[mask])
        print(radius,rcom)
        radius = 1000/3**i
    return rcom

def getAngularMomentum(vectors,masses,velocities):
    return np.sum(np.cross(vectors,masses[:,None]*velocities),axis=0)

def getAngularMomentumSquared(vectors,masses,velocities):
    ltot = np.sum(# sum in quadrature |lx|,|ly|,|lz|
        np.sum( # sum over particles 
            np.abs(np.cross( # |L| = |(r x mv )|
                vectors,
                masses[:,None]*velocities))
            ,axis=0)**2
        )**0.5 # msun - kpc - km/s units of L

    return ltot**2

    Li = np.cross(vectors,masses[:,None]*velocities)
    L2i = np.sum(Li*Li,axis=1)

    return np.sum(L2i)
