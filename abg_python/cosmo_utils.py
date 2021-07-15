import numpy as np 
from abg_python.all_utils import *

import h5py


### Constants
G = 4.301e-9 #km^2 Mpc MSun^-1 s^-2

##### THIS CHANGES WHAT HUBBLE IS
REDSHIFT=0

#from WMAP
H0=71 # km /s / Mpc
OMEGA_0 = 0.27
OMEGA_B = 0.044
OMEGA_LAMBDA = 1-OMEGA_0
HUBBLE=H0*(OMEGA_0*(1+REDSHIFT)**3+OMEGA_LAMBDA)**0.5 #km/s / Mpc
RHOCRIT = 3*HUBBLE**2./(8*np.pi*G)# 139 Msun / Mpc^3

### Hernquist/NFW Profiles
def findCCFromHernquistA(m200,akpc):
    r200 = (3*m200/(800*np.pi*RHOCRIT))**(1./3)#pc
    CC=getNFWCCFromHernquist(r200*1e6,a=akpc*1e3)
    return CC

def getNFWCCFromHernquist(R200,a=300,plot=0):
    """Inverts relationship between R200, nfw concentration, 
        nfw scale radius, and hernquist scale radius for concentration"""
    cs = np.linspace(1e-2,1e3,1e5)
    hernquist_as = R200/cs * (2 * (np.log(1 + cs) - cs / (1 + cs)))**0.5
    if plot:
        plt.plot(cs,hernquist_as,label=r'a=$\frac{r200}{c}\sqrt{2(\log(1+c)-\frac{c}{1+c})}$',lw=3)
        plt.plot(cs,[a]*len(cs),label='a=%.2f'%a,lw=3)
        plt.gca().legend(loc=0)
        plt.gca().set_xlabel('c')
        plt.gca().set_ylabel('a (kpc)')
        plt.show()
    return cs[np.argmin((hernquist_as-a)**2.)]

def hernquist_profile(mtot,a,r,nfw=0):
    """Defines a herqnuist profile with mass mtot 
        and scale radius a. has the option of 
        being nfw instead using a flag."""
    return a*mtot/(2*np.pi)*(r*(a+r)**(3-nfw))**-1.

def contained_hernquist_profile(MR,R,a,r):
    """Defines a hernquist profile with mass mtot
        and scale radius a. Howver, this is done by measuring
        the mass contained in a radius R and scaling it up to 
        Mtot"""
    #for convenience
    u,U=r/a,R/a
    #scale up to Mtot
    Mtot =MR*(1-2*(2*U-1)/(1+U)**2.)**-1
    return Mtot/(2*np.pi*a**3.)*u**-1*(1+u)**-3

### Lookback time (in context of converting stellar ages)
def convertStellarAges(HubbleParam,Omega0,stellar_tform,Time):
    """ Assumes a flat cosmology"""
    
    km_per_kpc = 3.086e16
    UnitTime_in_seconds = km_per_kpc / HubbleParam #/ 1 kms suppressed
    UnitTime_in_Megayears = UnitTime_in_seconds/3.1536e13
    
    Hubble_H0_CodeUnits = 3.2407789e-18 * UnitTime_in_seconds 
    
    
    a0 = stellar_tform
    a2 = Time
    
    x0 = (Omega0/(1-Omega0))/(a0*a0*a0)
    x2 = (Omega0/(1-Omega0))/(a2*a2*a2)
    age = (2./(3.*np.sqrt(1-Omega0)))*np.log(np.sqrt(x0*x2)/((np.sqrt(1+x2)-1)*(np.sqrt(1+x0)+1)))
    age *= 1./Hubble_H0_CodeUnits

    age *= 0.001*UnitTime_in_Megayears/HubbleParam
    return age

def approximateRedshiftFromGyr(HubbleParam,Omega0,gyrs):

    ## many zs..., uniformly in log(1+z) from z=0 to z=15
    zs = 10**np.linspace(0,np.log10(1000),np.max([2*gyrs.size,10**4]),endpoint=True)-1

    ## standard FIRE cosmology...
    #HubbleParam = 0.7
    #Omega0 = 0.272

    scale_factors = 1./(1+zs)
    close_times = convertStellarAges(HubbleParam,Omega0,1e-16,scale_factors)

    ## find indices of close_times that match closest to gyrs
    indices = findArrayClosestIndices(gyrs,close_times)
    
    ## make sure we're close enough...
    try:
        assert np.mean(close_times[indices]-gyrs)<=1e-2
    except:
        print('Redshift range was not fine enough: 0.01 <',np.mean(close_times[indices]-gyrs))
        if recurse_depth > 3:
            raise ValueError("Failed to downsample the SFH to the snapshot times")
        else:
            return approximateRedshiftFromGyr(HubbleParam,Omega0,gyrs,recurse_depth+1)

    return zs[indices]

def convertReadsnapTimeToGyr(snap):
    cur_time = snap['Time']
    HubbleParam = snap['HubbleParam']
    Omega0 = snap['Omega0']
    cur_time_gyr = convertStellarAges(HubbleParam,Omega0,1e-16,cur_time)
    return cur_time_gyr

def getAgesGyrs(open_snapshot):
    cosmo_sfts=open_snapshot['StellarFormationTime']
    cur_time = open_snapshot['Time']
    HubbleParam = open_snapshot['HubbleParam']
    Omega0 = open_snapshot['Omega0']
    ages = convertStellarAges(HubbleParam,Omega0,cosmo_sfts,cur_time)
    ages[ages<0] = 0 ## why does this happen? only noticed once, m12i_res7100_md@526
    return ages

def convertSnapSFTsToGyr(open_snapshot,snapshot_handle=None,arr=None):
    if snapshot_handle==None:
        if arr!=None:
            cosmo_sfts=arr
        else:
            cosmo_sfts=open_snapshot['StellarFormationTime']
        cur_time = open_snapshot['Time']
        HubbleParam = open_snapshot['HubbleParam']
        Omega0 = open_snapshot['Omega0']
        print('Using',HubbleParam,Omega0,'cosmology')
    else:
        raise Exception("Unimplemented you lazy bum!")

    cur_time_gyr = convertStellarAges(HubbleParam,Omega0,1e-16,cur_time)
    sfts = cur_time_gyr - convertStellarAges(HubbleParam,Omega0,cosmo_sfts,cur_time)
    return sfts,cur_time_gyr

## rockstar file opening
def load_rockstar(
    snapdir,snapnum,
    rockstar_path=None,
    extra_names_to_read=None,
    fname=None,
    which_host=0):


    ## does not allow for one to get main_halo indexed values out
    extra_names_to_read = [] if extra_names_to_read is None else extra_names_to_read

    if rockstar_path is None:
        rockstar_path = '../halo/rockstar_dm/'
        rockstar_path = os.path.join(snapdir,rockstar_path)

    fname = 'halo_%03d.hdf5'%snapnum if fname is None else fname


    if which_host == 0:
        which_host = 'host'
    elif which_host == 1:
        which_host = 'host2'
    
    path = os.path.join(rockstar_path,'catalog_hdf5',fname)

    with h5py.File(path,'r') as handle:
        main_host_index = handle[which_host+'.index'][0]
        ## in comoving kpc (NOT comoving kpc/h)
        rcom = handle['position'][main_host_index]
        scalefactor = handle['snapshot:scalefactor'][()]
        #vcom = handle['velocity'][main_host_index]

        ## in physical kpc? lmao
        rvir = handle['radius'][main_host_index]
        extra_values = [handle[key] for key in extra_names_to_read]

    return tuple(np.append([rcom*scalefactor,rvir],extra_values))
        

## AHF file opening
def load_AHF(
    snapdir,snapnum,
    current_redshift,
    hubble = 0.702,
    ahf_path=None,
    extra_names_to_read = None,
    fname = None):

    if extra_names_to_read is None:
        extra_names_to_read = ['Rstar0.5']

    ahf_path = '../halo/ahf/' if ahf_path is None else ahf_path
    fname = 'halo_00000_smooth.dat' if fname is None else fname

    path = os.path.join(snapdir,ahf_path,fname)

    if not os.path.isfile(path):
        raise IOError("path %s does not exist"%path)

    names_to_read = ['snum','Xc','Yc','Zc','Rvir']+extra_names_to_read

    names = list(np.genfromtxt(path,skip_header=0,max_rows = 1,dtype=str,comments='@'))
    ## ahf will sometimes "helpfully" put the 1-indexed index in the column header
    if '(' in names[0]: 
        names = [name[:name.index('(')] for name in names]
        
    ## this isn't a smooth halo tracing file produced by Zach,
    ##  this is a single snapshot, generic, AHF file

    ## determine if we have a smoothed halo history or just a single snapshot file
    ##  remove 'snum' from the names to read from the halo file if it's just a single
    ##  snapshot.
    if 'snum' not in names:
        names_to_read.pop(0)

    cols = []
    for name in names_to_read:
        if name not in names_to_read:
            raise ValueError(
                "%s is not in ahf file, available names are:"%name,
                names)
        else:
            cols+=[names.index(name)]

    if 'snum' not in names_to_read:
        row = np.genfromtxt(
            path,
            skip_header=0,
            max_rows = 2,
            usecols=cols,
            comments='@')[1]
    else:
        output = np.genfromtxt(
            path,
            delimiter='\t',
            usecols=cols,
            skip_header=1)

        index = output[:,0]==snapnum

        ## if there is no row that matches snapnum
        if np.sum(index)==0:
            ## snapnum is not in this halo file
            print(min(output[:,0]),'is the first snapshot in the halo file')
            print(output[:,0],'snapnums available')
            raise IOError("%d snapshot isn't in the AHF halo file"%snapnum)
        row = output[index][0]

    ## presumably in comoving kpc/h 
    scom = np.array([
        row[names_to_read.index('Xc')],
        row[names_to_read.index('Yc')],
        row[names_to_read.index('Zc')],
        ])/hubble*(1/(1+current_redshift))
    scom = scom.reshape(3,)

    ## comoving kpc to pkpc
    length_unit_fact = 1/hubble*(1/(1+current_redshift))

    rvir = row[names_to_read.index('Rvir')]*length_unit_fact
    return_val = np.array([scom, rvir],dtype=object)

    ## for everything that comes after Rvir
    for name in names_to_read[names_to_read.index('Rvir')+1:]:
        if name == 'Rstar0.5': 
            unit_fact = length_unit_fact
        else:
            unit_fact = 1
            print(name,'does not have units')
        return_val = np.append(
            return_val,
            np.array(
                row[names_to_read.index(name)]*unit_fact,
                dtype=object))

    return return_val

def addRedshiftAxis(ax,
    HubbleParam,
    Omega0,zs=None):
    if zs is None:
        zs = np.array([0,0.1,0.25,0.5,1,2,4])[::-1]
    #0**np.linspace(0,np.log10(1000),np.max([2*gyrs.size,1e4]),endpoint=True)-1
    ## standard FIRE cosmology... #HubbleParam = 0.7 #Omega0 = 0.272
    scale_factors = 1./(1+zs)
    close_times = convertStellarAges(
        HubbleParam,
        Omega0,
        1e-16,
        scale_factors)

    ax1 = ax.twiny()
    ax1.set_xticks(close_times)

    ax1.set_xlabel('Redshift')
    #ax.set_xlim(0,close_times[-1])
    #ax1.set_xlim(0,close_times[-1])
    ax.set_xlim(0,14)
    ax1.set_xlim(0,14)

    ax1.set_xticklabels(["%g"%red for red in zs])
    return ax1
