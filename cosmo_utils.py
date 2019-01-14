import numpy as np 
import os

def getAgesGyrs(open_snapshot):
    cosmo_sfts=open_snapshot['StellarFormationTime']
    cur_time = open_snapshot['Time']
    HubbleParam = open_snapshot['HubbleParam']
    Omega0 = open_snapshot['Omega0']
    return convertStellarAges(HubbleParam,Omega0,cosmo_sfts,cur_time)

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

def load_AHF(
    snapdir,snapnum,
    current_redshift,
    hubble = 0.702,
    ahf_path=None,
    extra_names_to_read = ['Rstar0.5'],
    fname = None):

    if fname is None:
        fname = 'halo_00000_smooth.dat'

    ahf_path = '../halo/ahf/' if ahf_path is None else ahf_path
    fname = 'halo_00000_smooth.dat' if fname is None else fname

    path = os.path.join(snapdir,ahf_path,fname)

    if 'anglesd' in snapdir:
        path = os.path.join(snapdir,'AHF',fname)

    if not os.path.isfile(path):
        raise IOError("path %s does not exist"%path)

    names_to_read = ['snum','Xc','Yc','Zc','Rvir']+extra_names_to_read

    names = list(np.genfromtxt(path,skip_header=0,max_rows = 1,dtype=str))
    cols = []

    for name in names_to_read:
        cols+=[names.index(name)]

    output = np.genfromtxt(
        path,delimiter='\t',usecols=cols,skip_header=1)

    ## unpack rows of output
    xs,ys,zs = output[:,1:4].T

    index = output[:,0]==snapnum
    if np.sum(index)==0:
        ## snapnum is not in this halo file
        print(min(output[:,0]),'is the first snapshot in the halo file')
        raise IOError("This snapshot isn't in the AHF halo file")
    ## psnapumably in comoving kpc/h 
    scom = np.array([xs[index],ys[index],zs[index]])/hubble*(1/(1+current_redshift))
    scom = scom.reshape(3,)

    ## comoving kpc to pkpc
    rvir = output[:,4][index][0]/hubble*(1/(1+current_redshift))

    return_val = [scom, rvir]
    if 'Rstar0.5' in names_to_read:
        rstar_half = output[:,names_to_read.index('Rstar0.5')][index][0]/hubble*(1/(1+current_redshift))
        return_val+=[rstar_half]
    return return_val
