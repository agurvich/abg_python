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
    return_val = [scom, rvir]

    ## for everything that comes after Rvir
    for name in names_to_read[names_to_read.index('Rvir')+1:]:
        if name == 'Rstar0.5': 
            unit_fact = length_unit_fact
        else:
            unit_fact = 1
            print(name,'does not have units')
        return_val = np.append(return_val,row[names_to_read.index(name)]*unit_fact)

    return return_val
