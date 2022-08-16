import h5py
import os

import numpy as np 

from .array_utils import findArrayClosestIndices
from .system_utils import getfinsnapnum
from .constants import cm_per_kpc

### Hernquist/NFW Profiles
def findCCFromHernquistA(m200,akpc):
    r200 = (3*m200/(800*np.pi*RHOCRIT))**(1./3)#pc
    CC=getNFWCCFromHernquist(r200*1e6,a=akpc*1e3)
    return CC

def getNFWCCFromHernquist(R200,a=300,ax=None):
    """Inverts relationship between R200, nfw concentration, 
        nfw scale radius, and hernquist scale radius for concentration"""
    cs = np.linspace(1e-2,1e3,1e5)
    hernquist_as = R200/cs * (2 * (np.log(1 + cs) - cs / (1 + cs)))**0.5
    if ax is not None:
        ax.plot(cs,hernquist_as,label=r'a=$\frac{r200}{c}\sqrt{2(\log(1+c)-\frac{c}{1+c})}$',lw=3)
        ax.plot(cs,[a]*len(cs),label='a=%.2f'%a,lw=3)
        ax.legend(loc=0)
        ax.set_xlabel('c')
        ax.set_ylabel('a (kpc)')
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
    """ Assumes a flat cosmology

        a0 = stellar_tform;
        a2 = All.Time;        

        /* use exact solution for flat universe, ignoring the radiation-dominated epoch [no stars forming then] */
            /* use simple trap rule integration */
            a1 = 0.5*(a0+a2);
            x0 = 1./(a0*hubble_function(a0));
            x1 = 1./(a1*hubble_function(a1));
            x2 = 1./(a2*hubble_function(a2));
            age = (a2-a0)*(x0+4.*x1+x2)/6.;
        }
    } 
    age *= UNIT_TIME_IN_GYR; // convert to absolute Gyr
    if((age<=1.e-5)||(isnan(age))) {age=1.e-5;}
    return age;
}
    """
    
    UnitTime_in_seconds = 1e5*cm_per_kpc / HubbleParam #/ 1 kms suppressed
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

def RydenLookbackTime(HubbleParam,Omega_Matter,scale_factors):
    """Compute the lookback time to a (list of) scale factor(s) analytically
        assuming a flat cosmology. """

    UnitTime_in_seconds = 1e5*cm_per_kpc / HubbleParam #/ 1 kms suppressed
    UnitTime_in_Megayears = UnitTime_in_seconds/3.1536e13
    Hubble_H0_CodeUnits = 3.2407789e-18 * UnitTime_in_seconds
    
    prefactor = 2/3 * 1/np.sqrt(1-Omega_Matter)
    prefactor *= 0.001*UnitTime_in_Megayears/HubbleParam/Hubble_H0_CodeUnits
    
    aml3 = (Omega_Matter/(1-Omega_Matter))
    
    return prefactor * np.log( scale_factors**(3/2)/np.sqrt(aml3) + np.sqrt(1+scale_factors**3/aml3))

def approximateRedshiftFromGyr(HubbleParam,Omega0,gyrs):

    ## many zs..., uniformly in log(1+z) from z=0 to z=15
    zs = 10**np.linspace(0,np.log10(1000),np.max([2*gyrs.size,10**4]),endpoint=True)-1

    ## standard FIRE cosmology...
    #HubbleParam = 0.7
    #Omega0 = 0.272

    scale_factors = 1./(1+zs)
    close_times = RydenLookbackTime(HubbleParam,Omega0,scale_factors)

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
    cur_time_gyr = RydenLookbackTime(HubbleParam,Omega0,cur_time)
    return cur_time_gyr

def getAgesGyrs(open_snapshot):
    cosmo_sfts=open_snapshot['StellarFormationTime']
    cur_time = open_snapshot['Time']
    HubbleParam = open_snapshot['HubbleParam']
    Omega0 = open_snapshot['Omega0'] if 'Omega0' in open_snapshot else open_snapshot['Omega_Matter']
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

    cur_time_gyr = RydenLookbackTime(HubbleParam,Omega0,cur_time)
    sfts = cur_time_gyr - convertStellarAges(HubbleParam,Omega0,cosmo_sfts,cur_time)
    return sfts,cur_time_gyr

def load_rockstar_tree(
    snapdir,
    rockstar_path=None,
    loud=False,
    which_halo=0,
    fancy_trace=False):
    """ from awetzel.halo_analysis.halo_io: 
Default/stored properties (the most important ones)

If you read the halo catalog (out_*.list, halos_*.ascii, or halo_*.hdf5) you have:
    'id' : catalog ID, valid at given snapshot (starts at 0)
    'position' : 3-D position, along simulations's (cartesian) x,y,z grid [kpc comoving]
    'velocity' : 3-D velocity, along simulations's (cartesian) x,y,z grid [km / s]
    'mass' : default total mass - M_200m is default overdensity definition [M_sun]
    'radius' : halo radius, for 'default' overdensity definition of R_200m [kpc physical]
    'scale.radius' : NFW scale radius [kpc physical]
    'mass' : total mass defined via 200 x mean matter density [M_sun]
    'mass.vir' : total mass defined via Bryan & Norman 1998
    'mass.200c' : total mass defined via 200 x critical density [M_sun]
    'mass.bound' : total mass within R_200m that is bound to the halo [M_sun]
    'vel.circ.max' : maximum of the circular velocity profile [km / s]
    'vel.std' : standard deviation of the velocity of particles [km / s]
    'mass.lowres' : mass from low-resolution dark-matter particles in halo [M_sun]
    'host.index' : catalog index of the primary host (highest halo mass) in catalog
    'host.distance' : 3-D distance wrt center of primary host [kpc physical]
    'host.velocity' : 3-D velocity wrt center of primary host [km / s]
    'host.velocity.tan' : tangential velocity wrt primary host [km / s]
    'host.velocity.rad' : radial velocity wrt primary host (negative = inward) [km / s]

If you read the halo main progenitor histories (hlist*.list or halo_*.hdf5) you also have:
    'major.merger.snapshot' : snapshot index of last major merger
    'mass.half.snapshot' : snapshot when first had half of current mass
    'mass.peak' : maximum of mass throughout history [M_sun]
    'mass.peak.snapshot': snapshot index at which above occurs
    'vel.circ.peak' : maximum of vel.circ.max throughout history [km / s]
    'infall.snapshot' : snapshot index when most recently fell into a host halo
    'infall.mass' : mass when most recently fell into host halo [M_sun]
    'infall.vel.circ.max' : vel.circ.max when most recently fell into a host halo [km / s]
    'infall.first.snapshot' : snapshot index when first became a satellite
    'infall.first.mass' : mass when first fell into a host halo (became a satellite) [M_sun]
    'infall.first.vel.circ.max' : vel.circ.max when first became a satellite [km / s]
    'accrete.rate' : instantaneous accretion rate [M_sun / yr]
    'accrete.rate.100Myr : mass growth rate averaged over 100 Myr [M_sun / yr]
    'accrete.rate.tdyn : mass growth rate averaged over dynamical time [M_sun / yr]

If you read the halo merger trees (tree*.dat or tree.hdf5) you have:
    'tid' : tree ID, unique across all halos across all snapshots (starts at 0)
    'snapshot' : snapshot index of halo
    'am.phantom' : whether halo is interpolated across snapshots
    'descendant.snapshot' : snapshot index of descendant
    'descendant.index' : tree index of descendant
    'am.progenitor.main' : whether am most massive progenitor of my descendant
    'progenitor.number' : number of progenitors
    'progenitor.main.index' : index of main (most massive) progenitor
    'progenitor.co.index' : index of next co-progenitor (with same descendant)
    'final.index' : tree index at final snapshot
    'dindex' : depth-first order (index) within tree
    'progenitor.co.dindex' : depth-first index of next co-progenitor
    'progenitor.last.dindex' : depth-first index of last progenitor - includes *all* progenitors
    'progenitor.main.last.dindex' : depth-first index of last progenitor - only via main progenitors
    'central.index' : tree index of most massive central halo (which must be a central)
    'central.local.index' : tree index of local (lowest-mass) central (which could be a satellite)
    'host.index' : tree index of the primary host (following back main progenitor branch)
    'host.distance' : 3-D distance wrt center of primary host [kpc physical]
    'host.velocity' : 3-D velocity wrt center of primary host [km / s]
    'host.velocity.tan' : tangential velocity wrt primary host [km / s]
    'host.velocity.rad' : radial velocity wrt primary host (negative = inward) [km / s]
    """

    if fancy_trace: raise ValueError("Fancy tracing not implemented yet for this")

    if rockstar_path is None:
        rockstar_path = '../halo/rockstar_dm/catalog_hdf5'
        rockstar_path = os.path.realpath(os.path.join(snapdir,rockstar_path))

    treefile = os.path.join(rockstar_path,'tree.hdf5')
    my_tree = {}
    with h5py.File(treefile,'r') as handle:
        halo_index = handle['progenitor.last.dindex'][0]+1

        while which_halo > 0:
            halo_index = handle['progenitor.last.dindex'][halo_index]+1
            which_halo -=1

        for key in handle.keys():
            try: my_tree[key] = handle[key][:halo_index]
            except: my_tree[key] = handle[key][()]
            
    return my_tree,treefile

def trace_rockstar(*args,**kwargs):

    ## load the merger tree for this halo
    my_tree,treefile = load_rockstar_tree(*args,**kwargs)

    ## follow the progenitor.main.index chain, 
    ##  each main progenitor has its own main progenitor (and a snapshot
    ##  corresponding to that progenitor). 
    ##  Ideally this halo appears in each snapshot but you never know :\
    chain = [0]
    index = chain[-1]
    while index > -1:
        chain += [my_tree['progenitor.main.index'][index]]
        index = chain[-1]
    chain = chain[:-1]

    snapshots = my_tree['snapshot'][chain]
    sort_inds = np.argsort(snapshots)
    halo_traj = my_tree['position'][chain] ## in comoving kpc!
    r200s = my_tree['radius'][chain]

    return snapshots[sort_inds],halo_traj[sort_inds],r200s[sort_inds],treefile

def fancy_trace_rockstar(snapdir,rockstar_path=None,fancy_trace=True,loud=False):

    if 'elvis' in snapdir:
        raise NotImplementedError("Can't handle multiple halos, yet")

    if rockstar_path is None:
        rockstar_path = '../halo/rockstar_dm/catalog_hdf5'
        rockstar_path = os.path.realpath(os.path.join(snapdir,rockstar_path))

    low_index = getfinsnapnum(
        rockstar_path,
        fname_to_match='halo_',getmin=True)
    high_index = getfinsnapnum(
        rockstar_path,
        fname_to_match='halo_',getmin=False)+1
    
    snapshots = np.arange(low_index,high_index)
    rcoms = np.zeros((high_index-low_index,3))+np.nan
    rvirs = np.zeros(high_index-low_index)+np.nan
    scalefactors = np.zeros(high_index-low_index)+np.nan

    for i,snapnum in enumerate(snapshots[::-1]):
        fname = 'halo_%03d.hdf5'%snapnum
        path = os.path.join(rockstar_path,fname)

        with h5py.File(path,'r') as handle:
            ## find the indices of the 3 most massive halos
            masses = handle['mass'][()]
            ## -masses b.c. argpartition orders from smallest to largest
            ##  since masses are all positive trick it into doing the
            ##  reverse by making them all negative

            N = min(20,masses.size-1)
            indices = np.argpartition(-masses,N,)[:N]
            ## argpartition does not guarantee an ordering
            ##  let's do it ourselves. order from most to least massive 
            indices = indices[np.argsort(masses[indices])][::-1]

            masses = masses[indices]
            hubble_param = handle['cosmology:hubble'][()]
            scalefactors[i] = handle['snapshot:scalefactor'][()]

            main_host_index = np.nanargmax(handle['mass']) # handle['host.index'][0] <-- removed from wetzel's code :[
            ## skip the first step because we have nothing to compare to
            if i != 0 and fancy_trace: 
                prev_rcom = rcoms[i-1]
                mass_rcoms = handle['position'][()][indices]
                ## km/s -> kpc/gyr
                mass_vcoms = handle['velocity'][()][indices]*1.022

                drs = mass_vcoms*(prev_time - handle['snapshot:time'][()])

                dists = np.sqrt(np.sum(((mass_rcoms+drs)-prev_rcom)**2,axis=1))

                closest_index = indices[np.argmin(dists)]

                mass_dists = np.sqrt((np.log10(masses)-np.log10(prev_mass))**2)
                mass_index = indices[np.argmin(mass_dists)]

                if closest_index != main_host_index or main_host_index != mass_index:
                    main_where = np.argwhere(indices==main_host_index)[0][0]
                    closest_where = np.argwhere(indices==closest_index)[0][0]
                    mass_where =  np.argwhere(indices==mass_index)[0][0]
                    mass_ratio = masses[closest_where]/masses[mass_where]
                    mass_dist_ratio = mass_dists[mass_where]/mass_dists[closest_where]
                    distance_ratio = dists[closest_where]/dists[mass_where]
                    if loud:
                        print(
                            '%03d'%snapnum,
                            'main host:',main_where,
                            'closest (kpc):',closest_where,
                            'closest (mass):',mass_where,
                            end='\t')
                    ## both physical and mass distances agree which one we should choose
                    ##  so let's choose it!
                    if closest_index == mass_index: 
                        if main_host_index != mass_index:
                            if loud:
                                print('ignoring rockstar')
                            main_host_index = mass_index 
                    ## we'll choose the one closest in physical space
                    ##  since the masses are kind of a toss-up but one 
                    ##  is pretty significantly closer than the other
                    elif distance_ratio < 0.5 and mass_ratio > 0.5:
                        if closest_index!=main_host_index:
                            if loud:
                                print('choosing closest (kpc)')
                        #print(
                            #'choosing the closest in physical space',
                            #'rather than the most massive.',
                            #'because the mass ratio is very close to 1:',mass_ratio,
                            #'but the distance ratio is pretty small:',distance_ratio)
                            main_host_index = closest_index
                    ## one is clearly closer in log-mass space, so we'll
                    ##  choose it
                    elif mass_dist_ratio < 0.5:
                        if mass_index!=main_host_index:
                            if loud:
                                print('choosing closest (mass)',distance_ratio,mass_ratio,dists)
                            #print(np.log10(prev_mass),np.log10(masses))
                            #print(mass_dists)
                            #import pdb; pdb.set_trace()
                            main_host_index = mass_index


            ## in comoving kpc (NOT comoving kpc/h), only needs scalefactor
            rcom = handle['position'][main_host_index]
            #vcom = handle['velocity'][main_host_index]
            ## in physical kpc? lmao
            rvir = handle['radius'][main_host_index]
            prev_time = handle['snapshot:time'][()]
            prev_mass = handle['mass'][main_host_index]

        rcoms[i,:] = rcom
        rvirs[i] = rvir

    rcoms*=scalefactors[:,None]
    return snapshots,rcoms[::-1],rvirs[::-1]


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
        rockstar_path = os.path.realpath(os.path.join(snapdir,rockstar_path))

    fname = 'halo_%03d.hdf5'%snapnum if fname is None else fname


    if 'elvis' in snapdir:
        if which_host == 0: which_host = 'host'
        elif which_host == 1: which_host = 'host2'
        main_index_fn = lambda handle: handle[which_host+'.index'][0]
    else: main_index_fn = lambda handle: np.argmax(handle['mass'])
    
    path = os.path.join(rockstar_path,'catalog_hdf5',fname)

    with h5py.File(path,'r') as handle:
        main_host_index = main_index_fn(handle)
        hubble_param = handle['cosmology:hubble'][()]
        ## in comoving kpc (NOT comoving kpc/h)
        rcom = handle['position'][main_host_index]
        scalefactor = handle['snapshot:scalefactor'][()]
        #vcom = handle['velocity'][main_host_index]

        ## in physical kpc? lmao
        rvir = handle['radius'][main_host_index]
        extra_values = [handle[key][()][main_host_index] for key in extra_names_to_read]

    return tuple([rcom*scalefactor,rvir] + extra_values)
        

## AHF file opening
def load_AHF(
    snapdir,snapnum,
    hubble = 0.702,
    ahf_path=None,
    extra_names_to_read = None,
    fname = None,
    return_full_halo_file = False):

    if extra_names_to_read is None:
        extra_names_to_read = ['Rstar0.5']

    ahf_path = '../halo/ahf/' if ahf_path is None else ahf_path
    fname = 'halo_00000_smooth.dat' if fname is None else fname

    path = os.path.join(snapdir,ahf_path,fname)

    if not os.path.isfile(path):
        raise IOError("path %s does not exist"%path)

    names_to_read = ['redshift','snum','Xc','Yc','Zc','Rvir']+extra_names_to_read

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
        names_to_read.pop(1)

    cols = []
    for name in names_to_read:
        if name not in names_to_read:
            raise ValueError(
                "%s is not in ahf file, available names are:"%name,
                names)
        else:
            cols+=[names.index(name)]

    if 'snum' not in names_to_read:
        
        if return_full_halo_file: 
            print('returning (without units):',names_to_read)
            output = np.genfromtxt(
                path,
                skip_header=1,
                usecols=cols,
                comments='@')
            return output
        else:
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
        if return_full_halo_file: 
            print('returning (without units):',names_to_read)
            return output

        index = output[:,1]==snapnum

        ## if there is no row that matches snapnum
        if np.sum(index)==0:
            ## snapnum is not in this halo file
            print(min(output[:,0]),'is the first snapshot in the halo file')
            print(output[:,0],'snapnums available')
            raise IOError("%d snapshot isn't in the AHF halo file"%snapnum)
        row = output[index][0]

    current_redshift = row[names_to_read.index('redshift')]

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
    close_times = RydenLookbackTime(
        HubbleParam,
        Omega0,
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
