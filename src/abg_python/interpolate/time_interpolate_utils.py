import warnings
import time

import numpy as np
import pandas as pd

from ..snapshot_utils import convertSnapToDF
from ..array_utils import filterDictionary
from ..math_utils import get_primehats,add_jhat_coords
from .. import kms_to_kpcgyr

def find_bordering_snapnums(
    snap_times_gyr,
    dGyr=.005,
    tmin=None,
    tmax=None,
    times_gyr=None):
    """ """

    if times_gyr is None:
        ## handle default maximum time
        tmax = snap_times_gyr[-1] if tmax is None else tmax
        
        ## handle default minimum time
        if tmin is None:
            tmin = snap_times_gyr[0]

        ## remove dGyr so that tmin is included in arange below
        elif tmin - dGyr > 0:
            tmin = tmin-dGyr

        ## create list of times, -1e-9 to avoid landing exactly on a snapshot number
        times_gyr = np.arange(tmax,tmin,-dGyr)[::-1]-1e-9
    
    inds_next = np.argmax((times_gyr - snap_times_gyr[:,None]) < 0 ,axis=0)
    inds_prev = inds_next-1
    return (
        times_gyr,
        np.array(list(zip(inds_prev,inds_next))),
        np.array(list(zip(snap_times_gyr[inds_prev],snap_times_gyr[inds_next]))))

def convertToDF(
    snapshot_dictionary,
    keys_to_extract=None,
    polar=True,
    extra_arrays_function=None):
    ## note, it might make more sense for them to compute it on the interpolated snapshot?
    ##  i guess it depends if they want to interpolate their own thing or compute their thing
    ##  on an interpolated quantity(ies)
    if extra_arrays_function is not None:
        raise NotImplementedError("Need to allow users to pass in a function that will compute"+
            " arbitrary quantity arrays from stuff computed in the snapshot.")

    if keys_to_extract is None: keys_to_extract = []
    keys_to_extract += [
        'Coordinates',
        'Masses',
        'SmoothingLength',
        'Velocities',
        'ParticleIDs',
        'ParticleChildIDsNumber',
        'AgeGyr',
        'Temperature']
    
    if polar: keys_to_extract+=['polarjhats','polarjhatCoordinates','polarjhatVelocities','CircularVelocities']

    pandas_kwargs = dict(
        keys_to_extract=keys_to_extract,
        total_metallicity_only=True)
    
    ## convert snapshot dictionaries to pandas dataframes
    return convertSnapToDF(snapshot_dictionary,**pandas_kwargs)

## find which stars in the previous gas df are now in the 
def cross_match_starformed_gas(
    t0,t1,
    prev_gas_df,next_gas_df,
    prev_star_df,next_star_df):
    ##  next star df (i.e. which ones turned into stars)
    starformed = prev_gas_df.index[prev_gas_df.index.isin(next_star_df.index)]
    ages = next_star_df.loc[starformed,'AgeGyr']
    dsnap = t1 - t0
    #print(starformed.shape[0],'many starformed particles')
    
    ## add gas data to the stars and set the ages s.t. they will
    ##  appear at the correct time.
    ## -----------
    ##  append the rows, but only columns which are shared
    which_keys = prev_gas_df.keys()[prev_gas_df.keys().isin(prev_star_df.keys())]
    prev_star_df = pd.concat([
        prev_star_df, ## df to modify
        prev_gas_df.loc[starformed,which_keys]], ## new rows
        axis=0, ## add rows
        )
    ##  linear interpolation from ages-dt -> ages-dt + dt = ages
    ##   will result in star particles disappearing at correct time
    prev_star_df.loc[starformed,'AgeGyr'] = ages-dsnap

    ##  copy the field values from the future for keys that aren't shared
    ##   between gas and stars
    other_keys = prev_star_df.keys()[~prev_star_df.keys().isin(prev_gas_df.keys())]
    if len(other_keys):
        prev_star_df.loc[starformed,other_keys] = next_star_df.loc[starformed,other_keys]

    ## add star data to the gas and set the ages s.t. they will
    ##  disappear at the correct time.
    ## -----------
    ##  append the rows, but only columns which are shared
    which_keys = next_star_df.keys()[next_star_df.keys().isin(next_gas_df.keys())]
    next_gas_df = pd.concat([
        next_gas_df, ## df to modify 
        next_star_df.loc[starformed,which_keys]], ## new rows
        axis=0, ## add rows
        )
    ## tauf = age/(age-dt) -- derivation on remarkable 4/9/22
    ##  starting at 1, will cross 0 at age
    next_gas_df.loc[starformed,'AgeGyr'] = ages/(ages-dsnap)

    ##  copy the field values from the past for keys that aren't shared
    ##   between gas and stars
    other_keys = next_gas_df.keys()[~next_gas_df.keys().isin(next_star_df.keys())]
    if len(other_keys):
        next_gas_df.loc[starformed,other_keys] = prev_gas_df.loc[starformed,other_keys]

    return prev_gas_df,next_gas_df,prev_star_df,next_star_df

def finalize_df(
    t0,t1,
    merged_df,
    polar=True,
    take_avg_L=True,
    extrapolate=False,
    rotate_support_thresh=0.333):
    """
    keys_to_extract = ['Coordinates','Masses','SmoothingLength','ParticleIDs','ParticleChildIDsNumber']

    1. extrapolates missing values if extrapolate == True (it's False by default)
    2. adds polar coordinates in angular momentum frame of particles
    3. sets interpolation order and method by adding a column of interpolation flags

    """

    if extrapolate:
        ## fill in any missing values extrapolating
        linear_extrapolate(t0,t1,merged_df)
        linear_extrapolate(t0,t1,merged_df,False)

    ## and if we have any repeats, let's keep the first one
    merged_df = merged_df.loc[~merged_df.index.duplicated(keep='first')]
    
    ## handle scenarios when you can't find a particle to match
    ##  TODO currently uses linear extrapolation but consider using 
    ##  polar where appropriate?
    ## don't look for parents split parents if you're a star particle. 
    ##  this happens already in handle_stars_formed_between_snapshots with the
    ##  extra_df
        
    ## remove any nans; there shouldn't be any unless someone passed in spooky
    ##  field values
    #print('fraction nans:',np.sum(np.any(merged_df.isna(),axis=1))/merged_df.shape[0])
    #foo = merged_df.isna()
    #for key in merged_df.keys():
        #print(key,':',np.sum(foo[key]/foo.shape[0]))

    merged_df = merged_df.dropna().copy(deep=True)

    ## have to add all the angular momentum stuff
    if polar:
        ## compute coordinates w.r.t. angular momentum plane of each particle
        add_polar_jhat_coords(merged_df,take_avg_L=take_avg_L)

        ## add jhat rotation angle, compute dot product and take arccos of it
        rotation_angle = 0
        for i in range(3):
            rotation_angle += (
                merged_df['polarjhats_%d'%i] *
                merged_df['polarjhats_%d_next'%i])
        rotation_angle = np.arccos(np.clip(rotation_angle,-1,1))

        ## doing it this way (setting 0 @ t = t0 and rotation angle at t = t1) 
        ##  will automatically add 
        ##  the interpolated jhat_rotangle to the interp_snap in make_interpolated_snap
        merged_df['jhat_rotangle_next'] = rotation_angle
        merged_df['jhat_rotangle'] = np.zeros(rotation_angle.values.shape)

    
        interpolation_flags = np.zeros(merged_df.shape[0],dtype=int)
        ## the non-rotationally supported particles should be cartesian (3rd,2nd, or 1st order TBD)
        interpolation_flags[get_cartesian_interpolation_mask(merged_df,rotate_support_thresh)]+=1
    ## all particles should be interpolated in cartesian coordinates
    else: interpolation_flags = np.ones(merged_df.shape[0],dtype=int)

    ## set interpolation order and method flag for each particle
    ##  and check if the 3rd order interpolation scheme will introduce extrema, if so
    ##  then fall back to the 2nd, if the 2nd will introduce extrema then fall back
    ##  to the first order scheme (which is guaranteed not to).
    ## 0: polar -> 3rd in phi and 1st in R and theta
    ## 1: cartesian -> 3rd
    ## 2: cartesian -> 2nd
    ## 3: cartesian -> 1st
    merged_df['interpolation_flags'] = increment_interpolation_flags(t0,t1,merged_df,interpolation_flags)

    return merged_df

def linear_extrapolate(t0,t1,merged_df,forward=True):
    if forward: 
        lookup_suffix = ''
        target_suffix = '_next'
    else: 
        lookup_suffix='_next'
        target_suffix = ''
        t1,t0 = t0,t1

    mask = np.logical_or(
        merged_df.isna()['Masses'+target_suffix],
        merged_df['Masses'+target_suffix]==0)

    ## extrapolate the coordinates
    for i in range(3): ## should only happen if gas particles are converted into stars
        merged_df.loc[mask,f'Coordinates_{i:d}'+target_suffix] =(
            merged_df.loc[mask,f'Coordinates_{i:d}'+lookup_suffix] + 
            merged_df.loc[mask,f'Velocities_{i:d}'+lookup_suffix]*(t1-t0)*kms_to_kpcgyr)

    ## carry field values forward as a constant
    for key in merged_df.keys():
        if '_next' in key: continue
        if 'Coordinates' in key: continue
        if 'AgeGyr' in key: continue
        merged_df.loc[mask,key+target_suffix] = (merged_df.loc[mask,key+lookup_suffix])

def add_polar_jhat_coords(merged_df,take_avg_L=True):

    coords = np.zeros((merged_df.shape[0],3))
    vels = np.zeros((merged_df.shape[0],3))
    if take_avg_L:
        avg_Ls = 0
        ## calculate the average angular momentum vector between the two snapshots
        for target_suffix in ['','_next']:
            for i in range(3):
                coords[:,i] = merged_df[f'Coordinates_{i:d}{target_suffix}']
                vels[:,i] = merged_df[f'Velocities_{i:d}{target_suffix}']
            avg_Ls += np.cross(coords,vels)
        avg_Ls/=2
    else: avg_Ls = None

    ## calculate the coordinates in the relevant frame
    for target_suffix in ['','_next']:
        ## fill buffer with pandas dataframe values
        for i in range(3):
            coords[:,i] = merged_df[f'Coordinates_{i:d}{target_suffix}']
            vels[:,i] = merged_df[f'Velocities_{i:d}{target_suffix}']

        jhat_coords,jhat_vels,jhats = add_jhat_coords(
            ## pass a dictionary because we check if 'AngularMomentum' 
            ##  is a key of the snapdict argument
            dict( 
                AngularMomentum=avg_Ls,
                Coordinates=coords,
                Velocities=vels))

        for i in range(jhat_coords.shape[1]):
            merged_df[f'polarjhatCoordinates_{i:d}{target_suffix}'] = jhat_coords[:,i]
            merged_df[f'polarjhatVelocities_{i:d}{target_suffix}'] = jhat_vels[:,i]
        for i in range(jhats.shape[1]): merged_df[f'polarjhats_{i:d}{target_suffix}'] = jhats[:,i]

def get_cartesian_interpolation_mask(merged_df,rotate_support_thresh=0.333):
    ## check for rotational support, inspired by Phil's routine
    ## average 1D velocity between snapshots
    #avg_vels2 = (rp_prev_vels**2+rp_next_vels**2)/2
    ## time interpolated 1D velocity 
    Vdenoms = np.zeros((merged_df.shape[0]))

    Vc_key = f'CircularVelocities'
    vel_key = 'polarjhatVelocities_%d'
    coord_key = 'polarjhatCoordinates_0'

    ## read radii
    prev_rs = merged_df[coord_key]
    next_rs = merged_df[coord_key+'_next']

    ## read phi velocity
    prev_vphis = getattr(merged_df,vel_key%1).values 
    next_vphis = getattr(merged_df,vel_key%1+'_next').values 

    ## use Vc^2 in the denominator if we can
    if Vc_key in merged_df.keys():
        Vdenoms += getattr(merged_df,Vc_key).values**2
        Vdenoms += getattr(merged_df,Vc_key+'_next').values**2
    
    ## don't have CircularVelocities so we need to fill denominator with Vtot rather than Vc
    else:
        for i in range(3):
            ## can happen when theta is 0 when take_avg_L = False
            if vel_key%i not in merged_df.keys(): continue

            if i == 0 :
                Vdenoms += getattr(merged_df,vel_key%i).values**2
                Vdenoms += getattr(merged_df,vel_key%i+'_next').values**2
            else:
                Vdenoms += (getattr(merged_df,vel_key%i)*prev_rs).values**2
                Vdenoms += (getattr(merged_df,vel_key%i+'_next')*next_rs).values**2

    ## vphi^2 / denom^2
    vrot2_frac = ( (prev_vphis*prev_rs)**2 + (next_vphis*next_rs)**2 ) / Vdenoms
    #print('polar fraction:',np.sum(vrot2_frac > rotate_support_thresh)/vrot2_frac.size)

    ## non-rotationally supported <==> |vphi|/|v| < 0.5; |vphi| comes from sqrt above
    ##  make a mask for those particles which are not rotationally supported and need
    ##  to be replaced with a simple cartesian interpolation
    cartesian_mask = np.logical_or(
        vrot2_frac < rotate_support_thresh,
        np.logical_or(prev_rs > 30,next_rs > 30))
    
    #print('cartesian fraction:',np.sum(cartesian_mask)/cartesian_mask.size)
    
    return cartesian_mask

def increment_interpolation_flags(t0,t1,merged_df,interpolation_flags):

    ## check the third order cartesian interpolation points,
    ##  are any of those going to introduce extrema? if so
    ##  then drop them to 2nd order
    third_order_mask = interpolation_flags == 1
    bad_mask = check_third_order_extrema(t0,t1,merged_df,third_order_mask)
    interpolation_flags[bad_mask]+=1

    ## now check the 2nd order cartesian interpolation points,
    ##  are any of those going to introduce extrema? if so
    ##  then drop them to 1st order (which is guaranteed to not introduce extrema)
    second_order_mask = interpolation_flags == 2
    bad_mask = check_second_order_extrema(t0,t1,merged_df,second_order_mask)
    interpolation_flags[bad_mask]+=1

    return interpolation_flags

def check_third_order_extrema(t0,t1,merged_df,mask):
    dsnap = t1-t0
    merged_df = merged_df[mask]
    bad_mask = np.zeros(mask.shape,dtype=bool)
    arg_mask = np.argwhere(mask)
    for i in range(3):
        ## unpack this coordinate array
        prev_coords = merged_df['Coordinates_%d'%i]
        next_coords = merged_df['Coordinates_%d_next'%i]
        prev_vels = merged_df['Velocities_%d'%i]
        next_vels = merged_df['Velocities_%d_next'%i]

        ## define helper variables
        dcoord = next_coords - prev_coords
        x2 = 3*dcoord - (2*prev_vels+next_vels)*dsnap
        x3 = -2*dcoord + (prev_vels+next_vels)*dsnap

        ## handle velocity extremum
        bad_mask[arg_mask[np.logical_and(0 < -x2/(3*x3), -x2/(3*x3) < 1)]] = True

        ## handle + position extremum
        ## easier than gating +ive discriminant
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extremum = (-x2+np.sqrt(x2**2-3*x3-prev_vels*dsnap))/(3*x3)
        bad_mask[arg_mask[np.logical_and(0 < extremum, extremum < 1)]] = True
        ## handle - position extremum
        ## easier than gating +ive discriminant
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extremum = (-x2-np.sqrt(x2**2-3*x3-prev_vels*dsnap))/(3*x3)
        bad_mask[arg_mask[np.logical_and(0 < extremum, extremum < 1)]] = True
        
    return bad_mask

def check_second_order_extrema(t0,t1,merged_df,mask):
    dsnap = t1-t0
    merged_df = merged_df[mask]
    bad_mask = np.zeros(mask.shape,dtype=bool)
    arg_mask = np.argwhere(mask)
    for i in range(3):
        prev_coords = merged_df['Coordinates_%d'%i]
        next_coords = merged_df['Coordinates_%d_next'%i]
        prev_vels = merged_df['Velocities_%d'%i]
        next_vels = merged_df['Velocities_%d_next'%i]

        ## define helper variables
        dcoord = next_coords - prev_coords
        x2 = (next_vels - prev_vels)/2 * dsnap
        x1 = dcoord - x2

        ## handle position extremum
        bad_mask[arg_mask[np.logical_and(0 < -x1/(2*x2), -x1/(2*x2) < 1)]]= True

    return bad_mask

def make_interpolated_snap(t0,t1,merged_df,t):
    
    interp_snap = {}

    ## create a new snapshot with linear interpolated values
    ##  between key and key_next using t0, t1, and t.
    for key in merged_df.keys():
        if '_next' in key: continue
        elif 'polarjhat' in key: continue ## will allow jhat_rotangle though
        elif 'CircularVelocities' in key: continue ## only using to figure out if polar interpolation is appropriate
        elif 'Coordinates' == key: continue
        elif 'interpolation_flags' == key: continue
        interp_snap[key] = linear_interpolate(
            getattr(merged_df,key),
            getattr(merged_df,key+'_next'),
            t0,t1,t).values

    ## interpolate coordinates using higher order/more complicated interpolations
    ##  will remove the Coordinates_i and Velocities_i keys from interp_snap
    interp_snap['Coordinates'],interp_snap['Velocities'] = interpolate_position(
        t,t0,t1,
        merged_df,
        interp_snap,
        merged_df['interpolation_flags'])

    ids = np.array([*merged_df.index.values])
    interp_snap['ParticleIDs'] = ids[:,0]
    interp_snap['ParticleChildIDsNumber'] = ids[:,1]

    ## remove stars that have not formed yet or gas particles that have split
    ##  or turned into stars
    if 'AgeGyr' in interp_snap: interp_snap = filterDictionary(interp_snap,interp_snap['AgeGyr']>0)
 
    return interp_snap

def linear_interpolate(
    x0,x1,
    t0,t1,
    t):
    return x0 + (x1-x0)/(t1-t0)*(t-t0)

def interpolate_position(
    t,t0,t1,
    merged_df,
    interp_snap,
    interpolation_flags,
    ):

    
    coords = np.zeros((merged_df.shape[0],3))
    vels = np.zeros((merged_df.shape[0],3))

    ## successively fall back to worse and worse interpolation schemes
    ##  as determined by the interpolation limiter in finalize_df
    for i in range(interpolation_flags.max()):
        this_mask = interpolation_flags == i
        ## 0 = polar
        if i == 0: 
            coords[this_mask],vels[this_mask] = polar_interpolate(
                t,t0,t1,merged_df,interp_snap,this_mask)
        ## 1,2,3 = third, second, and first order
        else: 
            coords[this_mask],vels[this_mask] = cartesian_interpolate(
                t,t0,t1,merged_df,mask=this_mask,order=4-i)

    return coords,vels

def cartesian_interpolate(
    t,t0,t1,
    merged_df,
    mask=None,
    order=2):


    if mask is None: mask = np.ones(merged_df.shape[0],dtype=bool)
    if not np.any(mask): return coords,vels

    ## apply the mask once to the df
    merged_df = merged_df[mask]

    coords = np.zeros((merged_df.shape[0],3))
    vels = np.zeros((merged_df.shape[0],3))

    for i in range(3):
        prev_coords = merged_df['Coordinates_%d'%i]
        prev_vels = merged_df['Velocities_%d'%i]

        next_coords = merged_df['Coordinates_%d_next'%i]
        next_vels = merged_df['Velocities_%d_next'%i]

        coords[:,i], vels[:,i] = interpolate_at_order(
            prev_coords,
            next_coords,
            prev_vels,
            next_vels,
            t,t0,t1,
            order=order)

    return coords,vels

def polar_interpolate(t,t0,t1,merged_df,interp_snap,mask=None):
    if mask is None: mask = np.ones(merged_df.shape[0],dtype=bool)

    ## mask the df so we only compute what we need to
    merged_df = merged_df[mask]

    coord_key = 'polarjhatCoordinates_%d'
    vel_key = 'polarjhatVelocities_%d'

    do_theta = False 
    for target_suffix in ['','_next']:
        for key in [coord_key%2,vel_key%2]:
            has_this_key = (key+target_suffix) in merged_df.keys()
            do_theta = do_theta or has_this_key
            if do_theta and not has_this_key:
                ## this happened out of the blue, no idea how this is even possible...
                ##  clearly something else is wrong further up the pipeline but at least
                ##  here we can catch it and raise the error
                raise KeyError(
                    f"{key+target_suffix:s} missing but have at least one other theta key:\n"+
                    f"{repr(list(merged_df.keys())):s}")

    ## may not actually have z because we've rotated into jhat plane
    ##  if take_avg_L = False (non-default)
    interpd_polar_coords = np.zeros((merged_df.shape[0],2+do_theta))
    interpd_polar_vels = np.zeros((merged_df.shape[0],2+do_theta))

    this_coord_key = coord_key%0
    this_vel_key = vel_key%0
    ## interpolate r coordinate
    prev_Rs = getattr(merged_df,this_coord_key).values
    next_Rs = getattr(merged_df,this_coord_key+'_next').values
    prev_vRs = getattr(merged_df,this_vel_key).values
    next_vRs = getattr(merged_df,this_vel_key+'_next').values

    interpd_polar_coords[:,0], interpd_polar_vels[:,0] = interpolate_at_order(
        prev_Rs,
        next_Rs,
        prev_vRs,
        next_vRs,
        t,t0,t1) ## defaults to order=1

    this_coord_key = coord_key%1
    this_vel_key = vel_key%1
    ## interpolate phi coordinate
    prev_vphis = getattr(merged_df,this_vel_key).values 
    next_vphis = getattr(merged_df,this_vel_key+'_next').values 

    interpd_polar_coords[:,1], interpd_polar_vels[:,1] = interpolate_at_order(
        getattr(merged_df,this_coord_key).values,
        getattr(merged_df,this_coord_key+'_next').values,
        prev_vphis, ## radians / gyr
        next_vphis, ## radians / gyr
        t,t0,t1,
        order=3,
        periodic=True)
    
    if not do_theta: prev_vthetas=next_vthetas=0
    else:
        this_coord_key = coord_key%2
        this_vel_key = vel_key%2
        ## interpolate phi coordinate
        prev_vthetas = getattr(merged_df,this_vel_key).values 
        next_vthetas = getattr(merged_df,this_vel_key+'_next').values 

        interpd_polar_coords[:,2], interpd_polar_vels[:,2] = interpolate_at_order(
            getattr(merged_df,this_coord_key).values,
            getattr(merged_df,this_coord_key+'_next').values,
            prev_vthetas, ## radians / gyr
            next_vthetas, ## radians / gyr
            t,t0,t1) ## defaults to order=1

    ## need to convert interpd_polar_coords and interpd_polar_vels from r' p' to x,y,z
    ##  do that by getting interpolated jhat vectors and then associated x',y' vectors
    coords,vels = convert_polar_to_cartesian(
        merged_df,
        interp_snap.pop('jhat_rotangle')[mask],
        interpd_polar_coords,
        interpd_polar_vels)

    return coords,vels

def interpolate_at_order(
    this_prev_coords,
    this_next_coords,
    this_prev_vels,
    this_next_vels,
    t,t0,t1,
    order=1,
    periodic=False):

    dt = (t-t0) 
    dsnap = (t1-t0)
    time_frac = dt/dsnap ## 'tau' in Phil's notation

    dcoord = this_next_coords - this_prev_coords
    if periodic:
        ## how far would we guess each particle goes at tfirst?
        ##  let's guess how many times it actually went around, basically
        ##  want to determine which of (N)*2pi + dcoord  
        ##  or (N+1)*2pi + dcoord, or (N-1)*2pi + dcoord is 
        ##  closest to approx_radians, (N = approx_radians//2pi)
        dcoord = guess_windings(dcoord,(this_prev_vels+this_next_vels)/2*dsnap,2*np.pi)
    #if periodic:dcoord = np.mod(dcoord,2*np.pi)
    
    ## basic linear interpolation
    if order == 1:
        interp_coords = this_prev_coords + dcoord*time_frac
        #interp_vels = this_prev_vels + (this_next_vels - this_prev_vels)*time_frac
        ## corresponds to the constant velocity that the particle is traveling with
        ##  for this interpolation scheme
        interp_vels = dcoord/dsnap 
    ## correction factor to minimize velocity difference, apparently
    elif order == 2:
        x2 = (this_next_vels - this_prev_vels)/2 * dsnap
        x1 = dcoord - x2
        interp_coords = this_prev_coords + x1*time_frac + x2*time_frac*time_frac
        ## splits the vf-v0 centered on dcoord/dsnap rather than
        ## splits the vf-v0 centered on dcoord/dsnap rather than on (vf+v0)/2
        interp_vels = x1/dsnap + 2*x2*time_frac/dsnap 
    ## "enables exact matching of x,v but can 'overshoot' " - phil
    elif order == 3:
        x2 = 3*dcoord - (2*this_prev_vels+this_next_vels)*dsnap
        x3 = -2*dcoord + (this_prev_vels+this_next_vels)*dsnap
        interp_coords = this_prev_coords + this_prev_vels*dt + x2*time_frac**2 + x3*time_frac**3
        interp_vels = this_prev_vels + 2*x2*time_frac/dsnap + 3*x3*time_frac**2/dsnap
    else: raise Exception("Bad order, should be 1,2, or 3")

    ## do a simple 1st order interpolation for the velocities
    ##  while we have them in scope
    return interp_coords,interp_vels

def convert_polar_to_cartesian(
    merged_df,
    rotangle,
    interpd_polar_coords,
    interpd_polar_vels):
    """ If interpd_polar_coords is shape Nx2 then we are to interpret as being r,phi in the rotated frame.
    If interpd_polar_coords is shape Nx3 then we are to interpret as it being spherical coordinates in some
    other frame. Typically this will be a fixed frame aligned with the average angular momentum vector between the
    two snapshots. In this case rotangle will be 0, but in principle can be in any two frames and this will still 
    work.

    Parameters
    ----------
    interp_snap : _type_
        _description_
    interpd_polar_coords : _type_
        _description_
    interpd_polar_vels : _type_
        _description_
    vrot2_frac : _type_, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    prev_jhats = np.zeros((merged_df.shape[0],3))
    next_jhats = np.zeros((merged_df.shape[0],3))
    for i in range(3):
        jhat_key = 'polarjhats_%d'%i
        prev_jhats[:,i] = getattr(merged_df,jhat_key).values
        next_jhats[:,i] = getattr(merged_df,jhat_key+'_next').values

    ## if we were only passed r and phi then we don't need to convert
    ##  fixed theta, everything is at z=0 in the rotated frame
    if interpd_polar_coords.shape[1] != 3: thetas = np.pi/2 
    else: thetas = interpd_polar_coords[:,2]

    khats = np.cross(prev_jhats,next_jhats) ## vector about which jhat is rotated
    ohats = np.cross(khats,prev_jhats) ## vector pointing from ji toward jf in plane of rotation

    ## make sure all our vectors are normalized properly 
    prev_jhats = prev_jhats/np.linalg.norm(prev_jhats,axis=1)[:,None]
    knorms = np.linalg.norm(khats,axis=1)
    khats[knorms>0] /= knorms[knorms>0,None]
    onorms = np.linalg.norm(ohats,axis=1)
    ohats[onorms>0] /= onorms[onorms>0,None]

    ## last term is canceled b.c. by definition k . j_i = 0
    interp_jhats = prev_jhats*np.cos(rotangle)[:,None] + ohats*np.sin(rotangle)[:,None] #+ khats*(np.dot(khats,prev_jhats))*(1-np.cos(rotangle))
    interp_jhats/=np.linalg.norm(interp_jhats,axis=1)[:,None]
    ## note that in case rotangle = 0 then interp_jhats = prev_jhats (= next_jhats)

    xprimehats,yprimehats = get_primehats(interp_jhats) ## x' & y' in x,y,z coordinates

    interpd_xyz_vels = np.zeros((interpd_polar_vels.shape[0],3))
    interpd_xyz_coords = np.zeros((interpd_polar_vels.shape[0],3))

    ## replace rp w/ xy
    #  vx = vr * cos(phi) * sin(theta) + ...
    interpd_xyz_vels[:,0] +=  interpd_polar_vels[:,0]*np.cos(interpd_polar_coords[:,1])*np.sin(thetas) 
    #  vy = vr * cos(phi) * sin(theta) + ...
    interpd_xyz_vels[:,1] +=  interpd_polar_vels[:,0]*np.sin(interpd_polar_coords[:,1])*np.sin(thetas)
    #  vz = vr * cos(theta) + ...
    interpd_xyz_vels[:,2] +=  interpd_polar_vels[:,0]*np.cos(thetas)

    # vx = [-vphi/r * sin(phi)]*r + ...
    interpd_xyz_vels[:,0] += -interpd_polar_vels[:,1]*np.sin(interpd_polar_coords[:,1])*interpd_polar_coords[:,0]
    # vx =  [vphi/r * cos(phi)]*r + ...
    interpd_xyz_vels[:,1] +=  interpd_polar_vels[:,1]*np.cos(interpd_polar_coords[:,1])*interpd_polar_coords[:,0]
    ## no contribution to vz from vphi

    if interpd_polar_vels.shape[1] == 3:
        #  vx = [vtheta/r * cos(phi) * sin(theta)]*r + ...
        interpd_xyz_vels[:,0] +=  -interpd_polar_vels[:,2]*np.cos(interpd_polar_coords[:,1])*np.cos(thetas)*interpd_polar_coords[:,0]
        #  vy = [vtheta/r * cos(phi) * sin(theta)]*r + ...
        interpd_xyz_vels[:,1] +=  interpd_polar_vels[:,2]*np.sin(interpd_polar_coords[:,1])*np.cos(thetas)*interpd_polar_coords[:,0]
        #  vz = [-vtheta/r * cos(theta)]*r + ...
        interpd_xyz_vels[:,2] +=  -interpd_polar_vels[:,2]*np.sin(thetas)*interpd_polar_coords[:,0]

    interpd_xyz_coords[:,0] = interpd_polar_coords[:,0]*np.cos(interpd_polar_coords[:,1])*np.sin(thetas) # r * cos(phi) * sin(theta)
    interpd_xyz_coords[:,1] = interpd_polar_coords[:,0]*np.sin(interpd_polar_coords[:,1])*np.sin(thetas) # r * sin(phi) * sin(theta)
    interpd_xyz_coords[:,2] = interpd_polar_coords[:,0]*np.cos(thetas) # r * cos(theta)

    ## unit primehat vectors are in simulation coordinate frame, multiply by 
    ##  components in jhat frame to get simulation coordinates
    coords = (
        interpd_xyz_coords[:,0,None]*xprimehats +
        interpd_xyz_coords[:,1,None]*yprimehats +
        interpd_xyz_coords[:,2,None]*interp_jhats)

    vels = (
        interpd_xyz_vels[:,0,None]*xprimehats +
        interpd_xyz_vels[:,1,None]*yprimehats +
        interpd_xyz_vels[:,2,None]*interp_jhats)

    return coords,vels

def guess_windings(dphi,vphi_dt,period=1):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_windings = vphi_dt//period
    ## now we have to decide, do we go back a whole winding
    ##  to match dphi or do we go forward part of a winding?
    guesses = (base_windings+np.array([-1,0,1])[:,None])*period + dphi
    
    ## find the guess that is closest to the predicted vphi_dt
    adjust_windings = np.argmin(np.abs(guesses-vphi_dt),axis=0)

    ## take advantage of the fact that options are -1,0,1 to 
    ## turn min index from 0,1,2 -> -1,0,1
    adjust_windings-=1

    return (base_windings+adjust_windings)*period+dphi

def search_multi_ids(
    lookup_df,
    target_df,
    forward=True):
    """ extrapolates coordinates/fields forward in time if in prev but not next
        extrapolats coordinates/fields backward in time if in next but not prev
        attempts to avoid extrapolation if there is a particle split (looks for parent)"""

    if forward: target_suffix = '_next'
    else: target_suffix = ''

    mask = (target_df.isna()['Masses'+target_suffix]) | (target_df['Masses'+target_suffix]==0)
    init_time = time.time()
    print(np.sum(target_df.isna()['Masses'+target_suffix]) , np.sum(target_df['Masses'+target_suffix]==0),mask.size)
    print('fraction missing before:',np.sum(mask)/mask.size)

    ## find the particles who don't exist in the prev snapshot
    multi_indices = np.array(mask[mask].index.values.tolist())

    ## child id !=0 suggests it was a gas split, if == 0 it turned into a star
    ##  and we won't be able to find it anyway.
    if lookup_df is target_df: multi_indices = multi_indices[multi_indices[:,1] !=0]

    if len(multi_indices):
        print('fixing:',len(multi_indices))
        assign_from_ancestor(
            multi_indices, 
            ## df to look for parents in, 
            ##  pass only relevant particles for improved performance
            lookup_df.loc[np.unique(multi_indices[:,0])], 
            target_df,
            forward=forward) ## df to assign values to

    post_mask = (target_df.isna()['Masses'+target_suffix]) | (target_df['Masses'+target_suffix]==0)
    print('fraction missing after:',np.sum(post_mask)/post_mask.size,f'({time.time()-init_time:.1f} s elapsed)')

    return target_df

def assign_from_ancestor(
    orphaned_multi_indices,
    lookup_df,
    target_df,
    forward=False):

    if forward: 
        target_suffix = '_next'
        lookup_suffix = ''
    else: 
        target_suffix = ''
        lookup_suffix='_next'

    for orphaned_multi_index in orphaned_multi_indices:
        ## convert to tuple so it can actually index .loc
        orphaned_multi_index = tuple(orphaned_multi_index)
        ## find all particles that have a matching base index
        base_index = orphaned_multi_index[0]
        ## if there are none, then we have to skip :[
        if base_index not in lookup_df.index: continue
        possible_parents = lookup_df.loc[base_index]

        ## no parents to choose from, we have to just extrapolate
        ##  tbh this should not really be possible??
        if possible_parents.shape[0] == 1: continue 
        ## only one possible candidate it could've split from
        ##  let's just choose the other particle
        elif possible_parents.shape[0] == 2:
            parent_multi_index = (
                orphaned_multi_index[0],
                possible_parents[possible_parents.index != orphaned_multi_index[1]].index[0])
        elif possible_parents.shape[0] > 2: 
            ## so there are a bunch of candidates, we'll have to use the formula
            ##  for new child IDs and then check below that the ID is actually contained
            ##      new child ID = parent child ID + 2^(number of times split) 
            gen_key = 'ParticleIDGenerationNumber'
            generation = target_df.loc[orphaned_multi_index,gen_key+lookup_suffix]
            parent_multi_index = (orphaned_multi_index[0],orphaned_multi_index[1]-2**(generation-1))

        ## so this can only happen if we hit the >2 case and the calculated value wasn't there
        ##  in this scenario there is lots of splitting happening rapidly, I guess,
        ##  here is one scenario that I can imagine where this is possible:
        ## ----------------------------------------------------------------------
        ##  prev                        between                                 next
        ##  grandparent  grandparent || parent; parent||child; parent -> star  child
        ## ----------------------------------------------------------------------
        ##  which is a wild set of scenarios, should be pretty unlikely and yet it
        ##  happens pretty frequently
        if parent_multi_index not in lookup_df.index:
            continue
            parent_multi_index = (orphaned_multi_index[0],possible_parents.iloc[0].name)
            print("NOTE: ambiguous parentage for child particle.")

        parent = lookup_df.loc[parent_multi_index]

        ## well we found the parent but it's missing a value, somehow, 
        ##  I guess that can happen if you have a grandparent -> parent -> child 
        ##  all between one snapshot but not sure
        if np.any(np.isnan(parent)): continue
        
        ## alright, we're here, we made it. now we're going to copy over
        ##  all the keys from the parent to the child
        for key in lookup_df.keys():
            if '_next' in key: continue
            if 'AgeGyr' in key: continue
            ## if the parent has more information than the child needs
            ##  that's fine. such is life. we don't need it.
            if key+target_suffix not in target_df.keys(): continue
            ## copy all fields from the parent particle
            parent_val = parent[key+lookup_suffix]
            ## no idea why this fails, probably because the df is too big
            ##  and the hash table is messed up. but i literally went into
            ##  the skunkworks of pandas to debug this and it's a nightmare. 
            ##  basically what happens is that you try and fill a scalar value
            ##  with a scalar value but somewhere along the way you end up trying
            ##  to set an array value with a sequence but right before doing so 
            ##  you try and take the truth value of that array and you get the
            ##  any(), all() exception... very dumb.
            try: target_df.loc[orphaned_multi_index,key+target_suffix] = parent_val
            except:
                ## however, if you index *this* way, pandas asserts there's ambiguity over
                ##  whether the DF is a copy and it raises a warning (but doesn't seem to
                ##  try and set an array value with a sequence or compute the truth value
                ##  of an array)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    target_df[key+target_suffix].loc[orphaned_multi_index] = parent_val

                ## double check that we didn't change a copy as pandas
                ##  suspects we might've (i hate you pandas)
                if (not np.isnan(parent_val) and 
                    np.isnan(target_df.loc[orphaned_multi_index,key+target_suffix])):
                    raise ValueError("Value wasn't changed properly")
