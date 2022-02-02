import warnings

import numpy as np
import pandas as pd

from ..snapshot_utils import convertSnapToDF
from ..galaxy.gal_utils import Galaxy
from ..array_utils import filterDictionary
from ..math_utils import getThetasTaitBryan, rotateEuler
from ..math_utils import get_primehats
from ..physics_utils import get_IMass
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

def index_match_snapshots_with_dataframes(
    prev_sub_snap,
    next_sub_snap,
    keys_to_extract=None,
    extra_arrays_function=None,
    t0=None,
    t1=None,
    polar=True,
    extra_df=None):
    """
        if you use Metallicity  or Velocities then the keys will be different when you try to access them
          in your render call using render_kwargs.
        Velocities ->  Velocities_0,Velocities_1,Velocities_2
        Metallicity -> Metallicity_0,Metallicity_1,...
    keys_to_extract = ['Coordinates','Masses','SmoothingLength','ParticleIDs','ParticleChildIDsNumber']
    """
    
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
    
    if polar: keys_to_extract+=['polarjhats','polarjhatCoordinates','polarjhatVelocities']

    pandas_kwargs = dict(
        keys_to_extract=keys_to_extract,
        total_metallicity_only=True)
    
    ## convert snapshot dictionaries to pandas dataframes
    prev_df_snap = convertSnapToDF(prev_sub_snap,**pandas_kwargs)
    
    next_df_snap = convertSnapToDF(next_sub_snap,**pandas_kwargs)
        
    ## remove particles that do not exist in the previous snapshot, 
    ##  difficult to determine which particle they split from
    next_df_snap_reduced = next_df_snap.reindex(prev_df_snap.index,copy=False)

    ## add back stars that formed between snapshots
    if 'AgeGyr' in keys_to_extract and 'AgeGyr' in next_sub_snap:
        ## find stars w/ age in last snapshot < snapshot spacing
        new_star_mask = next_sub_snap['AgeGyr'] < (t1-t0)
        next_young_star_snap = filterDictionary(next_sub_snap,new_star_mask)

        next_young_star_df = convertSnapToDF(next_young_star_snap,**pandas_kwargs)

        ## now we have to initialize the young stars in the prev_snap
        prev_young_star_snap = {}#copy.copy(next_young_star_snap)
        ## set their ages to be negative

        if extra_df is not None:
            prev_young_star_snap['AgeGyr'] = next_young_star_df['AgeGyr'] - (t1-t0)
            next_stars_with_parents_mask = next_young_star_df.index.isin(extra_df.index)
            next_young_star_df = next_young_star_df[next_stars_with_parents_mask]
            gas_parent_mask = extra_df.index.isin(next_young_star_df.index)

            ## filter and then reorder
            gas_parent_df = extra_df.loc[gas_parent_mask].loc[next_young_star_df.index]

            for key in next_young_star_df.keys():
                if key == 'AgeGyr': continue
                prev_young_star_snap[key] = gas_parent_df[key]
            
            del gas_parent_df, extra_df

            prev_young_star_df = pd.DataFrame(
                prev_young_star_snap,
                index=next_young_star_df.index)

        ## NOTE: could this be why i'm getting spherical shells of very young stars?
        ##  maybe I should be extrapolating instead...
        else: raise ValueError("Pass time_merged_gas_df to find progenitor gas particles")

        ## append the young stars to each dataframe
        next_df_snap_reduced = next_df_snap_reduced.append(next_young_star_df)
        prev_df_snap = prev_df_snap.append(prev_young_star_df)

        ## and for good measure let's re-sort now that we 
        ##  added new indices into the mix
        next_df_snap_reduced.sort_index(inplace=True)
        prev_df_snap.sort_index(inplace=True)

    ## merge rows of dataframes based on particle ID
    prev_next_merged_df_snap = prev_df_snap.join(
        next_df_snap_reduced,
        rsuffix='_next')
    prev_next_merged_df_snap = prev_next_merged_df_snap.loc[
        ~prev_next_merged_df_snap.index.duplicated(keep='first')]

    ## remove particles that do not exist in the next snapshot, 
    ##  difficult to tell if they turned into stars or were merged. 
    ## NOTE: this was breaking when we were calculating spherical coords--
    ##  like half the rows were being thrown out??
    #prev_next_merged_df_snap = prev_next_merged_df_snap.dropna()
    ## TODO: should extrapolate missing particles in each snapshot

    ## add jhat rotation angle
    if polar:
        rotation_angle = 0
        for i in range(3):
            rotation_angle += prev_next_merged_df_snap['polarjhats_%d'%i]*prev_next_merged_df_snap['polarjhats_%d_next'%i]
        rotation_angle = np.arccos(rotation_angle)

        ## doing it this way will automatically add 
        ##  the interpolated jhat_rotangle to the interp_snap in make_interpolated_snap
        prev_next_merged_df_snap['jhat_rotangle_next'] = rotation_angle
        prev_next_merged_df_snap['jhat_rotangle'] = np.zeros(rotation_angle.values.shape)

    prev_next_merged_df_snap.first_time = t0
    prev_next_merged_df_snap.next_time = t1

    return prev_next_merged_df_snap

def make_interpolated_snap(time_merged_df,t,polar=True):
    
    interp_snap = {}

    t0,t1 = time_merged_df.first_time,time_merged_df.next_time
    ## create a new snapshot with linear interpolated values
    ##  between key and key_next using t0, t1, and t.
    for key in time_merged_df.keys():
        if '_next' in key: continue
        elif 'polarjhat' in key: continue ## will allow jhat_rotangle though
        interp_snap[key] = linear_interpolate(
            getattr(time_merged_df,key),
            getattr(time_merged_df,key+'_next'),
            t0,t1,t).values

    ## interpolate coordinates using higher order/more complicated interpolations
    ##  will remove the Coordinates_i and Velocities_i keys from interp_snap
    interp_snap['Coordinates'],interp_snap['Velocities'] = interpolate_position(
        t,t0,t1,
        time_merged_df,
        interp_snap,
        polar=polar)

    ## remove stars that have not formed yet
    if 'AgeGyr' in interp_snap: interp_snap = filterDictionary(interp_snap,interp_snap['AgeGyr']>0)
 
    return interp_snap

def linear_interpolate(
    x0,x1,
    t0,t1,
    t):
    return x0 + (x1-x0)/(t1-t0)*(t-t0)

def interpolate_position(t,t0,t1,time_merged_df,interp_snap,polar=True):

    if not polar:
        coords = np.zeros((time_merged_df.shape[0],3))
        coords[:,0] = interp_snap.pop('Coordinates_0')
        coords[:,1] = interp_snap.pop('Coordinates_1')
        coords[:,2] = interp_snap.pop('Coordinates_2')
        vels = np.zeros((time_merged_df.shape[0],3))
        vels[:,0] = interp_snap.pop('Velocities_0')
        vels[:,1] = interp_snap.pop('Velocities_1')
        vels[:,2] = interp_snap.pop('Velocities_2')
        return coords,vels
    else:
        ## doesn't actually have z because we've rotated into jhat plane
        rpz_interp_coords = np.zeros((time_merged_df.shape[0],2))
        rpz_interp_vels = np.zeros((time_merged_df.shape[0],2))

        first_jhats = np.zeros((time_merged_df.shape[0],3))
        next_jhats = np.zeros((time_merged_df.shape[0],3))

        ## interpolate r coordinate
        coord_key = 'polarjhatCoordinates_0'
        vel_key = 'polarjhatVelocities_0'
        first_Rs = getattr(time_merged_df,coord_key).values
        next_Rs = getattr(time_merged_df,coord_key+'_next').values
        rpz_interp_coords[:,0], rpz_interp_vels[:,0] = interpolate_at_order(
            first_Rs,
            next_Rs,
            getattr(time_merged_df,vel_key).values,
            getattr(time_merged_df,vel_key+'_next').values,
            t,t0,t1) ## defaults to order=1

        ## interpolate phi coordinate
        coord_key = 'polarjhatCoordinates_1'
        vel_key = 'polarjhatVelocities_1'
        first_renorm = kms_to_kpcgyr/first_Rs ## convert to radians
        next_renorm = kms_to_kpcgyr/next_Rs ## convert to radians

        rpz_interp_coords[:,1], rpz_interp_vels[:,1] = interpolate_at_order(
            getattr(time_merged_df,coord_key).values,
            getattr(time_merged_df,coord_key+'_next').values,
            getattr(time_merged_df,vel_key).values*first_renorm, 
            getattr(time_merged_df,vel_key+'_next').values*next_renorm, 
            t,t0,t1,
            order=3,
            periodic=True,
            vels_renorm=(1/first_renorm,1/next_renorm))

        ## unpack flattened rpz arrays from pandas dataframe which did our id matching
        for i in range(3):
            jhat_key = 'polarjhats_%d'%i
            first_jhats[:,i] = getattr(time_merged_df,jhat_key).values
            next_jhats[:,i] = getattr(time_merged_df,jhat_key+'_next').values

        ## need to convert rpz_interp_coords and rpz_interp_vels from r' p' to x,y,z
        ##  do that by getting interpolated jhat vectors and then associated x',y' vectors
        return convert_rp_to_xyz(
            interp_snap,
            rpz_interp_coords,
            rpz_interp_vels,
            first_jhats,
            next_jhats)

def interpolate_at_order(
    this_first_coords,
    this_next_coords,
    this_first_vels,
    this_next_vels,
    t,t0,t1,
    order=1,
    periodic=False,
    vels_renorm=None):

    dt = (t-t0)
    dsnap = (t1-t0)
    time_frac = dt/dsnap ## 'tau' in Phil's notation

    if this_next_coords is not None:
        dcoord = this_next_coords - this_first_coords
        if periodic:
            ## how far would we guess each particle goes at tfirst?
            ##  let's guess how many times it actually went around, basically
            ##  want to determine which of (N)*2pi + dcoord  
            ##  or (N+1)*2pi + dcoord, or (N-1)*2pi + dcoord is 
            ##  closest to approx_radians, (N = approx_radians//2pi)
            dcoord = guess_windings(dcoord,this_first_vels*dsnap,2*np.pi)
    else:
        ## handle extrapolation case
        dcoord = this_first_vels*dsnap
        #if periodic:dcoord = np.mod(dcoord,2*np.pi)
    
    ## basic linear interpolation
    if order == 1:
        interp_coords = this_first_coords + dcoord*time_frac
    ## correction factor to minimize velocity difference, apparently
    elif order == 2:
        x2 = (this_next_vels - this_first_vels)/2 * dsnap
        x1 = dcoord - x2
        interp_coords = this_first_coords + x1*time_frac * x2*time_frac*time_frac
    ## "enables exact matching of x,v but can 'overshoot' " - phil
    elif order == 3:
        x2 = 3*dcoord - (2*this_first_vels+this_next_vels)*dsnap
        x3 = -2*dcoord + (this_first_vels+this_next_vels)*dsnap
        interp_coords = this_first_coords + this_first_vels*dt + x2*time_frac**2 + x3*time_frac**3
    else: raise Exception("Bad order, should be 1,2, or 3")

    ## when periodic = True need to convert velocities back to 
    ##  km/s when we do the interpolation below
    if vels_renorm is not None:
        this_first_vels = this_first_vels * vels_renorm[0]
        this_next_vels = this_next_vels * vels_renorm[1]
    ## do a simple 1st order interpolation for the velocities
    ##  while we have them in scope
    return interp_coords, this_first_vels + (this_next_vels - this_first_vels)*(t-t0)/(t1-t0)

def convert_rp_to_xyz(
    interp_snap,
    rpz_interp_coords,
    rpz_interp_vels,
    first_jhats,
    next_jhats):

    ## need to convert rpz_interp_coords and rpz_interp_vels from r' p' to x,y,z
    ##  do that by getting interpolated jhat vectors and then associated x',y' vectors
    rotangle = interp_snap.pop('jhat_rotangle') ## interpolated value computed in calling function

    khats = np.cross(first_jhats,next_jhats) ## vector about which the vector is rotated
    ohats = np.cross(khats,first_jhats) ## vector pointing from ji toward jf in plane of rotation

    ## make sure all our vectors are normalized properly 
    first_jhats = first_jhats/np.linalg.norm(first_jhats,axis=1)[:,None]
    khats = khats/np.linalg.norm(khats,axis=1)[:,None]
    ohats = ohats/np.linalg.norm(ohats,axis=1)[:,None]

    ## last term is canceled b.c. by definition k . j_i = 0
    interp_jhats = first_jhats*np.cos(rotangle)[:,None] + ohats*np.sin(rotangle)[:,None] #+ khats*(np.dot(khats,first_jhats))*(1-np.cos(rotangle))
    interp_jhats/=np.linalg.norm(interp_jhats,axis=1)[:,None]

    xprimehats,yprimehats = get_primehats(interp_jhats) ## x' & y' in x,y,z coordinates

    xyz_interp_vels = np.zeros(rpz_interp_vels.shape)
    xyz_interp_coords = np.zeros(rpz_interp_coords.shape)

    ## replace rp w/ xy
    xyz_interp_vels[:,0] +=  rpz_interp_vels[:,0]*np.cos(rpz_interp_coords[:,1]) #  vr * cos(phi)
    xyz_interp_vels[:,1] +=  rpz_interp_vels[:,0]*np.sin(rpz_interp_coords[:,1]) #  vr * sin(phi)
    xyz_interp_vels[:,0] += -rpz_interp_vels[:,1]*np.sin(rpz_interp_coords[:,1]) # -vphi * sin(phi)
    xyz_interp_vels[:,1] +=  rpz_interp_vels[:,1]*np.cos(rpz_interp_coords[:,1]) #  vphi * cos(phi)

    xyz_interp_coords[:,0] = rpz_interp_coords[:,0]*np.cos(rpz_interp_coords[:,1]) # r * cos(phi)
    xyz_interp_coords[:,1] = rpz_interp_coords[:,0]*np.sin(rpz_interp_coords[:,1]) # r * sin(phi)

    ## unit primehat vectors are in simulation coordinate frame, multiply by 
    ##  components in jhat frame to get simulation coordinates
    coords = xyz_interp_coords[:,0,None]*xprimehats + xyz_interp_coords[:,1,None]*yprimehats
    vels = xyz_interp_vels[:,0,None]*xprimehats + xyz_interp_vels[:,1,None]*yprimehats

    ## check for rotational support, inspired by Phil's routine
    ## average 1D velocity between snapshots
    #avg_vels2 = (rpz_first_vels**2+rpz_next_vels**2)/2
    ## time interpolated 1D velocity 
    avg_vels2 = rpz_interp_vels**2
    norms2 = np.sum(avg_vels2,axis=1)

    ## non-rotationally supported <==> |vphi|/|v| < 0.5; |vphi| comes from sqrt above
    ##  make a mask for those particles which are not rotationally supported and need
    ##  to be replaced with a simple cartesian interpolation
    support_thresh = 0.50  ## 0.5
    non_rot_support = np.logical_or(
        avg_vels2[:,1]/norms2 < support_thresh,
        rpz_interp_coords[:,0] > 30)

    for i in range(3): 
        ## replace the masked coordinates with their simulation-cartesian interpolated counterparts
        ##  computed in the loop in our calling function (which put them into interp_snap in the first place)
        these_coords = interp_snap.pop('Coordinates_%d'%i)
        these_vels = interp_snap.pop('Velocities_%d'%i)
        if np.any(non_rot_support): 
            coords[non_rot_support][:,i] = these_coords[non_rot_support]
            vels[non_rot_support][:,i] = these_vels[non_rot_support]
    
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
