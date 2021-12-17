import numpy as np
import pandas as pd

from ..snapshot_utils import convertSnapToDF
from ..galaxy.gal_utils import Galaxy
from ..array_utils import filterDictionary
from ..math_utils import getThetasTaitBryan, rotateEuler
from ..math_utils import get_cylindrical_velocities,get_cylindrical_coordinates,get_spherical_coordinates,get_spherical_velocities

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
    coord_interp_mode='spherical'):
    """
        if you use Metallicity  or Velocities then the keys will be different when you try to access them
          in your render call using render_kwargs.
        Velocities -> vx,vy,vz 
        Metallicity -> met0,met1,met2,...
    keys_to_extract = ['Coordinates','Masses','SmoothingLength','ParticleIDs','ParticleChildIDsNumber']
    """
    
    ## note, it might make more sense for them to compute it on the interpolated snapshot?
    ##  i guess it depends if they want to interpolate their own thing or compute their thing
    ##  on an interpolated quantity(ies)
    if extra_arrays_function is not None:
        raise NotImplementedError("Need to allow users to pass in a function that will compute"+
            " arbitrary quantity arrays from stuff computed in the snapshot.")

    if keys_to_extract is None: keys_to_extract = []
    keys_to_extract += ['Coordinates','Masses','SmoothingLength','ParticleIDs','ParticleChildIDsNumber']

    pandas_kwargs = dict(
        keys_to_extract=keys_to_extract,
        #cylindrical_coordinates=coord_interp_mode=='cylindrical',
        #spherical_coordinates=coord_interp_mode=='spherical',
        total_metallicity_only=True)
    
    ## convert snapshot dictionaries to pandas dataframes
    prev_df_snap = convertSnapToDF(prev_sub_snap,**pandas_kwargs)
    
    next_df_snap = convertSnapToDF(next_sub_snap,**pandas_kwargs)
        
    ## remove particles that do not exist in the previous snapshot, 
    ##  difficult to determine which particle they split from
    next_df_snap_reduced = next_df_snap.reindex(prev_df_snap.index,copy=False)

    ## add back stars that formed between snapshots
    if 'AgeGyr' in keys_to_extract:
        ## find stars w/ age in last snapshot < snapshot spacing
        new_star_mask = next_sub_snap['AgeGyr'] < (t1-t0)
        next_young_star_snap = filterDictionary(next_sub_snap,new_star_mask)
        next_young_star_snap['AgeGyr'] = t0 - next_young_star_snap['AgeGyr']
        next_young_star_df = convertSnapToDF(next_young_star_snap,**pandas_kwargs)

        next_df_snap_reduced = next_df_snap_reduced.append(next_young_star_df)
        next_df_snap_reduced.sort_index(inplace=True)

    ## merge rows of dataframes based on 
    """
    lol this used to work, obviously, but now it
    raises an internal error when trying to extract the
    multi-indices. pandas sucks.

    prev_next_merged_df_snap = pd.merge(
        prev_df_snap,
        next_df_snap_reduced,
        how='inner',
        on=prev_df_snap.index,
        suffixes=('','_next'),
        copy=False).set_index('key_0')
    """
    prev_next_merged_df_snap = prev_df_snap.join(
        next_df_snap_reduced,
        rsuffix='_next')

    ## remove particles that do not exist in the next snapshot, 
    ##  difficult to tell if they turned into stars or were merged. 
    ## NOTE: this was breaking when we were calculating spherical coords--
    ##  like half the rows were being thrown out??
    #prev_next_merged_df_snap = prev_next_merged_df_snap.dropna()

    if coord_interp_mode in ['cylindrical','spherical']:
        Ls = np.zeros((prev_next_merged_df_snap.shape[0],3))
        this_coords = np.zeros((prev_next_merged_df_snap.shape[0],3))
        this_vels = np.zeros((prev_next_merged_df_snap.shape[0],3))
        axes = ['xs','ys','zs']

        ## find the average of the momentum vectors between the two snapshots
        for suffix in ['','_next']:
            masses = prev_next_merged_df_snap['Masses'+suffix].values
            for i in range(3):
                this_coords[:,i] = prev_next_merged_df_snap['coord_%s'%(axes[i])+suffix]
                this_vels[:,i] = prev_next_merged_df_snap['v%s'%(axes[i])+suffix]

            ## find angular momentum frame for each particle
            Ls+= np.cross(this_coords,this_vels*masses[:,None])
        Ls/=2 

        ## get rotation matrices
        theta_TBs,phi_TBs = getThetasTaitBryan(Ls.T)
        rot_matrices = rotateEuler(theta_TBs,phi_TBs,0,None,loud=False)
        ## reshape to be Npart x 3 x 3
        rot_matrices = np.rollaxis(rot_matrices,-1,0)
        
        ## fill a dictionary with the rotated coordinates
        copy_snap = {}
        for suffix in ['','_next']:
            ## let's make new memory buffers each time just to make sure the 
            ##  dataframe doesn't end up aliasing particle positions :\
            this_coords = np.zeros((prev_next_merged_df_snap.shape[0],3))
            this_vels = np.zeros((prev_next_merged_df_snap.shape[0],3))

            ## open this snapshot's coordinates and velocities
            for i in range(3):
                ckey,vkey = 'coord_%s'%(axes[i])+suffix,'v%s'%(axes[i])+suffix
                this_coords[:,i] = prev_next_merged_df_snap[ckey]
                this_vels[:,i] = prev_next_merged_df_snap[vkey]

            ## trick for multiply N matrices by N vectors pairwise
            this_coords = (rot_matrices*this_coords[:,None]).sum(-1)
            this_vels = (rot_matrices*this_vels[:,None]).sum(-1)

            ## did we request spherical coordinates?
            if coord_interp_mode == 'spherical':
                (copy_snap['coord_rs'+suffix],
                copy_snap['coord_thetas'+suffix],
                copy_snap['coord_phis'+suffix]) = get_spherical_coordinates(
                    this_coords)

                (copy_snap['vrs'+suffix],
                copy_snap['vthetas'+suffix],
                copy_snap['vphis'+suffix]) = get_spherical_velocities(
                    this_vels,
                    this_coords)

            ## did we request cylindrical coordinates?
            if coord_interp_mode == 'cylindrical':
                (copy_snap['coord_Rs'+suffix],
                copy_snap['coord_R_phis'+suffix],
                copy_snap['coord_R_zs'+suffix]) = get_cylindrical_coordinates(
                    this_coords)

                (copy_snap['vRs'+suffix],
                copy_snap['vRphis'+suffix],
                copy_snap['vRzs'+suffix]) = get_cylindrical_velocities(
                    this_vels,
                    this_coords)

        ## append these columns to the dataframe
        prev_next_merged_df_snap = prev_next_merged_df_snap.join(
            pd.DataFrame(copy_snap,index=prev_next_merged_df_snap.index))
                
        ## we'll need the inverse rotation matrices
        ##  later, so let's store them as the inverse (transpose)
        rot_matrices = np.transpose(rot_matrices,axes=(0,2,1))

        ## however, we need to append rotation matrix elements as 9 extra columns -__-
        ##  because pandas doesn't like 2D arrays as elements
        prev_next_merged_df_snap = prev_next_merged_df_snap.join(pd.DataFrame(
            rot_matrices.reshape(rot_matrices.shape[0],-1), ## flatten from 3d to 2d array
            columns=['InvRotationMatrix_%d'%i for i in range(9)],
            index=prev_next_merged_df_snap.index))

    return prev_next_merged_df_snap

def make_interpolated_snap(
    t,
    time_merged_df,
    t0,
    t1,
    coord_interp_mode='spherical'):
    
    interp_snap = {}

    ## create a new snapshot with linear interpolated values
    ##  between key and key_next using t0, t1, and t.
    for key in time_merged_df.keys():
        if '_next' in key:
            continue
        elif (key in [
            'coord_xs','coord_ys','coord_zs',
            'coord_rs','coord_thetas','coord_phis',
            'coord_Rs','coord_R_phis','coord_R_zs'
            ] or 'InvRotationMatrix_' in key):
            #'vxs','vys','vzs',
            #'vrs','vthetas','vphis'
            # 'vRs','vRphis','vRzs']:

            continue
        interp_snap[key] = linear_interpolate(
            getattr(time_merged_df,key),
            getattr(time_merged_df,key+'_next'),
            t0,t1,
            t).values

    ckeys = {'cartesian':['coord_xs','coord_ys','coord_zs'],
            'spherical':['coord_rs','coord_thetas','coord_phis'],
            'cylindrical':['coord_Rs','coord_R_phis','coord_R_zs']}[coord_interp_mode]

    vkeys = {'cartesian':['vxs','vys','vzs'],
            'spherical':['vrs','vthetas','vphis'],
            'cylindrical':['vRs','vRphis','vRzs']}[coord_interp_mode]

    first_coords = np.zeros((time_merged_df.shape[0],3))
    next_coords = np.zeros((time_merged_df.shape[0],3))

    first_vels = np.zeros((time_merged_df.shape[0],3))
    next_vels = np.zeros((time_merged_df.shape[0],3))

    vels = np.zeros(first_coords.shape)

    for i,(coord_key,vel_key) in enumerate(zip(ckeys,vkeys)):
        first_coords[:,i] = getattr(time_merged_df,coord_key)
        next_coords[:,i] = getattr(time_merged_df,coord_key+'_next')

        first_vels[:,i] = getattr(time_merged_df,vel_key)
        next_vels[:,i] = getattr(time_merged_df,vel_key+'_next')

        vels[:,i] = interp_snap[vel_key]

    coords = interpolate_coords(
        first_coords,
        first_vels,
        next_coords,
        next_vels,
        t,t0,t1,
        coord_interp_mode=coord_interp_mode)
    
    if 'InvRotationMatrix_0' in time_merged_df:# and coord_interp_mode in ['spherical','cylindrical']:
        ## reconstruct the inverse rotation matrices
        inv_rot_matrices = np.zeros((first_coords.shape[0],9))
        for i in range(9):
            this_key = "InvRotationMatrix_%d"%i
            inv_rot_matrices[:,i] = time_merged_df[this_key]

        inv_rot_matrices = inv_rot_matrices.reshape(-1,3,3)

        coords = (inv_rot_matrices*coords[:,None]).sum(-1)
        vels = (inv_rot_matrices*vels[:,None]).sum(-1)

    interp_snap['Coordinates'] = coords
    interp_snap['Velocities'] = vels

    ## remove stars that will form between this and the next snapshot
    ##  but haven't formed yet
    ## TODO what is the AgeGyr value for stars that haven't formed w/o the _next prefix??
    if 'AgeGyr' in interp_snap: interp_snap = filterDictionary(interp_snap,interp_snap['AgeGyr']>0)
 
    return interp_snap

def interpolate_coords(
    first_coords,
    first_vels,
    next_coords,
    next_vels,
    t,t0,t1,
    coord_interp_mode='spherical'):

    delta_t = (t-t0)
    dsnap = (t1-t0)
    time_frac = delta_t/dsnap

    coords = np.zeros((first_coords.shape))
    rtp_coords = np.zeros((first_coords.shape))
    
    for i in range(3):
        this_first_coords = first_coords[:,i]
        this_next_coords = next_coords[:,i]

        this_first_vels = first_vels[:,i]
        this_next_vels = next_vels[:,i]
        
        vbar = (this_first_vels+this_next_vels)/2.

        
        dcoord = this_next_coords - this_first_coords

        ## r theta phi
        if coord_interp_mode == 'spherical':
            phi_index = 2
            r2d = rtp_coords[:,0]*np.sin(rtp_coords[:,1])
        elif coord_interp_mode == 'cylindrical':
            phi_index = 1
            r2d = rtp_coords[:,0]
        else:
            phi_index = None

        if i == phi_index: 
            ## relevant angular velocity is v/r2d
            vbar = vbar/r2d

            ## how far would we guess each particle could go?
            approx_radians = vbar*dsnap

            ## let's guess how many times it went around, basically
            ##  want to determine which of (N)*2pi + dcoord  
            ##  or (N+1)*2pi + dcoord, or (N-1)*2pi + dcoord is 
            ##  closest to approx_radians, (N = approx_radians//2pi)
            dcoord = guess_windings(dcoord,approx_radians,2*np.pi)

        rtp_coords[:,i] = (this_first_coords + dcoord*time_frac)

    ## cast spherical coordinates to cartesian coordinates
    if coord_interp_mode=='spherical': 
        coords[:,0] = rtp_coords[:,0] * np.sin(rtp_coords[:,1]) * np.cos(rtp_coords[:,2])
        coords[:,1] = rtp_coords[:,0] * np.sin(rtp_coords[:,1]) * np.sin(rtp_coords[:,2])
        coords[:,2] = rtp_coords[:,0] * np.cos(rtp_coords[:,1])
    ## cast cylindrical coordinates to cartesian coordinates
    elif coord_interp_mode == 'cylindrical':
        coords[:,0] = rtp_coords[:,0] * np.cos(rtp_coords[:,1])
        coords[:,1] = rtp_coords[:,0] * np.sin(rtp_coords[:,1])
        coords[:,2] = rtp_coords[:,2]
    else: coords = rtp_coords

    return coords

def linear_interpolate(
    x0,x1,
    t0,t1,
    t):
    return x0 + (x1-x0)/(t1-t0)*(t-t0)


def guess_windings(dphi,vphi_dt,period=1):
    base_windings = vphi_dt//period
    ## now we have to decide, do we go back a whole winding
    ##  to match dphi or do we go forward part of a winding?
    guesses = (base_windings+np.array([-1,0,1])[:,None])*period+(dphi-vphi_dt)
    
    ## take advantage of the fact that options are -1,0,1, turn corresponding indices from 0,1,2 -> -1,0,1
    adjust_windings = np.argmin(np.abs(guesses),axis=0)-1

    return (base_windings+adjust_windings)*period+dphi