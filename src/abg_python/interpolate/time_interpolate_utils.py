import numpy as np

from ..snapshot_utils import convertSnapToDF
from ..galaxy.gal_utils import Galaxy
from ..array_utils import filterDictionary

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
    t1=None):
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
        cylindrical_coordinates=True,
        spherical_coordinates=True,
        total_metallicity_only=True,
        specific_coords=True)
    
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

    return prev_next_merged_df_snap

def make_interpolated_snap(
    t,
    time_merged_df,
    t0,
    t1,
    mode='cylindrical'):
    interp_snap = {}

    ## create a new snapshot with linear interpolated values
    ##  between key and key_next using t0, t1, and t.
    for key in time_merged_df.keys():
        if '_next' in key:
            continue
        elif key in [
            'coord_xs','coord_ys','coord_zs',
            'coord_Rs','coord_rs',
            'coord_thetas','coord_phis',]:
            #'vxs','vys','vzs',
            #'vrs','vRs',
            #'vthetas','vphis']:

            continue
        interp_snap[key] = linear_interpolate(
            getattr(time_merged_df,key),
            getattr(time_merged_df,key+'_next'),
            t0,t1,
            t).values

    ckeys = {'cartesian':['coord_xs','coord_ys','coord_zs'],
            'spherical':['coord_rs','coord_thetas','coord_phis'],
            'cylindrical':['coord_Rs','coord_R_phis','coord_R_zs']}[mode]

    vkeys = {'cartesian':['vxs','vys','vzs'],
            'spherical':['vrs','vthetas','vphis'],
            'cylindrical':['vRs','vRphis','vRzs']}[mode]

    first_coords = np.zeros((time_merged_df.shape[0],3))
    next_coords = np.zeros((time_merged_df.shape[0],3))

    first_vels = np.zeros((time_merged_df.shape[0],3))
    next_vels = np.zeros((time_merged_df.shape[0],3))

    vels = np.zeros(coords.shape)

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
        mode=mode)
    
    if 'InvRotationMatrix_0' in time_merged_df and mode in ['spherical','cylindrical']:
        ## reconstruct the inverse rotation matrices
        inv_rot_matrices = np.zeros((first_coords.shape[0],9))
        for i in range(9):
            this_key = "InvRotationMatrix_%d"%i
            ## explicitly assume that L does not change much
            ##  by averaging the rotation matrices (yikes)
            inv_rot_matrices[:,i] = (time_merged_df[this_key]+time_merged_df[this_key+'_next']).values/2

        inv_rot_matrices = inv_rot_matrices.reshape(-1,3,3)

        coords = (inv_rot_matrices*coords[:,None]).sum(-1)
        vels = (inv_rot_matrices*vels[:,None]).sum(-1)

    interp_snap['Coordinates'] = coords
    interp_snap['Velocities'] = vels

    ## remove stars that will form between this and the next snapshot
    ##  but haven't formed yet
    if 'AgeGyr' in interp_snap: interp_snap = filterDictionary(interp_snap,interp_snap['AgeGyr']>0)

    return interp_snap

def interpolate_coords(
    first_coords,
    first_vels,
    next_coords,
    next_vels,
    t,t0,t1,
    mode='cylindrical'):

    delta_t = (t-t0)
    time_frac = delta_t/(t1-t0)

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
        if mode == 'spherical':
            phi_index = 2
            theta_index = 1
        elif mode == 'cylindrical':
            phi_index = 1
            theta_index = None
        else:
            phi_index = None
            theta_index = None

        if phi_index is not None:
            vbar = vbar/rtp_coords[:,0]
            ## handle partial windings, change sign as necessary
            if i == phi_index:
                dcoord[(dcoord>0) & (vbar < 0)]-=2*np.pi
                dcoord[(dcoord<0) & (vbar > 0)]+=2*np.pi

            ## handle full windings
            if i == phi_index or (theta_index is not None and i == theta_index): 
                nwindings = ((vbar*(t1-t0))/(2*np.pi)).astype(int)
                dcoord += nwindings*2*np.pi

        rtp_coords[:,i] = (this_first_coords + dcoord*time_frac)

    ## cast spherical coordinates to cartesian coordinates
    if mode=='spherical': 
        coords[:,0] = rtp_coords[:,0] * np.sin(rtp_coords[:,1]) * np.cos(rtp_coords[:,2])
        coords[:,1] = rtp_coords[:,0] * np.sin(rtp_coords[:,1]) * np.sin(rtp_coords[:,2])
        coords[:,2] = rtp_coords[:,0] * np.cos(rtp_coords[:,1])
    ## cast cylindrical coordinates to cartesian coordinates
    elif mode == 'cylindrical':
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