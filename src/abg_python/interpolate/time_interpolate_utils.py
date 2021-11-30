import numpy as np

from ..snapshot_utils import convertSnapToDF
from ..galaxy.gal_utils import Galaxy

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
    extra_keys_to_extract=None,
    extra_arrays_function=None):
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

    print('Creating a merged DF')
    keys_to_extract = ['Coordinates','Masses','SmoothingLength','ParticleIDs','ParticleChildIDsNumber']
    if extra_keys_to_extract is not None:
        keys_to_extract += list(extra_keys_to_extract)

    ## convert snapshot dictionaries to pandas dataframes
    prev_df_snap = convertSnapToDF(
        prev_sub_snap,
        keys_to_extract=keys_to_extract,
        spherical_coordinates=True)
    
    ## apparently index operations go faster if you sort by index
    prev_df_snap.sort_index(inplace=True)
    
    next_df_snap = convertSnapToDF(
        next_sub_snap,
        keys_to_extract=keys_to_extract,
        spherical_coordinates=True)
    
    ## apparently index operations go faster if you sort by index
    next_df_snap.sort_index(inplace=True)
        
    ## remove particles that do not exist in the previous snapshot, 
    ##  difficult to determine which particle they split from
    next_df_snap_reduced = next_df_snap.reindex(prev_df_snap.index,copy=False)
    

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
    prev_next_merged_df_snap = prev_next_merged_df_snap.dropna()

    return prev_next_merged_df_snap

def make_interpolated_snap(t,time_merged_df,t0,t1,spherical=True):
    interp_snap = {}

    ## create a new snapshot with linear interpolated values
    ##  between key and key_next using t0, t1, and t.
    for key in time_merged_df.keys():
        if '_next' in key:
            continue
        elif key in [
            'coord_xs','coord_ys','coord_zs',
            'vxs','vys','vzs',
            'coord_rs','coord_thetas','coord_phis',
            'vrs','vthetas','vphis']:
            continue
        interp_snap[key] = linear_interpolate(
            getattr(time_merged_df,key),
            getattr(time_merged_df,key+'_next'),
            t0,t1,
            t).values
    
    coords = np.zeros((time_merged_df.shape[0],3))
    rtp_coords = np.zeros((time_merged_df.shape[0],3))
    #vels = np.zeros((time_merged_df.shape[0],3))

    ckeys = ['coord_xs','coord_ys','coord_zs'] if not spherical else ['coord_rs','coord_thetas','coord_phis']
    vkeys = ['vxs','vys','vzs'] if not spherical else ['vrs','vthetas','vphis']

    delta_t = (t-t0)
    time_frac = delta_t/(t1-t0)
    for i,(coord_key,vel_key) in enumerate(zip(ckeys,vkeys)):


        first_coords = getattr(time_merged_df,coord_key)
        next_coords = getattr(time_merged_df,coord_key+'_next')

        first_vels = getattr(time_merged_df,vel_key)
        next_vels = getattr(time_merged_df,vel_key+'_next')
        vbar = (first_vels+next_vels)/2.

        
        dcoord = next_coords - first_coords

        if spherical:
            vbar = vbar/rtp_coords[:,0]
            ## handle partial windings, change sign as necessary
            if i == 2:
                dcoord[(dcoord>0) & (vbar < 0)]-=2*np.pi
                dcoord[(dcoord<0) & (vbar > 0)]+=2*np.pi

            ## handle full windings
            if i > 0: 
                nwindings = ((vbar.values*(t1-t0))/(2*np.pi)).astype(int)
                dcoord += nwindings*2*np.pi

        rtp_coords[:,i] = (first_coords + dcoord*time_frac).values

    if not spherical: coords = rtp_coords
    else: 
        ## cast spherical coordinates to cartesian coordinates
        coords[:,0] = rtp_coords[:,0] * np.sin(rtp_coords[:,1]) * np.cos(rtp_coords[:,2])
        coords[:,1] = rtp_coords[:,0] * np.sin(rtp_coords[:,1]) * np.sin(rtp_coords[:,2])
        coords[:,2] = rtp_coords[:,0] * np.cos(rtp_coords[:,1])
        
    interp_snap['Coordinates'] = coords

    return interp_snap

def linear_interpolate(
    x0,x1,
    t0,t1,
    t):
    return x0 + (x1-x0)/(t1-t0)*(t-t0)