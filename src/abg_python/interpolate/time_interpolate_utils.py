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

def single_threaded_control_flow(
    function_to_call_on_interpolated_dataframe,
    times,
    snap_pairs,
    snap_pair_times,
    galaxy_kwargs,
    extra_keys_to_extract):
    """ """

    prev_galaxy,next_galaxy = None,None
    prev_snapnum,next_snapnum = None,None

    return_values = []

    for i,(pair,pair_times) in enumerate(zip(snap_pairs,snap_pair_times)):     
        if not i%10: print(i,pair,pair_times)
        ## determine if the galaxies in the pair are actually
        ##  changed, and if so, open their data from the disk.
        prev_galaxy,next_galaxy,changed = load_gals_from_disk(
            prev_snapnum,next_snapnum,
            pair,
            prev_galaxy,
            next_galaxy,
            **galaxy_kwargs)

        ## update the previous/next snapnums
        prev_snapnum,next_snapnum = pair

        this_time = times[i]
        
        if changed:
            ## make an interpolated snapshot with these galaxies,
            ##  this takes a while so we'll hold onto it and only 
            ##  make a new one if necessary.
            t0,t1 = pair_times
            time_merged_df = index_match_snapshots_with_dataframes(
                prev_galaxy.sub_snap,
                next_galaxy.sub_snap,
                extra_keys_to_extract=extra_keys_to_extract)

        ## update the interp_snap with new values for the new time
        interp_snap = make_interpolated_snap(this_time,time_merged_df,t0,t1)
        interp_snap['prev_snapnum'] = prev_snapnum
        interp_snap['next_snapnum'] = next_snapnum
        interp_snap['prev_time'] = t0
        interp_snap['next_time'] = t1
        interp_snap['this_time'] = this_time
        interp_snap['name'] = next_galaxy.name 
        ## let's put the FIREstudio projections into a sub-directory of our Galaxy class instance
        interp_snap['studio_datadir'] = next_galaxy.datadir

        ## call the function we were passed
        return_values += [function_to_call_on_interpolated_dataframe(interp_snap)]
    return return_values

def load_gals_from_disk(
    prev_snapnum,next_snapnum,
    pair,
    prev_galaxy,next_galaxy,
    testing=False,
    **kwargs):
    """ Determines whether it needs to load a new galaxy from disk
        or if we already have what we need."""

    ## -- check the prev galaxy
    ## keep the current snapnum
    if pair[0] == prev_snapnum:
        prev_galaxy=prev_galaxy
    ## step forward in time, swap pointers
    elif pair[0] == next_snapnum:
        prev_galaxy = next_galaxy
        next_galaxy = None
    ## will need to read from disk
    else:
        prev_galaxy = None
    
    ## -- now the next galaxy
    ## keep the current snapnum
    if pair[1] == next_snapnum:
        next_galaxy = next_galaxy
    ## will need to read from disk
    else:
        next_galaxy = None

    changed = False ## flag for whether we loaded something from disk
    if prev_galaxy is None:
        print('loading',pair[0],'from disk')
        if not testing:
            prev_galaxy = Galaxy(snapnum=pair[0],**kwargs)
            prev_galaxy.extractMainHalo()
        else: prev_galaxy = pair[0]
        changed = True
    if next_galaxy is None:
        print('loading',pair[1],'from disk')
        if not testing:
            next_galaxy = Galaxy(snapnum=pair[1],**kwargs)
            next_galaxy.extractMainHalo()
        else: next_galaxy = pair[1]
        changed = True
        
    return prev_galaxy,next_galaxy,changed

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

def make_interpolated_snap(t,time_merged_df,t0,t1,velocity_interpolate=True,spherical=True):
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