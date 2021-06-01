import os
import time
import copy
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abg_python.snapshot_utils import convertSnapToDF
from abg_python.galaxy.gal_utils import Galaxy

class TimeInterpolationHandler(object):

    def __init__(
        self,
        snapnums,
        dGyr_or_times_gyr,
        snap_times_gyr=None,
        dGyr_tmin=None,
        dGyr_tmax=None):
        """ """

        print("TODO: in cache file projections from different times are only differentiated to Myr precision.")
        ## need to check/decide how one would convert to Myr if necessary. I think only
        ##  the snap interpolation would break.

        if snap_times_gyr is None:
            raise NotImplementedError("Need to open each snapshot and extract the time in Gyr")

        dGyr_or_times_gyr = np.array(dGyr_or_times_gyr)

        if len(dGyr_or_times_gyr.shape) == 0:
            ## we were passed a time spacing
            times_gyr,snap_pairs,snap_pair_times = find_bordering_snapnums(
                    snap_times_gyr,
                    dGyr=dGyr_or_times_gyr,
                    tmin=dGyr_tmin,
                    tmax=dGyr_tmax)

        elif len(dGyr_or_times_gyr) == 1:
            ## we were passed an array of times
            times_gyr = dGyr_or_times_gyr
            inds_next = np.argmax((times_gyr - snap_times_gyr[:,None]) < 0 ,axis=0)
            inds_prev = inds_next-1
            snap_pairs = np.array(list(zip(inds_prev,inds_next)))
            snap_pair_times = np.array(list(zip(
                snap_times_gyr[inds_prev],
                snap_times_gyr[inds_next])))

        else:
            raise ValueError("Could not interpret dGyr_or_times_gyr",dGyr_or_times_gyr)

        self.times_gyr = times_gyr
        self.snap_pairs = snap_pairs
        self.snap_pair_times = snap_pair_times

    def interpolate_on_snap_pairs(
        self,
        function_to_call_on_interpolated_dataframe,
        galaxy_kwargs=None,
        multi_threads=1,
        extra_keys_to_extract=None):
        """ """

        if galaxy_kwargs is None:
            galaxy_kwargs = {}

        if multi_threads == 1:

            return single_threaded_control_flow(
                function_to_call_on_interpolated_dataframe,
                self.times_gyr,
                self.snap_pairs,
                self.snap_pair_times,
                galaxy_kwargs,
                extra_keys_to_extract)
            
        elif multi_threads > 1:
            ## split the pairs of snapshots into approximately equal chunks
            ##  prioritizing  matching pairs of snapshots
            mps_indices = split_into_n_approx_equal_chunks(self.snap_pairs,multi_threads)
            split_times_gyr = np.array_split(self.times_gyr,mps_indices)
            split_snap_pairs = np.array_split(self.snap_pairs,mps_indices)
            split_snap_pair_times = np.array_split(self.snap_pair_times,mps_indices)
            
            argss = zip(
                itertools.repeat(function_to_call_on_interpolated_dataframe),
                split_times_gyr,
                split_snap_pairs,
                split_snap_pair_times,
                itertools.repeat(galaxy_kwargs),
                itertools.repeat(extra_keys_to_extract))

            with multiprocessing.Pool(multi_threads) as my_pool:
                    my_pool.starmap(single_threaded_control_flow,argss)
        else:
            raise ValueError("Specify a number of threads >=1, not",multi_threads)

def find_matching_split_indices(snap_pairs):
    pass

def find_bordering_snapnums(
    snap_times_gyr,
    dGyr=.005,
    tmin=None,
    tmax=None):
    """ """

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
    prev_snapnum,next_snapnum=None,None

    return_values = []

    for i,(pair,pair_times) in enumerate(zip(snap_pairs,snap_pair_times)):     
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

        ## call the function we were passed
        return_values += [function_to_call_on_interpolated_dataframe(interp_snap)]
    return return_values

def load_gals_from_disk(
    prev_snapnum,next_snapnum,
    pair,
    prev_galaxy,next_galaxy,
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
        prev_galaxy = Galaxy(snapnum=pair[0],**kwargs)
        prev_galaxy.extractMainHalo()
        changed = True
    if next_galaxy is None:
        print('loading',pair[1],'from disk')
        next_galaxy = Galaxy(snapnum=pair[1],**kwargs)
        next_galaxy.extractMainHalo()
        changed = True
        
    return prev_galaxy,next_galaxy,changed
        

def linear_interpolate(
    x0,x1,
    t0,t1,
    t):
    return x0 + (x1-x0)/(t1-t0)*(t-t0)

def make_interpolated_snap(t,time_merged_df,t0,t1):
    interp_snap = {}

    ## create a new snapshot with linear interpolated values
    ##  between key and key_next using t0, t1, and t.
    for key in time_merged_df.keys():
        if '_next' in key:
            continue
        elif key in ['coord_xs','coord_ys','coord_zs']:
            continue
        interp_snap[key] = linear_interpolate(
            getattr(time_merged_df,key),
            getattr(time_merged_df,key+'_next'),
            t0,t1,
            t).values
    
    ## handle coordinates separately
    ## TODO add {velocity correction}:
    ##  xt = x0 + (x1 - x0)f_t + {(1-f_t)(v0 (t-t0) + 1/2 (v1-v0)/(t1-t0) (t-t0)^2) } 
    coords = np.zeros((time_merged_df.shape[0],3))
    for i,key in enumerate(['coord_xs','coord_ys','coord_zs']):
        coords[:,i] = linear_interpolate(
            getattr(time_merged_df,key),
            getattr(time_merged_df,key+'_next'),
            t0,t1,
            t).values
        
    interp_snap['Coordinates'] = coords

    return interp_snap

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

    init=time.time()
    print('Creating a merged DF')
    keys_to_extract = ['Coordinates','Masses','SmoothingLength','ParticleIDs','ParticleChildIDsNumber']
    if extra_keys_to_extract is not None:
        keys_to_extract += list(extra_keys_to_extract)

    ## convert snapshot dictionaries to pandas dataframes
    prev_df_snap = convertSnapToDF(
        prev_sub_snap,
        keys_to_extract=keys_to_extract)
    
    ## apparently index operations go faster if you sort by index
    prev_df_snap.sort_index(inplace=True)
    
    next_df_snap = convertSnapToDF(
        next_sub_snap,
        keys_to_extract=keys_to_extract)
    
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

def split_into_n_approx_equal_chunks(snap_pairs,nchunks):
    
    nrenders = len(snap_pairs)
    ## split matching pairs into groups
    indices = split_pairs(snap_pairs[:])
    splitted = np.array_split(snap_pairs,indices)
    
    ## determine how many of each matching pair there are
    n_renders_per_split = [len(this_split) for this_split in splitted]
    
    mps_chunks = []
    this_chunk = 0
    # 1 would bias early, do this if the remainder would dump a lot of pairs on the last
    ##  process, i.e. when the remainder > nchunks/2
    per_chunk = nrenders//nchunks + (nrenders%nchunks > nchunks//2)
    
    for i in range(len(indices)):
        #print(this_chunk,per_chunk)
        if this_chunk >= per_chunk:
            mps_chunks+=[indices[i-1]]
            this_chunk=0
        this_chunk+=n_renders_per_split[i]
    
    mps_chunks = np.array(mps_chunks)#-indices[0]
    mps_chunks=list(mps_chunks)
    print('split into:',np.diff([0]+mps_chunks+[len(snap_pairs)]))   
    return mps_chunks

def split_pairs(snap_pairs):
    indices = []
    changed = split_head(snap_pairs)
    cur_index = 0
    while changed > 0:
        changed = split_head(snap_pairs[cur_index:])
        cur_index+=changed
        indices+=[cur_index]

    return indices[:-1]#np.array_split(snap_pairs,indices[:-1])
    
def split_head(snap_pairs):
    split_index = np.argmax(np.logical_not(np.all(snap_pairs==snap_pairs[0],axis=1)))
    return split_index

def main():

    savename = 'm10q_res250'
    snapdir = "/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m10q_res250/output"
    snapnum = 600 
    datadir = '/scratch/04210/tg835099/data/metal_diffusion'

    ## load up a galaxy object to get its snapnums and snap_gyrs attribtues
    prev_galaxy = Galaxy(
        savename,
        snapdir,
        snapnum,
        datadir=datadir)

    ## decide what we want to pass to the GasStudio
    render_kwargs = {
        'weight_name':'Masses',
        'quantity_name':'Temperature',
        #min_quantity=2,
        #max_quantity=7,
        'quantity_adjustment_function':np.log10,
        'quick':True,
        'min_weight':-0.5,
        'max_weight':3,
        'weight_adjustment_function':lambda x: np.log10(x/(30**2/1200**2)) + 10 - 6, ## msun/pc^2,
        'cmap':'afmhot',
        'quick':True}

    name = copy.copy(prev_Galaxy.name) ## copy to ensure there's not an errant reference to prev_galaxy
    ## let's put the FIREstudio projections into a sub-directory of our Galaxy class instance
    studio_datadir = os.path.join(os.path.dirname(next_galaxy.datadir),'firestudio')

    from firestudio.studios.gas_studio import GasStudio
    def my_func(interp_snap):

        my_gasStudio = GasStudio(
            studio_datadir,
            interp_snap['next_snapnum'], ## attribute this data to the next_snapnum's projection file
            name,
            gas_snapdict=interp_snap)
        
        ## differentiate this time to Myr precision
        my_gasStudio.this_setup_id += "_time%.3f"%interp_snap['this_time'] 


        ## create a new figure for this guy
        fig = plt.figure()
        my_gasStudio.render(plt.gca(),**render_kwargs)
        return fig


    time_handler = TimeInterpolationHandler(
        prev_galaxy.snapnums,
        dGyr=2,
        snap_times=prev_galaxy.snap_gyrs)

    galaxy_kwargs = {
        'name':savename,
        'snapdir':snapdir,
        'datadir':datadir}

    time_handler.interpolate_on_snap_pairs(
        my_func,
        galaxy_kwargs=galaxy_kwargs,
        multi_threads=1)

if __name__ == '__main__':
    main()

