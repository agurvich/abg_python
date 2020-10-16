import os
import time
import copy

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


        if snap_times is None:
            raise NotImplementedError("Need to open each snapshot and extract the time in Gyr")

        dGyr_or_times_gyr = np.array(dGyr_or_times_gyr)

        if len(dGyr_or_times_gyr.shape) == 0:
            ## we were passed a time spacing
            times_gyr,snap_pairs = find_bordering_snapnums(
                    prev_galaxy.snap_gyrs,
                    dGyr=dGyr_or_times_gyr,
                    tmin=dGyr_tmin,
                    tmax=dGyr_tmax)

        elif len(dGyr_or_times_gyr) == 1:
            ## we were passed an array of times
            times_gyr = dGyr_or_times_gyr
            inds_next = np.argmax((times_gyr - snap_times_gyr[:,None]) < 0 ,axis=0)
            inds_prev = inds_next-1
            snap_pairs = np.array(list(zip(inds_prev,inds_next)))

        else:
            raise ValueError("Could not interpret dGyr_or_times_gyr",dGyr_or_times_gyr)

        self.times_gyr = times_gyr
        self.snap_pairs = snap_pairs

    def interpolate_on_snap_pairs(
        self,
        function_to_call_on_interpolated_dataframe,
        galaxy_kwargs=None,
        multi_threads=1):
        """ """

        if galaxy_kwargs is None:
            galaxy_kwargs = {}

        if multi_threads == 1:

            return single_threaded_control_flow(
                function_to_call_on_interpolated_dataframe,
                self.times_gyr,
                self.snap_pairs,
                galaxy_kwargs)
            
        elif multi_threads > 1
            raise NotImplementedError("Need to break pairs into chunks and dispatch them to python processes")
        else:
            raise ValueError("Specify a number of threads >=1, not",multi_threads)

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
    return times_gyr,np.array(list(zip(inds_prev,inds_next)))

def single_threaded_control_flow(
    function_to_call_on_interpolated_dataframe,
    times,
    snap_pairs,
    galaxy_kwargs):
    """ """

    prev_galaxy,next_galaxy = None,None
    prev_snapnum,next_snapnum=None,None
    for i,pair in enumerate(snap_pairs):     
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
            t0,t1 =  prev_galaxy.current_time_Gyr,next_galaxy.current_time_Gyr
            time_merged_df = index_match_snapshots_with_dataframes(
                prev_galaxy.sub_snap,
                next_galaxy.sub_snap,
                extra_keys_to_extract=['Temperature'])

        ## update the interp_snap with new values for the new time
        interp_snap = make_interpolated_snap(this_time,time_merged_df,t0,t1)
        interp_snap['prev_snapnum'] = prev_snapnum
        interp_snap['next_snapnum'] = next_snapnum

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
        #print('loading',prev_snapnum,'from disk')
        prev_galaxy = Galaxy(snapnum=pair[0],**kwargs)
        prev_galaxy.extractMainHalo()
        changed = True
    if next_galaxy is None:
        #print('loading',next_snapnum,'from disk')
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
    prev_next_merged_df_snap = pd.merge(
        prev_df_snap,
        next_df_snap_reduced,
        how='inner',
        on=prev_df_snap.index,
        suffixes=('','_next'),
        copy=False).set_index('key_0')
    
    ## remove particles that do not exist in the next snapshot, 
    ##  difficult to tell if they turned into stars or were merged. 
    prev_next_merged_df_snap = prev_next_merged_df_snap.dropna()
    
    return prev_next_merged_df_snap

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
        
        my_gasStudio.this_setup_id += "_time%.3f"%this_time ## differentiate this time to Myr precision

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

