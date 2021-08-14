import numpy as np

from scipy.interpolate import interp1d

def boxcar_average(
    time_edges,
    ys,
    boxcar_width,
    loud=False,
    average=True, ## vs. just counting non-nan entries in a window
    assign='right'):

    """
    for lists with many nans need to first run w/ average=False, then 
    run with ys=np.ones and average=False
    to get divisor for each window.

    idea is that you subtract off previous window from each window
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    def running_mean(x, N):
        cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    potentially accrues floating point error for many points... (>1e5?)
    there is another solution on that page that uses scipy instead 

    from scipy.ndimage.filters import uniform_filter1d
    uniform_filter1d(x, size=N) <--- requires one to explicitly deal 
        with edges of window
    """

    ## apply a finite filter... no infinities allowed!
    ys[np.logical_not(np.isfinite(ys))] = np.nan

    dts = np.unique(time_edges[1:]-time_edges[:-1])
    if not np.allclose(dts,dts[0]): raise ValueError("ys must be uniformly spaced for this to work...")

    ## prepend a dt to the beginning if we were passed something missing an initial bin edge
    if time_edges.shape[0] != (ys.shape[0]+1): time_edges = np.append([time_edges[0]-dts[0]],time_edges)

    ## number of points per boxcar is:
    N = int(boxcar_width//dts[0] + ((boxcar_width%dts[0])/dts[0]>=0.5))
    if loud: print("boxcar'ing %d points/car, dt: %.2e window: %.2e"%(N,dts[0],boxcar_width))

    cumsum = np.nancumsum(np.insert(ys, 0, 0).astype(np.float64)) 
    ## cumsum[N:] is the first window, then second window + extra first point,
    ##  then third window + extra 2 first points, etc... 
    ys = (cumsum[N:]-cumsum[:-N])

    ## allow a user to specify if they just want to take the sum
    if average: ys = ys/N

    ## tack on a window's worth of nans to the end since we couldn't
    ##  evaluate it
    ys = np.append([np.nan]*(N-1),ys)

    ## shift the window based on centering
    ##  because conditionals should default true
    if assign == 'right':
        pass
    ##  shift left by a whole window width
    elif assign == 'left':
        ys[:-N] = ys[N:]
        ys[-N:] = np.nan
    ##  shift left by a half window width
    elif assign == 'center':
        n = max(int(N/2),1)
        ys[:-n] = ys[n:]
        ys[-n:] = np.nan
    else: raise ValueError("unknown bin assignment %s, should be left, right, or center"%assign) 

    return time_edges,ys
    
def uniform_resampler(xs,arrs,DT=None):
    """resamples xs,[ys,...] arrays evenly by interpolation. """
    new_xs = xs
    return_value = arrs
    if len(np.unique(np.diff(xs))) != 1: ## not uniform 
        ## not told what what spacing to use, so we'll conserve 
        ##  number of points.
        if DT is None: DT = (np.nanmax(xs) - np.nanmin(xs))/(len(xs)-1)

        ## create new xs to sample at
        new_xs = np.arange(np.nanmax(xs),np.nanmin(xs)-DT,-DT)[::-1]

        ## sample each of the arrays we're passed
        return_value = [
            interp1d(
                xs,
                arr,
                fill_value=np.nan,
                bounds_error=False)(new_xs)
            for arr in arrs]

    return new_xs,return_value
    
def smooth_x_varying_curve(
    xs,
    ys,
    boxcar_width,
    log=False,
    assign='center',
    DT=0.01):

    assert len(xs) == len(ys)

    if log: ys = np.log10(ys)
 
    new_xs,[new_ys] = uniform_resampler(xs,[ys],DT)

    smooth_xs,smooth_ys = boxcar_average(new_xs,new_ys,boxcar_width,assign=assign)
    smooth_xs,smooth_ys2 = boxcar_average(new_xs,new_ys**2,boxcar_width,assign=assign)
    smooth_xs,smooth_nan_count = boxcar_average(new_xs,np.isnan(new_ys),boxcar_width,assign=assign)

    ## boxcar average would've added an extra point because it expects edges
    if len(smooth_xs) != len(smooth_ys) and len(new_xs) == len(new_ys): smooth_xs = smooth_xs[1:]

    ## exclude region that we *just* filled with nans, 
    ##  because those points had their average diluted
    dx = smooth_xs[1]-smooth_xs[0]
    if assign == 'center':
        dtop_should_be = int(boxcar_width/2/dx)
        dbottom_should_be = int(boxcar_width/2/dx)
    elif assign == 'left':
        dtop_should_be = int(boxcar_width/dx)
        dbottom_should_be = 0
    elif assign == 'right':
        dtop_should_be = 0
        dbottom_should_be = int(boxcar_width/dx)
    else: raise ValueError("invalid assign")

    if dtop_should_be !=0:
        smooth_xs = smooth_xs[dbottom_should_be:-dtop_should_be]
        smooth_ys = smooth_ys[dbottom_should_be:-dtop_should_be]
        smooth_ys2 = smooth_ys2[dbottom_should_be:-dtop_should_be]
        smooth_nan_count = smooth_nan_count[dbottom_should_be:-dtop_should_be]
    else:
        smooth_xs = smooth_xs[dbottom_should_be:]
        smooth_ys = smooth_ys[dbottom_should_be:]
        smooth_ys2 = smooth_ys2[dbottom_should_be:]
        smooth_nan_count = smooth_nan_count[dbottom_should_be:]

    ## TODO
    ## this is hard-coded for double-smoothing
    ##  with the same window. one day i may regret this
    nan_mask = smooth_nan_count  > 0
    ## if there is an extra window's worth of nans
    ##  we must have smoothing this before (or maybe
    ##  we're smoothing a running scatter)
    ##  let's get rid of all the nan's and hope that there
    ##  aren't any that were in the middle, just at the edges
    ##  from not having enough points in the window :\
    if np.sum(nan_mask!=0) == (int(boxcar_width/dx)):
        nan_mask = smooth_nan_count == 0
        smooth_xs = smooth_xs[nan_mask]
        smooth_ys = smooth_ys[nan_mask]
        smooth_ys2 = smooth_ys2[nan_mask]

    ## it's possible that we removed nans from in the middle, so we have to do this for safety
    smooth_xs,[smooth_ys,smooth_ys2] = uniform_resampler(smooth_xs,[smooth_ys,smooth_ys2],DT)

    ## have to skip first window's width of points
    sigmas = (smooth_ys2-smooth_ys**2)**0.5

    ## exponentiate to put everything back
    if log:
        lowers = 10**(smooth_ys-sigmas)
        uppers = 10**(smooth_ys+sigmas)
        smooth_ys = 10**smooth_ys
        sigmas = sigmas ## dex
        ys = 10**ys
    else:
        lowers = smooth_ys-sigmas
        uppers = smooth_ys+sigmas
    
    assert len(smooth_xs) == len(smooth_ys)
    return smooth_xs,smooth_ys,sigmas,lowers,uppers

def find_first_window(xs,ys,bool_fn,window,last=False):
    ## averages boolean over window at time xs
    bool_xs,bool_ys = boxcar_average(
        xs,
        bool_fn(xs,ys),
        window) 

    ##  by taking the floor, it requires that
    ##   all times in the window fulfill the boolean
    bool_ys[np.isfinite(ys)] = np.floor(bool_ys[np.isfinite(ys)]).astype(float)
    bool_ys[np.isnan(ys)] = np.nan

    ## no window matches
    if np.nansum(bool_ys) == 0:
        return np.nan,np.nan

    ## find the first instance of true
    rindex = np.nanargmax(bool_ys==1)

    if last:
        ## nightmare code that finds the last instance
        ##  of a true in a time series
        offset = None
        while offset != 0:
            ## find the next time the time series has a false
            next_false = np.nanargmax(bool_ys[rindex:]==0)

            ## find the next time there's a true after that false
            ##  3 cases since we're starting on a false:
            ##  1) there are no more trues -> offset=0
            ##  2) there are only trues following this false -> here offset=1, but will next be case 1)
            ##  3) there are trues and falses -> offset = distance to next True
            offset = np.nanargmax(bool_ys[rindex+next_false:])

            ## if that false eventually has a true after it, let's
            ##  move there and start the process over again.
            if offset > 0:
                rindex += offset + next_false 
        
    ## finds the point a window's width before from the right edge
    lindex = np.argmin((bool_xs - (bool_xs[rindex] - window))**2)

    return bool_xs[lindex],bool_xs[rindex]

def find_last_instance(xs,ys,bool_fn):
    """ search the reversed array for the first time bool_fn returns True,
        aka the last time it returns True!"""
    bool_ys = bool_fn(xs,ys)
    if np.nansum(bool_ys) == 0:
        return np.nan,np.nan
    rev_index = np.argmin(np.logical_not(bool_ys[::-1]))
    return xs[xs.size-rev_index-(rev_index==0)],ys[xs.size-rev_index-(rev_index==0)]
    
def find_local_minima_maxima(xs,ys,smooth=None,ax=None):

    ## approximate the slope of the curve
    slopes = np.diff(ys)/np.diff(xs)
    xs = (xs[1:]+xs[:-1])/2

    ## smooth the slope if requested
    if smooth is not None:
        ## x could also be uniform and this will work
        xs,slopes,_,_,_ = smooth_x_varying_curve(xs,slopes,smooth,assign='center')
    
    xs = (xs[1:]+xs[:-1])/2
    ## find locations where slope changes sign
    zeros = xs[np.diff(slopes>=0).astype(bool)]

    ## sure, why not, we'll plot them
    if ax is not None:
        ax.plot(xs,slopes[1:])
        ax.axhline(0,ls='--',c='k')
        for zero in zeros:
            ax.axvline(zero)

    return zeros