from scipy.interpolate import interp1d

def boxcar_average(
    time_edges,
    ys,
    boxcar_width,
    loud=False,
    average=True, ## vs. just counting non-nan entries in a window
    edges = True,
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
    if not np.allclose(dts,dts[0]):
        print(dts)
        raise ValueError("ys must be uniformly spaced for this to work...")

    if not edges:
        time_edges = np.append([time_edges[0]-dts[0]],time_edges)

    ## number of points per boxcar is 
    N = int(boxcar_width//dts[0] + ((boxcar_width%dts[0])/dts[0]>=0.5))
    if loud:
        print("boxcar'ing with %d points/car, dt: %.2e window: %.2e"%(N,dts[0],boxcar_width))
    cumsum = np.nancumsum(np.insert(ys, 0, 0).astype(np.float64)) 

    ## cumsum[N:] is the first window, then second window + extra first point,
    ##  then third window + extra 2 first points, etc... 
    ys = (cumsum[N:]-cumsum[:-N])
    if average:
        ys = ys/N
    ys = np.append([np.nan]*(N-1),ys)

    ## because conditionals should default true
    if assign == 'right':
        pass
    ## shift left by a whole window width
    elif assign == 'left':
        ys[:-N] = ys[N:]
        ys[-N:] = np.nan
    ## shift left by a half window width
    elif assign == 'center':
        n = max(int(N/2),1)
        ys[:-n] = ys[n:]
        ys[-n:] = np.nan
    else:
        raise ValueError("unknown bin assignment %s, should be left, right, or center"%assign) 

    return time_edges[:],ys
    
def smooth_x_varying_curve(xs,ys,smooth,log=False,assign='center'):

    if log:
        ys = np.log10(ys)

    times = np.arange(np.nanmax(xs),np.nanmin(xs)-0.01,-0.01)[::-1]
    fn = interp1d(
        xs,
        ys,
        fill_value=np.nan,
        bounds_error=False)
    values = fn(times)

    smooth_xs,smooth_ys = boxcar_average(times,values,smooth,assign=assign)
    smooth_xs,smooth_ys2 = boxcar_average(times,values**2,smooth,assign=assign)
    smooth_xs,smooth_nan_count = boxcar_average(times,np.isnan(values),smooth,assign=assign)

    ## exclude region that we filled with nans, or might have its
    ##  average otherwise diluted

    dx = smooth_xs[1]-smooth_xs[0]
    if assign == 'center':
        dtop_should_be = int(smooth/2/dx)
        dbottom_should_be = int(smooth/2/dx)
    elif assign == 'left':
        dtop_should_be = int(smooth/dx)
        dbottom_should_be = 0
    elif assign == 'right':
        dtop_should_be = 0
        dbottom_should_be = int(smooth/dx)
    else:
        raise NotImplementedError("invalid assign")

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
    if np.sum(nan_mask!=0) == (int(smooth/dx)):
        nan_mask = smooth_nan_count == 0
        smooth_xs = smooth_xs[nan_mask]
        smooth_ys = smooth_ys[nan_mask]
        smooth_ys2 = smooth_ys2[nan_mask]

    ## need to resample what we had to make sure points are evenly spaced
    if len(np.unique(np.diff(smooth_xs))) != 1:
        ## evenly spaced times
        times = np.arange(smooth_xs.max(),smooth_xs.min()-0.01,-0.01)[::-1]

        ## replace ys
        fn = interp1d(
            smooth_xs,
            smooth_ys,
            fill_value=np.nan,
            bounds_error=False)
        smooth_ys = fn(times)

        ## replace ys2
        fn = interp1d(
            smooth_xs,
            smooth_ys2,
            fill_value=np.nan,
            bounds_error=False)
        smooth_ys2 = fn(times)
        
        ## replace smooth_xs
        smooth_xs = times

    ## have to skip first window's width of points
    sigmas = (smooth_ys2-smooth_ys**2)**0.5


    if log:
        lowers = 10**(smooth_ys-sigmas)
        uppers = 10**(smooth_ys+sigmas)
        smooth_ys = 10**smooth_ys
        sigmas = sigmas ## dex
        ys = 10**ys
    else:
        lowers = smooth_ys-sigmas
        uppers = smooth_ys+sigmas
    
    return smooth_xs,smooth_ys,sigmas,lowers,uppers

def find_first_window(xs,ys,bool_fn,window,last=False):
    ## averages boolean over window,
    ##  by taking the floor, it requires that 
    ##  all times in the window fulfill the boolean
    bool_xs,bool_ys = boxcar_average(
        xs,
        bool_fn(xs,ys),
        window) 

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
            ##  2) there are only trues following this false -> offset=1, will next be 0
            ##  3) there are trues and falses -> offset = distance to next True
            offset = np.nanargmax(bool_ys[rindex+next_false:])

            ## if that false eventually has a true after it, let's
            ##  move there and start the process over again.
            if offset > 0:
                rindex += offset + next_false 
        
    ## finds the point a window's width away from the right edge
    lindex = np.argmin((bool_xs - (bool_xs[rindex] - window))**2)

    return bool_xs[lindex],bool_xs[rindex]

def find_last_instance(xs,ys,bool_fn):
    bool_ys = bool_fn(xs,ys)
    if np.nansum(bool_ys) == 0:
        return np.nan,np.nan
    rev_index = np.argmin(np.logical_not(bool_ys[::-1]))
    return xs[xs.size-rev_index-(rev_index==0)],ys[xs.size-rev_index-(rev_index==0)]
    
def find_local_minima_maxima(xs,ys,smooth=None,ax=None):

    ## calculate the slope of the curve
    slopes = np.diff(ys)/np.diff(xs)
    xs = xs[1:]

    ## smooth the slope if requested
    if smooth is not None:
        ## x could also be uniform and this will work
        xs,slopes,foo,bar,foo = smooth_x_varying_curve(xs,slopes,smooth)
    
    xs = xs[1:]
    ## find where slope changes sign
    zeros = xs[np.diff(slopes>0).astype(bool)]

    if ax is not None:
        ax.plot(xs,slopes[1:])
        ax.axhline(0,ls='--',c='k')
        #for zero in zeros:
            #ax.axvline(zero)

    return zeros
