import numpy as np
from scipy.optimize import leastsq as opt

def fitAXb(xs,ys,yerrs,fixed_a=None,fixed_b=None):
    """Fits a linear trendline to some data"""
    if yerrs is None:
        yerrs=np.array([1]*len(xs))
    weights=yerrs**-2.
    X=np.matrix([[sum(ys*weights)],[sum(xs*ys*weights)]])
    M=np.matrix([
        [sum(weights),sum(xs*weights)],
        [sum(xs*weights),sum(xs**2.*weights)]
        ])
    if fixed_a is not None:
        b = ((X[0] - fixed_a*M[0,1])/M[0,0])[0,0]
        a = fixed_a
    elif fixed_b is not None:
        a = ((X[1] - fixed_b*M[1,0])/M[1,1])[0,0]
        b = fixed_b
    else:
        [b],[a]=(M.I*X).A
    return a,b

def fit_running_AXb(time_edges,boxcar_width,xs,ys,yerrs):
    """fits a trendline using f(x) ~ y in bins 
    of t that are boxcar_width wide"""
    
    xs,ys = pairReplace(xs,ys,np.nan,np.isinf)
    if yerrs is None:
        yerrs=np.ones(len(xs))
    weights=yerrs**-2.
    
    boxcar_xs,sum_weights = boxcar_average(
        time_edges,
        weights,
        boxcar_width,
        average=False)
    
    boxcar_xs,sum_xs_weights = boxcar_average(
        time_edges,
        xs*weights,
        boxcar_width,
        average=False)
    
    boxcar_xs,sum_xs2_weights = boxcar_average(
        time_edges,
        xs*xs*weights,
        boxcar_width,
        average=False)
    
    boxcar_xs,sum_ys_weights = boxcar_average(
        time_edges,
        ys*weights,
        boxcar_width,
        average=False)
    
    boxcar_xs,sum_ys_xs_weights = boxcar_average(
        time_edges,
        ys*xs*weights,
        boxcar_width,
        average=False)


    X = np.zeros((sum_ys_weights.size,2))
    X[:,0] = sum_ys_weights
    X[:,1] = sum_ys_xs_weights
    
    M = np.zeros((sum_weights.size,2,2))
    M[:,0,0] = sum_weights
    M[:,0,1] = sum_xs_weights
    M[:,1,0] = sum_xs_weights
    M[:,1,1] = sum_xs2_weights

    test_arr = sum_xs_weights==0
    if np.any(test_arr):
        inds = np.argwhere(test_arr)[:,0]
        M[test_arr] = [[np.nan,np.nan],[np.nan,np.nan]]
    
    #print("X=",X[4399])
    #print("M=",M[4399])
    try:
        invs = np.linalg.inv(M)
    except:
        ## one of the matrices was singular... 
        ##  none should have 0 pivots so is 
        ##  that just bad luck...? overflow error?
        ##  who could say. regardless, this will do
        ##  **something**
        invs = np.linalg.inv(M)
    
    ### https://stackoverflow.com/questions/46213851/python-multiplying-a-list-of-vectors-by-a-list-of-matrices-as-a-single-matrix-o
    ##  only god knows why this works
    pars = np.einsum('ij,ikj->ik',X,invs)
    fit_bs, fit_as = pars.T
    return fit_as,fit_bs

def fitVoigt(xs,ys,yerrs=None):
    p0 = [np.sum(xs*ys)/np.sum(ys),
        (np.max(xs)-np.min(xs))/4.,
        (np.max(xs)-np.min(xs))/4.,
        np.max(ys)]

    ## define a gaussian with amplitude A, mean mu, and width sigma
    lorentz_fn = lambda pars,x: (pars[3]*
        pars[2]/(
        (x-pars[0])**2 + pars[2]**2) )

    gauss_fn = lambda pars,x: pars[3]/np.sqrt(
        2*np.pi*pars[1]**2
        )*np.exp(-(x-pars[0])**2./(2*pars[1]**2.))

    fn = lambda pars,x: lorentz_fn(pars,x)*gauss_fn(pars,x)


    pars = fitLeastSq(fn,p0,xs,ys,yerrs)
    return pars,lambda x: fn(pars,x)


def fitLorentzian(xs,ys,yerrs=None):
    p0 = [np.sum(xs*ys)/np.sum(ys),(np.max(xs)-np.min(xs))/4.,np.max(ys)]

    ## define a gaussian with amplitude A, mean mu, and width sigma
    fn = lambda pars,x: pars[2]*(np.pi*pars[1])**-1*(pars[1]**2/(
        (x-pars[0])**2 + pars[1]**2) )

    pars = fitLeastSq(fn,p0,xs,ys,yerrs)
    return pars,lambda x: fn(pars,x)
    
def fitGauss(xs,ys,yerrs=None,format_str=None):
    ## initial parameter estimate
    p0 = [np.sum(xs*ys)/np.sum(ys),(np.max(xs)-np.min(xs))/4.,np.max(ys)]

    ## define a gaussian with amplitude A, mean mu, and width sigma
    fn = lambda pars,x: pars[2]*np.exp(-(x-pars[0])**2./(2*pars[1]**2.))

    pars = fitLeastSq(fn,p0,xs,ys,yerrs)
    if format_str is None:
        pass
    return pars,lambda x: fn(pars,x)

def covarianceEigensystem(xs,ys):
    """ calculates the covariance matrix and the principle axes
        (eigenvectors) of the data. 
        Eigenvalues represent variance along those principle axes.

        ## choose new x-axis to be evecs[0], rotation angle is
        ##  angle between it and old x-axis, i.e.
        ##  ehat . xhat = cos(angle)
        angle = np.arccos(evecs[0][0])

        ## evals are variance along principle axes
        rx,ry = evals**0.5 ## in dex
        cx,cy = 10**np.mean(xs),10**np.mean(ys) ## in linear space

        for evec,this_eval in zip(evecs,evals):
            dx,dy = evec*this_eval**0.5
            ax.plot(
                [cx,10**(np.log10(cx)+dx)],
                [cy,10**(np.log10(cy)+dy)],
                lw=3,ls=':',c='limegreen')

        plotEllipse(
            ax,
            cx,cy,
            rx,ry,
            angle=angle*180/np.pi,
            log=True,
            color='limegreen')"""

    if len(xs) == len(ys) == 0:
        return np.array([[np.nan,np.nan],[np.nan,np.nan]]),np.array([np.nan,np.nan])
    cov = np.cov([xs,ys])
    evals,evecs = np.linalg.eig(cov)

    evecs = evecs.T ## after transpose becomes [e1,e2], which makes sense...? lol

    ## re-arrange so semi-major axis is always 1st
    sort_mask = np.argsort(evals)[::-1]
    evals,evecs = evals[sort_mask],evecs[sort_mask]
    evecs[np.all(evecs<0,axis=1)]*=-1 ## take the positive version

    return evecs,evals

def getCovarianceEllipse(xs,ys):
    evecs,evals = covarianceEigensystem(xs,ys)

    ## choose new x-axis to be evecs[0], rotation angle is
    ##  angle between it and old x-axis, i.e.
    ##  ehat . xhat = cos(angle)
    angle = np.arccos(evecs[0][0])

    ## evals are variance along principle axes
    rx,ry = evals**0.5 ## in dex
    cx,cy = 10**np.mean(xs),10**np.mean(ys) ## in linear space

    return cx,cy,rx,ry,angle,evecs


def fitSkewGauss(xs,ys,yerrs=None):
    ## initial parameter estimate
    p0 = [np.sum(xs*ys)/np.sum(ys),(np.max(xs)-np.min(xs))/4.,np.max(ys),.5]

    ## define a gaussian with amplitude A, mean mu, and width sigma
    fn = lambda pars,x: pars[2]*np.exp(-(x*pars[3]-pars[0])**2./(2*pars[1]**2.))

    pars = fitLeastSq(fn,p0,xs,ys,yerrs)
    return pars,lambda x: fn(pars,x)

def fitLeastSq(fn,p0,xs,ys,yerrs=None,log_fit=0):
    """ Example fitting a parabola:
        fn = lambda p,xs: p[0]+p[1]*xs**2
        xs,ys=np.arange(-10,10),fn((1,2),xs)
        plt.plot(xs,ys,lw=3)
        pars = fitLeastSq(fn,[15,2],xs,ys)
        plt.plot(xs,fn(pars,xs),'r--',lw=3)"""
    if yerrs is not None:
        if log_fit:
            fit_func= lambda p: np.log10(ys) - np.log10(fn(p,xs))
        else:
            fit_func= lambda p: (ys - fn(p,xs))/yerrs
    else:
        if log_fit:
            fit_func= lambda p: np.log10(ys) - np.log10(fn(p,xs))
        else:
            fit_func= lambda p: (ys - fn(p,xs))
    pars,res = opt(fit_func,p0)

    return pars
    
def modelVariance(fn,xs,ys,yerrs=None):
    """takes a function and returns the variance compared to some data"""
    if yerrs is None:
        yerrs=[1]*len(xs)
    return sum([(fn(x)-ys[i])**2./yerrs[i]**2. for i,x in enumerate(xs)])

def brokenPowerLaw(x,a1,b1,a2,xoff):
    """A helper function to evaluate a broken power law given some
        parameters-- since lambda functions create unwanted aliases"""

    ## handle when we're passed an array, the lazy way
    try:
        iter(x)
        return np.array([brokenPowerLaw(this_x,a1,b1,a2,xoff) for this_x in x])
    except:
        if x < xoff:
            return a1*x+b1
        else:
            return a2*(x-xoff)+(a1*xoff+b1)

def fit_broken_AXb(xs,ys,yerrs=None):
    """Finds the best fit broken linear trendline for a set of x and y 
        data. It does this by finding the chi^2 of placing a joint at each 
        point and finding the best fit linear trendline for the data on either 
        side. The joint that produces the minimum chi^2 is accepted. 
        Input: 
            xs - the x values 
            ys - the y values 
            yerrs - the yerrors, defaults to None -> constant error bars
        Output: 
            what it advertises
    """
    vars=[]
    models=[]

    xs,ys = pairFilter(xs,ys,np.isfinite)

    if yerrs is None:
        yerrs=np.array([1]*len(xs))

    for i,xoff in enumerate(xs):
        if i==0 or i==1 or i==(len(xs)-2) or i==(len(xs)-1):
            #skip the first  and second guy, lol
            continue
        these_xs = xs[:i]
        ## fit a line to the first half
        a1,b1=fitAXb(these_xs,ys[:i],yerrs[:i])

        ## fit a line to the second half, 
        ##   offset the points and pin them 
        ##   to the end of the first line. 
        a2,b2=fitAXb(
            xs[i:]-xoff,
            ys[i:],
            yerrs[i:],
            fixed_b=a1*xoff+b1)
            
        params=(a1,b1,a2,xoff)
        models+=[params]
        model=lambda x: brokenPowerLaw(x,*params)
        vars+=[modelVariance(model,xs,ys,yerrs)]

    #there is a hellish feature of python that refuses to evaluate lambda functions
    #so i can't save the models in their own list, I have to save their parameters
    #and recreate the best model
    params=models[np.argmin(vars)]
    model=lambda x: brokenPowerLaw(x,*params)

    return model,params

def fitExponential(xs,ys):
    """Fits an exponential log y = ax +b => y = e^b e^(ax)"""
    return fitAXb(xs[ys>0],np.log(ys[ys>0]),yerrs=None)


