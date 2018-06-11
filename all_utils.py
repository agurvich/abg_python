import numpy as np 
import copy
import random
import os
from scipy.optimize import leastsq as opt
import h5py
import pandas as pd
try:
    from memory_profiler import profile,memory_usage
except:
    print "Couldn't find memory profiler"

from readsnap import readsnap

from scipy.spatial.distance import cdist as cdist

#GLOBAL VARIABLES   

# Code mass -> g , (code length)^-3 -> cm^-3 , g -> nH
DENSITYFACT=2e43*(3.086e21)**-3/(1.67e-24)


## dictionary helper functions
def filterDictionary(dict0,indices,dict1 = None,key_exceptions=[],free_mem = 0):
    if dict1 is None:
        dict1={}
    for key in dict0:
        if key in key_exceptions:
            continue
        try:
            if np.shape(dict0[key])[0]==indices.shape[0]:
                dict1[key]=dict0[key][indices]
            ## should only be center of mass and center of mass velocity
            else:
                raise Exception("Save this array verbatim")
        except:
            dict1[key]=dict0[key]
    if free_mem:
        del dict0
    return dict1

## physics helper functions
def getTemperature(U_code,y_helium,ElectronAbundance):
    """U_codes = res['u']
        y_heliums = res['z'][:,1]
        ElectronAbundance=res['ne']"""
    U_cgs = U_code*1e10
    gamma=5/3.
    kB=1.38e-16 #erg /K
    m_proton=1.67e-24 # g
    mu = (1.0 + 4*y_helium) / (1+y_helium+ElectronAbundance) 
    mean_molecular_weight=mu*m_proton
    return mean_molecular_weight * (gamma-1) * U_cgs / kB

def get_IMass(age,mass):
    ## based off mass loss in gizmo plot, averaged over 68 stars
    b = 0.62
    a = -0.071
    factors = b*age**a 
    factors[factors > 1]=1
    factors[factors < 0.8]=0.8
    return mass/factors

#fitting functions
def fitAXb(xs,ys,yerrs):
    """Fits a linear trendline to some data"""
    if yerrs==None:
        yerrs=np.array([1]*len(xs))
    weights=yerrs**-2.
    X=np.matrix([[sum(ys*weights)],[sum(xs*ys*weights)]])
    M=np.matrix([
        [sum(weights),sum(xs*weights)],
        [sum(xs*weights),sum(xs**2.*weights)]
        ])
    [y0],[a]=(M.I*X).A
    return a,y0

def fitGauss(xs,ys,yerrs=None):
    ## initial parameter estimate
    p0 = [np.sum(xs*ys)/np.sum(ys),(np.max(xs)-np.min(xs))/4.,np.max(ys)]

    ## define a gaussian with amplitude A, mean mu, and width sigma
    fn = lambda (mu,sigma,A),x: A*np.exp(-(x-mu)**2./(2*sigma**2.))

    pars = fitLeastSq(fn,p0,xs,ys,yerrs)
    return pars,lambda x: fn(pars,x)

def fitSkewGauss(xs,ys,yerrs=None):
    ## initial parameter estimate
    p0 = [np.sum(xs*ys)/np.sum(ys),(np.max(xs)-np.min(xs))/4.,np.max(ys),.5]

    ## define a gaussian with amplitude A, mean mu, and width sigma
    fn = lambda (mu,sigma,A,skew),x: A*np.exp(-(x*skew-mu)**2./(2*sigma**2.))

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
    if yerrs==None:
        yerrs=[1]*len(xs)
    return sum([(fn(x)-ys[i])**2./yerrs[i]**2. for i,x in enumerate(xs)])

def brokenPowerLaw(a1,b1,a2,b2,xoff,x):
    """A helper function to evaluate a broken power law given some
        parameters-- since lambda functions create unwanted aliases"""
    if x < xoff:
        return a1*x+b1
    else:
        return a2*x+b2

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
    if yerrs==None:
        yerrs=np.array([1]*len(xs))
    for i,xoff in enumerate(xs):
        if i==0 or i==1 or i==(len(xs)-2) or i==(len(xs)-1):
            #skip the first  and second guy, lol
            continue
        b1,a1=fitAXb(xs[:i],ys[:i],yerrs[:i])
        b2,a2=fitAXb(xs[i:],ys[i:],yerrs[i:])
        params=(a1,b1,a2,b2,xoff)
        models+=[params]
        model=lambda x: brokenPowerLaw(params[0],params[1],params[2],params[3],
            params[4],x)
        vars+=[modelVariance(model,xs,ys,yerrs)]

    #there is a hellish feature of python that refuses to evaluate lambda functions
    #so i can't save the models in their own list, I have to save their parameters
    #and recreate the best model
    params=models[np.argmin(vars)]
    model=lambda x: brokenPowerLaw(params[0],params[1],params[2],params[3],
        params[4],x)
    return model,params

def fitExponential(xs,ys):
    """Fits an exponential log y = ax +b => y = e^b e^(ax)"""
    b,a = fitAXb(xs[ys>0],np.log(ys[ys>0]),yerrs=None)
    return (b,a)


#math functions
def rotateVectorsZY(thetay,thetaz,vectors):
    rotatedCoords=rotateVectors(rotationMatrixZ(thetaz),vectors)
    rotatedCoords=rotateVectors(rotationMatrixY(thetay),rotatedCoords)
    return rotatedCoords

def rotateVectors(rotationMatrix,vectors):
    return np.dot(rotationMatrix,vectors.T).T

def rotationMatrixY(theta):
    return np.array([
            [np.cos(theta),0,-np.sin(theta)],
            [0,1,0],
            [np.sin(theta),0,np.cos(theta)]
        ])

def rotationMatrixX(theta):
    return np.array([
            [1,0,0],
            [0,np.cos(theta),np.sin(theta)],
            [0,-np.sin(theta),np.cos(theta)]
        ])

def rotationMatrixZ(theta):
    return np.array([
            [np.cos(theta),np.sin(theta),0],
            [-np.sin(theta),np.cos(theta),0],
            [0,0,1]
        ])

def distance(obj1,obj2):
    """Returns the cartesian distance between two objects that have
        x,y,z attributes"""
    return ((obj1.x-obj2.x)**2+(obj1.y-obj2.y)**2+(obj1.z-obj2.z)**2)**0.5

#list operations
def substep(arr,N):
    my_arr = np.array([])
    for lx,rx in zip(arr[:-1],arr[1:]):
        my_arr=np.append(my_arr,np.linspace(lx,rx,N+1)[:-1])
        
    ## excluded the right end, need to include the final right end
    my_arr = np.append(my_arr,rx)
    return my_arr

def pairFilter(xs,ys,bool_fn):
    """filters both x and y corresponding pairs by
        bool_fn"""

    new_xs = xs[bool_fn(ys)]
    new_ys = ys[bool_fn(ys)]

    new_ys = new_ys[bool_fn(new_xs)]
    new_xs = new_xs[bool_fn(new_xs)]
    return new_xs,new_ys

def findArrayClosestIndices(xs,ys):
    try:
        assert len(xs) < len(ys)
    except:
        raise Exception("Ys should be some large sample that Xs is subsampling!")

    dists = cdist(
        xs.reshape(-1,1),
        ys.reshape(-1,1))

    indices = np.argmin(dists,axis=1)
    return indices

def findIntersection(xs,ys,ys1):
    argmin = np.argmin((ys-ys1)**2)
    return xs[argmin],ys[argmin]
    
#quality of life 
def suppressSTDOUTToFile(fn,args,fname,mode='a+',debug=1):
    """Hides the printed output of a python function to remove clutter, but
        still saves it to a file for later inspection. 
        Input: 
            fn - The function you want to call 
            args - A dictionary with keyword arguments for the function
            fname - The path to the output text file you want to pipe to. 
            mode - The file open mode you want to use, defaults to a+ to append
                to the same debug/output file but you might want w+ to replace it
                every time. 
            debug - Prints a warning message that the STDOUT is being suppressed
        Output: 
            ret - The return value of fn(**args)
    """
    
    orgstdout=sys.stdout
    ret=-1
    try:
        handle=StringIO.StringIO()
        if debug:
            print 'Warning! Supressing std.out...'
        sys.stdout=handle

        ret=fn(**args)

        with file(fname,mode) as fhandle:
            fhandle.write(handle.getvalue())
    finally:
        sys.stdout=orgstdout
        if debug:
            print 'Warning! Unsupressing std.out...'

    return ret

def suppressSTDOUT(fn,args,debug=1):
    """Hides the printed output of a python function to remove clutter. 
        Input: 
            fn - The function you want to call 
            args - A dictionary with keyword arguments for the function
            debug - Prints a warning message that the STDOUT is being suppressed
        Output: 
            ret - The return value of fn(**args)
    """
    orgstdout=sys.stdout
    ret=-1
    try:
        handle=StringIO.StringIO()
        if debug:
            print 'Warning! Supressing std.out...'
        sys.stdout=handle

        ret=fn(**args)

    finally:
        sys.stdout=orgstdout
        if debug:
            print 'Warning! Unsupressing std.out...'

    return ret

#histogram operations
def adaptivelyGenerateBins(ys,hist,binedges):
    """Takes a histogram and adds bins to selectively add sampling in order to reduce 
        erroneous peaks. Helpful for smoothing a PDF. 
        Input: 
            ys - The original y values that were used to generate the histogram 
                (the things that were binned) 
            hist - The histogram of y values
            binedges - the edges of the bins
        Output: 
            hist - The new histogram of y values 
            binedges - The correpsonding bin edges 
    """
    avgbinvalue=np.mean(hist)
    #no bin peaks above 4x the mean 
    iters=0
    maxiters=25
    while max(hist) > avgbinvalue*3.0:
        assert len(hist)==(len(binedges)-1)
        badbini=np.argmax(hist)
        print 'Adding a bin between %.2f and %.2f'%(binedges[badbini],
            binedges[badbini+1])
        binedges=(list(binedges[:badbini+1])+
            [(binedges[badbini]+binedges[badbini+1])/2.]+list(binedges[badbini+1:]))
        hist,binedges=np.histogram(ys,binedges)
        avgbinvalue=np.mean(hist)
        if iters > maxiters:
            raise Exception('Too single valued!')
    return hist,binedges

def getBinMidsAndWidths(bin_edges):
    """Returns two inline operations that I use often-- taking the width of each bin and 
        finding their midpoint."""
    return np.array(zip(*[((bin_edges[i+1]+bin_edges[i])/2.0,1.0*bin_edges[i+1]-bin_edges[i]) 
        for i in xrange(len(bin_edges)-1)]))

def sortHistogrammedObjects(llist,edges,key):
    """Takes a list of objects and sorts them into bins defined by some
        comparison function. Compares the output of key(llist[i]) to 
        [,) bin edges
        Input: 
            llist - the list of objects to be sorted
            edges - the bin edges to sort into 
            key - a function that takes an object and returns a value to be compared
                to bin edges 
        Returns: 
            sorted_list - a 2d list of objects organized by bins
    """
    sorted_list=[[] for i in xrange((len(edges)-1))]
    for obj in llist:
        comparison=key(obj)
        #if less than right edge, it should be in that bin
        #e.x. if less than 0th right edge, it should be in 0th bin
        for j,edge in enumerate(edges[1:]): 
            if comparison < edge: 
                break
        #print comparison,edge,j
        sorted_list[j]+=[obj]
    return sorted_list
    

#plotting functions
def plotSideBySide(plt,rs,rcom,indices,weights=None):
    fig,[ax1,ax2]=plt.subplots(1,2)
    xs,ys,zs = (rs[indices]-rcom).T
    twoDHist(plt,ax1,xs,ys,bins=200,weights=weights)
    twoDHist(plt,ax2,xs,zs,bins=200,weights=weights)
    fig.set_size_inches(12,6)
    nameAxes(ax1,None,'x (kpc)','y (kpc)')
    nameAxes(ax2,None,'x (kpc)','z (kpc)')
    return fig,ax1,ax2

def twoDHist(plt,ax,xs,ys,bins,weights=None,norm='',cbar=0):
    if norm=='':
        from matplotlib.colors import LogNorm
        norm=LogNorm()
    cmap=plt.get_cmap('afmhot')
    h,xedges,yedges=np.histogram2d(xs,ys,weights=weights,bins=bins)
    ax.imshow(h.T,cmap=cmap,origin='lower',
    norm=norm,extent=[min(xedges),max(xedges),min(yedges),max(yedges)])
    if cbar:
        plt.colorbar()
    return h,xedges,yedges

def slackifyAxes(ax,width=8,height=6):
    fig = ax.get_figure()
    fig.set_size_inches(width,height)
    fig.set_facecolor('white')
    
def nameAxes(ax,title,xname,yname,logflag=(0,0),make_legend=0,verty=0,
            subtitle=None,supertitle=None,off_legend=0,axesKeys={'scatterpoints':1,
            'numpoints':1},loc=0,fontsize=16,slackify=0,width=8,height=6,
            xlow=None,xhigh=None,ylow=None,yhigh=None,font_color='k',font_weight='regular'):
    """Convenience function for adjusting axes and axis labels
    Input:
        ax - Axis to label, for single plot pass plt.gca(), for subplot pass 
            the subplot's axis.
        title - The title of the plot.
        xname - The xaxis label
        yname - The yaxis label
        logflag - Flags for log scaling the axes, (x,y) uses simple true/false
        make_legend - A flag for making a legend using each line's label passed
            from the plot(xs,ys,label=)
        verty - A flag for changing the orientation of the yaxis label
        subtitle - Puts a subtitle in the bottom left corner of the axis panel
            if not None
        off_legend - Offsets the legend such that it appears outside of the 
            plot. You MUST add the artist to the bbox_extra_artists list in
            savefig otherwise it WILL be cut off. 
        axesKeys - Keys to pass to legend, look at matplotlib documentation for
            options"""

    ## axes limits
    if xlow is not None:
        ax.set_xlim(left=xlow)
    if ylow is not None:
        ax.set_ylim(bottom=ylow)
    if xhigh is not None:
        ax.set_xlim(right=xhigh)
    if yhigh is not None:
        ax.set_ylim(top=yhigh)

    if yname!=None:
        if verty:
            ax.set_ylabel(yname,fontsize=16)
        else:
            ax.set_ylabel(yname,fontsize=fontsize)
    if xname!=None:
        ax.set_xlabel(xname,fontsize=fontsize)
    if logflag[0]:
        ax.set_xscale('log')
    if logflag[1] :
        ax.set_yscale('log',nonposy='clip')
    if title!=None:
        ax.set_title(title)

    if supertitle:
        ax.text(.01,.96,supertitle,transform=ax.transAxes,
            verticalalignment='center',horizontalalignment='left',
            color = font_color,weight=font_weight)

    if subtitle:
        ax.text(.01,.04,subtitle,transform=ax.transAxes,
            verticalalignment='center',horizontalalignment='left',
            color=font_color,weight=font_weight)
    if slackify:
        slackifyAxes(ax,width,height)

    if make_legend:
        if off_legend:
            return ax.legend(bbox_to_anchor=(1.02,1),frameon=0,**axesKeys)
        else:
            ax.legend(loc=loc,fontsize=16,frameon=0,**axesKeys)
            return ax.get_legend_handles_labels()

###### DIRECTORY STUFF ######
def add_directory_tree(datadir):
    """This function probably already exists lmfao..."""
    if not os.path.isdir(datadir):
	directories=datadir.split('/')
	directories_to_make=[]
	for i in xrange(len(directories)):
	    trialdir='/'.join(directories[:-i])
	    if os.path.isdir(trialdir):
		i-=1
		for j in xrange(i):
		    toadd='/'.join(directories[:-j-1])
		    directories_to_make+=[toadd]
		break
	directories_to_make+=[datadir]
	for directory_to_make in directories_to_make:
	    os.mkdir(directory_to_make)

def getfinsnapnum(snapdir,getmin=0):
    if not getmin:
        maxnum = 0
        for snap in os.listdir(snapdir):
            if 'snapshot' in snap and 'hdf5' in snap:
                snapnum = int(snap[len('snapshot_'):-len('.hdf5')])
                if snapnum > maxnum:
                    maxnum=snapnum
            elif 'snapdir' in snap:
                snapnum = int(snap[len('snapdir_'):])
                if snapnum > maxnum:
                    maxnum=snapnum
        return maxnum
    else:
        minnum=1e8
        for snap in os.listdir(snapdir):
            if 'snapshot' in snap:
                snapnum = int(snap[len('snapshot_'):-len('.hdf5')])
                if snapnum < minnum:
                    minnum=snapnum
            elif 'snapdir' in snap:
                snapnum = int(snap[len('snapdir_'):])
                if snapnum < minnum:
                    minnum=snapnum
        return minnum

def extractMaxTime(snapdir):
    """Extracts the time variable from the final snapshot"""
    maxsnapnum = getfinsnapnum(snapdir)
    if 'snapshot_%3d.hdf5'%maxsnapnum in os.listdir(snapdir):
        h5path = 'snapshot_%3d.hdf5'
    elif 'snapdir_%03d'%maxsnapnum in os.listdir(snapdir):
        h5path = "snapdir_%03d/snapshot_%03d.0.hdf5"%(maxsnapnum,maxsnapnum)
    else:
        print "Couldn't find maxsnapnum in"
        print os.listdir(snapdir)
        raise Exception("Couldn't find snapshot")

    with h5py.File(os.path.join(snapdir,h5path)) as handle:
        maxtime = handle['Header'].attrs['Time']
    return maxtime

## INDICES THOUGH

def extractRectangularVolumeIndices(rs,rcom,radius,height):
   x_indices = (rs-rcom)[:,0]**2<radius**2
   y_indices = (rs-rcom)[:,1]**2<radius**2

   height = radius if height==0 else height
   z_indices = (rs-rcom)[:,2]**2<height**2
   return np.logical_and(np.logical_and(x_indices,y_indices),z_indices)

def extractCylindricalVolumeIndices(coords,r,h,rcom=None):
    if rcom==None:
        rcom = np.array([0,0,0])
    gindices = np.sum((coords[:,:2])**2.,axis=1) < r**2.
    gzindices = (coords[:,2])**2. < h**2.
    indices = np.logical_and(gindices,gzindices)
    return indices

def extractSphericalVolumeIndices(rs,rcom,radius2,rotationAngle=None):
    if rotationAngle != None : 
        rs = np.dot(rotationMatrix(rotationAngle),rs.T).T
        rcom = np.dot(rotationMatrix(rotationAngle),rcom)
    
    indices = np.sum((rs - rcom)**2.,axis=1) < radius2
    if rotationAngle!=None:
        return indices,rs,rcom
    return indices


## USEFUL PHYSICS 
def calculateSigma1D(vels,masses):
    vcom = np.sum(vels*masses[:,None],axis=0)/np.sum(masses)
    vels = vels - vcom # if this has already been done, then subtracting out 0 doesn't matter
    v_avg_2 = (np.sum(vels*masses[:,None],axis=0)/np.sum(masses))**2
    v2_avg = (np.sum(vels**2*masses[:,None],axis=0)/np.sum(masses))
    return (np.sum(v2_avg-v_avg_2)/3)**0.5

def denToff_time(den):
    Gcgs = 6.67e-8 # cm^3 /g /s^2
    ff_time = (
        3*np.pi/(32*Gcgs) /
        den  # g/cc
        )**0.5 #s

    ff_time /=3.15e7 # yr
    return ff_time

