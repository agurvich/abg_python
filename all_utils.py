## from builtin
import numpy as np 
import h5py
import os
from scipy.optimize import leastsq as opt
from scipy.spatial.distance import cdist as cdist
from scipy.interpolate import interp1d

from matplotlib.ticker import NullFormatter

#GLOBAL VARIABLES   

# Code mass -> g , (code length)^-3 -> cm^-3 , g -> nH
DENSITYFACT=2e43*(3.086e21)**-3/(1.67e-24)
HYDROGENMASS = 1.67e-24  # g
cm_per_kpc = 3.08e21 # cm/kpc
Gcgs = 6.674e-8 #cm3/(g s^2)


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
def getTemperature(
    U_code,
    helium_mass_fraction=None,
    ElectronAbundance=None,
    mu = None):
    """U_code = snapdict['InternalEnergy']
    helium_mass_fraction = snapdict['Metallicity'][:,1]
    ElectronAbundance= snapdict['ElectronAbundance']"""
    U_cgs = U_code*1e10
    gamma=5/3.
    kB=1.38e-16 #erg /K
    m_proton=1.67e-24 # g
    if mu is None:
        ## not provided from chimes, hopefully you sent me helium_mass_fraction and
        ##  electron abundance!
        try: 
            assert helium_mass_fraction is not None
            assert ElectronAbundance is not None
        except AssertionError:
            raise ValueError(
                "You need to either provide mu or send helium mass fractions and electron abundances to calculate it!")
        y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))
        mu = (1.0 + 4*y_helium) / (1+y_helium+ElectronAbundance) 
    mean_molecular_weight=mu*m_proton
    return mean_molecular_weight * (gamma-1) * U_cgs / kB

def get_IMass(age,mass):
    ## age must be in Gyr
    ## based off mass loss in gizmo plot, averaged over 68 stars
    b = 0.587
    a = -8.26e-2
    factors = b*age**a 
    factors[factors > 1]=1
    factors[factors < 0.76]=0.76
    return mass/factors

def calculateKappa(vcs,rs):
    """calculate the epicyclic frequency"""

    dvcdr = (vcs[1:] - vcs[:-1])/(rs[1:]-rs[:-1])
    mid_rs = (rs[1:]+rs[:-1])/2.
    mid_vcs = (vcs[1:]+vcs[:-1])/2.
    kappas = np.sqrt(4*mid_vcs**2/mid_rs**2+mid_rs**2*dvcdr)
    kappa_fn = interp1d(mid_rs,kappas,fill_value="extrapolate",kind='linear')
    return kappa_fn(rs)

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
    
def fitGauss(xs,ys,yerrs=None):
    ## initial parameter estimate
    p0 = [np.sum(xs*ys)/np.sum(ys),(np.max(xs)-np.min(xs))/4.,np.max(ys)]

    ## define a gaussian with amplitude A, mean mu, and width sigma
    fn = lambda pars,x: pars[2]*np.exp(-(x-pars[0])**2./(2*pars[1]**2.))

    pars = fitLeastSq(fn,p0,xs,ys,yerrs)
    return pars,lambda x: fn(pars,x)

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
def vectorsToRAAndDec(vectors):
    xs,ys,zs = vectors.T
    ## puts the meridian at x = 0
    ra = np.arctan2(ys,xs)

    ## puts the equator at z = 0
    dec = np.arctan2(zs,(xs**2+ys**2))

    return ra,dec

def rotateVectorsZY(thetay,thetaz,vectors):
    rotatedCoords=rotateVectors(rotationMatrixZ(thetaz),vectors)
    rotatedCoords=rotateVectors(rotationMatrixY(thetay),rotatedCoords)
    return rotatedCoords

def unrotateVectorsZY(thetay,thetaz,vectors):
    rotatedCoords=rotateVectors(rotationMatrixY(-thetay),vectors)
    rotatedCoords=rotateVectors(rotationMatrixZ(-thetaz),rotatedCoords)
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
            print('Warning! Supressing std.out...')
        sys.stdout=handle

        ret=fn(**args)

        with file(fname,mode) as fhandle:
            fhandle.write(handle.getvalue())
    finally:
        sys.stdout=orgstdout
        if debug:
            print('Warning! Unsupressing std.out...')

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
            print('Warning! Supressing std.out...')
        sys.stdout=handle

        ret=fn(**args)

    finally:
        sys.stdout=orgstdout
        if debug:
            print('Warning! Unsupressing std.out...')

    return ret

#plotting functions
def plotSideBySide(plt,rs,rcom,indices,weights=None,axs=None):
    if axs is None:
        fig,[ax1,ax2]=plt.subplots(1,2)
    else:
        fig = axs[0].get_figure()
        ax1,ax2=axs
    print(axs,ax1,ax2)
    xs,ys,zs = (rs[indices]-rcom).T
    twoDHist(plt,ax1,xs,ys,bins=200,weights=weights)
    twoDHist(plt,ax2,xs,zs,bins=200,weights=weights)
    fig.set_size_inches(12,6)
    fig.set_facecolor('white')
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
    

import matplotlib.ticker
def my_log_formatter(x,y):
    """inspired by the nightmare mess that Jonathan Stern
        sent me after being offended by my ugly log axes"""
    if x in [1e-2,1e-1,1,10,100]:
        return r"$%g$"%x
    elif 1e-2 < x < 100 and np.isclose(0,(x*100)%1):
        return r"$%g$"%x
    else:
        return matplotlib.ticker.LogFormatterMathtext()(x)

my_log_ticker = matplotlib.ticker.FuncFormatter(my_log_formatter)

def addSecondAxis(ax,new_tick_labels,new_tick_locations=None,mirror='y'):
    if mirror == 'y':
        ax1 = ax.twiny()
    elif mirror == 'x':
        ax1 = ax.twinx()
    
    ax1.set_xticks(ax.get_xticks() if new_tick_locations is None else new_tick_locations)
    ax1.set_xticklabels(new_tick_labels)
    return ax1
    

def bufferAxesLabels(
    axs,
    nrows,ncols,
    ylabels = False,
    xlabels = False,
    share_ylabel = None,
    share_xlabel = None,
    label_offset = 0.075):
    """Changes the vertical/horizontal alignment of the first & last ytick/xtick 
    such that adjacent panels don't have overlapping labels. For some ridiculous
    reason if you are using a log scale the first and last ticks are denoted by -2 and 1 
    instead of -1 and 0 (and really why are they reversed in the first place??)
    Input:
        axs - flattened axis array
        nrows - number of rows
        ncols - number of columns
        ylabels - flag to turn off ylabels
        xlabels - flag to turn off xlabels """
    axss = axs.reshape(nrows,ncols)

    ## for each column that isn't the first
    for col_i in range(ncols):
        this_col = axss[:,col_i]
        for ax in this_col:
            if ylabels and (col_i+1) >= ylabels:
                ax.set_ylabel('')
            try:
                xticks = ax.get_xticklabels()
                if len(xticks) == 0:
                    continue
                xscale = ax.get_xscale()=='log'
                ##  change the first tick
                if col_i != 0:
                    xticks[xscale].set_horizontalalignment('left')
                ## if we're in the right most 
                ##  column we don't need to change the last tick
                #if col_i != (ncols-1):
                xticks[-1-xscale].set_horizontalalignment('right')
            except IndexError:
                pass ## this can fail if share_x = True

    for row_i in range(nrows):
        axs = axss[row_i,:]
        for col_i,ax in enumerate(axs):
            if xlabels:
                ax.set_xlabel('')
            try:
                yticks = ax.get_yticklabels()
                yscale = ax.get_yscale()=='log'
                ## if we're in the first row don't 
                if len(yticks) == 0:
                    continue
                ##  need to mess with the top tick
                if row_i != 0:
                    ## NOTE wtf?? why is the top tick at 0???
                    ##  if th ebottom tick is at 1??????
                    if yscale:
                        yticks[-2].set_verticalalignment('top')
                    else:
                        yticks[-2].set_verticalalignment('top')
                ## if we're in the last row we 
                ##  don't need to mess with the bottom tick
                if row_i != (nrows-1):
                    yticks[yscale].set_verticalalignment('bottom')
            except IndexError as e:
                pass ## this can fail if share_y = True
    
    fig = axs[0].get_figure()
    if share_ylabel is not None:
        fig.text(
            label_offset,0.5,
            share_ylabel,
            rotation=90,va='center',ha='center',fontsize=16)

    if share_xlabel is not None:
        fig.text(
            0.5,label_offset,
            share_xlabel,
            va='center',ha='center',fontsize=16)


def nameAxes(
    ax,title,xname,yname,logflag=(0,0),
    subtitle=None,supertitle=None,
    make_legend=0,off_legend=0,
    loc=0,
    slackify=0,width=8,height=6,
    xlow=None,xhigh=None,
    ylow=None,yhigh=None,
    subfontsize=None,fontsize=None,
    xfontsize=None,yfontsize=None,
    font_color=None,font_weight='regular',
    legendkwargs=None,
    swap_annotate_side=False):
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
            """

    legendkwargs = {} if legendkwargs is None else legendkwargs

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
        if yfontsize is None:
            ax.set_ylabel(yname)
        else:
            ax.set_ylabel(yname,fontsize=yfontsize)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(yfontsize)
    if xname!=None:
        if xfontsize is None:
            ax.set_xlabel(xname)
        else:
            ax.set_xlabel(xname,fontsize=xfontsize)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(xfontsize)
    if logflag[0]:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(my_log_ticker)
        #ax.xaxis.set_minor_formatter(my_log_ticker))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if logflag[1] :
        ax.set_yscale('log',nonposy='clip')
        ax.yaxis.set_major_formatter(my_log_ticker)
        #ax.yaxis.set_minor_formatter(my_log_ticker))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if title!=None:
        ax.set_title(title)

    subtextkwargs={}
    if font_color is not None:
        subtextkwargs['color']=font_color
    if subfontsize is not None:
        subtextkwargs['fontsize']=subfontsize

    if swap_annotate_side:
        x_pos = 1-0.01
        halign = 'right'
    else:
        x_pos = 0.01
        halign = 'left'
    if supertitle:
        ax.text(x_pos,.96,supertitle,transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment=halign,
            weight=font_weight,**subtextkwargs)

    if subtitle:
        ax.text(x_pos,.04,subtitle,transform=ax.transAxes,
            verticalalignment='center',
            horizontalalignment=halign,
            weight=font_weight,**subtextkwargs)

    if slackify:
        slackifyAxes(ax,width,height)

    ## add the subtext kwargs to legendkwargs
    legendkwargs.update(subtextkwargs)

    if make_legend:
        if off_legend:
            return ax.legend(bbox_to_anchor=(1.02,1),frameon=0,**legendkwargs)
        else:
            ax.legend(
                loc=loc+(supertitle is not None),
                frameon=0,**legendkwargs)
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
                break
        for j in xrange(i):
            toadd='/'.join(directories[:-j-1])
            directories_to_make+=[toadd]
        directories_to_make+=[datadir]
        for directory_to_make in directories_to_make:
            os.mkdir(directory_to_make)

def getfinsnapnum(snapdir,getmin=0):
    if not getmin:
        maxnum = 0
        for snap in os.listdir(snapdir):
            if 'snapshot' in snap and 'hdf5' in snap and snap.index('snapshot')==0:
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
            if 'snapshot' in snap and 'hdf5' in snap:
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
        h5path = 'snapshot_%3d.hdf5'%maxsnapnum
    elif 'snapdir_%03d'%maxsnapnum in os.listdir(snapdir):
        h5path = "snapdir_%03d/snapshot_%03d.0.hdf5"%(maxsnapnum,maxsnapnum)
    else:
        print("Couldn't find maxsnapnum in")
        print(os.listdir(snapdir))
        raise Exception("Couldn't find snapshot")

    with h5py.File(os.path.join(snapdir,h5path),'r') as handle:
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

def ff_timeToDen(ff_time):
    """ff_time must be in yr"""
    Gcgs = 6.67e-8 # cm^3 /g /s^2
    den = 3*np.pi/(32*Gcgs)/(ff_time * 3.15e7)**2 # g/cc
    return den 

def denToff_time(den):
    """den must be in g/cc"""
    Gcgs = 6.67e-8 # cm^3 /g /s^2
    ff_time = (
        3*np.pi/(32*Gcgs) /
        den  # g/cc
        )**0.5 # s

    ff_time /=3.15e7 # yr
    return ff_time


## from https://gist.github.com/benmaier/31f5fa109cf8fae077bde3d2d68a3883
def add_curve_label(
    ax,
    curve_x,
    curve_y,
    label,
    label_pos_abs=None,
    label_pos_rel=None,
    bbox_pad=1.0,
    **kwargs):
    """
    Add a label to a curve according to the curve's slope
    on the displayed figure.
    Parameters
    ----------
    ax : matplotlib.Axes
        The ax object where to put the label on. Use
        `pyplot.gca()` to get the current focal axes.
    curve_x : numpy.ndarray
        The curve's x-data.
    curve_y : numpy.ndarray
        The curve's y-data.
    label : str
        The label.
    label_pos_abs : float, default : None
        The absolute x-position at which to pose the label.
        Must be smaller than `curve_x`'s last element.
        If None, `label_pos_rel` must be given.
    label_pos_rel : float, default : None
        The relative x-position at which to pose the label.
        Must be 0 <= label_pos_rel < 1.
        If None, `label_pos_abs` must be given.
    bbox_pad : float, default : 1.0
        Padding of the bounding box around the label.
    **kwargs
        Will be passed to pyplot.text.
    """
    if label_pos_abs is None and label_pos_rel is not None:

        # get xmin and xmax in display coordinates
        xmin = ax.transData.transform(
            np.array( [ curve_x[1],  curve_y[1]  ] ))[0]
        xmax = ax.transData.transform(
            np.array( [ curve_x[-1], curve_y[-1] ] ))[0]

        # compute label x-position in display coordinates according to
        # demanded relative label position
        new_display_x = xmin + label_pos_rel * (xmax - xmin)

        # convert back to data coordinates and save absolute position
        new_data_x = ax.transData.inverted().transform(np.array([new_display_x,1.0]))
        label_pos_abs = new_data_x[0]

    elif label_pos_abs is None and label_pos_rel is None:
        raise ValueError('Please provide either `label_pos_abs` or `label_pos_rel`.')
    elif label_pos_abs is not None and label_pos_rel is not None:
        raise ValueError('Please provide either `label_pos_abs` or `label_pos_rel`, not both.')

    # find ndx in data for demanded label position
    ndx = np.where(curve_x < label_pos_abs)[0][-1]


    # convert data at this point to display coordinates
    x0, y0 = ax.transData.transform( np.array( [ curve_x[ndx], curve_y[ndx] ] ))
    x1, y1 = ax.transData.transform( np.array( [ curve_x[ndx+1], curve_y[ndx+1] ] ))

    # compute slope and angle at this point in display coordinates
    dx = x1 - x0
    dy = y1 - y0
    angle = np.arctan2(dy,dx) / np.pi * 180

    # convert back to data coordinates
    x0 = label_pos_abs
    y0 = np.interp(x0, curve_x, curve_y)
    # define bounding box for label
    bbox = dict(facecolor='w', alpha=1, edgecolor='none', pad=bbox_pad)

    if not ('ha' in kwargs or 'horizontalalignment' in kwargs):
        kwargs['ha'] = 'center'

    if not ('va' in kwargs or 'verticalalignment' in kwargs):
        kwargs['va'] = 'center'

    # add label
    ax.text(
        x0,y0,
        label,
        rotation=angle,
        rotation_mode='anchor',
        #bbox=bbox,
        transform=ax.transData,
        **kwargs)
