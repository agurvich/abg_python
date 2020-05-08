## from builtin
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from abg_python.all_utils import my_log_formatter

"""
try:
    from distinct_colours import get_distinct
    from cycler import cycler
    colors = get_distinct(5)
    plt.rc('axes', prop_cycle=(cycler('color', colors) ))
except:
    print "Couldn't reset default colors"
"""

## try and use my default matplotlib rcparams style file
#try:
    #plt.style.use('ABG_default')
#except:
    #pass


def add_to_legend(
    ax,
    label='',
    shape='line',
    loc=0,
    legend_kwargs=None,
    make_new_legend=False,
    **kwargs):

    legend = ax.get_legend()
    ## add the current legend to the tracked artists
    ##  and then pretend we weren't passed one
    if make_new_legend and legend is not None:
        ax.add_artist(legend)
        legend=None

    if legend is not None:
        lines = legend.get_lines()
        labels = [text.get_text() for text in legend.get_texts()]
    else:
        lines,labels=[],[]

    if legend_kwargs is None:
        legend_kwargs = {}

    ## make the new line
    if shape == 'line':
        line = Line2D(
        [0],[0],
        **kwargs)
    else:
        raise NotImplementedError

    if label not in labels:
        lines.append(line)
        labels.append(label)

    if loc in legend_kwargs:
        loc = legend_kwargs.pop('loc')
    ax.legend(lines,labels,loc=loc,**legend_kwargs)

    return ax

def plotCircle(
    ax,
    x,y,
    radius,
    fill=False,
    lw=3,
    **kwargs):
    """kwargs you might like are: 
        ls
        color
    """
    return ax.add_artist(
        plt.Circle((x,y),radius,fill=fill,lw=lw,**kwargs))

def addColorbar(
    ax,cmap,
    vmin,vmax,
    label,logflag = 0,
    fontsize=8,cmap_number=0,
    tick_tuple=None,
    horizontal = False):
    if logflag:
        from matplotlib.colors import LogNorm as norm
        ticks = np.linspace(np.log10(vmin),np.log10(vmax),5,endpoint=True)
        ticks = 10**ticks
        tick_labels= [my_log_formatter(tick,None) for tick in ticks]
    else:
        from matplotlib.colors import Normalize as norm
        ticks = np.linspace(vmin,vmax,5,endpoint=True)
        tick_labels= ["%.2f" % tick for tick in ticks]
    
    if tick_tuple is not None:
        ticks,tick_labels = tick_tuple
    
    fig = ax.get_figure()
    ## x,y of bottom left corner, width,height in percentage of figure size
    ## matches the default aspect ratio of matplotlib
    cur_size = fig.get_size_inches()*fig.dpi        

    cur_height = cur_size[1]
    cur_width = cur_size[0]
    offset = 0.05 + cmap_number*(25/cur_width+50/cur_width)

    if not horizontal:
        ax1 = fig.add_axes([0.95 + offset, 0.125, 25./cur_width, 0.75])
    else:
        ax1 = fig.add_axes([0.3, -.05 - offset,0.4, 50./cur_height])

    cb1 = matplotlib.colorbar.ColorbarBase(
        ax1, cmap=cmap,
        extend='both',
        extendfrac=0.05,
        norm=norm(vmin=vmin,vmax=vmax),
        orientation='vertical' if not horizontal else 'horizontal')


    cb1.set_label(label,fontsize=fontsize)

    cb1.set_ticks(ticks)
    cb1.set_ticklabels(tick_labels)
    cb1.ax.tick_params(labelsize=fontsize)
    return cb1,ax1


def addSegmentedColorbar(ax,colors,vmin,vmax,label,logflag=0,fontsize=16,cmap_number=0,
                           tick_tuple = None):
    ## find figure sizes
    fig = ax.get_figure()
    cur_height,cur_width = fig.get_size_inches()*fig.dpi
    offset = cmap_number * (150/cur_width)
    
    ## make the colorbar axes
    ax1 = fig.add_axes([0.95 + offset, 0.125, 15./cur_width, 0.75])
    
    ## setup segmented colormap
    cmap = matplotlib.colors.ListedColormap(colors)
    cmap.set_over(colors[-1])
    cmap.set_under(colors[0])
    
    if logflag:
        ticks = 10**np.linspace(np.log10(vmin),np.log10(vmax),len(cmap.colors)+1)[1:-1]
        norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
        tick_labels = [r"$10^{%.2f}$"%tick for tick in np.log10(ticks)]
    else:
        ticks = np.linspace(vmin,vmax,len(cmap.colors)+1)[1:-1]
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        tick_labels=ticks
        
    ## allow explicit tick placement and labelling
    if tick_tuple is not None: 
        ticks,tick_labels = tick_tuple
    
    cb = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    extend='both',
                                    extendfrac=.05,
                                    extendrect=False,
                                    ticks=ticks,
                                    orientation='vertical')
    cb.set_label(label,fontsize=fontsize)
    cb.set_ticklabels(tick_labels)
    cb.ax.tick_params(labelsize=fontsize-2)
    return lambda x: cmap(norm(x))

def plotMulticolorLine(ax,xs,ys,zs,cmap,n_interp=50,**kwargs):
    """
        takes x/y values and creates a line collection object
        of line segments between points in x/y colored by cmap(zs). 
        zs should be between 0 and 1
    """

    xs = linearInterpolate(xs,n_interp)
    ys = linearInterpolate(ys,n_interp)
    zs = linearInterpolate(zs,n_interp)

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap,norm=plt.Normalize(0, 1),**kwargs)
    lc.set_array(zs)
    lc.set_linewidth(3)
    ax.add_collection(lc)

    
def linearInterpolate(xs,n_interp):
    final = np.array([])
    for i,xr in enumerate(xs[1:]):
        xl = xs[i]
        final = np.append(final,np.linspace(xl,xr,n_interp))
    return final



## alternatively use 
#from matplotlib.colors import LinearSegmentedColormap
#cm = LinearSegmentedColormap.from_list('rgb', [[1,0,0,1],[0,1,0,1],[0,0,1,1]], N=50)

def make_colormap(mycolors,ninterp=100):
    """ Takes a series of RGBA arrays and interpolates between 
        them to create a colormap. Results are not guaranteed
        to be aesthetically pleasing """ 

    thecolors = np.array([])
    for i in range(len(mycolors)-1):
        rs = np.linspace(mycolors[i][0],mycolors[i+1][0],ninterp,endpoint=1)
        gs = np.linspace(mycolors[i][1],mycolors[i+1][1],ninterp,endpoint=1)
        bs = np.linspace(mycolors[i][2],mycolors[i+1][2],ninterp,endpoint=1)
        
        thecolors = np.append(thecolors,np.array([rs,gs,bs,[1]*len(rs)]).T)
    thecolors = thecolors.reshape(-1,4)
    indices = 1.0*np.arange(len(thecolors))/len(thecolors)


    def my_cmap(i):
        try:
            len(i)
            argmin = np.argmin((indices[:,None]-i)**2,axis=0)
        except:
            argmin = np.argmin((indices-i)**2)
            
        return thecolors[argmin]

    return my_cmap

def plotMultiColorHist(ax,edges,h,vmin,vmax,ncolors = 4, clabel =''):

    ## setup ticks and colors
    viridis = plt.get_cmap('viridis')
    colors = [viridis(i) for i in np.linspace(0,1,ncolors)]
    ticks = 10**np.linspace(np.log10(vmin),np.log10(vmax),ncolors+1)

    for i,(xl,xr,y) in enumerate(zip(edges[:-1],edges[1:],h)):
        cindex = get_cindex(y,ticks)
        ## plot the horizontal bars
        ax.plot([xl,xr],[y,y],color=colors[cindex],lw=3)
        
        ## have to plot the vertical part, let's do the left edge
        if i != 0: 
            y_prev = h[i-1]
            prev_cindex = get_cindex(y_prev,ticks)
            ## we crossed a threshold
            if prev_cindex != cindex:
                if y > y_prev:
                    ax.plot([xl]*2,[ticks[cindex],y],c=colors[cindex],lw=3)
                    ax.plot([xl]*2,[ticks[cindex],y_prev,],c=colors[prev_cindex],lw=3)
                else:
                    ax.plot([xl]*2,[ticks[prev_cindex],y_prev],c=colors[prev_cindex],lw=3)
                    ax.plot([xl]*2,[ticks[prev_cindex],y],c=colors[cindex],lw=3)
                    
            ## simple, let's just plot a single color between the two points
            else:
                ax.plot([xl]*2,[y_prev,y],lw=3,c=colors[cindex])

    
    color_mapper = addSegmentedColorbar(
        ax,colors,
        vmin,vmax,
        clabel,
        logflag=1,
        tick_tuple=(ticks[1:-1],["%.2f"%tick for tick in ticks[1:-1]]))

def get_cindex(y,ticks):

    try:
        cindex = np.where(ticks > y)[0][0]-1
        cindex+= 1 if cindex == -1 else 0 
    except:
        cindex = -1
    return cindex

def talkifyAxes(axs,lw=2,labelsize=24,ticklabelsize=16):
    axs = np.array(axs).flatten() ## klugey way of accepting single or  multiple axes
    for ax in axs:
        for axis in ['top','bottom','left','right']:
              ax.spines[axis].set_linewidth(lw)
        ax.xaxis.label.set_size(labelsize)
        ax.yaxis.label.set_size(labelsize)
        if ax.is_first_col():
            ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(),fontsize=ticklabelsize)
        if ax.is_last_row():
            ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(),fontsize=ticklabelsize)

