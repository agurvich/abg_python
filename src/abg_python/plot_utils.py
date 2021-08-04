## from builtin
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from abg_python.all_utils import pairFilter,covarianceEigensystem
from scipy.interpolate import interp1d

latex_pagewidth=6.9738480697 ## in #7.125
latex_columnwidth=3.32 ## in


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

## stupid way of handling black faceolor... 
GLOBAL_linecolor='k'

def add_many_to_legend(
    ax,
    line_labels,
    line_kwargss=None,
    make_new_legend=False,
    **legend_kwargs):
    """ use this for markers because matplotlib won't fix their bug!"""

    
    if line_kwargss is None:
        line_kwargss = [{} for i in len(line_labels)]

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
    for i,label in enumerate(line_labels):
        line_kwargs = line_kwargss[i]
        line = Line2D([0],[0],**line_kwargs)

        if label not in labels:
            lines.append(line)
            labels.append(label)

    ## for backwards compatibility...
    ax.legend(lines,labels,**legend_kwargs)

    return ax



    
        
def add_to_legend(
    ax,
    label='',
    shape='line',
    loc=0,
    ls='-',
    c='k',
    alpha=1,
    lw=None,
    legend_kwargs=None,
    make_new_legend=False,
    **kwargs):

    legend = ax.get_legend()
    ## add the current legend to the tracked artists
    ##  and then pretend we weren't passed one
    prev_loc = None
    if make_new_legend and legend is not None:
        prev_loc = legend._loc
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
        ## this is wild, but have to handle when plotting
        ##  markers, apparently when you read the line from 
        ##  the legend it loses memory of the marker
        if 'ls' in kwargs and kwargs['ls'] == '':
            ax.markers = []
        line_kwargs = {}

        if lw is not None:
            line_kwargs['lw'] = lw

        line = Line2D(
        [0],[0],
        ls=ls,
        c=c,
        alpha=alpha,
        **line_kwargs
        )
    else:
        raise NotImplementedError

    if label not in labels:
        lines.append(line)
        labels.append(label)

    if loc in legend_kwargs:
        loc = legend_kwargs.pop('loc')

    for line in lines:
        if line.get_linestyle() == 'None':
            print(line.get_marker(),'marker')

    if prev_loc is not None and loc == prev_loc:
        loc+=1

    ## for backwards compatibility...
    legend_kwargs.update(kwargs)
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

def plotEllipse(
    ax,
    x,y,
    semi_major,
    semi_minor,
    angle=0,
    fill=False,
    lw=3,
    log=False,
    **kwargs):
    from matplotlib.patches import Ellipse
    from matplotlib.transforms import ScaledTranslation

    if log:
        # use the axis scale tform to figure out how far to translate 
        circ_offset = ScaledTranslation(x,y,ax.transScale)

        # construct the composite tform
        circ_tform = circ_offset + ax.transLimits + ax.transAxes

        # create the circle centred on the origin, apply the composite tform
        circ = Ellipse(
            (0,0),
            2*semi_major,
            2*semi_minor,
            angle=angle,
            fill=fill,lw=lw,
            transform=circ_tform,
            **kwargs)
        ax.add_artist(circ)

    else:
        return ax.add_artist(
            Ellipse(
                (x,y),
                semi_major,
                semi_minor,
                angle=angle,
                fill=fill,
                lw=lw,
                **kwargs))

def addColorbar(
    ax,cmap,
    vmin,vmax,
    label,logflag = 0,
    fontsize=12,cmap_number=0,
    tick_tuple=None,
    horizontal = False,
    span_full_figure=True):

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
    bbox = ax.get_position()
    extents = bbox.extents
    offset = 10
    if span_full_figure:
        for ax in fig.get_axes():
            bbox = ax.get_position()
            these_extents = bbox.extents
            for i in range(2):
                if these_extents[i] < extents[i]:
                    extents[i] = these_extents[i]
            for i in range(2,4):
                if these_extents[i] > extents[i]:
                    extents[i] = these_extents[i]
        height = extents[3] - extents[1]
        width = extents[2] - extents[0]
    else:
        height = bbox.height
        width = bbox.width

    fig_x0,fig_y0,fig_x1,fig_y1 = extents

    if not horizontal:
        thickness = 20./cur_size[0] * fig.dpi/100
        ax1 = fig.add_axes([fig_x1+offset/cur_size[0], fig_y0, thickness, height])

    else:
        xlabel = ax.xaxis.get_label()
        if xlabel.get_text() != '':
            print("addColorbar does not support finding xaxis text, this will look bad")
        thickness = 20./cur_size[1] * fig.dpi/100
        ax1 = fig.add_axes([fig_x0,fig_y0 - thickness - offset/cur_size[1],width, thickness])

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)

    cb1 = matplotlib.colorbar.ColorbarBase(
        ax1, cmap=cmap,
        extend='both',
        extendfrac=0.05,
        norm=norm(vmin=vmin,vmax=vmax),
        orientation='vertical' if not horizontal else 'horizontal')


    cb1.set_label(label,fontsize=fontsize)

    cb1.set_ticks(ticks)
    if tick_labels is not None:
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
    axs = np.array(axs)
    axss = axs.reshape(nrows,ncols)

    if ylabels:
        for i,ax in enumerate(axs.flatten()):
            if i != nrows//2:
                ax.set_ylabel(ax.get_ylabel(),color=ax.get_facecolor())

    ## for each column that isn't the first
    for col_i in range(ncols):
        this_col = axss[:,col_i]
        for ax in this_col:
            if ylabels and not ax.is_first_col():
                ax.set_ylabel('')
            try:
                xticks = ax.get_xticklabels()
                xtick_strings = np.array([xtick.get_text() for xtick in xticks])
                if len(xticks) == 0:
                    continue

                ##  change the first tick
                if not ax.is_first_col():
                    xticks[0].set_horizontalalignment('left')
                ## if we're in the right most 
                ##  column we don't need to change the last tick
                #if col_i != (ncols-1):
                xticks[-1].set_horizontalalignment('right')
            except IndexError:
                pass ## this can fail if share_x = True

    for ax in axss.flatten():
        if xlabels:
            ax.set_xlabel('')
        try:
            yticks = ax.get_yticklabels()
            ## if we're in the first row don't 
            if len(yticks) == 0:
                continue
            ##  need to mess with the top tick
            if not ax.is_first_row():
                yticks[-1].set_verticalalignment('top')
            ## if we're in the last row we 
            ##  don't need to mess with the bottom tick
            if not ax.is_last_row():
                yticks[0].set_verticalalignment('bottom')
        except IndexError as e:
            pass ## this can fail if share_y = True
    
    fig = axs.flatten()[0].get_figure()
    if share_ylabel is not None:
        bbox = ax.get_position()
        fig.text(
            0-label_offset,0.5,
            share_ylabel,
            rotation=90,va='center',ha='right',fontsize=13)

    if share_xlabel is not None:
        fig.text(
            0.5,label_offset-0.05,
            share_xlabel,
            va='center',ha='center',fontsize=13)


def nameAxes(
    ax,title,xname,yname,logflag=(0,0),
    subtitle=None,supertitle=None,
    make_legend=0,off_legend=0,
    loc=0,
    slackify=0,width=8,height=6,
    yrotation=90,
    xlow=None,xhigh=None,
    ylow=None,yhigh=None,
    subfontsize=12,fontsize=None,
    xfontsize=None,yfontsize=None,
    font_color=None,font_weight='regular',
    legendkwargs=None,
    swap_annotate_side=False,
    subtextkwargs = None):
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
            ax.set_ylabel(yname,rotation=yrotation)
        else:
            ax.set_ylabel(yname,fontsize=yfontsize,rotation=yrotation)
            #for tick in ax.yaxis.get_major_ticks():
                #tick.label.set_fontsize(yfontsize)

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

    subtextkwargs={} if subtextkwargs is None else subtextkwargs
    if font_color is not None:
        subtextkwargs['color']=font_color
    if subfontsize is not None:
        subtextkwargs['fontsize']=subfontsize

    bbox = ax.get_position()
    if swap_annotate_side:
        x_pos = 1-0.01/bbox.width
        halign = 'right'
    else:
        x_pos = 0.01/bbox.width
        halign = 'left'

    if supertitle:

        y_pos = 1-(0.01/bbox.height)
        ax.text(x_pos,y_pos,supertitle,transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment=halign,
            weight=font_weight,**subtextkwargs)

    if subtitle:
        y_pos = (0.01/bbox.height)
        ax.text(x_pos,y_pos,subtitle,transform=ax.transAxes,
            verticalalignment='bottom',
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
            loc = loc+(supertitle is not None)
            if 'loc' in legendkwargs:
                loc = legendkwargs.pop('loc')
            ax.legend(
                loc=loc,
                frameon=0,**legendkwargs)
            return ax.get_legend_handles_labels()

def plot_histogram_contour_log(
    ax,
    xs,ys,
    xedges,yedges,
    plot_histogram=True,
    plot_points=False,
    contour_kwargs=None,
    plot_ellipse=True):

    ## initialize contour_kwargs, controls what contours are plotted, 
    ##  their linestyles (as an array), and their color (mainly... I'm sure there 
    ##  are other kwargs that can be passed to ax.contour)
    if contour_kwargs is None:
        contour_kwargs = {}

    if 'colors' not in contour_kwargs and 'cmap' not in contour_kwargs:
        contour_kwargs.update({'cmap':None,'colors':GLOBAL_linecolor})

    if 'percentiles' not in contour_kwargs:
        contour_kwargs['percentiles'] = [0.5,0.9]

    ## initialize 2d histogram
    dx,dy = xedges[1]-xedges[0],yedges[1]-yedges[0]
    xs,ys = pairFilter(np.log10(xs),np.log10(ys),np.isfinite)

    ## pad the edges with 1 more bin on each side
    #xedges = np.concatenate([[xedges[0]-dx],xedges,[xedges[-1]+dx]])

    ## make the 2d histogram
    h,xedges,yedges = np.histogram2d(
        xs,ys,
        bins=[xedges,yedges])
    X,Y = np.meshgrid((xedges[1:]+xedges[:-1])/2,(yedges[1:]+yedges[:-1])/2)

    from palettable.colorbrewer.sequential import Oranges_3,Greens_3,Blues_3
    from matplotlib.colors import LinearSegmentedColormap
    color = (1,1,1) if 'k' in GLOBAL_linecolor else(0,0,0)
    new_cmap = LinearSegmentedColormap.from_list(1,[color]+Oranges_3.mpl_colors)

    if plot_histogram:
        ax.pcolor(
            10**X,10**Y,
            h.T,
            lw=0,
            alpha=1,
            edgecolors='face',
            snap=True,
            cmap=new_cmap)

    if plot_points:
        ax.plot(10**xs,10**ys,'.',markeredgewidth=0,c='g')

    evecs,evals = covarianceEigensystem(xs,ys)

    ## choose new x-axis to be evecs[0], rotation angle is
    ##  angle between it and old x-axis, i.e.
    ##  ehat . xhat = cos(angle)
    angle = np.arccos(evecs[0][0])

    ## evals are variance along principle axes
    rx,ry = evals**0.5 ## in dex
    cx,cy = 10**np.mean(xs),10**np.mean(ys) ## in linear space

    if plot_ellipse:
        ## doesn't work for log
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
            color='limegreen')

    return_value = plot_percentile_contours(ax,10**X,10**Y,h,**contour_kwargs)

    return h,rx,ry,evecs,return_value

def plot_percentile_contours(
    ax,
    X,Y,h,
    percentiles,
    cmap='viridis',
    **contour_kwargs):
    """ from 
https://stackoverflow.com/questions/37890550/python-plotting-percentile-contour-lines-of-a-probability-distribution"""

    h= h/h.sum()
    n = 1000
    t = np.linspace(0, h.max(), n,endpoint=True)
    integral = ((h >= t[:, None, None]) * h).sum(axis=(1,2))

    ## contour levels must be "increasing" (so percentiles must be decreasing)
    ##  e.g. [0.9, 0.5, 0.1]
    percentiles.sort()
    percentiles = percentiles[::-1]

    if 'linestyles' not in contour_kwargs:
        linestyles=['-','-.','--',':'][::-1]
        contour_kwargs['linestyles'] = linestyles[-len(percentiles):]

    f = interp1d(integral, t)
    try:
        t_contours = f(np.array(percentiles))
        contours = ax.contour(
            X,Y,
            h.T,
            cmap=cmap,
            levels=t_contours,
            **contour_kwargs)
        return contours.levels
    except ValueError:
        print(percentiles,"not possible with given h, try smaller bins?")
        return []


def plot_avg_and_percentiles(
    ax,
    xs,
    yss,
    average_function,
    color,
    label,
    plot_avg=True,
    plot_many=True,
    linestyle='-'):

    my_averages = average_function(
        yss,
        axis=0)

    if plot_avg and ax is not None:
        ## plot the average of the terms
        ax.plot(
            xs,
            my_averages,
            color=color,
            label=label,
            ls=linestyle)

    ## plot the inter-quartile range of the terms
    sigmas = plot_percentiles_shaded_region(
        ax if plot_many else None,
        xs,
        yss,
        color)

    return my_averages,sigmas

def plot_percentiles_shaded_region(ax,xs,yss,color,percentiles=None):
    """plots a shaded region at percentiles of
    a distribution of lines"""
    
    percentiles = [25,75] if percentiles is None else percentiles
    bottom_qs,top_qs = np.nanpercentile(yss,percentiles,axis=0)
    
    if ax is not None:
        ## pretty sure this is handling whether the axis face-color
        ##  is black...
        facecolor = ax.get_facecolor() 
        if (facecolor[0] == facecolor[1] == facecolor[2]):
           alpha = 0.45 
        else:
            alpha = 0.15
            
        ax.fill_between(
            xs,
            bottom_qs,
            top_qs,
            lw=0,
            color=color,
            alpha=alpha)

    ## half-width of shaded region in log space
    return (np.log10(top_qs/bottom_qs))/2.

def set_matplotlib_params(matplotlib):
    matplotlib.rcParams['legend.frameon'] = False
 
    matplotlib.rcParams["xtick.direction"] = 'in'
    matplotlib.rcParams["ytick.direction"] = 'in'

    # Make the x and y ticks bigger                                                    
    matplotlib.rcParams['xtick.labelsize'] = 11
    matplotlib.rcParams['xtick.major.size'] = 5
    matplotlib.rcParams['xtick.major.width'] = .5
    matplotlib.rcParams['ytick.labelsize'] = 11
    matplotlib.rcParams['ytick.major.size'] = 5
    matplotlib.rcParams['ytick.major.width'] = .5
                                                                                       
    # Make the axes linewidths bigger                                                  
    matplotlib.rcParams['axes.linewidth'] = 1

    matplotlib.rcParams['lines.linewidth'] = 1.5

    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor'] = 'white'

    matplotlib.rcParams['axes.edgecolor'] = 'black'
    matplotlib.rcParams['axes.labelcolor'] = 'black'
    matplotlib.rcParams['text.color'] = 'black'

    matplotlib.rcParams['xtick.color'] = 'black'
    matplotlib.rcParams['ytick.color'] = 'black'

    ## font family 
    ##  math
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['mathtext.fontset'] = 'stix'

    matplotlib.rcParams['font.size'] = 10
    matplotlib.rcParams['axes.labelsize'] = 11
    matplotlib.rcParams['legend.fontsize'] = 11

    ##  outside math
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
    #matplotlib.rcParams['font.sans-serif'] = 'Computer Modern Sans serif'
    #matplotlib.rcParams['font.monospace'] = 'Computer Modern Typewriter'

    matplotlib.rcParams['figure.figsize'] = [latex_pagewidth/2,latex_pagewidth/2]
    matplotlib.rcParams['figure.dpi'] = 120

    matplotlib.rcParams['figure.subplot.bottom'] = 0
    matplotlib.rcParams['figure.subplot.top'] = 1
    matplotlib.rcParams['figure.subplot.left'] = 0
    matplotlib.rcParams['figure.subplot.right'] = 1

    matplotlib.rcParams['figure.subplot.hspace'] = 0
    matplotlib.rcParams['figure.subplot.wspace'] = 0

def set_matplotlib_params_black_talk(matplotlib):
    matplotlib.rcParams['legend.frameon'] = False

    matplotlib.rcParams["xtick.direction"] = 'in'
    matplotlib.rcParams["ytick.direction"] = 'in'

    # Make the x and y ticks bigger                                                    
    matplotlib.rcParams['xtick.major.size'] = 5
    matplotlib.rcParams['xtick.major.width'] = 1
    matplotlib.rcParams['ytick.major.size'] = 5
    matplotlib.rcParams['ytick.major.width'] = 1
                                                                                       
    # Make the axes linewidths bigger                                                  
    matplotlib.rcParams['axes.linewidth'] = 1

    matplotlib.rcParams['lines.linewidth'] = 1.5

    matplotlib.rcParams['figure.facecolor'] = 'k'
    matplotlib.rcParams['axes.facecolor'] = 'black'

    matplotlib.rcParams['axes.edgecolor'] = 'white'
    matplotlib.rcParams['axes.labelcolor'] = 'white'
    matplotlib.rcParams['text.color'] = 'white'

    matplotlib.rcParams['xtick.color'] = 'white'
    matplotlib.rcParams['ytick.color'] = 'white'

    matplotlib.rcParams['xtick.labelsize'] = 20
    matplotlib.rcParams['ytick.labelsize'] = 20
    matplotlib.rcParams['axes.labelsize'] = 20

    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['legend.fontsize'] = 16

    ## font family 
    ##  math
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['mathtext.fontset'] = 'stix'


    ##  outside math
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    #matplotlib.rcParams['font.serif'] = 'Computer Modern Roman'
    #matplotlib.rcParams['font.sans-serif'] = 'Computer Modern Sans serif'
    #matplotlib.rcParams['font.monospace'] = 'Computer Modern Typewriter'

    matplotlib.rcParams['figure.figsize'] = [latex_pagewidth/2,latex_pagewidth/2]
    matplotlib.rcParams['figure.dpi'] = 120

    matplotlib.rcParams['figure.subplot.bottom'] = 0
    matplotlib.rcParams['figure.subplot.top'] = 1
    matplotlib.rcParams['figure.subplot.left'] = 0
    matplotlib.rcParams['figure.subplot.right'] = 1

    matplotlib.rcParams['figure.subplot.hspace'] = 0
    matplotlib.rcParams['figure.subplot.wspace'] = 0

    global GLOBAL_linecolor
    GLOBAL_linecolor='white'

def ffmpeg_frames(
    framedir,
    frame_heads,
    savename=None,
    framerate=15,
    extension='.mp4',
    outdir=None,
    copy=True):

    savename = '' if savename is None else savename

    if outdir is None:
        outdir = os.path.dirname(framedir)
    
    for frame_head in frame_heads:
        cmd = 'ffmpeg -framerate %d'%framerate
        cmd += ' -i %s'%os.path.join(framedir,frame_head)
        cmd += '_%03d.png'
        cmd += ' -q:v 1'
        cmd += ' %s%s -y'%(os.path.join(outdir,frame_head),extension)
        print(cmd)
        os.system(cmd)

        ## copy the movie to ~/movies
        if copy:
            src = os.path.join(outdir,frame_head+extension)
            dst = os.path.join(
                os.environ['HOME'],
                'movies',
                savename + ('_'*(savename!='')) + frame_head+extension)
            cmd = 'cp %s %s'%(src,dst)
            print(cmd)
            os.system(cmd)
