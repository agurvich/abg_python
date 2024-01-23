from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

def load_my_custom_color_tables():
    fna='heat_red'
    cdict_tmp={\
    'red':   ((0., 0.0416, 0.0416),(0.365079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'green': ((0., 0., 0.),(0.71, 0.000000, 0.000000),\
            (1., 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'blue':  ((0., 0., 0.),(0.71, 0.000000, 0.000000),(1.0, 1.0, 1.0))}
    my_cmap = LinearSegmentedColormap(fna,cdict_tmp,256)
    mpl.colormaps.register(name=fna,cmap=my_cmap)

    fna='heat_blue'
    cdict_tmp={\
    'blue':   ((0., 0.0416, 0.0416),(0.365079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'green': ((0., 0., 0.),(0.365079, 0.000000, 0.000000),\
            (0.746032, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'red':  ((0., 0., 0.),(0.746032, 0.000000, 0.000000),(1.0, 1.0, 1.0))}
    my_cmap = LinearSegmentedColormap(fna,cdict_tmp,256)
    mpl.colormaps.register(name=fna,cmap=my_cmap)

    fna='heat_green'
    cdict_tmp={\
    'green':   ((0., 0.0416, 0.0416),(0.365079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'red': ((0., 0., 0.),(0.365079, 0.000000, 0.000000),\
            (1.000000, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'blue':  ((0., 0., 0.),(0.1, 0.000000, 0.000000),(1.0, 1.0, 1.0))}
    my_cmap = LinearSegmentedColormap(fna,cdict_tmp,256)
    mpl.colormaps.register(name=fna,cmap=my_cmap)

    fna='heat_redyellow'
    cdict_tmp={\
    'red':   ((0., 0.0416, 0.0416),(0.365079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'green': ((0., 0., 0.),(0.365079, 0.000000, 0.000000),\
            (0.746032, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'blue':  ((0., 0., 0.),(0.746032, 0.000000, 0.000000),(1.0, 1.0, 1.0))}
    my_cmap = LinearSegmentedColormap(fna,cdict_tmp,256)
    mpl.colormaps.register(name=fna,cmap=my_cmap)

    fna='heat_yellow'
    cdict_tmp={\
    'red':   ((0., 0.0416, 0.0416),(0.365079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'green':   ((0., 0.0416, 0.0416),(0.365079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'blue':  ((0., 0., 0.),(0.746032, 0.000000, 0.000000),(1.0, 1.0, 1.0))}
    my_cmap = LinearSegmentedColormap(fna,cdict_tmp,256)
    mpl.colormaps.register(name=fna,cmap=my_cmap)

    fna='heat_purple'
    cdict_tmp={\
    'red':   ((0., 0.0416, 0.0416),(0.565079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'green': ((0., 0., 0.),(0.565079, 0.000000, 0.000000),\
            (0.946032, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'blue':   ((0., 0.0416, 0.0416),(0.565079, 1.000000, 1.000000),(1.0, 1.0, 1.0))}
    my_cmap = LinearSegmentedColormap(fna,cdict_tmp,256)
    mpl.colormaps.register(name=fna,cmap=my_cmap)

    fna='heat_orange'
    cdict_tmp={\
    'red':   ((0., 0.0416, 0.0416),(0.365079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),\
    'green':   ((0., 0.0416, 0.0416),(0.365079, 0.400000, 0.400000),(1.0, 1.0, 1.0)),\
    'blue':  ((0., 0., 0.),(0.746032, 0.000000, 0.000000),(1.0, 1.0, 1.0))}
    my_cmap = LinearSegmentedColormap(fna,cdict_tmp,256)
    mpl.colormaps.register(name=fna,cmap=my_cmap)

    """
    fna='rainbow'
    cdict_tmp = {'red':   ((0.0, 80/256., 80/256.),
                       (0.2, 0.0, 0.0),
                       (0.4, 0.0, 0.0),
                       (0.6, 256/256., 256/256.),
                       (0.95, 256/256., 256/256.),
                       (1.0, 150/256., 150/256.)),
             'green': ((0.0, 0/256., 0/256.),
                       (0.2, 0/256., 0/256.),
                       (0.4, 130/256., 130/256.),
                       (0.6, 256/256., 256/256.),
                       (1.0, 0.0, 0.0)),
             'blue':  ((0.0, 80/256., 80/256.),
                       (0.2, 220/256., 220/256.),
                       (0.4, 0.0, 0.0),
                       (0.6, 20/256., 20/256.),
                       (1.0, 0.0, 0.0))}
    my_cmap = LinearSegmentedColormap(fna,cdict_tmp,256)
    mpl.colormaps.register(name=fna,cmap=my_cmap)
    """
