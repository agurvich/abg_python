import numpy as np
import matplotlib
# -*- coding: iso-8859-1 -*-

"""
Colour-blind proof distinct colours module, based on work by Paul Tol
Pieter van der Meer, 2011
SRON - Netherlands Institute for Space Research
"""

# colour table in HTML hex format
hexcols = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', 
           '#CC6677', '#882255', '#AA4499', '#661100', '#6699CC', '#AA4466',
           '#4477AA']

greysafecols = ['#809BC8', '#FF6666', '#FFCC66', '#64C204']

xarr = [[12], 
        [12, 6], 
        [12, 6, 5], 
        [12, 6, 5, 3], 
        [0, 1, 3, 5, 6], 
        [0, 1, 3, 5, 6, 8], 
        [0, 1, 2, 3, 5, 6, 8], 
        [0, 1, 2, 3, 4, 5, 6, 8], 
        [0, 1, 2, 3, 4, 5, 6, 7, 8], 
        [0, 1, 2, 3, 4, 5, 9, 6, 7, 8], 
        [0, 10, 1, 2, 3, 4, 5, 9, 6, 7, 8], 
        [0, 10, 1, 2, 3, 4, 5, 9, 6, 11, 7, 8]]

# get specified nr of distinct colours in HTML hex format.
# in: nr - number of colours [1..12]
# returns: list of distinct colours in HTML hex
def get_distinct(nr):
    """Returns a list of colorblind safe 
        and distinct colors of size nr"""

    #
    # check if nr is in correct range
    #
    
    if nr < 1 or nr > 12:
        print("wrong nr of distinct colours!")
        return

    #
    # get list of indices
    #
    
    lst = xarr[nr-1]
    
    #
    # generate colour list by stepping through indices and looking them up
    # in the colour table
    #

    i_col = 0
    col = [0] * nr
    for idx in lst:
        col[i_col] = hexcols[idx]
        i_col+=1
    return col

#colormaps

# Deviation around zero colormap (blue--red)
cols = []
for x in np.linspace(0,1, 256):
    rcol = 0.237 - 2.13*x + 26.92*x**2 - 65.5*x**3 + 63.5*x**4 - 22.36*x**5
    gcol = ((0.572 + 1.524*x - 1.811*x**2)/(1 - 0.291*x + 0.1574*x**2))**2
    bcol = 1/(1.579 - 4.03*x + 12.92*x**2 - 31.4*x**3 + 48.6*x**4 - 23.36*x**5)
    cols.append((rcol, gcol, bcol))

cm_plusmin = matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_plusmin", cols)

# Linear colormap (white--red)
from scipy.special import erf

cols = []
for x in np.linspace(0,1, 256):
    rcol = (1 - 0.392*(1 + erf((x - 0.869)/ 0.255)))
    gcol = (1.021 - 0.456*(1 + erf((x - 0.527)/ 0.376)))
    bcol = (1 - 0.493*(1 + erf((x - 0.272)/ 0.309)))
    cols.append((rcol, gcol, bcol))

cm_linear = matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_linear", cols)

# Linear colormap (rainbow)
cols = [(0,0,0)]
for x in np.linspace(0,1, 254):
    rcol = (0.472-0.567*x+4.05*x**2)/(1.+8.72*x-19.17*x**2+14.1*x**3)
    gcol = 0.108932-1.22635*x+27.284*x**2-98.577*x**3+163.3*x**4-131.395*x**5+40.634*x**6
    bcol = 1./(1.97+3.54*x-68.5*x**2+243*x**3-297*x**4+125*x**5)
    cols.append((rcol, gcol, bcol))

cols.append((1,1,1))
cm_rainbow = matplotlib.colors.LinearSegmentedColormap.from_list("PaulT_rainbow", cols)


## ------------------------------------------------------------------------------- ##

##  my favorite colors for pressure terms
colors = get_distinct(6)
cr_pressure_colors = [colors[0],colors[-2],colors[1],colors[-1],colors[-3],colors[-4],colors[0]]
pressure_labels = ['disp','therm','bulk','cr','total','weight','kin']
pressure_colors_dict = dict(zip(
    pressure_labels,
    cr_pressure_colors))

pressure_colors = [colors[0],colors[-2],colors[1],colors[-3],colors[-4],colors[0]]

##  my favorite colors for gas phases
colors = get_distinct(4)
extended_colors = get_distinct(12)[::3]
extended_phases = ['cold','warm','hot','all','wnm','wim','star','dark']
phase_colors = [colors[0],colors[2],colors[1],colors[3],extended_colors[1],extended_colors[2],'green','purple']
phase_colors_dict = dict(zip(
    extended_phases,
    phase_colors))

## darken the yellow for the warm phase and total pressure
from matplotlib.colors import rgb_to_hsv,hex2color,hsv_to_rgb
rgb = hex2color(phase_colors_dict['warm'])
hsv = rgb_to_hsv(rgb)
hsv[-2]*=1.2
hsv[-1]/=1.1

phase_colors_dict['warm'] = hsv_to_rgb(hsv)
pressure_colors_dict['total'] = hsv_to_rgb(hsv)
phase_colors_dict['wim'] = hsv_to_rgb(hsv)

my_qualitative_colors = [
    phase_colors_dict['cold'],
    phase_colors_dict['warm'],
    phase_colors_dict['hot'],
    phase_colors_dict['all'],
    pressure_colors_dict['cr'],
    pressure_colors_dict['bulk'],
]

my_orange = rgb_to_hsv((
    np.array(hex2color(my_qualitative_colors[1])) + 
    np.array(hex2color(my_qualitative_colors[2])))/2)

my_orange[-2]*=1.4
my_orange[-1]*=1.1

my_qualitative_colors+=[hsv_to_rgb(my_orange) ]

