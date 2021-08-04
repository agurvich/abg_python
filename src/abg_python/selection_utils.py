import numpy as np

def rectangularVolumeMask(coords,radius,height=None,rcom=None):
    if rcom is None: rcom = np.zeros(3)
    if height is None: height = radius  

    r2s = (coords-rcom)**2
    x_indices = r2s[:,0]<=radius**2
    y_indices = r2s[:,1]<=radius**2

    z_indices = r2s[:,2]<=height**2
    return np.logical_and(np.logical_and(x_indices,y_indices),z_indices)

def cylindricalVolumeMask(coords,radius,height=None,rcom=None):
    if rcom is None: rcom = np.zeros(3)
    if height is None: height = radius  

    r2s = (coords-rcom)**2
    xy_indices = np.sum(r2s[:,:2],axis=1) <= radius**2.
    zindices = r2s[:,2] <= height**2.
    return np.logical_and(xy_indices,zindices)

def sphericalVolumeMask(coords,radius,rcom=None):
    if rcom is None: rcom = np.zeros(3)
    return np.sum((coords - rcom)**2.,axis=1) <= radius**2
