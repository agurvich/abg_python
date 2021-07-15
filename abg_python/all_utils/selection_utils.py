def extractRectangularVolumeIndices(rs,rcom,radius,height):
   x_indices = (rs-rcom)[:,0]**2<radius**2
   y_indices = (rs-rcom)[:,1]**2<radius**2

   height = radius if height==0 else height
   z_indices = (rs-rcom)[:,2]**2<height**2
   return np.logical_and(np.logical_and(x_indices,y_indices),z_indices)

def extractCylindricalVolumeIndices(coords,r,h,rcom=None):
    if rcom is None:
        rcom = np.array([0,0,0])
    gindices = np.sum((coords[:,:2])**2.,axis=1) < r**2.
    gzindices = (coords[:,2])**2. < h**2.
    indices = np.logical_and(gindices,gzindices)
    return indices

def extractSphericalVolumeIndices(rs,rcom,radius):
    return np.sum((rs - rcom)**2.,axis=1) < radius**2
