import numpy as np 

#math functions (trig and linear algebra...)
def vectorsToRAAndDec(vectors):
    xs,ys,zs = np.transpose(vectors)
    ## puts the meridian at x = 0
    ra = np.arctan2(ys,xs)

    ## puts the equator at z = 0
    dec = np.arctan2(zs,(xs**2+ys**2))

    return ra,dec

def getThetasTaitBryan(angMom):
    """As Euler angles but xyz vs. zxz. Returns degrees!"""
    theta_TB = np.arctan2(angMom[1],np.sqrt(angMom[0]**2+angMom[2]**2))*180/np.pi
    phi_TB = np.arctan2(-angMom[0],angMom[2])*180/np.pi

    return theta_TB,phi_TB

def rotateEuler(
    theta,phi,psi,
    pos,
    order='xyz', ## defaults to Tait-Bryan, actually
    rotation_point=None,
    loud=True,
    inverse=False):

    ## if need to rotate at all really -__-
    if theta==0 and phi==0 and psi==0: return pos

    if rotation_point is None: rotation_point = np.zeros(3)
    
    pos=pos-rotation_point

    # rotate particles by angle derived from frame number
    theta_rad = np.pi*theta/ 180
    phi_rad   = np.pi*phi  / 180
    psi_rad   = np.pi*psi  / 180

    c1 = np.cos(theta_rad)
    s1 = np.sin(theta_rad)
    c2 = np.cos(phi_rad)
    s2 = np.sin(phi_rad)
    c3 = np.cos(psi_rad)
    s3 = np.sin(psi_rad)

    # construct rotation matrix
    ##  Tait-Bryan angles
    if order == 'xyz':
        if loud: print('Using Tait-Bryan angles (xyz). Change with order=zxz.')
        rot_matrix = np.array([
            [c2*c3           , - c2*s3         , s2    ],
            [c1*s3 + s1*s2*c3, c1*c3 - s1*s2*s3, -s1*c2],
            [s1*s3 - c1*s2*c3, s1*c3 + c1*s2*s3, c1*c2 ]],
            dtype = np.float32)

    ##  classic Euler angles
    elif order == 'zxz':
        if loud: print('Using Euler angles (zxz). Change with order=xyz.')
        rot_matrix = np.array([
            [c1*c3 - c2*s1*s3, -c1*s3 - c2*c3*s1, s1*s2 ],
            [c3*s1 + c1*c2*s3, c1*c2*c3 - s1*s3 , -c1*s2],
            [s2*s3           , c3*s2            , c2    ]])
    else:
        raise Exception("Bad order")

    ## the inverse of a rotation matrix is its tranpose
    if inverse: rot_matrix = rot_matrix.T

    ## rotate about each axis with a matrix operation
    pos_rot = np.matmul(rot_matrix,pos.T).T

    ## on 11/23/2018 (the day after thanksgiving) I discovered that 
    ##  numpy will change to column major order or something if you
    ##  take the transpose of a transpose, as above. Try commenting out
    ##  this line and see what garbage you get. ridiculous.
    ##  also, C -> "C standard" i.e. row major order. lmfao
    pos_rot = np.array(pos_rot,order='C')
    
    ### add the frame_center back
    pos_rot+=np.array(np.matmul(rot_matrix,rotation_point.T).T,order='C')

    return pos_rot

def applyRandomOrientation(coords,vels,random_orientation):
    ## interpret the random_orientation variable as a seed
    np.random.seed(random_orientation)

    ## position angles of random orientation vector 
    theta = np.arccos(1-2*np.random.random())
    phi = np.random.random()*2*np.pi

    ## convert from position angles to rotation angles
    orientation_vector = np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)])
    new_theta,new_phi = getThetasTaitBryan(orientation_vector)

    ## rotate the coordinates and velocities 
    if coords is not None:
        coords = rotateEuler(new_theta,new_phi,0,coords,loud=False)
    if vels is not None:
        vels = rotateEuler(new_theta,new_phi,0,vels,loud=False)

    return orientation_vector,new_theta,new_phi,coords,vels

def rotateSnapshot(which_snap,theta,phi,psi):
    if 'Coordinates' in which_snap:
        which_snap['Coordinates']=rotateEuler(theta,phi,psi,which_snap['Coordinates'],loud=False)
    if 'Velocities' in which_snap:
        which_snap['Velocities']=rotateEuler(theta,phi,psi,which_snap['Velocities'],loud=False)
    return which_snap

try:
    from numba import jit
    @jit(nopython=True)
    def get_cylindrical_velocities(vels,coords):
        this_coords_xy = coords[:,:2]
        this_radii_xy = np.sqrt(
            np.array([
                np.linalg.norm(this_coords_xy[pi,:]) for
                pi in range(len(this_coords_xy))])**2)

        rhats = np.zeros((len(this_coords_xy),2))
        rhats[:,0] = this_coords_xy[:,0]/this_radii_xy
        rhats[:,1] = this_coords_xy[:,1]/this_radii_xy

        vrs = np.sum(rhats*vels[:,:2],axis=1)
        #vrs = np.zeros(len(this_coords))
        #for pi in range(len(this_coords)):
            #vrs[pi] = np.sum(this_coords[pi,:2]/np.sum

        vzs = vels[:,2]

        vphis = np.sqrt(
            np.array([
                np.linalg.norm(vels[i,:]) for
                i in range(len(vels))
            ])**2 -
            vrs**2 -
            vzs**2)
        return vrs,vphis,vzs
except ImportError:
    print("Couldn't import numba. Missing:")
    print("abg_python.all_utils.get_cylindrical_velocities")
