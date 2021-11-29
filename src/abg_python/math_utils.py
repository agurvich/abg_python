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

def construct_quaternion(angles,order='zyx',units='deg'):

    axes = {'x':[1,0,0],
        'y':[0,1,0],
        'z':[0,0,1]}

    if units == 'deg': angles = np.array(angles)/180*np.pi

    quat = None
    for angle,axis in zip(angles,order):
        axis = axes[axis]
        if quat is None: quat = axisangle_to_q(axis,angle)
        else: quat = q_mult(axisangle_to_q(axis,angle),quat)

    return quat

def rotateQuaternion(quat,pos,inverse=False):

    ## basically throw away the entire point of having a quaternion
    ##  in order to broadcast the rotation. at least we can interpret
    ##  quaternion input /shrug
    rot_matrix = q_to_rotation_matrix(quat,inverse=inverse)

    ## rotate about each axis with a matrix operation
    pos_rot = np.array(np.matmul(rot_matrix,pos.T).T,order='C')

    ## on 11/23/2018 (the day after thanksgiving) I discovered that 
    ##  numpy will change to column major order or something if you
    ##  take the transpose of a transpose, as above. Try commenting out
    ##  this line and see what garbage you get. ridiculous.
    ##  also, C -> "C standard" i.e. row major order. lmfao
    return np.array(pos_rot,order='C')

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
        this_radii_xy = np.sqrt(np.sum(this_coords_xy[:,:2]**2,axis=1))

        rhats = np.zeros((len(this_coords_xy),2))
        rhats[:,0] = this_coords_xy[:,0]/this_radii_xy
        rhats[:,1] = this_coords_xy[:,1]/this_radii_xy

        vrs = np.sum(rhats*vels[:,:2],axis=1)
        #vrs = np.zeros(len(this_coords))
        #for pi in range(len(this_coords)):
            #vrs[pi] = np.sum(this_coords[pi,:2]/np.sum

        vzs = vels[:,2]

        vphis = np.sqrt(np.sum(vels**2,axis=1) - vrs**2 - vzs**2)

        return vrs,vphis,vzs

    @jit(nopython=True)
    def get_cylindrical_coordinates(coords):
    
        rs = np.sqrt(np.sum(coords[:,:2]**2,axis=1))

        phis = np.arctan2(coords[:,1],coords[:,0])
        phis[phis < 0 ]+=2*np.pi

        return rs,phis,coords[:,2]

    @jit(nopython=True)
    def get_spherical_velocities(vels,coords):

        ## phi is shared between cylindrical and spherical coordinates
        ##  this avoids having to dot each velocity with that particle's
        ##  phi hat vector
        _,vphis,_ = get_cylindrical_velocities(vels,coords)


        rs = np.sqrt(np.sum(coords**2,axis=1))

        rhats = np.zeros(coords.shape)
        rhats[:,0] = coords[:,0]/rs
        rhats[:,1] = coords[:,1]/rs
        rhats[:,2] = coords[:,2]/rs

        vrs = np.sum(rhats*vels,axis=1)

        ## vtheta is the remaining velocity
        vthetas = np.sqrt(
            np.sum(vels**2,axis=1) - 
            vrs**2 -
            vphis**2)

        return vrs,vthetas,vphis

    @jit(nopython=True)
    def get_spherical_coordinates(coords):
    
        rs = np.sqrt(np.sum(coords**2,axis=1))

        phis = np.arctan2(coords[:,1],coords[:,0])
        phis[phis < 0 ]+=2*np.pi

        ## reciprocal from what you might expect, z/R, because measured from the pole
        thetas = np.arctan2(np.sqrt(np.sum(coords[:,:2]**2,axis=1)),coords[:,2])

        return rs,thetas,phis


except ImportError:
    print("Couldn't import numba. Missing:")
    print("abg_python.math_utils.get_cylindrical_velocities")
    print("abg_python.math_utils.get_spherical_velocities")

## quaternion helper functions:
## https://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
def old_q_mult(q1, q2):
    """ rip """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2 # q1[0]*q2[0] - np.dot(q1[1:],q2[1:]) 
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.array([w, x, y, z])

def q_mult(q1,q2):
    outer = np.outer(q1,q2)
    vector_components = np.cross(np.identity(3),outer[1:,1:])
    q3 = np.zeros(4)

    signs = np.identity(4)
    signs[1:,1:]*=-1
    
    q3[0] = np.trace(signs*outer)
    outer[1:,1:] = vector_components
    q3[1:] = np.sum(outer,axis=0)[1:]+outer[1:,0]
    return q3

def multi_q_mult(q1,q2s):
    q1s = q1.reshape(1,4)
    outers = np.tensordot(q1s[None,:],q2s[None,:],axes=(0,0))
    outers = np.moveaxis(outers,2,1)[0]

    vector_componentss = np.cross(np.identity(3)[None,:],outers[:,1:,1:])
    
    q3s = np.zeros((q2s.shape[0],4))
    signs = np.identity(4)
    signs[1:,1:]*=-1
    q3s[:,0] = np.trace(signs[None,:]*outers,axis1=1,axis2=2)

    outers[:,1:,1:] = vector_componentss

    q3s[:,1:] = np.sum(outers,axis=1)[:,1:]+outers[:,1:,0]
    return q3s
    
def multi_qv_mult(q1,v2s):
    q1s = q1.reshape(1,4)
    outers = np.tensordot(q1s[None,:],v2s[None,:],axes=(0,0))
    outers = np.moveaxis(outers,2,1)[0]

    vector_componentss = np.cross(np.identity(3)[None,:],outers[:,1:,:])

    outers[:,1:,:] = vector_componentss

    return np.sum(outers,axis=1)

def q_conjugate(q):
    q_conj = np.repeat(np.nan,4)
    q_conj[0] = q[0]
    q_conj[1:] = -q[1:]
    return q_conj

def qv_mult(q1, v1):
    q2 = np.zeros(4)
    q2[1:] = v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def axisangle_to_q(axis, theta):
    ## normalize the rotation axis
    axis = axis/np.linalg.norm(axis) 
    ## initialize the output quaternion
    q = np.repeat(np.nan,4)
    
    ## fill scalar part
    q[0]  = np.cos(theta/2)
    
    ## fill vector part
    q[1:] = axis*np.sin(theta/2)
    return q

def q_to_rotation_matrix(quat,inverse=False):

    ## ensure quaternion is unit length
    quat = quat/np.linalg.norm(quat)

    ## the inverse of a quaternion rotation is just
    ##  setting theta=-theta. because the scalar
    ##  part is cos that is even, vector part is 
    ##  sin (odd) so it picks up a minus sign
    if inverse: quat[1:]*=-1

    rotation_matrix = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            foo = 2*quat[1+i]*quat[1+j]
            if i == j: foo = 1 + (foo - 2*np.linalg.norm(quat[1:])**2)
            rotation_matrix[i,j] = foo
    ## quat = r,i,j,k
    rotation_matrix[0,1]+=-2*quat[0]*quat[3]  ## qr*qk
    rotation_matrix[1,0]+= 2*quat[0]*quat[3]

    rotation_matrix[0,2]+= 2*quat[0]*quat[2] ## qr*qj
    rotation_matrix[2,0]+=-2*quat[0]*quat[2]

    rotation_matrix[1,2]+=-2*quat[0]*quat[1] ## qr*qi
    rotation_matrix[2,1]+= 2*quat[0]*quat[1]

    return rotation_matrix