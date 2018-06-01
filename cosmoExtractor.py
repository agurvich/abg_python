import h5py,sys,getopt,os
import numpy as np
from abg_python.all_utils import filterDictionary,rotationMatrixZ,rotationMatrixY,rotateVectors

from abg_python.snapshot_utils import openSnapshot

def makeOutputDir(snapdir):
    datadir=os.path.join(snapdir,'subsnaps')
    if 'subsnaps' not in os.listdir(snapdir):
        print 'making directory subsnaps in %s' % snapdir
        os.mkdir(datadir)
    return datadir

## geometry functions
def rotateVectorsZY(thetay,thetaz,vectors):
    rotatedCoords=rotateVectors(rotationMatrixZ(thetaz),vectors)
    rotatedCoords=rotateVectors(rotationMatrixY(thetay),rotatedCoords)
    return rotatedCoords

## volume index extraction
def extractSphericalVolumeIndices(rs,rcom,radius2):
    return np.sum((rs - rcom)**2.,axis=1) < radius2

def extractRectangularVolumeIndices(rs,rcom,radius,height):
   x_indices = (rs-rcom)[:,0]**2<radius**2
   y_indices = (rs-rcom)[:,1]**2<radius**2

   height = radius if height==0 else height
   z_indices = (rs-rcom)[:,2]**2<height**2
   return np.logical_and(np.logical_and(x_indices,y_indices),z_indices)

def extractCylindricalVolumeIndices(rs,rcom,radius,height):
    xyindices = np.sum((rs-rcom)[:,:2]**2.,axis=1)<radius**2.
    #absolute value is expensive?
    zindices = (rs[:,2]-rcom[2])**2. < height**2.
    indices = np.logical_and(xyindices,zindices)
    return indices

## physics helper functions
def getVcom(masses,velocities):
    return np.sum(masses[:,None]*velocities,axis=0)/np.sum(masses)

def getAngularMomentum(vectors,masses,velocities):
    return np.sum(np.cross(vectors,masses[:,None]*velocities),axis=0)

def getAngularMomentumSquared(vectors,masses,velocities):
    ltot = np.sum(# sum in quadrature |lx|,|ly|,|lz|
        np.sum( # sum over particles 
            np.abs(np.cross( # |L| = |(r x mv )|
                vectors,
                masses[:,None]*velocities))
            ,axis=0)**2
        )**0.5 # msun - kpc - km/s units of L

    return ltot**2

    Li = np.cross(vectors,masses[:,None]*velocities)
    L2i = np.sum(Li*Li,axis=1)

    return np.sum(L2i)

def getThetas(angMom):
    thetay = np.arctan2(np.sqrt(angMom[0]**2+angMom[1]**2),(angMom[2]))
    thetaz = np.arctan2(angMom[1],angMom[0])
    #print "want to rotate by",thetay*180/np.pi,thetaz*180/np.pi
    return thetay,thetaz

def iterativeCoM(coords,masses,n=4,r0=np.array([0,0,0])):
    rcom = r0
    for i in xrange(n):
        sindices= extractSphericalVolumeIndices(coords,rcom,1000**2/10**i)
        #sindices= extractSphericalVolumeIndices(coords,rcom,10000**2/100**i)
        rcom = np.sum(coords[sindices]*masses[sindices][:,None],axis=0)/np.sum(masses[sindices])
    return rcom

def orientDisk(smasses,svs,srs,scom):
    ## find com velocity of disk
    vscom = getVcom(smasses,svs)

    ## find angular momentum vector
    angMom = getAngularMomentum((srs-scom),
        smasses,(svs-vscom))
    ## find angles necessary to rotate coordinates to align with angMom
    thetay,thetaz = getThetas(angMom)
    return vscom,thetay,thetaz

def findContainedScaleHeight(zs,ms):
    """ a more "physical" scale height?""" 
    return np.sqrt(np.sum(zs**2*ms)/np.sum(ms))

##### main extraction protocols
def extractDiskFromReadsnap(star_snap,snap,radius,scom=None,orient_stars=0):
    if star_snap is None:
        srs,svs,smasses=None,None,None
    else:
        srs,svs,smasses=star_snap['Coordinates'],star_snap['Velocities'],star_snap['Masses']

    return extractDiskFromArrays(
        srs,svs,smasses,
        snap['Coordinates'],snap['Velocities'],snap['Masses'],snap['Density'],
        radius,scom=scom,orient_stars=orient_stars)

def extractDiskFromArrays(
    srs,svs,smasses,
    rs,vs,masses,rhos,
    radius,scom=None,orient_stars=0):
    """Takes arrays from a snapshot and returns the information required
        from extractDisk. Useful to separate so that external protocols can 
        call it (multiple snapshots, for instance)
        Input: 
            srs/svs/smasses - positions,velocities, and masses of star particles
            rs/rhos - positions and densities of gas particles 
            radius - radius to extract particles from
    """
    #scom = None
    if scom is None:
        ## find com using iterative shells
        scom = np.sum(smasses[:,None]*srs,axis=0)/np.sum(smasses)
        scom = iterativeCoM(srs,smasses,r0=scom)#rs[np.argmax(rhos)])

    if radius is None:
        raise Exception("Should be using 5rstarhalf!")
        #mass_fact = np.sum(smasses)+np.sum(masses) 
        big_radius = 25  # if baryonic mass is > 10^10 solar masses then should use 25 kpc as outer radius
        sindices= extractSphericalVolumeIndices(srs,scom,big_radius**2)
        gindices= extractSphericalVolumeIndices(rs,scom,big_radius**2)
        spherical_galactocentric_radii = np.sum((rs[gindices]-scom)**2,axis=1)**0.5
        radius = np.sum(spherical_galactocentric_radii*masses[gindices])/np.sum(masses[gindices])
        print "Determined radius to be:",radius

    ## extract particles within radius cube
    #gindices = extractRectangularVolumeIndices(rs,scom,radius,height=radius) 
    #sindices = extractRectangularVolumeIndices(srs,scom,radius,height=radius)

    ## extract particles within radius sphere
    if srs is not None:
        sindices = extractSphericalVolumeIndices(srs,scom,radius**2)
    else:
        sindices = None
    gindices= extractSphericalVolumeIndices(rs,scom,radius**2)
    
    ## orient along angular momentum vector
    if orient_stars:
        vscom,thetay,thetaz = orientDisk(smasses[sindices],svs[sindices],srs[sindices],scom)
    else:
        vscom,thetay,thetaz = orientDisk(masses[gindices],vs[gindices],rs[gindices],scom)

    return thetay,thetaz,scom,vscom,gindices,sindices,radius

def diskFilterDictionary(
    star_snap,snap,
    radius,cylinder='',
    scom=None,dark_snap=None,orient_stars=0,
    rect_buffer=1.1):
    """ Takes two openSnapshot dictionaries and returns a filtered subset of the particles
        that are in the disk, with positions and velocities rotated"""
    ## make sure someone didn't pass no stars but ask us to orient the disk about the stars
    if star_snap is None:
        orient_stars=0

    thetay,thetaz,scom,vscom,gindices,sindices,radius=extractDiskFromReadsnap(
        star_snap,snap,radius,scom=scom,orient_stars=orient_stars)
    
    ## store com information in case its relevant
    new_snap = {'scale_radius':radius,
        'scom':scom,'vscom':vscom,
        'thetay':thetay,'thetaz':thetaz,}
    ## rotate the coordinates of the spherical extraction
    new_rs = rotateVectorsZY(thetay,thetaz,snap['Coordinates']-scom)
    new_vs = rotateVectorsZY(thetay,thetaz,snap['Velocities']-vscom)

    if star_snap is not None:
        new_star_snap = {'scale_radius':radius,
            'scom':scom,'vscom':vscom,
            'thetay':thetay,'thetaz':thetaz}
        ## rotate the coordinates of the spherical extraction
        new_star_rs = rotateVectorsZY(thetay,thetaz,star_snap['Coordinates']-scom)
        new_star_vs = rotateVectorsZY(thetay,thetaz,star_snap['Velocities']-vscom)

    if dark_snap is not None:
        new_dark_snap = {'scale_radius':radius,
            'scom':scom,'vscom':vscom,
            'thetay':thetay,'thetaz':thetaz}
        ## rotate position/velocity vectors
        new_dark_rs = rotateVectorsZY(thetay,thetaz,dark_snap['Coordinates']-scom)
        new_dark_vs = rotateVectorsZY(thetay,thetaz,dark_snap['Velocities']-vscom)
        ## extract spherical volume
        dindices = extractSphericalVolumeIndices(new_dark_rs,np.zeros(3),radius**2)

    if orient_stars:
        ## calculate stars "diskiness"
        angMom = getAngularMomentum(new_star_rs[sindices],
            star_snap['Masses'][sindices]*1e10,new_star_vs[sindices])# msun - kpc - km/s units of L
            ## post-rotation, lz == ltot by definition, lx, ly = 0 
        lz = angMom[2]

            ## add up ltot as sum(|l_i|), doesn't cancel counter-rotating stuff
        ltot = getAngularMomentumSquared(new_star_rs[sindices],
            star_snap['Masses'][sindices]*1e10,new_star_vs[sindices])**0.5 # msun - kpc - km/s units of L

        new_star_snap['star_lz']=lz
        new_star_snap['star_ltot']=ltot
    else:
        ## calculate gas "diskiness"
        angMom = getAngularMomentum(new_rs[gindices],
            snap['Masses'][gindices]*1e10,new_vs[gindices])# msun - kpc - km/s units of L
            ## post-rotation, lz == ltot by definition, lx, ly = 0 
        lz = angMom[2]

            ## add up ltot as sum(|l_i|), doesn't cancel counter-rotating stuff
        ltot = getAngularMomentumSquared(new_rs[gindices],
            snap['Masses'][gindices]*1e10,new_vs[gindices])**0.5 # msun - kpc - km/s units of L

        new_snap['lz']=lz
        new_snap['ltot']=ltot

    #overwrite gindices/sindices/dindices to get a square instead of a disk
    if cylinder != '':
        if cylinder is None:
            cylinder = findContainedScaleHeight(new_rs[:,2][gindices],snap['Masses'][gindices])
        gindices = extractRectangularVolumeIndices(new_rs,np.zeros(3),radius*rect_buffer,cylinder) 
        if star_snap is not None:
            sindices = extractRectangularVolumeIndices(new_star_rs,np.zeros(3),radius*rect_buffer,cylinder)
        if dark_snap is not None:
            dindices = extractRectangularVolumeIndices(new_dark_rs,np.zeros(3),radius*rect_buffer,cylinder)
        new_snap['scale_h']=cylinder

    ## add positions and velocities separately, since they need to be rotated and indexed
    new_snap['Coordinates'] = new_rs[gindices]
    new_snap['Velocities'] = new_vs[gindices]
    ## index the snapt of the keys
    new_snap = filterDictionary(snap,gindices,dict1 = new_snap, key_exceptions = ['Coordinates','Velocities'])

    if star_snap is not None:
        ## add positions and velocities separately, since they need to be rotated and indexed
        new_star_snap['Coordinates'] = new_star_rs[sindices]
        new_star_snap['Velocities'] = new_star_vs[sindices]
        ## index the rest of the keys
        new_star_snap = filterDictionary(
            star_snap,sindices,
            dict1 = new_star_snap,
            key_exceptions = ['Coordinates','Velocities'])

    if dark_snap is not None:
        ## update positions/velocities 
        new_dark_snap['Coordinates'] = new_dark_rs[dindices]
        new_dark_snap['Velocities'] = new_dark_vs[dindices]
        new_dark_snap = filterDictionary(
            dark_snap,dindices,
            dict1 = new_dark_snap,
            key_exceptions = ['Coordinates','Velocities'])

    return_list = [new_snap]

    if star_snap is not None:
        return_list+=[new_star_snap]
    if dark_snap is not None:
        return_list+=[new_dark_snap]

    return return_list
