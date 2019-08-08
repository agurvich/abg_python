from __future__ import print_function
import sys,getopt,os
import numpy as np
from abg_python.all_utils import filterDictionary,rotationMatrixZ,rotationMatrixY,rotateVectors,unrotateVectorsZY

from abg_python.snapshot_utils import openSnapshot

def makeOutputDir(snapdir):
    datadir=os.path.join(snapdir,'subsnaps')
    if 'subsnaps' not in os.listdir(snapdir):
        print('making directory subsnaps in %s' % snapdir)
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
    assert np.sum(masses) > 0 
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
    #print("want to rotate by",thetay*180/np.pi,thetaz*180/np.pi)
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
def extractDiskFromSnapdict(
    star_snap,
    snap,
    radius,
    scom=None,
    orient_stars=0):

    if star_snap is None:
        srs,svs,smasses=None,None,None
    else:
        srs,svs,smasses=star_snap['Coordinates'],star_snap['Velocities'],star_snap['Masses']

    return extractDiskFromArrays(
        srs,svs,smasses,
        snap['Coordinates'],snap['Velocities'],snap['Masses'],
        radius,scom=scom,orient_stars=orient_stars)

def extractDiskFromArrays(
    srs,svs,smasses,
    rs,vs,masses,
    orient_radius,
    scom=None,
    orient_stars=0):
    """Takes arrays from a snapshot and returns the information required
        from extractDisk. Useful to separate so that external protocols can 
        call it (multiple snapshots, for instance)
        Input: 
            srs/svs/smasses - positions,velocities, and masses of star particles
            rs - positions and densities of gas particles 
            radius - radius to extract particles from
    """
    ## orient along angular momentum vector
    if orient_stars:
        if srs is None:
            raise ValueError("Was not passed star coordinates")
        sindices = extractSphericalVolumeIndices(
            srs,scom,orient_radius**2)
        vscom,thetay,thetaz = orientDisk(
            smasses[sindices],
            svs[sindices],
            srs[sindices],scom)
        gindices = None
    else:
        gindices= extractSphericalVolumeIndices(
            rs,scom,orient_radius**2)
        if not np.sum(gindices):
            print(scom,orient_radius)
            raise ValueError("No gas particles to orient the disk on!")

        vscom,thetay,thetaz = orientDisk(
            masses[gindices],
            vs[gindices],
            rs[gindices],scom)
        sindices = None

    return thetay,thetaz,scom,vscom,gindices,sindices

def offsetRotateSnapshot(snap,scom,vscom,thetay,thetaz,orient_stars):

    if 'overwritten' in snap and snap['overwritten']:
        if (snap['thetay'] == thetay and 
            snap['thetaz'] == thetaz and
            np.all(snap['scom'] == scom) and
            np.all(snap['vscom'] == vscom)):
            print("Already offset this snapshot.")
            return snap
    ## rotate the coordinates of the spherical extraction
    new_rs = rotateVectorsZY(thetay,thetaz,snap['Coordinates']-scom)
    new_vs = rotateVectorsZY(thetay,thetaz,snap['Velocities']-vscom)

    ## add positions and velocities separately, since they need to be rotated and indexed
    snap['Coordinates'] = new_rs
    snap['Velocities'] = new_vs

    ## store com information in case its relevant
    add_to_dict = {
        'thetay':thetay,'thetaz':thetaz,
        'overwritten':1,
        'orient_stars':orient_stars}

    ## if the snap is already overwritten this info is in there
    if 'scom' not in snap:
        add_to_dict['scom']=scom
    if 'vscom' not in snap:
        add_to_dict['vscom']=vscom

    ## add relevant keys
    snap.update(add_to_dict)

    return snap

def unrotateSnapshots(
    snaps,
    thetay,
    thetaz):
    ## let's put it back the way we found it
    for this_snap in snaps:
        #snaps[0]['scom'] + 
        this_snap['Coordinates'] = unrotateVectorsZY(
            thetay,thetaz,
            this_snap['Coordinates'])

        #snaps[0]['vscom'] + 
        this_snap['Velocities'] = unrotateVectorsZY(
            thetay,thetaz,
            this_snap['Velocities'])


def diskFilterDictionary(
    star_snap,
    snap,
    radius,
    orient_radius,
    cylinder='',
    scom=None,
    dark_snap=None,
    orient_stars=0,
    rect_buffer=1.1,
    ):
    """ Takes two openSnapshot dictionaries and returns a filtered subset of the particles
        that are in the disk, with positions and velocities rotated"""

    ## make sure someone didn't pass no stars but 
    ##  ask us to orient the disk about the stars
    if star_snap is None and orient_stars:
        raise ValueError("Can't orient on what isn't provided...")

    ## collect the snapshots we were passed
    snaps = ([snap] + 
        [star_snap]*(star_snap is not None)+
        [dark_snap]*(dark_snap is not None))
    indicess = [[] for i in range(len(snaps))]

    ## check if we've already worked on this snapshot
    if 'thetay' in snap:
        scom = vscom = np.zeros(3)
        old_thetay,old_thetaz = snap['thetay'],snap['thetaz']
        unrotateSnapshots(snaps,old_thetay,old_thetaz) 
        ## remove the old theta to represent that it has been 
        ##  derotated
        for this_snap in snaps:
            this_snap.pop('thetay')
            this_snap.pop('thetaz')
            this_snap.pop('overwritten')

    print("Reorienting...",)
    (thetay,thetaz,
    scom,vscom,
    orient_gindices,
    orient_sindices)=extractDiskFromSnapdict(
        star_snap,
        snap,
        orient_radius,
        scom=scom,
        orient_stars=orient_stars)
    print("Done.")

    for this_snap in snaps:
        ## overwrites the coordinates in the snapshot
        offsetRotateSnapshot(
            this_snap,
            scom,vscom,
            thetay,thetaz,
            orient_stars)

    ## dictionary to add to extracted snapshot
    add_to_dict = {
        'scale_radius':radius,
        'rect_buffer':rect_buffer}

    #overwrite gindices/sindices/dindices to get a square instead of a disk
    if cylinder != '':
        if cylinder is None:
            ## find the gas scale height through a constant
            ##  approximation
            cylinder = findContainedScaleHeight(
                snap['Coordinates'][:,2][gindices],
                snap['Masses'][gindices])

        ## replace th erelevant indices for each snapshot
        for i,this_snap in enumerate(snaps):
            indicess[i] = extractRectangularVolumeIndices(
                this_snap['Coordinates'],
                np.zeros(3),radius*rect_buffer,20)#radius*rect_buffer) 

        ## add the scale height to the snapshot
        add_to_dict.update({'scale_height':cylinder})
    else:
        for i,this_snap in enumerate(snaps):
            indicess[i] = extractSphericalVolumeIndices(
                this_snap['Coordinates'],
                np.zeros(3),radius**2)#radius*rect_buffer) 

    sub_snaps = []
    ## create the volume filtered dictionaries, snapshots are already rotated/offset
    for i,this_snap in enumerate(snaps):
        sub_snaps+=[filterDictionary(this_snap,indicess[i])]
        sub_snaps[i].update(add_to_dict)

    ## calculate the diskiness --  but only in the quantity that 
    ##  we actually aligned the galaxy with respect to, for consistency
    if orient_stars:
        ## calculate stars "diskiness," alias new_star_snap
        my_snap = snaps[1]
        my_sub_snap = sub_snaps[1]
        key_add = "star_"
        orient_indices = orient_sindices
    else:
        my_snap = snaps[0]
        my_sub_snap = sub_snaps[0]
        key_add = ""
        orient_indices = orient_gindices

    angMom = getAngularMomentum(
        my_snap['Coordinates'][orient_indices],
        my_snap['Masses'][orient_indices]*1e10,
        my_snap['Velocities'][orient_indices])# msun - kpc - km/s units of L
        ## post-rotation, lz == ltot by definition, lx, ly = 0 
    lz = angMom[2]

        ## add up ltot as sum(|l_i|), doesn't cancel counter-rotating stuff
    ltot = getAngularMomentumSquared(
        my_snap['Coordinates'][orient_indices],
        my_snap['Masses'][orient_indices]*1e10,
        my_snap['Velocities'][orient_indices])**0.5 # msun - kpc - km/s units of L

    my_sub_snap[key_add+'lz']=lz
    my_sub_snap[key_add+'ltot']=ltot

    ## figure out what we're returning
    ## the extracted snapshots, obviously

    return sub_snaps
