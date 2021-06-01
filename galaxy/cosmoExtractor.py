import sys,getopt,os
import numpy as np

from abg_python.all_utils import filterDictionary,rotateEuler,getVcom,getAngularMomentum,getAngularMomentumSquared,extractSphericalVolumeIndices
from abg_python.snapshot_utils import openSnapshot

## angle calculation and rotation
def getThetas(angMom):
    """ not sure which rotation matrix this corresponds to... 
        I guess it's yzx based on name of variables """

    thetay = np.arctan2(np.sqrt(angMom[0]**2+angMom[1]**2),(angMom[2]))
    thetaz = np.arctan2(angMom[1],angMom[0])
    #print("want to rotate by",thetay*180/np.pi,thetaz*180/np.pi)
    ## RETURNS RADIANS
    return thetay,thetaz

def getThetasTaitBryan(angMom):
    """ as Euler angles but xyz vs. zxz"""
    theta_TB = np.arctan2(angMom[1],np.sqrt(angMom[0]**2+angMom[2]**2))*180/np.pi
    phi_TB = np.arctan2(-angMom[0],angMom[2])*180/np.pi

    #new_angMom = rotateEuler(
        #theta_TB,phi_TB,0,
        #angMom,
        #order='xyz',loud=False)
    #print('old:',angMom,'new:',new_angMom)

    ## RETURNS DEGREES
    return theta_TB,phi_TB

def offsetRotateSnapshot(snap,scom,vscom,theta_TB,phi_TB,orient_stars):

    if 'overwritten' in snap and snap['overwritten']:
        if (snap['theta_TB'] == theta_TB and 
            snap['phi_TB'] == phi_TB and
            np.all(snap['scom'] == scom) and
            np.all(snap['vscom'] == vscom)):
            print("Already offset this snapshot.")
            return snap

    ## rotate the coordinates of the spherical extraction
    new_rs = rotateEuler(
        theta_TB,phi_TB,0,
        snap['Coordinates']-scom,
        order='xyz',loud=False)
    new_vs = rotateEuler(
        theta_TB,phi_TB,0,
        snap['Velocities']-vscom,
        order='xyz',loud=False)

    ## add positions and velocities separately, since they need to be rotated and indexed
    snap['Coordinates'] = new_rs
    snap['Velocities'] = new_vs

    ## store com information in case its relevant
    add_to_dict = {
        'theta_TB':theta_TB,
        'phi_TB':phi_TB,
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
    theta_TB,
    phi_TB):

    ## let's put it back the way we found it 
    for this_snap in snaps:
        this_snap['Coordinates'] = rotateEuler(
            theta_TB,phi_TB,0,
            this_snap['Coordinates'],
            order='xyz',inverse=True,
            loud=False)

        this_snap['Velocities'] = rotateEuler(
            theta_TB,phi_TB,0,
            this_snap['Velocities'],
            order='xyz',inverse=True,
            loud=False)


##### main extraction protocols
def extractDiskFromSnapdicts(
    star_snap,
    snap,
    radius,
    orient_radius,
    scom=None,
    dark_snap=None,
    orient_stars=0,
    force_theta_TB=None,
    force_phi_TB=None,
    ):
    """ Takes two openSnapshot dictionaries and returns a filtered subset of the particles
        that are in the disk, with positions and velocities rotated"""

    ## collect the snapshots we were passed
    snaps = ([snap] + 
        [star_snap]*(star_snap is not None)+
        [dark_snap]*(dark_snap is not None))
    indicess = [[] for i in range(len(snaps))]

    ## check if we've already worked on this snapshot
    if 'theta_TB' in snap:
        scom = vscom = np.zeros(3)
        old_theta_TB,old_phi_TB = snap['theta_TB'],snap['phi_TB']
        unrotateSnapshots(snaps,old_theta_TB,old_phi_TB) 
        ## remove the old theta to represent that it has been 
        ##  derotated
        for this_snap in snaps:
            this_snap.pop('theta_TB')
            this_snap.pop('phi_TB')
            this_snap.pop('overwritten')

    print("Reorienting...",)
    theta_TB,phi_TB,vscom=orientDiskFromSnapdicts(
        star_snap,
        snap,
        orient_radius,
        scom=scom,
        orient_stars=orient_stars,
        force_theta_TB=force_theta_TB,
        force_phi_TB=force_phi_TB)
    print("Done.")

    for this_snap in snaps:
        ## overwrites the coordinates in the snapshot
        offsetRotateSnapshot(
            this_snap,
            scom,vscom,
            theta_TB,phi_TB,
            orient_stars)

    ## dictionary to add to extracted snapshot
    add_to_dict = {
        'scale_radius':np.round(radius,5)}

    #overwrite gindices/sindices/dindices
    for i,this_snap in enumerate(snaps):
        indicess[i] = extractSphericalVolumeIndices(
            this_snap['Coordinates'],
            np.zeros(3),radius) 

    sub_snaps = []
    ## create the volume filtered dictionaries, snapshots are already rotated/offset
    for i,this_snap in enumerate(snaps):
        sub_snaps+=[filterDictionary(this_snap,indicess[i])]
        sub_snaps[i].update(add_to_dict)

    return sub_snaps

def orientDiskFromSnapdicts(
    star_snap,
    snap,
    radius,
    scom,
    orient_stars=0,
    force_theta_TB=None,
    force_phi_TB=None):
    """ Takes arrays from a snapshot and returns orientation.
        Input: 
            srs/svs/smasses - positions,velocities, and masses of star particles
            rs - positions and densities of gas particles 
            radius - radius to extract particles from
    """

    if orient_stars:
        if star_snap is None:
            raise ValueError("Can't orient on stars if stars are not passed")

        these_rs = star_snap['Coordinates']
        these_vs = star_snap['Velocities']
        these_masses = star_snap['Masses']
    else:
        if snap is None:
            raise ValueError("Can't orient on gas if gas is not passed")

        these_rs = snap['Coordinates']
        these_vs = snap['Velocities']
        these_masses = snap['Masses']

    mask = extractSphericalVolumeIndices(
            these_rs,scom,radius)

    if not np.sum(mask):
        print(scom,radius)
        raise ValueError("No particles to orient the disk on!")

    ## find com velocity of disk
    vscom = getVcom(these_masses[mask],these_vs[mask])

    ## find angular momentum vector
    angMom = getAngularMomentum(
        these_rs[mask]-scom,
        these_masses[mask],
        these_vs[mask]-vscom)

    if force_theta_TB is None or force_phi_TB is None:
        ## find angles necessary to rotate coordinates to align with angMom
        theta_TB, phi_TB = getThetasTaitBryan(angMom)
    else:
        theta_TB, phi_TB = force_theta_TB, force_phi_TB

    return theta_TB,phi_TB,vscom
