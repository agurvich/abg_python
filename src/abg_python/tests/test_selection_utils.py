from abg_python.selection_utils import cylindricalVolumeMask, rectangularVolumeMask, sphericalVolumeMask
import pytest
import numpy as np

@pytest.mark.parametrize('radius', [1,3,5])
@pytest.mark.parametrize('height', [None, 2, 3])
@pytest.mark.parametrize('rcom', [None, [0,0,3], [0,0,10],[0,0,-10]])
def test_rectangularVolumeMask(radius,height,rcom):

    coords = np.empty(33)
    coords.shape = (-1,3)
    coords[:,0] = 0
    coords[:,1] = 0
    coords[:,2] = np.arange(-5,6)

    mask = rectangularVolumeMask(coords,radius,height,rcom)
    if height is None: height = radius
    if rcom is None: num_expected = 2*height+1
    else: 
        coords[:,2]-=rcom[-1]
        num_expected = np.sum(coords[:,2]**2<=height**2)
    assert np.sum(mask) == num_expected,(np.sum(mask),num_expected,radius,height,rcom)

@pytest.mark.parametrize('radius', [1,3,5])
@pytest.mark.parametrize('height', [None, 2, 3])
@pytest.mark.parametrize('rcom', [None, [0,0,2],[0,0,-2]])
def test_cylindricalVolumeMask(radius,height,rcom):

    coords = np.empty(33)
    coords.shape = (-1,3)
    coords[:,0] = np.arange(-5,6)
    coords[:,1] = 0
    coords[:,2] = np.arange(-5,6)

    mask = cylindricalVolumeMask(coords,radius,height,rcom)
    if height is None: height = radius
    if rcom is None: num_expected = min(2*height+1,2*radius+1)
    else: 
        coords -= rcom
        xy_mask = np.sum(coords[:,:2]**2,axis=1) <= radius**2
        z_mask = coords[:,2]**2 <= height**2
        num_expected = np.sum(xy_mask*z_mask)
    assert np.sum(mask) == num_expected,(
        "act. %d"%np.sum(mask),
        "exp. %d"%num_expected,
        "rad. %d"%radius,
        "ht. %d"%height,
        "com %s"%str(rcom))


@pytest.mark.parametrize('radius', [1,3,5])
@pytest.mark.parametrize('rcom', [None, [0,0,2],[0,0,-2]])
def test_sphericalVolumeMask(radius,rcom):

    coords = np.empty(33)
    coords.shape = (-1,3)
    coords[:,0] = np.arange(-5,6)
    coords[:,1] = 0
    coords[:,2] = 0

    mask = sphericalVolumeMask(coords,radius,rcom)

    if rcom is None: num_expected = 2*radius+1
    ## literally the same line :\
    else: num_expected = np.sum(np.sum((coords-rcom)**2,axis=1) <= radius**2)

    assert np.sum(mask) == num_expected,(
        "act. %d"%np.sum(mask),
        "exp. %d"%num_expected,
        "rad. %d"%radius,
        "com %s"%str(rcom))