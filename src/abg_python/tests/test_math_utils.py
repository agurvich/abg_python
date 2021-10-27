from ..math_utils import *
import pytest

#math functions (trig and linear algebra...)
@pytest.mark.parametrize('vector,answer', list(zip(
    [[0,1,0],
    [0,-1,0],
    [1,0,0],
    [-1,0,0],
    [0,0,1],
    [0,0,-1]] ,
    [(90,0),
    (-90,0),
    (0,0),
    (180,0),
    (0,90),
    (0,-90)]
)))
def test_vectorsToRAAndDec(vector,answer):
    ## puts the meridian at x = 0
    ## puts the equator at z = 0
    ra,dec = vectorsToRAAndDec(vector)
    ra/=np.pi/180
    dec/=np.pi/180

    assert ra == answer[0] and dec == answer[1]

@pytest.mark.parametrize('angMom,answer', list(zip(
    [[0,1,0],
    [0,-1,0],
    [1,0,0],
    [-1,0,0],
    [0,0,1],
    [0,0,-1]] ,
    [(90,0),
    (-90,0),
    (0,-90), 
    (0,90), 
    (0,0),
    (0,180)] ## 180, 0 is also a valid answer but has a phase spin
)))
def test_getThetasTaitBryan(angMom,answer):
    theta_TB, phi_TB = getThetasTaitBryan(angMom)
    assert (theta_TB == answer[0] and phi_TB == answer[1]),(
        theta_TB,phi_TB,answer)

@pytest.mark.parametrize('theta',[0,90])
@pytest.mark.parametrize('phi',[0,90])
@pytest.mark.parametrize('psi',[0,90])
@pytest.mark.parametrize('pos',np.array([[0,0,1],[0,1,0],[1,0,0]]))
@pytest.mark.parametrize('order',['xyz','zxz'])
@pytest.mark.parametrize('rotation_point',np.array([[0,0,0],[1,1,1]]))
def test_rotateEuler_norm(
    theta,phi,psi,
    pos,
    order, 
    rotation_point):

    rot_pos = rotateEuler(theta,phi,psi,pos,order,rotation_point,loud=False)

    og_norm = np.linalg.norm(pos)
    rot_norm = np.linalg.norm(rot_pos)

    ## first check that the norm is preserved
    assert np.isclose(og_norm,rot_norm),(og_norm,rot_norm)

    ## check that the inverse works
    inv_rot_pos = rotateEuler(
        theta,phi,psi,
        rot_pos,order,
        rotation_point,
        loud=False,
        inverse=True)

    assert np.all(np.isclose(pos,inv_rot_pos))

@pytest.mark.parametrize('theta',[0,90])
@pytest.mark.parametrize('phi',[0,90])
@pytest.mark.parametrize('psi',[0,90])
def test_rotateEuler_dir_xyz(
    theta,phi,psi):

    rot_pos = rotateEuler(theta,phi,psi,[0,0,1],loud=False)

    ## psi actually doesn't matter since we're using z-axis
    if phi == 90: assert np.all(np.isclose(rot_pos , [1,0,0]))
    elif theta == 90: assert np.all(np.isclose(rot_pos , [0,-1,0]))
    else: assert np.all(np.isclose(rot_pos , [0,0,1])) 