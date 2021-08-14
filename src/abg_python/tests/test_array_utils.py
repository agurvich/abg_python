from ..array_utils import *

def test_findArrayClosestIndices():
    ret = findArrayClosestIndices([1,2,3,4],[1,2,3,4,5])
    assert np.all(ret == [0,1,2,3]), ret

def test_findIntersection():
    ## define a line
    xs = np.linspace(-2,2,10000)
    ys = 0.5*xs-0.5
    ## define a perpendicular line
    ys1 = -0.5*xs+0.5

    ## they should cross at 1,0
    x,y = findIntersection(xs,ys,ys1)

    assert np.isclose(np.round(x,3),1), x
    assert np.isclose(np.round(y,3),0), y

def test_getFWHM():
    
    xs = np.linspace(-4,4,10000)
    ys = 1/np.sqrt(2*np.pi)*np.exp(-xs**2/2)
    ## FWHM = 2.355 sigma, here sigma = 1
    fwhm,_ = getFWHM(xs,ys)
    assert np.isclose(np.round(fwhm,3),2.355), fwhm

def test_pairReplace():
    xs = np.ones(10)
    ys = np.ones(10)

    xs[1] = np.nan
    ys[-2] = np.nan

    xs,ys = pairReplace(xs,ys,0,np.isnan)

    assert xs[1] == 0, xs[1]
    assert ys[-2] == 0, ys[2]

def test_substep():
    """linearly interpolates between the values in array arr using N steps"""

    arr = np.arange(0,12,3)
    arr = substep(arr,3)

    assert np.all(arr==np.arange(0,10)),arr

def test_manyFilter():
    
    arrs = np.arange(30)
    arrs.shape = (-1,10)


    xs,ys,zs = manyFilter(lambda x: np.mod(x,2),*arrs)

    assert np.all(xs == np.arange(1,10,2)), xs
    assert np.all(ys == np.arange(11,20,2)), ys
    assert np.all(zs == np.arange(21,30,2)), zs


def test_pairFilter():
    """filters both x and y corresponding pairs by
        bool_fn"""

    arrs = np.arange(20)
    arrs.shape = (-1,10)

    xs,ys = pairFilter(arrs[0],arrs[1],lambda x: np.mod(x,2))

    assert np.all(xs == np.arange(1,10,2)), xs
    assert np.all(ys == np.arange(11,20,2)), ys


def test_filterDictionary():

    test_dict = {
        "const":1,
        "arr1":np.arange(10),
        "arr2":np.arange(10),
        "consts":np.arange(3)
    }

    new_dict = filterDictionary(test_dict,np.mod(np.arange(10),2).astype(bool))

    assert new_dict['const'] == 1, new_dict['const']
    assert np.all(new_dict['consts'] == [0,1,2]), new_dict['consts']
    assert np.all(new_dict['arr1'] == np.arange(1,10,2)), new_dict['arr1']
    assert np.all(new_dict['arr2'] == np.arange(1,10,2)), new_dict['arr2']
