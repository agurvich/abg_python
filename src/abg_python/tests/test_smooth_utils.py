import pytest

from ..smooth_utils import *


@pytest.mark.parametrize('average', [True,False])
@pytest.mark.parametrize('assign', ['right','left','center'])
def test_boxcar_average(average,assign):

    edges = np.linspace(-1,1,21,endpoint=True)
    ys = np.ones(edges.size-1)

    _, smooth_ys = boxcar_average(
        edges,ys,
        boxcar_width=0.3, ## 3 points per boxcar
        average=average,
        assign=assign,
        loud=False)

    npoints_per_car = 3

    if average: smooth_ys*=npoints_per_car

    if assign == 'right':
        ## should be missing a window at the beginning
        assert np.all(np.isnan(smooth_ys[:2])),smooth_ys
    elif assign == 'left':
        ## should be missing a window at the end
        assert np.all(np.isnan(smooth_ys[-2:])),smooth_ys
    elif assign == 'center':
        ## should be missing half a window on either side
        assert np.isnan(smooth_ys[0]),smooth_ys
        assert np.isnan(smooth_ys[-1]),smooth_ys
    else: raise ValueError("Bad assign %s"%assign)

    assert np.all(smooth_ys[np.isfinite(smooth_ys)] == npoints_per_car)

def test_uniform_resampler():

    xs = [0,1,1.1,1.2,1.3,5]
    ys = xs

    uni_xs,[uni_ys,uni_ys2] = uniform_resampler(xs,[ys,ys])

    ## test respacing of xs
    assert np.all(uni_xs == np.arange(6)),uni_xs

    ## test packing/unpacking of arrays and interpolation
    assert np.all(uni_ys == np.arange(6)),uni_ys
    assert np.all(uni_ys2 == np.arange(6)),uni_ys2
    
def test_smooth_x_varying_curve():
    #xs,ys,smooth,log=False,assign='center'
    pass

@pytest.mark.parametrize('last', [False,True])
def test_find_first_window(last):
    #xs,ys,bool_fn,window
    xs = np.arange(0,360)
    ys = np.sin(2*xs/180*np.pi)
    bool_fn = lambda x,y: y <= 0 

    lx,rx = find_first_window(xs,ys,bool_fn,xs[10]-xs[0],last=last)
    if not last: assert (lx == 89 and rx == 99),(lx,rx)
    else: assert (lx ==  269 and rx == 279),(lx,rx) 

def test_find_last_instance():
    #xs,ys,bool_fn
    xs = np.arange(10)
    ys = np.mod(xs,3)
    loc,value = find_last_instance(xs,ys,lambda x,y: y ==0)
    assert  loc == 9 and value == 0, (loc,value)

@pytest.mark.parametrize('smooth', [None,10])
def test_find_local_minima_maxima(smooth):
    #xs,ys,smooth=None,ax=None
    xs = np.arange(0,360)
    ys = np.sin(2*xs/180*np.pi)

    zeros = find_local_minima_maxima(xs,ys,smooth)

    assert len(zeros) == 4, len(zeros)
    assert zeros[0] == 45,zeros#zeros[0]
    assert zeros[1] == 45+90,zeros#zeros[1]
    assert zeros[2] == 45+180,zeros[2]
    assert zeros[3] == 45+270,zeros[3]
