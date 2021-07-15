def findArrayClosestIndices(xs,ys):
    try:
        assert len(xs) < len(ys)
    except:
        raise ValueError(
            "Ys (%d)"%len(ys),
            "should be some large sample that",
            "Xs (%d) is subsampling!"%len(xs))

    dists = cdist(
        xs.reshape(-1,1),
        ys.reshape(-1,1))

    indices = np.argmin(dists,axis=1)
    return indices

def findIntersection(xs,ys,ys1):
    argmin = np.argmin((ys-ys1)**2)
    return xs[argmin],ys[argmin]

def getFWHM(xs,ys):
    argmax = np.argmax(ys)
    xl,yl = findIntersection(xs[:argmax],(ys/np.max(ys))[:argmax],0.5)
    xr,yr = findIntersection(xs[argmax:],(ys/np.max(ys))[argmax:],0.5)
    return (xr-xl),(xl,xr,yl,yr)

def substep(arr,N):
    """linearly interpolates between the values in array arr using N steps"""
    my_arr = np.array([])
    for lx,rx in zip(arr[:-1],arr[1:]):
        my_arr=np.append(my_arr,np.linspace(lx,rx,N+1)[:-1])
        
    ## excluded the right end, need to include the final right end
    my_arr = np.append(my_arr,rx)
    return my_arr

def manyFilter(bool_fn,*args):
    """filters an arbitrary number of arrays in 
        corresponding tuples by bool_fn"""
    mask = np.ones(args[0].size)

    for arg in args:
        mask = np.logical_and(bool_fn(arg),mask)

    return [arg[mask] for arg in args]

def pairReplace(xs,ys,value,bool_fn):
    """filters both x and y corresponding pairs by
        bool_fn"""

    xs,ys = copy.copy(xs),copy.copy(ys)

    xs[bool_fn(ys)] = value
    ys[bool_fn(ys)] = value

    xs[bool_fn(xs)] = value
    ys[bool_fn(xs)] = value

    return xs,ys

def pairFilter(xs,ys,bool_fn):
    """filters both x and y corresponding pairs by
        bool_fn"""

    new_xs = xs[bool_fn(ys)]
    new_ys = ys[bool_fn(ys)]

    new_ys = new_ys[bool_fn(new_xs)]
    new_xs = new_xs[bool_fn(new_xs)]
    return new_xs,new_ys

def filterDictionary(dict0,indices,dict1 = None,key_exceptions=[],free_mem = 0):
    if dict1 is None:
        dict1={}
    for key in dict0:
        if key in key_exceptions:
            continue
        try:
            if np.shape(dict0[key])[0]==indices.shape[0]:
                dict1[key]=dict0[key][indices]
            ## should only be center of mass and center of mass velocity
            else:
                raise Exception("Save this array verbatim")
        except:
            dict1[key]=dict0[key]
    if free_mem:
        del dict0
    return dict1
