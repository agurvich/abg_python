import os
import sys

def add_directory_tree(datadir):
    """This function probably already exists lmfao..."""
    if not os.path.isdir(datadir):
        directories=datadir.split('/')
        directories_to_make=[]
        for i in range(len(directories)):
            trialdir='/'.join(directories[:-i])
            if os.path.isdir(trialdir):
                i-=1
                break
        for j in range(i):
            toadd='/'.join(directories[:-j-1])
            directories_to_make+=[toadd]
        directories_to_make+=[datadir]
        for directory_to_make in directories_to_make:
            os.mkdir(directory_to_make)

def getfinsnapnum(snapdir,getmin=0):
    if not getmin:
        maxnum = 0
        for snap in os.listdir(snapdir):
            if 'snapshot' in snap and 'hdf5' in snap and snap.index('snapshot')==0:
                snapnum = int(snap[len('snapshot_'):-len('.hdf5')])
                if snapnum > maxnum:
                    maxnum=snapnum
            elif 'snapdir' in snap:
                snapnum = int(snap[len('snapdir_'):])
                if snapnum > maxnum:
                    maxnum=snapnum
        return maxnum
    else:
        minnum=1e8
        for snap in os.listdir(snapdir):
            if 'snapshot' in snap and 'hdf5' in snap:
                snapnum = int(snap[len('snapshot_'):-len('.hdf5')])
                if snapnum < minnum:
                    minnum=snapnum
            elif 'snapdir' in snap:
                snapnum = int(snap[len('snapdir_'):])
                if snapnum < minnum:
                    minnum=snapnum
        return minnum

def extractMaxTime(snapdir):
    """Extracts the time variable from the final snapshot"""
    maxsnapnum = getfinsnapnum(snapdir)
    if 'snapshot_%3d.hdf5'%maxsnapnum in os.listdir(snapdir):
        h5path = 'snapshot_%3d.hdf5'%maxsnapnum
    elif 'snapdir_%03d'%maxsnapnum in os.listdir(snapdir):
        h5path = "snapdir_%03d/snapshot_%03d.0.hdf5"%(maxsnapnum,maxsnapnum)
    else:
        print("Couldn't find maxsnapnum in")
        print(os.listdir(snapdir))
        raise Exception("Couldn't find snapshot")

    with h5py.File(os.path.join(snapdir,h5path),'r') as handle:
        maxtime = handle['Header'].attrs['Time']
    return maxtime

def suppressSTDOUT(fn,args,debug=1):
    """Hides the printed output of a python function to remove clutter. 
        Input: 
            fn - The function you want to call 
            args - A dictionary with keyword arguments for the function
            debug - Prints a warning message that the STDOUT is being suppressed
        Output: 
            ret - The return value of fn(**args)
    """
    orgstdout=sys.stdout
    ret=-1
    try:
        handle=StringIO.StringIO()
        if debug:
            print('Warning! Supressing std.out...')
        sys.stdout=handle

        ret=fn(**args)

    finally:
        sys.stdout=orgstdout
        if debug:
            print('Warning! Unsupressing std.out...')

    return ret

def get_size(obj, seen=None):
    """Recursively finds size of objects
        https://goshippo.com/blog/measure-real-size-any-python-object/
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def suppressSTDOUTToFile(fn,args,fname,mode='a+',debug=1):
    """Hides the printed output of a python function to remove clutter, but
        still saves it to a file for later inspection. 
        Input: 
            fn - The function you want to call 
            args - A dictionary with keyword arguments for the function
            fname - The path to the output text file you want to pipe to. 
            mode - The file open mode you want to use, defaults to a+ to append
                to the same debug/output file but you might want w+ to replace it
                every time. 
            debug - Prints a warning message that the STDOUT is being suppressed
        Output: 
            ret - The return value of fn(**args)
    """
    
    orgstdout=sys.stdout
    ret=-1
    try:
        handle=StringIO.StringIO()
        if debug:
            print('Warning! Supressing std.out...')
        sys.stdout=handle

        ret=fn(**args)

        with open(fname,mode) as fhandle:
            fhandle.write(handle.getvalue())
    finally:
        sys.stdout=orgstdout
        if debug:
            print('Warning! Unsupressing std.out...')

    return ret


