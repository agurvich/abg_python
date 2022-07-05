from io import StringIO

import sys
import os

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

def suppressSTDOUTToFile(fn,args,fname=None,mode='a+',loud=False,**kwargs):
    """Hides the printed output of a python function to remove clutter, but
        still saves it to a file for later inspection. 
        Input: 
            fn - The function you want to call 
            args - A dictionary with keyword arguments for the function
            fname - The path to the output text file you want to pipe to. 
            mode - The file open mode you want to use, defaults to a+ to append
                to the same debug/output file but you might want w+ to replace it
                every time. 
            loud - Prints a warning message that the STDOUT is being suppressed
        Output: 
            ret - The return value of fn(**args)
    """
    
    orgstdout=sys.stdout
    ret=-1
    try:
        handle=StringIO()
        if loud:
            print('Warning! Supressing std.out...')
        sys.stdout=handle

        ret=fn(*args,**kwargs)

        ## by default just suppress STDOUT, but we'll 
        ##  let you choose a file if you'd like
        if fname is not None:
            with open(fname,mode) as fhandle:
                fhandle.write(handle.getvalue())
    finally:
        sys.stdout=orgstdout
        if loud:
            print('Warning! Unsupressing std.out...')

    return ret

def getfinsnapnum(
    snapdir,
    getmin=0,
    fname_to_match='snapshot_',
    dir_to_match='snapdir_'):

    if not getmin:
        maxnum = -1e8
        for snap in os.listdir(snapdir):
            if fname_to_match in snap and snap.index(fname_to_match)==0 and '.hdf5' in snap:
                snapnum = int(snap[len(fname_to_match):-len('.hdf5')])
                if snapnum > maxnum:
                    maxnum=snapnum
            elif dir_to_match in snap and snap.index(dir_to_match)==0:
                snapnum = int(snap[len(dir_to_match):])
                if snapnum > maxnum:
                    maxnum=snapnum
        return maxnum
    else:
        minnum=1e8
        for snap in os.listdir(snapdir):
            if fname_to_match in snap and snap.index(fname_to_match)==0 and '.hdf5' in snap:
                snapnum = int(snap[len(fname_to_match):-len('.hdf5')])
                if snapnum < minnum:
                    minnum=snapnum
            elif dir_to_match in snap and snap.index(dir_to_match)==0:
                snapnum = int(snap[len(dir_to_match):])
                if snapnum < minnum:
                    minnum=snapnum
        return minnum

# Print iterations progress
def printProgressBar (iteration, total, prefix = 'Progress:', suffix = 'complete', decimals = 1, length = 50, fill = '█', printEnd = "\r",force=False):
    """
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters?noredirect=1&lq=1
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    n_iter_update = int(total//(100*10**(decimals))) ## update every 0.1% for decimals = 1

    if force or n_iter_update <= 1 or not (iteration % n_iter_update): 
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
