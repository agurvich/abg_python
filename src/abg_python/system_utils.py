from io import StringIO

import sys

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
