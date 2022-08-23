import sys
import multiprocessing
try:
    from multiprocessing import shared_memory
except ImportError:
    ## check MP version
    version = sys.version_info[:2]
    version = float("%d.%d"%version)
    if version < 3.8:
        print("Upgrade to Python 3.8 to use multiprocessing with shared memory.")


import numpy as np
import os


def copyNumpyArrayToMPSharedMemory(
    input_arr,
    finally_flag=False,
    loud=False):
    """ must clean up anything that contains a reference to a shared
        memory object. globals() must be purged before the shm_buffers
        are unlinked or python will crash."""
 
    ## check MP version
    version = sys.version_info[:2]
    version = float("%d.%d"%version)
    if version < 3.8:
        raise OSError("Upgrade to Python 3.8 to use multiprocessing with shared memory.")

    if not finally_flag:
        raise BufferError(
            "Set finally_flag=True to confirm "+
            "that you understand the risks associated with shared memory "+
            "and are prepared to unlink the returned buffer when you're done. "+
            "In a try except finally clause! With great power comes great responsibility.")


    shm = shared_memory.SharedMemory(create=True, size=input_arr.nbytes)
    # Now create a NumPy array backed by shared memory
    shm_arr = np.ndarray(input_arr.shape, dtype=input_arr.dtype, buffer=shm.buf)
    shm_arr[:] = input_arr[:]  # Copy the original data into shared memory
    if loud: print('Copied an array to the buffer.')
    del input_arr
    return shm,shm_arr

def copySnapshotNamesToMPSharedMemory(
    arr_names,
    snapdict,
    **kwargs):

    this_snapdict = {}
    shm_buffers = []
    for arr_name in arr_names:
        if arr_name in snapdict:
            shm_buffer,shm_arr = copyNumpyArrayToMPSharedMemory(snapdict[arr_name],**kwargs)
        
        ## track these shared memory buffers so they can be cleaned
        ##  up later.
        shm_buffers.append(shm_buffer)
        this_snapdict[arr_name] = shm_arr

    del snapdict,arr_names
    return this_snapdict,shm_buffers
