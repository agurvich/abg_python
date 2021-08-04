import itertools
import ctypes
from multiprocessing import Pool, Array
import os
import time
import numpy as np
import psutil

n, m = 10**2,10**3

def printme(this_i,this_j):
    counter=0
    for i in range(10**3):
        counter+=1
    #mp_arr[this_i,this_j]=this_i*m+this_j
    return counter
    
dtype = ctypes.c_float

## allocate memory
#mp_arr = Array(dtype, n*m,lock=False) # shared, can be used from multiple processes
#mp_arr = np.frombuffer(mp_arr,dtype=dtype) # mp_arr and arr share the same memory
# make it two-dimensional
#mp_arr = mp_arr.reshape((n,m)) # b and arr share the same memory

#fork 
pool = Pool(6)
pixis,pixjs = np.meshgrid(np.arange(n,dtype=int),np.arange(m,dtype=int))
pixis,pixjs=pixis.flatten(),pixjs.flatten()

# can set data after fork
argss = zip(
    pixis,
    pixjs,)

init_time = time.time()
pool.starmap(printme,argss)
pool.close()
pool.join()
print(time.time()-init_time,'s elapsed mp')

argss = zip(
    pixis,
    pixjs,)
init_time = time.time()
for args in argss:
    printme(*args)
print(time.time()-init_time,'s elapsed')


#print(np.all((mp_arr-np.arange(n*m).reshape(n,m))==0))
#del mp_arr
