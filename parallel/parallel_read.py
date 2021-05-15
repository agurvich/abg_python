import h5py
from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.rank  # The process ID (integer 0-3 for 4-process run)
size = comm.size

def MPI_open(comm,handle,key,nparts):

    ## figure out how many particles each MPI task is supposed to read
    num_per_task = int(nparts//comm.size)+1

    ## go ahead and read the data from the handle
    sendbuf = handle[key][rank*num_per_task : (rank+1)*num_per_task].astype(np.float64)

    recvbuf = None
    if rank == 0:
        ## initialize the  receiving memory buffer
        recvbuf = np.empty(
            [comm.size]+list(sendbuf.shape),
            dtype=sendbuf.dtype)

    ## gather the data
    comm.Gather(sendbuf, recvbuf, root=0)

    if rank == 0:
        recvbuf = np.concatenate(recvbuf,axis=0)[:nparts]

    return recvbuf


## to test functionality use fname = 'foo.hdf5'
fname = 'foo.hdf5'
fname = '/scratch/projects/xsede/GalaxiesOnFIRE/metal_diffusion/m12i_res7100/output/snapdir_600/snapshot_600.0.hdf5'

## how many particles to read
nparts = 21

if rank == 0:
    if fname == 'foo.hdf5':
        ## initialize a dummy file just for this test
        with h5py.File(fname, 'w') as f:
            group = f.create_group("PartType0")
            dset = group.create_dataset('Coordinates', (nparts,3), dtype='i')
            dset[:] = np.arange(nparts*3).reshape(-1,3)

init_time = time.time()
## open the hdf5 file with the mpio driver for parallel read (?)
with h5py.File(fname, 'r', driver='mpio', comm=comm) as handle:

    if 'Header' in handle.keys():
        nparts = int(handle['Header'].attrs['NumPart_ThisFile'][0])
    ## read the coordinates from the handle
    coords = MPI_open(comm,handle['PartType0'],'Coordinates',nparts)


## clean up child MPI tasks
if rank != 0:
    exit()

print(time.time()-init_time,'s elapsed reading %d particles'%nparts)
