#!/bin/bash

echo "mpiexec -n 1:" `mpiexec -n 1 python parallel_read.py`
echo "mpiexec -n 2:" `mpiexec -n 2 python parallel_read.py` 
echo "mpiexec -n 4:" `mpiexec -n 4 python parallel_read.py` 
echo "mpiexec -n 8:" `mpiexec -n 8 python parallel_read.py`
echo "mpiexec -n 16:" `mpiexec -n 16 python parallel_read.py`
