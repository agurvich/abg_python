import os

from ..system_utils import *

def test_get_size():

    ## bigger lists should be bigger than smaller lists
    assert get_size(range(100)) > get_size(range(10))
    ## lists should be bigger than ints
    assert get_size([10]) > get_size(10)
    ## dicts should be bigger than lists
    assert get_size({10:10}) > get_size(10)

def test_suppressSTDOUTToFile():
    
    fname="test_output.txt"

    ## have a function that automatically
    ##  tests if its being executed correctly
    ##  re: args and kwargs
    def func(a,b,c,d=0):
        assert (a+b+d) == c
        print("%d + %d + %d = %d"%(a,b,d,c))

    ## save print statement to file
    suppressSTDOUTToFile(func,[1,2,3],fname=fname,mode='w')

    ## test that file was created correctly
    assert os.path.isfile(fname)
    with open(fname,'r') as handle:
        line = handle.readlines()[0].replace('\n','')
        assert line=="1 + 2 + 0 = 3"

    ## clean up
    os.remove(fname)

    ## test kwarg and null pipe
    suppressSTDOUTToFile(func,[1,2,4],fname=None,mode='w',d=1)
    assert not os.path.isfile(fname)