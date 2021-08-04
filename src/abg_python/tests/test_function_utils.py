from ..function_utils import *
import numpy as np

def test_filter_kwargs():

    def func(foo=1,bar=2):
        return foo + bar
    
    kwargs = {
        'foo':2,
        'bar':3,
        'cat':None,
        'dog':1
    }

    good,bad =  filter_kwargs(func,kwargs)

    assert good  == {'foo':2,'bar':3}
    assert bad == {'cat':None,'dog':1}


def test_append_function_docstring():
    """tests both append_function_docstring and append string docstring"""

    def func1():
        """This is a default docstring.
        Input:
        Output:
            foo - an integer"""

        foo = 1
        return foo

    def func2(bar):
        """Additional docstring.
        Input:
            bar - an integer
        Output:
            bar + 1"""
        return bar + 1
        
    
    append_function_docstring(func1,func2,use_md=False)

    answer = """This is a default docstring.
        Input:
        Output:
            foo - an integer
--------
 test_append_function_docstring.<locals>.func2:
 Additional docstring.
        Input:
            bar - an integer
        Output:
            bar + 1"""

    chars = np.array(list(func1.__doc__))
    ans_chars = np.array(list(answer))

    assert np.all(chars == ans_chars),(len(chars),len(ans_chars))