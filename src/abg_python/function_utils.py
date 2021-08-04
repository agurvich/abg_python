import inspect

def filter_kwargs(func,kwargs):
    good = {}
    bad = {}

    ## get the args that func takes
    allowed_args_dict = inspect.signature(func).parameters
    allowed_args = allowed_args_dict.keys()
    for arg in kwargs.keys():
        ## ignore self if we're inspecting a method
        if arg == 'self':
            continue

        if arg in allowed_args:
            good[arg] = kwargs[arg]
        else:
            bad[arg] = kwargs[arg]
    return good,bad 

def append_function_docstring(function_1,function_2,**kwargs):
    #print(function_1,function_2)
    #print(function_1.__doc__,function_2.__doc__)
    function_1.__doc__ = append_string_docstring(
        function_1.__doc__,function_2,**kwargs)

def append_string_docstring(string,function_2,use_md=True,prepend_string=''):
    prepend_string += use_md*"### "
    name = function_2.__qualname__
    if use_md:
        name = '`' + name + '`' 

    string+="\n--------\n %s:\n %s"%(
        prepend_string + name,
        function_2.__doc__)
    return string

    

