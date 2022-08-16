import inspect
import functools
import sys
import getopt

def CLI_args():
    def decorator(func):
        ## automatically read the argument names from the function we're passed
        signature = inspect.signature(func)
        cli_args = []

        ## add each argument to the list of getopt options
        for key,parameter in signature.parameters.items(): cli_args += [key]
            #print(key,parameter.default,'empty:',parameter.default == parameter.empty)

        ## use getopt to parse the CLI into tuples and raise errors if we're passed a bad one
        argv = sys.argv[1:]
        opts,args = getopt.getopt(argv,'',[arg+'=' for arg in cli_args])

        ## evaluate CLI args
        for i,opt in enumerate(opts):
            if opt[1]=='': opts[i]=('mode',opt[0].replace('-',''))
            else:
                ## if it's an int or a float this should work
                try: opts[i]=(opt[0].replace('-',''),eval(opt[1]))
                ## if it's a string... not so much
                except: opts[i]=(opt[0].replace('-',''),opt[1])

        ## define a wrapper so that stack traces go through
        @functools.wraps(func)
        def wrapper(): func(**dict(opts))

        return wrapper
    return decorator

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

    

