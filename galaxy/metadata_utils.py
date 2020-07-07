### function decorators 
import functools
import time
import numpy as np

import h5py

from abg_python.all_utils import filter_kwargs

class Metadata(object):
    """Read in metadata for a class and reference it 
        opaquely through the instance.metadata object without worrying 
        about datagroup prefixes."""
         
    def __init__(self,metapath,loud_metadata=1,upfront_load=False):
        self.metapath = metapath
        self.loud_metadata = loud_metadata
        self.file_keys = []
        try:
            with h5py.File(metapath,'r') as handle:
                ## handle groups
                for group in handle.keys():
                    for key in handle[group].keys():
                        self.file_keys+=["%s_%s"%(group,key)]
                        if upfront_load: 
                            value = np.array(handle['%s/%s'%(group,key)])
                            ## stupid way of reading dataset and copying it into permanent memory
                            if value.size == 1:
                                value = value.reshape(1)
                                value = value[0]
                            setattr(self,'%s_%s'%(group,key),value)
                ## handle header, if it exists
                for key in handle.attrs.keys():
                    value = handle.attrs[key]
                    setattr(self,key,value)
        except IOError:
            if self.loud_metadata:
                print("Couldn't find a metadata file... for\n",self )

    def hasattr(self,attr):
        return attr in dir(self) or attr in self.file_keys

    def __getattr__(self,attr,loud=False):
        try:
            if attr in self.__dict__.keys():
                return self.__dict__[attr] 
            elif attr in self.file_keys:
                self.lazy_load_from_file(attr)
                return self.__dict__[attr] 
            else:
                 raise KeyError("%s isn't in the file or the live memory"%attr)
        except KeyError:
            key_matches = []
            for key in self.__dict__.keys():
                if attr in key:
                    key_matches+=[key]
            
            if len(key_matches)==0:
                for key in self.file_keys:
                    if attr in key:
                        key_matches+=[key]

            ## otherwise use a partial match
            if len(key_matches)>1:
                if loud:
                    print(attr,key_matches)
                raise KeyError("Too many metadata attrs match that key, be more precise!",key_matches)
            elif len(key_matches)==0:
                raise AttributeError("No metadata attrs matches %s!"%attr)
            else:
                raise KeyError("Partial match %s"%key_matches[0])

    def __repr__(self):
        return "Metadata object at %s" % (self.metapath)# + self.__dict__.keys()

    def _save_to_metadata_object(self,group,key,value,overwrite=0):
        if group == 'header':
            key_to_save = key
        else:
            key_to_save = "%s_%s"%(group,key)
        if key_to_save in self.__dict__ and not overwrite:
            raise Exception("This is already in the metadata object")
        else:
            setattr(self,key_to_save,value)
        
    def save_to_metadata(self,group,key,value,mode='a',overwrite=0):
        ## first determine if this should be saved as an attribute of
        ##  the currently open metadata object
        self._save_to_metadata_object(group,key,value,overwrite)
        with h5py.File(self.metapath,mode) as handle:
            if group == 'header':
                if key not in handle.attrs.keys():
                    handle.attrs[key]=value
                else:
                    if overwrite:
                        del handle.attrs[key]
                        handle.attrs[key]=value
                    else:
                        print('value already exists, but not overwriting')
            else:
                if group not in handle.keys():
                    handle.create_group(group)
                if key not in handle[group].keys():
                    handle['%s/%s'%(group,key)]=value
                else:
                    if overwrite:
                        del handle[group][key]
                        handle[group][key]=value
                    else:
                        print('value already exists, but not overwriting')

    def inspect_metadata(self,this_group=None):
        unloaded_keys = []
        try:
            with h5py.File(self.metapath,'r') as handle:
                print('Found a metadata file')
                print('-- Header -- ')
                for key in handle.attrs.keys():
                    unloaded_keys += ["%s"%key]
                    print(key,)

                print()

                for group in handle.keys():
                    if (this_group is None) or (group == this_group):
                        print('-- %s -- '%group)
                    for key in handle[group].keys():
                        if (this_group is None) or (group == this_group):
                            print('%s - %s '%(group,key),)
                        unloaded_keys += ["%s_%s"%(group,key)]
                    if (this_group is None) or (group == this_group):
                        print()
            my_set = set(unloaded_keys)-set(self.__dict__.keys())
            if len(my_set):
                print(list(my_set),'keys are unloaded')

            my_set = set(self.__dict__.keys())-set(unloaded_keys)
            if len(my_set):
                print(list(my_set),'keys are unsaved')
        except IOError:
            if self.loud_metadata:
                print("Couldn't find a metadata file...",self.metapath)

    def purge_metadata_group(self,group_name,loud=False,force=0):
        if not force:
            raise Exception("I'm sorry Dave, I'm afraid I can't do that.")
        else:
            with h5py.File(self.metapath,'a') as handle:
                if group_name == 'Header':
                    for key in handle.attrs.keys():
                        del handle.attrs[key]
                if group_name in handle.keys():
                    if self.loud_metadata or loud:
                        print(handle.keys(),'...before...')
                    del handle[group_name]
                    if self.loud_metadata or loud:
                        print(handle.keys(),"...after. I hope you're happy.")
                else:
                    if self.loud_metadata or loud:
                        print("This metadata doesn't have %s."%group_name)

            ## now get it out of the instance itself
            for key in list(self.__dict__.keys()):
                if key[:len(group_name)] == group_name:
                    self.__dict__.pop(key)

    def lazy_load_from_file(self,key,load_entire_group=False):
        if key in self.file_keys:
            with h5py.File(self.metapath,'r') as handle:
                ## handle groups
                for group in handle.keys():
                    ## is this the group this key lives in?
                    if key[:len(group)] == group:
                        ## load the key that was requested
                        key = key[len(group)+1:]
                        value = np.array(handle['%s/%s'%(group,key)])
                        setattr(self,'%s_%s'%(group,key),value)
                        if load_entire_group:
                            ## read the entire group to minimize re-opening the file.
                            for key in handle[group].keys():
                                try:
                                    value = np.array(handle['%s/%s'%(group,key)])
                                except KeyError:
                                    print("--------------------------------------------------------")
                                    print("corrupt group/key combo")
                                    print(group,key)
                                    print("--------------------------------------------------------")
                                    raise
                                ## stupid way of reading dataset and copying it into permanent memory
                                if value.size == 1:
                                    value = value.reshape(1)
                                    value = value[0]
                                setattr(self,'%s_%s'%(group,key),value)
        else:
            raise KeyError("%s isn't in the metadata file")
    
    def purge_metadata_key(self,group_name,key_name,force=0):
        if not force:
            raise Exception("I'm sorry Dave, I'm afraid I can't do that.")
        else:
            with h5py.File(self.metapath,'a') as handle:
                if group_name in handle.keys() and key_name in handle[group_name].keys():
                    if self.loud_metadata:
                        print(handle[group_name].keys(),'...before...')
                    del handle[group_name][key_name]
                    if self.loud_metadata:
                        print(handle[group_name].keys(),"...after. I hope you're happy.")
                else:
                    if self.loud_metadata:
                        print("This metadata doesn't have that group, or that group doesn't have that key.")

## creates a collection of metadata instances that can be read as a chain
class MultiMetadata(Metadata):
    def __init__(self,snapnums,galaxies,metapath,loud_metadata=1):
        self.loud_metadata = loud_metadata
        if galaxies is not None:
            self.snap_metadatas = [galaxy.metadata for galaxy in galaxies]
        else:
            self.snap_metadatas = [
                Metadata(
                    os.path.join(metapath,'meta_Galaxy_%03d.hdf5'%snapnum)
                    ,loud_metadata=self.loud_metadata)
                for snapnum in snapnums]

    def __getattr__(self,attr):
        llist = []
        for gal_i, metadata in enumerate(self.snap_metadatas):
            try:
                llist += [ getattr(metadata,attr)]
            except AttributeError:
                #raise KeyError(
                #    self.metapath[gal_i],"doesn't have",attr)
                print(self.metapath[gal_i],"doesn't have",attr)
        try:
            """
            ## it's a z-slab map of pixels
            if np.shape(llist)[-1] == 51:
                return np.array(llist) 
            ## it's just a list
            else:
                return np.concatenate(llist,axis=0)
            """
            return np.array(llist)
        except:
            return llist

    def __repr__(self):
        return str(self.snap_metadatas)

    def __getitem__(self,index):
        return self.snap_metadatas[index]
    
    def inspect_metadata(self,index=0,*args):
        return self[index].inspect_metadata(*args)

## wrapper function that can be applied to memoize the output of functions
def metadata_cache(
    group,keys,
    use_metadata=1,
    save_meta=0,
    assert_cached=0,
    loud=1,
    **kwargs):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(
            *func_args,
            **func_kwargs) :

            ## NOTE could put something that prints ignored_kwargs
            func_kwargs,ignored_kwargs = filter_kwargs(func,func_kwargs)

            self = func_args[0]
            func_name = "%s%s%s"%(
                func.__name__,
                repr(func_args),
                repr(func_kwargs))
            try:
                ## does the function call want us to use the metadata?
                ##  allow users to opt out
                assert use_metadata
                for key in keys:
                    try:
                        value = getattr(self,key)
                    except AttributeError:
                        value = getattr(self.metadata,"%s_%s"%(group,key)) 
                        setattr(self,key,value)
                return_value = tuple([getattr(self,key) for key in keys])
                if loud:
                    print("cache",
                        group,func_name,
                        "success!")
            except (AssertionError,KeyError,AttributeError):
                if loud and use_metadata:
                    print("cache",
                        group,func_name,
                        "fail :[")
                if assert_cached:
                    raise AssertionError("User asserted cached")
                init = time.time()
                ## go ahead and actually call the function
                return_value = func(*func_args,**func_kwargs)
                if loud:
                    print(func_name,'%.2f s elapsed'%(time.time()-init))
                
                ## must be explicitly asked to save to metadata
                ##  with a kwarg
                if save_meta:
                    ## confirm we returned the number of things
                    ##  we expected to
                    if type(return_value) == tuple:
                        if len(keys) != len(return_value):
                            raise ValueError(
                                "Number of keys doesn't match"+
                                " number of returned values!")
                        for key,value in zip(keys,return_value):
                            self.metadata.save_to_metadata(
                                group,
                                key,
                                value,
                                overwrite=1)
                    else:
                        if len(keys) != 1:
                            raise ValueError(
                                "Number of keys doesn't match"+
                                " number of returned values!")
                        self.metadata.save_to_metadata(
                            group,
                            keys[0],
                            return_value,
                            overwrite=1)

                ## set the attribute to self regardless of save_meta
                for ikey,key in enumerate(keys):
                    #if loud:
                        #print('setting',key)
                    if len(keys)>1:
                        setattr(self,key,return_value[ikey])
                    else:
                        setattr(self,key,return_value)
            return return_value
        return wrapper
    return decorator
