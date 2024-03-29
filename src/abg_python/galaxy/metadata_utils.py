### function decorators 
import functools
import time
import numpy as np

import h5py
import os

from ..function_utils import filter_kwargs

class Metadata(object):
    """Read in metadata for a class and reference it 
        opaquely through the instance.metadata object without worrying 
        about datagroup prefixes."""
         
    def __init__(
        self,
        metapath,
        loud_metadata=1,
        upfront_load=False,
        groups_to_sub_load_with_index=None,
        sub_load_low_indices=None,
        sub_load_high_indices=None,
        target_last_sizes=None,
        sub_load_exclude=None,
        loud=True):


        self.metapath = metapath
        self.loud_metadata = loud_metadata
        self.file_keys = []

        if groups_to_sub_load_with_index is not None:
            if (sub_load_low_indices is None or 
                sub_load_high_indices is None or 
                target_last_sizes is None or
                len(sub_load_low_indices) != len(groups_to_sub_load_with_index) or
                len(sub_load_high_indices) != len(groups_to_sub_load_with_index) or
                len(target_last_sizes) != len(groups_to_sub_load_with_index)):

                raise ValueError(
                    "Pass in  valid sub_load_<low/high>_indices",
                    groups_to_sub_load_with_index,
                    sub_load_low_indices,
                    sub_load_high_indces,
                    target_last_sizes)

            ## must be a list because we use the .index method
            self.groups_to_sub_load_with_index = list(groups_to_sub_load_with_index)
        else:
            self.groups_to_sub_load_with_index = None
            
        self.sub_load_low_indices = sub_load_low_indices
        self.sub_load_high_indices = sub_load_high_indices
        self.target_last_sizes = target_last_sizes
        self.sub_load_exclude = [] if sub_load_exclude is None else sub_load_exclude

        self.file_groups = []

        try:
            with h5py.File(metapath,'r') as handle:
                ## handle groups
                for group in handle.keys():
                    self.file_groups += [group]
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
            if self.loud_metadata and loud:
                print("Couldn't find a metadata file... for\n",self )

    def hasattr(self,attr):
        return attr in dir(self) or attr in self.file_keys

    def check_for_partial_match(self,attr):
        dict_attrs_match = []
        for key in self.__dict__.keys():
            if attr in key:
                dict_attrs_match+=[key]

        file_attrs_match = []
        for key in self.file_keys:
            if attr in key:
                file_attrs_match+=[key]

        return dict_attrs_match,file_attrs_match

    def __getattr__(self,attr,loud=False,**kwargs):
        try:
            if attr in self.__dict__.keys():
                return self.__dict__[attr] 
            elif attr in self.file_keys:
                self.lazy_load_from_file(attr,**kwargs)
                return self.__dict__[attr] 
            else:
                 raise KeyError("%s isn't in the file or the live memory"%attr)
        except KeyError:
            dict_keys,file_keys = self.check_for_partial_match(attr)

            ## if we haven't loaded it, check the disk
            if len(dict_keys)==0:
                ## we have no matches
                if len(file_keys) == 0:
                    raise AttributeError("No unloaded or loaded metadata attrs matches %s!"%attr)
                ## we have unloaded multiple partial matches
                elif len(file_keys)>1:
                    if loud: print(attr,file_keys)
                    raise KeyError("Too many unloaded metadata attrs match that key, be more precise!",file_keys)
                ## we have a single partial match
                else:
                    raise KeyError("Partial unloaded match %s"%file_keys[0],attr)

            ## if we have loaded multiple partial matches
            elif len(dict_keys)>1:
                if loud: print(attr,dict_keys)
                raise KeyError("Too many loaded metadata attrs match that key, be more precise!",dict_keys)

            ## we have a single partial match
            else:
                raise KeyError("Partial loaded match %s"%dict_keys[0])

    def __repr__(self):
        return "Metadata object at %s" % (self.metapath)# + self.__dict__.keys()

    def __save_to_metadata_object(self,group,key,value,overwrite=0):
        if group == 'header':
            key_to_save = key
        else:
            key_to_save = "%s_%s"%(group,key)
        if key_to_save in self.__dict__ and not overwrite:
            raise Exception("This is already in the metadata object")
        else:
            setattr(self,key_to_save,value)
        
    def save_to_metadata(self,group,key,value,mode='a',overwrite=0,attempts=0):
        ## first determine if this should be saved as an attribute of
        ##  the currently open metadata object
        if attempts <=0: self.__save_to_metadata_object(group,key,value,overwrite)
        try: self.__save_to_metadata_file(group,key,value,mode,overwrite)
        except OSError as e:
            ## likely attempting to save to a metadata file that another thread
            ##  has open. tsk tsk.
            if attempts < 4:
                ## let's wait a bit to try and de-sync from the other thread
                ##  and then try again
                time.sleep(np.random.randint(30,60))
                self.save_to_metadata(group,key,value,mode,overwrite,attempts+1)
            ## well, we tried 5 times so we might as well give up. 
            else: raise e

    def __save_to_metadata_file(self,group,key,value,mode,overwrite):
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

    def inspect_groups(self,substr=''):
        for group in self.file_groups:
            if substr in group: print(group,end='\t')
        print()

    def inspect_metadata(self,this_group=None,print_unloaded=True):
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
            if len(my_set) and print_unloaded:
                print(list(my_set),'keys are unloaded')

            my_set = set(self.__dict__.keys())-set(unloaded_keys)
            if len(my_set):
                print(list(my_set),'keys are unsaved')
        except IOError:
            if self.loud_metadata:
                print("Couldn't find a metadata file...",self.metapath)

    def purge_metadata_group(self,group_name,key_name=None,loud=False,force=0):
        if not force:
            raise Exception("I'm sorry Dave, I'm afraid I can't do that.")
        else:
            with h5py.File(self.metapath,'a') as handle:
                if group_name == 'Header':
                    if key_name is None:
                        ## delete all the keys
                        for key in handle.attrs.keys():
                            del handle.attrs[key]
                    ## delete only the specified key
                    else: del handle.attrs[key_name]
                    return
                
                ## take advantage of h5py group syntax to delete only
                ##  the specified key
                if key_name is not None: group_name = "%s/%s"%(group_name,key_name)

                if group_name in handle:
                    if self.loud_metadata or loud:
                        print(np.array([
                            (key,len(handle[key])) for key in handle.keys()]),
                            '...before...')
                    del handle[group_name]
                    if self.loud_metadata or loud:
                        print(np.array([
                            (key,len(handle[key])) for key in handle.keys()]),
                            "...after. I hope you're happy.")
                else:
                    if self.loud_metadata or loud:
                        print("This metadata doesn't have %s."%group_name)

            if key_name is not None: 
                group_name = group_name.replace('/','_')
                if group_name in self.__dict__.keys():
                    self.__dict__.pop(key_name)
            else:
                ## now get the whole group out of the instance
                for key in list(self.__dict__.keys()):
                    if key[:len(group_name)] == group_name:
                        self.__dict__.pop(key)

    def lazy_load_from_file(self,key,load_entire_group=False,attempts=0):
        try: self.__lazy_load_from_file(key,load_entire_group=load_entire_group)
        except OSError as e: 
            ## likely attempting to load from a metadata file that another thread
            ##  has open. tsk tsk.
            if attempts < 4:
                ## let's wait a bit to try and de-sync from the other thread
                ##  and then try again
                time.sleep(np.random.randint(30,60))
                self.lazy_load_from_file(key,load_entire_group,attempts+1)
            ## well, we tried 5 times so we might as well give up. 
            else: raise e
    
    def __lazy_load_from_file(self,key,load_entire_group=False):
        if key in self.file_keys:
            with h5py.File(self.metapath,'r') as handle:
                ## handle groups
                for group in handle.keys():
                    ## is this the group this key lives in?
                    if key[:len(group)] == group:
                        ## load the key that was requested

                        sub_key = key[len(group)+1:]

                        if sub_key not in handle[group].keys():
                            ## degenerate group/key combo, keep looking!
                            continue

                        value = self.__load_from_open_handle(handle,group,sub_key)
                        setattr(self,'%s_%s'%(group,sub_key),value)
                        if load_entire_group:
                            ## read the entire group to minimize re-opening the file.
                            for key in handle[group].keys():
                                try:
                                    value = self.__load_from_open_handle(handle,group,key)
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
            raise KeyError("%s isn't in the metadata file"%key)

    def __load_from_open_handle(self,handle,group,key):
        pathh = '%s/%s'%(group,key)
        shape = handle[pathh].shape
        ## determine if we are trying to load a subset of the data
        ##  for speed and memory purposes
        if (self.groups_to_sub_load_with_index is not None and
            group in self.groups_to_sub_load_with_index): 

            ## what index is this group at so we can find matching
            ##  sub load parameters
            group_index = self.groups_to_sub_load_with_index.index(group)

            ## does this array match the target we are trying to mask?
            if (shape != () and
                shape[-1] == self.target_last_sizes[group_index] and
                key not in self.sub_load_exclude):
                ## load only from [low:high]
                low = self.sub_load_low_indices[group_index]
                high = self.sub_load_high_indices[group_index]
                return np.array(handle[pathh][...,low:high])
            else:
                return np.array(handle[pathh])
        else:
            return np.array(handle[pathh])

    def export_to_file(self,target,group_name,keys=None,write_mode='a'):
        if type(target) == dict:
            raise NotImplementedError
        
        ## relative paths are w.r.t. to home for convenience
        if target[0] != os.sep:
            target = os.path.join(os.environ['HOME'],target)
            print('Detected relative path, prepending $HOME, new target:',target)
        
        dirname = os.path.dirname(target)
        if not os.path.isdir(dirname): os.makedirs(dirname)

        ## make a new hdf5 file
        with h5py.File(target,write_mode) as target_handle:
            ## open this hdf5 file
            with h5py.File(self.metapath,'r') as handle:
                ## check if this group/key combo can/should be exported
                if group_name in handle.keys():
                    ## for the first key we need to create a home in the 
                    ##  target file
                    if group_name not in target_handle.keys():
                        target_handle.create_group(group_name)

                    ## loop through all keys and check if they're allowed
                    for key in handle[group_name].keys():
                        if keys is None or (key in keys):
                            if self.loud_metadata:
                                print("Copying %s/%s from"%(group_name,key),self,'to',target)
                            ## copy the data over
                            ## in 'w' mode this won't happen, in 'a' mode we don't want to overwrite
                            if key in target_handle[group_name].keys(): continue
                            target_handle[group_name][key] = handle[group_name][key][()]
                else:
                    raise IOError("%s doesn't have %s."%(repr(self),group_name))

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
                raise AttributeError(self.metapath[gal_i],"doesn't have",attr)
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
        return repr(self.snap_metadatas)

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
    force_from_file=False,
    check_cached_only=0,
    clean_object=False,
    clean_metadata=False,
    **kwargs):

    ## overwrite loud if we're only checking if something is cached
    if check_cached_only:
        loud=False

    def decorator(func):

        @functools.wraps(func)
        def wrapper(
            *func_args,
            **func_kwargs) :

            ## need to declare these two as nonlocal since
            ##  we write to them and that makes python
            ##  think they will be local variables
            nonlocal check_cached_only,force_from_file,loud,clean_object,clean_metadata

            ## not every function I've written has these explicitly passed,
            ##  so peel them out of any kwargs for good measure
            if 'check_cached_only' in func_kwargs:
                check_cached_only = func_kwargs.pop('check_cached_only')

            if check_cached_only:
                loud=False

            if 'force_from_file' in func_kwargs:
                force_from_file = func_kwargs.pop('force_from_file')

            if 'clean_object' in func_kwargs:
                clean_object = func_kwargs.pop('clean_object')
                
            if 'clean_metadata' in func_kwargs:
                clean_metadata = func_kwargs.pop('clean_metadata')

            ## NOTE could put something that prints ignored_kwargs
            func_kwargs,ignored_kwargs = filter_kwargs(func,func_kwargs)

            self = func_args[0]

            func_name = "%s%s%s"%(
                func.__name__,
                repr(func_args),
                repr(func_kwargs))


            if clean_object:
                print('Clearing %s and exiting.'%func_name)
                for key in keys:
                    if hasattr(self,key): delattr(self,key)
                return

            if clean_metadata:
                print('Clearing metadata of %s and exiting.'%func_name)
                for key in keys:
                    self.metadata.purge_metadata_group(
                        group,
                        key,
                        force=True,
                        loud=False)
                return
                    
            try:
                ## does the function call want us to use the metadata?
                ##  allow users to opt out
                if not use_metadata:
                    raise AssertionError("Told to not use metadata.")

                for key in keys:
                    if not check_cached_only:
                        try:
                            if force_from_file:
                                raise AttributeError
                            value = getattr(self,key)
                        except AttributeError:
                            value = getattr(self.metadata,"%s_%s"%(group,key)) 
                            setattr(self,key,value)
                    else:
                        if not hasattr(self.metadata,"%s_%s"%(group,key)):
                            raise AttributeError("Missing: %s - %s"%(group,key))

                if check_cached_only: return

                if len(keys) > 1:
                    return_value = tuple([getattr(self,key) for key in keys])
                else:
                    return_value = getattr(self,key)

                if loud: print("cache",group,func_name,"success!")
            except (AssertionError,KeyError,AttributeError) as e:
                if loud and use_metadata: print("cache",group,func_name,"fail :[",e)
                if assert_cached:
                    raise AssertionError("User asserted cached for %s - %s"%(group,func_name),keys)
                init = time.time()
                ## go ahead and actually call the function
                return_value = func(*func_args,**func_kwargs)
                duration = time.time()-init
                print(func_name,'%.2f s elapsed'%(duration))
                
                ## must be explicitly asked to save to metadata
                ##  with a kwarg
                if save_meta:
                    ## save the time it took to get the solution
                    try:
                        self.metadata.save_to_metadata(
                            group,
                            '%s_duration'%func_name,
                            [duration],
                            overwrite=True)
                    except:
                        ## TODO debug this if it's failing somewhere....
                        pass
                    ## confirm we returned the number of things
                    ##  we expected to
                    if type(return_value) == tuple:
                        if len(keys) != len(return_value):
                            raise ValueError(
                                "Number of keys doesn't match",
                                len(keys),keys,
                                "number of returned values!",
                                len(return_value))
                        for key,value in zip(keys,return_value):
                            self.metadata.save_to_metadata(
                                group,
                                key,
                                value,
                                overwrite=1)
                    else:
                        if len(keys) != 1:
                            ## otherwise if you returned just a list alone it could try and 
                            ##  bind the elements to keys?
                            raise ValueError(
                                "Number of keys should be 1",
                                len(keys),keys,
                                "if you return a list",
                                type(return_value),len(return_value))

                        self.metadata.save_to_metadata(
                            group,
                            keys[0],
                            return_value,
                            overwrite=1)

                ## set the attribute to self regardless of save_meta
                for key,value in zip(keys,return_value):
                    setattr(self,key,value)

            if type(return_value) == tuple and len(keys) == 1 and len(return_value) == 1:
                return_value=return_value[0]
            return return_value
        return wrapper
    return decorator
