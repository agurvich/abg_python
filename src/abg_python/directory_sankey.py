#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import os
import numpy as np
HOMEenviron = os.environ['HOME']

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')




import plotly.graph_objects as go
import plotly
#plotly.offline.init_notebook_mode(connected=True)
#import plotly.io as pio
#pio.renderers.default = "iframe"
#from abg_python.distinct_colours import get_distinct




def main(
    du_file_name,
    top_level_name=None,
    min_size=0.1): ## TB

    ## what is the name of the left-most node (name of current directory). 
    ##  in file it is just '.'
    if top_level_name is None: top_level_name = du_file_name

    print('reading',du_file_name)

    ## read the du -b > storage.txt output
    storage = pd.read_csv(
        os.path.join(HOMEenviron,du_file_name),
        delimiter='\t',
        header=None)


    ## convert to numpy because pandas makes my head hurt
    mems = storage[0].array.to_numpy()
    paths = storage[1].array.to_numpy()


    ## remove prepended ./ from each path
    for i,path in enumerate(paths):
        paths[i] = paths[i].replace('./','')


    ## keep only files larger minimum size
    mask = mems > (1024**4*min_size)
    big_paths = paths[mask]
    big_mems = mems[mask]
    print(np.sum(mask),'many files > 1 GB')

    ## sort the paths alphabetically
    ##  this will align sub-directories 
    ##  underneath parent directories
    indices = np.argsort(big_paths)
    big_mems = big_mems[indices]
    big_paths = big_paths[indices]

    ## create a look-up dictionary for directory sizes
    global size_dict
    size_dict = dict(zip(big_paths,big_mems))


    ## parse the top-level directory manually
    ##  create a dictionary of nodes with sub-directory keys
    ##  and a 'size' key
    top_folders = inner_parse_strings(big_paths[:])
    top_folders = parse_file_string_list(top_folders)
    top_folders['size'] = None


    ## remove '.' from the node dictionary, store its size for labels later 
    print('top level folder keys:',top_folders.keys())
    total_size = top_folders.pop('.')['size']

    ## walk the node dictionary and flatten it 
    ##  into input go.sankey expects
    labels, xs, origins, targets, sizes = sankify(
        top_folders,
        level=1,
        min_size=min_size)


    sizes = np.array(sizes)
    xs = np.array(xs)
    ys = np.zeros(xs.size)

    ## create the x and y positions of the nodes of the 
    ##  sankey diagram, x is in discrete levels, y ranges
    ##  from 0->1 for each node in that level
    print(xs.max(),'directory levels')
    for i in range(1,xs.max()+1):
        this_mask = xs==i
        n_this_level = np.sum(this_mask)
        print('level %d:'%i,n_this_level)

        ys[this_mask]=np.linspace(
            0.001, ## can't quite be 0, that seems to break things
            1.0,
            n_this_level,
            endpoint=True)
        
    xs = (xs+0.01)/np.max(xs)

    ## add the top level back into the label list
    labels = np.append([top_level_name],labels)
    sizes = np.append([total_size],sizes)
    ## manually place it
    xs = np.append([0],xs)
    ys = np.append([0.5],ys)

    print('labels',len(labels),
        'xs',len(xs),'ys',len(ys),
        'sizes',len(sizes),
        'origins',len(origins),
        'targets',len(targets))

    these_labels = ["%s - %.2f T"%(label,size) for label,size in zip(labels[:],sizes)]

    ## crate the sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 5,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = these_labels,
            #label = ["%s x=%.2f, y=%.2f"%(label,x,y) for label,x,y in zip(these_labels,xs,ys)] ,
            color = 'blue',
        y=ys,
        x=xs),
        link = dict(
            source = origins,
            target = targets,
            value = sizes[1:]),
        orientation='h',
        arrangement='snap')])

    ## make it big
    fig.layout=go.Layout(
        autosize=False,
        width=500*3,
        height=2000,
        margin=dict(l=20, r=20, t=20, b=200))

    ## write output to html
    fig.write_html(os.path.join(
        HOMEenviron,
        du_file_name.replace('.txt','.html')))


def inner_parse_strings(paths,pre_str=''):
    node = {}
    
    for fname in paths:
        ## skip 'snapdir' and 'output' directories, we already
        ##  get the simulation name 1 level above
        if 'snapdir' in fname or 'output' in fname: continue
        llist = fname.split(os.sep)
        
        directory = llist[0]
        
        if directory not in node:
            node[directory]= {
                ## use global size look-up dictionary
                'size':size_dict[pre_str+fname],
                'fnames':[]}
        ## add sub-directories to this directory
        else: node[directory]['fnames']+=[os.sep.join(llist[1:])]
        
    ## b -> Kb -> Mb -> Gb -> Tb
    for directory in node:
        node[directory]['size'] = node[directory]['size']/1024**4
    return node

def parse_file_string_list(node,pre_str=''):
    for directory in list(node.keys()):
        fnames = node[directory].pop('fnames')

        ## if there are sub-directories, create a node
        ##  for each of them
        if len(fnames):
            sub_node = parse_file_string_list(
                inner_parse_strings(
                    fnames,
                ## need to pass pre_str so the size look-up 
                ##  dictionary works.
                    pre_str=pre_str+directory+os.sep),
                pre_str=pre_str+directory+os.sep) 

            ## attach the sub-node
            node[directory].update(sub_node)

    return node


def sankify(my_dict,level=0,origin=0,offset=0,min_size=None):
    
    labels = list(my_dict.keys())
    labels.pop(labels.index('size'))

    ## filter out sub-directories which are too small
    new_labels = []
    for label in labels:
        if my_dict[label]['size'] >= min_size: new_labels+=[label]
    labels = new_labels

    ## create this layer's lists
    sizes = [my_dict[label]['size'] for label in labels]
    xs = [level]*len(labels)
    origins = [origin]*len(labels)
    targets = list(np.arange(1,len(labels)+1)+offset)
    
    ## loop through each sub-directory and recursively sankify
    for j in range(len(labels)):
        key = labels[j]
        this_labels,this_xs,this_origins,this_targets,this_sizes = sankify(
            my_dict[key],
            level+1, ## 1 layer deeper
            origin=1+offset+j, ## set the origin to be this sub-directory
            offset=len(labels)+offset, ## see below, labels gets longer
            min_size=min_size)

        ## append this layer to the flattened list(s)
        labels+=this_labels
        xs+=this_xs
        origins+=this_origins
        targets+=this_targets
        sizes+=this_sizes

    return labels, xs, origins, targets, sizes

if __name__ == '__main__':

    du_file_name = 'stampede_2_usage_11.15.2021.txt'
    top_level_name = 'GalaxiesOnFIRE'
    du_file_name = 'frontera_usage_11.15.21.txt'
    top_level_name = 'pfh-frontera-scratch'
    du_file_name = 'fire3_usage.txt'
    top_level_name = 'stampede2/GalaxiesOnFire/fire3'
    du_file_name = 'stampede2_fire3.txt'
    top_level_name = 'fire3_compatability'
    du_file_name = 'frontera_5.4.22.txt'
    top_level_name = 'frontera'

    main(du_file_name,top_level_name,min_size=1) ## TB
