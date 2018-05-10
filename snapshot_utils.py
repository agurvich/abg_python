import h5py,sys,os
from readsnap import readsnap
import numpy as np

def makeHeader(snapshot,build_snapshot):
    ref_header = dict(zip(['Flag_IC_Info','NumFilesPerSnapshot','MassTable','Time',
                            'HubbleParam','Flag_Metals','BoxSize','NumPart_Total',
                           'NumPart_ThisFile',"NumPart_Total_HighWord",'Redshift','Omega0',
                           'OmegaLambda','Flag_Sfr','Flag_Cooling','Flag_StellarAge',
                          'Flag_Feedback', 'Flag_DoublePrecision'],
                         [2,1,np.array([0,0,0,0,0]),0,1,17,100,
                          np.array([0,0,0,0,0]),np.array([0,0,0,0,0]),
                         np.array([0,0,0,0,0]),0,0,0,1,1,1,1,0]))
    print 'ref header created',ref_header.keys()
    with h5py.File(snapshot,'r') as snap:
        with h5py.File(build_snapshot,'w') as build_snap:
            try:
                build_snap.create_group('Header')
            except ValueError:
                pass
            ngas = 0
            nstar = 0
            numpart = np.array([ngas,0,0,0,nstar,0])
            build_snap['Header'].attrs['NumPart_ThisFile']=numpart
            build_snap['Header'].attrs['NumPart_Total']=numpart
            print build_snap['Header'].attrs['NumPart_ThisFile']
            print build_snap['Header'].attrs['NumPart_Total']
            
            for key in snap['Header'].attrs:
                try:
                    ref_val = ref_header[key]
                    snap_val = getHeaderValue(snap,key)
                    bool_check =ref_val==snap_val 
                    if type(bool_check)!=bool:
                        bool_check=bool_check.all()

                    if bool_check:
                        setHeaderValue(build_snap,key,ref_val)
                    else:
                        print key,'disagreed, setting manually'
                        if key in ['Flag_IC_Info','MassTable','Time',
                            'HubbleParam']:
                            setHeaderValue(build_snap,key,ref_val)
                        elif key in ['Flag_Metals','NumFilesPerSnapshot',
                            'Omega0','OmegaLambda','Flag_Sfr']:
                            setHeaderValue(build_snap,key,snap_val)
                        elif key in ['BoxSize']:
                            setHeaderValue(build_snap,key,1000)
                        elif key in ['NumPart_Total','NumPart_ThisFile']:
                            setHeaderValue(build_snap,key,np.zeros(5))
                        else:

                            print '-----'
                            print "Didn't set",key
                            print 'snap',snap_val
                            print 'ref',ref_val
                            print '-----'
                except:
                    print key,'failed raised an uncaught error'
                    print 'snap',getHeaderValue(snap,key)
                    print 'ref',ref_header[key]
                    raise
                    pass

            print
            print 'Latte-ISO contents:'
            print build_snap.keys()
            print build_snap['Header'].attrs.keys()

def addGas(snapshot,build_snapshot):
    with h5py.File(snapshot,'r') as snap:    
        print '\nLatte Gas contents:\n',snap['PartType0'].keys(),'\n'
        HubbleParam=snap['Header'].attrs['HubbleParam']

        with h5py.File(build_snapshot,'a') as build_snap:
            print '\nmaking build snap...\n'
            
            try:                 
                for key in snap['PartType0'].keys():
                    if key in ['SmoothingLength','Masses','Coordinates']:
                        build_snap['PartType0/%s'%key]=np.array(snap['PartType0/%s'%key])/HubbleParam
                    elif key == 'Density':
                        build_snap['PartType0/%s'%key]=np.array(snap['PartType0/%s'%key])*HubbleParam**2.
                    else:
                        print 'Copying:',key
                        build_snap['PartType0/%s'%key]=np.array(snap['PartType0/%s'%key])
                
                #set FIRE-2 ID requirements to keep track of what split when
                try:
                    build_snap['PartType0/ParticleChildIDsNumber']=np.zeros(len(build_snap['PartType0/Masses']))
                    build_snap['PartType0/ParticleIDGenerationNumber']=np.zeros(len(build_snap['PartType0/Masses']))
                except RuntimeError:
                    pass
                
                ngas=len(build_snap['PartType0/Masses'])
            except:
                raise

                print 'Latte-ISO Contents:'
                print build_snap.keys()
                print build_snap['PartType0'].keys()
    return ngas

def addStars(snapshot,build_snapshot):
    with h5py.File(snapshot,'r') as snap:    
        print '\nLatte Star contents:\n',snap['PartType4'].keys(),'\n'
        HubbleParam=snap['Header'].attrs['HubbleParam']

        with h5py.File(build_snapshot,'a') as build_snap:
            print '\nmaking build snap...\n'
            
            try:                 
                for key in snap['PartType4'].keys():
                    if key in ['SmoothingLength','Masses','Coordinates']:
                        build_snap['PartType4/%s'%key]= np.array(snap['PartType4/%s'%key])/HubbleParam
                    elif key == 'StellarFormationTime':
                        
                        #set ages, which are *different now!!*
                        Omega0 = getHeaderValue(snap,'Omega0')
                        Time = getHeaderValue(snap,'Time')

                        ages = convertStellarAges(HubbleParam,
                            Omega0,np.array(snap['PartType4/StellarFormationTime']),Time)
                        #stellarformationtime should be negative since they were created 
                        #*before* the start of the simulation
                        build_snap['PartType4/StellarFormationTime']=-ages
                    else:
                        print 'Copying:',key
                        build_snap['PartType4/%s'%key]=np.array(snap['PartType4/%s'%key])
                        
                #set FIRE2 ids, which didn't exist before
                try:
                    build_snap['PartType4/ParticleChildIDsNumber']=np.zeros(len(snap['PartType4/Masses']))
                    build_snap['PartType4/ParticleIDGenerationNumber']=np.zeros(len(snap['PartType4/Masses']))
                except RuntimeError:
                    pass
                
                nstars=len(build_snap['PartType4/Masses'])
            except:
                raise

                print 'Latte-ISO Contents:'
                print build_snap.keys()
                print build_snap['PartType4'].keys()
    return nstars

def translateSnap(snapnum,multisnap=0):
    if multisnap:
        snapshot = os.path.join(snapdir,'snapshot_600.%d.hdf5'%snapnum)
        build_snapshot = os.path.join(new_snapdir,'snapshot_000.%d.hdf5'%snapnum)
    else:
        snapshot = os.path.join(snapdir,'snapshot_600.hdf5')
        build_snapshot = os.path.join(new_snapdir,'snapshot_000.hdf5')
        
    makeHeader(snapshot,build_snapshot)
    ngas = addGas(snapshot,build_snapshot)
    nstar = addStars(snapshot,build_snapshot)
    ndm = addDM(snapshot,build_snapshot)
    
    with h5py.File(build_snapshot,'a') as build_snap:
        numpart = np.array([ngas,ndm,0,0,nstar])
        build_snap['Header'].attrs['NumPart_ThisFile']=numpart  
        print build_snap['Header'].attrs['NumPart_ThisFile']
         
    return numpart

def setNumpartTotal(snapnum,numpart_total,multisnap=0):
    if multisnap:
        build_snapshot = os.path.join(new_snapdir,'snapshot_000.%d.hdf5'%snapnum)
    else:
        build_snapshot = os.path.join(new_snapdir,'snapshot_000.hdf5')
        
    with h5py.File(build_snapshot,'a') as build_snap:
        build_snap['Header'].attrs['NumPart_Total']=numpart_total
        print build_snap['Header'].attrs['NumPart_Total']


#numpart_total=np.array([0,0,0,0,0])

#for snapnum in [0,1,2,3]:
    #numpart=translateSnap(snapnum,multisnap=multisnap)
    #numpart_total+=numpart

#for snapnum in range(4):
    #setNumpartTotal(snapnum,numpart_total,multisnap=multisnap)

######################## Rewriting readsnap tho ##########
HEADER_KEYS = [
    #u'NumPart_ThisFile', u'NumPart_Total', u'NumPart_Total_HighWord',
    #u'NumFilesPerSnapshot',u'MassTable',
    u'Time', u'Redshift', u'BoxSize', 
    u'Omega0', u'OmegaLambda', u'HubbleParam',
    #u'Flag_Sfr', u'Flag_Cooling',u'Flag_StellarAge', u'Flag_Metals',
    #u'Flag_Feedback', u'Flag_DoublePrecision', u'Flag_IC_Info'
    ]

def get_fnames(snapdir,snapnum):
    fnames = [os.path.join(snapdir,fname) for fname in os.listdir(snapdir) if "%03d"%snapnum in fname]
    if len(fnames) > 1:
        raise Exception("Too many files found for that snapnum!",fnames)

    if os.path.isdir(fnames[0]):
        fnames = [os.path.join(snapdir,fnames[0],fname) for fname in os.listdir(fnames[0])]

    return fnames

def fillHeader(dictionary,handle):
    for hkey in HEADER_KEYS:
        dictionary[hkey] = handle['Header'].attrs[hkey]

def get_unit_conversion(new_dictionary,pkey,cosmological):
    unit_fact = 1
    hinv = new_dictionary['HubbleParam']**-1
    if pkey in ['SmoothingLength','Masses','Coordinates']:
        unit_fact*=hinv
    if cosmological:
        ascale = (1+new_dictionary['Redshift'])
        if pkey in ['SmoothingLength','Coordinates']:
            unit_fact*=ascale
        if pkey in ['Density']:
            unit_fact*=(hinv/((ascale*hinv)**3))
        if pkey in ['Velocity']:
            unit_fact*=ascale**0.5
    return unit_fact

def openSnapshot(
    snapdir,snapnum,ptype,
    snapshot_name='snapshot', extension='.hdf5',
    cosmological=0,header_only=0):

    fnames = get_fnames(snapdir,snapnum)
    
    new_dictionary = {}

    for i,fname in enumerate(sorted(fnames)):
        with h5py.File(fname,'r') as handle:
            if i == 0:
                ## read header once
                fillHeader(new_dictionary,handle)

                if not header_only:

                    ## initialize particle arrays
                    for pkey in handle['PartType%d'%ptype].keys():
                        unit_fact = get_unit_conversion(new_dictionary,pkey,cosmological)
                        new_dictionary[pkey] = np.array(handle['PartType%d/%s'%(ptype,pkey)])*unit_fact

            else:
                if not header_only:
                    ## append particle array for each file
                    for pkey in handle['PartType%d'%ptype].keys():
                        unit_fact = get_unit_conversion(new_dictionary,pkey,cosmological)
                        new_dictionary[pkey] = np.append(
                            new_dictionary[pkey],
                            np.array(handle['PartType%d/%s'%(ptype,pkey)])*unit_fact,axis=0)

    return new_dictionary



## pandas dataframe stuff
def readsnapToDF(snapdir,snapnum,parttype):
    res = readsnap(snapdir,snapnum,parttype,cosmological='m12i' in snapdir)
    
    ids = res.pop('id')

    vels = res.pop('v')
    coords = res.pop('p')

    res['xs'],res['vxs'] = coords[:,0],vels[:,0]
    res['ys'],res['vys'] = coords[:,1],vels[:,1]
    res['zs'],res['vzs'] = coords[:,2],vels[:,2]


    metallicity = res.pop('z')
    for i,zarray in enumerate(metallicity.T):
        res['met%d'%i]=zarray
    
    snap_df = pd.DataFrame(res,index=ids)
    return snap_df
