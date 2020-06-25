## from builtin
import numpy as np 
import h5py
import os

## from abg_python
from abg_python.snapshot_utils import openSnapshot,get_unit_conversion


from abg_python.plot_utils import add_to_legend
from abg_python.distinct_colours import get_distinct

import abg_python.all_utils as all_utils
import abg_python.cosmo_utils as cosmo_utils 

from abg_python.galaxy.cosmoExtractor import diskFilterDictionary,offsetRotateSnapshot
from abg_python.galaxy.movie_utils import Draw_helper,FIREstudio_helper
from abg_python.galaxy.sfr_utils import SFR_helper
from abg_python.galaxy.metadata_utils import metadata_cache,Metadata,MultiMetadata

## mapping between particle type and what I called them
sub_snap_dict = {
    0:'sub_snap',
    1:'sub_dark_snap',
    4:'sub_star_snap',
}

snap_dict = {
    0:'snap',
    1:'dark_snap',
    4:'star_snap'
}

## function to determine what the "main" halo is
def halo_id(name):
    if 'm10v_res250' in name:
        return 2
    else:
        return 0 

## bread and butter galaxy class
class Galaxy(
    Draw_helper,
    FIREstudio_helper,
    SFR_helper):
    """------- Galaxy
        Input:
            name - name of the simulation directory
            snapdir - location that the snapshots live in, should end in "output"
            snapnum - snapshot number
            datadir = None - directory where any new files are saved to
            data_name = None - name of the data directory, if different than name
            plot_color = 0 - color this instance will use when plotting onto an axis
            loud_metadata = 1 - whether the data cache will show optional print statements
            multi_thread = 1 - number of threads this instance will use if multi-threading
                is available
            ahf_path = None - path to AHF halo files, defaults first to my halo directory
                and snapdir/../halo/ahf if halo second.
            ahf_fname = None - name of the AHF file, typically smooth_halo_00000.dat

        Provided functions:

    """ 

    __doc__+= (
        "\n"+Draw_helper.__doc__ + 
        "\n"+FIREstudio_helper.__doc__ +
        "\n"+SFR_helper.__doc__)

    def hasattr(self,attr):
        return attr in dir(self)

    def __repr__(self):
        return "%s at %d" % (self.data_name,self.snapnum)# + self.__dict__.keys()

    ## convenience function to add this simulation to a legend
    ##  with its name and plot_color
    def add_to_legend(
        self,
        *args,
        **kwargs):
            
        kwargs['label'] = self.data_name
        kwargs['color'] = self.plot_color
        add_to_legend(*args,**kwargs)

        return ax

    ## GALAXY
    def __init__(
        self,
        name,
        snapdir,
        snapnum,
        datadir=None,
        data_name = None,
        plot_color = 0,
        loud_metadata=1,
        multi_thread = 1,
        ahf_path = None,
        ahf_fname = None,
        ):

        self.name = name

        ## snapdir is sometimes None if an instance is 
        ##  created just to access the metadata and cached
        ##  methods
        if snapdir is not None and snapdir[-1]==os.sep:
            snapdir = snapdir[:-1]

        self.data_name = self.name if data_name is None else data_name

        ## append _md to the data_name for my own sanity
        if '_md' not in self.data_name and 'metal_diffusion' in self.snapdir:
            self.data_name = self.data_name + '_md'

        ## name that should appear on plots
        ##  i.e. remove the resXXX from the name
        pretty_name = self.name.split('_')
        pretty_name = [
            strr if 'res' not in strr else '' 
            for strr in pretty_name] 
        self.pretty_name = '_'.join(pretty_name)
        self.pretty_name = self.pretty_name.replace('__','_')

        ## bind input
        self.snapdir = snapdir
        self.snapnum = snapnum
        self.multi_thread = multi_thread

        ## TODO??
        self.snapdir_name = 'snapdir' if (
            'angles' in self.snapdir or 
            '_md' in self.name 
            and 'm11' not in self.name) else ''

        if type(plot_color) is int:
            plot_color=get_distinct(9)[plot_color] # this is a dumb way to do this
        self.plot_color = plot_color
        
        ## handle datadir creation
        if datadir is None:
            self.datadir = "/home/abg6257/projects/data/"
        else:
            self.datadir = datadir

        ## make top level datadir if it doesn't exist...
        if not os.path.isdir(self.datadir):
            os.mkdir(self.datadir)

        if name not in self.datadir and name!='temp':
            self.datadir = os.path.join(self.datadir,self.data_name)

        if not os.path.isdir(self.datadir):
            os.mkdir(self.datadir)

        ## handle metadatadir creation
        self.metadatadir = os.path.join(self.datadir,'metadata')
        if not os.path.isdir(self.metadatadir):
            os.mkdir(self.metadatadir)

        ## are we just trying to open this simulation's constant header info?
        if self.snapnum is None:
            ## load the header from my catalog file
            ##  if it's not in the header, let this raise an 
            ##  unholy error
            self.header = self.loadHeaderFromCatalog() 

            ## set filler values for variables that would be set if we were
            ##  actually opening the hdf5 file

            ## metadata file 
            self.metapath = None

            ## simulation timing
            self.current_time_Gyr = None
            self.current_redshift = None

            ## main halo information
            self.scom = None
            self.rvir = None
            self.rstar_half = None

        else:
            ## open/create the metadata object & file
            ##  nb that file will only be created after
            ##  first entry is saved to metadata
            self.metapath = os.path.join(
                self.metadatadir,
                'meta_Galaxy_%03d.hdf5'%self.snapnum)
            self.metadata = Metadata(
                self.metapath,
                loud_metadata=loud_metadata)

            ## attempt to open the header in the hdf5 file
            try:
                self.header = openSnapshot(
                    self.snapdir,
                    self.snapnum,
                    0, ## dummy particle index, not used if header_only is True
                    header_only=1,
                    snapdir_name=self.snapdir_name,
                    cosmological=True)

                ## save the header to our catalog of simulation headers
                ##  so that we can open the header in scenarios where
                ##  snapnum is None, as above.
                try:
                    self.saveHeaderToCatalog()
                except ValueError:
                    pass


            except IOError:
                print("Couldn't find snapshot",self.snapnum,"in",self.snapdir)
                return 

            ## simulation timing
            ##  load snapshot times to convert between snapnum and time_Gyr
            try:
                self.createSnapshotTimes()
                self.current_time_Gyr = self.snap_gyrs[self.snapnums==self.snapnum][0]
                self.current_redshift = self.snap_zs[self.snapnums==self.snapnum][0]
            except:
                print("Couldn't load or create snapshot times, opening the header.")
                self.current_redshift = self.header['Redshift']
                self.current_time_Gyr = cosmo_utils.convertReadsnapTimeToGyr(self.header)

            ## attempt to read halo location and properties from AHF
            if ahf_fname is None:
                ahf_fname='halo_0000%d_smooth.dat'%halo_id(self.name)

            if ahf_path is None:
                ## assumes we are on stampede2
                ahf_path = "/work/04210/tg835099/stampede2/halo_files/%s/%s"

                if 'metal_diffusion' in self.snapdir:
                    ahf_path = ahf_path%('metal_diffusion',self.name)
                elif 'HL000' in self.name:
                    ahf_path = ahf_path%('xiangcheng',self.name)
                elif 'core' in self.snapdir:
                    ahf_path = ahf_path%('core',self.name)
                else: ## set myself up for failure below
                    ahf_path = ahf_path%('foo',self.name)

            ## check if this first guess at the ahf_name and ahf_path
            ##  is right
            if not os.path.isfile(os.path.join(ahf_path,ahf_fname)):
                ## try looking in the simulation directory
                ahf_path = os.sep.join(
                    self.snapdir.split(os.sep)[:-1] ## remove output from the snapdir
                    +['halo','ahf']) ## look in <simdir>/halo/ahf

                ## okay no smooth halo file but there is a halo/ahf directory at least
                if (os.path.isdir(ahf_path) and (
                    not os.path.isfile(os.path.join(ahf_path,ahf_fname)))):

                    ## let's scan through the files in the directory and try and 
                    ##  find an AHF halo file that corresponds to just this snapshot. 
                    fnames = []
                    snap_string = "%03d"%self.snapnum

                    for fname in os.listdir(ahf_path):
                        if (snap_string in fname and 
                            'AHF_halos' in fname):
                            fnames+=[fname]
                    if len(fnames) == 1:
                        fname = fnames[0]
                    else:
                        raise IOError("can't find a halo file (or found too many)",fnames)
            else:
                raise IOError("can't find a halo file with",ahf_path,'and',ahf_name)
                

            self.ahf_path = ahf_path
            self.ahf_fname = ahf_fname

            ## now that we've attempted to identify an AHF file lets open 
            ##  this puppy up
            try:
                self.scom, self.rvir, self.rstar_half = cosmo_utils.load_AHF(
                    self.snapdir,
                    self.snapnum,
                    self.current_redshift,
                    ahf_path = ahf_path,
                    fname=ahf_fname,
                    hubble = self.header['HubbleParam'])

            ## TODO make sure there's some proper error handling in cosmo_utils.load_AHF
            ## no rstar 1/2 in this AHF file, we'll have to calculate it ourselves in our first extraction
            except ValueError:
                self.scom, self.rvir = cosmo_utils.load_AHF(
                    self.snapdir,
                    self.snapnum,
                    self.current_redshift,
                    extra_names_to_read = [],
                    ahf_path = ahf_path,
                    fname=ahf_fname,
                    hubble = self.header['HubbleParam'])

                self.rstar_half = None
                ## have we already calculated it and cached it?
                for attr in ['gas_extract_rstar_half','star_extract_rstar_half']:
                    if hasattr(self.metadata,attr):
                        print('using cached',attr,'for rstar_half')
                        self.rstar_half = getattr(self.metadata,attr)
                        break

                ## I guess not
                if self.rstar_half is None:
                    print("No rstar 1/2 in AHF or metadata files, we will need to calculate it ourselves.")

        ## determine what the final snapshot of this simulation is
        ##  by checking the snapdir and sorting the files by snapnum
        self.finsnap = all_utils.getfinsnapnum(self.snapdir)

    def createSnapshotTimes(self,snaptimes='snapshot_times'):

        ## try looking in the simulation directory for a snapshot_times.txt file
        snap_FIRE_SN_times = os.path.join(self.snapdir,'..','%s.txt'%snaptimes)
        ## try loading from the datadir
        data_FIRE_SN_times = os.path.join(self.datadir,'%s.txt'%snaptimes)

        for pathh in [
            snap_FIRE_SN_times,
            data_FIRE_SN_times]:

            if os.path.isfile(pathh):
                (self.snapnums,
                    self.sfs,
                    self.zs,
                    self.gyrs,
                    self.dTs) = np.genfromtxt(pathh,unpack=1)
                return
        
        ## if we didn't manage to find an existing snapshot times file
        ##  we'll try and make one ourself using the available snapshots
        ##  in the snapdir
        finsnap = all_utils.getfinsnapnum(self.snapdir)
        print("Oh boy, have to open %d files to output their snapshot timings"%finsnap)

        snapnums = []
        sfs = []
        gyrs = []
        zs = []
        dTs = [0]

        for snapnum in range(finsnap+1):
            try:
                ## open the snapshot, but only load the header info
                header_snap = openSnapshot(
                    self.snapdir,
                    snapnum,
                    0, ## dummy particle index, ignored for header_only = True
                    header_only=True,
                    snapdir_name=self.snapdir_name,
                    cosmological=True)

                ## save each one to a list
                snapnums += [snapnum]
                sfs += [header_snap['ScaleFactor']]
                gyrs += [header_snap['TimeGyr']]
                zs += [header_snap['Redshift']]

                if len(gyrs) >= 2 :
                    dTs += [gyrs[-1]-gyrs[-2]]

            ## this snapshot doesn't exist, but maybe there are others
            except IOError:
                print("snapshot %d doesn't exist, skipping..."%snapnum)
                continue

        ## write out our snapshot_times.txt to datadir
        np.savetxt(
            data_FIRE_SN_times,
            np.array([snapnums,sfs,zs,gyrs,dTs]).T,
            fmt = ['%d','%.6f','%.6f','%.6f','%.6f'])

        (self.snapnums,
            self.sfs,
            self.zs,
            self.gyrs,
            self.dTs) = np.genfromtxt(data_FIRE_SN_times,unpack=1)

    ## I keep a single hdf5 file that stores header information
    ##  for every simulation I have opened with a Galaxy class
    def loadHeaderFromCatalog(self,catalog_name = 'FIRE_headers.hdf5'):
        header = {}

        master_datadir = os.path.split(self.datadir)[0]
        with h5py.File(os.path.join(master_datadir,catalog_name),'r') as handle:
            ## have we saved this simulation to the catalog?
            if self.data_name in handle.keys():
                this_group = handle[self.data_name]
                for key in this_group.keys():
                    ## transfer the keys from the hdf5 file to dictionary
                    header[key] = this_group[key].value
            else:
                raise IOError("This halo is not in the catalog")

        return header

    def saveHeaderToCatalog(
        self,
        catalog_name = 'FIRE_headers.hdf5',
        overwrite=0):

        master_datadir = os.path.split(self.datadir)[0]

        with h5py.File(os.path.join(master_datadir,catalog_name),'a') as handle:
            if self.data_name in handle.keys() and overwrite:
                ## if we want to overwrite, can just delete it and "start from scratch"
                ##  recursively
                del handle[self.data_name]
                return self.saveHeaderToCatalog(catalog_name)
            elif self.data_name not in handle.keys():
                this_group = handle.create_group(self.data_name)
                for key in self.header.keys():
                    ## only take the keys that are constant throughout the simulation
                    if key not in ['Time','Redshift','TimeGyr','ScaleFactor']:
                        val = self.header[key]
                        if key == 'fname':
                            val = [bar.encode() for bar in val]
                            this_group[key] = val
            else:
                raise ValueError("This halo is already in the catalog") 
         
        print("Saved header of %s to %s header table"%(self.data_name,master_datadir))

#### use the parameters set in __init__ to find the main halo, rotate
####  coordinates to lie along gas/star angular momentum axis
####  and optionally output this new "sub-snapshot" to a cache file
    def extractMainHalo(
        self,
        save_meta = False, 
        orient_stars = True, ## which angular momentum axis to orient along
        overwrite_full_snaps_with_rotated_versions = False,
        free_mem = True, ## delete the full snapshot from RAM
        extract_DM = True, ## do we want the DM particles? 
        **kwargs):
        """
        radius = None, cylinder = '',
        use_saved_subsnapshots = True,
        force = False,
        """
        
        if orient_stars:
            group_name = 'star_extract'
        else:
            group_name = 'gas_extract'

        @metadata_cache(
            group_name,
            ['sub_radius',
            'orient_stars',
            'thetay',
            'thetaz',
            'sphere_lz',
            'sphere_ltot',
            'rvir',
            'rstar_half'],
            use_metadata=False,
            save_meta=save_meta,
            loud=1)
        def extract_halo_inner(
            self,
            orient_stars=True,
            radius = None,
            cylinder = '',
            use_saved_subsnapshots = True,
            force = False):

            ## handle default remappings

            if radius is None:
                radius = self.rvir ## radius of the sub-snapshot

                ## manually calcualte rstar half using the star particles
                ##  rather than relying on the output of AHF
                if self.rstar_half is None:
                    self.load_stars()
                    self.rstar_half = self.calculate_rstar_half() 

                ## radius to calculate angular momentum
                ##  to orient on 
                orient_radius = 5*self.rstar_half 

            ## if this is not the first time extract_halo_inner has been 
            ##  called these properties may or may not exist
            subsnap_missing = (
                not hasattr(self,'sub_snap') or
                not hasattr(self,'sub_star_snap') or
                (not hasattr(self,'sub_dark_snap') and extract_DM) or
                force)

            ## initialize which_dark_snap
            which_dark_snap = None

            ## attempt to open up the cached subsnaps 
            ##  (if they exist) and bind them
            fname = os.path.join(
                self.datadir,
                'subsnaps',
                'snapshot_%03d.hdf5'%self.snapnum)
            already_saved = os.path.isfile(fname)


            ## if we've already bound these to our Galaxy instance
            ##  let's go ahead and pass these to the rotation routine
            if not subsnap_missing:
                which_snap = self.sub_snap
                which_star_snap = self.sub_star_snap
                if extract_DM:
                    which_dark_snap = self.sub_dark_snap

            ## if the relevant sub_snap(s) are not bound to the
            ##  Galaxy instance we need to try and load
            ##  them from disk
            else:
                try:
                    assert use_saved_subsnapshots
                    print("Using the saved sub-snapshots")

                    self.sub_snap = openSnapshot(
                        None,None,
                        0, ## gas index
                        fnames = [fname], ## directly load from a specific file
                        cosmological=True,
                        abg_subsnap=1)

                    self.sub_star_snap = openSnapshot(
                        None,None,
                        4, ## star index
                        fnames = [fname], ## directly load from a specific file
                        cosmological=True,
                        abg_subsnap=1)

                    self.sub_dark_snap = openSnapshot(
                        None,None,
                        1, ## high-res DM index
                        fnames = [fname], ## directly load from a specific file
                        cosmological=True,
                        abg_subsnap=1)

                    ## check if the extraction radius is what we want, to 5 decimal places
                    if np.round(radius,5) != np.round(self.sub_snap['scale_radius'],5):
                        raise ValueError("scale_radius is not the same",
                            radius,
                            self.sub_snap['scale_radius'])

                    ## check if halo center is the same to 5 decimal places
                    if (np.round(self.scom,5) != np.round(self.sub_snap['scom'],5)).any():
                        raise ValueError("Halo center is not the same",
                            self.scom ,self.sub_snap['scom'])

                    print("Successfully loaded a pre-extracted subsnap")

                    ## pass these sub-snapshots into the rotation routine
                    which_snap = self.sub_snap
                    which_star_snap = self.sub_star_snap
                    if extract_DM:
                        which_dark_snap = self.sub_dark_snap

                except (AttributeError,AssertionError,ValueError,IOError) as error:
                    message = "Failed to open saved sub-snapshots"
                    message+= ' %s'%error.__class__  
                    if hasattr(error,'message'):
                        message+= ' %s'%error.message
                    print(message)

                    ## have to load the full snapshots...
                    if 'snap' not in self.__dict__:
                        self.load_gas()
                    if 'star_snap' not in self.__dict__:
                        self.load_stars()
                    if (extract_DM and 
                        'dark_snap' not in self.__dict__):
                        self.load_dark_matter()

                    ## pass these full snapshots into the rotation routine
                    which_snap = self.snap
                    which_star_snap = self.star_snap
                    if extract_DM:
                        which_dark_snap = self.dark_snap
                

            ## pass the snapshots into the rotation routine
            sub_snaps = diskFilterDictionary(
                which_star_snap,
                which_snap,
                radius, ## radius to extract particles within
                orient_radius, ## radius to orient on
                scom=self.scom,
                dark_snap=which_dark_snap, ## dark_snap = None will ignore dark matter particles
                orient_stars=orient_stars)

            ## unpack the return value
            if not extract_DM:
                 self.sub_snap,self.sub_star_snap = sub_snaps 
            else:
                (self.sub_snap,
                self.sub_star_snap,
                self.sub_dark_snap) = sub_snaps

            ## bind the radius
            self.sub_radius = orient_radius
            ## denote whether the most recent extraction was 
            ##  oriented on stars or gas
            self.orient_stars = orient_stars

            ## save for later, if requested
            if (use_saved_subsnapshots and
                extract_DM and
                not already_saved):
                self.outputSubsnapshot()

            ## read out the angular momentum from the sub_snap
            ##  to return it below
            if orient_stars:
                lz = self.sub_star_snap['star_lz']
                ltot = self.sub_star_snap['star_ltot']
            else:
                lz = self.sub_snap['lz']
                ltot = self.sub_snap['ltot']

            return (self.sub_radius,
                self.orient_stars,
                self.sub_snap['thetay'],
                self.sub_snap['thetaz'],
                lz,
                ltot,
                self.rvir,
                self.rstar_half)

        return_value = extract_halo_inner(
            self,
            orient_stars=orient_stars,
            **kwargs)

        ## this should happen by default when you do an extraction but if you are 
        ##  loading from a cached sub-snap
        if overwrite_full_snaps_with_rotated_versions:
            self.overwrite_full_snaps_with_rotated_versions(extract_DM)

        if free_mem: 
            if 'dark_snap' in self.__dict__.keys():
                del self.dark_snap

            ## delete dark matter extraction if not necessary
            if 'sub_dark_snap' in self.__dict__.keys() and not extract_DM:
                del self.sub_dark_snap

            if 'star_snap' in self.__dict__.keys():
                del self.star_snap

            if 'snap' in self.__dict__.keys():
                del self.snap
            print("Snapshot memory free")

        return return_value

    def load_stars(self,**kwargs):
        print("Loading star particles of",self)
        self.star_snap = openSnapshot(
            self.snapdir,
            self.snapnum,4,
            cosmological=True,
            snapdir_name=self.snapdir_name,
            **kwargs)

    def load_gas(self,**kwargs):
        print("Loading gas particles of",self)
        self.snap = openSnapshot(
            self.snapdir,
            self.snapnum,0,
            cosmological=True,
            snapdir_name=self.snapdir_name,
            **kwargs)

    def load_dark_matter(self,**kwargs):
        print("Loading dark matter particles of",self)
        self.dark_snap = openSnapshot(
            self.snapdir,self.snapnum,1,
            cosmological=True,
            snapdir_name=self.snapdir_name,
            **kwargs)

    def calculate_rstar_half(self):
        print("Calculating the stellar half mass radius")

        ## find the stars within the virial radius
        coords = self.star_snap['Coordinates']-self.scom
        radii = np.sum(coords**2,axis=1)**0.5
        halo_indices = radii < self.rvir
        
        edges = np.linspace(0,self.rvir/2,5000,endpoint=True)
        h,edges = np.histogram(radii[halo_indices],bins=edges,weights = self.star_snap['Masses'][halo_indices])
        h/=1.0*np.sum(h)
        cdf = np.cumsum(h)

        return all_utils.findIntersection(edges[1:],cdf,0.5)[0]

    ## load and save things to object
    def outputSubsnapshot(
        self,
        ptypes = [0,1,4],
        new_extraction = 0):

        if new_extraction:
            self.extractMainHalo()

        ## make the subsnap directory if necessary
        subsnapdir = os.path.join(self.datadir,'subsnaps')
        if not os.path.isdir(subsnapdir):
            os.mkdir(subsnapdir)


        ## get keys directly from the parent snapshot:
        with h5py.File(self.snap['fnames'][0],'r') as handle:
            header_keys = list(handle['Header'].attrs.keys())
            pkeyss = [list(handle['PartType%d'%ptype].keys()) for ptype in ptypes]

        outpath = os.path.join(subsnapdir,'snapshot_%03d.hdf5'%self.snapnum)
        extra_keyss = []
        with h5py.File(outpath,'w') as handle:
            ## make header
            header = handle.create_group('Header').attrs

            ## count the particles of each type to have an accurate numpart total in the header
            numparts = np.array([0 for i in range(5)])

            for ptype in ptypes:
                numparts[ptype]+=getattr(self,sub_snap_dict[ptype])['Masses'].shape[0]
                
            for header_key in header_keys:
                if header_key == 'NumPart_ThisFile' or header_key == 'NumPart_Total':
                    header[header_key] = numparts
                else:
                    header[header_key] = self.sub_snap[header_key]

            ## save each of the particle typos info
            for ptype,pkeys in zip(ptypes,pkeyss):
                pgroup = handle.create_group('PartType%d'%ptype)
                this_sub_snap = getattr(self,sub_snap_dict[ptype])
                extra_keyss+=[set(this_sub_snap.keys()) - set(pkeys) - set(header_keys)]
                for pkey in pkeys:
                    ## divide by the unit fact, since we want it to be in the FIRE units
                    unit_fact = get_unit_conversion(this_sub_snap,pkey,True)
                    pgroup[pkey] = this_sub_snap[pkey]/unit_fact

            ## handle a bunch of set operations that will find the stuff i added
            ##  that wasn't in the original snapshot, and figure out where it has to live
            common_keys = extra_keyss[0]
            for extra_keys in extra_keyss[1:]:
                common_keys = common_keys & extra_keys

            ## put the common keys into ABG_Header
            ABG_Header = handle.create_group("ABG_Header")
            for key in common_keys:
                ABG_Header.attrs[key]=self.sub_snap[key]

            derived_arrays = set(['Temperature','AgeGyr'])
            for ptype,extra_keys in zip(ptypes,extra_keyss):
                abg_pgroup = ABG_Header.create_group('PartType%d'%ptype)
                this_sub_snap = getattr(self,sub_snap_dict[ptype])
                for key in (extra_keys-common_keys-derived_arrays):
                    abg_pgroup[key]=this_sub_snap[key]
        print('Finished, output sub-snapshot to:',outpath)
        
    def overwrite_full_snaps_with_rotated_versions(self,extractDM):
        ## which snaps to offset and rotate?
        snaps = [self.snap,self.star_snap] 
        if extract_DM:
            snaps += [self.dark_snap]

        ## get the extraction parameters
        thetay,thetaz,scom,vscom,orient_stars = (
            self.sub_snap['thetay'],self.sub_snap['thetaz'],
            self.sub_snap['scom'],self.sub_snap['vscom'],
            self.sub_snap['orient_stars'])

        ## if snaps doesn't have dark snap then zipped will stop at [0,4]
        for ptype,snap in zip([0,4,1],snaps):
            if 'overwritten' not in snap or not snap['overwritten']:
                ## set snap to new snapdict that has offset/rotated coords
                snap = offsetRotateSnapshot(
                    snap,
                    scom,vscom,
                    thetay,thetaz,
                    orient_stars)

                ## snapdict holds ptype -> "snap"/"star_snap"/"dark_snap"
                setattr(self,snap_dict[ptype],snap)

    def outputIndexCache(
        self,
        ptypes = [0,1,4],
        new_extraction = 0,
        heavy_data = False):
        """use h5py fancy indexing to load only a subset of the data
            this actually takes forever and should never be used..."""

        if new_extraction:
            self.extractMainHalo()

        ## make the subsnap directory if necessary
        subsnapdir = os.path.join(self.datadir,'subsnaps')
        if not os.path.isdir(subsnapdir):
            os.mkdir(subsnapdir)

        ## do we want to just produce a carbon copy of the subsnapshot or an
        ##  index cache?
        if heavy_data:
            return self.outputSubsnapshot(ptypes)

        outpath = os.path.join(subsnapdir,'index_cache_%03d.hdf5'%self.snapnum)

        ## point to the snapdir
        with h5py.File(outpath,'w') as sub_handle:
            header = sub_handle.create_group('Header')
            header.attrs['HeavyDataPath'] = os.path.split(self.snap['fnames'][0])[0]

        indices = {}
        for ptype in ptypes:
            ## find the extracted sub snapshot
            this_sub_snap = getattr(self,sub_snap_dict[ptype])

            for h5file in self.snap['fnames']:
                ## the group in the file will be named snapshot_600.0, etc... 
                group_name = os.path.split(h5file)[1].split('.hdf5')[0]

                with h5py.File(h5file,'r') as full_handle:
                    ## load the particle ids for this snapshot and this particle type
                    full_snap_ids = full_handle['PartType%d'%ptype]['ParticleIDs'].value

                    this_file_indices = np.isin(
                                full_snap_ids,
                                self.sub_snap['ParticleIDs'])

                    with h5py.File(outpath,'a') as sub_handle:
                        ## create the group if we need to
                        if group_name not in sub_handle.keys():
                            group = sub_handle.create_group(group_name)
                        else:
                            group = sub_handle[group_name]

                        ## save the indices
                        group['ThisSnapMask_PartType%d'%ptype] = this_file_indices 

    def get_tex_galaxy_table_row(
        self,
        disk_scale_height=None,
        disk_scale_radius=None,
        SFH_tavg=0.6,
        table=None,
        ):
        """disk_scale_height = None -> spherical mask
            disk_scale_radius = None -> 5 rstar_half"""

        ## initialize disk_scale_radius if necessary
        disk_scale_radius = 5*self.rstar_half if disk_scale_radius is None else disk_scale_radius

        ## intialize this row with the header for a table. can avoid this by passing table = ""
        if table is None:
            table = r"""## name & $M_\mathrm{halo}$ (M$_\odot$) \tnote{a} & $R_{*,1/2}$ (kpc) \tnote{b}& $M_*$ (M$_\odot$) \tnote{c} & $f_g$ \tnote{d} & $\langle SFR \rangle_\mathrm{600\,Myr}$ (M$_\odot$ yr$^{-1}$) \tnote{e}\\ \hline
    """
        ## open the particle data
        self.extractMainHalo()

        ## find the mask corresponding to the galactic disk
        scoords = self.sub_star_snap['Coordinates']
        coords = self.sub_snap['Coordinates']
 
        masks = []
        for this_coords in [coords,scoords]:
            mask = np.sum(this_coords**2,axis=1)**0.5 < disk_scale_radius

            if disk_scale_height is not None:
                zmask = np.abs(this_coords[:,-1]) <= disk_scale_height
                mask = np.logical_and(mask,zmask)

            masks+=[mask]

        ## apply mask to get only particles within the disk
        disk_masses = self.sub_snap['Masses'][masks[0]]
        disk_smasses = self.sub_star_snap['Masses'][masks[1]]
        
        mstar = disk_smasses.sum()
        mgas = disk_masses.sum()
        
        ## load the star formation history to find recent SFR
        self.get_SFH()
        last_time = self.SFH_time_edges[-1]
        avg_sfr = np.mean(self.sfrs[(last_time-self.SFH_time_edges[1:])<=SFH_tavg])
        
        print(self,end='\n----\n')
        print('mh: %.1e'%(self.sub_dark_snap['Masses'].sum()*1e10))
        print('rstarhalf: %.2f'%self.rstar_half)
        print('m*: %.1e'%(mstar*1e10))
        print('fg: %.2f'%(mgas/(mstar+mgas)))
        print('sfr: %.2f'%avg_sfr)
        print("----")

        table+=r"%s & %.1e & %.1f & %.1e & %.2f & %.2f\\"%(
            self.pretty_name,
            self.sub_dark_snap['Masses'].sum()*1e10,
            self.rstar_half,
            (mstar*1e10),
            (mgas/(mstar+mgas)),
            avg_sfr
        )
        table+='\n'
        table = table.replace('+','')

        return table
    
###### MANY GALAXY FUNCTIONS
class ManyGalaxy(Galaxy):
    def __repr__(self):
        return "%s many-galaxy wrapper"%(self.name)

    def __init__(
        self,
        name,
        snapdir,
        datadir=None,
        data_name=None,
        load_snapnums=None,
        population_kwargs=None,
        **galaxy_kwargs):
        """ a wrapper that will allow one to open multiple galaxies at the same time,
            most useful for creating and accessing MultiMetadata instances while 
            using the same plotting scripts that a Galaxy instance would work for 
            (in general, this must be done consciously while making a plotting script). """

        ## snapdir is sometimes None if an instance is 
        ##  created just to access the metadata and cached
        ##  methods
        if snapdir is not None and snapdir[-1]==os.sep:
            snapdir = snapdir[:-1]

        ## bind input
        self.snapdir = snapdir

        self.name = name
        self.data_name = self.name if data_name is None else data_name

        ## append _md to the data_name for my own sanity
        if 'metal_diffusion' in self.snapdir and '_md' not in self.data_name:
            self.data_name = self.data_name + '_md'

        ## name that should appear on plots
        ##  i.e. remove the resXXX from the name
        pretty_name = self.name.split('_')
        pretty_name = [
            strr if 'res' not in strr else '' 
            for strr in pretty_name] 
        self.pretty_name = '_'.join(pretty_name)
        self.pretty_name = self.pretty_name.replace('__','_')

        ## TODO??
        self.snapdir_name = 'snapdir' if (
            'angles' in self.snapdir or 
            '_md' in self.name 
            and 'm11' not in self.name) else ''

        ## handle datadir creation
        if datadir is None:
            self.datadir = "/home/abg6257/projects/data/"
        else:
            self.datadir = datadir

        ## make top level datadir if it doesn't exist...
        if not os.path.isdir(self.datadir):
            os.mkdir(self.datadir)

        if name not in datadir and name!='temp':
            self.datadir = os.path.join(self.datadir,self.data_name)

        if not os.path.isdir(self.datadir):
            os.mkdir(self.datadir)

        ## handle metadatadir creation
        self.metadatadir = os.path.join(self.datadir,'metadata')
        if not os.path.isdir(self.metadatadir):
            os.mkdir(self.metadatadir)

        ## allow a MultiGalaxy wrapper to open histories files
        if self.snapdir is not None:
            self.finsnap = all_utils.getfinsnapnum(self.snapdir)
            ## get the minimum snapnum
            self.minsnap = all_utils.getfinsnapnum(self.snapdir,True)
        else:
            ## filler values
            self.finsnap=self.minsnap=None

        ## for anything else you'd like to pass to future loaded galaxies
        self.galaxy_kwargs = galaxy_kwargs


        if population_kwargs is not None:
            load_snapnums = self.find_galaxy_population(**population_kwargs)

        ## if i was given snapshots to open, let's create an object that opens the 
        ##  Galaxy instances
        if load_snapnums is not None:
            self.galaxies = [self.loadAtSnapshot(snapnum) for snapnum in load_snapnums]
            self.metadata = MultiMetadata(
                load_snapnums,
                self.galaxies,
                os.path.join(self.datadir,'metadata'))

            ## define convenient access patterns
            def __getattr__(self,attr):
                return np.array(
                    [getattr(gal,attr) for gal 
                    in self.galaxies])

            def __getitem__(self,index):
                return self.galaxies[index]

            setattr(self,'__getattr__',__getattr__)
            setattr(self,'__getitem__',__getitem__)

    def find_galaxy_population(
        self,
        N=10,
        cursnap=600,
        objects=False, ## do we want to return functioning Galaxy instances or just their snapnums?
        DTMyr=50):
        """ Finds evenly spaced snapshots by iteratively loading snapshots in reverse"""

        ## initialize the search at our current snapshot
        found = 1
        return_list = [self.loadAtSnapshot(cursnap)]
        cur_time = return_list[0].current_time_Gyr
        new_snap = cursnap-1
        while found < N:
            new_galaxy = self.loadAtSnapshot(new_snap)
            if (cur_time - new_galaxy.current_time_Gyr)*1000 > DTMyr:
                cur_time = new_galaxy.current_time_Gyr
                return_list +=[new_galaxy]
                found+=1
            new_snap-=1
        if not objects:
            ## return only the snapnums
            return [galaxy.snapnum for galaxy in return_list[::-1]]
        else: 
            return return_list[::-1]

    def loadAtSnapshot(self,snapnum,**kwargs):
        """ Create a Galaxy instance at snapnum using the stored 
            galaxy kwargs (and any additional ones passed). 

            Options are:
                plot_color=0,
                loud_metadata=1,
                multi_thread=1,
                ahf_path=None,
                ahf_fname=None,
            """

        new_kwargs = {}
        new_kwargs.update(self.galaxy_kwargs)
        new_kwargs.update(kwargs)

        return Galaxy(
            self.name,
            self.snapdir,
            snapnum,
            datadir=self.datadir,
            data_name=self.data_name,
            **new_kwargs)
