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

from abg_python.galaxy.cosmoExtractor import extractDiskFromSnapdicts,offsetRotateSnapshot
from abg_python.galaxy.movie_utils import Draw_helper,FIREstudio_helper
from abg_python.galaxy.sfr_utils import SFR_helper
from abg_python.galaxy.metadata_utils import metadata_cache,Metadata,MultiMetadata
from abg_python.galaxy.firefly_utils import Firefly_helper

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

h_official_names = {'h206':'A1','h29':'A2','h113':'A4','h2':'A8'}

elvis_partners = {
    'Romeo':('Juliet',0),
    'Juliet':('Romeo',1),

    'Romulus':('Remus',0),
    'Remus':('Romulus',1),

    'Thelma':('Louise',0),
    'Louise':('Thelma',1)}

def get_elvis_snapdir_name(savename):

    print(savename)
    this_name = None

    ## check each of the partners and see if they're in the savename
    ##  savename should be of the form ex.: Romeo_res3500
    for partner in elvis_partners.keys():
        if partner in savename:
            ## which one do we want and what resolution
            this,resolution = savename.split('_')
            ## get the matching partner and figure which_host we are
            that,which_host = elvis_partners[this]

            ## ensure RomeoJuliet not JulietRomeo
            if which_host == 0:
                this_name = this+that
            else:
                this_name = that+this

            ## format the final name
            this_name = 'm12_elvis_%s_%s'%(this_name,resolution)

    ## if we didn't do anything in the loop above, just return the savename
    this_name = savename if this_name is None else this_name

    return this_name

## function to determine what the "main" halo is
def halo_id(name):
    return 0 

## bread and butter galaxy class
class Galaxy(
    Firefly_helper,
    Draw_helper,
    FIREstudio_helper,
    SFR_helper):
    """------- Galaxy
        Input:
            name - name of the simulation directory
            snapdir - location that the snapshots live in, should end in "output"
            snapnum - snapshot number
            datadir = None - directory where any new files are saved to
            datadir_name = None - name of the data directory, if different than name
            plot_color = 0 - color this instance will use when plotting onto an axis
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
        return "%s at %d" % (self.name,self.snapnum)# + self.__dict__.keys()

    ## convenience function to add this simulation to a legend
    ##  with its name and plot_color
    def add_to_legend(
        self,
        *args,
        **kwargs):
            
        kwargs['label'] = self.name
        kwargs['color'] = self.plot_color
        add_to_legend(*args,**kwargs)

    ## GALAXY
    def __init__(
        self,
        name,
        snapdir,
        snapnum,
        datadir=None,
        datadir_name = None,
        snapdir_name = None,
        plot_color = 0,
        multi_thread = 1,
        ahf_path = None,
        ahf_fname = None,
        save_header_to_table = True,
        meta_name = None,
        suite_name = None,
        **metadata_kwargs 
        ):

        if meta_name is None:
            meta_name = 'meta_Galaxy'

        if suite_name is None:
            if 'metal_diffusion' in snapdir:
                self.suite_name = 'metal_diffusion'
            elif 'HL000' in snapdir:
                self.suite_name = 'xiangcheng'
            elif 'core' in snapdir:
                self.suite_name = 'core'
            elif 'cr_heating_fix' in snapdir:
                self.suite_name = 'cr_heating_fix'
            else: ## set myself up for failure below
                self.suite_name = 'unknown'
        else:
            self.suite_name = suite_name

        ## bind input
        self.snapnum = snapnum
        self.multi_thread = multi_thread
        self.name = name

        ## snapdir is sometimes None if an instance is 
        ##  created just to access the metadata and cached
        ##  methods
        if snapdir is not None and snapdir[-1]==os.sep:
            snapdir = snapdir[:-1]
        self.snapdir = snapdir

        self.datadir_name = self.name if datadir_name is None else datadir_name
        self.snapdir_name = self.datadir_name if snapdir_name is None else snapdir_name

        ## append _md to the name for my own sanity
        #if '_md' not in self.name and 'metal_diffusion' in self.snapdir:
            #self.name = self.name + '_md'

        ## name that should appear on plots
        ##  i.e. remove the resXXX from the name
        pretty_name = self.name.split('_')
        pretty_name = np.array([
            strr if 'res' not in strr else '' 
            for strr in pretty_name])
        pretty_name = pretty_name[pretty_name!= '']
        self.pretty_name = '_'.join(pretty_name)
        self.pretty_name = self.pretty_name.replace('__','_')

        if self.pretty_name in ['h2','h206','h29','h113']:
            self.pretty_name += '-%s'%h_official_names[self.pretty_name]

        if type(plot_color) is int:
            try:
                plot_color=get_distinct(9)[plot_color] # this is a dumb way to do this
            except:
                plot_color="C%d"%((plot_color-9)%13)

        self.plot_color = plot_color
        
        ## handle datadir creation
        if datadir is None:
            self.datadir = os.environ['HOME']+"/scratch/data/%s"%self.suite_name
        else:
            self.datadir = datadir

        ## make top level datadir if it doesn't exist...
        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir)

        if name not in self.datadir and name!='temp':
            self.datadir = os.path.join(self.datadir,self.datadir_name)

        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir)

        ## handle metadatadir creation
        self.metadatadir = os.path.join(self.datadir,'metadata')
        if not os.path.isdir(self.metadatadir):
            os.makedirs(self.metadatadir)

        ## handle plotdir creation
        self.plotdir = os.path.join(self.datadir,'plots')
        if not os.path.isdir(self.plotdir):
            os.makedirs(self.plotdir)

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
                '%s_%03d.hdf5'%(meta_name,self.snapnum))
            self.metadata = Metadata(
                self.metapath,
                **metadata_kwargs)

            ## attempt to open the header in the hdf5 file
            try:
                self.header = openSnapshot(
                    self.snapdir,
                    self.snapnum,
                    0, ## dummy particle index, not used if header_only is True
                    header_only=True)

                ## save the header to our catalog of simulation headers
                ##  so that we can open the header in scenarios where
                ##  snapnum is None, as above.
                if save_header_to_table:
                    try:
                        self.saveHeaderToCatalog()
                    except (ValueError,OSError):
                        ## OSError is when parallel processes try to write to header at the same time...
                        ##  Value Error is probably when the simulation is already in the file?
                        pass


            except IOError as e:
                print("Couldn't find header.")
                print(e) 
                return 

            ## simulation timing
            ##  load snapshot times to convert between snapnum and time_Gyr
            self.current_redshift = self.header['Redshift']
            self.current_time_Gyr = self.header['TimeGyr']

        
        if self.header['cosmological']:
            ## opens the halo file to find the halo center and virial radius
            self.load_halo_file()
        else:
            self.scom = np.zeros(3)
            self.rvir = 300 ## what should i do here...
            self.rstar_half = None

        ## have we already calculated it and cached it?
        if self.rstar_half is None:
            for attr in ['gas_extract_rstar_half','star_extract_rstar_half']:
                if hasattr(self.metadata,attr):
                    self.rstar_half = getattr(self.metadata,attr)
                    break

            ## I guess not
            if self.rstar_half is None:
                print("No rstar 1/2 in halo or metadata files, we will need to calculate it ourselves.")

        ## determine what the final snapshot of this simulation is
        ##  by checking the snapdir and sorting the files by snapnum
        self.finsnap = all_utils.getfinsnapnum(self.snapdir)

    def load_halo_file(self,halo_fname=None,halo_path=None):

        if halo_path == 'None' or halo_fname == 'None':
            self.scom,self.rvir,self.rstar_half = None,None,None
            print(
                'Make sure to set:',
                'scom',
                'rvir',
                'rstar_half',
                'attributes manually')
        else:
            try:
                if 'elvis' in self.snapdir:
                    raise IOError("No AHF files for Elvis runs")
                self.load_ahf(ahf_fname=halo_fname,ahf_path=halo_path)
            except IOError:
                try:
                    
                    if 'elvis' in self.snapdir:
                        which_host = self.snapdir.split('m12_elvis_')[1].split('_')[0]
                        if which_host[:len(self.pretty_name)] == self.pretty_name:
                            which_host = 0 
                        else:
                            if which_host[-len(self.pretty_name):] != self.pretty_name:
                                raise IOError("invalid name, should be one of %s"%which_host)
                            which_host = 1
                    else:
                        which_host = 0

                    self.load_rockstar(
                        rockstar_fname=halo_fname,
                        rockstar_path=halo_path,
                        which_host=which_host)
                except IOError:
                    print("Couldn't find AHF nor Rockstar halo files")
                    raise

    def load_ahf(self,**kwargs):

        ## automatically search for the ahf_path and ahf_fname
        self.ahf_path, self.ahf_fname = self.auto_search_ahf(**kwargs)
    
        ## now that we've attempted to identify an AHF file lets open 
        ##  this puppy up
        try:
            ## first try and read the stellar half-mass radius (default args)
            self.scom, self.rvir, self.rstar_half = cosmo_utils.load_AHF(
                self.snapdir,
                self.snapnum,
                self.current_redshift,
                ahf_path = self.ahf_path,
                fname = self.ahf_fname,
                hubble = self.header['HubbleParam'])

        except ValueError:
            ## no rstar 1/2 in this AHF file, we'll have to calculate it ourselves 
            self.scom, self.rvir = cosmo_utils.load_AHF(
                self.snapdir,
                self.snapnum,
                self.current_redshift,
                extra_names_to_read = [],
                ahf_path = self.ahf_path,
                fname = self.ahf_fname,
                hubble = self.header['HubbleParam'])

            self.rstar_half = None

        return self.scom,self.rvir,self.rstar_half
            
    def auto_search_ahf(self,ahf_fname=None,ahf_path=None):
        ## attempt to read halo location and properties from AHF
        if ahf_fname is None:
            ahf_fname='halo_0000%d_smooth.dat'%halo_id(self.snapdir_name)

        if ahf_path is None:
            ## system blind if you put the soft link in
            ahf_path = os.path.join(
                os.environ['HOME'],'halo_files',
                "%s","%s")

            ahf_path = ahf_path%(self.suite_name,self.snapdir_name)

        ## check if this first guess at the ahf_fname and ahf_path
        ##  is right
        if (not os.path.isfile(os.path.join(ahf_path,ahf_fname)) and 
            ahf_path != 'None' and 
            ahf_fname != 'None'):
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

        return ahf_path, ahf_fname

    def load_rockstar(self,rockstar_fname=None,rockstar_path=None,which_host=0):

        self.scom,self.rvir = cosmo_utils.load_rockstar(
            self.snapdir,
            self.snapnum,
            fname=rockstar_fname,
            rockstar_path=rockstar_path,
            which_host=which_host)
        self.rstar_half=None

        return self.scom,self.rvir,self.rstar_half

    def get_snapshotTimes(self,snaptimes='snapshot_times',assert_cached=False): 
        ## try looking in the simulation directory for a snapshot_times.txt file
        #snap_FIRE_SN_times = os.path.join(self.snapdir,'..','%s.txt'%snaptimes)
        ## try loading from the datadir

        #for pathh in [
            #snap_FIRE_SN_times,
            #data_FIRE_SN_times]:

        ## TODO
        ## ignore the snapshot times that are saved in the FIRE
        ##  directory because I can't re-derive them :\
        ##  using the stated cosmology and convertStellarAges
        ##  better to consistently rederive age using stated cosmology
        ##  and the provided redshift, as is done in openSnapshot
        
        ## if we didn't manage to find an existing snapshot times file
        ##  we'll try and make one ourself using the available snapshots
        ##  in the snapdir

        data_FIRE_SN_times = os.path.join(self.datadir,'%s.txt'%snaptimes)

        if os.path.isfile(data_FIRE_SN_times):
            (self.snapnums,
                self.snap_sfs,
                self.snap_zs,
                self.snap_gyrs,
                self.dTs) = np.genfromtxt(data_FIRE_SN_times,unpack=1)
            return


        if assert_cached:
            raise AssertionError("User asserted that the snapshot times should be saved to disk.")

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
                    header_only=True)

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
            self.snap_scale_factors,
            self.snap_zs,
            self.snap_gyrs,
            self.dTs) = np.genfromtxt(data_FIRE_SN_times,unpack=1)

    ## I keep a single hdf5 file that stores header information
    ##  for every simulation I have opened with a Galaxy class
    def loadHeaderFromCatalog(self,catalog_name = 'FIRE_headers.hdf5'):
        header = {}

        master_datadir = os.path.split(self.datadir)[0]
        with h5py.File(os.path.join(master_datadir,catalog_name),'r') as handle:
            ## have we saved this simulation to the catalog?
            if self.snapdir_name in handle.keys():
                this_group = handle[self.snapdir_name]
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
            if self.snapdir_name in handle.keys() and overwrite:
                ## if we want to overwrite, can just delete it and "start from scratch"
                ##  recursively
                del handle[self.snapdir_name]
                return self.saveHeaderToCatalog(catalog_name)
            elif self.snapdir_name not in handle.keys():
                this_group = handle.create_group(self.snapdir_name)
                for key in self.header.keys():
                    ## only take the keys that are constant throughout the simulation
                    if key not in ['Time','Redshift','TimeGyr','ScaleFactor']:
                        val = self.header[key]
                        if key == 'fname':
                            val = [bar.encode() for bar in val]
                            this_group[key] = val
            else:
                raise ValueError("This halo is already in the catalog") 
         
        print("Saved header of %s to %s header table"%(self.snapdir_name,master_datadir))

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
        radius = None -- radius of final sub_snap extraction, *not* orient_radius, 
            which is fixed to 5*rstarhalf
        use_saved_subsnapshots = True -- save/load hdf5 cache files. if want to overwrite, set to True
            and force=True
        force = False -- force extraction, despite existing extractions attached to instance or
            in cache files. 
        force_theta_TB = None -- force orientation
        force_phi_TB = None -- force orientation
        """
        
        if orient_stars:
            group_name = 'star_extract'
        else:
            group_name = 'gas_extract'

        @metadata_cache(
            group_name,
            ['sub_radius',
            'orient_stars',
            'theta_TB',
            'phi_TB',
            'rvir',
            'rstar_half',
            'rgas_half'],
            use_metadata=False,
            save_meta=save_meta,
            loud=1)
        def extract_halo_inner(
            self,
            orient_stars=True,
            radius=None,
            use_saved_subsnapshots=True,
            force=False,
            force_theta_TB=None,
            force_phi_TB=None):

            ## handle default remappings

            if self.scom is None:
                self.load_stars()
                self.scom = all_utils.iterativeCoM(
                    self.star_snap['Coordinates'],
                    self.star_snap['Masses'],
                    n=4)

            if radius is None:
                radius = self.rvir ## radius of the sub-snapshot

                ## manually calcualte rstar half using the star particles
                ##  rather than relying on the output of AHF
                if self.rstar_half is None:
                    self.load_stars()
                    self.rstar_half = self.calculate_half_mass_radius() 


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
                ## avoid making multiple subsnaps of the same simulation
                ##  snapshots...
                self.datadir.replace(self.datadir_name,self.snapdir_name),  
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
                    print("Using the saved sub-snapshots for",self)

                    self.sub_snap = openSnapshot(
                        None,None,
                        0, ## gas index
                        fnames = [fname], ## directly load from a specific file
                        abg_subsnap=1)

                    self.sub_star_snap = openSnapshot(
                        None,None,
                        4, ## star index
                        fnames = [fname], ## directly load from a specific file
                        abg_subsnap=1)

                    self.sub_dark_snap = openSnapshot(
                        None,None,
                        1, ## high-res DM index
                        fnames = [fname], ## directly load from a specific file
                        abg_subsnap=1)

                    ## check if the extraction radius is what we want, to 4 decimal places
                    if np.round(radius,4) != np.round(self.sub_snap['scale_radius'],4):
                        ## delete it because it is GARBAGE
                        os.remove(fname)
                        already_saved = False
                        raise ValueError("scale_radius is not the same",
                            radius-self.sub_snap['scale_radius'],
                            radius,self.sub_snap['scale_radius'])

                    ## check if halo center is the same to 4 decimal places
                    if (np.round(self.scom,4) != np.round(self.sub_snap['scom'],4)).any():
                        ## delete it because it is GARBAGE
                        os.remove(fname)
                        already_saved = False
                        raise ValueError("Halo center is not the same",
                            self.scom -self.sub_snap['scom'],
                            self.scom ,self.sub_snap['scom'])

                    print("Successfully loaded a pre-extracted subsnap")

                    ## pass these sub-snapshots into the rotation routine
                    which_snap = self.sub_snap
                    which_star_snap = self.sub_star_snap
                    if extract_DM:
                        which_dark_snap = self.sub_dark_snap

                except (AttributeError,AssertionError,ValueError,IOError,KeyError,TypeError) as error:
                    message = "Failed to open saved sub-snapshots"
                    #message+= ' %s'%error.__class__  
                    message+= ' %s'%repr(error)
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
            sub_snaps = extractDiskFromSnapdicts(
                which_star_snap,
                which_snap,
                radius, ## radius to extract particles within
                orient_radius, ## radius to orient on
                scom=self.scom,
                dark_snap=which_dark_snap, ## dark_snap = None will ignore dark matter particles
                orient_stars=orient_stars,
                force_theta_TB=force_theta_TB,
                force_phi_TB=force_phi_TB)

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

            if not hasattr(self,'rgas_half'):
                self.rgas_half = self.calculate_half_mass_radius(
                    which_snap=self.sub_snap) 

            return (self.sub_radius,
                self.orient_stars,
                self.sub_snap['theta_TB'],
                self.sub_snap['phi_TB'],
                self.rvir,
                self.rstar_half,
                self.rgas_half)

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
        if not hasattr(self,'star_snap'):
            self.star_snap = openSnapshot(
                self.snapdir,
                self.snapnum,4,
                **kwargs)

    def load_gas(self,**kwargs):
        print("Loading gas particles of",self)
        self.snap = openSnapshot(
            self.snapdir,
            self.snapnum,0,
            **kwargs)

    def load_dark_matter(self,**kwargs):
        print("Loading dark matter particles of",self)
        self.dark_snap = openSnapshot(
            self.snapdir,self.snapnum,1,
            **kwargs)

    def calculate_half_mass_radius(
        self,
        which_snap=None,
        geometry='spherical',
        within_radius=None):

        within_radius = self.rvir if within_radius is None else within_radius

        ## find the stars within the virial radius
        if which_snap is None:
            which_snap = self.star_snap

        if 'overwritten' in which_snap.keys() and which_snap['overwritten']:
            coords = which_snap['Coordinates']
        else:
            coords = which_snap['Coordinates']-self.scom

        masses = which_snap['Masses']

        edges = np.linspace(0,within_radius,5000,endpoint=True)

        if geometry in ['cylindrical','scale_height']:
            radii = np.sum(coords[:,:2]**2,axis=1)**0.5
        elif geometry == 'spherical':
            print("Calculating the half mass radius")
            radii = np.sum(coords**2,axis=1)**0.5

        within_mask = radii <= within_radius

        ## let's co-opt this method to calculate a scale height as well
        if geometry == 'scale_height':
            ## take the z-component
            radii = np.abs(coords[:,-1])
            edges = np.linspace(0,10*within_radius,5000,endpoint=True)

        h,edges = np.histogram(
            radii[within_mask],
            bins=edges,
            weights = masses[within_mask])

        h/=1.0*np.sum(h)
        cdf = np.cumsum(h)

        return all_utils.findIntersection(edges[1:],cdf,0.5)[0]
    
    def get_simple_radius_and_height(
        self,
        component='gas',
        save_meta=True,
        use_metadata=True,
        loud=True,
        **kwargs):

        if component not in ['gas','stars']:
            raise ValueError("Invalid component %s must be gas or star."%component)

        group_name = 'SimpleGeom_%s'%component

        @metadata_cache(
            group_name,
            ['%s_simple_r'%component,
            '%s_simple_h'%component],
            use_metadata=use_metadata,
            save_meta=save_meta,
            loud=loud)
        def compute_simple_radius_and_height(self,component):

            if component == 'gas':
                which_snap = self.sub_snap
            elif component == 'stars':
                which_snap = self.sub_star_snap

            ## calculate the cylindrical half-mass radius using mass 
            ##  within 20% virial radius
            radius = self.calculate_half_mass_radius(
                which_snap=which_snap,
                geometry='cylindrical',
                within_radius=0.2*self.rvir)

            ## calculate the half-mass height w/i cylinder
            ##  of radius previously calculated
            height = self.calculate_half_mass_radius(
                which_snap=which_snap,
                geometry='scale_height',
                within_radius=radius)
                
            return radius,height
        return compute_simple_radius_and_height(self,component,**kwargs)

    ## load and save things to object
    def outputSubsnapshot(
        self,
        ptypes = [0,1,4],
        new_extraction = 0):

        if new_extraction:
            self.extractMainHalo()

        ## make the subsnap directory if necessary
        subsnapdir = os.path.join(
            ## avoid making multiple subsnaps of the same simulation
            ##  snapshots...
            self.datadir.replace(self.datadir_name,self.snapdir_name),  
            'subsnaps')
        if not os.path.isdir(subsnapdir):
            os.makedirs(subsnapdir)


        outpath = os.path.join(subsnapdir,'snapshot_%03d.hdf5'%self.snapnum)

        ## if we haven't already created a subsnapshot, read keys from the 
        ##  parent snapshot
        if not os.path.isfile(outpath):
            fname = self.snap['fnames'][0]
        else: 
            ## actually a subsnapshot already exists, let's use that one
            fname = outpath
            
        ## read keys from existing snapshot
        with h5py.File(fname,'r') as handle:
            header_keys = list(handle['Header'].attrs.keys())
            pkeyss = [list(handle['PartType%d'%ptype].keys()) for ptype in ptypes]

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
        
    def overwrite_full_snaps_with_rotated_versions(self,extract_DM):
        ## which snaps to offset and rotate?
        snaps = [self.snap,self.star_snap] 
        if extract_DM:
            snaps += [self.dark_snap]

        ## get the extraction parameters
        theta_TB,phi_TB,scom,vscom,orient_stars = (
            self.sub_snap['theta_TB'],self.sub_snap['phi_TB'],
            self.sub_snap['scom'],self.sub_snap['vscom'],
            self.sub_snap['orient_stars'])

        ## if snaps doesn't have dark snap then zipped will stop at [0,4]
        for ptype,snap in zip([0,4,1],snaps):
            if 'overwritten' not in snap or not snap['overwritten']:
                ## set snap to new snapdict that has offset/rotated coords
                snap = offsetRotateSnapshot(
                    snap,
                    scom,vscom,
                    theta_TB,phi_TB,
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
            os.makedirs(subsnapdir)

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
        avg_sfr = np.mean(self.SFRs[(last_time-self.SFH_time_edges[1:])<=SFH_tavg])
        
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
    """ """

    def __repr__(self):
        return "%s many-galaxy wrapper"%(self.name)

    def __init__(
        self,
        name,
        snapdir,
        datadir=None,
        datadir_name=None,
        snapdir_name=None,
        load_snapnums=None,
        population_kwargs=None,
        name_append='',
        suite_name='metal_diffusion',
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

        self.name = name+name_append
        self.datadir_name = self.name if datadir_name is None else datadir_name
        self.snapdir_name = self.datadir_name if snapdir_name is None else snapdir_name
        self.suite_name = suite_name

        ## save this for any Galaxy instances we create as well
        galaxy_kwargs['suite_name'] = suite_name


        ## append _md to the datadir_name for my own sanity
        #if '_md' not in self.name and 'metal_diffusion' in self.snapdir:
            #self.name = self.name + '_md'

        ## name that should appear on plots
        ##  i.e. remove the resXXX from the name
        pretty_name = self.name.split('_')
        pretty_name = np.array([
            strr if 'res' not in strr else '' 
            for strr in pretty_name])
        pretty_name = pretty_name[pretty_name!= '']
        self.pretty_name = '_'.join(pretty_name)
        self.pretty_name = self.pretty_name.replace('__','_')

        ## handle datadir creation
        if datadir is None:
            self.datadir = os.environ['HOME']+"/scratch/data/%s"%self.suite_name
        else:
            self.datadir = datadir

        ## make top level datadir if it doesn't exist...
        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir)

        if name not in self.datadir and name!='temp':
            self.datadir = os.path.join(self.datadir,self.datadir_name)

        if not os.path.isdir(self.datadir):
            os.makedirs(self.datadir)

        ## handle metadatadir creation
        self.metadatadir = os.path.join(self.datadir,'metadata')
        if not os.path.isdir(self.metadatadir):
            os.makedirs(self.metadatadir)

        ## handle plotdir creation
        self.plotdir = os.path.join(self.datadir,'plots')
        if not os.path.isdir(self.plotdir):
            os.makedirs(self.plotdir)

        ## allow a MultiGalaxy wrapper to open histories files
        if self.snapdir is not None:
            self.finsnap = all_utils.getfinsnapnum(self.snapdir)
            ## get the minimum snapnum
            self.minsnap = all_utils.getfinsnapnum(self.snapdir,True)
        else:
            ## filler values
            self.finsnap=self.minsnap=None

        try:
            self.get_snapshotTimes(assert_cached=True)
        except AssertionError:
            print("No snapshot times, create one manually with a Galaxy object and .get_snapshotTimes.")

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
                if 'galaxies' in self.__dict__.keys():
                    return np.array(
                        [getattr(gal,attr) for gal 
                        in self.galaxies])
                else:
                    return getattr(self,attr) ## might fail but that's what we want

            def __getitem__(self,index):
                return self.galaxies[index]

            setattr(self,'__getattr__',__getattr__)
            setattr(self,'__getitem__',__getitem__)

    def find_galaxy_population(
        self,
        cursnap=600,
        N=10,
        objects=False, ## do we want to return functioning Galaxy instances or just their snapnums?
        DTMyr=50,
        loud=True):
        """ Finds evenly spaced snapshots by iteratively loading snapshots in reverse"""

        found = 1
        return_list = [cursnap]

        def get_time_from_snapnum(snapnum):
            
            if not hasattr(self,'snap_gyrs'):
                gal = self.loadAtSnapshot(cursnap)
                this_time = gal.current_time_Gyr
            else:
                this_time = self.snap_gyrs[self.snapnums==snapnum][0]
            return this_time


        ## initialize the search at our current snapshot

        cur_time = get_time_from_snapnum(cursnap)
        new_snap = cursnap-1

        while found < N:
            new_time = get_time_from_snapnum(new_snap)
            if loud:
                print(new_snap,'%.2f'%((cur_time - new_time)*1000),end='\t')
            if (cur_time - new_time)*1000 > DTMyr:
                cur_time = new_time
                return_list +=[new_snap]
                found+=1
            new_snap-=1

        if not objects:
            ## return only the snapnums
            return return_list[::-1]
        else: 
            return [self.loadAtSnapshot(snapnum) for snapnum in return_list[::-1]]

    def loadAtSnapshot(self,snapnum,**kwargs):
        """ Create a Galaxy instance at snapnum using the stored 
            galaxy kwargs (and any additional ones passed). 

            Options are:
                plot_color=0,
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
            datadir=os.path.dirname(self.datadir),
            datadir_name=self.datadir_name,
            snapdir_name=self.snapdir_name,
            **new_kwargs)
