## from builtin
import numpy as np 
import h5py
import os
import multiprocessing
import itertools
import copy
import time

from scipy.interpolate import interp1d

## from abg_python
from ..snapshot_utils import openSnapshot,get_unit_conversion
from ..plot_utils import add_to_legend
from ..color_utils import get_distinct
from ..array_utils import findIntersection
from ..system_utils import getfinsnapnum, printProgressBar
from ..physics_utils import iterativeCoM
from ..cosmo_utils import load_AHF,load_rockstar,trace_rockstar
from ..smooth_utils import smooth_x_varying_curve
from ..math_utils import add_jhat_coords
from ..cosmo_utils import RydenLookbackTime

from .cosmoExtractor import extractDiskFromSnapdicts,offsetRotateSnapshot
from .movie_utils import Draw_helper,FIREstudio_helper
from .sfr_utils import SFR_helper
from .metadata_utils import metadata_cache,Metadata,MultiMetadata
from .firefly_utils import Firefly_helper
from .scale_height_utils import ScaleHeight_helper

try: from firestudio.utils.stellar_utils.load_stellar_hsml import get_particle_hsml
except: print("FIRE studio is not installed, Galaxy.get_HSML will not work")


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


cr_heating_fixed = [
    'm09_res30',
    'm09_res250',
    'm10q_res30',
    'm10q_res250',
    'm10v_res30 ',
    'm10v_res250',
    'm11b_res2100_no-swb',
    'm11b_res2100_no-swb_v2 ',
    'm11b_res2100',
    'm11b_res2100_no-swb_contaminated',
    'm11e_res7100',
    'm12i_res7100',
    'm12i_res57000',
    'm12f_res7100',
    'm12s_res113000',
    'm12w_res57000']

def get_elvis_snapdir_name(savename):

    this_name = None

    ## check each of the partners and see if they're in the savename
    ##  savename should be of the form ex.: Romeo_res3500
    for partner in elvis_partners.keys():
        if (partner in savename) and (len(savename.split('_')) == 2):
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
    SFR_helper,
    ScaleHeight_helper):
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
            halo_path = None - path to AHF halo files, defaults first to my halo directory
                and snapdir/../halo/ahf if halo second.
            halo_fname = None - name of the AHF file, typically smooth_halo_00000.dat

        Provided functions:

    """ 

    __doc__+= (
        "\n"+Draw_helper.__doc__ + 
        "\n"+FIREstudio_helper.__doc__ +
        "\n"+SFR_helper.__doc__+
        "\n"+ScaleHeight_helper.__doc__)


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
        snapnum,
        suite_name = 'metal_diffusion',
        snapdir=None,
        datadir=None,
        datadir_name=None,
        snapdir_name=None,
        plot_color=0,
        multi_thread=1,
        full_init=True,
        save_header_to_table=True,
        meta_name='meta_Galaxy',
        **metadata_kwargs):


        if suite_name == 'metal_diffusion' and name in cr_heating_fixed:
            suite_name = 'metal_diffusion/cr_heating_fix'

        self.suite_name = suite_name

        if self.suite_name == 'cr_suite':
            if name == 'm12i_res7100':
                name = 'm12i_mass7000_MHDCR_tkFIX/cr_700'
            else:
                name +='/cr_700'
            name=name.replace('cr_700/cr_700','cr_700')
            
        ## bind input
        self.snapnum = snapnum
        self.multi_thread = multi_thread
        self.name = name

        if snapdir is None: snapdir = os.path.join(
            os.environ['HOME'],
            'snaps',
            suite_name,
            name,
            'output')
        elif snapdir[-1]==os.sep: snapdir = snapdir[:-1]

        ## will replace name with name if not an elvis name
        ##  otherwise will replace Romeo_res3500 w/ RomeoJuliet_res3500
        ##  or Juliet_res3500 w/ RomeoJuliet_res3500
        snapdir = snapdir.replace(name,get_elvis_snapdir_name(name))

        self.snapdir = snapdir

        ## determine what the final snapshot of this simulation is
        ##  by checking the snapdir and sorting the files by snapnum
        self.finsnap = getfinsnapnum(self.snapdir)

        self.datadir_name = self.name if datadir_name is None else datadir_name
        self.snapdir_name = self.datadir_name if snapdir_name is None else snapdir_name

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

        if suite_name == 'cr_suite':
            self.pretty_name = self.pretty_name.split('_')[0]+'_cr'

        elif suite_name == 'metal_diffusion/cr_heating_fix':
            #self.pretty_name += '+'
            pass
        elif suite_name == 'fire3_compatability':
            self.pretty_name += '_fire3'

        colors = get_distinct(9)
        if type(plot_color) is int:
            if plot_color < len(colors): plot_color=colors[plot_color] 
            else: plot_color="C%d"%((plot_color-len(colors))%13)

        self.plot_color = plot_color
        
        if datadir is None: self.datadir = os.environ['HOME']+"/scratch/data/%s"%self.suite_name
        else: self.datadir = datadir

        ## make top level datadir if it doesn't exist...
        if not os.path.isdir(self.datadir): os.makedirs(self.datadir)

        if self.name not in self.datadir and self.name!='temp':
            self.datadir = os.path.join(self.datadir,self.datadir_name)

        ## handle datadir creation
        if not os.path.isdir(self.datadir): os.makedirs(self.datadir)

        ## handle metadatadir creation
        self.metadatadir = os.path.join(self.datadir,'metadata')
        if not os.path.isdir(self.metadatadir): os.makedirs(self.metadatadir)

        ## handle plotdir creation
        self.plotdir = os.path.join(self.datadir,'plots')
        if not os.path.isdir(self.plotdir): os.makedirs(self.plotdir)

        ## there is no metadata file for a Galaxy w.o. a snapshot
        if self.snapnum is None: self.metapath = None
        ## open/create the metadata object & file
        else:
            self.metapath = os.path.join(
                self.metadatadir,
                '%s_%03d.hdf5'%(meta_name,self.snapnum))
            self.metadata = Metadata(
                self.metapath,
                **metadata_kwargs)

        ## let's try and make a header object, shall we?
        if self.snapnum is None:
            ## load the header from my catalog file
            try: self.header = self.loadHeaderFromCatalog() 
            except OSError: self.header = {}

            ## snapshot timing is not available because snapnum is None
            self.header['Redshift'] = None
            self.header['TimeGyr'] = None
        else: 
            ## attempt to open the header from the hdf5 file
            ##  let openSnapshot attempt to parse the snapdir/snapnum
            ##  situation and it'll raise an IOError if it can't find the file
            try:
                self.open_header(save_header_to_table=save_header_to_table)
                self.__set_snapshot_info(dummy=not full_init)
                return 
            except IOError as e:
                print(f"Couldn't find snapshot {self.snapnum:d} in {self.snapdir}.")
                print(e) 

        self.__set_snapshot_info(dummy=True)
 
    def open_header(self,save_header_to_table=True):
        self.header = openSnapshot(
            self.snapdir,
            self.snapnum,
            0, ## dummy particle index, not used if header_only is True
            header_only=True)

        if save_header_to_table:
            ## save the header to our catalog of simulation headers
            ##  so that we can open the header in scenarios where
            ##  snapnum is None, as above.
            try: self.saveHeaderToCatalog()
            ## OSError is when parallel processes try to write to header at the same time...
            ##  Value Error is probably when the simulation is already in the file?
            except (ValueError,OSError): pass

    def __set_snapshot_info(
        self,
        use_rockstar_first=True,
        dummy=False):

        if dummy:
            ## main halo information
            self.scom = None
            self.rvir = None
            self.rstar_half = None

        else:
            ## opens the halo file to find the halo center and virial radius
            if self.header['cosmological']: self.load_halo_file()#use_rockstar_first=use_rockstar_first)
            else:
                self.scom = np.zeros(3)
                self.rvir = 300 ## what should i do here...
                self.rstar_half = None

                if 'r30r' in self.snapdir:
                    self.scom = get_idealized_center(self.name,self.snapnum)

            if self.rstar_half is None:
                ## have we already calculated it and cached it?
                for attr in ['gas_extract_rstar_half','star_extract_rstar_half']:
                    if hasattr(self.metadata,attr):
                        self.rstar_half = getattr(self.metadata,attr)
                        break

        self.current_redshift = self.header['Redshift']
        self.current_time_Gyr = self.header['TimeGyr']

    def load_halo_file(
        self,
        halo_fname=None,
        halo_path=None,
        use_rockstar_first=True,
        **kwargs):

        ## decide which one is the fallback
        if not use_rockstar_first:
            first_fn = self.load_ahf
            second_fn = self.load_rockstar
        else:
            first_fn = self.load_rockstar
            second_fn = self.load_ahf

        if halo_path == 'None' or halo_fname == 'None':
            self.scom,self.rvir,self.rstar_half = None,None,None
            print(
                'Make sure to set:',
                'scom',
                'rvir',
                'rstar_half',
                'attributes manually')
        else:
            try: first_fn(halo_fname,halo_path,**kwargs)
            except IOError as e:
                try: second_fn(halo_fname,halo_path,**kwargs)
                except IOError:
                    print("Couldn't find AHF nor Rockstar halo files")
                    raise

    def load_ahf(self,halo_fname=None,halo_path=None,**kwargs):

        print("We don't use ahf, only rockstar. join us. ",end='\t')
        time.sleep(3)
        print("but we'll allow it this time.")

        if 'elvis' in self.snapdir:
            raise IOError("No AHF files for Elvis runs")

        ## automatically search for the halo_path and halo_fname
        self.halo_path, self.halo_fname = self.auto_search_ahf(**kwargs)
    
        ## now that we've attempted to identify an AHF file lets open 
        ##  this puppy up
        try:
            ## first try and read the stellar half-mass radius (default args)
            self.scom, self.rvir, self.rstar_half = load_AHF(
                self.snapdir,
                self.snapnum,
                ahf_path = self.halo_path,
                fname = self.halo_fname,
                hubble = self.header['HubbleParam'])

        except ValueError:
            ## no rstar 1/2 in this AHF file, we'll have to calculate it ourselves 
            self.scom, self.rvir = load_AHF(
                self.snapdir,
                self.snapnum,
                extra_names_to_read = [],
                ahf_path = self.halo_path,
                fname = self.halo_fname,
                hubble = self.header['HubbleParam'])

            self.rstar_half = None

        return self.scom,self.rvir,self.rstar_half
            
    def auto_search_ahf(self,halo_fname=None,halo_path=None):
        ## attempt to read halo location and properties from AHF
        if halo_fname is None:
            halo_fname='halo_0000%d_smooth.dat'%halo_id(self.snapdir_name)

        if halo_path is None:
            ## system blind if you put the soft link in
            halo_path = os.path.join(
                os.environ['HOME'],'halo_files',
                "%s","%s")

            halo_path = halo_path%(self.suite_name,self.snapdir_name)

        ## check if this first guess at the halo_fname and halo_path
        ##  is right
        if (not os.path.isfile(os.path.join(halo_path,halo_fname)) and 
            halo_path != 'None' and 
            halo_fname != 'None'):
            ## try looking in the simulation directory
            halo_path = os.sep.join(
                self.snapdir.split(os.sep)[:-1] ## remove output from the snapdir
                +['halo','ahf']) ## look in <simdir>/halo/ahf

            ## okay no smooth halo file but there is a halo/ahf directory at least
            if (os.path.isdir(halo_path) and (
                not os.path.isfile(os.path.join(halo_path,halo_fname)))):

                ## let's scan through the files in the directory and try and 
                ##  find an AHF halo file that corresponds to just this snapshot. 
                fnames = []
                snap_string = "%03d"%self.snapnum

                for fname in os.listdir(halo_path):
                    if (snap_string in fname and 
                        'AHF_halos' in fname):
                        fnames+=[fname]
                if len(fnames) == 1:
                    fname = fnames[0]
                else:
                    raise IOError("can't find a halo file (or found too many)",fnames)

        return halo_path, halo_fname
    
    def get_rockstar_file_output(
        self,
        use_metadata=True,
        save_meta=True,
        loud=True,
        assert_cached=False,
        force_from_file=False,
        fancy_trace=False,
        smooth=None,#0.1, ## results in snapshots that are way mis-centered
        **kwargs):
        """fancy_trace=True - use additional phase space info to minimize jumps in rcom coordinates.
            smooth=0.1 - whether to smooth by a window in Gyr, None for no smoothing"""

        prefix = ''
        if fancy_trace:
            prefix+='fancy_'
        if smooth is not None:
            prefix+='smoothed%.3fGyr_'%smooth

        """
        ## separate cachefile that we can save to that all snapshots can use
        hdf5_path = os.path.join(
            os.environ['HOME'],
            'halo_files',
            'rockstar',
            self.suite_name,
            self.name,
            prefix+'rockstar_trace.hdf5')
        """

        kwargs['smooth'] = smooth
        kwargs['fancy_trace'] = fancy_trace
 
        @metadata_cache(
            prefix+'rockstar_history',
            [prefix+'snapnums',prefix+'rcoms',prefix+'rvirs',prefix+'treefile'],
            use_metadata=use_metadata,
            save_meta=save_meta,
            loud=loud,
            assert_cached=assert_cached,
            force_from_file=force_from_file)
        def compute_rockstar_file_output(self,fancy_trace=False,smooth=0.1):

            #print(f'Tracing the rockstar halo files with fancy:{fancy_trace} and {smooth} Gyr smoothing.')
            snapnums,rcoms,rvirs,treefile = trace_rockstar(self.snapdir,fancy_trace=fancy_trace)

            self.get_snapshotTimes()
            scale_factors = 1/(1+self.snap_zs[snapnums])
            rcoms*=scale_factors[:,None]

            if smooth is not None:
                new_rcoms = copy.copy(rcoms)
                for i in range(3):
                    smooth_times,smooth_coords,_,_,_ = smooth_x_varying_curve(
                        self.snap_gyrs[-len(snapnums):],
                        rcoms[:,i],
                        smooth,
                        assign='center')
                    new_coords = interp1d(
                        smooth_times,
                        smooth_coords,
                        bounds_error=False,
                        fill_value=np.nan)(
                        self.snap_gyrs[-len(snapnums):])
                    new_rcoms[-len(snapnums):,i] = new_coords

                    ## can't smooth the first and last half-window
                    ##  /shrug
                    outside_window_mask = np.isnan(new_rcoms[:,i])
                    new_rcoms[outside_window_mask,i] = rcoms[outside_window_mask,i]
                rcoms = new_rcoms

            return snapnums,rcoms,rvirs,treefile

        """
        ## try loading from a cached trace
        if os.path.isfile(hdf5_path) and use_metadata:
            if loud: print("reading from",hdf5_path)
            with h5py.File(hdf5_path,'r') as handle:
                snapnums = handle[prefix+'rockstar_history'][prefix+'snapnums'][()]
                rcoms = handle[prefix+'rockstar_history'][prefix+'rcoms'][()]
                rvirs = handle[prefix+'rockstar_history'][prefix+'rvirs'][()]
        ## trace the halo and save it if allowed
        else:
            if save_meta: self.metadata.export_to_file(hdf5_path,prefix+'rockstar_history',write_mode='w')
        """

        snapnums,rcoms,rvirs,treefile = compute_rockstar_file_output(self,**kwargs)

        self.halo_file = str(treefile)

        return snapnums,rcoms,rvirs
    
    def get_ahf_file_output(
        self,
        use_metadata=True,
        save_meta=False,
        loud=True,
        assert_cached=False,
        force_from_file=False,
        **kwargs):

        raise Exception("We don't use ahf, only rockstar. join us.")

        @metadata_cache(
            'ahf_history',
            ['snapnums','rcoms','rvirs'],
            use_metadata=use_metadata,
            save_meta=save_meta,
            loud=loud,
            assert_cached=assert_cached,
            force_from_file=force_from_file)
        def compute_ahf_file_output(self):
            ## get the self.halo_path to point to rockstar if possible
            self.load_ahf()
            snapnums = load_AHF(
                self.snapdir,
                self.snapnum,
                ahf_path = self.halo_path,
                fname = self.halo_fname,
                hubble = self.header['HubbleParam'],
                return_full_halo_file=True,
                extra_names_to_read=[])[:,1]

            rcoms = np.zeros((snapnums.size,3))+np.nan
            rvirs = np.zeros(snapnums.size)+np.nan

            for i in snapnums:
                rcom,rvir = load_AHF(
                    self.snapdir,
                    i,
                    ahf_path = self.halo_path,
                    fname=self.halo_fname,
                    hubble=self.header['HubbleParam'])[:2]

                rcoms[int(i-snapnums[0]),:] = rcom
                rvirs[int(i-snapnums[0])] = rvir

            return snapnums[::-1],rcoms,rvirs
        return compute_ahf_file_output(self,**kwargs)

    def load_rockstar(self,halo_fname=None,halo_path=None,**kwargs):

        ## have default kwargs but allow them to be ignored
        ##  in most human readable way
        if 'use_metadata' not in kwargs:
            kwargs['use_metadata'] = True
        if 'save_meta' not in kwargs:
            kwargs['save_meta'] = True
        if 'loud' not in kwargs:
            kwargs['loud'] = False

        try:
            snapnums,rcoms,rvirs = self.get_rockstar_file_output(**kwargs)

            index = np.argwhere(snapnums==self.snapnum).astype(int)[0][0]
            self.scom = rcoms[index]
            self.rvir = rvirs[index]
            self.rstar_half = None

        except IOError:
            raise
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
            if halo_path is None:
                self.halo_path = '../halo/rockstar_dm/'
                self.halo_path = os.path.realpath(os.path.join(self.snapdir,self.halo_path))

            self.halo_fname = 'halo_%03d.hdf5'%self.snapnum if halo_fname is None else halo_fname

            self.scom,self.rvir = load_rockstar(
                self.snapdir,
                self.snapnum,
                fname=halo_fname,
                rockstar_path=halo_path,
                which_host=which_host)
            self.rstar_half=None

        return self.scom,self.rvir,self.rstar_half

    def get_snapshotTimes(
        self,
        snaptimes='my_snapshot_times',
        assert_cached=False,
        use_metadata=True,
        target=None): 
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

        if target is None: data_FIRE_SN_times = os.path.join(self.datadir,'%s.txt'%snaptimes)
        else: data_FIRE_SN_times = target

        if os.path.isfile(data_FIRE_SN_times) and use_metadata:
            (self.snapnums,
                self.snap_sfs,
                self.snap_zs,
                self.snap_gyrs,
                self.dTs) = np.genfromtxt(data_FIRE_SN_times,unpack=1)
            return

        ## well alright, I guess we have to create the file then...
        ##  check if there's a list of scale factors we can just convert
        FIRE_SN_times = os.path.join(self.snapdir,'..','snapshot_times.txt')

        if os.path.isfile(FIRE_SN_times):
            try:
                (snapnums,
                    sfs,
                    zs,
                    _,
                    dTs) = np.genfromtxt(FIRE_SN_times,unpack=1)
            except:
                # i scale-factor redshift time[Gyr] lookback-time[Gyr] time-width[Myr]
                (snapnums,
                    sfs,
                    zs,
                    _,
                    _,
                    dTs) = np.genfromtxt(FIRE_SN_times,unpack=1)

            if 'Omega0' in self.header.keys(): Omega0 = self.header['Omega0']
            else: Omega0 = self.header['Omega_Matter']

            gyrs = RydenLookbackTime(
                self.header['HubbleParam'],
                Omega0,
                sfs)

        else:
            ## well, alright I guess. then we have to open every snapshot :[
            if assert_cached:
                raise AssertionError("User asserted that the snapshot times should be saved to disk.")

            finsnap = getfinsnapnum(self.snapdir)
            print("Oh boy, have to open %d files to output their snapshot timings"%finsnap)

            snapnums = []
            sfs = []
            gyrs = []
            zs = []
            dTs = [0]

            printProgressBar(0,finsnap+1)
            for snapnum in range(finsnap+1):
                printProgressBar(snapnum+1,finsnap+1)
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

                    if len(gyrs) >= 2 : dTs += [gyrs[-1]-gyrs[-2]]

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
        save_meta=False, 
        orient_component='cold_gas', ## which angular momentum axis to orient along
        overwrite_full_snaps_with_rotated_versions=False,
        free_mem=True, ## delete the full snapshot from RAM
        extract_DM=True, ## do we want the DM particles? 
        compute_stellar_hsml=False,
        loud=True,
        jhat_coords=False,
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
        
        group_name = f"{orient_component}_extract"

        @metadata_cache(
            group_name,
            ['sub_radius',
            'orient_component',
            'theta_TB',
            'phi_TB',
            'rvir',
            'rstar_half',
            'rgas_half'],
            use_metadata=False,
            save_meta=save_meta,
            loud=loud)
        def extract_halo_inner(
            self,
            orient_component='cold_gas',
            radius=None,
            use_saved_subsnapshots=True,
            force=False,
            force_theta_TB=None,
            force_phi_TB=None):

            ## handle default remappings

            if self.scom is None:
                self.load_stars()
                self.scom = iterativeCoM(
                    self.star_snap['Coordinates'],
                    self.star_snap['Masses'],
                    n=4)

            ## calculate the stellar half-mass radius to have--
            ##  some analyses require 5rstar_half
            if self.rstar_half is None: self.get_rstar_half(
                save_meta=save_meta,
                force_from_file=True,
                loud=False,
                within_radius=0.2*self.rvir)

            if radius is None: radius = self.rvir ## radius of the sub-snapshot 

            ## radius to calculate angular momentum to orient on
            #orient_radius = 5*self.rstar_half 
            ## switched on 1/16/2023 for paper 3, also orient on cold_gas
            orient_radius =  0.1 * self.rvir 

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
                    if not use_saved_subsnapshots: raise AssertionError("Told not to use saved sub-snapshots")

                    if loud: print("Using the saved sub-snapshots for",self)

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
                    if not np.isclose(np.round(radius,4),np.round(self.sub_snap['scale_radius'],4)):
                        this_sim = self.snapdir.split('snaps')[1].split('/output')[0]
                        if this_sim not in fname:
                            raise ValueError("%s %s do not correspond"%(self.snapdir,fname))

                        sub_scale_radius = self.sub_snap['scale_radius']
                        ## delete it because it is GARBAGE
                        os.remove(fname)
                        del self.sub_snap
                        del self.sub_star_snap
                        del self.sub_dark_snap
                        already_saved = False

                        raise ValueError("virial radius is not the same",
                            radius-sub_scale_radius,
                            radius,sub_scale_radius)

                    ## check if halo center is the same to 4 decimal places
                    if not np.isclose(np.round(self.scom,4),np.round(self.sub_snap['scom'],4)).all():
                        this_sim = self.snapdir.split('snaps')[1].split('/output')[0]
                        if this_sim not in fname:
                            raise ValueError("%s %s do not correspond"%(self.snapdir,fname))
                        sub_scom = self.sub_snap['scom']
                        ## delete it because it is GARBAGE
                        os.remove(fname)
                        del self.sub_snap
                        del self.sub_star_snap
                        del self.sub_dark_snap
                        already_saved = False
                        raise ValueError("Halo center is not the same",
                            self.scom - sub_scom,
                            self.scom , sub_scom)

                    if loud: print("Successfully loaded a pre-extracted subsnap")

                    ## pass these sub-snapshots into the rotation routine
                    which_snap = self.sub_snap
                    which_star_snap = self.sub_star_snap
                    if extract_DM:
                        which_dark_snap = self.sub_dark_snap

                except (AttributeError,AssertionError,ValueError,IOError,KeyError,TypeError) as error:
                    message = "Failed to open saved sub-snapshots"
                    #message+= ' %s'%error.__class__  
                    message+= ' %s'%repr(error)
                    if loud: print(message)

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
                orient_component=orient_component,
                force_theta_TB=force_theta_TB,
                force_phi_TB=force_phi_TB,
                loud=loud)

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
            self.orient_component = orient_component

            if compute_stellar_hsml and 'SmoothingLength' not in self.sub_star_snap: 
                already_saved=False
                self.sub_star_snap['SmoothingLength'] = self.get_HSML(
                    loud=loud,
                    save_meta=False)

            ## save for later, if requested
            if (use_saved_subsnapshots and
                extract_DM and
                not already_saved):
                self.outputSubsnapshot()

            if not hasattr(self,'rgas_half'):
                self.rgas_half = self.calculate_half_mass_radius(
                    which_snap=self.sub_snap) 
            

            return (self.sub_radius,
                self.orient_component,
                self.sub_snap['theta_TB'],
                self.sub_snap['phi_TB'],
                self.rvir,
                self.rstar_half,
                self.rgas_half)

        return_value = extract_halo_inner(
            self,
            orient_component=orient_component,
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
            if loud: print("Snapshot memory free")
        
        snapdicts = [self.sub_snap,self.sub_star_snap]
        if extract_DM: snapdicts += [self.sub_dark_snap]

        ## store a couple of things in the dictionary to identify it:
        for snapdict in snapdicts:
            snapdict['name'] = self.name
            snapdict['snapnum'] = self.snapnum
            snapdict['datadir'] = self.datadir

            if jhat_coords: add_jhat_coords(snapdict)

        return return_value

    def get_rstar_half(self,
        use_metadata=True,
        save_meta=False,
        loud=True,
        assert_cached=False,
        force_from_file=False,
        **kwargs):
    
        @metadata_cache(
            'star_extract',
            ['rstar_half'],
            use_metadata=use_metadata,
            save_meta=save_meta,
            loud=loud,
            assert_cached=assert_cached,
            force_from_file=force_from_file)
        def compute_rstar_half(self,within_radius=None):
            self.load_stars()
            return self.calculate_half_mass_radius(within_radius=within_radius),
        return compute_rstar_half(self,**kwargs)

    def get_HSML(
        self,
        snapdict_name='star',
        use_metadata=True,
        save_meta=True,
        assert_cached=False,
        loud=True,
        **kwargs, 
        ):
        """Compute smoothing lengths for particles that don't have them,
            typically collisionless particles (like stars). 

            Input:

                snapdict_name -- name in the form of `'%s_snapdict'%snapdict_name`
                    that will be used to compute smoothing lengths for. 

                use_metadata = True -- flag to search cache for result
                save_meta = True -- flag to cache the result
                assert_cached = False -- flag to require a cache hit
                loud = True -- whether cache hits/misses should be announced
                    to the console.
                
            Output:

                smoothing_lengths -- numpy array of estimated smoothing lengths"""

        @metadata_cache(
            '%s_data'%snapdict_name,  ## hdf5 file group name
            ['%s_SmoothingLengths'%snapdict_name],
            use_metadata=use_metadata,
            save_meta=save_meta,
            assert_cached=assert_cached,
            loud=loud,
            force_from_file=True) ## read from cache file, not attribute of object
        def compute_HSML(self,snapdict_name):
            snapdict_name = ('sub_%s_snap'%snapdict_name).replace('sub_gas','sub')
            snapdict = getattr(self,snapdict_name)
            pos = snapdict['Coordinates']
            smoothing_lengths = get_particle_hsml(pos[:,0],pos[:,1],pos[:,2])
            return smoothing_lengths

        return compute_HSML(self,snapdict_name,**kwargs)

    def load_stars(self,**kwargs):
        if self.metadata.loud_metadata:
            print("Loading star particles of",self,'at',self.snapdir)
        if not hasattr(self,'star_snap'):
            self.star_snap = openSnapshot(
                self.snapdir,
                self.snapnum,4,
                **kwargs)

    def load_gas(self,**kwargs):
        if self.metadata.loud_metadata:
            print("Loading gas particles of",self,'at',self.snapdir)
        self.snap = openSnapshot(
            self.snapdir,
            self.snapnum,0,
            **kwargs)

    def load_dark_matter(self,**kwargs):
        if self.metadata.loud_metadata:
            print("Loading dark matter particles of",self,'at',self.snapdir)
        self.dark_snap = openSnapshot(
            self.snapdir,self.snapnum,1,
            **kwargs)

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
                if key == 'AngularMomentum': 
                    print('why is AngularMomentum being output to ABG_Header in abg_python.galaxy.gal_utils?')
                    continue
                ABG_Header.attrs[key]=self.sub_snap[key]

            derived_arrays = set(['Temperature','AgeGyr','AngularMomentum'])
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
        theta_TB,phi_TB,scom,vscom,orient_component = (
            self.sub_snap['theta_TB'],self.sub_snap['phi_TB'],
            self.sub_snap['scom'],self.sub_snap['vscom'],
            self.sub_snap['orient_component'])

        ## if snaps doesn't have dark snap then zipped will stop at [0,4]
        for ptype,snap in zip([0,4,1],snaps):
            if 'overwritten' not in snap or not snap['overwritten']:
                ## set snap to new snapdict that has offset/rotated coords
                snap = offsetRotateSnapshot(
                    snap,
                    scom,vscom,
                    theta_TB,phi_TB,
                    orient_component)

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
    
    def purge_metadata(
        self,
        group_name,
        key_name=None,
        snaplow=None,
        snaphigh=None,
        force=False,
        loud=True,
        mps=1):

        if snaplow is None: snaplow = self.minsnap
        if snaphigh is None: snaphigh = self.finsnap

        argss = zip(
            itertools.repeat(self),
            np.arange(snaplow,snaphigh+1),
            itertools.repeat(group_name),
            itertools.repeat(key_name),
            itertools.repeat(force),
            itertools.repeat(loud))

        if mps > 1:
            with multiprocessing.Pool(multiprocessing.cpu_count()) as my_pool:
                my_pool.starmap(purge_metadata_group_wrapper,argss)
                my_pool.close()
                my_pool.join()
        else:
            for args in argss: purge_metadata_group_wrapper(*args)

    def __repr__(self):
        return "%s many-galaxy wrapper"%(self.name)

    def __init__(
        self,
        name,
        name_append='',
        load_snapnums=None,
        population_kwargs=None,
        **galaxy_kwargs):
        """ a wrapper that will allow one to open multiple galaxies at the same time,
            most useful for creating and accessing MultiMetadata instances while 
            using the same plotting scripts that a Galaxy instance would work for 
            (in general, this must be done consciously while making a plotting script). """
        
        ## do all the snapdir, datadir, w.e. voodoo but don't do any of the hard stuff
        Galaxy.__init__(self,name,None,**galaxy_kwargs,full_init=False)

        ## now do the ManyGalaxy specific stuff
        self.name = name+name_append

        ## allow a MultiGalaxy wrapper to open histories files
        if self.snapdir is not None:
            self.finsnap = getfinsnapnum(self.snapdir)
            ## get the minimum snapnum
            self.minsnap = getfinsnapnum(self.snapdir,True)
        ## filler values
        else: self.finsnap,self.minsnap=None,None

        try: self.get_snapshotTimes(assert_cached=True)
        except (AssertionError,AttributeError,KeyError): 
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
                else: return getattr(self,attr) ## might fail but that's what we want

            def __getitem__(self,index):
                return self.galaxies[index]

            setattr(self,'__getattr__',__getattr__)
            setattr(self,'__getitem__',__getitem__)

    def get_final_orientation(self):
        """Get Tait-Bryan xyz rotation angles from the final snapshot"""
        last_galaxy = self.loadAtSnapshot(self.finsnap)
        try:
            ## attempt to read orientation from the cached metadata
            phi_TB = last_galaxy.metadata.star_extract_phi_TB
            theta_TB = last_galaxy.metadata.star_extract_theta_TB
        except AttributeError:
            ## extract the main halo and cache the output
            last_galaxy.extractMainHalo(save_meta=True)
            phi_TB = last_galaxy.metadata.star_extract_phi_TB
            theta_TB = last_galaxy.metadata.star_extract_theta_TB

        return theta_TB,phi_TB

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
                halo_path=None,
                halo_fname=None,
            """

        new_kwargs = {}
        new_kwargs.update(self.galaxy_kwargs)
        new_kwargs.update(kwargs)

        return Galaxy(self.name,snapnum,**new_kwargs)

def get_idealized_center(savename,snapnum):

    center_name = {
        'm11_def_r30r':'centers_feedback_normal',
        'm11_light_def_r30r':'centers_feedback_light',
        'm11_extralight_def_r30r':'centers_feedback_extralight',
    }[savename]

    center_path = os.path.join(
        os.environ['HOME'],
        'scratch',
        'data',
        'jonathan_centers',
        center_name+'.npz')

    try:
        return np.load(center_path)['centers'][snapnum]
    except:
        return np.load(center_path)['centers'][snapnum-1]


## MPS wrapper functions

def purge_metadata_group_wrapper(many_galaxy,snapnum,group_name,key_name,force,loud):
    galaxy = many_galaxy.loadAtSnapshot(snapnum)
    galaxy.metadata.purge_metadata_group(
        group_name=group_name,
        key_name=key_name,
        force=force,
        loud=loud)
