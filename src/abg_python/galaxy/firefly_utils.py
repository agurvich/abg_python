import os
import numpy as np

from ..math_utils import rotateEuler

try:
    from firefly.data_reader import Reader,ParticleGroup
except (ModuleNotFoundError,ImportError):
    print("Missing firefly, obtain it at http://github.com/ageller/Firefly or pip install firefly.")


from matplotlib.colors import to_rgba

colors = {
    'gas': [1,0,0,1],
    'stars' : [0,0,1,1],
    'dark' : [1,1,0,1]
    }

class Firefly_helper(object):

    def initialize_reader(self,JSONdir=None,write_startup='append',firefly_data_dir=None,**kwargs):
        """Kwargs passed to this function will create a new self.reader instance:\n"""
        
        if firefly_data_dir is None: firefly_data_dir = os.path.join(self.datadir,'firefly')

        if JSONdir is None:
            JSONdir = os.path.join(
                firefly_data_dir,
                '%s_%03d'%(self.name,self.snapnum))
        if not os.path.isdir(os.path.dirname(JSONdir)):
            os.makedirs(os.path.dirname(JSONdir))

        self.reader = Reader(JSONdir=JSONdir,clean_JSONdir=True,write_startup=write_startup,**kwargs)
        return self.reader
    try:
        initialize_reader.__doc__ += Reader.__init__.__doc__
    ## Reader won't exist if we didn't successfully import
    except NameError as e1:
        try:
            from firefly_api.reader import Reader
            from firefly_api.particlegroup import ParticleGroup
            raise e1
        except ImportError:
            pass
        

    @property
    def reader(self):
        if not hasattr(self,'_reader'):
            print('initializing')
            JSONdir = os.path.join(
                self.datadir,
                'firefly',
                '%s_%03d'%(self.name,self.snapnum))
            self._reader = Reader(
                JSONdir=JSONdir,
                clean_JSONdir=True,
                write_startup='append')
        return self._reader

    @reader.setter
    def reader(self,value):
        self._reader = value

    @reader.deleter
    def reader(self):
        del self._reader

    def add_firefly_particles(
        self,
        pname='gas',
        UIname=None,
        full_snap=False,
        keys=None,
        mask=None,
        decimation_factor=1000,
        color=None):

        UIname = pname if UIname is None else UIname

        ## determine which particles we want
        which_snap = getattr(
            self,
            'sub_'*(not full_snap) + 
                ('' if pname == 'gas' else 
                pname+'_')
                + 'snap')

        if keys is None:
            keys = []

        coords = which_snap['Coordinates']
        if mask is not None:
            coords = coords[mask]

        tracked_arrays = {}
        for key in keys:

            if key == 'Radius':
                value = np.sqrt(np.sum(coords**2,axis=1))/self.rvir
            else:
                if key[:5] != 'log10':
                    value = which_snap[key] 
                else:
                    value = np.log10(which_snap[key[5:]])

                if mask is not None:
                    value = value[mask]

            tracked_arrays[key] = value

        this_ParticleGroup = self.track_firefly_particles(
            UIname,
            coords,
            decimation_factor,
            tracked_arrays=tracked_arrays)

        ## start with velocity vectors enabled
        self.reader.settings['showVel'][UIname] = True
        self.reader.settings['velType'][UIname] = 'arrow'

        if color is None:
            self.reader.settings['color'][UIname] = colors[pname]
        else:
            if len(color) != 4 and type(color) != list:
                if type(color) == str:
                    color = to_rgba(color)
                elif len(color) == 3:
                    color = list(np.append(color,[1]))
                else:
                    print(color,type(color))
                    raise ValueError('Color must be an RGBA list')
            self.reader.settings['color'][UIname] = color

        self.reader.settings['sizeMult'][UIname] = 5 if UIname != 'hot' else 8
        
        return this_ParticleGroup 

    def add_firefly_axes(
        self,
        reorient_angles=None,
        radius=None,
        height=None):
    

        ## handle default arguments
        radius = 5*self.rstar_half if radius is None else radius
        height = 10 if height is None else height

        n_repeat = 5
        disk_coords = np.zeros((360*n_repeat,3))
        thetas = np.linspace(0,2*np.pi,int(disk_coords.shape[0]/n_repeat),endpoint=False)
        disk_coords[:,0] = np.tile(radius*np.cos(thetas),n_repeat)
        disk_coords[:,1] = np.tile(radius*np.sin(thetas),n_repeat)
        disk_coords[:,2] = np.repeat(np.linspace(-1,1,n_repeat),360)*height

        zaxis_coords = np.zeros((20,3))
        zaxis_coords[:,-1] = np.linspace(-height,height,20,endpoint=True)

        for key,coords in zip(
            ['disk','zaxis'],[disk_coords,zaxis_coords]):

            if reorient_angles is not None:
                ## unpack the angles
                unrotate_theta,unrotate_phi,fixed_theta,fixed_phi = reorient_angles

                ## re-rotate
                coords = rotateEuler(unrotate_theta,unrotate_phi,0,coords,inverse=True, loud=False)
                coords = rotateEuler(fixed_theta,fixed_phi,0,coords,loud=False)

            if reorient_angles is not None:
                key+='_orient'

            self.track_firefly_particles(key,coords,1)

            self.reader.settings['UIparticle'][key] = False 
            self.reader.settings['color'][key] = [1.0,1.0,1.0,1.0]
            self.reader.settings['sizeMult'][key] = 2.5
                
    def track_firefly_particles(
        self,
        UIname,
        coords,
        decimation_factor=1,
        **kwargs):

        self.reader.addParticleGroup(ParticleGroup(
            UIname,
            coords,
            decimation_factor=decimation_factor,
            **kwargs))

        return self.reader.particleGroups[-1]
