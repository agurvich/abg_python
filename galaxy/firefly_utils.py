import os
import numpy as np

from abg_python.all_utils import rotateEuler

try:
    from firefly_api.reader import Reader
    from firefly_api.particlegroup import ParticleGroup
except ImportError:
    print("missing firefly api")


from matplotlib.colors import to_rgba

colors = {
    'gas': [1,0,0,1],
    'stars' : [0,0,1,1],
    'dark' : [1,1,0,1]
    }

class Firefly_helper(object):

    def initialize_reader(self,JSONdir=None,write_startup='append',**kwargs):
        """Kwargs passed to this function will create a new self.reader instance:\n"""
        
        if JSONdir is None:
            JSONdir = os.path.join(
                self.datadir,
                'firefly',
                '%s_%03d'%(self.name,self.snapnum))

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

        this_ParticleGroup = self.track_firefly_particles(
            UIname,
            coords,
            decimation_factor)

        for key in keys:

            if key[:5] != 'log10':
                value = which_snap[key] 
            else:
                value = np.log10(which_snap[key[5:]])

            if mask is not None:
                value = value[mask]
            this_ParticleGroup.trackArray(
                key,
                value)

        ## start with velocity vectors enabled
        self.reader.options['showVel'][UIname] = True
        self.reader.options['velType'][UIname] = 'arrow'

        if color is None:
            self.reader.options['color'][UIname] = colors[pname]
        else:
            if len(color) != 4 and type(color) != list:
                if type(color) == str:
                    color = to_rgba(color)
                elif len(color) == 3:
                    color = list(np.append(color,[1]))
                else:
                    print(color,type(color))
                    raise ValueError('Color must be an RGBA list')
            self.reader.options['color'][UIname] = color

        self.reader.options['sizeMult'][UIname] = 5 if UIname != 'hot' else 8
        
        return this_ParticleGroup 

    def add_firefly_axes(
        self,
        reorient_angles=None,
        radius=None,
        height=None):
    

        ## handle default arguments
        radius = 5*self.rstar_half if radius is None else radius
        height = 10 if height is None else height

        disk_coords = np.zeros((180,3))
        thetas = np.linspace(0,2*np.pi,disk_coords.shape[0],endpoint=False)
        disk_coords[:,0] = radius*self.rstar_half*np.cos(thetas)
        disk_coords[:,1] = radius*self.rstar_half*np.sin(thetas)

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

            this_ParticleGroup = self.track_firefly_particles(
                key,
                coords,
                1)
            self.reader.options['UIparticle'][key] = False 
            self.reader.options['color'][key] = [1.0,1.0,1.0,1.0]
            self.reader.options['sizeMult'][key] = 2.5
                
    def track_firefly_particles(
        self,
        UIname,
        coords,
        decimation_factor = 1
        ):

        self.reader.addParticleGroup(ParticleGroup(
            UIname,
            coords,
            decimation_factor=decimation_factor))

        return self.reader.particleGroups[-1]
