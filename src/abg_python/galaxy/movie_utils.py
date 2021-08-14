#from movie_maker_gasTwoColour import plot_image_grid
#from movie_maker_gasDensity_v2 import compute_image_grid as compute_density_grid
#from movie_maker_gasTemperature_v2 import compute_image_grid as compute_temp_grid
#from sne_utils import findClusterExtent,drawSN
#from all_utils import getTemperature

from abg_python.distinct_colours import get_distinct

from abg_python.all_utils import rotateVectors
from abg_python.plot_utils import nameAxes,addColorbar

import matplotlib.pyplot as plt
import numpy as np 
import os

import h5py

import copy 

class Draw_helper(object):
    """------- Draw_helper 
    """

    def drawGasGalaxy(
        self,
        thetax=None,thetay=None,thetaz=None,
        indices=None,
        axs=None,
        radius=None,
        full_snap=False,
        **kwargs):
        if not full_snap:
            coords = self.sub_snap['Coordinates']
        else:
            coords = self.snap['Coordinates']
        if radius is not None:
            indices = np.sum(coords**2,axis=1)<radius**2
        return self.drawGalaxy(
            coords,
            thetax,thetay,thetaz,
            indices,axs=axs,
            **kwargs)
    
    def drawStellarGalaxy(
        self,
        thetax=None,thetay=None,thetaz=None,
        indices=None,axs=None,
        radius=None,
        **kwargs):
        coords = self.sub_star_snap['Coordinates']

        if radius is not None:
            indices = np.sum(coords**2,axis=1)<radius**2
       
        return self.drawGalaxy(
            coords,
            thetax,thetay,thetaz,
            indices,axs=axs,
            **kwargs)
 

    def drawDarkGalaxy(
        self,
        thetax=None,thetay=None,thetaz=None,
        indices=None,axs=None,
        **kwargs):
        coords = self.sub_dark_snap['Coordinates']
        return self.drawGalaxy(
            coords,
            thetax,thetay,thetaz,
            indices,axs=axs,
            **kwargs)

    def drawGalaxy(
        self,
        coords,
        thetax=None,thetay=None,thetaz=None,
        indices=None,
        axs=None,
        **kwargs):
        if thetax is not None:
            coords = rotateVectors(rotationMatrixX(thetax),coords)
        if thetay is not None:
            coords = rotateVectors(rotationMatrixY(thetay),coords)
        if thetaz is not None:
            coords = rotateVectors(rotationMatrixZ(thetaz),coords)
        if indices is None:
            indices = np.ones(len(coords),dtype=bool)
        return plotSideBySide(
            coords,
            np.zeros(3),
            indices,
            axs=axs,
            **kwargs)

    def fauxrender(self,ax,savefig=0,noaxis=0,bins=1200,
        theta=0,phi=0,psi=0,cylinder=None,
        **kwargs):

        frame_width=self.sub_radius
        frame_half_thickness=cylinder if cylinder else self.sub_radius
        frame_center=np.zeros(3)

        indices = extractRectangularVolumeIndices(self.sub_snap['Coordinates'],
            frame_center,frame_width,frame_width if frame_half_thickness is None else frame_half_thickness)

        pos = self.sub_snap['Coordinates']# want to rotate about frame_center

        ## don't rotate anything

        ## add back the offset post rotation...?
        radii = np.sum(pos**2,axis=1)**0.5
        pos_rot = rotateEuler(theta,phi,psi,pos,frame_center)
        rot_radii = np.sum(pos_rot**2,axis=1)**0.5

        assert np.allclose(radii,rot_radii)

        xs,ys = pos_rot[:,:2][indices].T

        twoDHist(ax,xs,ys,bins=bins)

        ax.set_ylim(frame_center[1]-frame_width,frame_center[1]+frame_width)
        ax.set_xlim(frame_center[0]-frame_width,frame_center[0]+frame_width)
        ax.get_figure().set_size_inches(6,6)


        if noaxis:
            ax.axis('off')
        return ax

class FIREstudio_helper(object):
    """------- FIREstudio_helper
    """

    def initialize_FIREstudio(self):
        ## guard import to avoid
        try:
            from firestudio.studios.gas_studio import GasStudio
            from firestudio.studios.star_studio import StarStudio

            self.firestudio_GasStudio = GasStudio
            self.firestudio_StarStudio = StarStudio

        except ImportError as error:
            print(error)
            print(error.message)
            print('download firestudio from github')
            raise

    def render(
        self,
        ax,
        assert_cached=False,
        use_metadata=True,
        save_meta=True,
        edgeon=False,
        frame_half_thickness=None,
        frame_half_width=15,
        min_weight=-0.5,
        max_weight=1.6,
        min_quantity=2,
        max_quantity=7,
        quick=False,
        cmap='viridis',
        **kwargs):

        frame_half_thickness = frame_half_width if frame_half_thickness is None else frame_half_thickness 


        ## attempt to import FIRE_studio
        self.initialize_FIREstudio()

        gasStudio = self.firestudio_GasStudio(
            datadir=os.path.join(self.datadir,'firestudio'),
            snapnum=self.snapnum,
            sim_name=self.name.split('-')[0], ## split for anything tacked on to end of name
            snapdir=self.snapdir,
            gas_snapdict=self.sub_snap if not assert_cached else None,
            star_snapdict=self.sub_star_snap if not assert_cached else None,
            frame_half_thickness=frame_half_thickness,
            frame_half_width=frame_half_width,
            **kwargs)

        ## manually set the aspect ratio to an integer number of the disk
        ##  aspect ratios, since we know what that is. don't use the 
        ##  built in edgeon flag in render in this case. 
        if edgeon:
            gasStudio.set_ImageParams(
                theta = 90,
                aspect_ratio = frame_half_thickness/frame_half_width)

        kwargs.update(
            {'min_weight':min_weight,
            'max_weight':max_weight,
            'min_quantity':min_quantity,
            'max_quantity':max_quantity})

        gasStudio.render(
            ax,
            quantity_name='LogTemperature',
            assert_cached=assert_cached,
            #quantity_adjustment_function=np.log10,
            weight_adjustment_function=lambda x: np.log10(x*1e10/1e6/gasStudio.Acell),
            use_metadata=use_metadata,
            save_meta=save_meta,
            quick=quick,
            cmap=cmap,
            **kwargs)

        ## free up any memory associated with that object
        return ax,gasStudio

    def star_render(
        self,
        ax,
        assert_cached=False,
        frame_half_width=15,
        frame_half_thickness=None,
        edgeon=False,
        quick=False,
        master_loud=False,
        **kwargs):

        frame_half_thickness = frame_half_width if frame_half_thickness is None else frame_half_thickness

        starStudio = self.firestudio_StarStudio(
            datadir=os.path.join(self.datadir,'firestudio'),
            snapnum=self.snapnum,
            sim_name=self.name.split('-')[0], ## split for anything tacked on to end of name
            snapdir=self.snapdir,
            gas_snapdict=self.sub_snap if not assert_cached else None,
            star_snapdict=self.sub_star_snap if not assert_cached else None,
            frame_half_thickness=frame_half_thickness,
            frame_half_width=frame_half_width,
            master_loud=master_loud,
            **kwargs)
            
            ## manually set the aspect ratio to an integer number of the disk
            ##  aspect ratios, since we know what that is. don't use the 
            ##  built in edgeon flag in render in this case. 
        if edgeon:
            starStudio.set_ImageParams(
                theta = 90,
                aspect_ratio =frame_half_thickness/frame_half_width)
 
        print(starStudio.npix_x,starStudio.npix_y,'pixels input')

        starStudio.render(ax,assert_cached=assert_cached,quick=quick,**kwargs)

        ## free up any memory associated with that object
        return ax,starStudio

    def renderPatch(self,ax,
        frame_half_width,frame_center,
        frame_half_thickness=None,
        savefig=0,
        noaxis=0,
        snap=None,
        **kwargs):
        """Renders a patch of the galaxy at frame_center, of dimensions
            frame_width x frame_half_thickness"""

        snap = self.sub_snap if snap is None else snap
        self.firestudio_renderGalaxy(
            ax,
            self.snapdir,self.snapnum,
            frame_half_width = frame_half_width,
            frame_half_thickness = frame_width if frame_half_thickness is None else frame_half_thickness,
            frame_center = frame_center,
            extract_galaxy=False,
            datadir = self.datadir,
            snapdict=snap,
            savefig=savefig,noaxis=noaxis,
            Hsml = snap['SmoothingLength'],
            **kwargs)

    def starRenderPatch(self,ax,
        frame_width,frame_center,
        frame_half_thickness=None,savefig=0,noaxis=0,**kwargs):
        raise Exception("Not tested!")

        self.firestudio_renderStellarGalaxy(
            ax,
            self.snapdir,self.snapnum,
            frame_width = frame_width,
            frame_half_thickness = frame_width if frame_half_thickness is None else frame_half_thickness,
            frame_center = frame_center,
            extract_galaxy=False,
            datadir = self.datadir,
            snapdict=self.sub_snap,
            savefig=savefig,noaxis=noaxis,
            **kwargs)

### helper functions
def plotSideBySide(
    rs,
    rcom,
    indices,
    weights=None,
    axs=None,
    **kwargs):

    if axs is None:
        fig,[ax1,ax2]=plt.subplots(1,2)
    else:
        fig = axs[0].get_figure()
        ax1,ax2=axs
    xs,ys,zs = (rs[indices]-rcom).T
    rs = np.sqrt(xs**2+ys**2+zs**2)

    twoDHist(ax1,xs,ys,bins=200,weights=weights,**kwargs)
    if 'cbar' in kwargs:
        kwargs.pop('cbar')
    twoDHist(ax2,xs,zs,bins=200,weights=weights,**kwargs)
    fig.set_size_inches(12,6)
    fig.set_facecolor('white')
    nameAxes(ax1,None,'x (kpc)','y (kpc)')
    nameAxes(ax2,None,'x (kpc)','z (kpc)')
    return fig,ax1,ax2

def fauxrenderPatch(sub_snap,ax,
    frame_center,frame_width,
    frame_half_thickness=None,savefig=0,noaxis=0,
    theta=0,phi=0,psi=0,**kwargs):

    indices = extractRectangularVolumeIndices(sub_snap['p'],
        frame_center,frame_width,frame_width if frame_half_thickness is None else frame_half_thickness)

    pos = sub_snap['p'] - frame_center # want to rotate about frame_center
    pos_rot = rotateEuler(theta,phi,psi,pos) +frame_center # add back the offset post rotation...?

    xs,ys = pos_rot[:,:2][indices].T

    twoDHist(ax,xs,ys,bins=1200)

    ax.set_ylim(frame_center[1]-frame_width,frame_center[1]+frame_width)
    ax.set_xlim(frame_center[0]-frame_width,frame_center[0]+frame_width)

    if noaxis:
        ax.axis('off')
    return ax

def twoDHist(
    ax,
    xs,ys,
    bins,
    weights=None,
    norm='',
    cbar=0,
    vmin=1,
    vmax=5e3,):
    if norm=='':
        from matplotlib.colors import LogNorm
        norm=LogNorm(vmin=vmin,vmax=vmax)
    cmap=plt.get_cmap('afmhot')

    h,xedges,yedges=np.histogram2d(
        xs,ys,
        weights=weights,
        bins=bins)
    
    ax.imshow(h.T,cmap=cmap,origin='lower',
        norm=norm,
        extent=[min(xedges),max(xedges),min(yedges),max(yedges)])

    if cbar:
        addColorbar(
            ax,cmap,
            vmin,vmax,
            r'$N_{particles}$' if weights is None else 'N . weight',
            logflag = 0,
            fontsize=12,
            cmap_number=0)

    return h,xedges,yedges

def rotateEuler(
    theta,phi,psi,
    pos,frame_center):

    ## if need to rotate at all really -__-
    if theta==0 and phi==0 and psi==0:
        return pos
    # rotate particles by angle derived from frame number
    theta_rad = np.pi*theta/ 1.8e2
    phi_rad   = np.pi*phi  / 1.8e2
    psi_rad   = np.pi*psi  / 1.8e2

    # construct rotation matrix
    #print('theta = ',theta_rad)
    #print('phi   = ',phi_rad)
    #print('psi   = ',psi_rad)

    ## explicitly define the euler rotation matrix 
    rot_matrix = np.array([
	[np.cos(phi_rad)*np.cos(psi_rad), #xx
	    -np.cos(phi_rad)*np.sin(psi_rad), #xy
	    np.sin(phi_rad)], #xz
	[np.cos(theta_rad)*np.sin(psi_rad) + np.sin(theta_rad)*np.sin(phi_rad)*np.cos(psi_rad),#yx
	    np.cos(theta_rad)*np.cos(psi_rad) - np.sin(theta_rad)*np.sin(phi_rad)*np.sin(psi_rad),#yy
	    -np.sin(theta_rad)*np.cos(phi_rad)],#yz
	[np.sin(theta_rad)*np.sin(psi_rad) - np.cos(theta_rad)*np.sin(phi_rad)*np.cos(psi_rad),#zx
	    np.sin(theta_rad)*np.cos(psi_rad) - np.cos(theta_rad)*np.sin(phi_rad)*np.sin(psi_rad),#zy
	    np.cos(theta_rad)*np.cos(phi_rad)]#zz
	],dtype=np.float32)
	    
    ## translate the particles so we are rotating about the center of our frame
    pos-=frame_center

    ## rotate each of the vectors in the pos array
    pos = np.dot(rot_matrix,pos.T).T

    # translate particles back
    pos+=frame_center

    pos_rot = copy.copy(pos)
    del pos
    
    return pos_rot
