from matplotlib.patches import Ellipse
import numpy as np

from ..plot_utils import nameAxes,plt
from ..array_utils import findIntersection
from ..fitting_utils import fitExponential
from ..function_utils import append_function_docstring

from .metadata_utils import metadata_cache

class Plot_ScaleHeight(object):
    
    def draw_inertia_ellipsoid(
        self,
        subtitle='',
        mask=None,
        component='gas',
        rmax=None,
        **kwargs):
     
        ## handle default arguments
        if rmax is None: rmax = self.scale_height_rmax_rvir*self.rvir

        if component == 'gas': which_snap = self.sub_snap
        else: which_snap = self.sub_star_snap

        if mask is None:
            ## select only particles within rmax and those not in the halo (if gas)
            mask = np.sum(which_snap['Coordinates']**2,axis=1)**0.5 < rmax

        ## draw galaxy background
        if component == 'gas': fig,ax1,ax2 = self.drawGasGalaxy(mask=mask)
        else: fig,ax1,ax2 = self.drawStellarGalaxy(mask=mask)
        lengths,evecs,angles_rad = self.get_inertia_ellipsoid(component=component,rmax=rmax,**kwargs)
        
        for i in list(range(2))[::-1]:
            these_evecs = evecs[3*i:3*(i+1)]
            these_lengths = lengths[3*i:3*(i+1)]
            
            ## unpack eigenvectors and eigenvalues
            a,b,c = these_lengths 
            e1,e2,e3 = these_evecs 

            print("a=%.2f"%a,e1)
            print("b=%.2f"%b,e2)
            print("c=%.2f"%c,e3)
        
            ## convert to degrees
            angles = angles_rad/np.pi*180 

            ax1.add_patch(Ellipse(
                (0,0),
                a*2*np.sqrt(e1[0]**2+e1[1]**2),
                b*2*np.sqrt(e2[0]**2+e2[1]**2),
                fill=None,
                color='darkblue',
                lw=3,
                angle=angles[0]))    
        
            ax1.plot([0,e1[0]*a],[0,e1[1]*a],color='darkgreen',lw=3)
            ax1.plot([0,e2[0]*b],[0,e2[1]*b],color='darkblue',lw=3)
        
            #ax1.plot([0,a*np.cos(angles_rad[0])],[0,a*np.sin(angles_rad[0])],color='darkblue',lw=3)
            #ax1.plot([0,b*np.cos(angles_rad[0]+np.pi/2)],[0,b*np.sin(angles_rad[0]+np.pi/2)],color='darkblue',lw=3)

            ax2.add_patch(Ellipse(
                (0,0),
                a*2*np.sqrt(e1[0]**2+e1[-1]**2),
                c*2*np.sqrt(e3[0]**2+e3[-1]**2),
                fill=None,
                color='darkblue',
                lw=3,
                angle=angles[-1]))
        
            ax2.plot([0,e1[0]*a],[0,e1[-1]*a],color='darkgreen',lw=3)
            ax2.plot([0,e3[0]*c],[0,e3[-1]*c],color='darkblue',lw=3)
        
            #ax2.plot([0,a*np.cos(angles_rad[-1])],[0,a*np.sin(angles_rad[-1])],color='darkblue',lw=3)
            #ax2.plot([0,c*np.cos(angles_rad[-1]+np.pi/2)],[0,c*np.sin(angles_rad[-1]+np.pi/2)],color='darkblue',lw=3)
        
        fig.set_size_inches(16,8)
        nameAxes(
            ax1,None,None,None,
            subtitle=r"$\theta_x=%.2f$ deg"%angles[0])
        nameAxes(
            ax2,None,None,None,
            subtitle="a=%.2f\nc=%.2f\nc/a=%.2f\n"%(max(a,b),c,c/max(a,b))+r"$\theta_z=%.2f$ deg"%angles[-1],
            supertitle=subtitle)
        return fig,[ax1,ax2]
    
    def plot_exponential_profiles(self,axs=None,component='gas',rmax=None,zmax=None,**kwargs):
        if axs is None:
            fig,axs = plt.subplots(nrows=1,ncols=2)
        else: fig = axs[0].get_figure()

        fig.subplots_adjust(wspace=0.25)
 
        radius,height,b_r,b_h=self.get_exponential_radius_and_height(component=component,rmax=rmax,zmax=zmax,**kwargs)

        ## handle default arguments
        if rmax is None: rmax = self.scale_height_rmax_rvir*self.rvir
        if zmax is None: zmax = self.scale_height_rmax_rvir*self.rvir

        ## figure out which particles to use
        if component == 'gas': which_snap = self.sub_snap
        elif 'star' in component: which_snap = self.sub_star_snap
        
        coords,masses = which_snap['Coordinates'],which_snap['Masses']

        rs = np.sum(coords**2,axis=1)**0.5
        redges = np.logspace(np.log10(rmax)-2,np.log10(rmax),40)

        mask = rs < redges[-1]

        h,_ = np.histogram(
            rs[mask],
            bins=redges,
            weights=masses[mask])

        ## fit the exponential profile to the radial surface density profile
        ##  convert to surface area
        dA = np.diff(redges**2)*np.pi
        xs,ys = redges[1:],h/dA

        ax = axs[0]
        ax.plot(xs,ys)
        ax.plot(xs,b_r*np.exp(-1/radius*xs),label="$%.2g e^{-x/%.4g}$"%(b_r,radius))
        nameAxes(ax,None,'R [kpc]',None,make_legend=True)

        ## --------- fit the scale height now
        ## mix above and below the midplane
        zs = np.abs(coords[:,-1])
        zedges = np.logspace(np.log10(zmax)-2,np.log10(zmax),40)

        h,_ = np.histogram(
            zs[mask],
            bins=zedges,
            weights=masses[mask])

        ## fit  the exponential profile to the vertical mass profile
        xs,ys = zedges[1:],h/np.diff(zedges)

        ax = axs[1]
        ax.plot(xs,ys)
        ax.plot(xs,b_h*np.exp(-1/height*xs),label="$%.2g e^{-x/%.4g}$"%(b_h,height))
        nameAxes(ax,None,'z [kpc]',None,make_legend=True)

        return fig,axs

class ScaleHeight_helper(Plot_ScaleHeight):
    """------- ScaleHeight_helper 
    """

    scale_height_rmax_rvir = 0.1

    def calculate_half_mass_radius(
        self,
        which_snap=None,
        geometry='spherical',
        within_radius=None,
        mask=None):

        within_radius = self.rvir if within_radius is None else within_radius

        ## find the stars within the virial radius
        if which_snap is None:
            which_snap = self.star_snap

        if 'overwritten' in which_snap.keys() and which_snap['overwritten']:
            coords = which_snap['Coordinates']
        else:
            coords = which_snap['Coordinates']-self.scom

        masses = which_snap['Masses']

        if mask is not None:
            coords = coords[mask]
            masses = masses[mask]

        edges = np.linspace(0,within_radius,5000,endpoint=True)

        if geometry in ['cylindrical','scale_height']:
            radii = np.sum(coords[:,:2]**2,axis=1)**0.5
        elif geometry == 'spherical':
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

        return findIntersection(edges[1:],cdf,0.5)[0]

    def run_ScaleHeight_helper(self,rmax_rvirs=None,**kwargs):
        """ Run all calculations in the ScaleHeight_helper module, contains:"""

        if rmax_rvirs is None: rmax_rvirs = np.linspace(0.05,1,20)

        for component in self.components:
            for rmax_rvir in rmax_rvirs:
                self.get_simple_radius_and_height(component,rmax_rvir=rmax_rvir,**kwargs)
                self.get_simple_radius_and_height('ann_'+component,rmax_rvir=rmax_rvir,**kwargs)
                self.get_inertia_ellipsoid(component,rmax_rvir=rmax_rvir,**kwargs)
                self.get_exponential_radius_and_height(component,rmax_rvir=rmax_rvir,**kwargs)
                self.get_angmom_radius_and_height(component,rmax_rvir=rmax_rvir,**kwargs)
    
    def get_simple_radius_and_height(
        self,
        component='gas',
        rmax_rvir=0.1,
        save_meta=True,
        use_metadata=True,
        loud=True,
        **kwargs):

        group_name = 'SimpleGeom_%s'%component

        @metadata_cache(
            group_name,
            ['%s_half_r_%.2f_rvir'%(component,rmax_rvir),
            '%s_quarter_h_%.2f_rvir'%(component,rmax_rvir)],
            use_metadata=use_metadata,
            save_meta=save_meta,
            loud=loud)
        def compute_simple_radius_and_height(self,component,rmax=None):

            annulus = False
            ## do in an annulus
            if 'ann_' in component:
                annulus = True
                component = component[4:]
            
            if rmax is None: rmax = self.scale_height_rmax_rvir*self.rvir

            ## pick particles in this phase/particle type
            which_snap = self.particle_picker(component)

            mask = None

            ## calculate the cylindrical half-mass radius using mass 
            ##  within 20% virial radius
            radius = self.calculate_half_mass_radius(
                which_snap=which_snap,
                geometry='cylindrical',
                within_radius=rmax,
                mask=mask)
            
            if not annulus: rmax = radius
            else:
                rmax = 1.25*radius
                new_mask = np.sum(which_snap['Coordinates'][:,:2]**2,axis=1) > (0.75*radius)**2
                if mask is None: mask = new_mask
                else: mask = np.logical_and(mask,new_mask)


            ## calculate the half-mass height w/i cylinder
            ##  of radius previously calculated
            height = self.calculate_half_mass_radius(
                which_snap=which_snap,
                geometry='scale_height',
                within_radius=rmax,
                mask=mask)
                
            return radius,height
        return compute_simple_radius_and_height(self,component,rmax=rmax_rvir*self.rvir,**kwargs)

    def get_inertia_ellipsoid(
        self,
        component='gas',
        rmax_rvir=0.1,
        use_metadata=True,
        save_meta=False,
        loud=True,
        assert_cached=False,
        force_from_file=False,
        **kwargs):
    
        @metadata_cache(
            "%s_inertia_ellipsoid_%.2f_rvir"%(component,rmax_rvir),
            ["%s_lengths_%.2f_rvir"%(component,rmax_rvir),
            "%s_evecs_%.2f_rvir"%(component,rmax_rvir),
            "%s_angles_rad_%.2f_rvir"%(component,rmax_rvir)],
            use_metadata=use_metadata,
            save_meta=save_meta,
            loud=loud,
            assert_cached=assert_cached,
            force_from_file=force_from_file)
        def compute_inertia_ellipsoid(self,component='gas',rmax=None):

            if rmax is None: rmax = self.scale_height_rmax_rvir*self.rvir

            ## pick particles in this phase/particle type
            which_snap = self.particle_picker(component)

            ## select only particles within rmax and those not in the halo (if gas)
            mask = np.sum(which_snap['Coordinates']**2,axis=1)**0.5 < rmax
                
            masses,coords = which_snap['Masses'][mask],which_snap['Coordinates'][mask]

            if masses.size <= 1:
                if rmax < self.rvir*0.5:
                    return compute_inertia_ellipsoid(self,component,rmax=2*rmax)
                else:
                    return np.repeat(np.nan,6),np.repeat(np.nan,18).reshape(-1,3),np.repeat(np.nan,2)
            lengths,evecs = computeClumpRadius(masses,coords)

            e1 = evecs[0]
                 
            angles_rad = np.array([np.arctan2(e1[1],e1[0]),np.arctan2(e1[-1],e1[0])])
            angles_rad[angles_rad<-np.pi/2] = np.pi+angles_rad[angles_rad<-np.pi/2]

            return lengths,evecs,angles_rad
            
        return compute_inertia_ellipsoid(self,component=component,rmax=rmax_rvir*self.rvir,**kwargs)
    
    def get_exponential_radius_and_height(
        self,
        component='gas',
        rmax_rvir=0.1,
        use_metadata=True,
        save_meta=False,
        loud=True,
        assert_cached=False,
        force_from_file=False,
        **kwargs):
    
        @metadata_cache(
            "%s_exponential_scale_height_%.2f_rvir"%(component,rmax_rvir),
            ["%s_exponential_radius_%.2f_rvir"%(component,rmax_rvir),
            "%s_exponential_height_%.2f_rvir"%(component,rmax_rvir),
            "%s_radius_norm_%.2f_rvir"%(component,rmax_rvir),
            "%s_height_norm_%.2f_rvir"%(component,rmax_rvir)],
            use_metadata=use_metadata,
            save_meta=save_meta,
            loud=loud,
            assert_cached=assert_cached,
            force_from_file=force_from_file)
        def compute_exponential_radius_and_scale_height(
            self,
            rmax=None,
            zmax=None,
            component='gas'):

            ## handle default arguments
            if rmax is None: rmax = self.scale_height_rmax_rvir*self.rvir
            if zmax is None: zmax = self.scale_height_rmax_rvir*self.rvir

            which_snap = self.particle_picker(component)
            
            coords,masses = which_snap['Coordinates'],which_snap['Masses']

            rs = np.sum(coords**2,axis=1)**0.5
            redges = np.logspace(np.log10(rmax)-2,np.log10(rmax),40)

            mask = rs < redges[-1]

            h,_ = np.histogram(
                rs[mask],
                bins=redges,
                weights=masses[mask])

            ## fit the exponential profile to the radial surface density profile
            ##  convert to surface area
            dA = np.diff(redges**2)*np.pi
            xs,ys = redges[1:],h/dA
            if np.sum(ys>0) > 1: a_r,b_r = fitExponential(xs,ys)
            else: a_r,b_r = np.nan,np.nan

            ## --------- fit the scale height now
            ## mix above and below the midplane
            zs = np.abs(coords[:,-1])
            zedges = np.logspace(np.log10(zmax)-2,np.log10(zmax),40)

            h,_ = np.histogram(
                zs[mask],
                bins=zedges,
                weights=masses[mask])

            ## fit  the exponential profile to the vertical mass profile
            xs,ys = zedges[1:],h/np.diff(zedges)

            if np.sum(ys>0) > 1: a_h,b_h = fitExponential(xs,ys)
            else: a_h,b_h = np.nan,np.nan

            return -1/a_r,-1/a_h,np.exp(b_r),np.exp(b_h)
            

        return compute_exponential_radius_and_scale_height(self,component=component,rmax=rmax_rvir*self.rvir,**kwargs)

    def get_angmom_radius_and_height(
        self,
        component='gas',
        rmax_rvir=0.1,
        use_metadata=True,
        save_meta=False,
        loud=True,
        assert_cached=False,
        force_from_file=False,
        **kwargs):
    
        @metadata_cache(
            '%s_angmom_scale_height_%.2f_rvir'%(component,rmax_rvir),
            ['%s_angmom_radius_%.2f_rvir'%(component,rmax_rvir),
            '%s_angmom_height_%.2f_rvir'%(component,rmax_rvir),
            '%s_angmom_vector_%.2f_rvir'%(component,rmax_rvir)],
            use_metadata=use_metadata,
            save_meta=save_meta,
            loud=loud,
            assert_cached=assert_cached,
            force_from_file=force_from_file)
        def compute_angmom_radius_and_height(self,component='gas',rmax=None,):
    
            if rmax is None: rmax = self.scale_height_rmax_rvir*self.rvir
                
            ## apply component cut
            which_snap = self.particle_picker(component)
            
            coords = which_snap['Coordinates']
            r2s = np.sum(coords**2,axis=1)

            ## for applying radial cut
            rmask = r2s<(rmax**2)

            ## unpack snapshot values we need
            coords = coords[rmask]
            masses = which_snap['Masses'][rmask]
            mtot = np.sum(masses)
            vels = which_snap['Velocities'][rmask]
            r2s = r2s[rmask]

            ## calculate angular momentum vector for this component
            angmom = np.sum(np.cross(coords,vels*masses[:,None]),axis=0)
            angmom /= np.linalg.norm(angmom)

            ## find (squared) lengths along this axis 
            l_parallel2s = np.sum(coords*angmom,axis=1)**2

            ## find (squared) perpendicular lengths (and apply radial cut)
            l_perp2s = r2s-l_parallel2s
            
            ## mass weighted RMS height/radius
            height = np.sqrt(np.sum(l_parallel2s*masses)/mtot)
            radius = np.sqrt(np.sum(l_perp2s*masses)/mtot)

            return radius,height,angmom
        return compute_angmom_radius_and_height(self,rmax=rmax_rvir*self.rvir,component=component,**kwargs)

    ## handle programatic function docstrings
    append_function_docstring(run_ScaleHeight_helper,get_simple_radius_and_height)
    append_function_docstring(run_ScaleHeight_helper,get_inertia_ellipsoid)
    append_function_docstring(run_ScaleHeight_helper,get_exponential_radius_and_height)

### HELPER FUNCTIONS
def isolate_lengths(inertia,Is=None):
    if Is is None:
        ## get specific moments of inertia about x,y, and z axes
        Is = np.zeros(3)
        for i in range(3):
            arr = np.zeros(3)
            arr[i] = 1
            Is[i] = np.dot(np.dot(arr,inertia),arr)
        
    ## specific moments of inertia are combinations of lx^2, ly^2, and lz^2
    ##  here we will combine them to isolate lengths along individual axes
    lengths = np.zeros(3)
    ##  x = [(x + z) - (y + z) + (y + x)]/2
    lengths[0] = np.sqrt((Is[1]- Is[0]+Is[2])/2)
    ##  y = [(y + z) - (x + z) + (y + x)]/2
    lengths[1] = np.sqrt((Is[0]- Is[1]+Is[2])/2)
    ##  z = [(x + z) - (x + y) + (y + z)]/2
    lengths[2] = np.sqrt((Is[1]- Is[2]+Is[0])/2)
    return lengths

def computeClumpRadius(masses,coords):
    r2 = np.sum(coords**2,axis=1)
    inertia = np.identity(3)*np.sum(r2*masses)/np.sum(masses)-np.cov(coords.T*np.sqrt(masses))/(np.sum(masses)/(masses.size-1))
 
    Is,evecs = np.linalg.eig(inertia)
    
    ## first order eigenvectors by decreasing size
    sort_indices = np.argsort(Is)[::-1]
    evecs = evecs.T[sort_indices]
    Is = Is[sort_indices]
    
    ## ensure eigenvalues have opposite rank ordering as x,y,z RMS lengths
    Is_axs = np.sqrt(np.sum(coords**2*masses[:,None],axis=0)/np.sum(masses))*2
    
    ## figure out rank ordering brute force
    max_index = np.argmax(Is_axs)
    min_index = np.argmin(Is_axs)
    mid_index = list(set([0,1,2])-set([min_index,max_index]))[0]
    
    ## assign sorted Is indices (min = 0 , mid = 1, max = 2)
    ##  to whatever index corresponds to min,mid,max is in Is_axs
    sort_indices = np.zeros(3,dtype=int)
    sort_indices[min_index]=0
    sort_indices[mid_index]=1
    sort_indices[max_index]=2
    
    ## reorder the eigenvectors and moments of inertia
    evecs = evecs[sort_indices]
    Is = Is[sort_indices]
    
    ## initialize output arrays
    all_evecs = np.zeros((6,3))
    lengths = np.zeros(6)

    ## fill eigenvectors for principle axes
    all_evecs[:3,:] = evecs
    ## fill eigenvectors for x,y,z
    all_evecs[3:,:] = np.identity(3)
    
    ## find lengths along principle axes
    lengths[:3] = isolate_lengths(inertia,Is)
    ## find lengths along x,y,z
    lengths[3:] = isolate_lengths(inertia,None)

    ## okay hear me out on this one, the root (mass-weighted) mean square
    ##  radius turns out to be ~1/2 the extent of a uniform distribution.
    ##  (this is equivalent to getting "effective" lenths from
    ##  moments of inertia)
    ##  consider the following example:
    ## > masses = np.ones(20)
    ## > rs = np.arange(20)
    ## > reff = np.sqrt(np.sum(masses*rs**2)/np.sum(masses))
    ## reff = 11.113, in this example *i think* the length we really care about
    ##  is 20, but ofc the inner masses are going to pull the effective radius in (especially
    ##  the case when distrib is not uniform but centrally concentrated). 
    ##  so multiplying by 2 gets you the "right" answer in the uniform case ("is necessary")
    ##  so we'll do it in all cases :\ doesn't sound very convincing tbh
    #lengths*=2
        
    return lengths,all_evecs