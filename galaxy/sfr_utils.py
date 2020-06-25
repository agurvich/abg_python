import numpy as np 
import h5py
import os

import matplotlib.pyplot as plt

## from abg_python
from abg_python.plot_utils import add_to_legend

from abg_python.cosmo_utils import approximateRedshiftFromGyr

import abg_python.all_utils as all_utils

from abg_python.galaxy.metadata_utils import metadata_cache

class SFR_plotter(object):
    """------- SFR_plotter
    """
    
    def addSFRToAx(
        self,
        ax=None,
        only_axis=0,
        bump_pix=0,
        savefig=False,
        **kwargs):

        if ax is None:
            ax = plt.gca()
        fig = ax.get_figure()

        ## x,y of bottom left corner, width,height in percentage of figure size
        ## matches the default aspect ratio of matplotlib
        cur_size = fig.get_size_inches()*fig.dpi        
        cur_height = cur_size[1]
        cur_width = cur_size[0]
        
        offset_pix = 25
        height_pix = 100.

        xoffset = (15.+ bump_pix)/cur_width 
        width = 1-(10./cur_width)-xoffset

        ## should we add an axis to the figure or should we 
        ##  just plot to a single axis...
        ax1 = fig.add_axes([xoffset, 1.+(offset_pix)/cur_height, width, height_pix/cur_height])

        self.plotSFH(ax1,**kwargs)

        fig.tight_layout()
        if savefig:
            fig.savefig(savefig,
                bbox_extra_artists=[ax1],bbox_inches='tight')
            plt.close(fig)
        return ax1

    def plot_SFH(
        self,
        ax,
        near=None, ## centers SFH on the snapshot's current time, pass near=window_size
        specific=0, ## divide by integrated SFR
        bursty=0, ## plot a bursty line
        redshift_xs=0, ## plot as log(1+z) but label w/ z on x-axis
        snapshot_xs=0, ## plot snapshot numbers on the x-axis
        renormed=None, ## normalize by the running average
        color=None, ## what color to plot the SFH w/ 
        DT=0.001,
        plot_grayBand=False,
        **axkwargs
        ):
        """ near=None does not change x-axis limits to center on current time. 
                near=window_size will plot window_size/2 on either side of current time
            renormed=None does not renormalize the SFH, renormed=DT will renormalize by the 
            DT running average. """

        plot_color = self.plot_color if plot_color is None else plot_color

        ## just load up the 1 Myr SFR, it's
        if not hasattr(self,'SFH_dt') or self.SFH_dt!=0.001:
            try:
                self.get_(DT=0.001,assert_cached=True)
            except AssertionError:
                raise ValueError("Compute the SFH first using galaxy.get_SFH(DT=0.001)")
        
            xs,ys = all_utils.boxcar_average(
                self.SFH_time_edges,
                self.SFRs,
                DT)
            xs = xs[1:]
        else:
            xs = self.SFH_time_edges[1:]
            ys = self.SFRs

        ylabel = 'SFR (M$_\odot$/yr)'
        xlabel = 't (Gyr)'

        ## do we need to change the y-axis values at all?
        if renormed is not None:
            ## you want us to renormalize by the running average
            xs,long_ys = all_utils.boxcar_average(
                self.SFH_time_edges,
                self.SFRs,
                renormed)
            ys = ys/long_ys
            ylabel = r'SFR/$\langle SFR \rangle_{%d\,\mathrm{Myr}}$'%(renormed*1e3)
        elif specific: 
            ## you want us to divide by the integrated SFR
            mass_subset_sum = [ys[0]]
            for y in ys[1:]:
                mass_subset_sum+=[mass_subset_sum[-1]+y]
            mass_subset_sum = np.array(mass_subset_sum)*(xs[1]-xs[0])*1e9 #msun_formed(t)
            ys = ys / mass_subset_sum *1e9 # gyr-1

            ylabel = 'sSFR ($Gyr^{-1}$)'

        ## do we need to change the x-axis values at all?
        if redshift_xs: 
            ## you want us to plot by log(1+z) but label by z
            xs = np.log10(approximateRedshiftFromGyr(
                self.header['HubbleParam'],self.header['Omega0'],xs)+1)

            if zticks is not None:
                ax.set_xticks(np.log10(1+np.array(zticks)))
                ax.set_xticklabels(zticks)
            else:
                ax.set_xticks(np.linspace(min(xs),max(xs),10))
                ax.set_xticklabels(["%.3f"% (10**z-1) for z in np.linspace(min(xs),max(xs),10)])

            xlabel = 'z' 
        elif snapshot_xs:
            ## requires snapshot_times.txt to exist...
            ax.set_xticks(self.snap_gyrs[::snapshot_xs])
            ax.set_xticklabels(["%d"%snapnum  for snapnum in self.snapnums[::snapshot_xs]],rotation=-45)
            xlabel = ''

        ## actually plot the damn thing
        ax.plot(xs,ys,lw=1,
            color = plot_color)

        ## sprinkles on top, plot overlays that might be useful
        if bursty:
            ## should we plot a vertical line at the bursty time?
            bursty_index,bursty_time = self.get_bursty_regime()
            ax.axvline(xs[bursty_index],c=plot_color,ls='--',alpha=0.65)

        if plot_grayBand:
            ## should we plot a band showing the running average of the
            ##  the SFR +- a factor of sqrt(3)?
            if renormed:
                ax.fill_between(
                    xs,
                    np.ones(xs.size)/factor**0.5,
                    np.ones(xs.size)*factor**0.5,
                    lw=0,
                    color='gray',
                    alpha=0.15)
            else:
                self.plot_graySFRBand(
                    ax,
                    want_redshift_xs=redshift_xs,
                    color='gray')
    

        if near is not None:
            ## if near, plot the bead and crop the plot
            ## find the time on the SFH that is closest to the current time
            cur_index,ti = findIntersection(
                np.arange(self.SFRs.size),
                self.SFH_time_edges[1:],
                self.current_time_Gyr)

            sfri,xi=ys[cur_index],xs[cur_index]

            ## done obliquely this way to account for possibility of redshift xs
            ##  but tbh shouldn't really be doing near for redshift xs so....
            ##  automatically handles if xl or xr is outside SFH_time_edges
            xl = xs[np.argmin(
                ((self.SFH_time_edges[1:]-near/2)-self.SFH_time_edges)**2)]
            xr = xs[np.argmin(
                ((self.SFH_time_edges[1:]+near/2)-self.SFH_time_edges)**2)]

            ax.plot(xi,sfri,'ro',markeredgewidth=0,markersize=8)
            ax.set_xlim(xl,xr)

        nameAxes(ax,None,xlabel,ylabel,logflag=(0,1),**axkwargs)
        return ax

    def plot_graySFRBand(
        self,
        ax=None,
        window_size=0.3,
        want_redshift_xs=0,
        color=None,
        factor=3):

        color = self.plot_color if color is None else color

        try:
            self.get_SFH(DT=0.001,assert_cached=True)
        except AssertionError:
            raise ValueError("Compute the SFH first using galaxy.get_SFH(DT=0.001)")

        ## calculate the running average to be plotted as a gray bar
        xs,avg_long = all_utils.boxcar_average(
            self.SFH_time_edges, self.SFRs, window_size)

        if ax is not None:
            if want_redshift_xs: 
                xs = np.log10(approximateRedshiftFromGyr(
                    self.header['HubbleParam'],self.header['Omega0'],xs)+1)

            ## if we already divided by this function then we just want a constant window
            ax.fill_between(
                xs,
                ys/factor**0.5,
                ys*factor**0.5,
                lw=0,
                color=color,
                alpha=0.15)

        return ax

class SFR_helper(SFR_plotter):
    """------- SFR_helper
    """

    __doc__+="\n"+SFR_plotter.__doc__

    def get_sfr_string(self,DT):
        if DT > 0:
            sfr_string = "%dMyr"%(DT*1e3)
        else:
            sfr_string = "pcarry_inst"   
        return sfr_string

    def get_SFH(
        self, 
        use_metadata=True,
        save_meta=False,
        assert_cached=False,
        loud=True,
        DT = 0.001,
        **kwargs):

        sfr_string = self.get_sfr_string(DT)
        ### begin wrapped
        @metadata_cache(
            "SFR_%s"%sfr_string,[
                'SFH_time_edges',
                "SFRs",
                "SFR_metals",
                "SFH_dt"],
            use_metadata=use_metadata,
            save_meta=save_meta,
            assert_cached=assert_cached,
            loud=loud)
        def compute_SFH(
            self,
            radial_thresh=None):
            """radial_thresh = None -> spherical cut of 5*rstar_half"""
            
            ## initialize the spherical radial threshold
            radial_thresh = 5*self.rstar_half if radial_thresh is None else radial_thresh

            finsnap = self.finsnap 

            if finsnap != self.snapnum:
                print("Already opened the final snapshot!")
                temp_fin_gal = self
            else:
                from abg_python.galaxy.gal_utils import Galaxy
                ## have to open a whole new galaxy object!!
                temp_fin_gal = Galaxy(
                    self.name,
                    self.snapdir,
                    finsnap,
                    cosmological=self.cosmological,
                    datadir=self.datadir,
                    data_name=self.data_name,
                    ahf_path=self.ahf_path,
                    ahf_fname=self.ahf_fname)
                print(temp_fin_gal,'loaded to compute SFR archaeologically')


            ## do we need to extract the sub_star_snap? if 
            ##  finsnap is self.snapnum maybe not...
            if 'sub_star_snap' not in temp_fin_gal.__dict__.keys():
                temp_fin_gal.extractMainHalo(free_mem=1,extract_DM=0)

            ## apply the radial mask
            rmask = np.sum(star_snap['Coordinates']**2,axis=1)<radial_thresh**2
            star_snap = all_utils.filterDictionary(self.sub_star_snap,rmask)

            ## get initial stellar masses if possible, otherwise estimate them by "undoing" the 
            ##  integrated STARBURST99 mass loss rates
            smasses = star_snap['Masses'].astype(np.float64)*1e10 # solar masses, assumes 20% mass loss
            ages = star_snap['AgeGyr']  # gyr, as advertised
            smasses = all_utils.get_IMass(ages,smasses) # uses a fit function to figure out initial mass from current age

            ## calculate the star formation history
            SFTs,timemax = temp_fin_gal.current_time_Gyr - star_snap['AgeGyr'],temp_fin_gal.current_time_Gyr
            SFRs,time_edges = arch_method(
                smasses,
                SFTs,
                timemax,
                tmin=1e-16,
                DT=DT)

            ## calculate the stellar metallicity history-- which Katie Breivik once asked me for
            ##  /shrug
            metals = star_snap['Metallicity'][:,0] # mass fractions
            metal_SFRs,time_edges = arch_method(smasses*metals,SFTs,timemax,tmin=tmin,DT=DT)
            metal_History = metal_SFRs/SFRs # metal masses / total masses -> metal mass fraction in each bin
 
            ## free up temporary galaxy memory
            if temp_fin_gal is not self:
                del temp_fin_gal

            return SFH_time_edges, SFRs, SFR_metals, DT

        compute_SFH(self,**kwargs)

        ## output the SFH and metallicity history to an external "catalog"
        ##  file so it's easier to share (with say, Jonathan, for instance)
        catpath = os.path.join(self.datadir,'sfrcat_%d_%s'%(finsnap,sfr_string))
        if not os.path.isfile(catpath):
            sfrcat = {'metals':metal_History,'sfrs':SFRs,'time_edges':time_edges}
            print("outputting sfrcat to:",catpath,'radial_thresh:',radial_thresh)
            np.savez(catpath,**sfrcat)

        return self.SFH_time_edges, self.SFRs, self.SFR_metals, self.SFH_dt

    def downsample_SFH(
        self,
        DT_downsample=0.1,
        same_size=False):

        ## save the current SFH properties in case the below works
        orig = self.SFH_time_edges,self.SFRs,self.SFR_metals,self.SFH_dt
        try:
            ## see if we have already calculated this DT, in which case, 
            ##  just use it... (typically unlikely, though)
            return_value = self.get_SFH(save_meta=False,
                assert_cached=True,
                DT = DT_downsample)

            ## unpack the original values
            self.SFH_time_edges,self.SFRs,self.SFR_metals,self.SFH_dt = orig
            return return_value

        except AssertionError:
            ## if we haven't already loaded the 1 Myr SFH, go grab it
            if not hasattr(self,'SFH_dt') or self.SFH_dt != 0.001:
                self.get_SFH(DT = 0.001)

        ## number of dt sized bins in each of our wider DT sized bins
        nbins_to_merge = int(DT_downsample//0.001)

        ## number of DT sized bins we'll be able to make
        nbins = int(self.global_sfrs.size//nbins_to_merge)

        ## remainder of equal division is the number of 
        ##  dt sized bins we have have to skip (at the beginning)
        skip_index = self.global_sfrs.size%nbins

        ## group times associated w/ each SFR into nbins many 
        ##  equally sized groups, the new time associated with 
        ##  these wider bins is the right-most (max)
        downsampled_times = np.max(
            ## times[1:] are right edges of bins
            ##  want to skip skip_index many of them to make for 
            ##  equally sized groups
            np.split(self.global_sfrs_times[1:][skip_index:],nbins),
            axis=-1)

        ## SFR in DT window is sum(SFRs*dt)/(DT) = 
        ##  sum(SFRs*dt)/(nbins*dt) = mean(SFRs)_nbins
        downsampled_sfrs = np.mean(
            ## split SFRs into nbins many equally sized bins
            np.split(self.global_sfrs[skip_index:],nbins), 
            axis=-1)

        ## hyper sample our new DT resolution SFH back to 
        ##  the original dt resolution SFH we were given 
        if same_size:
            ## take the original times, since that's what we want anyway
            downsampled_times = orig[0]

            ## repeat the SFRs (in place, *not* tiling) DT/dt many times 
            downsampled_sfrs = np.repeat(downsampled_sfrs,nbins_to_merge)

            ## the first skip_index many points are invalid since you can't average
            ##  over DT_downsample in the past for them. 
            downsampled_sfrs = np.append([np.nan]*skip_index,downsampled_sfrs)

            if (downsampled_times.size != self.global_sfrs_times[1:].size-1 or 
                downsampled_sfrs.size != self.global_sfrs.size):
                raise ValueError("Resampling at same size failed... for some reason...")

        ## unpack the original values in case we replaced them
        self.SFH_time_edges,self.SFRs,self.SFR_metals,self.SFH_dt = orig

        return downsampled_times, downsampled_sfrs

    ## analysis functions
    def get_bursty_regime(
        self,
        use_metadata=True,
        save_meta=False,
        assert_cached=False,
        loud=True,
        **kwargs):
        """ thresh=0.5, ## dex of scatter
            window=0.3, ## size of window to compute scatter within"""

        ### begin wrapped
        @metadata_cache(
            "bursty",[
                'bursty_index',
                'bursty_time',
                'bursty_redshift',
                'SFR_rel_scatter_hist'],
            use_metadata=use_metadata,
            save_meta=save_meta,
            assert_cached=assert_cached,
            loud=loud)
        def compute_bursty_regime(
            self,
            thresh=0.5, ## dex of scatter
            window=0.3, ## size of window to compute scatter within
            ):

            ## ensure that we have the 1 Myr SFH loaded
            self.get_SFH(DT=0.001,loud=False)

            ## calculate the relative scatter as sigma/y
            xs,avg_long = all_utils.boxcar_average(
                self.SFH_time_edges,
                np.log10(self.SFRs),
                window_size)

            xs2,avg_long_2 = all_utils.boxcar_average(
                self.SFH_time_edges,
                np.log10(self.SFRs)**2,window_size)

            rel_scatters = (sfh_long_2-sfh_long**2)**0.5/sfh_long

            ## have to reverse the rel_scatters to find the "last point of crossing" 
            ##  after which the rel_scatter doesn't cross the threshold. 
            tindex = np.argmax(rel_scatters[::-1] > thresh)
            tindex = rel_scatters.size-tindex-1

            bursty_redshift = approximateRedshiftFromGyr(
                self.header['HubbleParam'],
                self.header['Omega0'],
                xs[tindex])

            return tindex, xs[tindex], bursty_redshift, rel_scatters
        
        return compute_bursty_regime(self,**kwargs)

#### HELPER FUNCTIONS
def arch_method(smasses,SFTs,timemax,tmin=None,DT=None):

    if DT==None:
        DT = .001 #1 Myr in Gyr

    if tmin == None:
        tmin = np.min(SFTs)

    edges = np.arange(tmin,timemax+DT,DT)

    SFRs,time_edges = np.histogram(SFTs,weights=smasses/(DT*1e9),bins=edges) #solar masses/year, Myr
    assert (time_edges==edges).all()
    #ignore the last point because it will often contain a "remainder bin", not a full bin width
    return SFRs, time_edges
