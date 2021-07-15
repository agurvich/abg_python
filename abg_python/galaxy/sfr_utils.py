import numpy as np 
import h5py
import os

import matplotlib.pyplot as plt

## from abg_python
from abg_python.plot_utils import add_to_legend,nameAxes

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

        color = self.plot_color if color is None else color

        ## just load up the 1 Myr SFR, it's
        if not hasattr(self,'SFH_dt') or DT!=0.001:
            try:
                self.get_SFH(DT=0.001,assert_cached=True,loud=False)
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
            xs = xs[1:]
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

            ax.set_xticks(np.linspace(min(xs),np.log10(1+7),10))
            ax.set_xticklabels(["%.3f"% (10**z-1) for z in np.linspace(min(xs),np.log10(1+7),10)])
            ax.set_xlim(min(xs),np.log10(1+7))

            xlabel = 'z' 
        elif snapshot_xs:
            ## requires snapshot_times.txt to exist...
            ax.set_xticks(self.snap_gyrs[::snapshot_xs])
            ax.set_xticklabels(["%d"%snapnum  for snapnum in self.snapnums[::snapshot_xs]],rotation=-45)
            xlabel = ''

        ## actually plot the damn thing
        ax.plot(xs,ys,lw=1,
            color = color)

        ## sprinkles on top, plot overlays that might be useful
        if bursty:
            ## should we plot a vertical line at the bursty time?
            bursty_index,bursty_time,bursty_redshift,rel_scatters = self.get_bursty_regime(loud=False)
            
            ax.axvline(xs[bursty_index],c=color,ls='--',alpha=0.65)

        if plot_grayBand:
            factor = 2
            ## should we plot a band showing the running average of the
            ##  the SFR +- a factor of sqrt(3)?
            if renormed:
                ax.fill_between(
                    xs,
                    np.ones(xs.size)/factor,
                    np.ones(xs.size)*factor,
                    lw=0,
                    color='gray',
                    alpha=0.25)
            else:
                self.plot_graySFRBand(
                    ax,
                    want_redshift_xs=redshift_xs,
                    color='gray',
                    factor=factor)
    

        if near is not None:
            ## if near, plot the bead and crop the plot
            ## find the time on the SFH that is closest to the current time
            cur_index,ti = all_utils.findIntersection(
                np.arange(self.SFRs.size),
                self.SFH_time_edges[1:],
                self.current_time_Gyr)

            sfri,xi=ys[cur_index],xs[cur_index]

            ## done obliquely this way to account for possibility of redshift xs
            ##  but tbh shouldn't really be doing near for redshift xs so....
            ##  automatically handles if xl or xr is outside SFH_time_edges
            foo = ((self.SFH_time_edges[1:]-near/2)-self.SFH_time_edges[1:])**2
            xl = xs[np.argmin(
                ((self.current_time_Gyr-near/2)-self.SFH_time_edges[1:])**2)]
            xr = xs[np.argmin(
                ((self.current_time_Gyr+near/2)-self.SFH_time_edges[1:])**2)]

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
        factor=2): ## approx 0.3 dex

        color = self.plot_color if color is None else color

        try:
            self.get_SFH(DT=0.001,assert_cached=True,loud=False)
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
                xs[1:],
                avg_long/factor,
                avg_long*factor,
                lw=0,
                color=color,
                alpha=0.25)

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
        """ radial_thresh=None - spherical cut,defaults to 5*rstar_half"""


        sfr_string = self.get_sfr_string(0.001)
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
            loud=loud,
            force_from_file=True)
        def compute_SFH(
            self,
            radial_thresh=None):
            """ radial_thresh=None - spherical cut,defaults to 5*rstar_half"""
            
            ## initialize the spherical radial threshold
            radial_thresh = 5*self.rstar_half if radial_thresh is None else radial_thresh

            finsnap = self.finsnap 

            if finsnap == self.snapnum:
                print("Already opened the final snapshot!")
                temp_fin_gal = self
            else:
                from abg_python.galaxy.gal_utils import Galaxy
                ## have to open a whole new galaxy object!!
                temp_fin_gal = Galaxy(
                    self.name,
                    self.snapdir,
                    finsnap,
                    datadir=os.path.dirname(self.datadir),
                    datadir_name=self.datadir_name,
                    ahf_path=self.ahf_path,
                    ahf_fname=self.ahf_fname)
                print(temp_fin_gal,'loaded to compute SFR archaeologically')
                return temp_fin_gal.get_SFH(radial_thresh=radial_thresh)

            ## do we need to extract the sub_star_snap? if 
            ##  finsnap is self.snapnum maybe not...
            if 'sub_star_snap' not in temp_fin_gal.__dict__.keys():
                temp_fin_gal.extractMainHalo(free_mem=1,extract_DM=0)

            ## apply the radial mask
            rmask = np.sum(self.sub_star_snap['Coordinates']**2,axis=1)<radial_thresh**2

            star_snap = all_utils.filterDictionary(self.sub_star_snap,rmask)

            ## get initial stellar masses if possible, otherwise estimate them by "undoing" the 
            ##  integrated STARBURST99 mass loss rates
            smasses = star_snap['Masses'].astype(np.float64)*1e10 # solar masses
            ages = star_snap['AgeGyr']  # gyr, as advertised
            smasses = all_utils.get_IMass(ages,smasses) # uses a fit function to figure out initial mass from current age

            ## calculate the star formation history
            SFTs,timemax = star_snap['TimeGyr'] - star_snap['AgeGyr'],star_snap['TimeGyr']

            ## make sure our last bin ends at the current time
            time_edges = np.arange(star_snap['TimeGyr'],0,-DT)[::-1]

            SFRs,SFH_time_edges = arch_method(
                smasses,
                SFTs,
                time_edges,
                DT=DT)

            ## calculate the stellar metallicity history-- which Katie Breivik once asked me for
            ##  /shrug
            metals = star_snap['Metallicity'][:,0] # mass fractions
            SFR_metals,SFH_time_edges = arch_method(smasses*metals,SFTs,time_edges,DT=DT)
            SFR_metals = SFR_metals/SFRs # metal masses / total masses -> metal mass fraction in each bin
 
            ## free up temporary galaxy memory
            if temp_fin_gal is not self:
                del temp_fin_gal

            ## output the SFH and metallicity history to an external "catalog"
            ##  file so it's easier to share (with say, Jonathan, for instance)
            catpath = os.path.join(self.datadir,'sfrcat_%d_%s'%(self.finsnap,sfr_string))
            if not os.path.isfile(catpath):
                sfrcat = {'metals':SFR_metals,'sfrs':SFRs,'time_edges':SFH_time_edges}
                print("outputting sfrcat to:",catpath,'radial_thresh:',radial_thresh)
                np.savez(catpath,**sfrcat)

            return SFH_time_edges, SFRs, SFR_metals, DT

        return_value = list(compute_SFH(self,**kwargs))

        return_value[-1] = DT
        foo,return_value[1] = all_utils.boxcar_average(
            return_value[0],
            return_value[1],
            DT)

        ## update the attributes of the instance
        self.SFRs = return_value[1]
        self.SFH_dt = return_value[-1]

        return return_value

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
        """ thresh=0.3, ## dex of scatter
            window_size=0.3, ## size of window to compute scatter within"""

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
            thresh=None, ## dex of scatter
            window_size=0.3, ## size of window to compute scatter within
            mode=None,
            thresh_window=1.5, ## size of window must remain below threshold for
            ):
                
            if thresh is None:
                thresh = 0.3#np.log10(2)

            if self.snapnum != self.finsnap:
                from abg_python.galaxy.gal_utils import Galaxy
                ## have to open a whole new galaxy object!!
                temp_fin_gal = Galaxy(
                    self.name,
                    self.snapdir,
                    self.finsnap,
                    datadir=os.path.dirname(self.datadir),
                    datadir_name=self.datadir_name,
                    ahf_path=self.ahf_path,
                    ahf_fname=self.ahf_fname)
                return temp_fin_gal.get_bursty_regime(
                    thresh=thresh,
                    window_size=window_size,
                    numerator_time=numerator_time,loud=False)

            ## ensure that we have the 1 Myr SFH loaded
            try:
                self.get_SFH(
                    DT=0.001,
                    loud=False,
                    use_metadata=True,
                    assert_cached=True)
            except:
                self.get_SFH(
                    DT=0.001,
                    loud=False,
                    save_meta=save_meta,
                    use_metadata=use_metadata,
                    assert_cached=assert_cached)


            adjusted_sfrs = (self.SFRs + self.SFRs[self.SFRs>0].min()/10)
 
            if mode == 'peaktrough':

                rel_scatters = np.zeros(adjusted_sfrs.size)
                per_ls = np.zeros(adjusted_sfrs.size)
                per_rs = np.zeros(adjusted_sfrs.size)
                medians = np.zeros(adjusted_sfrs.size)
                
                this_window_size = 0.05 #window_size
                window_size_n = int(this_window_size/self.SFH_dt/2)

                for i in range(adjusted_sfrs.size):
                    window = adjusted_sfrs[
                        max(0,i-window_size_n):
                        min(adjusted_sfrs.size-1,i+window_size_n)]

                    median = np.median(window)
                    per_l,per_r = np.quantile(
                        window/median,
                        [0.1,0.9])

                    rel_scatters[i] = (per_r - per_l)
                    #rel_scatters[i] = (per_r / per_l)
                    per_ls[i] = per_l
                    per_rs[i] = per_r
                    medians[i] = median
                xs,rel_scatters = all_utils.boxcar_average(
                    self.SFH_time_edges,
                    rel_scatters,
                    0.3,
                    assign='center')

                self.SFH_scatter_per_ls = per_ls
                self.SFH_scatter_per_rs = per_rs
                self.SFH_scatter_medians = medians

            elif mode == 'anna':
                ## calculate scatter using 10 Myr running average in 
                ##  window_size sized window
                xs,adjusted_sfrs = all_utils.boxcar_average(
                    self.SFH_time_edges,
                    adjusted_sfrs,
                    0.01,
                    loud=True)

                xs,boxcar_ys_300 = all_utils.boxcar_average(
                    self.SFH_time_edges,
                    adjusted_sfrs,
                    0.5,#window_size,
                    loud=True,
                    assign='center')

                xs,boxcar_ys2_300 = all_utils.boxcar_average(
                    self.SFH_time_edges,
                    adjusted_sfrs**2,
                    0.5,#window_size,
                    loud=True,
                    assign='center')

                ## <std>/<SFR>
                rel_scatters = np.sqrt(boxcar_ys2_300 - boxcar_ys_300**2)/boxcar_ys_300
            else:
                xs,boxcar_ys_300 = all_utils.boxcar_average(
                    self.SFH_time_edges,
                    np.log10(adjusted_sfrs),
                    window_size,
                    loud=True,
                    assign='center')

                xs,boxcar_ys2_300 = all_utils.boxcar_average(
                    self.SFH_time_edges,
                    np.log10(adjusted_sfrs)**2,
                    window_size,
                    loud=True,
                    assign='center')

                rel_scatters = np.sqrt(boxcar_ys2_300 - boxcar_ys_300**2)

            ## find the first 300 Myr window that is consistently below the threshold
            #print(thresh, thresh_window,rel_scatters)
            l_window, r_window = all_utils.find_first_window(
                self.SFH_time_edges,
                rel_scatters,
                lambda x,y: y < thresh,
                thresh_window,
                last=True)

            #print(mode,l_window,r_window,rel_scatters)
            tindex = np.nan
            bursty_redshift = np.nan
            if np.isfinite(l_window):
                tindex = np.argmin((l_window-self.SFH_time_edges)**2)
                bursty_redshift = approximateRedshiftFromGyr(
                    self.header['HubbleParam'],
                    self.header['Omega0'],
                    np.array([l_window]) )[0]

            #print(mode,tindex,l_window,r_window,bursty_redshift,rel_scatters)
            return tindex, l_window, bursty_redshift, rel_scatters
        
        return compute_bursty_regime(self,**kwargs)

#### HELPER FUNCTIONS
def arch_method(smasses,SFTs,time_edges,DT=None):

    if DT==None:
        DT = .001 #1 Myr in Gyr

    SFRs,time_edges = np.histogram(SFTs,weights=smasses/(DT*1e9),bins=time_edges) #solar masses/year, Myr

    #ignore the last point because it will often contain a "remainder bin", not a full bin width
    return SFRs, time_edges
