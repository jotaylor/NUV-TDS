#! /usr/bin/env python

import astropy
from astropy.io import fits as pf
from astropy.time import Time
import os
import glob
import numpy as np
import datetime
from scipy import stats
import warnings

from wavelength_ranges import *
from utils import linlsqfit

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

class TDSData():
    """
    Measure the TDS of COS data. 

    Attributes:
        dates_mjd (array-like): Observation date (MJD) of each dataset.
            Astropy Time object.
        dates (array-like): Observation date (datetime.datetime) of each 
            dataset. Datetime object.
        dates (array-like): Observation date (decimal year) of each dataset.
            Astropy Time object.
        infiles (array-like): Full path and filename of all input x1d files.
#        outdir (str): Output directory for.
        nfiles (int): Number of input x1d files.
        rootnames (array-like): Rootnames of all input x1d files.
        nets (array-like): NET array for each x1d file.
        wls (array-like): WAVELENGTH array for each x1d file.
        gratings (array-like): OPT_ELEM keyword for each x1d file.
        segments (array-like): SEGMENT keyword for each x1d file.
        ref (dict): Nested dictionary of NET, WAVELENGTH, and filename for 
            each reference (first in time) dataset's cenwave.
        ratios (array-like): For each x1d file, the ratio of its NET array 
            compared to the reference dataset.
        nets_intp (array-like): NET array of each x1d files interpolated onto the
            wavelength scale of the reference dataset.
        bins (array-like): Edges of the wavelength bins for each x1d file.
        nbins (array-like): Number of wavelength bins for each x1d file.
        means (array-like): The mean of the NET ratio in each wavelength bin 
            for each x1d file.
        stds (array-like): Standard deviation of the NET ratio in each 
            wavelength for each x1d file.
        means_net (array-like): The mean of the NET in each wavelength bin for
            each x1d file.
        stds_nets (array-like): Standard deviation of the NET in each wavelength
            bin for each x1d file.
        detector (str): NUV or FUV.
        nsegs (str): Number of segments, 3 for NUV or 2 for FUV.
    """

    def __init__(self, infiles, outdir=".", binsize=1000., pickle=True, plot=True):
        """
        Args: 
            infiles: List or wild-card or dir name of input datasets.
            outdir (str): Output directory for pickle file.
            binsize (int or float): Size of each wavelength bin.
            pickle (Bool): Switch to pickle resulting class instance. 
            plot (Bool): Switch to make plots.
        """

        infiles = self.parse_infiles(infiles)
        mjds = [Time(pf.getval(x, "expstart", 1), format="mjd") for x in infiles]
        order = np.argsort(mjds)
        self.infiles = infiles[order]
        self.dates_mjd = np.array(mjds)[order]
        self.dates = np.array([x.datetime for x in self.dates_mjd])
        self.dates_dec = np.array([x.decimalyear for x in self.dates_mjd])
        self.outdir = outdir
        self.nfiles = len(self.infiles)
        self.rootnames = np.array([pf.getval(x, "rootname", 0) for x in self.infiles])
        self.get_hduinfo()
        self.get_refdata()
        self.ratios, self.nets_intp = self.calc_ratios()
        self.bin_data(binsize)
        
        if pickle:
            self.pickle_tds()


#-----------------------------------------------------------------------------#

    def pickle_tds(self):
        """
        Pickle the TDSData class object.
        """

        import pickle
        import datetime

        now = datetime.datetime.now()
        pname = os.path.join(self.outdir, "costds_{0}.p".format(now.strftime("%Y%m%d_%M%S")))
        pickle.dump(self, open(pname, "wb"))
        print("Wrote pickle file {0}".format(pname))

#-----------------------------------------------------------------------------#

    def apply_wl_windows(self):
        """
        Truncate the wavelength arrays based on the defined ranges in
        wavelength ranges.
        
        Not currently used. 
        """

        wl_trunc = []
        for i in range(self.nfiles):
            wl_windows = []
            for j in range(self.nsegs):
                wl_range = WL_DICT[self.cenwaves[i]][self.segments[i][j]]
                wl_windows.append(np.where((self.wls[i][j] >= wl_range[0]) & (self.wls[i][j] <= wl_range[1]))[0])
            wl_trunc.append(np.array(wl_windows))
        
        self.wl_trunc = wl_trunc

#-----------------------------------------------------------------------------#

    def bin_data(self, binsize):
        """
        Bin the ratio and net data and determine the mean and standard deviation
        of each bin.

        Args:
            binsize (int or float): Size of each wavelength bin. 
        """

        means = []
        stds = []
        bins = []
        nbins = []
        means_net = []
        stds_net = []

        for i in range(self.nfiles):
            # These need to be lists because there can be multiple entries
            # depending on bin size.
            mean_segs = []
            stds_segs = []
            bins_segs = []
            nbins_segs = []
            meann_segs = []
            stdn_segs = []


            for j in range(self.nsegs):
                # Wavelength range defined by pre-defined values.
                wl_range = WL_DICT[self.cenwaves[i]][self.segments[i][j]]
                min_wl = wl_range[0]
                max_wl = wl_range[1]
                
                # If the bin size is bigger than the wavelength range,
                # make the bin the entire WL range. 
                if binsize >= (max_wl - min_wl):
                    bins_j = np.array([min_wl, max_wl])
                else:
                    bins_j = np.arange(min_wl, max_wl, binsize)
                nbins_j = len(bins_j) - 1

                # Determine the mean and STD for each bin for both the 
                # NET ratio and NET.
                mean = stats.binned_statistic(self.ref[self.cenwaves[i]]["wl"][j],
                                              self.ratios[i][j], 
                                              "mean", bins=bins_j)[0]
                std = stats.binned_statistic(self.ref[self.cenwaves[i]]["wl"][j],
                                             self.ratios[i][j], 
                                             np.std, bins=bins_j)[0]
                mean_n = stats.binned_statistic(self.ref[self.cenwaves[i]]["wl"][j],
                                              self.nets_intp[i][j], 
                                              "mean", bins=bins_j)[0]
                std_n = stats.binned_statistic(self.ref[self.cenwaves[i]]["wl"][j],
                                             self.nets_intp[i][j], 
                                             np.std, bins=bins_j)[0]

                mean_segs.append(mean)
                stds_segs.append(std) 
                bins_segs.append(bins_j)
                nbins_segs.append(nbins_j)
                meann_segs.append(mean_n)
                stdn_segs.append(std_n)

            means.append(np.array(mean_segs))
            stds.append(np.array(stds_segs))
            bins.append(np.array(bins_segs))
            nbins.append(np.array(nbins_segs))
            means_net.append(np.array(meann_segs))
            stds_net.append(np.array(stdn_segs))

        self.bins = np.array(bins)
        self.nbins = np.array(nbins)
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.means_net = np.array(means_net)
        self.stds_net = np.array(stds_net)

#-----------------------------------------------------------------------------#

    def get_refdata(self):
        """
        Determine the reference dataset (first in time) of each cenwave and
        store its NET, WAVELENGTH, and filename information in a dictionary.
        """

        ref_dict = {}
        for i in range(self.nfiles):
            if self.cenwaves[i] not in ref_dict.keys():
                ref_dict[self.cenwaves[i]] = {}
                ref_dict[self.cenwaves[i]]["net"] = self.nets[i]
                ref_dict[self.cenwaves[i]]["wl"] = self.wls[i]
                ref_dict[self.cenwaves[i]]["filename"] = self.infiles[i]

        self.ref = ref_dict

#-----------------------------------------------------------------------------#

# need to update for FUV.
    def get_hduinfo(self):
        """
        Get necessary information from the input files' HDU headers and 
        data extensions.
        """
    
        def remove_files(inds):
            self.infiles = np.delete(self.infiles, inds)
            self.dates = np.delete(self.dates, inds)
            self.dates_mjd = np.delete(self.dates_mjd, inds)
            self.dates_dec = np.delete(self.dates_dec, inds)
            self.rootnames = np.delete(self.rootnames, inds)
            self.nfiles = len(self.infiles)
            
        nets = []
        wls = []
        cenwaves = []
        gratings = []
        segments = []
        bad_inds = []
        detectors = []

        for i in range(self.nfiles):
            with pf.open(self.infiles[i]) as hdulist:
                data = hdulist[1].data
                hdr0 = hdulist[0].header
                detector = hdr0["detector"]
                if "NUV" in detector:
                    nsegs = 3
                else:
                    nsegs = 2

                detectors.append(detector)
        
                # Should move this to a separate script !!!!!
                # These observations failed and should be excluded.
                if hdr0["proposid"] == 13125 and (hdr0["obset_id"] == "M2" or hdr0["obset_id"] == "L2"):
                    bad_inds.append(i)
                    continue
                if hdr0["proposid"] == 11896 and hdr0["obset_id"] == "AC":
                    bad_inds.append(i)
                    continue

                # Some early observations used different targets, exclude these.
                if hdr0["opt_elem"] == "G230L":
                    if hdr0["targname"] != "WD1057+719":
                        bad_inds.append(i)
                        continue
                else:
                    if hdr0["targname"] != "G191B2B":
                        bad_inds.append(i)
                        continue

                nets.append(data["net"])
                segments.append(data["segment"])
                wls.append(data["wavelength"])
                gratings.append(hdr0["opt_elem"])
                cenwaves.append(hdr0["cenwave"]) 

        # Remove rows for all the bad files.
        remove_files(bad_inds)

        if len(set(detectors)) != 1:
            print("ERROR: data are not all same detector type, {0}".format(
                  set(detectors)))
            sys.exit()
        else:
            self.detector = detectors[0]

        self.nets = np.array(nets)
        self.wls = np.array(wls)
        self.gratings = np.array(gratings)
        self.cenwaves = np.array(cenwaves)
        self.segments = np.array(segments)
        self.nsegs = nsegs

#-----------------------------------------------------------------------------#

    def calc_ratios(self):
        """
        Calculate the ratio of each input dataset's NET array compared to its
        reference dataset (first dataset of the same cenwave).

        Returns:
            ratios (array-like): For each x1d file, the ratio of its NET array 
                compared to the reference dataset.
            nets_intp (array-like): NET array of each x1d files interpolated 
                onto the wavelength scale of the reference dataset.
        """

        ratios = []
        net_interp = []
        
        # Interpolate each dataset's NET values to the wavelength scale of its 
        # reference dataset (first in time).
        for i in range(self.nfiles):
            net_intp_allseg = []
            ratios_allseg = []

            for j in range(self.nsegs):
                net_intp_j = np.interp(self.ref[self.cenwaves[i]]["wl"][j], 
                                       self.wls[i][j], self.nets[i][j])
                net_intp_allseg.append(net_intp_j)
                net_intp_j.astype(np.float64)
                ref_net = self.ref[self.cenwaves[i]]["net"][j]
                ref_net.astype(np.float64)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                ratios_j = net_intp_j / ref_net
                warnings.resetwarnings()
                ratios_j = np.nan_to_num(ratios_j)
                ratios_allseg.append(ratios_j)

            net_intp_allseg_arr = np.array(net_intp_allseg)
            ratios_allseg_arr = np.array(ratios_allseg)
            net_interp.append(net_intp_allseg_arr)
            ratios.append(ratios_allseg_arr)

        return np.array(ratios), np.array(net_interp)

#-----------------------------------------------------------------------------#

    def parse_infiles(self, infiles):
        """
        Determine the list of all input files based on method of input.

        Args:
            infiles: List or wild-card string of input datasets.
        Returns:
            x1dfiles (array-like): Input x1d files.
        """

        if isinstance(infiles, list):
            x1dfiles = infiles
        
        elif isinstance(infiles, str):
            if os.path.isdir(infiles):
                x1dfiles = glob.glob(os.path.join(infiles, "*x1d.fits*"))
            
            elif "*" in infiles:
                x1dfiles = glob.glob(infiles)
                if not rawfiles:
                    print("ERROR No matching datasets: {0}".format(infiles))
                    sys.exit()
            else:
                print("ERROR Could not parse input files: {0}".format(infiles))
        
        else:
            print("ERROR Input files were not list or str: {0}".format(infiles))
            sys.exit()

        return np.array(x1dfiles)

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

#class TDSTrends(object):
#    """
#    Determine the linear fit to the TDS trends.
#    
#    Attributes:
#        
#    """
#    def __init__(self, gratings, cenwaves, dates, ratios, make_plots=True):
#        """
#        gratings (list or array): Gratings for which TDS was analyzed.
#        Cenwaves (list or array): Cenwaves for each grating.
#        Dates (list or array): Dates for each TDS measurement.
#        Ratios (list or array): Net ratio for each TDS measurement.
#            If there are multiple segments, this should be N_segments X length.
#        make_pltos (Bool): True if plots should be produced.
#        """ 
#        
#        this = "wut do"
#        ## Check type of dates first and convert to Time.decimalyear
#        #date = dates.flatten()[0]
#        #if type(date) is float:
#        #    x = [Time(i, format="decimalyear") for i in dates]
#        #elif type(date) is datetime.datetime:
#        #    x = [Time(i, format="datetime").decimalyear for i in dates]
#        #elif type(date) is astropy.time.core.Time:
#        #    x = [i.decimalyear for i in dates]
#        #else:
#        #    d_type = type(date)
#        #    print("Format of date array not recognized: {0}".format(d_type))
#        #    sys.exit()
#
#        #self.dates = x
#        #self.ratios = ratios
#        #self.gratings = gratings
#        #self.cenwaves = cenwaves
#
#        #self.trends = self.fit_trends()
#        #if make_plots:
#        #    self.plot()

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

class TDSTrends(object):
    def __init__(self, TDS):
        """
        Create a TDSTrends class instance  by using information from a
        TDSData instance.
        """ 
        
        self.detector = TDS.detector
        trends = {}
        for cenwave in set(TDS.cenwaves):
            grating = TDS.gratings[np.where(TDS.cenwaves == cenwave)[0][0]]
            if grating not in trends.keys():
                trends[grating] = {}
            trends[grating][cenwave] = {}

            inds = np.where(TDS.cenwaves == cenwave)[0]
            cenwave_x = TDS.dates_dec[inds]
            cenwave_y = TDS.means[inds]
            nbins = TDS.nbins[inds][0]
            bins = TDS.bins[inds][0]

            for i in range(TDS.nsegs):
                seg = TDS.segments[inds][0][i]
                trends[grating][cenwave][seg] = {}
                for j in range(0, nbins[i]):
                    cenwave_bin_y = cenwave_y[:, i, j]
                    bin_tup = (bins[i][j], bins[i][j+1])
                    trends[grating][cenwave][seg][bin_tup] = {}
                    trends[grating][cenwave][seg][bin_tup]["y"] = cenwave_bin_y
                    trends[grating][cenwave][seg][bin_tup]["x"] = cenwave_x
                    m,b = linlsqfit(cenwave_x, cenwave_bin_y)
                    trends[grating][cenwave][seg][bin_tup]["fit"] = (m,b)

        self.trends = trends

#-----------------------------------------------------------------------------#

#    def fit_trends(self):
#        trends = {}
#        for cenwave in set(self.cenwaves):
#            grating = self.gratings[np.where(self.cenwaves == cenwave)[0][0]]
#            if grating not in trends.keys():
#                trends[grating] = {}
#            trends[grating][cenwave] = {}
#            trends[grating][cenwave]["y"] = []
#            trends[grating][cenwave]["fit"] = np.zeros((1, self.nsegs,  
#
#            inds = np.where(self.cenwaves == cenwave)[0]
#            cenwave_x = self.dates_dec[inds]
#            cenwave_y = self.means[inds]
#            nbins = self.nbins[inds][0]
#            trends[grating][cenwave]["x"] = cenwave_x
#
#            
#            for i in range(self.nsegs):
#                slopes = []
#                ints = []
#                for j in range(1, nbins[i]+1):
#                    cenwave_bin_y = cenwave_y[:, i, j]
#                    m,b = linlsqfit(cenwave_x, cenwave_bin_y[:,i])
#                    slopes.append(m)
#                    ints.append(b)
#                    
#                    trends[grating][cenwave]["fit"].append((m,b))  
#                    trends[grating][cenwave]["y"].append(cenwave_y[i]) 
#                
#        return trends

#-----------------------------------------------------------------------------#
        
    def plot_trends(self, g285m_log=True):
        """
        Args:
            g285m_log (Bool): Switch to plot G285M trends with a log Y-axis. 
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        plt.style.use("niceplot")
        from matplotlib import gridspec
        
        if self.detector == "FUV":
            segs = ["FUVA", "FUVB"]
        elif self.detector == "NUV":
            segs = ["NUVA", "NUVB", "NUVC"]

        now = datetime.datetime.now()
        now_ymd = now.strftime("%Y%m%d")

        # Loop over each grating
        for grating in self.trends:
            # Loop over each cenwave in the grating.
            for cenwave in self.trends[grating]:
                fig = plt.figure(figsize=(8,11))
                nsegs = len(self.trends[grating][cenwave])
                gs = gridspec.GridSpec(nsegs, 1)
                # Loop over each segment in the cenwave.
                for i,seg in enumerate(self.trends[grating][cenwave]):
                    # Loop over each wavelength bin.
                    for wlbin in self.trends[grating][cenwave][seg]:
                        current_trends = self.trends[grating][cenwave][seg][wlbin]
                        
                        ax = plt.subplot(gs[i])
                        
                        if g285m_log and grating == "G285M":
                            ax.set_yscale("log")
                            ax.set_yticks([1, .8, .6, .5, .4, .3, .2, .1, .01, .001])
#                            ax.set_ylim(.003, 1.1)
                            ax.set_ylabel("Log(Relative Net)")
                        else:
                            ax.set_ylabel("Relative Net")
                        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    
                        m,b = current_trends["fit"]
                        x = current_trends["x"]
                        y = current_trends["y"]
                        
                        fit_x = np.linspace(2009., x[-1]+0.5, 200)
                        fit_y = m * fit_x + b 
    
                        # Plot linear fit and data points.
                        ax.plot(fit_x, fit_y, color="forestgreen", 
                                label="{0:.2f} %/yr".format(m*100.))
                        ax.plot(x, y, marker="*", linestyle="None", color="black",
                                markersize=8)
    
                        ax.legend(loc="best")
                        
                        ax.set_title("{0}/{1} {2}, {3}-{4}$\AA$".format(grating, cenwave, seg, wlbin[0], wlbin[1]))
                        ax.set_xlim(2009.5, x[-1] + 0.5)
                        
                        if i == (nsegs - 1):
                            ax.set_xlabel("Date")
                
                figname = "{0}_{1}_trends_{2}.png".format(grating, cenwave, now_ymd)
                fig.savefig(figname)
                fig.clear()
                print("Saved {0}".format(figname))

#-----------------------------------------------------------------------------#

    def make_summary_plot(self, g285m_log=True):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        plt.style.use("niceplot")
        from matplotlib import gridspec
        colors = ["gray", "crimson", "darkorange", "gold", "yellowgreen", 
                  "forestgreen", "darkturquoise", "royalblue", "mediumslateblue", 
                  "darkmagenta", "mediumvioletred", "pink"]
                   
        fig = plt.figure(figsize=(17,12))
        if self.detector == "NUV":
            n_gratings = 4
        else:
            n_gratings = 3
        # Gives 3 subplots for FUV, 4 for NUV.
        gs = gridspec.GridSpec(2, 2)
      
        import pdb
        pdb.set_trace() 
        for i,grating in enumerate(self.trends):
            c = 0
            ax = plt.subplot(gs[i])
            ax.set_title("{0}".format(grating))
            for cenwave in self.trends[grating]:
                for seg in self.trends[grating][cenwave]:
                    if len(self.trends[grating][cenwave][seg]) > 1:
                        # do something
                        # slope vs. wavelength, avg wavl. bins where overlap, smooth avg then plot that
                        this = 0.
                    else:
                        # For loop to get the key and value, but there is only one index
                        for wbin,current_trends in self.trends[grating][cenwave][seg].items():
                            x = current_trends["x"]
                            y = current_trends["y"]
                            m,b = current_trends["fit"]
                        ax.plot(x, y, marker="*", linestyle="None", color=colors[c],
                                label="{0}/{1}".format(cenwave, seg))
                        c += 1
            ax.legend(loc="best")
            ax.set_ylabel("Relative Net Counts")
            ax.set_xlabel("Date")
        figname = "allnuvtds.png"
        fig.savefig(figname)
        print("Saved {0}".format(figname))            

#-----------------------------------------------------------------------------#

    def create_tdstab(self, outfile=None, 
                      current_file="/grp/hst/cdbs/lref/u7d20378l_tds.fits"):
        """
        Use TDSTrends.trends to create a new TDSTAB based on the most recent
        TDS analysis.

        The TDSTAB is the stupidest reference file in existence. There is no
        cenwave dependency, only wavelength. For the NUV, we don't even do that
        right. If there are 2 or more cenwaves observed for a given grating, the
        trends are averaged and applied to all wavelengths for that grating.
        Beyond that, for each grating there is an array for each time chunk
        (breakpoint). Within this time array, the slope and intercept can be
        defined for each wavelength bin (same bins for all gratings).
        For the NUV we only modify one time array (2nd of 3) and the fits 
        are the same for the wavelengths that correspond to the grating in
        question. 
        """

        import shutil
        from datetime import datetime

        reftime_dec = 2003.772602739726

        now = datetime.now()
        current_dayb = now.strftime("%b %d %Y")
        current_dayb2 = now.strftime("%b%d%Y")
        current_daym = now.strftime("%d/%m/%Y")
        if not outfile:
            outfile = "{0}_tds.fits".format(current_dayb2)

        shutil.copy(current_file, outfile)
        with pf.open(outfile, mode="update") as hdulist:
            hdr0 = hdulist[0].header
            data = hdulist[1].data

            if self.detector == "NUV":
                segs = ["NUVA", "NUVB", "NUVC"]
            else:
                segs = ["FUVA", "FUVB"]
            
            for seg in segs:
                for grating in self.trends:
                    slopes = []
                    ints = []
                    for cenwave in self.trends[grating]:
                        for wbin, current_trends in self.trends[grating][cenwave][seg].items():
                            m,b = current_trends["fit"]
                            # You must use the reftime from the 1st header.
                            ints.append(m * reftime_dec + b)
                            slopes.append(m)

                    avg_m = np.average(slopes)
                    avg_b = np.average(ints)
                    indx = np.where((data["opt_elem"] == grating) 
                                    & (data["segment"] == seg)
                                    & (data["aperture"] == "PSA"))[0][0]
                    # !!! This is only for NUV !!!
                    # Accessing the 1st index gives you the 1st time index
                    # This is the only one we modify for NUV. 
                    # Looping through gives us the value for each wavelength
                    # bin. You must check if the value is non-1. because 1.0
                    # values designate wavelength out of range... yeah.
                    for i in range(len(data[indx]["intercept"][1])):
                        if data[indx]["intercept"][1][i] != 1.:
                            data[indx]["intercept"][1][i] = avg_b
                        if data[indx]["slope"][1][i] != 0.:
                            data[indx]["slope"][1][i] = avg_m
            
            descrip = "Updated with TDS trends based on data as of {0}".format(current_dayb)
            descrip67 = descrip.ljust(67, "-")
            
            hdr0["DESCRIP"] = descrip 
            hdr0["PEDIGREE"] = "INFLIGHT 01/09/2009 {0}".format(current_daym)
            hdr0.remove("COMMENT")
            hdr0.add_comment(" = Reference file created by Jo Taylor", before="HISTORY")
            hdr0.add_history("")
            hdr0.add_history("Updated by Jo Taylor on {0} from {1}".format(current_dayb, current_file))
            hdr0.add_history("and renamed to {0}".format(outfile))
            hdr0.add_history("Modified with cos_tds.py.")
            hdr0.add_history("Slopes and intercepts have been updated with data from")
            hdr0.add_history("most recent TDS analysis as of {0}.".format(current_dayb))
            hdr0.add_history("Created for the delivery of new COS NUV Synphot files.")
            hdr0.add_history("Note that only PSA entries have been updated for this purpose.")

        print("Wrote new TDSTAB {0}".format(outfile))
