#! /usr/bin/env python

from astropy.io import fits as pf
from astropy.time import Time
import os
import glob
import numpy as np
import matplotlib.pyplot as pl
import datetime
from scipy import stats
import warnings

from wavelength_ranges import *

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
        nets (list): NET array for each x1d file.
        wls (list): WAVELENGTH array for each x1d file.
        opt_elems (list): OPT_ELEM keyword for each x1d file.
        segments (list): SEGMENT keyword for each x1d file.
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
    """

    def __init__(self, infiles, outdir, binsize=1000., pickle=True):
        """
        Args: 
            infiles: List or wild-card string of input datasets.
            outdir (str): Output directory for pickle file.
            binsize (int or float): Size of each wavelength bin.
            pickle (Bool): Switch to pickle class or not. 
        """

        mjds = [Time(pf.getval(x, "expstart", 1), format="mjd") for x in infiles]
        order = np.argsort(mjds)
        self.dates_mjd = np.array(mjds)[order]
        self.dates = np.array([x.datetime for x in self.dates_mjd])
        self.dates_dec = np.array([x.decimalyear for x in self.dates_mjd])
        self.infiles = self.parse_infiles(infiles)[order]
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
        pname = "costds_{0}.p".format(now.strftime("%Y%m%d_%M%S"))
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
            for j in range(len(self.segments[i])):
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


            for j in range(len(self.segments[i])):
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

        self.bins = bins
        self.nbins = nbins
        self.means = means
        self.stds = stds
        self.means_net = means_net
        self.stds_net = stds_net

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

# put in a check to ensure all data same detector
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
            self.rootnames = np.delete(self.rootnames, inds)
            self.nfiles = len(self.infiles)

        nets = []
        wls = []
        cenwaves = []
        gratings = []
        segments = []
        bad_inds = []
        
        for i in range(self.nfiles):
            with pf.open(self.infiles[i]) as hdulist:
                data = hdulist[1].data
                hdr0 = hdulist[0].header
                
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

        self.nets = nets
        self.wls = wls
        self.opt_elems = gratings
        self.cenwaves = cenwaves
        self.segments = segments

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

            for j in range(len(self.segments[i])):
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

        return ratios, net_interp

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
            if "*" in infiles:
                x1dfiles = glob.glob(infiles)
                if not rawfiles:
                    print("ERROR: No matching datasets for {0}".format(infiles))
                    sys.exit()
            else:
                x1dfiles = infiles

        return np.array(x1dfiles)

#-----------------------------------------------------------------------------#

    def do_linlsqfit(x, y):
        from astropy.modeling import models, fitting
        t_init = models.Linear1D()
        fit_t = fitting.LinearLSQFitter()
        lsq_lin_fit = fit_t(t_init, x, y)                                                             
