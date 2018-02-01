#! /usr/bin/env python

from astropy.io import fits as pf
import glob
from matplotlib import pyplot as pl
import os
import numpy as np
import pdb
import argparse

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

def measure_snr(files, windows, plot_it):
    avg_snr = {}
    for item in files:
        hdulist = pf.open(item)
        data = hdulist[1].data
        hdr0 = hdulist[0].header
        hdulist.close()
    
        cenwave = hdr0["cenwave"]
        opt_elem = hdr0["opt_elem"]
        rootname = hdr0["rootname"]
        
        # SNR is measured at central wavelength in stripe B
        # Select by NUVB, not 1st index
        wl0 = data["wavelength"][1]
        flux0 = data["flux"][1]
        nonzero = np.where(flux0 > 0.0)
        wl = wl0[nonzero][5:-5] # cut off jumps on edges
        flux = flux0[nonzero][5:-5]
        
        # Define wavelength windows manually. 
        if windows:
            wl, flux = define_windows(wl, flux, hdr0) 

        # There is a bump in G230L/NUVB that needs to be removed.
        # make 80 pixels an argument
        if opt_elem == "G230L":
            cutoff = wl[0] + 80.
            bump = np.where(wl > cutoff)
            wl = wl[bump]
            flux = flux[bump] 

        # Smooth over a NUV resel size (3pix)
        wl_reg = downsample_1d(wl, 3)
        flux_reg = downsample_1d(flux, 3)
        
        # Fit a polynomial to the spectrum.    
        # See if centering X around 0 changes results
        poly2 = np.polyfit(wl_reg, flux_reg, 4)
        fit2 = np.poly1d(poly2)
    
        # Y values of polynomial given a wavelength array
        ycalc = fit2(wl_reg)
        # Deviation of the actual flux from the polynomial Y values
        dev = flux_reg - ycalc
        stdev_reg = np.std(dev)
   
        # SNR is measured around the central wavelength. 
        y_ceninds = np.where((wl_reg >= (cenwave-2)) & (wl_reg < (cenwave+2)))
        # polynomial Y values for wavelength around central wavelength
        y_cen = np.average(ycalc[y_ceninds])
        snr = y_cen / stdev_reg
        # Alternative way of defining SNR
        #avg_reg = np.average(ycalc)
        #snr = avg_reg / stdev_reg
        
        if plot_it:
            pl.plot(wl, flux, "b", wl_reg, ycalc, "ro")
            pl.title("{0} {1} {2}".format(rootname, opt_elem, cenwave))
            pl.xlabel("Wavelength (A)")
            pl.ylabel("Flux")
            pl.show()
            print("Close plot window to continue")
            pl.clf()
    
        print("SNR for {0} ({1}/{2}) using WL{{{3:4.0f}:{4:4.0f}}}= {5:4.1f}\n".format(
              rootname, opt_elem, cenwave, wl[0], wl[-1], snr))
        
        if cenwave not in avg_snr.keys():
            avg_snr[cenwave] = [snr]
        else:
            avg_snr[cenwave].append(snr)
    
    for cenwave in avg_snr:
        print("Average SNR for {0}= {1:4.1f}".format(cenwave, np.average(avg_snr[cenwave])))

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

def downsample_1d(myarr,factor):
    """
    Downsample a 1D array by averaging over *factor* pixels.
    Crops right side if the shape is not a multiple of factor.

    Got this specific function from "Adam Ginsburg's python codes" on agpy

    myarr : numpy array

    factor : how much you want to rebin the array by
    """
    xs = myarr.shape[0]
    crarr = myarr[:xs-(xs % int(factor))]
    dsarr = np.mean( np.concatenate(
                     [[crarr[i::factor] for i in range(factor)] ]
                     ),axis=0)

    return dsarr

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

def define_windows(wl, flux, hdr0):    
    pl.plot(wl, flux)
    pl.title("{0} {1} {2}".format(hdr0["rootname"], hdr0["opt_elem"], hdr0["cenwave"]))
    pl.show()
    # This needs work
    print("Enter start of WL region to fit\n")
    beg = input("")
    print("Enter end of WL region to fit\n")
    end = input("")
    beg = int(beg)
    end = int(end)
    pl.clf()
    inds = np.where((wl >= beg) & (wl < end))
    wl_reg = wl[inds]
    flux_reg = flux[inds]

    return wl_reg, flux_reg

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", dest="plot_it", action="store_true", 
                        default=False, help="Switch to plot spectra and fits")
    parser.add_argument("-w", dest="windows", action="store_true",
                        default=False, 
                        help="Switch to determine windows interactively")
    parser.add_argument("-d", dest="data_dir", 
                        default="/smov/cos/Data/14441/otfrdata/??-??-????",
                        help="Path to TDS data")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data_dir, "*x1d.fits*"))

    measure_snr(files, args.windows, args.plot_it)
