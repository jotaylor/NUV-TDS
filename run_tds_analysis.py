#! /usr/bin/env python

import argparse

from cos_tds import TDSData, TDSTrends

def run_tds(datapath, plotit, tdstab):
    """
    Args:
        datapath (str): Path to *all* TDS data. 
        plotit (Bool): True if plots should be produced and saved.
        tdstab (Bool): True if new TDSTAB should be created.
    """

    Data = TDSData(datapath)
    Trends = TDSTrends(Data)
    if plotit:
        Trends.plot_trends(g285m_log=True, one_plot=False)
        Trends.make_summary_plot(g285m_log=True)
    if tdstab:
        Trends.create_tdstab(outfile=None,
                             current_file="/grp/hst/cdbs/lref/u7d20378l_tds.fits")

#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", dest="datapath", 
                        help="Path to data to analyze")
    parser.add_argument("--plot", dest="plotit", action="store_true",
                        default=False, 
                        help="Switch to produce analysis plots")
    parser.add_argument("--tdstab", dest="tdstab", action="store_true", 
                        default=False,
                        help="Switch to create a new TDSTAB")
    args = parser.parse_args()

    run_tds(args.datapath, args.plotit, args.tdstab)
