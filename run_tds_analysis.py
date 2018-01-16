#! /usr/bin/env python

import argparse

from cos_tds import TDSData, TDSTrends

def run_tds(datapath):
    Data = TDSData(datapath)
    Trends = TDSTrends(Data)
    TDSTrends.plot_trends(Trends)
    TDSTrends.make_summary_plot(Trends)
    TDSTrends.create_tdstab(Trends)

#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", dest="datapath", 
                        help="Path to data to analyze")
    args = parser.parse_args()

    run_tds(args.datapath, plot)
