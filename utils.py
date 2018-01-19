#! /usr/bin/env python

def linlsqfit(x, y):
    from astropy.modeling import models, fitting                      
    t_init = models.Linear1D()
    fit_t = fitting.LinearLSQFitter()                         
    lsq_lin_fit = fit_t(t_init, x, y)                                            

    return lsq_lin_fit.slope.value, lsq_lin_fit.intercept.value
