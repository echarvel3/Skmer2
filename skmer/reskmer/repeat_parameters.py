#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy.optimize import minimize

from skmer.config import * 

def estim_oh(xi, hr, hcount, ref_hist, maxj):
    '''computes L2 error between estimated kmer histogram against the observed histogram (hcount)...'''
    try:
        errs = [(hcount[h] - np.dot(ref_hist.iloc[0:maxj,1], 
                                    np.array([np.exp(-j*xi) * np.power(j*xi,h) / math.factorial(h) for j in range(1, maxj+1)]))
            )/math.sqrt(hcount[h]) for h in hr]
        err = sum((e ** 2 for e in errs)) /  len(errs)
    except:
        err=default_error_rate
    return(err)

def xi_function(hrange, count, ref_hist, maxj):
    '''returns L2 error-computing function...'''
    fxn = (lambda xi : estim_oh(xi, hrange, count, ref_hist, maxj))
    return(fxn)

def estimate_cov_with_ref(sample, ref_hist_df, ksum, count, k, e, l):
        '''estimates lambda and epsilon using a repeat spectrum...'''
        # Check if repeat spectrum is one sample (reference assembly) or mutliple (respect)
        
        if ref_hist_df.shape[1] > 2:
            try:
                ref_hist = ref_hist_df[[0, sample]]
            except KeyError:
                print("[!WARNING!] {0} not found in estimated spectra file".format(sample))
                return(-1, 1)
        elif ref_hist_df.shape[1] == 2:
            ref_hist = ref_hist_df

        genome_size = np.dot(ref_hist.iloc[:, 0], ref_hist.iloc[:, 1]) 
        lam = float(ksum/genome_size)
        
        # minimizing L2 error between hcount and estimated kmer histogram...
        hl = 2
        erscore = {}
        maxj = min(ref_hist.shape[0], 50)
        for hl in range(2,10*(int(lam)+1)):
                hrange=range(max(int(lam),2),min(max((int(lam)+hl),5),len(count)))
                try:
                    xi = minimize(fun = xi_function(hrange, count, ref_hist, maxj), 
                                  x0 = lam*((1-0.003)**k), 
                                  bounds = [(lam*(1-0.03)**k, lam*((1-0.0001)**k))])
                    xi = xi.x[0]
                    eps = 1 - (xi / lam) ** (1.0 / k)
                    erscore[estim_oh(xi, hrange, count, ref_hist, maxj)] = (eps, hl)  
                except ValueError:
                    eps = -1
        
        if erscore.keys():
            eps = erscore[min(erscore.keys())][0]
        else:
            eps = -1

        eps = eps if e is None else e
        return (eps, lam)
