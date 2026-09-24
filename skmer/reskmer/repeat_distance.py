import os
import pandas as pd
import numpy as np
import math
from scipy.optimize import brenth
from subprocess import check_output, STDOUT

from skmer.utils import get_hist_data

def estimate_intersection(ref_hist, lam1, lam2, eps1, eps2, eta1, eta2, d, k):
    '''calculates expected size of intersection (exp|AuB|) given sequencing parameters... '''
    lam1 = 1.0 if lam1 == "NA" else lam1
    lam2 = 1.0 if lam2 == "NA" else lam2

    nonerr_term1 = 1 - np.power(1-eta1, ref_hist.iloc[:,0])
    nonerr_term2  = 1 - np.power((1-eta2*((1-d)**k)), ref_hist.iloc[:,0])
    nonerr_ins = np.dot(ref_hist.iloc[:,1], nonerr_term1*nonerr_term2)

    b = k*(1-math.exp(-1/(3*k)))

    if eps1:
        n1 = 1-np.exp(-1*ref_hist.iloc[:,0]*b*lam1*eps1*np.power(1-eps1,k-1))
    else:
        n1 = 0

    if eps2:
        n21 = np.power(1-d,k)*np.exp(-1*b*lam2*eps2*np.power(1-eps2, k-1))
        n22 = d*np.power(1-d,k-1)*b*(np.exp(-1*lam2*np.power(1-eps2, k))-1)
        n23 = 1 - np.power(1-d,k)
    else:
        n21 = 0
        n22 = 0
        n23 = 0

    term1 = n1
    term2 = 1-np.power(n21 + n22 + n23, ref_hist.iloc[:,0])
    extra_ins = 3*k*np.dot(ref_hist.iloc[:,1], term1*term2)

    return np.dot([1, 1], [nonerr_ins, extra_ins])

def intersection_fnctn(ref_hist, msh_int, lam1, lam2, eps_1, eps_2, k):
    '''returns FUNCTION of est exp|AuB| - obs|AuB|'''
    eta1 = 1.0 - np.exp(-lam1 * ((1-eps_1)**k)) if (lam1 != "NA" and eps_1) else 1.0
    eta2 = 1.0 - np.exp(-lam2 * ((1-eps_2)**k)) if (lam2 != "NA" and eps_2) else 1.0
    
    # if obs|AuB| is smaller than exp|AuB| at d = 0
    zde = estimate_intersection(ref_hist, lam1, lam2, eps_1, eps_2, eta1, eta2, 0.0, k)
    if (((zde - msh_int) / zde) < 0.01):
        msh_int = zde
        
    def g(est_d):
       return estimate_intersection(ref_hist, lam1, lam2, eps_1, eps_2, eta1, eta2, est_d, k) - msh_int

    return g 

def estimate_reskmer_dist(sample_1, sample_2, lib_1, lib_2, ce, le, ee, rl, k, cov_thres, tran, ref_hist_df):
    '''estimates repeat-aware genomic distance between samples...'''
    if ref_hist_df.shape[1] > 2:
        ref_hist = ref_hist_df[[0, sample_1]]
    elif ref_hist_df.shape[1] == 2:
        ref_hist = ref_hist_df

    try:
        if sample_1 == sample_2 and lib_1 == lib_2:
            return sample_1, sample_2, 0.0
        
        sample_dir_1 = os.path.join(lib_1, sample_1)
        sample_dir_2 = os.path.join(lib_2, sample_2)

        # error rates
        eps_1 = ee[sample_1] if ee[sample_1] != "NA" else None
        eps_2 = ee[sample_2] if ee[sample_2] != "NA" else None

        # get size (sum of kmers) and usize (count of unique kmers)
        _ , size_1, usize_1 = get_hist_data(lib_1, sample_1)
        _ , size_2, usize_2 = get_hist_data(lib_2, sample_2)

        # get i (obs|AuB|) from Mash
        msh_1 = os.path.join(sample_dir_1, sample_1 + ".msh")
        msh_2 = os.path.join(sample_dir_2, sample_2 + ".msh")
        dist_stderr = check_output(["mash", "dist", msh_1, msh_2], stderr=STDOUT, universal_newlines=True)
        j = float(dist_stderr.split()[4].split("/")[0]) / float(dist_stderr.split()[4].split("/")[1])
        i = j * (usize_1 + usize_2) / (1.0 + j)

        # genome size (from reference)
        genome_size = np.dot(ref_hist.iloc[:, 0], ref_hist.iloc[:, 1]) 

        # kmer coverage coverage (lambda)
        cov_1 = float(size_1/genome_size) if ce[sample_1] != "NA" else "NA"
        cov_2 = float(size_2/genome_size) if ce[sample_2] != "NA" else "NA"

        d = brenth(f=intersection_fnctn(ref_hist, i, cov_1, cov_2, eps_1, eps_2, k), a=0, b=1)

        # jukes-cantor transform
        if tran:
            if d < 0.75:
                d = max(0, -0.75 * np.log(1 - 4.0 * d / 3.0))
            else:
                d = 5.0
        return sample_1, sample_2, round(d, 5)
    
    except Exception as e:
        print(e)
        return sample_1, sample_2, None
