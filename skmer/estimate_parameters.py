#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import errno
import sys
import math
import numpy as np
from subprocess import check_output, STDOUT, run, call

from skmer.reskmer import estimate_cov_with_ref
from skmer.dipskmer import estimate_diploid_cov, estimate_theta_from_ref
from skmer.skmer import estimate_base_cov
from skmer.config import *
from skmer.utils import sequence_stat, write_error_file, count_kmers

def estimate_cov(sequence, lib, k, e, nth, skmer_ver, ref_hist = None, theta = None):
    sample = os.path.basename(sequence).rsplit('.f', 1)[0]
    sample_dir = os.path.join(lib, sample)
    try:
        os.makedirs(sample_dir)
    except OSError as Error:
        if Error.errno != errno.EEXIST:
            raise
            
    info_file = os.path.join(sample_dir, sample + '.dat')
    
    # Generates jellyfish histogram
    histo_stderr = count_kmers(sample_dir, sample, sequence, k, nth)
    count = [0]
    ksum = 0
    for item in histo_stderr.split('\n')[:-1]:
        count.append(int(item.split()[1]))
        ksum += int(item.split()[0]) * int(item.split()[1])

    # Calculates read length statistics 
    (l, max_len, tot_len, n_reads) = sequence_stat(sequence)
    #(l, max_len, tot_len, n_reads) = (150, 150, 7000, 70000)
    #(l, max_len, tot_len, n_reads) = (100, 100, 7000, 70000)
    # checks if input sequence is an assembly
    if max_len > seq_len_threshold:
        cov = "NA"
        g_len = tot_len
        eps = 0
        l = "NA"
        theta = "NA" if (skmer_ver == "dipskmer" or skmer_ver == "reskmer + dipskmer") else None
        write_error_file(info_file, cov, g_len, eps, l, theta)
        return (sample, cov, g_len, eps, l, theta)
    
    # check for coverage is too low:
    if len(count) < 3:
        sys.stderr.write('[WARNING] Coverage of {0} is too low, not able to estimate it; no correction applied\n'.format(sample))
        cov = "NA"
        g_len = "NA"
        eps = "NA"
        theta = "NA" if (skmer_ver == "dipskmer" or skmer_ver == "reskmer + dipskmer") else None
        write_error_file(info_file, cov, g_len, eps, l, theta)
        return sample, cov, g_len, eps, l, theta
    
    if skmer_ver == "dipskmer":
        (eps, lam, theta) = estimate_diploid_cov(sample, count, k, e, l, theta)
    elif (skmer_ver == "reskmer") or (skmer_ver == "reskmer + dipskmer"):
        (eps, lam) = estimate_cov_with_ref(sample, ref_hist, ksum, count, k, e, l)
        if (skmer_ver == "reskmer + dipskmer"):
            theta = estimate_theta_from_ref(lam/2.0, eps, k, count) if (theta is None) else theta
    elif skmer_ver == "skmer":
        (eps, lam) = estimate_base_cov(sample, count, k, e, l)
    cov = (1.0 * l / (l - k)) * lam
    tot_seq = 1.0 * ksum * l / (l - k)
    g_len = int(tot_seq / cov)
    
    if eps > error_rate_threshold or eps < 0:
        cov = "NA"
        g_len = "NA"
        eps = "NA"
        theta = "NA" if (skmer_ver == "dipskmer") else None

    write_error_file(info_file, cov, g_len, eps, l, theta)
    return(sample, cov, g_len, eps, l, theta)

# def estimate_skmer_cov(sequence, lib, k, e, nth, ref_hist = None):
#     sample = os.path.basename(sequence).rsplit('.f', 1)[0]
#     sample_dir = os.path.join(lib, sample)

#     try:
#         os.makedirs(sample_dir)
#     except OSError as Error:
#         if Error.errno != errno.EEXIST:
#             raise
            
#     info_file = os.path.join(sample_dir, sample + '.dat')
    
#     # Does not recalculate histogram if histogram already exists
#     histo_stderr = count_kmers(sample_dir, sample, sequence, k, nth)
#     # Calculate read stats: 
#     (l, max_len, tot_len, n_reads) = sequence_stat(sequence)
#     # if sample is assembly
#     if max_len > seq_len_threshold:
#         cov = "NA"
#         g_len = tot_len
#         eps = 0
#         l = "NA"
#         write_error_file(info_file, cov, g_len, eps, l)
#         return sample, cov, g_len, eps, l

#     count = [0]
#     ksum = 0
#     for item in histo_stderr.split('\n')[:-1]:
#         count.append(int(item.split()[1]))
#         ksum += int(item.split()[0]) * int(item.split()[1])
#     # If coverage is too low
#     if len(count) < 3:
#         sys.stderr.write('Coverage of {0} is too low, not able to estimate it; no correction applied\n'.format(sample))
#         cov = "NA"
#         g_len = "NA"
#         eps = "NA"
#         write_error_file(info_file, cov, g_len, eps, l)
#         return sample, cov, g_len, eps, l

#     ind = min(count.index(max(count[2:])), len(count) - 2) + (1 if ref_hist is not None else 0)
#     # If no reskmer reference
#     if (e is not None) and (ref_hist is None):
#         eps = e
#         p0 = np.exp(-k * eps)
#         if ind < 2:
#             r21 = 1.0 * count[2] / count[1]
#             cov = newton(cov_temp_func, 0.05, args=(r21, p0, k, l))
#         else:
#             cov = (1.0 / p0) * (1.0 * l / (l - k)) * (ind + 1) * count[ind + 1] / count[ind]
#     elif ind < 2:
#         sys.stderr.write('Not enough information to co-estimate coverage and error rate of {0}; '.format(sample) +
#                          'Using default error rate {0}\n'.format(default_error_rate))
#         eps = default_error_rate
#         p0 = np.exp(-k * eps)
#         r21 = 1.0 * count[2] / count[1]
#         cov = newton(cov_temp_func, 0.05, args=(r21, p0, k, l))
#     else:
#         if ref_hist is not None:
#             # repeat spectrum-based calculation of error and coverage (reskmer)
#             (eps, lam) = estimate_cov_with_ref(ref_hist, ksum, count, k, sample, l, e)
#         else:
#             gam = 1.0 * (ind + 1) * count[ind + 1] / count[ind]
#             lam = (np.exp(-gam) * (gam ** ind) / math.factorial(ind)) * count[1] / count[ind] + gam * (1 - np.exp(-gam))
#             eps = 1 - (gam / lam) ** (1.0 / k)
#         cov = (1.0 * l / (l - k)) * lam
#     tot_seq = 1.0 * ksum * l / (l - k)
#     g_len = int(tot_seq / cov)

#     if eps > error_rate_threshold or eps < 0:
#         cov = "NA"
#         g_len = "NA"
#         eps = "NA"

#     write_error_file(info_file, cov, g_len, eps, l)
#     return sample, cov, g_len, eps, l
