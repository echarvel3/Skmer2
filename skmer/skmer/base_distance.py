import os
import numpy as np
import math
from subprocess import check_output, STDOUT
import math

def estimate_skmer_dist(sample_1, sample_2, lib_1, lib_2, ce, le, ee, rl, k, cov_thres, tran):
    if sample_1 == sample_2 and lib_1 == lib_2:
        return sample_1, sample_2, 0.0
    sample_dir_1 = os.path.join(lib_1, sample_1)
    sample_dir_2 = os.path.join(lib_2, sample_2)
    msh_1 = os.path.join(sample_dir_1, sample_1 + ".msh")
    msh_2 = os.path.join(sample_dir_2, sample_2 + ".msh")
    dist_stderr = check_output(["mash", "dist", msh_1, msh_2], stderr=STDOUT, universal_newlines=True)
    mash_dist = float(dist_stderr.split()[2]) 
  #  return(sample_1, sample_2, mash_dist)
    j = float(dist_stderr.split()[4].split("/")[0]) / float(dist_stderr.split()[4].split("/")[1])
    gl_1 = le[sample_1]
    gl_2 = le[sample_2]
    if gl_1 == "NA" or gl_2 == "NA":
        gl_1 = 1
        gl_2 = 1
    cov_1 = ce[sample_1]
    cov_2 = ce[sample_2]
    eps_1 = ee[sample_1]
    eps_2 = ee[sample_2]
    l_1 = rl[sample_1]
    l_2 = rl[sample_2]
    r_1 = dist_temp_func(cov_1, eps_1, k, l_1, cov_thres)
    r_2 = dist_temp_func(cov_2, eps_2, k, l_2, cov_thres)
    wp = r_1[0] * r_2[0] * (gl_1 + gl_2) * 0.5
    zp = sum(r_1) * gl_1 + sum(r_2) * gl_2
    d = round(max(0, float(1 - (1.0 * zp * j / (wp * (1 + j))) ** (1.0 / k))), 5)
    if tran:
        if d < 0.75:
            d = max(0, -0.75 * np.log(1 - 4.0 * d / 3.0))
        else:
            d = 5.0
    return sample_1, sample_2, d

def dist_temp_func(cov, eps, k, l, cov_thres):
    if cov == "NA":
        return [1.0, 0]
    p = np.exp(-k * eps)
    copy_thres = int(1.0 * cov / cov_thres) + 1
    lam = 1.0 * cov * (l - k) / l
    if copy_thres == 1 or p == 1:
        return [1 - np.exp(-lam * p), lam * (1 - p)]
    else:
        s = [(lam * p) ** i / math.factorial(i) for i in range(copy_thres)]
        return [1 - np.exp(-lam * p) * sum(s), 0]
