import os
import errno
import math
import sys
import numpy as np
from scipy.optimize import minimize

from subprocess import run

from skmer.utils import write_error_file, count_kmers, sequence_stat, cov_temp_func
from skmer.config import *

def estimate_theta_from_ref(lam, eps, k, count):
    try:
        ind = min(count.index(max(count[2:])), len(count) - 2)
        r =  count[ind + 1] / count[ind]    
        xi = lam * (1-eps)**k
        theta = 1 - 2**(1/ k) * ((np.exp(xi) * (-r - ind* r + xi))/( 2**ind *r - 2 * np.exp(xi) * r + 2**ind * ind * r - 2* np.exp(xi) * ind* r - 2**(1 + ind) * xi + 2 * np.exp(xi)* xi))**(1/k)
        #print(ind, r, xi, theta)
    except:
        theta = default_theta
    return(theta)

def get_theta_arr(i, count):
    r =  count[i + 1] / count[i]
    rn =  count[i + 2] / count[i+1]
    if (8 * (2 + i) * rn > 9 * (1 + i) * r) or (theta_arg is not None):
        theta = default_theta if (theta_arg is None) else theta_arg
        Q = (1-theta)**k
        xi = minimize(lambda x: (r+x*(-2* np.exp(x)+2* (-2**i+np.exp(x)) * Q)/((1+i)* (2* np.exp(x)+(2**i-2* np.exp(x))*Q )))**2, 0.5, bounds = [[0,100]]).x[0]


def estimate_diploid_cov(sample, count, k, e, l, theta_arg):
    ind = min(count.index(max(count[2:])), len(count) - 2)
    
    if False:
        iters = 30
        #print(np.array([get_theta_arr(x, count) for x in range(iters)]))

    if (ind < 2):
        eps = e if (e is not None) else default_error_rate
        theta = default_theta if (theta_arg is None) else theta_arg
        p0 = np.exp(-k * eps)
        r21 = 1.0 * count[2] / count[1]
        cov = newton(cov_temp_func, 0.05, args=(r21, p0, k, l)) 
        sys.stderr.write('[WARNING] Not enough information to co-estimate coverage, theta, and error rate of {0}; '.format(sample) +
                         'Using {0} as default error rate and {1} as theta\n'.format(theta))
    else:
        r =  count[ind + 1] / count[ind]
        rn =  count[ind + 2] / count[ind+1]
        #print(r, rn)
        if (8 * (2 + ind) * rn > 9 * (1 + ind) * r) or (theta_arg is not None):
            theta = default_theta if (theta_arg is None) else theta_arg
            #print(r)
            Q = (1-theta)**k
            xi = minimize(lambda x: (r+x*(-2* np.exp(x)+2* (-2**ind+np.exp(x)) * Q)/((1+ind)* (2* np.exp(x)+(2**ind-2* np.exp(x))*Q )))**2, 0.5, bounds = [[0,100]]).x[0]
            if theta_arg is None:
                sys.stderr.write('[WARNING] Not enough information to co-estimate coverage, theta, and error rate of {0}; '.format(sample) +
                         'Using default theta but computing other values {0}\n'.format(default_theta))
        else:
            xi = 1./4 * (3 * (1 + ind) * r - math.sqrt((1 + ind) * r) *math.sqrt(max(0, -8 * (2 + ind) * rn + 9 * (1 + ind) * r)))
            theta = 1 - 2**(1/ k) * ((np.exp(xi) * (-r - ind* r + xi))/( 2**ind *r - 2 * np.exp(xi) * r + 2**ind * ind * r - 2* np.exp(xi) * ind* r - 2**(1 + ind) * xi + 2 * np.exp(xi)* xi))**(1/k) 
            Q = (1-theta)**k
        
        # sets error rate
        if e is not None:
            eps = e
            lam = xi / (1-eps)**k
        else:
           lam = count[1]/count[ind] * ( (2 *np.exp(-xi) + (2**ind * np.exp(-2*xi) - 2 * np.exp(-xi))* Q) * (xi**ind) )/(2 * math.factorial(ind) ) + (-1 + np.exp(xi)) * (np.exp(-xi) + np.exp(-2*xi) * Q) * xi
           eps = 1-(xi/lam)**(1/k)

    return eps, lam*2.0, theta
