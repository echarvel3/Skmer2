import numpy as np
import math
from scipy.optimize import newton
import sys

from skmer.utils import cov_temp_func
from skmer.config import *

def estimate_base_cov(sample, count, k, e, l):
    ind = min(count.index(max(count[2:])), len(count) - 2)
    # given a default error rate or is too low of a coverage
    if (e is not None) or (ind < 2):
        eps = e if (e is not None) else default_error_rate
        p0 = np.exp(-k * eps)
        if ind < 2:
            r21 = 1.0 * count[2] / count[1]
            cov = newton(cov_temp_func, 0.05, args=(r21, p0, k, l))
            sys.stderr.write('Not enough information to co-estimate coverage and error rate of {0}; '.format(sample) + 'Using {0} as a default error rate.\n'.format(eps))            
        else:
            cov = (1.0 / p0) * (1.0 * l / (l - k)) * (ind + 1) * count[ind + 1] / count[ind]
        lam = cov * (l-k)/l * 1.0
    else:
        gam = 1.0 * (ind + 1) * count[ind + 1] / count[ind]
        lam = (np.exp(-gam) * (gam ** ind) / math.factorial(ind)) * count[1] / count[ind] + gam * (1 - np.exp(-gam))
        eps = 1 - (gam / lam) ** (1.0 / k)
    return eps, lam
