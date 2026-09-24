
import os
import numpy as np
import math

from subprocess import check_output, STDOUT
from scipy.stats import poisson
from skmer.config import *

from skmer.utils import get_hist_data, check_mash_files


def estimate_dipskmer_dist(sample_1, sample_2, lib_1, lib_2, ce, le, ee, rl, k, cov_thres, tran, theta):
    if sample_1 == sample_2 and lib_1 == lib_2:
        return sample_1, sample_2, 0.0
    sample_dir_1 = os.path.join(lib_1, sample_1)
    sample_dir_2 = os.path.join(lib_2, sample_2)
    gl_1 = le[sample_1]
    gl_2 = le[sample_2]
    if gl_1 == "NA" or gl_2 == "NA":
        gl_1 = 1
        gl_2 = 1
    cov_1 = ce[sample_1]/2.0 if ce[sample_1] != "NA" else 1.0
    cov_2 = ce[sample_2]/2.0 if ce[sample_2] != "NA" else 1.0

    eps_1 = ee[sample_1] if ee[sample_1] != "NA" else default_error_rate
    eps_2 = ee[sample_2] if ee[sample_2] != "NA" else default_error_rate

    l_1 = rl[sample_1]
    l_2 = rl[sample_2]
    theta_1 = theta[sample_1]
    theta_2 = theta[sample_2]
    
    def dip_dist_temp_func(cov, eps, k, l, cov_thres, theta):
        if cov == "NA":
            return [1.0, 0]
        p = np.exp(-k * eps)
        lam = 1.0 * cov * (l - k) / l
        xi = lam * p
        copy_thres = int(1.0 * xi / (1.0* cov_thres)) + 1
        #print(cov,eps,cov_thres)
        if copy_thres == 1 or p == 1:
            return [1 - np.exp(- xi), lam * (1 - p)]
        else:
            #print("cov thresh: ", cov_thres, copy_thres, cov, cov/cov_thres, p)
            s = [np.exp(-lam*(1-eps)**k) * math.pow(lam*(1-eps)**k, i)/math.factorial(i) for i in range(copy_thres)] 
            #print(1-sum(s))
            return [1 - sum(s), 0]


    #print("pars used:", eps_1,eps_2,cov_1,cov_2)
    r_1 = dip_dist_temp_func(cov_1, eps_1, k, l_1, cov_thres, theta_1)
    r_2 = dip_dist_temp_func(cov_2, eps_2, k, l_2, cov_thres, theta_2)
    
    hist_1, size_1, usize_1 = get_hist_data(lib_1, sample_1)
    hist_2, size_2, usize_2 = get_hist_data(lib_2, sample_2)
    
    lam_1 = 1.0 * cov_1 * (l_1 - k) / l_1
    lam_2 = 1.0 * cov_2 * (l_2 - k) / l_2
    #psi_1 = 2*lam_1/2*(1-np.power(1-eps_1, k))
    #psi_2 = 2*lam_2/2*(1-np.power(1-eps_2, k))
    psi_1 = r_1[1]
    psi_2 = r_2[1]
    eta1= r_1[0]
    eta2= r_2[0]
    #print("psi1", psi_1, "psi_2", psi_2, "j", j)
    xi_1 = lam_1 * np.exp(-k*eps_1)
    xi_2 = lam_2 * np.exp(-k*eps_2)
    #t = int(1.0 * (xi_1+xi_2)/2.0 / (1.0* cov_thres)) + 1
    t1 = int(1.0 * (xi_1) / (1.0* cov_thres)) + 1
    t2 = int(1.0 * (xi_2) / (1.0* cov_thres)) + 1
    #print("   t: ",t1,t2)
    #print("   P: ",psi_1,psi_2)
    msh_1 = check_mash_files(os.path.join(sample_dir_1, sample_1 + "t" + str(t1)),t1) #os.path.join(sample_dir_1, sample_1 + ".msh")
    msh_2 = check_mash_files(os.path.join(sample_dir_2, sample_2 + "t" + str(t2)) ,t2) #os.path.join(sample_dir_2, sample_2 + ".msh")
    #print("Sketches:\n%s\n%s" %(msh_1,msh_2))
    if msh_1 is None :
        #print(os.path.join(sample_dir_1, sample_1 + "t" + str(t1), "not found"))
        raise FileNotFoundError(os.path.join(sample_dir_1, sample_1 + "t" + str(t1), "not found"))
    if msh_2 is None:
        #print(os.path.join(sample_dir_2, sample_2 + "t" + str(t2), "not found"))
        raise FileNotFoundError(os.path.join(sample_dir_2, sample_2 + "t" + str(t2), "not found"))
    dist_stderr = check_output(["mash", "dist", msh_1, msh_2], stderr=STDOUT, universal_newlines=True)
    j = float(dist_stderr.split()[4].split("/")[0]) / float(dist_stderr.split()[4].split("/")[1])
    #nt = lambda x, y : 1 - poisson.cdf(t-1, x*xi_1 + y*xi_2)
    i = j * (usize_1 + usize_2) / (1.0 + j)
    EI = i / gl_1
    #print(usize_1, usize_2)
    #numerator = (11*EI) + (4*r_1[0]*r_2[0] * ( r_1[0] + r_2[0] - 5))
    #denominator = r_1[0]*r_2[0] * (11*r_1[0]*r_2[0] - 18*(r_1[0]+r_2[0]) + 24)

    n11 = 1 - poisson.cdf(t1-1, 1*xi_1)
    n12 = 1 - poisson.cdf(t1-1, 2*xi_1)
    n21 = 1 - poisson.cdf(t2-1, 1*xi_2)
    n22 = 1 - poisson.cdf(t2-1, 2*xi_2)
    Qn = 1 + 11 * ( j *(n22 + n12) - (1 + j) * n12 * n22 + j*(2*psi_1 + 2*psi_2))/(6 * j* (2 * n21 + 2 * n11 - n22 - n12) + (j + 1) * (-4 * n21 * n11 - 4 * n22 * n11 - 4*n21 * n12 + 11 * n22 * n12) )
    d = 1 - np.power(Qn, (6/11 * 1/k))
    if False:
        if t > 1:
            
            nt = lambda x, y : 1 - poisson.cdf(t-1, x*xi_1 + y*xi_2)

            #print("printing t>1 equation, exact solution, since coverage is high:", cov_1, cov_2)
            #numerator = 11*(j*nt(2,2) - nt(0,2)*nt(2,0))
            #numerator = j*(4*nt(0,1)*(nt(1,0)+nt(2,0)-3)+4*nt(0,2)*nt(1,0) - 5*nt(0,2) - 12*nt(1,0) - 5*nt(2,0)) + 4*nt(1,0) *(nt(1,0) + nt(2,0)) + 4*nt(0,2)*nt(1,0)
            #denominator = 4*nt(1,0)*(j*(nt(0,1) + nt(0,2) - 3) + nt(0,1) + nt(0,2)) + nt(2,0)*(j*(4*nt(0,1) - 11*nt(0,2) + 6) + 4*nt(0,1) - 11 * nt(0,2)) + 6 *j*(2*nt(0,2) - 2*nt(0,1))
    #        print(numerator)
            #denom_1 = j*(4*nt(0,1) + nt(0,2) + 4*nt(1,0) + 4*nt(1,1) + 4*nt(1,2) + nt(2,0) + 4*nt(2,1) - 11*nt(2,2))
    #        print(denom_1)
            #denom_2 = -4*nt(0,1) * (nt(1,0) + nt(2,0)) + nt(0,2)*(11*nt(2,0) - 4*nt(1,0))
    #        print(denom_2)

        
            #Q = 1 + numerator / (denom_1 + denom_2)
            #Q = numerator / denominator

            numerator = (4* nt(0,2) *nt(1,0) + 4 *nt(0,1) *(nt(1,0) + nt(2,0)) + j* (-5* nt(0,2) - 12* nt(1,0) + 4* nt(0,2)* nt(1,0) - 5 *nt(2,0) + 4* nt(0,1) *(-3 + nt(1,0) + nt(2,0))))
            denominator = (6* j* (-2* nt(0,1) + nt(0,2)) +  4* (nt(0,1) + nt(0,2) + j *(-3 + nt(0,1) + nt(0,2)))* nt(1,0) + (4* nt(0,1) +  j* (6 + 4* nt(0,1) - 11* nt(0,2)) - 11* nt(0,2))* nt(2,0))
            
            Q= numerator/denominator
            #print("   Q:",Q, Qn)
            
            #numerator = 11*(j*nt(2,2) - nt(0,2)*nt(2,0))
            #denom_1 = j*(4*nt(0,1) + nt(0,2) + 4*nt(1,0) + 4*nt(1,1) + 4*nt(1,2) + nt(2,0) + 4*nt(2,1) - 11*nt(2,2))
            #denom_2 = -4*nt(0,1) * (nt(1,0) + nt(2,0)) + nt(0,2)*(11*nt(2,0) - 4*nt(1,0))
            #Q = 1 + numerator / (denom_1 + denom_2)
        
            d = 1 - np.power(Q, (6/11 * 1/k))
        else:
            #print("printing t=1 equation since coverage is low:", cov_1, cov_2)
            numerator = j * ( -5*(eta1**2 +eta2**2) + 22*(eta1+ eta2+ psi_1 + psi_2) ) + 4 * (1+j) * eta2*eta1 *( eta1 + eta2 - 5 )
            #print(numerator)
            power = (6/11 * 1/k)
            #print(power)
            denominator = eta1*eta2*(11*eta2*eta1 +24 -18*eta2 -18*eta1)*(1 + j) + 6*j*(eta2**2 + eta1**2)
            #print(denominator)
            Q=numerator/denominator
            #print("   eta:",n11,n12, eta1,eta1*(2-eta1),n21,n22, eta2,eta2*(2-eta2))
            #print("   Q:",Q, Qn)
            d = 1 - np.power(numerator/denominator, (6/11 * 1/k))

    if tran or math.isnan(d):
        if d < 0.75:
            d = max(0, -0.75 * np.log(1 - 4.0 * d / 3.0))
        else:
            d = 'nan'
    d = 0.0 if float(d) < 0.0 else round(float(d), 6)
    #print("distance:", sample_1, sample_2, ":", str(d), 'xi:', xi_1, xi_2)
    return sample_1, sample_2, d

# NOTE: OBSOLETE
# def estimate_dipskmer_dist_approx(sample_1, sample_2, lib_1, lib_2, ce, le, ee, rl, k, cov_thres, tran, theta):
#     if sample_1 == sample_2 and lib_1 == lib_2:
#         return sample_1, sample_2, 0.0
#     sample_dir_1 = os.path.join(lib_1, sample_1)
#     sample_dir_2 = os.path.join(lib_2, sample_2)
#     msh_1 = os.path.join(sample_dir_1, sample_1 + ".msh")
#     msh_2 = os.path.join(sample_dir_2, sample_2 + ".msh")
#     dist_stderr = check_output(["mash", "dist", msh_1, msh_2], stderr=STDOUT, universal_newlines=True)
#     j = float(dist_stderr.split()[4].split("/")[0]) / float(dist_stderr.split()[4].split("/")[1])
#     gl_1 = le[sample_1]
#     gl_2 = le[sample_2]
#     if gl_1 == "NA" or gl_2 == "NA":
#         gl_1 = 1
#         gl_2 = 1
#     cov_1 = ce[sample_1]
#     cov_2 = ce[sample_2]
#     eps_1 = ee[sample_1]
#     eps_2 = ee[sample_2]
#     l_1 = rl[sample_1]
#     l_2 = rl[sample_2]
#     theta_1 = theta[sample_1]
#     theta_2 = theta[sample_2]

#     #print("pars used:", eps_1,eps_2,cov_1,cov_2)
#     r_1 = dip_dist_temp_func(cov_1, eps_1, k, l_1, cov_thres, theta_1)
#     r_2 = dip_dist_temp_func(cov_2, eps_2, k, l_2, cov_thres, theta_2)
    
#     hist_1, size_1, usize_1 = get_hist_data(lib_1, sample_1)
#     hist_2, size_2, usize_2 = get_hist_data(lib_2, sample_2)
    
#     i = j * (usize_1 + usize_2) / (1.0 + j)
#     EI = i / gl_1
#     #print(usize_1, usize_2)
#     #numerator = (11*EI) + (4*r_1[0]*r_2[0] * ( r_1[0] + r_2[0] - 5))
#     #denominator = r_1[0]*r_2[0] * (11*r_1[0]*r_2[0] - 18*(r_1[0]+r_2[0]) + 24)
#     power = (6/11 * 1/k)
#     lam_1 = 1.0 * cov_1 * (l_1 - k) / l_1
#     lam_2 = 1.0 * cov_2 * (l_2 - k) / l_2
#     #psi_1 = 2*lam_1/2*(1-np.power(1-eps_1, k))
#     #psi_2 = 2*lam_2/2*(1-np.power(1-eps_2, k))
#     psi_1 = r_1[1]
#     psi_2 = r_2[1]
#     eta1= r_1[0]
#     eta2= r_2[0]
#     #print("psi1", psi_1, "psi_2", psi_2, "j", j)

#     numerator = j * ( -5*(eta1**2 +eta2**2) + 22*(eta1+ eta2+ psi_1 + psi_2) ) + 4 * (1+j) * eta2*eta1 *( eta1 + eta2 - 5 )
#     #4*eta1*eta2*(eta1 + eta2 - 5) + j *((4*eta1-5)*(eta2**2) + 4*(eta1-5)*eta1*eta2+(22-5*eta1)*eta1 + 11*(2*eta2 + psi_1 + psi_2))
#     denominator = eta1*eta2*(11*eta2*eta1 +24 -18*eta2 -18*eta1)*(1 + j) + 6*j*(eta2**2 + eta1**2)
#     #(r_2[0]**2)*(11*(r_1[0]**2)*(j+1) - (18*r_1[0]*(j+1)) + 6*j) + 6*(r_1[0]**2)*j - 6*(3*r_1[0]-4)*r_1[0]*r_2[0]*(j+1)
#     #print("eta1", r_1[0], "eta2", r_2[0], "int per pos",  EI, "int", i, "num", numerator, "denom", denominator, "pow", power)
#     d = 1 - np.power(numerator/denominator,power)
    
#     if tran or math.isnan(d):
#         if d < 0.75:
#             d = max(0, -0.75 * np.log(1 - 4.0 * d / 3.0))
#         else:
#             d = 'nan'
#     print(d)
#     return sample_1, sample_2, d
