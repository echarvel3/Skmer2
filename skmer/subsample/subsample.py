import os
import random
import fnmatch
import errno
import pandas as pd
import sys
import math
import multiprocessing as mp
import numpy as np
import pyfastx
from subprocess import run

from skmer.config import *
from skmer.utils import sequence_stat, sketch, write_error_file, assign_skmer_label
from skmer.skmer import estimate_skmer_dist
from skmer.dipskmer import estimate_dipskmer_dist
from skmer.reskmer import parse_reference, estimate_reskmer_dist
from skmer.estimate_parameters import estimate_cov

def create_sketch_dir(sequence, lib, ce, ge, ee, le,  nth):
    sample = os.path.basename(sequence).rsplit('.f', 1)[0]
    sample_dir = os.path.join(lib, sample)
    try:
        os.makedirs(sample_dir)
    except OSError as Error:
        if Error.errno != errno.EEXIST:
            raise
    info_file = os.path.join(sample_dir, sample + '.dat')

    cov = ce[sample]
    g_len = ge[sample]
    eps = ee[sample]
    l = le[sample]
    write_error_file(info_file, cov, g_len, eps, l)
    return

def estimate_stats(sequence, nth):
    sample = os.path.basename(sequence).rsplit('.f', 1)[0]

    (l, ml, tl, n_reads) = sequence_stat(sequence)
    if ml > seq_len_threshold:
        cov = "NA"
        g_len = tl
        eps = 0
        l = "NA"
    else:
       # Set to dummy values for reads to initialize dictionaries. 
       # Will be recomputed for each subsample.
        cov = 0.0
        g_len = tl
        eps = 0.0
        l = l
    return sample, cov, g_len, eps, l, n_reads

def sample_reads(sequence, seed, bl_sz, bs_dir):
    #print(sequence, seed, bl_sz, bs_dir)
    try:
        os.makedirs(bs_dir)
    except OSError as Error:
        if Error.errno != errno.EEXIST:
            raise

    opener = gzip.open if sequence.endswith('gz') else open
    basename = os.path.split(sequence)[-1]
    if sequence.endswith('.gz'):
        basename = basename[:-3]
    bs_rep = os.path.join(bs_dir, basename)

    # First pass: count total reads
    with opener(sequence, 'rt') as f:
        total = sum(1 for _ in f) // 4

    # Reservoir sample read indices
    #random.seed(seed)
    chosen = set(random.sample(range(total), min(bl_sz, total)))

    # Second pass: write sampled reads
    with opener(sequence, 'rt') as fin, open(bs_rep, 'w') as fout:
        for i, lines in enumerate(zip(*[fin]*4)):
            if i in chosen:
                fout.writelines(lines)

#def sample_reads(sequence, seed, bl_sz, bs_dir):
#    print(sequence, seed, bl_sz, bs_dir)
#    try:
#        os.makedirs(bs_dir)
#    except OSError as Error:
#        if Error.errno != errno.EEXIST:
#            raise
#    
#    if sequence.endswith('gz'):
#        bs_rep = os.path.join(bs_dir, os.path.split(sequence)[-1][:-3])
#    else:
#        bs_rep = os.path.join(bs_dir, os.path.split(sequence)[-1])
#  
#    with open(bs_rep, 'w') as fp: 
#        run(["seqtk", "sample",  "-s", str(seed), sequence, str(bl_sz)], stdout=fp) 
#
#    return 

def subsample(args):
    # Creating a directory for subsample
    skmer_ver = assign_skmer_label(args)

    try:
        os.makedirs(args.sub)
    except OSError as Error:
        if Error.errno != errno.EEXIST:
            raise
    
    # ReSkmer reference processing...
    ref_hist=parse_reference(args.r, args.k, args.p, args.sub, skmer_ver) if args.r else None

    # Making a list of sample names
    files_names = [f for f in os.listdir(args.input_dir)
                   if True in (fnmatch.fnmatch(f, '*' + form) for form in formats)]
    #print(files_names)
    samples_names = [f.rsplit('.f', 1)[0] for f in files_names]
    # Check if refs have duplicate entry
    if len(samples_names) != len(set(samples_names)):
        raise ValueError('Duplicate inputs (possibly same name with different extensions), please change '
                         'the file name(s) and try again')

    # Making a list of genome-skim files
    sequences = [os.path.join(args.input_dir, f) for f in files_names]

    # Initializing distance dataframe
    index = pd.MultiIndex.from_product([samples_names, samples_names], names=['sample', 'sample_2'])
    result_df = pd.DataFrame(columns=index)


    # Initializing coverage, genome length, error rate, and read length dictionaries
    cov_est = dict()
    len_est = dict()
    err_est = dict()
    read_len = dict()
    bs_kmer_sum = dict()
    sample_read_cnt = dict()
    # Exclusively for dipskmer...
    if (skmer_ver == "dipskmer") or (skmer_ver == "reskmer + dipskmer"):
        theta = dict()

    # Number of pools and threads for multi-processing
    n_pool = min(args.p, len(sequences))
    n_thread_cov = int(args.p / n_pool)
    n_proc_cov = n_pool * n_thread_cov
    n_pool_dist = min(args.p, len(sequences) ** 2)

    
    # Computing coverage, genome length, error rate, read length and k-mer count
    sys.stderr.write('[{0}] Starting subsampling using {1} processors...\n'.format(skmer_ver, n_proc_cov))
    pool_cov = mp.Pool(n_pool)
    
    results_cov = [pool_cov.apply_async(estimate_stats, args=(seq, n_proc_cov))
                    for seq in sequences]
    
    #print("-1")
    for result in results_cov:
        (name, coverage, genome_length, error_rate, read_length, rd_cnt) = result.get(9999999)
        #print(name, coverage, genome_length, error_rate, read_length, rd_cnt)

        cov_est[name] = coverage
        len_est[name] = genome_length
        err_est[name] = error_rate
        read_len[name] = read_length
        sample_read_cnt[name] = rd_cnt
        bs_kmer_sum[name] = args.s
    
    #print("0")
    pool_cov.close()
    pool_cov.join()
    #print(sample_read_cnt)

    #print("1")
    # Check whether inputs are reads or assemblies
    if "NA" in list(read_len.values()):
        input_data = 'assemblies'
    else:
        input_data = 'reads'

    
    ### Choose procedure for reads or assemblies ###
    sys.stderr.write('[{0}] Input processed as {1}...\n'.format(skmer_ver, input_data))

    # Compute block size

    np.random.seed(args.S)
    rand_seed_list = list(np.random.randint(low = 0, high = 4294967294, size = args.b))
    #print(rand_seed_list)

    bs_block_sz = {}
    bs_sample_sz = {}
    coef = args.c
    asm_sketch_sz = 0

    if input_data == 'reads':
        bs_sample_sz = sample_read_cnt
        for key, value in sample_read_cnt.items():
            bs_block_sz [key] = round((value)**(coef))
    else:
        bs_sample_sz = bs_kmer_sum
        mean_bs_kmer_count = np.mean(list(bs_kmer_sum.values()))
        asm_sketch_sz = round((mean_bs_kmer_count)**(coef))
        for key, value in bs_kmer_sum.items():
           bs_block_sz[key] = asm_sketch_sz

    #print(bs_block_sz)
    #print(bs_sample_sz)



    # Computing replicates
    #print(args.b)
    for b in range (0, args.b):

        sys.stderr.write('[{0}] Computing replicate {1} using {2} processors...\n'.format(skmer_ver, b, n_pool))

        # Creating replicate directory
        sub_rep = os.path.join(args.sub, "rep" + str(args.i))
        #print(sub_rep)
        args.i +=1
        try:
            os.makedirs(sub_rep)
        except OSError as Error:
            if Error.errno != errno.EEXIST:
                raise

        # Creating replicate/library directory
        sub_lib = os.path.join(sub_rep, 'library')
        #print(sub_lib)
        try:
            os.makedirs(sub_lib)
        except OSError as Error:
            if Error.errno != errno.EEXIST:
                raise

        # Update paths for subsampled replicate
        bs_sequences = [os.path.join(sub_rep, os.path.split(seq)[-1]) for seq in sequences]


        # Creating a config file for subsample  replicate
        config_file = os.path.join(sub_rep, 'CONFIG')
        with open(config_file, mode='w') as f:
            f.write('kmer_length\t{0}\n'.format(args.k) + 'sketch_size\t{0}\n'.format(args.s) +
                'sketching_seed\t{0}\n'.format(rand_seed_list[b]))


        # Write sample size dictionary to file
        np.save(os.path.join(sub_rep, 'block_size.npy'), bs_block_sz)        
        np.save(os.path.join(sub_rep, 'sample_size.npy'), bs_sample_sz)

        # Update  coverage and error estimates for subsample
        if input_data == 'reads':

            # Generate subsample replicates and save to bootstrap directory
            pool_sketch = mp.Pool(n_pool)
            results_sketch = [pool_sketch.apply_async(sample_reads, args=(seq, rand_seed_list[b], bs_block_sz[(os.path.split(seq)[-1]).rsplit('.f', 1)[0]], sub_rep)) for seq in sequences]
            for result in results_sketch:
                result.get(int(9999999))
            pool_sketch.close()
            pool_sketch.join()


            # Computing coverage, genome length, error rate, and read length of replicates  using reference function
            pool_cov = mp.Pool(n_pool)
            
            results_cov = [estimate_cov(seq, sub_lib, args.k, args.e, args.p, skmer_ver, ref_hist, args.theta) for seq in bs_sequences]
    
            for result in results_cov:
                (name, coverage, genome_length, error_rate, read_length, theta_val) = result
                cov_est[name] = coverage
                len_est[name] = genome_length
                err_est[name] = error_rate
                read_len[name] = read_length
                if (skmer_ver == "dipskmer") or (skmer_ver == "reskmer + dipskmer"):
                    theta[name] = theta_val
            
            # if skmer_ver == "dipskmer":
            #     results_cov = [pool_cov.apply_async(estimate_diploid_cov, args=(seq, sub_lib, args.k, args.e, n_thread_cov, args.theta))
            #                 for seq in bs_sequences]
            # else:
            #     results_cov = [pool_cov.apply_async(estimate_cov, args=(seq, sub_lib, args.k, args.e, n_thread_cov, ref_hist))
            #                 for seq in bs_sequences]
            
            # for result in results_cov:
            #     if skmer_ver == "dipskmer":
            #         (name, coverage, genome_length, error_rate, read_length, theta_est) = result.get(9999999)
            #         theta[name] = theta_est
            #     else:    
            #         (name, coverage, genome_length, error_rate, read_length) = result.get(9999999)
            #     cov_est[name] = coverage
            #     len_est[name] = genome_length
            #     err_est[name] = error_rate
            #     read_len[name] = read_length
            #     print(name, coverage, genome_length, error_rate, read_length)
            pool_cov.close()
            pool_cov.join()


            # Sketching genome-skims
            pool_sketch = mp.Pool(n_pool)
            #reads_sketch_sz = 100000
            if skmer_ver == "reskmer":
                results_sketch = [pool_sketch.apply_async(sketch, args=(seq, sub_lib, cov_est, err_est, args.k, args.s,
                                                                    coverage_threshold, rand_seed_list[b], True, False)) for seq in bs_sequences]
            elif (skmer_ver == "dipskmer") or (skmer_ver == "reskmer + dipskmer"):
                results_sketch = [pool_sketch.apply_async(sketch, args=(seq, sub_lib, cov_est, err_est, args.k, args.s,
                                                                    dip_coverage_threshold, rand_seed_list[b], False, True, read_len)) for seq in bs_sequences]
            else:
                results_sketch = [pool_sketch.apply_async(sketch, args=(seq, sub_lib, cov_est, err_est, args.k, args.s,
                                                                    coverage_threshold, rand_seed_list[b], False, False)) for seq in bs_sequences]
            
            for result in results_sketch:
                result.get(9999999)
            pool_sketch.close()
            pool_sketch.join()


            # Estimating pair-wise distances
            pool_dist = mp.Pool(n_pool_dist)

            if skmer_ver == "reskmer":
                results_dist = [pool_dist.apply_async(estimate_reskmer_dist, 
                                                      args=(s1, s2, sub_lib, sub_lib, cov_est, len_est,
                                                            err_est, read_len, args.k, coverage_threshold, args.t, ref_hist))
                                                            for s1 in samples_names for s2 in samples_names]
            elif (skmer_ver == "dipskmer") or (skmer_ver == "reskmer + dipskmer"):
                results_dist = [pool_dist.apply_async(estimate_dipskmer_dist, 
                                                      args=(s1, s2, sub_lib, sub_lib, cov_est, len_est,
                                                            err_est, read_len, args.k, dip_coverage_threshold, args.t, theta))
                                                            for s1 in samples_names for s2 in samples_names]
            else:
                results_dist = [pool_dist.apply_async(estimate_skmer_dist, 
                                                      args=(s1, s2, sub_lib, sub_lib, cov_est, len_est,
                                                            err_est, read_len, args.k, coverage_threshold, args.t))
                                                            for s1 in samples_names for s2 in samples_names]

            for result in results_dist:
                dist_output = result.get(9999999)
                result_df[(dist_output[0], dist_output[1])] = [repr(dist_output[2])]


        else:

            # Prepare genome directory structure
            pool_sketch = mp.Pool(n_pool)
            #print(seq, sub_lib, cov_est, len_est, err_est)
            results_sketch = [pool_sketch.apply_async(create_sketch_dir, args=(seq, sub_lib, cov_est, len_est, err_est, 
                                                                               read_len, args.t)) for seq in sequences]
            pool_sketch.close()
            pool_sketch.join()



            # Sketching genomes
            pool_sketch = mp.Pool(n_pool)
            if skmer_ver == "reskmer":
                results_sketch = [pool_sketch.apply_async(sketch, args=(seq, sub_lib, cov_est, err_est, args.k, asm_sketch_sz,
                                                                    coverage_threshold, rand_seed_list[b], True)) for seq in sequences]
            elif (skmer_ver == "dipskmer") or (skmer_ver == "reskmer + dipskmer"):
                results_sketch = [pool_sketch.apply_async(sketch, args=(seq, sub_lib, cov_est, err_est, args.k, asm_sketch_sz,
                                                                    dip_coverage_threshold, rand_seed_list[b], False)) for seq in sequences]
            else:
                results_sketch = [pool_sketch.apply_async(sketch, args=(seq, sub_lib, cov_est, err_est, args.k, asm_sketch_sz,
                                                                    coverage_threshold, rand_seed_list[b], False)) for seq in sequences]
            
            for result in results_sketch:
                result.get(9999999)
            pool_sketch.close()
            pool_sketch.join()



            # Estimating pair-wise distances
            pool_dist = mp.Pool(n_pool_dist)
            if skmer_ver == "reskmer":
                results_dist = [pool_dist.apply_async(estimate_reskmer_dist, 
                                                      args=(s1, s2, sub_lib, sub_lib, cov_est, len_est,
                                                            err_est, read_len, args.k, coverage_threshold, args.t, ref_hist))
                                                            for s1 in samples_names for s2 in samples_names]
            elif (skmer_ver == "dipskmer") or (skmer_ver == "reskmer + dipskmer"):
                results_dist = [pool_dist.apply_async(estimate_dipskmer_dist, 
                                                      args=(s1, s2, sub_lib, sub_lib, cov_est, len_est,
                                                            err_est, read_len, args.k, dip_coverage_threshold, args.t, theta))
                                                            for s1 in samples_names for s2 in samples_names]
            else:
                results_dist = [pool_dist.apply_async(estimate_skmer_dist, 
                                                      args=(s1, s2, sub_lib, sub_lib, cov_est, len_est,
                                                            err_est, read_len, args.k, coverage_threshold, args.t))
                                                            for s1 in samples_names for s2 in samples_names]

            for result in results_dist:
                dist_output = result.get(9999999)
                result_df[(dist_output[0], dist_output[1])] = [repr(dist_output[2])]



        # Writing distances to file
        sys.stderr.write('[{0}] Writing to file...\n'.format(skmer_ver))
        result_dfm = pd.melt(result_df, value_name='distance')
        result_mat = result_dfm.pivot(index='sample', columns='sample_2', values='distance')
        final_path = os.path.join(sub_rep, "dimtrx_rep" + ".txt")
        result_mat.to_csv(final_path, float_format='%f', sep='\t', mode='w')


        # Cleaning up 
        if args.fa:
                for fi in bs_sequences:
                    try:
                        os.remove(fi)
                    except OSError:
                        pass

        if args.msh:
            sketch_fi = []
            for (dirpath, dirnames, filenames) in os.walk(sub_lib):
                sketch_fi += [os.path.join(dirpath, file) for file in filenames if file.endswith(".msh") ]
            #print(sketch_fi)
            for fi in sketch_fi:
                try:
                    os.remove(fi)
                except OSError:
                    pass


    # Clean up subsample folders 
    #import shutil
    #shutil.rmtree(sub_lib)
    #shutil.rmtree(args.bs)
    #shutil.rmtree(args.l)

def correction(args):
     
    # Making a list of sample names
    try:
        #with open(args.main,"r") as f:
        df = pd.read_csv(args.main, header = 0, sep='\t', skiprows = 0)
        samples_names = list(df.iloc[:,0])
        #print(samples_names)
    except:
        raise ValueError('Please check file name for main distance matrix and try again')


    # Initializing distance dataframe
    index = pd.MultiIndex.from_product([samples_names, samples_names], names=['sample', 'sample_2'])
    result_df = pd.DataFrame(columns=index)
    combo_result_df = pd.DataFrame()


    # Round distances up to 12 digits since fastme doesn't except more than 12 decimals
    no_strap_dfm = pd.melt(df, id_vars=['sample'], value_vars= list(df.columns[1:]) )
    no_strap_dfm.rename(columns={'variable':'sample_2'}, inplace=True) 
    no_strap_dfm.rename(columns={'value':'no_strapped_dist'}, inplace=True)    

    decimals = 12
    no_strap_dfm['no_strapped_dist'] = no_strap_dfm['no_strapped_dist'].apply(pd.to_numeric, errors='coerce')
    no_strap_dfm['no_strapped_dist'] =  no_strap_dfm['no_strapped_dist'].apply(lambda x: round(x, decimals))

    no_strap_mat = no_strap_dfm.pivot(index='sample', columns='sample_2', values='no_strapped_dist')
    no_strap_mat.to_csv(os.path.splitext(args.main)[0] + "_cor_" +  ".txt", float_format='%f', sep='\t', mode='w')

    # List replicate directories
    try:
        for dir in [name for name in os.listdir(args.sub) if 'rep' in name]:
            #print(dir)
            rep_mtrx = os.path.join(args.sub, dir, "dimtrx_rep.txt")
            df = pd.read_csv(rep_mtrx, header = 0, sep='\t', skiprows = 0)

            # Append estimates to combo dataframe
            result_dfm = pd.melt(df, id_vars=['sample'], value_vars= list(df.columns[1:]) )
            result_dfm.rename(columns={'variable':'sample_2'}, inplace=True)
            result_dfm.rename(columns={'value':'uncorrected_dist'}, inplace=True)
            result_dfm['rep'] = int(dir.split('rep', 1)[-1])

            # Load dictionaries
            bs_block_sz = np.load(os.path.join(args.sub, dir, 'block_size.npy'), allow_pickle='TRUE').item()
            bs_sample_sz = np.load(os.path.join(args.sub, dir, 'sample_size.npy'), allow_pickle='TRUE').item()
            
            result_dfm['b_s1'] = result_dfm['sample'].map(bs_block_sz)
            result_dfm['b_s2'] = result_dfm['sample_2'].map(bs_block_sz)
            result_dfm['b_s1'] = result_dfm['b_s1'].apply(pd.to_numeric, errors='coerce')
            result_dfm['b_s2'] = result_dfm['b_s2'].apply(pd.to_numeric, errors='coerce')
            result_dfm['b_mean'] = result_dfm[['b_s1', 'b_s2']].mean(axis=1, skipna=True)
            
            result_dfm['n_s1'] = result_dfm['sample'].map(bs_sample_sz)
            result_dfm['n_s2'] = result_dfm['sample_2'].map(bs_sample_sz)
            result_dfm['n_s1'] = result_dfm['n_s1'].apply(pd.to_numeric, errors='coerce')
            result_dfm['n_s2'] = result_dfm['n_s2'].apply(pd.to_numeric, errors='coerce')
            result_dfm['N_mean'] = result_dfm[['n_s1', 'n_s2']].mean(axis=1, skipna=True)
            combo_result_df = combo_result_df.append(result_dfm, ignore_index = True)
            #print(combo_result_df)

    except:
        raise ValueError('Please check subsample directory and try again')


   
    # Computing distance correction

    combo_result_df['uncorrected_dist'] = combo_result_df['uncorrected_dist'].apply(pd.to_numeric, errors='coerce')
    res = combo_result_df.groupby(['sample', 'sample_2'], as_index=False)['uncorrected_dist'].mean()
    res.rename({'uncorrected_dist': 'subsample_mean_dist'}, axis=1, inplace=True)    
    new_df = pd.merge(combo_result_df, res,  how='left', left_on=['sample','sample_2'], right_on = ['sample','sample_2'])

    new_df_out = pd.merge(new_df, no_strap_dfm,  how='left', left_on=['sample','sample_2'], right_on = ['sample','sample_2'])
    new_df_out['no_strapped_dist'] = new_df_out['no_strapped_dist'].apply(pd.to_numeric, errors='coerce')
    
    new_df_out['corrected_dist'] = ((new_df_out['b_mean']/new_df_out['N_mean'])**(1/2))*(new_df_out['uncorrected_dist']-new_df_out['subsample_mean_dist'])+new_df_out['no_strapped_dist']
    new_df_out['corrected_dist_cons'] = ((new_df_out['b_mean']/new_df_out['N_mean'])**(1/2))*(new_df_out['uncorrected_dist']-new_df_out['subsample_mean_dist'])+new_df_out['subsample_mean_dist']
    
    #replace negative values with 0.0 so fastme can handle matrices
    new_df_out.corrected_dist = np.where(new_df_out.corrected_dist < 0, 0.0, new_df_out.corrected_dist)
    new_df_out.corrected_dist_cons = np.where(new_df_out.corrected_dist_cons < 0, 0.0, new_df_out.corrected_dist_cons)

    # round distances up to 12 digits since fastme doesn't except more than 12 decimals
    new_df_out['corrected_dist'] =  new_df_out['corrected_dist'].apply(lambda x: round(x, decimals))
    new_df_out['corrected_dist_cons'] =  new_df_out['corrected_dist_cons'].apply(lambda x: round(x, decimals))
    new_df_out.to_csv(os.path.join(args.sub, "_summary" + ".csv"), float_format='%f', sep=',', mode='w')



    # Writing distances to file
    sys.stderr.write('[skmer] Writing to file...\n')
    
    b_list = list((new_df_out.loc[:, 'rep']).unique())
    #print(b_list)
    for b in b_list:
        sub_dfm = new_df_out.loc[(new_df_out['rep'] == b)]
        
        #-mean+main
        sub_dfm_main = sub_dfm[['sample','sample_2', 'corrected_dist']]
        result_mat_main = sub_dfm_main.pivot(index='sample', columns='sample_2', values='corrected_dist')
        result_mat_main.to_csv(os.path.join(args.sub, "rep" + str(b),  "dimtrx_rep_cor.txt"), float_format='%f', sep='\t', mode='w')
        
        #-mean+mean
        sub_dfm_cons = sub_dfm[['sample','sample_2', 'corrected_dist_cons']]
        result_mat_cons = sub_dfm_cons.pivot(index='sample', columns='sample_2', values='corrected_dist_cons')
        result_mat_cons.to_csv(os.path.join(args.sub, "rep" + str(b),  "dimtrx_rep_cor_cons.txt"), float_format='%f', sep='\t', mode='w')
