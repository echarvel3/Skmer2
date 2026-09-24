import os
import fnmatch
import sys
import pandas as pd
import numpy as np
import pyfastx

from subprocess import check_output, STDOUT, call, run
from skmer.config import *

def pop(args):
    return

import pyfastx

def sequence_stat(sequence):
    total_length = 0
    n_reads = 0
    max_length = 0

    fx = pyfastx.Fastx(sequence)  # autodetect fasta/fastq
    for name, seq, *rest in fx:
        s = seq.upper()
        read_length = s.count("A") + s.count("C") + s.count("G") + s.count("T")
        total_length += read_length
        max_length = max(max_length, read_length)
        n_reads += 1

    if n_reads == 0:
        return 0, 0, 0, 0
    return int(round(total_length / n_reads)), max_length, total_length, n_reads


#def sequence_stat(sequence):
#    total_length = 0
#    n_reads = 0
#    max_length = 0
#    # TODO: seqtk comp causing slowing issues in some clusters especially with Skmer subsample.
#    # NOTE: potential fix is to replace with "bbduk.sh in=X lhist=X.txt"
#    comp_stdout = check_output(["seqtk", "comp", sequence], stderr=STDOUT, universal_newlines=True)
#    reads_stat = comp_stdout.split('\n')
#    for stat in reads_stat:
#        if not stat.strip():
#            continue
#        read_length = sum([int(x) for x in stat.split('\t')[2:6]])
#        total_length += read_length
#        max_length = max(max_length, read_length)
#        n_reads += 1
#    return int(round(1.0 * total_length / n_reads)), max_length, total_length, n_reads

def check_mash_files(msh, copy_thres):
    #for msht in [msh,msh + "t%d" %copy_thres]:
    mshf = msh+".msh"
    #print("mshf",mshf)
    if (os.path.exists(mshf)):
        mashcounts = run(["mash","info","-c",mshf],capture_output=True, text=True)
        et = int(mashcounts.stdout.split('\n')[1].split('\t')[1])
        #print("Detected sketch %s with t=%d looking for %d" %(mshf,et,copy_thres))
        if et == copy_thres:
            return mshf
        #print("et", et, "copy thres", copy_thres)
    return None

def sketch(sequence, lib, ce, ee, k, s, cov_thres, seed, has_spectrum = False, is_diploid = False, re = None):
    #print("sketching")
    sample = os.path.basename(sequence).rsplit('.f', 1)[0]
    sample_dir = os.path.join(lib, sample)
    msh = os.path.join(sample_dir, sample)
    
    cov = ce[sample]/2.0 if is_diploid and (ce[sample] != 'NA') else ce[sample]

    eps = ee[sample]
    r_len = re[sample] if re is not None else None
    #print(r_len)
    #print(cov, cov_thres)
    if cov == "NA" and eps == 0:
        #print("1")
        call(["mash", "sketch", "-k", str(k), "-s", str(s), "-S", str(seed), "-o", msh, sequence], stderr=open(
            os.devnull, 'w'))
        return
    elif eps == "NA":
        #print("2")
        if is_diploid:
            msh = (msh+ "t1") if r_len is not None else (msh)
            if check_mash_files(msh, 1):
                #print("Skipping sketching, as existing sketch seems right" )
                return
        call(["mash", "sketch", "-k", str(k), "-s", str(s), "-S", str(seed), "-r", "-o", msh, sequence], stderr=open(
            os.devnull, 'w'))
        return
    if r_len is not None:
        lam =  1.0 * cov * (r_len - k) / r_len
        cov = xi = lam * np.exp(-k*eps)
    copy_thres = int(cov / cov_thres) + 1
    #print("t:",copy_thres, "cov", xi, cov, "r", r_len)
    
    msh = (msh+ "t%d" %copy_thres) if r_len is not None else (msh)
    if check_mash_files(msh, copy_thres):
        #print("Skipping sketching, as existing sketch seems right" )
        return

    #print("sketching infto %s" %msh)
    if copy_thres ==1 or eps == 0.0 or has_spectrum:
        #print("3")
        # ReSkmer or below Skmer high-cov threshold...
        call(["mash", "sketch", "-k", str(k), "-s", str(s), "-S", str(seed), "-r", "-o", msh, sequence], stderr=open(
            os.devnull, 'w'))
    else:
        # high-coverage Skmer
        #print("HERE")
        call(["mash", "sketch", "-m", str(copy_thres), "-k", str(k), "-s", str(s), "-S", str(seed), "-o", msh,
              sequence], stderr=open(os.devnull, 'w'))
    return

def get_samples_from_files(args):
    files_names = [f for f in os.listdir(args.input_dir)
                   if True in (fnmatch.fnmatch(f, '*' + form) for form in formats)]
    samples_names = [f.rsplit('.f', 1)[0] for f in files_names]

    # Check if refs have duplicate entry or no entries
    if not samples_names:
        raise FileNotFoundError("No files with extensions %s found" % " ".join(formats))
    elif len(samples_names) != len(set(samples_names)):
        raise ValueError('Duplicate inputs (possibly same name with different extensions), please change '
                         'the file name(s) and try again')
    return(files_names, samples_names)

def write_config_file(args):
    '''Creating a config file for references'''
    config_file = os.path.join(args.l, 'CONFIG')
    with open(config_file, mode='w') as f:
        f.write('kmer_length\t{0}\n'.format(args.k) + 'sketch_size\t{0}\n'.format(args.s) +
                'sketching_seed\t{0}\n'.format(args.S))
    #TODO: add skmer version to config files, maybe write 
    return()

def assign_skmer_label(args):
    '''assigns "skmer", "reskmer", or "dipskmer" based on input arguments'''
    if args.r:
        skmer_ver = "reskmer"
    elif args.d:
        skmer_ver = "dipskmer"
    else:
        skmer_ver = "skmer"

    if args.r and args.d:
        skmer_ver = "reskmer + dipskmer"
    return(skmer_ver)

def write_error_file(info_file, cov, g_len, eps, l, theta = None):
    cov = float(round(cov, 5)) if type(cov) != str else cov
    eps = float(round(eps, 5)) if type(eps) != str else eps

    with open(info_file, mode='w') as f:
        f.write('coverage\t{0}\n'.format(cov) + 'genome_length\t{0}\n'.format(g_len) +
                'error_rate\t{0}\n'.format(eps) + 'read_length\t{0}\n'.format(l))
        if theta is not None:
            theta = float(round(theta, 5)) if type(theta) != str else theta
            f.write('theta\t{0}\n'.format(theta))

def count_kmers(sample_dir, sample, sequence, k, nth):
    '''runs jellyfish if jellyfish file does not already exist'''
    #TODO: add alternatives to jellyfish to count k-mers
    mercnt = os.path.join(sample_dir, sample + '.jf')
    histo_file = os.path.join(sample_dir, sample + '.hist')

    # Runs jellyfish if .hist file does not exist.
    if (not os.path.exists(histo_file)) or (os.path.getsize(histo_file) == 0):
        mercnt = os.path.join(sample_dir, sample + '.jf')
        # Reads gzipped data
        # NOTE: only works in specific environments? Explore alternative k-mer counters with friendlier gzip reading.
        if sequence.endswith("gz"):
            jellyfish_cmd = ["zcat", sequence, "|", "jellyfish", "count", "-m", str(k), "-s", "100M", "-t", str(nth), "-C", "-o", mercnt, "/dev/fd/0"]
            run(" ".join(jellyfish_cmd), shell=True, check=True)
        else:
            call(["jellyfish", "count", "-m", str(k), "-s", "100M", "-t", str(nth), "-C", "-o", mercnt, sequence],
                stderr=open(os.devnull, 'w'))
        histo_stderr = check_output(["jellyfish", "histo", "-h", "1000000", mercnt], stderr=STDOUT, universal_newlines=True)
        with open(histo_file, mode='w') as f:
            f.write(histo_stderr)
        os.remove(mercnt)
    else:  
        sys.stderr.write("\033[91m" + '[!WARNING!] {0}.hist already exists. Using existing file.\n'.format(sample) + "\033[0m")
        histo_stderr = open(histo_file).read()
    return(histo_stderr)

def get_hist_data(lib, sample):
    '''reads a skim's kmer histogram...'''

    sample_dir = os.path.join(lib, sample)
    histo_file = os.path.join(sample_dir, sample + '.hist')
    ref_hist = pd.read_csv(histo_file, sep=' ', header=None)
    # sum of all kmers in a histogram
    ksum = np.dot(ref_hist.iloc[:, 0], ref_hist.iloc[:, 1])
    # count of all unique kmers
    usum = sum(ref_hist.iloc[:, 1])
    return ref_hist, ksum, usum

def cov_temp_func(x, r, p, k, l):
    lam = x * (1.0 * (l - k)) / l
    return lam * (p ** 2) * np.exp(-lam * p) - 2 * r * (p * np.exp(-lam * p) + 1 - p)
