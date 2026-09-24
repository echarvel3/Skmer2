import sys
import numpy as np
import math
import os
import io
import pandas as pd
import numpy as np
from numpy import random
from subprocess import call, check_output, STDOUT

def genome_size_correction(ref_hist, ref_names, ref_lengths):
    '''scale repeat spectrum bins by genome size'''
    for sample in ref_names:
        genome_length = ref_lengths[sample].loc['genome_length']
        #print(ref_hist.iloc[:, 0])
        #print(ref_hist[sample])
        
        ksum = np.dot(ref_hist.iloc[:, 0], ref_hist[sample].astype('float64'))

        if genome_length > ref_hist[sample].iloc[0]:
            # upscales bins (2 to 50) by estimate of genome size
            new_R_factor = 1 if (ksum - ref_hist[sample].iloc[0] < 0.00001) else (genome_length - ref_hist[sample].iloc[0]) / (ksum - ref_hist[sample].iloc[0])
            new_Rs = new_R_factor * ref_hist[sample].iloc[1:]
            ref_hist.loc[1:,sample] = [int(x) for x in new_Rs]
        else:
            # if bin one is already larger than the genome size, downscale all bins by genome size
            new_R_factor = genome_length/ksum
            new_Rs = new_R_factor * ref_hist[sample].iloc[0:]
            ref_hist.loc[0:,sample] = [int(x) for x in new_Rs]  
    return(ref_hist)

def smooth_repeat_spectrum(ref_hist, ref_names):
    '''Perform smoothing of respect-inferred repeat spectrum'''
    stop_count = 0
    pseudo_count = math.pow(10, -10)
    for sample in ref_names: 
        # target slope for smoothing
        slope = (math.log(ref_hist.loc[10, sample] + pseudo_count) - math.log(ref_hist.loc[len(ref_hist)-3, sample] + pseudo_count)/2 - math.log(ref_hist.loc[len(ref_hist)-2, sample] + pseudo_count)/2) / 40

        stop_count = 0
        for i in random.randint(10,ref_hist.shape[0]-2, 5000):
            # if the absolute difference between one spectra and the next is LARGER than the slope:
            if abs(math.log(ref_hist.loc[i, sample] + pseudo_count) - math.log(ref_hist.loc[i+1, sample] + pseudo_count)) > slope:     
                # if first bin is smaller than the next...
                if ref_hist.iloc[i,1] < ref_hist.iloc[i+1, 1]:
                    y = (i*ref_hist.loc[i, sample]  + (3*i+1) * ref_hist.loc[i+1, sample]            ) / (2 * (2*i+1))
                # if first bin is larger than the next...
                else:
                    y = ( (i+2) * ref_hist.loc[i+1, sample]  + 3 *(i) * ref_hist.loc[i, sample] ) / (2 * (2*i+1))
                # correct remaining bin
                x = (ref_hist.loc[i, sample] *i + ref_hist.loc[i+1, sample] *(i+1) - y * (i+1)) / i

                ref_hist.loc[i, sample] = int(x)
                ref_hist.loc[i+1, sample]  = int(y)
                stop_count = 0
            else:
                stop_count = stop_count + 1
            
            # end smoothing for sample if many bin comparisons match target slope
            if (stop_count == 50):
                break
    return(ref_hist)

def parse_reference(reference_path, k, nth, library, skmer_ver, correct_bin_size = True, smooth_spectrum = True):
    '''read either input assembly, histogram, or respect dataframe'''
    sys.stderr.write('[{0}] Parsing repeat spectrum with {1} processors...\n'.format(skmer_ver, nth))
    ext = reference_path.split('.')[-1]
    ref_hist = None
    # read jellyfish histogram
    if (ext == 'hist'):
        ref_hist = pd.read_csv(reference_path, sep = ' ', header = None)
    # read inferred repeat spectrum (respect)
    elif (ext == 'txt'):
        #TODO: verify that input file is correct
        # reads respect tsv: gets sample names, genome lengths, and repeat spectrum
        ref_hist = pd.read_csv(reference_path, sep='\t', header = 0)
        ref_names = pd.Series([x.rsplit('.', 1)[0] for x in ref_hist.pop('sample')])

        ref_lengths = ref_hist.pop('genome_length')
        ref_lengths = pd.concat([ref_lengths], axis = 1).transpose().rename(columns = ref_names)
        #print(ref_lengths)

        ref_hist = ref_hist.transpose()
        ref_hist = ref_hist.rename(columns = ref_names)
        ref_hist = ref_hist.reset_index(drop = True)
        ref_hist = pd.concat([pd.Series([float(x) for x in range(1,ref_hist.shape[0]+1)]), ref_hist], axis = 1)
        #print(ref_hist)
        if correct_bin_size:
        # scales bins to correct genome size
            ref_hist = genome_size_correction(ref_hist, ref_names, ref_lengths)
        if smooth_spectrum:
        # performs spectrum smoothing procedure
            ref_hist = smooth_repeat_spectrum(ref_hist, ref_names)
        
    # Turn input assembly into Jellyfish histogram reference...
    elif (ext[0] == 'f'):
        sample = os.path.basename(reference_path).rsplit('.f', 1)[0]
        mercnt = os.path.join(library, sample + ".jf")
        call(["jellyfish", "count", "-m", str(k), "-s", "100M", "-t", str(nth), "-C", "-o", mercnt, reference_path], stderr=open(os.devnull, 'w'))
        histo_stderr = check_output(["jellyfish", "histo", "-h", "1000000",  mercnt], stderr=STDOUT, universal_newlines=True)
        os.remove(mercnt)
        ref_hist = pd.read_csv(io.StringIO(histo_stderr), sep=' ', header=None)
        hist_file = os.path.basename(reference_path).rsplit('.f', 1)[0] + ".hist"
        #print(hist_file)
        with open(hist_file, "w")  as f:
            f.write(histo_stderr)


    return ref_hist
