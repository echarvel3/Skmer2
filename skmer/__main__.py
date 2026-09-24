#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import random
import argparse
import os
import sys
import multiprocessing as mp

from skmer import __version__
from skmer.skmer_functions import reference, distance, query
from skmer.subsample import subsample, correction
from skmer.utils import pop

def main():
    # Arguments parser
    parser = argparse.ArgumentParser(description='{0} - Estimating genomic distances between '.format(__version__) +
                                                 'genome-skims',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('-v', '--version', action='store_true', help='print the current version')
    parser.add_argument('--debug', action='store_true', help='Print the traceback when an exception is raised')
    subparsers = parser.add_subparsers(title='commands',
                                       description='reference   Process a library of reference genome-skims or assemblies\n'
                                                   'distance    Compute pairwise distances for a processed library\n'
                                                   'query       Compare a genome-skim or assembly against a reference library\n'
                                                   'subsample   Performs  subsample on a library of reference genome-skims or assemblies\n'
                                                   'correct     Performs correction of subsampled distance matrices obtained for reference' 
                                                   ' genome-skims or assemblies'
                                                   ,
                                       help='Run skmer {commands} [-h] for additional help',
                                       dest='{commands}')

    python_version = sys.version_info
    if (python_version[0] * 10 + python_version[1]) >= 33: # to make sure that subcommand is required in python >= 3.3
        subparsers.required = True  

    # =================================
    # 'reference' command subparser
    # Description:
    #   Processes a directory with sequencing files
    #   (fna/fasta for genomes or fastq/fq for skims)
    #   and outputs a Skmer library and distance matrix.
    # =================================
    parser_ref = subparsers.add_parser('reference', description='Process a library of reference genome-skims or assemblies')
    parser_ref.add_argument('input_dir',
                            help='Directory of input genome-skims or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_ref.add_argument('-l', default=os.path.join(os.getcwd(), 'library'),
                            help='Directory of output (reference) library. Default: working_directory/library')
    parser_ref.add_argument('-o', default='ref-dist-mat',
                            help='Output (distances) prefix. Default: ref-dist-mat')
    parser_ref.add_argument('-k', type=int, choices=list(range(1, 32)), default=31, 
                            help='K-mer length [1-31]. ' + 'Default: 31', metavar='K')
    parser_ref.add_argument('-s', type=int, default=10 ** 5, 
                            help='Sketch size. Default: 100000')
    parser_ref.add_argument('-S', type=int, default=42, 
                            help='Sketching random seed. Default: 42')
    parser_ref.add_argument('-e', type=float, 
                            help='Base error rate. By default, the error rate is automatically estimated.')
    parser_ref.add_argument('-t', action='store_true',
                            help='Apply Jukes-Cantor transformation to distances. Output 5.0 if not applicable')
    parser_ref.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                            help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                 'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_ref.add_argument('-r', 
                            help='Path to reference genome, histogram, or repeat spectra data. Runs ReSkmer equations for repeat-aware distances')
    parser_ref.add_argument('-d', action='store_true', 
                            help='Applies DipSkmer equations for diploid distance')
    parser_ref.add_argument('-theta', type=float, 
                            help="uses default theta value to compute diploid coverage")
    parser_ref.set_defaults(func=reference)

    # =================================
    # 'distance' command subparser
    # Description:
    #   computes pairwise distances for samples in 
    #   an existing library.
    # =================================
    parser_dist = subparsers.add_parser('distance', description='Compute the distance matrix for a processed library')
    parser_dist.add_argument('library', 
                            help='Directory of the input (processed) library')
    parser_dist.add_argument('-o', default='ref-dist-mat',
                             help='Output (distances) prefix. Default: ref-dist-mat')
    parser_dist.add_argument('-t', action='store_true',
                             help='Apply Jukes-Cantor transformation to distances. Output 5.0 if not applicable')
    parser_dist.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                             help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                  'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_dist.add_argument('-r', 
                            help='Path to reference genome, histogram, or repeat spectra data')
    parser_dist.add_argument('-d', action='store_true',
                            help='Applies DipSkmer equations for diploid distance')
    parser_dist.set_defaults(func=distance)

    # =================================
    # 'query' command subparser
    # Description:
    #   processes a single genome-skim/assembly and 
    #   computes distances to all samples in an 
    #   existing library.
    # =================================
    parser_qry = subparsers.add_parser('query', description='Compare an input genome-skim or assembly against a reference library')
    parser_qry.add_argument('input', 
                            help='Input (query) genome-skim or assembly (a .fastq/.fq/.fa/.fna/.fasta file)')
    parser_qry.add_argument('library', 
                            help='Directory of (reference) library')
    parser_qry.add_argument('-a', action='store_true',
                            help='Add the processed input (query) to the (reference) library')
    parser_qry.add_argument('-o', default='dist',
                            help='Output (distances) prefix. Default: dist')
    parser_qry.add_argument('-e', type=float, help='Base error rate. By default, the error rate is automatically estimated.')
    parser_qry.add_argument('-t', action='store_true',
                            help='Apply Jukes-Cantor transformation to distances. Output 5.0 if not applicable')
    parser_qry.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                            help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                 'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P') 
    parser_qry.add_argument('-r', 
                            help='Path to reference genome, histogram, or repeat spectra data')
    parser_qry.add_argument('-d', action='store_true', 
                            help='Applies DipSkmer equations for diploid distance')
    parser_qry.add_argument('-theta', type=float, help="uses default theta value to compute diploid coverage")
    parser_qry.set_defaults(func=query)

    # =================================
    # 'subsample' command subparser
    # Description:
    #   generates a distances from a series
    #   of random subsamples of a directory 
    #   of sequencing files.
    # =================================
    parser_bt = subparsers.add_parser('subsample', description='Performs subsample on a library of reference genome-skims or assemblies')
    parser_bt.add_argument('input_dir',
                            help='Directory of input genome-skims or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_bt.add_argument('-sub', default=os.path.join(os.getcwd(), 'subsample'),
                            help='Directory of output for subsample replicates. Default: working_directory/subsample')
    parser_bt.add_argument('-fa', action='store_false',
                            help='Save subsampled genome-skims. Default: false')
    parser_bt.add_argument('-msh', action='store_false',
                            help='Save sketches. Default: false')
    parser_bt.add_argument('-k', type=int, choices=list(range(1, 32)), default=31, 
                            help='K-mer length [1-31]. ' + 'Default: 31', metavar='K')
    parser_bt.add_argument('-s', type=int, default=10 ** 5, 
                            help='Sketch size. Default: 100000')
    parser_bt.add_argument('-S', type=int, default=42, 
                            help='Sketching random seed. Default: 42')
    parser_bt.add_argument('-i', type=int, default=0, 
                            help='Start index of subsampled replicate (eg 5 for dir rep5). Default: 0')
    parser_bt.add_argument('-b', type=int, default=100, 
                            help='Number of subsampled replicates. Default: 100')    
    parser_bt.add_argument('-c', type=float, default=0.9, 
                            help='Exponent value for subsampling. Default: 0.9')
    parser_bt.add_argument('-e', type=float, 
                            help='Base error rate. By default, the error rate is automatically estimated.')
    parser_bt.add_argument('-t', action='store_true',
                            help='Apply Jukes-Cantor transformation to distances. Output 5.0 if not applicable')
    parser_bt.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                            help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                 'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_bt.add_argument('-r', 
                            help='Path to reference genome, histogram, or repeat spectra data')
    parser_bt.add_argument('-d', action='store_true', 
                            help='Applies dipskmer equations for diploid distance')
    parser_bt.add_argument('-theta', type=float, 
                            help='uses default theta value to compute diploid coverage')
    parser_bt.set_defaults(func=subsample)
   
    # =================================
    # 'correct' command subparser
    # Description:
    #   Corrects subsampled distance matrices given a 
    #   distance matrix from a full-coverage library.
    # =================================
    parser_cor = subparsers.add_parser('correct',
                                       description='Performs correction of subsampled distance matrices obtained for reference genome-skims or assemblies')
    parser_cor.add_argument('-main',
                            help='Distance matrix of main estimate')
    parser_cor.add_argument('-sub', default=os.path.join(os.getcwd(), 'subsample'),
                            help='Directory of output for subsample replicates. Default: working_directory/subsample')
    parser_cor.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                            help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                 'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_cor.set_defaults(func=correction)

    # =================================
    # 'pop' command subparser
    # Description:
    #   Used to create an assortment of
    #   pop-gen estimates.
    # =================================
    parser_fst = subparsers.add_parser('pop', description='Calculates pop-gen estimates for subgroups within a distance matrix')
    parser_fst.add_argument('matrix', 
                            help='Path to distance matrix')
    parser_fst.add_argument('annotation', 
                            help='Path to annotation file TSV with the format (sample\tpopulation).')
    parser_fst.add_argument('-o', 
                            help='Path to output matrices.')
    parser_fst.set_defaults(func=pop)

    args = parser.parse_args()

    # Handling traceback on exceptions
    def exception_handler(exception_type, exception, traceback, debug_hook=sys.excepthook):
        if args.debug:
            debug_hook(exception_type, exception, traceback)
        else:
            print("{0}: {1}".format(exception_type.__name__, exception))

    sys.excepthook = exception_handler

    args.func(args)

if __name__ == "__main__":
    main()
