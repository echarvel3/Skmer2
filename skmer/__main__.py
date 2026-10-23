#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import multiprocessing as mp

from skmer import __version__

# NOTE: command modules import pandas, numpy, and scipy, which are slow to load. They are imported
# when a command runs, so that "skmer --help" and argument errors are fast.
def reference(args):
    from skmer.skmer_functions import reference as run
    return run(args)

def distance(args):
    from skmer.skmer_functions import distance as run
    return run(args)

def query(args):
    from skmer.skmer_functions import query as run
    return run(args)

def subsample(args):
    from skmer.subsample import subsample as run
    return run(args)

def correction(args):
    from skmer.subsample import correction as run
    return run(args)

def pop(args):
    from skmer.popgen import pop as run
    return run(args)

def selftest(args):
    from skmer.selftest import selftest as run
    return run(args)

def main():
    # Arguments parser
    parser = argparse.ArgumentParser(description='{0} - Estimating genomic distances between '.format(__version__) +
                                                 'genome-skims',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='skmer {0}'.format(__version__), help='print the current version')
    parser.add_argument('--debug', action='store_true', help='Print the traceback when an exception is raised')
    subparsers = parser.add_subparsers(title='commands',
                                       description='reference   Process a library of reference genome-skims or assemblies\n'
                                                   'distance    Compute pairwise distances for a processed library\n'
                                                   'query       Compare a genome-skim or assembly against a reference library\n'
                                                   'subsample   Performs  subsample on a library of reference genome-skims or assemblies\n'
                                                   'correct     Performs correction of subsampled distance matrices obtained for reference' 
                                                   ' genome-skims or assemblies\n'
                                                   'pop         Calculates theta, Dxy, and Fst for populations in a distance matrix\n'
                                                   'test        Checks the installation and runs every command on the bundled test data'
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
                            help='Path to reference genome, histogram, or repeat spectra data. Runs ReSkmer equations for repeat-aware distances. '
                                 'Use "assemblies" if all inputs are assemblies to use their own repeat spectra')
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
                            help='Path to reference genome, histogram, or repeat spectra data (or "assemblies" if all inputs are assemblies)')
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
                            help='Path to reference genome, histogram, or repeat spectra data (or "assemblies" if all inputs are assemblies)')
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
                            help='Keep subsampled genome-skims (by default they are deleted). Default: false')
    parser_bt.add_argument('-msh', action='store_false',
                            help='Keep sketches (by default they are deleted). Default: false')
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
    parser_bt.add_argument('-chunk', type=int, choices=list(range(1, 33)), default=10, metavar='N',
                            help='Number of replicates sampled together with one pass over each input file [1-32]. Larger values read '
                                 'the inputs fewer times but keep more subsampled files on disk at once. Default: 10')
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
                            help='Path to reference genome, histogram, or repeat spectra data (or "assemblies" if all inputs are assemblies)')
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
                            help='Prefix of output files (<prefix>-thetas.txt, -Dxy.txt, -NeiFst.txt, -SlatkinFst.txt, '
                                 '-WeirCockerhamFst.txt). Default: distance matrix path without extension')
    parser_fst.set_defaults(func=pop)

    # =================================
    # 'test' command subparser
    # Description:
    #   Checks dependencies and runs each
    #   command on the bundled test data.
    # =================================
    parser_test = subparsers.add_parser('test', description='Checks the installation and runs every command on the bundled test data')
    parser_test.add_argument('--quick', action='store_true',
                             help='Only check dependencies and run the unit tests (skips running commands on test data)')
    parser_test.add_argument('--keep', action='store_true',
                             help='Keep the temporary directory of test outputs')
    parser_test.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=min(2, mp.cpu_count()),
                             help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                  'Default: {0}'.format(min(2, mp.cpu_count())), metavar='P')
    parser_test.set_defaults(func=selftest)

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
