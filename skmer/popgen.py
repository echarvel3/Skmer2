import os
import sys
import errno
import pandas as pd

from skmer.config import *

def read_annotations(annotation_file):
    '''reads a (sample\tpopulation) tsv into a dictionary. An optional header row is ignored.'''
    annots = dict()
    with open(annotation_file) as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 2 or not fields[0].strip():
                continue
            annots[fields[0].strip()] = fields[1].strip()
    # drops header row (eg "sample\tpopulation")
    if 'sample' in annots and annots['sample'] in ('population', 'pop'):
        del annots['sample']
    if not annots:
        raise ValueError('No (sample\tpopulation) rows found in {0}'.format(annotation_file))
    return annots

def read_matrix(matrix_file):
    '''reads a square distance matrix tsv. Entries that are not numbers (eg None) become NaN.'''
    df = pd.read_csv(matrix_file, sep='\t', header=0, index_col=0, dtype=str)
    if df.shape[0] != df.shape[1]:
        raise ValueError('Distance matrix {0} is not square ({1} rows, {2} columns)'.format(matrix_file, df.shape[0], df.shape[1]))
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def assign_populations(samples, annots):
    '''maps each matrix sample to a population (matching either the full name or the name without extension)'''
    pops = dict()
    for sample in samples:
        if sample in annots:
            pops[sample] = annots[sample]
        elif sample.rsplit('.f', 1)[0] in annots:
            pops[sample] = annots[sample.rsplit('.f', 1)[0]]
    if not pops:
        raise ValueError('None of the matrix samples were found in the annotation file')
    return pops

def group_distances(df, pops):
    '''collects pairwise distances within each population and between each pair of populations'''
    pop_names = sorted(set(pops.values()))
    within = {p: [] for p in pop_names}
    between = {(p1, p2): [] for p1 in pop_names for p2 in pop_names if p1 < p2}
    for s1 in pops:
        for s2 in pops:
            if s1 == s2:
                continue
            d = df.loc[s1, s2]
            if pd.isna(d):
                continue
            p1, p2 = pops[s1], pops[s2]
            if p1 == p2:
                within[p1].append(float(d))
            elif p1 < p2:
                between[(p1, p2)].append(float(d))
            else:
                between[(p2, p1)].append(float(d))
    return within, between

def mean(values):
    '''mean of a list, None if empty'''
    return sum(values) / len(values) if len(values) > 0 else None

def theta(within):
    '''within-population diversity (pi_s): mean pairwise distance between samples of one population'''
    return mean(within)

def dxy(between):
    '''between-population divergence (Dxy): mean pairwise distance between samples of two populations'''
    return mean(between)

def _fst_terms(within_1, within_2, between):
    '''returns (average within-pop diversity, Dxy) or None if either is undefined'''
    theta_1, theta_2, d_xy = theta(within_1), theta(within_2), dxy(between)
    if theta_1 is None or theta_2 is None or d_xy is None:
        return None
    return (theta_1 + theta_2) / 2.0, d_xy

def nei_fst(within_1, within_2, between):
    '''Nei / Hudson-Boos-Kaplan Fst: (pi_t - pi_s) / pi_t, where pi_t is the mean of all pairwise distances'''
    terms = _fst_terms(within_1, within_2, between)
    if terms is None:
        return None
    pi_s, _ = terms
    pi_t = mean(within_1 + within_2 + between)
    return (pi_t - pi_s) / pi_t if pi_t != 0 else None

def slatkin_fst(within_1, within_2, between):
    '''Slatkin Fst: pi_d / (2 pi_s + pi_d), where pi_d = Dxy - pi_s'''
    terms = _fst_terms(within_1, within_2, between)
    if terms is None:
        return None
    pi_s, d_xy = terms
    pi_d = d_xy - pi_s
    return pi_d / (2 * pi_s + pi_d) if (2 * pi_s + pi_d) != 0 else None

def wc_fst(within_1, within_2, between):
    '''Weir-Cockerham / Hudson-Slatkin-Madison Fst: pi_d / (pi_s + pi_d) = (Dxy - pi_s) / Dxy'''
    terms = _fst_terms(within_1, within_2, between)
    if terms is None:
        return None
    pi_s, d_xy = terms
    pi_d = d_xy - pi_s
    return pi_d / (pi_s + pi_d) if (pi_s + pi_d) != 0 else None

def pairwise_matrix(pop_names, fxn):
    '''builds a symmetric population-by-population matrix with fxn(pop_1, pop_2); undefined values are "NA"'''
    mat = pd.DataFrame(index=pop_names, columns=pop_names, dtype=object)
    for p1 in pop_names:
        for p2 in pop_names:
            value = fxn(p1, p2)
            mat.loc[p1, p2] = 'NA' if value is None else round(value, 5)
    mat.index.name = 'sample'
    return mat

def calculate_popgen(df, pops):
    '''returns theta per population and the Dxy, Nei, Slatkin, and Weir-Cockerham matrices'''
    within, between = group_distances(df, pops)
    pop_names = sorted(within.keys())

    def get_between(p1, p2):
        return between[(p1, p2)] if p1 < p2 else between[(p2, p1)]

    def fst_fxn(fst):
        # an undefined (NA) fst for a population versus itself, a defined one is 0
        return lambda p1, p2: (0.0 if theta(within[p1]) is not None else None) if p1 == p2 else fst(within[p1], within[p2], get_between(p1, p2))

    thetas = pd.DataFrame({'population': pop_names, 'theta': ['NA' if theta(within[p]) is None else round(theta(within[p]), 5) for p in pop_names]})
    results = {
        'Dxy': pairwise_matrix(pop_names, lambda p1, p2: theta(within[p1]) if p1 == p2 else dxy(get_between(p1, p2))),
        'NeiFst': pairwise_matrix(pop_names, fst_fxn(nei_fst)),
        'SlatkinFst': pairwise_matrix(pop_names, fst_fxn(slatkin_fst)),
        'WeirCockerhamFst': pairwise_matrix(pop_names, fst_fxn(wc_fst)),
    }
    return thetas, results

def write_popgen(thetas, results, out_prefix):
    '''writes thetas and matrices as tab-separated files: <prefix>-<name>.txt'''
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        try:
            os.makedirs(out_dir)
        except OSError as Error:
            if Error.errno != errno.EEXIST:
                raise
    thetas.to_csv(out_prefix + '-thetas.txt', sep='\t', header=False, index=False)
    for name, mat in results.items():
        mat.to_csv(out_prefix + '-' + name + '.txt', sep='\t', mode='w')
    return [out_prefix + '-' + name + '.txt' for name in ['thetas'] + list(results.keys())]

def pop(args):
    '''calculates theta, Dxy, and Fst for populations defined in an annotation file'''
    df = read_matrix(args.matrix)
    annots = read_annotations(args.annotation)
    pops = assign_populations(list(df.index), annots)

    missing = [s for s in df.index if s not in pops]
    if missing:
        sys.stderr.write('[!WARNING!] {0} samples not in the annotation file were ignored: {1}\n'.format(len(missing), ', '.join(missing)))

    out_prefix = args.o if args.o else os.path.splitext(args.matrix)[0]
    sys.stderr.write('[pop] Calculating theta, Dxy, and Fst for {0} populations...\n'.format(len(set(pops.values()))))
    thetas, results = calculate_popgen(df, pops)
    sys.stderr.write('[pop] Writing to file...\n')
    return write_popgen(thetas, results, out_prefix)
