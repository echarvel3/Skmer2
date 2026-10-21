import os
import gzip
import shutil
import numpy as np
import pytest
from scipy.stats import poisson

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def pytest_collection_modifyitems(config, items):
    '''skips integration tests when an external tool is missing'''
    missing = [exe for exe in ['jellyfish', 'mash', 'seqkit'] if shutil.which(exe) is None]
    if not missing:
        return
    skip = pytest.mark.skip(reason='missing tools: ' + ', '.join(missing))
    for item in items:
        if 'integration' in item.keywords:
            item.add_marker(skip)

@pytest.fixture
def data_dir():
    return os.path.join(repo_dir, 'data')

def write_fastq(path, reads):
    '''writes a list of (name, sequence) as fastq (gzipped if the path ends in .gz)'''
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(str(path), 'wt') as f:
        for name, seq in reads:
            f.write('@{0}\n{1}\n+\n{2}\n'.format(name, seq, 'I' * len(seq)))
    return str(path)

def write_fasta(path, contigs):
    with open(str(path), 'w') as f:
        for name, seq in contigs:
            f.write('>{0}\n{1}\n'.format(name, seq))
    return str(path)

def random_seq(n, seed=0):
    return ''.join(np.random.RandomState(seed).choice(list('ACGT'), n))

def poisson_hist(genome_size, lam, eps, k):
    '''expected jellyfish count histogram (list indexed by multiplicity, with a leading 0) of a haploid genome with
    unique k-mers sequenced at k-mer coverage lam with error rate eps. Erroneous k-mers are all singletons.'''
    xi = lam * (1 - eps) ** k
    count = [0] + [int(round(genome_size * poisson.pmf(h, xi))) for h in range(1, 60)]
    count[1] += int(round(genome_size * lam * (1 - (1 - eps) ** k)))
    return count

def hist_text(count):
    '''jellyfish histo output for a count list'''
    return ''.join('{0} {1}\n'.format(h, c) for h, c in enumerate(count) if h > 0)

def write_hist(lib, sample, count):
    sample_dir = os.path.join(str(lib), sample)
    os.makedirs(sample_dir, exist_ok=True)
    with open(os.path.join(sample_dir, sample + '.hist'), 'w') as f:
        f.write(hist_text(count))

class FakePool(object):
    '''runs apply_async synchronously in-process so that mocks work'''
    def __init__(self, *args, **kwargs):
        pass
    def apply_async(self, fxn, args=(), kwds=None):
        class Result(object):
            def __init__(self, value):
                self.value = value
            def get(self, timeout=None):
                return self.value
        return Result(fxn(*args, **(kwds or {})))
    def close(self):
        pass
    def join(self):
        pass

def diploid_hist(genome_size, coverage, theta, k):
    '''expected jellyfish count histogram (list indexed by multiplicity) of an error-free diploid genome, where coverage is the
    k-mer coverage of each haplotype and theta is the heterozygosity. Homozygous k-mers (a fraction (1 - theta) ** k) are present on
    both haplotypes, and heterozygous k-mers on one.'''
    Q = (1 - theta) ** k
    return [0] + [int(round(genome_size * Q * poisson.pmf(h, 2 * coverage) + 2 * genome_size * (1 - Q) * poisson.pmf(h, coverage))) for h in range(1, 80)]
