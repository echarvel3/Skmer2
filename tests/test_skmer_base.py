import math
import numpy as np
import pytest
from unittest import mock

from skmer.skmer import estimate_base_cov, estimate_skmer_dist
from skmer.skmer.base_distance import dist_temp_func
from skmer.config import *
from conftest import poisson_hist

class TestEstimateBaseCov(object):
    def test_recovers_error_rate_and_coverage(self):
        count = poisson_hist(1000000, 12.0, 0.005, 31)
        eps, lam = estimate_base_cov('s', count, 31, None, 150)
        assert eps == pytest.approx(0.005, rel=0.2)
        assert lam == pytest.approx(12.0, rel=0.1)

    def test_given_error_rate_is_used(self):
        count = poisson_hist(1000000, 12.0, 0.005, 31)
        eps, lam = estimate_base_cov('s', count, 31, 0.005, 150)
        assert eps == 0.005
        assert lam == pytest.approx(12.0, rel=0.1)

    def test_low_coverage_uses_default_error_rate(self, capsys):
        # no k-mer is seen twice, so the error rate cannot be estimated
        eps, lam = estimate_base_cov('s', [0, 1000, 0, 0, 0], 31, None, 150)
        assert eps == default_error_rate
        assert 'Not enough information' in capsys.readouterr().err
        assert lam >= 0

class TestDistTempFunc(object):
    def test_assembly(self):
        assert dist_temp_func('NA', 0, 31, 'NA', 5) == [1.0, 0]

    def test_low_coverage_single_copy(self):
        cov, eps, k, l = 2.0, 0.01, 31, 150
        p = np.exp(-k * eps)
        lam = cov * (l - k) / l
        eta, psi = dist_temp_func(cov, eps, k, l, 5)
        assert eta == pytest.approx(1 - np.exp(-lam * p))
        assert psi == pytest.approx(lam * (1 - p))

    def test_no_errors(self):
        eta, psi = dist_temp_func(2.0, 0.0, 31, 150, 5)
        assert psi == 0
        assert eta == pytest.approx(1 - np.exp(-2.0 * 119 / 150))

    def test_high_coverage_uses_copy_threshold(self):
        cov, eps, k, l = 20.0, 0.01, 31, 150
        p = np.exp(-k * eps)
        lam = cov * (l - k) / l
        eta, psi = dist_temp_func(cov, eps, k, l, 5)
        # copy threshold = int(20 / 5) + 1 = 5
        expected = 1 - np.exp(-lam * p) * sum((lam * p) ** i / math.factorial(i) for i in range(5))
        assert eta == pytest.approx(expected)
        assert psi == 0

    def test_eta_increases_with_coverage(self):
        etas = [dist_temp_func(c, 0.0, 31, 150, 5)[0] for c in [0.5, 1, 2, 4]]
        assert etas == sorted(etas)

class TestEstimateSkmerDist(object):
    def mash(self, j, n=100000):
        return mock.patch('skmer.skmer.base_distance.check_output', return_value='a\tb\t0.0\t0\t{0}/{1}\n'.format(int(j * n), n))

    def dist(self, j, tran=False, **kwargs):
        args = dict(ce={'a': 'NA', 'b': 'NA'}, le={'a': 1000, 'b': 1000}, ee={'a': 0, 'b': 0}, rl={'a': 'NA', 'b': 'NA'})
        args.update(kwargs)
        with self.mash(j):
            return estimate_skmer_dist('a', 'b', 'lib', 'lib', args['ce'], args['le'], args['ee'], args['rl'], 31, 5, tran)

    def test_same_sample_is_zero(self):
        assert estimate_skmer_dist('a', 'a', 'lib', 'lib', {}, {}, {}, {}, 31, 5, False) == ('a', 'a', 0.0)

    def test_identical_assemblies_are_distance_zero(self):
        assert self.dist(1.0)[2] == 0.0

    def test_assembly_distance_matches_formula(self):
        # for assemblies d = 1 - (2j / (1 + j)) ** (1 / k)
        j = 0.5
        assert self.dist(j)[2] == pytest.approx(round(1 - (2 * j / (1 + j)) ** (1 / 31), 5))

    def test_distance_decreases_with_jaccard(self):
        ds = [self.dist(j)[2] for j in [0.2, 0.4, 0.6, 0.8]]
        assert ds == sorted(ds, reverse=True)

    def test_jukes_cantor_transformation(self):
        d = self.dist(0.5)[2]
        assert self.dist(0.5, tran=True)[2] == pytest.approx(-0.75 * np.log(1 - 4 * d / 3))

    def test_reads_with_sequencing_parameters(self):
        reads = dict(ce={'a': 8.0, 'b': 8.0}, le={'a': 1000, 'b': 1000}, ee={'a': 0.01, 'b': 0.01}, rl={'a': 150, 'b': 150})
        d_reads = self.dist(0.3, **reads)[2]
        d_assemblies = self.dist(0.3)[2]
        assert d_reads != d_assemblies and 0 < d_reads < 1
