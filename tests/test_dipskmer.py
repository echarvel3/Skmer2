import numpy as np
import pytest
from unittest import mock

from skmer.dipskmer import estimate_diploid_cov, estimate_theta_from_ref, estimate_dipskmer_dist
from skmer.config import *
from conftest import poisson_hist, diploid_hist, write_hist

class TestEstimateThetaFromRef(object):
    @pytest.mark.parametrize('theta', [0.001, 0.005, 0.01])
    def test_recovers_heterozygosity(self, theta):
        count = diploid_hist(1000000, 10.0, theta, 31)
        assert estimate_theta_from_ref(10.0, 0.0, 31, count) == pytest.approx(theta, rel=0.05)

    def test_failure_returns_default_theta(self):
        # histogram too short to find the peak
        assert estimate_theta_from_ref(10.0, 0.01, 31, [0, 5]) == default_theta

class TestEstimateDiploidCov(object):
    def test_low_coverage_uses_defaults(self, capsys):
        # no k-mer is seen twice, so nothing can be co-estimated
        eps, lam, theta = estimate_diploid_cov('s', [0, 1000, 0, 0, 0], 31, None, 150, None)
        assert eps == default_error_rate and theta == default_theta
        assert lam >= 0
        assert 'Not enough information' in capsys.readouterr().err

    def test_low_coverage_uses_given_theta_and_error_rate(self):
        eps, _, theta = estimate_diploid_cov('s', [0, 1000, 0, 0, 0], 31, 0.02, 150, 0.004)
        assert eps == 0.02 and theta == 0.004

    @pytest.mark.parametrize('theta', [0.001, 0.005])
    def test_recovers_theta_and_coverage(self, theta):
        count = diploid_hist(1000000, 10.0, theta, 31)
        eps, lam, est_theta = estimate_diploid_cov('s', count, 31, None, 150, None)
        assert est_theta == pytest.approx(theta, rel=0.05)
        # coverage of both haplotypes
        assert lam == pytest.approx(20.0, rel=0.01)
        assert eps == pytest.approx(0.0, abs=1e-4)

    def test_given_theta_at_high_coverage(self):
        count = poisson_hist(1000000, 20.0, 0.005, 31)
        eps, lam, theta = estimate_diploid_cov('s', count, 31, None, 150, 0.003)
        assert theta == 0.003
        assert 0 < eps < 0.03
        assert np.isfinite(lam) and lam > 0

    def test_given_error_rate_at_high_coverage(self):
        count = poisson_hist(1000000, 20.0, 0.005, 31)
        eps, lam, _ = estimate_diploid_cov('s', count, 31, 0.005, 150, 0.003)
        assert eps == 0.005


class TestEstimateDipskmerDist(object):
    def run(self, tmp_path, j, tran=False, same_lib=True, cov=20.0):
        write_hist(tmp_path, 'a', [0, 1000, 500, 100])
        write_hist(tmp_path, 'b', [0, 1000, 500, 100])
        d = {'a': cov, 'b': cov}
        mash = 'a\tb\t0.0\t0\t{0}/{1}\n'.format(int(j * 100000), 100000)
        with mock.patch('skmer.dipskmer.diploid_distance.check_mash_files', return_value='x.msh'), \
                mock.patch('skmer.dipskmer.diploid_distance.check_output', return_value=mash):
            return estimate_dipskmer_dist('a', 'b', str(tmp_path), str(tmp_path), d, {'a': 1000000, 'b': 1000000},
                                          {'a': 0.005, 'b': 0.005}, {'a': 150, 'b': 150}, 31, dip_coverage_threshold, tran,
                                          {'a': 0.005, 'b': 0.005})

    def test_same_sample_is_zero(self):
        assert estimate_dipskmer_dist('a', 'a', 'l', 'l', {}, {}, {}, {}, 31, 2, False, {}) == ('a', 'a', 0.0)

    def test_distance_decreases_with_jaccard(self, tmp_path):
        ds = [self.run(tmp_path, j)[2] for j in [0.5, 0.6, 0.7, 0.8, 0.9]]
        assert all(0 <= d < 1 for d in ds)
        assert ds == sorted(ds, reverse=True)

    def test_distance_is_not_negative(self, tmp_path):
        assert self.run(tmp_path, 0.99)[2] >= 0

    def test_jukes_cantor_transformation(self, tmp_path):
        d = self.run(tmp_path, 0.5)[2]
        assert self.run(tmp_path, 0.5, tran=True)[2] == pytest.approx(-0.75 * np.log(1 - 4 * d / 3), abs=1e-5)

    def test_missing_sketch_raises(self, tmp_path):
        write_hist(tmp_path, 'a', [0, 10])
        write_hist(tmp_path, 'b', [0, 10])
        with mock.patch('skmer.dipskmer.diploid_distance.check_mash_files', return_value=None):
            with pytest.raises(FileNotFoundError):
                estimate_dipskmer_dist('a', 'b', str(tmp_path), str(tmp_path), {'a': 10.0, 'b': 10.0}, {'a': 1000, 'b': 1000},
                                       {'a': 0.01, 'b': 0.01}, {'a': 150, 'b': 150}, 31, 2, False, {'a': 0.005, 'b': 0.005})

    @pytest.mark.xfail(strict=True, raises=TypeError, reason='known issue: assemblies have read length "NA", which diploid distances subtract from k')
    def test_assemblies(self, tmp_path):
        write_hist(tmp_path, 'a', [0, 1000])
        write_hist(tmp_path, 'b', [0, 1000])
        with mock.patch('skmer.dipskmer.diploid_distance.check_mash_files', return_value='x.msh'), \
                mock.patch('skmer.dipskmer.diploid_distance.check_output', return_value='a\tb\t0\t0\t50000/100000\n'):
            estimate_dipskmer_dist('a', 'b', str(tmp_path), str(tmp_path), {'a': 'NA', 'b': 'NA'}, {'a': 5000, 'b': 5000}, {'a': 0, 'b': 0},
                                   {'a': 'NA', 'b': 'NA'}, 31, 2, False, {'a': .005, 'b': .005})
