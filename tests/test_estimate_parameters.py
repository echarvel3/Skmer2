import os
import pytest
from unittest import mock

from skmer import estimate_parameters
from skmer.config import *
from conftest import poisson_hist, hist_text

def estimate(tmp_path, count, stat, skmer_ver='skmer', **kwargs):
    '''runs estimate_cov with a mocked k-mer histogram and read statistics'''
    with mock.patch('skmer.estimate_parameters.count_kmers', return_value=hist_text(count)), \
            mock.patch('skmer.estimate_parameters.sequence_stat', return_value=stat):
        return estimate_parameters.estimate_cov('/data/s1.fastq', str(tmp_path), 31, kwargs.get('e'), 1, skmer_ver,
                                                kwargs.get('ref_hist'), kwargs.get('theta'))

def read_dat(tmp_path):
    return dict(line.split('\t') for line in (tmp_path / 's1' / 's1.dat').read_text().strip().split('\n'))

class TestEstimateCov(object):
    def test_assembly(self, tmp_path):
        count = poisson_hist(1000, 1.0, 0.0, 31)
        out = estimate(tmp_path, count, (5000, 5000, 5000, 1))
        assert out == ('s1', 'NA', 5000, 0, 'NA', None)
        assert read_dat(tmp_path) == {'coverage': 'NA', 'genome_length': '5000', 'error_rate': '0.0', 'read_length': 'NA'}

    def test_diploid_assembly_has_na_theta(self, tmp_path):
        out = estimate(tmp_path, poisson_hist(1000, 1.0, 0.0, 31), (5000, 5000, 5000, 1), skmer_ver='dipskmer')
        assert out[-1] == 'NA'
        assert read_dat(tmp_path)['theta'] == 'NA'

    def test_creates_sample_directory(self, tmp_path):
        estimate(tmp_path, poisson_hist(1000, 1.0, 0.0, 31), (5000, 5000, 5000, 1))
        assert os.path.isdir(str(tmp_path / 's1'))

    def test_skmer_reads(self, tmp_path):
        G, cov, l = 1000000, 15.0, 150
        lam = cov * (l - 31) / l
        count = poisson_hist(G, lam, 0.005, 31)
        ksum = sum(h * c for h, c in enumerate(count))
        out = estimate(tmp_path, count, (l, l, int(G * cov), int(G * cov / l)))
        name, coverage, g_len, eps, read_length, theta = out
        assert name == 's1' and read_length == l and theta is None
        assert coverage == pytest.approx(cov, rel=0.1)
        assert g_len == pytest.approx(G, rel=0.1)
        assert eps == pytest.approx(0.005, rel=0.3)
        assert read_dat(tmp_path)['read_length'] == '150'

    def test_too_low_coverage(self, tmp_path, capsys):
        out = estimate(tmp_path, [0, 100], (150, 150, 15000, 100))
        assert out[:5] == ('s1', 'NA', 'NA', 'NA', 150)
        assert 'too low' in capsys.readouterr().err
        assert read_dat(tmp_path)['coverage'] == 'NA'

    def test_error_rate_above_threshold_is_na(self, tmp_path):
        G, l = 1000000, 150
        count = poisson_hist(G, 15.0 * 119 / 150, 0.005, 31)
        out = estimate(tmp_path, count, (l, l, G * 15, G // 10), e=0.05)
        assert out[1:4] == ('NA', 'NA', 'NA')

    def test_diploid_returns_theta(self, tmp_path):
        G, l = 1000000, 150
        count = poisson_hist(G, 20.0 * 119 / 150, 0.005, 31)
        out = estimate(tmp_path, count, (l, l, G * 20, G // 7), skmer_ver='dipskmer', theta=0.003)
        assert out[-1] == 0.003
        assert read_dat(tmp_path)['theta'] == '0.003'

    def test_reskmer_reads(self, tmp_path):
        import pandas as pd
        G, cov, l = 1000000, 3.0, 150
        lam = cov * (l - 31) / l
        count = poisson_hist(G, lam, 0.005, 31)
        ref = pd.DataFrame({0: [float(i) for i in range(1, 51)], 'x': [G] + [0] * 49})
        out = estimate(tmp_path, count, (l, l, int(G * cov), int(G * cov / l)), skmer_ver='reskmer', ref_hist=ref)
        assert out[1] == pytest.approx(cov, rel=0.05)
        assert 0 < out[3] < 0.02

    def test_error_rate_above_threshold_gives_na_theta_for_reskmer_dipskmer(self, tmp_path):
        import pandas as pd
        G, l = 1000000, 150
        count = poisson_hist(G, 3.0 * 119 / 150, 0.005, 31)
        ref = pd.DataFrame({0: [float(i) for i in range(1, 51)], 'x': [G] + [0] * 49})
        out = estimate(tmp_path, count, (l, l, G * 3, G // 50), skmer_ver='reskmer + dipskmer', ref_hist=ref, e=0.05, theta=0.003)
        assert out[1:4] == ('NA', 'NA', 'NA') and out[-1] == 'NA'
