import os
import numpy as np
import pandas as pd
import pytest
from unittest import mock

from skmer.reskmer import parse_spectrum, repeat_parameters, repeat_distance
from skmer.reskmer import parse_reference, build_assembly_spectra, estimate_cov_with_ref, estimate_reskmer_dist
from skmer.config import *
from conftest import poisson_hist, write_hist, hist_text

def spectrum(counts, name='s'):
    '''repeat spectrum dataframe: multiplicity in column 0, k-mer counts in column <name>'''
    return pd.DataFrame({0: np.arange(1, len(counts) + 1, dtype=float), name: counts})

class TestGenomeSizeCorrection(object):
    def test_upscales_repeat_bins_to_genome_length(self):
        ref_hist = spectrum([1000, 100, 10, 0, 0])
        lengths = pd.DataFrame({'s': [5000]}, index=['genome_length'])
        out = parse_spectrum.genome_size_correction(ref_hist, ['s'], lengths)
        assert out['s'].iloc[0] == 1000
        assert np.dot(out[0], out['s']) == pytest.approx(5000, abs=10)

    def test_downscales_when_unique_bin_exceeds_genome_length(self):
        ref_hist = spectrum([1000, 100, 10])
        lengths = pd.DataFrame({'s': [500]}, index=['genome_length'])
        ksum = np.dot(ref_hist[0], ref_hist['s'])
        out = parse_spectrum.genome_size_correction(ref_hist, ['s'], lengths)
        assert out['s'].iloc[0] == int(1000 * 500 / ksum)

    def test_spectrum_without_repeats_is_unchanged(self):
        ref_hist = spectrum([1000, 0, 0])
        lengths = pd.DataFrame({'s': [1500]}, index=['genome_length'])
        out = parse_spectrum.genome_size_correction(ref_hist, ['s'], lengths)
        assert out['s'].tolist() == [1000, 0, 0]

class TestSmoothRepeatSpectrum(object):
    def test_smooth_spectrum_is_unchanged(self):
        counts = [int(1e9 * 0.8 ** i) + 1 for i in range(60)]
        ref_hist = spectrum(counts)
        np.random.seed(0)
        out = parse_spectrum.smooth_repeat_spectrum(ref_hist.copy(), ['s'])
        assert out['s'].tolist() == counts

    def test_spike_is_smoothed_and_kmers_conserved(self):
        counts = [int(1e9 * 0.8 ** i) + 1 for i in range(60)]
        counts[30] *= 50
        ref_hist = spectrum(counts)
        np.random.seed(0)
        out = parse_spectrum.smooth_repeat_spectrum(ref_hist.copy(), ['s'])
        assert out['s'].tolist() != counts
        # k-mers are conserved when bins are weighted by their position in the spectrum
        weights = np.arange(60, dtype=float)
        assert np.dot(weights, out['s'].astype(float)) == pytest.approx(np.dot(weights, np.array(counts, dtype=float)), rel=1e-4)

class TestBuildAssemblySpectra(object):
    def test_combines_histograms(self, tmp_path):
        write_hist(tmp_path, 'a', [0, 100, 20, 0, 0])
        write_hist(tmp_path, 'b', [0, 200, 0, 5, 0])
        out = build_assembly_spectra({'a': str(tmp_path), 'b': str(tmp_path)})
        assert out.columns.tolist() == [0, 'a', 'b']
        assert out[0].tolist() == [1.0, 2.0, 3.0]
        assert out['a'].tolist() == [100, 20, 0]
        assert out['b'].tolist() == [200, 0, 5]

    def test_single_assembly_has_two_columns(self, tmp_path):
        write_hist(tmp_path, 'a', [0, 100, 20])
        assert build_assembly_spectra({'a': str(tmp_path)}).shape == (2, 2)

    def test_trailing_empty_bins_are_trimmed(self, tmp_path):
        write_hist(tmp_path, 'a', [0, 100, 20] + [0] * 1000)
        assert build_assembly_spectra({'a': str(tmp_path)}).shape[0] == 2

    def test_different_library_directories(self, tmp_path):
        (tmp_path / 'l1').mkdir()
        (tmp_path / 'l2').mkdir()
        write_hist(tmp_path / 'l1', 'a', [0, 10])
        write_hist(tmp_path / 'l2', 'b', [0, 20])
        out = build_assembly_spectra({'a': str(tmp_path / 'l1'), 'b': str(tmp_path / 'l2')})
        assert out[['a', 'b']].values.tolist() == [[10, 20]]

class TestParseReference(object):
    def test_assemblies_keyword_returns_none(self):
        assert parse_reference('assemblies', 31, 1, '.', 'reskmer') is None

    def test_jellyfish_histogram(self, tmp_path):
        f = tmp_path / 'g.hist'
        f.write_text('1 100\n2 20\n')
        out = parse_reference(str(f), 31, 1, str(tmp_path), 'reskmer')
        assert out.values.tolist() == [[1, 100], [2, 20]]

    def make_respect(self, tmp_path):
        f = tmp_path / 'spectra.txt'
        cols = ['sample'] + ['r{0}'.format(i) for i in range(1, 51)] + ['genome_length']
        rows = [['s1.fastq', 900000] + [1000] * 49 + [1000000], ['s2.fastq', 800] + [0] * 49 + [1000]]
        f.write_text('\t'.join(cols) + '\n' + '\n'.join('\t'.join(map(str, r)) for r in rows) + '\n')
        return str(f)

    def test_respect_spectra(self, tmp_path):
        out = parse_reference(self.make_respect(tmp_path), 31, 1, str(tmp_path), 'reskmer', correct_bin_size=False, smooth_spectrum=False)
        assert out.columns.tolist() == [0, 's1', 's2']
        assert out.shape == (50, 3)
        assert out[0].iloc[0] == 1.0 and out[0].iloc[-1] == 50.0
        assert out['s1'].iloc[0] == 900000

    def test_respect_spectra_are_corrected_to_genome_length(self, tmp_path):
        np.random.seed(0)
        out = parse_reference(self.make_respect(tmp_path), 31, 1, str(tmp_path), 'reskmer', correct_bin_size=True, smooth_spectrum=False)
        assert np.dot(out[0], out['s1'].astype(float)) == pytest.approx(1000000, abs=2000)
        # a spectrum without repeats is not rescaled
        assert out['s2'].iloc[0] == 800

    @pytest.mark.xfail(strict=True, reason='known issue: gzipped assemblies are silently ignored (extension is "gz")')
    def test_gzipped_assembly(self, tmp_path):
        asm = tmp_path / 'genome.fna.gz'
        asm.write_text('x')
        assert parse_reference(str(asm), 21, 1, str(tmp_path), 'reskmer') is not None

    def test_assembly_is_counted_with_jellyfish(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        asm = tmp_path / 'genome.fna'
        asm.write_text('>x\nACGT\n')
        def fake_call(cmd, **kwargs):
            open(cmd[cmd.index('-o') + 1], 'w').close()
        with mock.patch('skmer.reskmer.parse_spectrum.call', side_effect=fake_call) as call, \
                mock.patch('skmer.reskmer.parse_spectrum.check_output', return_value='1 50\n2 5\n'):
            out = parse_reference(str(asm), 21, 3, str(tmp_path), 'reskmer')
        assert out.values.tolist() == [[1, 50], [2, 5]]
        assert call.call_args[0][0][:8] == ['jellyfish', 'count', '-m', '21', '-s', '100M', '-t', '3']
        assert (tmp_path / 'genome.hist').read_text() == '1 50\n2 5\n'
        assert not (tmp_path / 'genome.jf').exists()

class TestEstimOh(object):
    ref = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [1000000, 0, 0]})

    def test_error_is_minimal_at_true_xi(self):
        count = poisson_hist(1000000, 10.0, 0.0, 31)
        hr = range(5, 15)
        at_truth = repeat_parameters.estim_oh(10.0, hr, count, self.ref, 3)
        assert at_truth < 0.5
        for xi in [7.0, 13.0]:
            assert repeat_parameters.estim_oh(xi, hr, count, self.ref, 3) > 100 * max(at_truth, 0.01)

    def test_failure_returns_default_error_rate(self):
        assert repeat_parameters.estim_oh(10.0, range(5, 15), [0, 1], self.ref, 3) == default_error_rate

    def test_xi_function_wraps_estim_oh(self):
        count = poisson_hist(1000000, 10.0, 0.0, 31)
        fxn = repeat_parameters.xi_function(range(5, 15), count, self.ref, 3)
        assert fxn(9.0) == repeat_parameters.estim_oh(9.0, range(5, 15), count, self.ref, 3)

class TestEstimateCovWithRef(object):
    def reads(self, lam=3.0, eps=0.005, k=31, G=1000000):
        count = poisson_hist(G, lam, eps, k)
        ksum = sum(h * c for h, c in enumerate(count))
        ref = spectrum([G] + [0] * 49)
        return count, ksum, ref

    def test_recovers_error_rate_and_coverage(self):
        count, ksum, ref = self.reads()
        eps, lam = estimate_cov_with_ref('s', ref, ksum, count, 31, None, 150)
        assert eps == pytest.approx(0.005, rel=0.5)
        assert lam == pytest.approx(3.0, rel=0.02)

    def test_given_error_rate_overrides_estimate(self):
        count, ksum, ref = self.reads()
        eps, _ = estimate_cov_with_ref('s', ref, ksum, count, 31, 0.002, 150)
        assert eps == 0.002

    def test_selects_sample_from_multi_sample_spectra(self):
        count, ksum, ref = self.reads()
        multi = ref.copy()
        multi['other'] = [5] * 50
        eps, lam = estimate_cov_with_ref('s', multi, ksum, count, 31, None, 150)
        assert lam == pytest.approx(3.0, rel=0.02)

    def test_sample_missing_from_spectra_returns_sentinel(self, capsys):
        count, ksum, ref = self.reads()
        multi = ref.rename(columns={'s': 'x'})
        multi['y'] = 1
        assert estimate_cov_with_ref('s', multi, ksum, count, 31, None, 150) == (-1, 1)
        assert 'not found' in capsys.readouterr().out

class TestEstimateIntersection(object):
    ref = pd.DataFrame({0: [1.0], 1: [100000]})

    def test_assemblies_without_repeats(self):
        # exp|AuB| = G (1 - d) ** k
        for d in [0.0, 0.01, 0.1]:
            out = repeat_distance.estimate_intersection(self.ref, 'NA', 'NA', None, None, 1.0, 1.0, d, 31)
            assert out == pytest.approx(100000 * (1 - d) ** 31)

    def test_intersection_decreases_with_distance(self):
        vals = [repeat_distance.estimate_intersection(self.ref, 5.0, 5.0, 0.01, 0.01, 0.99, 0.99, d, 31) for d in [0.0, 0.01, 0.05, 0.2]]
        assert vals == sorted(vals, reverse=True)

    def test_repeated_kmers_are_more_likely_to_be_shared(self):
        # a k-mer present in j copies is shared unless all j copies are mutated: 1 - (1 - (1 - d) ** k) ** j
        d, k = 0.05, 31
        for j in [2, 3]:
            repeats = pd.DataFrame({0: [1.0, float(j)], 1: [0, 1000]})
            out = repeat_distance.estimate_intersection(repeats, 'NA', 'NA', None, None, 1.0, 1.0, d, k)
            assert out == pytest.approx(1000 * (1 - (1 - (1 - d) ** k) ** j))

class TestIntersectionFunction(object):
    ref = pd.DataFrame({0: [1.0], 1: [100000]})

    def test_root_is_true_distance(self):
        observed = 100000 * (1 - 0.02) ** 31
        g = repeat_distance.intersection_fnctn(self.ref, observed, 'NA', 'NA', None, None, 31)
        assert g(0.02) == pytest.approx(0, abs=1e-6)
        assert g(0.0) > 0 > g(0.05)

    def test_observed_above_expected_is_clipped(self):
        # an observed intersection larger than expected at d = 0 gives d = 0
        g = repeat_distance.intersection_fnctn(self.ref, 100000 * 1.2, 'NA', 'NA', None, None, 31)
        assert g(0.0) == pytest.approx(0)

class TestEstimateReskmerDist(object):
    G = 100000

    def setup_lib(self, tmp_path):
        write_hist(tmp_path, 'a', [0, self.G])
        write_hist(tmp_path, 'b', [0, self.G])
        return str(tmp_path)

    def mash_for(self, d, k=31):
        i = self.G * (1 - d) ** k
        j = i / (2 * self.G - i)
        return mock.patch('skmer.reskmer.repeat_distance.check_output', return_value='a\tb\t0.0\t0\t{0}/{1}\n'.format(int(round(j * 1e6)), 1000000))

    def run(self, lib, d, ref_hist=None, tran=False, k=31):
        ref_hist = spectrum([self.G], 'x') if ref_hist is None else ref_hist
        with self.mash_for(d, k):
            return estimate_reskmer_dist('a', 'b', lib, lib, {'a': 'NA', 'b': 'NA'}, {}, {'a': 0, 'b': 0}, {}, k, 5, tran, ref_hist)

    def test_same_sample_is_zero(self, tmp_path):
        lib = self.setup_lib(tmp_path)
        assert estimate_reskmer_dist('a', 'a', lib, lib, {}, {}, {}, {}, 31, 5, False, spectrum([1], 'x')) == ('a', 'a', 0.0)

    def test_recovers_assembly_distance(self, tmp_path):
        lib = self.setup_lib(tmp_path)
        for d in [0.01, 0.05]:
            assert self.run(lib, d)[2] == pytest.approx(d, abs=2e-3)

    def test_multi_sample_spectra_select_the_first_samples_column(self, tmp_path):
        lib = self.setup_lib(tmp_path)
        multi = pd.DataFrame({0: [1.0, 2.0], 'a': [self.G, 0], 'b': [1, 1]})
        assert self.run(lib, 0.02, multi)[2] == pytest.approx(0.02, abs=2e-3)

    def test_jukes_cantor(self, tmp_path):
        lib = self.setup_lib(tmp_path)
        d = self.run(lib, 0.05)[2]
        assert self.run(lib, 0.05, tran=True)[2] == pytest.approx(-0.75 * np.log(1 - 4 * d / 3), abs=1e-4)

    def test_failure_returns_none(self, tmp_path, capsys):
        # no histograms in this library
        assert estimate_reskmer_dist('a', 'b', str(tmp_path), str(tmp_path), {'a': 'NA', 'b': 'NA'}, {}, {'a': 0, 'b': 0}, {}, 31, 5, False, spectrum([1], 'x')) == ('a', 'b', None)
