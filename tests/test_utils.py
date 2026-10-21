import os
import argparse
import numpy as np
import pytest
from unittest import mock

from skmer import utils
from skmer.config import *
from conftest import write_fastq, write_fasta, random_seq, write_hist

class TestSequenceStat(object):
    def test_fastq(self, tmp_path):
        f = write_fastq(tmp_path / 'r.fastq', [('a', 'A' * 10), ('b', 'C' * 20), ('c', 'G' * 30)])
        assert utils.sequence_stat(f) == (20, 30, 60, 3)

    def test_fasta(self, tmp_path):
        f = write_fasta(tmp_path / 'a.fna', [('c1', 'ACGT' * 5), ('c2', 'ACGT' * 10)])
        assert utils.sequence_stat(f) == (30, 40, 60, 2)

    def test_gzipped(self, tmp_path):
        f = write_fastq(tmp_path / 'r.fastq.gz', [('a', 'ACGTACGT')])
        assert utils.sequence_stat(f) == (8, 8, 8, 1)

    def test_only_counts_acgt(self, tmp_path):
        f = write_fastq(tmp_path / 'r.fastq', [('a', 'ACGTNNNNacgt')])
        assert utils.sequence_stat(f) == (8, 8, 8, 1)

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / 'empty.fasta'
        f.write_text('')
        with pytest.raises(ValueError, match='not a valid'):
            utils.sequence_stat(str(f))

class TestCheckMashFiles(object):
    def fake_run(self, threshold):
        return mock.Mock(stdout='header\nsketch\t{0}\n'.format(threshold))

    def test_matching_threshold_returns_sketch(self, tmp_path):
        (tmp_path / 's.msh').write_text('x')
        with mock.patch('skmer.utils.run', return_value=self.fake_run(3)) as run:
            assert utils.check_mash_files(str(tmp_path / 's'), 3) == str(tmp_path / 's') + '.msh'
        assert run.call_args[0][0] == ['mash', 'info', '-c', str(tmp_path / 's') + '.msh']

    def test_different_threshold_returns_none(self, tmp_path):
        (tmp_path / 's.msh').write_text('x')
        with mock.patch('skmer.utils.run', return_value=self.fake_run(2)):
            assert utils.check_mash_files(str(tmp_path / 's'), 3) is None

    def test_missing_sketch_returns_none(self, tmp_path):
        with mock.patch('skmer.utils.run') as run:
            assert utils.check_mash_files(str(tmp_path / 'nope'), 1) is None
        run.assert_not_called()

class TestSketch(object):
    def sketch(self, tmp_path, cov, eps, has_spectrum=False, is_diploid=False, r_len=None, cov_thres=5, existing=None):
        with mock.patch('skmer.utils.call') as call, mock.patch('skmer.utils.check_mash_files', return_value=existing):
            utils.sketch('/data/s1.fastq', str(tmp_path), {'s1': cov}, {'s1': eps}, 21, 1000, cov_thres, 7, has_spectrum, is_diploid,
                         {'s1': r_len} if r_len else None)
        return call.call_args[0][0] if call.called else None

    def test_assembly_sketches_without_read_flag(self, tmp_path):
        cmd = self.sketch(tmp_path, 'NA', 0)
        assert cmd == ['mash', 'sketch', '-k', '21', '-s', '1000', '-S', '7', '-o', str(tmp_path / 's1' / 's1'), '/data/s1.fastq']

    def test_unknown_error_rate_sketches_reads(self, tmp_path):
        cmd = self.sketch(tmp_path, 'NA', 'NA')
        assert '-r' in cmd and '-m' not in cmd

    def test_low_coverage_reads_use_read_flag(self, tmp_path):
        cmd = self.sketch(tmp_path, 2.0, 0.01)
        assert '-r' in cmd and '-m' not in cmd

    def test_high_coverage_skmer_uses_copy_threshold(self, tmp_path):
        cmd = self.sketch(tmp_path, 20.0, 0.01)
        # copy threshold = int(20 / 5) + 1
        assert cmd[:3] == ['mash', 'sketch', '-m'] and cmd[3] == '5'
        assert '-r' not in cmd

    def test_reskmer_always_uses_read_flag(self, tmp_path):
        cmd = self.sketch(tmp_path, 20.0, 0.01, has_spectrum=True)
        assert '-r' in cmd and '-m' not in cmd

    def test_zero_error_rate_uses_read_flag(self, tmp_path):
        cmd = self.sketch(tmp_path, 20.0, 0.0)
        assert '-r' in cmd

    def test_diploid_sketch_name_has_copy_threshold(self, tmp_path):
        cmd = self.sketch(tmp_path, 20.0, 0.01, is_diploid=True, r_len=100, cov_thres=dip_coverage_threshold)
        assert cmd[cmd.index('-o') + 1].endswith('s1t') or 't' in os.path.basename(cmd[cmd.index('-o') + 1])
        assert os.path.basename(cmd[cmd.index('-o') + 1]).startswith('s1t')

    def test_existing_sketch_is_not_recomputed(self, tmp_path):
        assert self.sketch(tmp_path, 2.0, 0.01, r_len=100, existing='x.msh') is None

class TestGetSamplesFromFiles(object):
    def test_lists_supported_formats(self, tmp_path):
        for name in ['a.fastq', 'b.fna', 'c.fq.gz', 'notes.txt']:
            (tmp_path / name).write_text('x')
        files, samples = utils.get_samples_from_files(argparse.Namespace(input_dir=str(tmp_path)))
        assert sorted(files) == ['a.fastq', 'b.fna', 'c.fq.gz']
        assert sorted(samples) == ['a', 'b', 'c']

    def test_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            utils.get_samples_from_files(argparse.Namespace(input_dir=str(tmp_path)))

    def test_duplicate_sample_names_raise(self, tmp_path):
        (tmp_path / 'a.fastq').write_text('x')
        (tmp_path / 'a.fna').write_text('x')
        with pytest.raises(ValueError):
            utils.get_samples_from_files(argparse.Namespace(input_dir=str(tmp_path)))

def test_write_config_file(tmp_path):
    utils.write_config_file(argparse.Namespace(l=str(tmp_path), k=21, s=500, S=3))
    assert (tmp_path / 'CONFIG').read_text() == 'kmer_length\t21\nsketch_size\t500\nsketching_seed\t3\n'

@pytest.mark.parametrize('r, d, label', [(None, False, 'skmer'), ('x.hist', False, 'reskmer'), (None, True, 'dipskmer'),
                                         ('x.hist', True, 'reskmer + dipskmer'), ('assemblies', False, 'reskmer')])
def test_assign_skmer_label(r, d, label):
    assert utils.assign_skmer_label(argparse.Namespace(r=r, d=d)) == label

class TestWriteErrorFile(object):
    def test_numbers_are_rounded(self, tmp_path):
        f = str(tmp_path / 's.dat')
        utils.write_error_file(f, 10.123456789, 1000, 0.0123456, 100)
        assert open(f).read() == 'coverage\t10.12346\ngenome_length\t1000\nerror_rate\t0.01235\nread_length\t100\n'

    def test_strings_are_kept(self, tmp_path):
        f = str(tmp_path / 's.dat')
        utils.write_error_file(f, 'NA', 5000, 0, 'NA', 'NA')
        assert open(f).read() == 'coverage\tNA\ngenome_length\t5000\nerror_rate\t0.0\nread_length\tNA\ntheta\tNA\n'

    def test_theta_is_appended(self, tmp_path):
        f = str(tmp_path / 's.dat')
        utils.write_error_file(f, 5.0, 1000, 0.01, 100, 0.0033333)
        assert open(f).read().split('\n')[4] == 'theta\t0.00333'

class TestCountKmers(object):
    histo = '1 100\n2 50\n3 10\n'

    def fake_call(self, cmd, **kwargs):
        # jellyfish count writes the .jf file
        open(cmd[cmd.index('-o') + 1], 'w').close()

    def test_runs_jellyfish_and_writes_histogram(self, tmp_path):
        with mock.patch('skmer.utils.call', side_effect=self.fake_call) as call, mock.patch('skmer.utils.check_output', return_value=self.histo) as out:
            histo = utils.count_kmers(str(tmp_path), 's', 'reads.fastq', 21, 4)
        assert histo == self.histo
        assert call.call_args[0][0] == ['jellyfish', 'count', '-m', '21', '-s', '100M', '-t', '4', '-C', '-o', str(tmp_path / 's.jf'), 'reads.fastq']
        assert out.call_args[0][0] == ['jellyfish', 'histo', '-h', '1000000', str(tmp_path / 's.jf')]
        assert (tmp_path / 's.hist').read_text() == self.histo
        assert not (tmp_path / 's.jf').exists()

    def test_existing_histogram_is_reused(self, tmp_path):
        (tmp_path / 's.hist').write_text(self.histo)
        with mock.patch('skmer.utils.call') as call, mock.patch('skmer.utils.check_output') as out:
            assert utils.count_kmers(str(tmp_path), 's', 'reads.fastq', 21, 4) == self.histo
        call.assert_not_called()
        out.assert_not_called()

    def test_empty_histogram_is_recomputed(self, tmp_path):
        (tmp_path / 's.hist').write_text('')
        with mock.patch('skmer.utils.call', side_effect=self.fake_call), mock.patch('skmer.utils.check_output', return_value=self.histo):
            assert utils.count_kmers(str(tmp_path), 's', 'reads.fastq', 21, 4) == self.histo

    def test_gzipped_input_is_piped_through_zcat(self, tmp_path):
        (tmp_path / 's.jf').write_text('')
        with mock.patch('skmer.utils.run') as run, mock.patch('skmer.utils.check_output', return_value=self.histo):
            utils.count_kmers(str(tmp_path), 's', 'reads.fastq.gz', 21, 4)
        cmd = run.call_args[0][0]
        assert cmd.startswith('zcat reads.fastq.gz | jellyfish count -m 21') and run.call_args[1]['shell'] is True

def test_get_hist_data(tmp_path):
    write_hist(tmp_path, 's', [0, 100, 50, 10])
    hist, ksum, usum = utils.get_hist_data(str(tmp_path), 's')
    assert ksum == 1 * 100 + 2 * 50 + 3 * 10
    assert usum == 160
    assert hist.shape == (3, 2)

class TestCovTempFunc(object):
    def test_matches_formula(self):
        x, r, p, k, l = 10.0, 0.3, 0.9, 21, 100
        lam = x * (l - k) / l
        expected = lam * p ** 2 * np.exp(-lam * p) - 2 * r * (p * np.exp(-lam * p) + 1 - p)
        assert utils.cov_temp_func(x, r, p, k, l) == pytest.approx(expected)

    def test_is_zero_when_ratio_is_consistent_with_poisson(self):
        # for no errors (p = 1) the root is where lam * exp(-lam) / (2 * r) = ... check that a root exists between 0.01 and 50
        values = [utils.cov_temp_func(x, 0.1, 1.0, 21, 100) for x in [0.01, 50]]
        assert values[0] * values[1] < 0

class TestCheckAllAssemblies(object):
    def test_assemblies_pass(self):
        utils.check_all_assemblies({'a': 'NA', 'b': 'NA'})

    def test_reads_raise(self):
        with pytest.raises(ValueError, match='reads: b'):
            utils.check_all_assemblies({'a': 'NA', 'b': 150})
