import os
import argparse
import numpy as np
import pandas as pd
import pytest
from unittest import mock

import importlib
ss = importlib.import_module("skmer.subsample.subsample")
from conftest import write_fastq, write_fasta, random_seq

reads = [('r{0}'.format(i), random_seq(50, i)) for i in range(200)]

def read_names(path):
    with open(str(path)) as f:
        return [line.strip() for i, line in enumerate(f) if i % 4 == 0]

class TestCreateSketchDir(object):
    def test_writes_dat_file(self, tmp_path):
        ss.create_sketch_dir('/x/g1.fna', str(tmp_path), {'g1': 'NA'}, {'g1': 5000}, {'g1': 0}, {'g1': 'NA'}, 1)
        assert (tmp_path / 'g1' / 'g1.dat').read_text() == 'coverage\tNA\ngenome_length\t5000\nerror_rate\t0.0\nread_length\tNA\n'

    def test_existing_directory_is_fine(self, tmp_path):
        (tmp_path / 'g1').mkdir()
        ss.create_sketch_dir('/x/g1.fna', str(tmp_path), {'g1': 'NA'}, {'g1': 5000}, {'g1': 0}, {'g1': 'NA'}, 1)
        assert (tmp_path / 'g1' / 'g1.dat').exists()

class TestEstimateStats(object):
    def test_reads(self, tmp_path):
        f = write_fastq(tmp_path / 's.fastq', [('a', 'ACGT' * 25), ('b', 'ACGT' * 25)])
        assert ss.estimate_stats(f, 1) == ('s', 0.0, 200, 0.0, 100, 2)

    def test_assembly(self, tmp_path):
        f = write_fasta(tmp_path / 'g.fna', [('c1', random_seq(3000))])
        assert ss.estimate_stats(f, 1) == ('g', 'NA', 3000, 0, 'NA', 1)

class TestSampleReads(object):
    @pytest.fixture
    def fastq(self, tmp_path):
        return write_fastq(tmp_path / 's.fastq', reads)

    def test_python_fallback_samples_requested_number(self, tmp_path, fastq):
        out = str(tmp_path / 'out.fastq')
        ss.sample_reads_python(fastq, 3, 20, out)
        names = read_names(out)
        assert len(names) == 20 and len(set(names)) == 20
        assert set(names) <= {'@' + n for n, _ in reads}

    def test_python_fallback_is_reproducible_by_seed(self, tmp_path, fastq):
        for i, seed in enumerate([3, 3, 4]):
            ss.sample_reads_python(fastq, seed, 20, str(tmp_path / 'o{0}.fastq'.format(i)))
        a, b, c = (read_names(tmp_path / 'o{0}.fastq'.format(i)) for i in range(3))
        assert a == b and a != c

    def test_python_fallback_keeps_records_intact(self, tmp_path, fastq):
        out = str(tmp_path / 'out.fastq')
        ss.sample_reads_python(fastq, 3, 5, out)
        lines = open(out).read().split('\n')[:-1]
        seqs = dict(reads)
        for i in range(0, len(lines), 4):
            assert lines[i + 1] == seqs[lines[i][1:]] and lines[i + 2] == '+'

    def test_python_fallback_more_than_available_returns_all(self, tmp_path, fastq):
        out = str(tmp_path / 'out.fastq')
        ss.sample_reads_python(fastq, 3, 1000, out)
        assert len(read_names(out)) == 200

    def test_python_fallback_reads_gzip(self, tmp_path):
        gz = write_fastq(tmp_path / 's.fastq.gz', reads)
        out = str(tmp_path / 'out.fastq')
        ss.sample_reads_python(gz, 1, 10, out)
        assert len(read_names(out)) == 10

    def test_uses_seqkit_two_pass_sampling(self, tmp_path, fastq):
        with mock.patch('skmer.subsample.subsample.which', return_value='/bin/seqkit'), mock.patch('skmer.subsample.subsample.run') as run:
            ss.sample_reads(fastq, 11, 20, str(tmp_path / 'bs'))
        assert run.call_args[0][0] == ['seqkit', 'sample', '-n', '20', '-s', '11', '-2', '-j', '1', '-o', str(tmp_path / 'bs' / 's.fastq'), fastq]
        assert run.call_args[1]['check'] is True
        assert (tmp_path / 'bs').is_dir()

    def test_output_name_drops_gz_extension(self, tmp_path):
        gz = write_fastq(tmp_path / 's.fastq.gz', reads)
        with mock.patch('skmer.subsample.subsample.which', return_value='/bin/seqkit'), mock.patch('skmer.subsample.subsample.run') as run:
            ss.sample_reads(gz, 1, 5, str(tmp_path / 'bs'))
        assert run.call_args[0][0][run.call_args[0][0].index('-o') + 1] == str(tmp_path / 'bs' / 's.fastq')

    def test_falls_back_to_python_without_seqkit(self, tmp_path, fastq, capsys):
        with mock.patch('skmer.subsample.subsample.which', return_value=None):
            ss.sample_reads(fastq, 1, 10, str(tmp_path / 'bs'))
        assert len(read_names(tmp_path / 'bs' / 's.fastq')) == 10
        assert 'seqkit not found' in capsys.readouterr().err

    @pytest.mark.integration
    def test_seqkit_samples_exact_number_reproducibly(self, tmp_path, fastq):
        for i in range(2):
            ss.sample_reads(fastq, 5, 30, str(tmp_path / 'bs{0}'.format(i)))
        a, b = (read_names(tmp_path / 'bs{0}'.format(i) / 's.fastq') for i in range(2))
        assert len(a) == 30 and a == b

    @pytest.mark.integration
    def test_seqkit_reads_gzip(self, tmp_path):
        gz = write_fastq(tmp_path / 's.fastq.gz', reads)
        ss.sample_reads(gz, 5, 30, str(tmp_path / 'bs'))
        assert len(read_names(tmp_path / 'bs' / 's.fastq')) == 30

class TestChooseReads(object):
    def test_each_replicate_has_exactly_the_requested_reads(self):
        chosen = ss.choose_reads(1000, 100, [1, 2, 3])
        for r in range(3):
            assert ((chosen >> r) & 1).sum() == 100

    def test_choices_depend_only_on_the_seed(self):
        a = ss.choose_reads(1000, 100, [5, 6, 7])
        b = ss.choose_reads(1000, 100, [7])
        assert np.array_equal((a >> 2) & 1, b & 1)

    def test_seeds_give_different_replicates(self):
        chosen = ss.choose_reads(1000, 100, [1, 2])
        assert not np.array_equal(chosen & 1, (chosen >> 1) & 1)

    def test_more_requested_than_available_chooses_all(self):
        assert (ss.choose_reads(10, 50, [1]) == 1).all()

class TestSampleReadsMulti(object):
    @pytest.fixture
    def fastq(self, tmp_path):
        return write_fastq(tmp_path / 's.fastq', reads)

    def dirs(self, tmp_path, n):
        return [str(tmp_path / 'rep{0}'.format(i)) for i in range(n)]

    def test_writes_one_subsample_per_seed(self, tmp_path, fastq):
        dirs = self.dirs(tmp_path, 3)
        ss.sample_reads_multi(fastq, [1, 2, 3], 20, 200, dirs)
        samples = [read_names(os.path.join(d, 's.fastq')) for d in dirs]
        assert all(len(x) == 20 and len(set(x)) == 20 for x in samples)
        assert samples[0] != samples[1]

    def test_matches_the_chosen_reads(self, tmp_path, fastq):
        dirs = self.dirs(tmp_path, 2)
        ss.sample_reads_multi(fastq, [4, 9], 30, 200, dirs)
        chosen = ss.choose_reads(200, 30, [4, 9])
        for r, d in enumerate(dirs):
            assert read_names(os.path.join(d, 's.fastq')) == ['@r{0}'.format(i) for i in np.flatnonzero((chosen >> r) & 1)]

    def test_replicate_does_not_depend_on_the_others(self, tmp_path, fastq):
        ss.sample_reads_multi(fastq, [4, 9], 30, 200, [str(tmp_path / 'a0'), str(tmp_path / 'a1')])
        ss.sample_reads_multi(fastq, [9], 30, 200, [str(tmp_path / 'b0')])
        assert open(str(tmp_path / 'a1' / 's.fastq')).read() == open(str(tmp_path / 'b0' / 's.fastq')).read()

    def test_records_are_intact(self, tmp_path, fastq):
        dirs = self.dirs(tmp_path, 1)
        ss.sample_reads_multi(fastq, [4], 20, 200, dirs, block_sz=7)
        lines = open(os.path.join(dirs[0], 's.fastq')).read().split('\n')[:-1]
        seqs = dict(reads)
        for i in range(0, len(lines), 4):
            assert lines[i + 1] == seqs[lines[i][1:]] and lines[i + 2] == '+' and lines[i + 3] == 'I' * 50

    def test_block_size_does_not_change_the_output(self, tmp_path, fastq):
        ss.sample_reads_multi(fastq, [4], 50, 200, [str(tmp_path / 'a')], block_sz=7)
        ss.sample_reads_multi(fastq, [4], 50, 200, [str(tmp_path / 'b')], block_sz=100000)
        assert open(str(tmp_path / 'a' / 's.fastq')).read() == open(str(tmp_path / 'b' / 's.fastq')).read()

    def test_gzipped_input_gives_plain_output(self, tmp_path):
        gz = write_fastq(tmp_path / 's.fastq.gz', reads)
        ss.sample_reads_multi(gz, [4], 20, 200, [str(tmp_path / 'a')])
        assert len(read_names(tmp_path / 'a' / 's.fastq')) == 20

    def test_wrong_read_count_raises(self, tmp_path, fastq):
        with pytest.raises(ValueError, match='expected 300'):
            ss.sample_reads_multi(fastq, [1], 20, 300, self.dirs(tmp_path, 1))
        with pytest.raises(ValueError, match='complete fastq reads'):
            ss.sample_reads_multi(fastq, [1], 20, 100, self.dirs(tmp_path, 1))

    def test_too_many_replicates_raises(self, tmp_path, fastq):
        with pytest.raises(ValueError, match='At most 32'):
            ss.sample_reads_multi(fastq, list(range(33)), 20, 200, self.dirs(tmp_path, 33))

    def test_is_fastq(self, tmp_path, fastq):
        assert ss.is_fastq(fastq) and ss.is_fastq(write_fastq(tmp_path / 'g.fastq.gz', reads))
        assert not ss.is_fastq(write_fasta(tmp_path / 'a.fna', [('c', 'ACGT')]))

def make_replicates(sub, main_dist, replicates, block=50, size=200):
    '''writes the files `skmer correct` reads: a main matrix and per-replicate matrices'''
    names = ['a', 'b']
    pd.DataFrame(main_dist, index=names, columns=names).rename_axis('sample').to_csv(str(sub / 'main.txt'), sep='\t')
    for i, dist in enumerate(replicates):
        rep = sub / 'rep{0}'.format(i)
        rep.mkdir()
        pd.DataFrame(dist, index=names, columns=names).rename_axis('sample').to_csv(str(rep / 'dimtrx_rep.txt'), sep='\t')
        np.save(str(rep / 'block_size.npy'), {'a': block, 'b': block})
        np.save(str(rep / 'sample_size.npy'), {'a': size, 'b': size})

class TestCorrection(object):
    def test_corrects_replicates_towards_main_estimate(self, tmp_path):
        make_replicates(tmp_path, [[0, .10], [.10, 0]], [[[0, .09], [.09, 0]], [[0, .11], [.11, 0]]])
        ss.correction(argparse.Namespace(main=str(tmp_path / 'main.txt'), sub=str(tmp_path), p=1))
        cor = pd.read_csv(str(tmp_path / 'rep0' / 'dimtrx_rep_cor.txt'), sep='\t', index_col=0)
        cons = pd.read_csv(str(tmp_path / 'rep0' / 'dimtrx_rep_cor_cons.txt'), sep='\t', index_col=0)
        # scale = sqrt(b / N) = 0.5; rep0 is 0.01 below the replicate mean (0.10)
        assert cor.loc['a', 'b'] == pytest.approx(0.5 * (0.09 - 0.10) + 0.10, abs=1e-5)
        assert cons.loc['a', 'b'] == pytest.approx(0.5 * (0.09 - 0.10) + 0.10, abs=1e-5)
        assert (tmp_path / '_summary.csv').exists()
        assert (tmp_path / 'main_cor_.txt').exists()

    def test_negative_corrected_distances_are_zero(self, tmp_path):
        make_replicates(tmp_path, [[0, .001], [.001, 0]], [[[0, .0], [.0, 0]], [[0, .2], [.2, 0]]], block=200, size=200)
        ss.correction(argparse.Namespace(main=str(tmp_path / 'main.txt'), sub=str(tmp_path), p=1))
        cor = pd.read_csv(str(tmp_path / 'rep0' / 'dimtrx_rep_cor.txt'), sep='\t', index_col=0)
        assert cor.loc['a', 'b'] == 0.0

    def test_bad_main_matrix_raises(self, tmp_path):
        with pytest.raises(ValueError, match='main distance matrix'):
            ss.correction(argparse.Namespace(main=str(tmp_path / 'missing.txt'), sub=str(tmp_path), p=1))

    def test_bad_subsample_directory_raises(self, tmp_path):
        make_replicates(tmp_path, [[0, .1], [.1, 0]], [])
        (tmp_path / 'rep0').mkdir()
        with pytest.raises(ValueError, match='subsample directory'):
            ss.correction(argparse.Namespace(main=str(tmp_path / 'main.txt'), sub=str(tmp_path), p=1))

@pytest.mark.integration
class TestSubsampleCommand(object):
    def args(self, tmp_path, inp, **kwargs):
        a = dict(input_dir=str(inp), sub=str(tmp_path / 'sub'), fa=True, msh=True, k=21, s=1000, S=42, i=0, b=2, chunk=10, c=0.9, e=None, t=False, p=1, r=None, d=False, theta=None)
        a.update(kwargs)
        return argparse.Namespace(**a)

    def test_reads(self, tmp_path, data_dir):
        ss.subsample(self.args(tmp_path, os.path.join(data_dir, 'ref_dir')))
        for rep in ['rep0', 'rep1']:
            mat = pd.read_csv(str(tmp_path / 'sub' / rep / 'dimtrx_rep.txt'), sep='\t', index_col=0)
            assert mat.shape == (3, 3)
            sizes = np.load(str(tmp_path / 'sub' / rep / 'block_size.npy'), allow_pickle=True).item()
            assert all(v > 0 for v in sizes.values())

    def test_same_seed_gives_same_replicates(self, tmp_path, data_dir):
        for name in ['x', 'y']:
            ss.subsample(self.args(tmp_path, os.path.join(data_dir, 'ref_dir'), sub=str(tmp_path / name), b=1))
        a, b = (pd.read_csv(str(tmp_path / n / 'rep0' / 'dimtrx_rep.txt'), sep='\t', index_col=0) for n in 'xy')
        pd.testing.assert_frame_equal(a, b)

    def test_replicates_do_not_depend_on_the_chunk_size(self, tmp_path, data_dir):
        for name, chunk in [('one', 1), ('many', 3)]:
            ss.subsample(self.args(tmp_path, os.path.join(data_dir, 'ref_dir'), sub=str(tmp_path / name), b=3, chunk=chunk))
        for rep in ['rep0', 'rep1', 'rep2']:
            a, b = (pd.read_csv(str(tmp_path / n / rep / 'dimtrx_rep.txt'), sep='\t', index_col=0) for n in ['one', 'many'])
            pd.testing.assert_frame_equal(a, b)

    def test_sampled_reads_are_removed_unless_kept(self, tmp_path, data_dir):
        ss.subsample(self.args(tmp_path, os.path.join(data_dir, 'ref_dir'), b=1))
        assert not (tmp_path / 'sub' / 'rep0' / 'ref_1.fastq').exists()

    def test_gzipped_reads(self, tmp_path):
        inp = tmp_path / 'inp'
        inp.mkdir()
        for i in range(2):
            write_fastq(inp / 's{0}.fastq.gz'.format(i), [('r{0}'.format(j), random_seq(100, i * 10000 + j)) for j in range(3000)])
        ss.subsample(self.args(tmp_path, inp, b=1))
        assert (tmp_path / 'sub' / 'rep0' / 'dimtrx_rep.txt').exists()

    def test_replicate_index_offset(self, tmp_path, data_dir):
        ss.subsample(self.args(tmp_path, os.path.join(data_dir, 'ref_dir'), i=5, b=1))
        assert (tmp_path / 'sub' / 'rep5' / 'dimtrx_rep.txt').exists()
