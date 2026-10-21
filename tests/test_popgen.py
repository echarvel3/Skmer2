import os
import argparse
import numpy as np
import pandas as pd
import pytest

from skmer import popgen

samples = ['a1', 'a2', 'b1', 'b2']
# within A = 0.1, within B = 0.2, between = 0.5
matrix = pd.DataFrame([[0, .1, .5, .5], [.1, 0, .5, .5], [.5, .5, 0, .2], [.5, .5, .2, 0]], index=samples, columns=samples)
pops = {'a1': 'A', 'a2': 'A', 'b1': 'B', 'b2': 'B'}

def write_matrix(path, df=matrix):
    df = df.copy()
    df.index.name = 'sample'
    df.to_csv(str(path), sep='\t')
    return str(path)

def write_annotations(path, lines):
    with open(str(path), 'w') as f:
        f.write(lines)
    return str(path)

class TestReadAnnotations(object):
    def test_reads_pairs(self, tmp_path):
        f = write_annotations(tmp_path / 'a.tsv', 'a1\tA\nb1\tB\n')
        assert popgen.read_annotations(f) == {'a1': 'A', 'b1': 'B'}

    def test_ignores_header_and_blank_lines(self, tmp_path):
        f = write_annotations(tmp_path / 'a.tsv', 'sample\tpopulation\n\na1\tA\n')
        assert popgen.read_annotations(f) == {'a1': 'A'}

    def test_empty_file_raises(self, tmp_path):
        f = write_annotations(tmp_path / 'a.tsv', '\n')
        with pytest.raises(ValueError):
            popgen.read_annotations(f)

class TestReadMatrix(object):
    def test_reads_numbers(self, tmp_path):
        df = popgen.read_matrix(write_matrix(tmp_path / 'm.txt'))
        assert df.shape == (4, 4)
        assert df.loc['a1', 'b1'] == pytest.approx(0.5)

    def test_non_numbers_become_nan(self, tmp_path):
        f = tmp_path / 'm.txt'
        f.write_text('sample\tx\ty\nx\t0.0\tNone\ny\tNone\t0.0\n')
        df = popgen.read_matrix(str(f))
        assert pd.isna(df.loc['x', 'y'])

    def test_non_square_raises(self, tmp_path):
        f = tmp_path / 'm.txt'
        f.write_text('sample\tx\ty\nx\t0.0\t0.1\n')
        with pytest.raises(ValueError):
            popgen.read_matrix(str(f))

class TestAssignPopulations(object):
    def test_exact_names(self):
        assert popgen.assign_populations(['a1', 'b1'], {'a1': 'A', 'b1': 'B'}) == {'a1': 'A', 'b1': 'B'}

    def test_matches_names_without_extension(self):
        assert popgen.assign_populations(['a1.fna', 'b1.fastq'], {'a1': 'A', 'b1': 'B'}) == {'a1.fna': 'A', 'b1.fastq': 'B'}

    def test_unannotated_samples_are_dropped(self):
        assert popgen.assign_populations(['a1', 'zzz'], {'a1': 'A'}) == {'a1': 'A'}

    def test_no_matches_raises(self):
        with pytest.raises(ValueError):
            popgen.assign_populations(['x'], {'a1': 'A'})

class TestGroupDistances(object):
    def test_within_and_between(self):
        within, between = popgen.group_distances(matrix, pops)
        assert within['A'] == [.1, .1] and within['B'] == [.2, .2]
        assert between[('A', 'B')] == [.5] * 8

    def test_missing_distances_are_skipped(self):
        df = matrix.copy()
        df.loc['a1', 'a2'] = float('nan')
        within, _ = popgen.group_distances(df, pops)
        assert within['A'] == [.1]

    def test_diagonal_is_excluded(self):
        within, _ = popgen.group_distances(matrix, {'a1': 'A'})
        assert within['A'] == []

class TestSummaries(object):
    def test_mean(self):
        assert popgen.mean([1, 2, 3]) == 2
        assert popgen.mean([]) is None

    def test_theta_and_dxy(self):
        assert popgen.theta([.1, .3]) == pytest.approx(.2)
        assert popgen.theta([]) is None
        assert popgen.dxy([.5]) == .5
        assert popgen.dxy([]) is None

class TestFst(object):
    a, b, ab = [.1, .1], [.2, .2], [.5] * 8

    def test_weir_cockerham(self):
        # pi_s = .15, Dxy = .5, (Dxy - pi_s) / Dxy
        assert popgen.wc_fst(self.a, self.b, self.ab) == pytest.approx(0.7)

    def test_slatkin(self):
        assert popgen.slatkin_fst(self.a, self.b, self.ab) == pytest.approx(.35 / .65)

    def test_nei(self):
        pi_t = (.2 + .4 + 4.0) / 12
        assert popgen.nei_fst(self.a, self.b, self.ab) == pytest.approx((pi_t - .15) / pi_t)

    def test_identical_populations_have_zero_fst(self):
        same = [.2, .2]
        for fst in [popgen.wc_fst, popgen.slatkin_fst, popgen.nei_fst]:
            assert fst(same, same, same) == pytest.approx(0.0)

    def test_undefined_without_within_population_distances(self):
        for fst in [popgen.wc_fst, popgen.slatkin_fst, popgen.nei_fst]:
            assert fst([], self.b, self.ab) is None
            assert fst(self.a, self.b, []) is None

    def test_zero_denominators_are_undefined(self):
        for fst in [popgen.wc_fst, popgen.slatkin_fst, popgen.nei_fst]:
            assert fst([0.0], [0.0], [0.0]) is None

    def test_wc_fst_can_be_negative(self):
        assert popgen.wc_fst([.5, .5], [.5, .5], [.4]) < 0

class TestCalculatePopgen(object):
    def test_values(self):
        thetas, results = popgen.calculate_popgen(matrix, pops)
        assert thetas['population'].tolist() == ['A', 'B']
        assert thetas['theta'].tolist() == [.1, .2]
        assert results['Dxy'].loc['A', 'B'] == .5
        assert results['WeirCockerhamFst'].loc['A', 'B'] == .7
        assert results['SlatkinFst'].loc['A', 'B'] == pytest.approx(.53846)
        assert results['NeiFst'].loc['A', 'B'] == pytest.approx(.6087)

    def test_nei_pools_every_pair_of_samples_once(self):
        # pi_t is the mean of all off-diagonal distances of the two populations' samples
        _, results = popgen.calculate_popgen(matrix, pops)
        pi_t = matrix.values[~np.eye(4, dtype=bool)].mean()
        assert results['NeiFst'].loc['A', 'B'] == pytest.approx(round((pi_t - .15) / pi_t, 5))

    def test_matrices_are_symmetric_with_expected_diagonal(self):
        _, results = popgen.calculate_popgen(matrix, pops)
        for name, mat in results.items():
            assert mat.loc['A', 'B'] == mat.loc['B', 'A']
        assert results['Dxy'].loc['A', 'A'] == .1
        assert results['WeirCockerhamFst'].loc['A', 'A'] == 0.0

    def test_single_member_population_is_na(self):
        df = matrix.iloc[:3, :3]
        _, results = popgen.calculate_popgen(df, {'a1': 'A', 'a2': 'A', 'b1': 'B'})
        thetas, _ = popgen.calculate_popgen(df, {'a1': 'A', 'a2': 'A', 'b1': 'B'})
        assert thetas['theta'].tolist() == [.1, 'NA']
        assert results['WeirCockerhamFst'].loc['A', 'B'] == 'NA'
        assert results['WeirCockerhamFst'].loc['B', 'B'] == 'NA'
        # divergence is still defined
        assert results['Dxy'].loc['A', 'B'] == .5

    def test_matrix_index_is_named_sample(self):
        _, results = popgen.calculate_popgen(matrix, pops)
        assert results['Dxy'].index.name == 'sample'

class TestWritePopgen(object):
    def test_writes_five_files(self, tmp_path):
        thetas, results = popgen.calculate_popgen(matrix, pops)
        files = popgen.write_popgen(thetas, results, str(tmp_path / 'out' / 'x'))
        assert sorted(os.path.basename(f) for f in files) == sorted(['x-thetas.txt', 'x-Dxy.txt', 'x-NeiFst.txt', 'x-SlatkinFst.txt', 'x-WeirCockerhamFst.txt'])
        assert all(os.path.exists(f) for f in files)
        assert open(str(tmp_path / 'out' / 'x-thetas.txt')).read() == 'A\t0.1\nB\t0.2\n'
        assert open(str(tmp_path / 'out' / 'x-Dxy.txt')).read().split('\n')[0] == 'sample\tA\tB'

class TestPop(object):
    def test_end_to_end(self, tmp_path):
        m = write_matrix(tmp_path / 'm.txt')
        a = write_annotations(tmp_path / 'a.tsv', 'sample\tpopulation\na1\tA\na2\tA\nb1\tB\nb2\tB\n')
        files = popgen.pop(argparse.Namespace(matrix=m, annotation=a, o=str(tmp_path / 'res')))
        wc = pd.read_csv(str(tmp_path / 'res-WeirCockerhamFst.txt'), sep='\t', index_col=0)
        assert wc.loc['A', 'B'] == pytest.approx(.7)
        assert len(files) == 5

    def test_default_prefix_is_matrix_path_without_extension(self, tmp_path):
        m = write_matrix(tmp_path / 'm.txt')
        a = write_annotations(tmp_path / 'a.tsv', 'a1\tA\na2\tA\nb1\tB\nb2\tB\n')
        popgen.pop(argparse.Namespace(matrix=m, annotation=a, o=None))
        assert os.path.exists(str(tmp_path / 'm-thetas.txt'))

    def test_warns_about_unannotated_samples(self, tmp_path, capsys):
        m = write_matrix(tmp_path / 'm.txt')
        a = write_annotations(tmp_path / 'a.tsv', 'a1\tA\na2\tA\nb1\tB\n')
        popgen.pop(argparse.Namespace(matrix=m, annotation=a, o=str(tmp_path / 'r')))
        assert 'b2' in capsys.readouterr().err
