import os
import argparse
import numpy as np
import pandas as pd
import pytest
from unittest import mock

from skmer import selftest

def write_matrix(path, values, names):
    df = pd.DataFrame(values, index=names, columns=names)
    df.index.name = 'sample'
    df.to_csv(str(path), sep='\t')
    return str(path)

class TestFindDataDir(object):
    def test_finds_bundled_data(self):
        data = selftest.find_data_dir()
        assert data is not None and os.path.exists(os.path.join(data, 'ref_4.fna'))

class TestCheckDependencies(object):
    def test_reports_each_dependency(self):
        names = [name for name, _, _ in selftest.check_dependencies()]
        assert 'python module: numpy' in names and 'tool: seqkit' in names and 'tool: jellyfish' in names and 'tool: mash' in names

    def test_missing_tool_fails_with_hint(self):
        with mock.patch('skmer.selftest.which', return_value=None):
            checks = {name: (ok, details) for name, ok, details in selftest.check_dependencies()}
        assert checks['tool: mash'][0] is False and 'conda install' in checks['tool: mash'][1]

    def test_missing_python_module_fails(self):
        with mock.patch('skmer.selftest.importlib.import_module', side_effect=ImportError):
            checks = {name: ok for name, ok, _ in selftest.check_dependencies()}
        assert checks['python module: numpy'] is False

class TestCheckDistanceMatrix(object):
    names = ['a', 'b', 'c']
    good = [[0, .1, .2], [.1, 0, .3], [.2, .3, 0]]

    def test_valid_matrix(self, tmp_path):
        assert selftest.check_distance_matrix(write_matrix(tmp_path / 'm.txt', self.good, self.names), 3).shape == (3, 3)

    def test_wrong_size(self, tmp_path):
        with pytest.raises(AssertionError, match='shape'):
            selftest.check_distance_matrix(write_matrix(tmp_path / 'm.txt', self.good, self.names), 2)

    def test_nonzero_diagonal(self, tmp_path):
        bad = [[.1, .1, .2], [.1, 0, .3], [.2, .3, 0]]
        with pytest.raises(AssertionError, match='diagonal'):
            selftest.check_distance_matrix(write_matrix(tmp_path / 'm.txt', bad, self.names), 3)

    def test_zero_distance_between_samples(self, tmp_path):
        bad = [[0, 0, .2], [0, 0, .3], [.2, .3, 0]]
        with pytest.raises(AssertionError, match='non-positive'):
            selftest.check_distance_matrix(write_matrix(tmp_path / 'm.txt', bad, self.names), 3)

    def test_missing_distance(self, tmp_path):
        f = tmp_path / 'm.txt'
        f.write_text('sample\ta\tb\na\t0.0\tNone\nb\tNone\t0.0\n')
        with pytest.raises(AssertionError, match='missing'):
            selftest.check_distance_matrix(str(f), 2)

    def test_asymmetric(self, tmp_path):
        bad = [[0, .1, .2], [.9, 0, .3], [.2, .3, 0]]
        with pytest.raises(AssertionError, match='symmetric'):
            selftest.check_distance_matrix(write_matrix(tmp_path / 'm.txt', bad, self.names), 3)

class TestMutateAssembly(object):
    def make(self, tmp_path):
        src = tmp_path / 'a.fna'
        src.write_text('>c1 description\n' + 'ACGT' * 5000 + '\n')
        return str(src)

    def test_zero_rate_is_identical(self, tmp_path):
        src = self.make(tmp_path)
        selftest.mutate_assembly(src, str(tmp_path / 'b.fna'), 0.0, 1)
        assert open(src).read() == (tmp_path / 'b.fna').read_text()

    def test_fraction_of_bases_substituted(self, tmp_path):
        src = self.make(tmp_path)
        selftest.mutate_assembly(src, str(tmp_path / 'b.fna'), 0.05, 1)
        a, b = open(src).read().split('\n')[1], (tmp_path / 'b.fna').read_text().split('\n')[1]
        assert len(a) == len(b)
        assert sum(x != y for x, y in zip(a, b)) / len(a) == pytest.approx(0.05, abs=0.01)

    def test_header_and_seed(self, tmp_path):
        src = self.make(tmp_path)
        selftest.mutate_assembly(src, str(tmp_path / 'b.fna'), 0.05, 1)
        selftest.mutate_assembly(src, str(tmp_path / 'c.fna'), 0.05, 1)
        assert (tmp_path / 'b.fna').read_text().startswith('>c1 description\n')
        assert (tmp_path / 'b.fna').read_text() == (tmp_path / 'c.fna').read_text()

def test_pop_check_runs_the_pop_command(tmp_path):
    assert 'theta' in selftest.test_pop(None, str(tmp_path), 1)

@pytest.fixture(autouse=True)
def no_nested_pytest():
    with mock.patch('skmer.selftest.test_unit_tests', return_value=(None, 'mocked')):
        yield

class TestSelftest(object):
    def test_quick_run_reports_and_passes(self, capsys):
        selftest.selftest(argparse.Namespace(quick=True, keep=False, p=1))
        out = capsys.readouterr().out
        assert 'passed' in out and 'commands on test data' in out and 'SKIP' in out

    def test_failed_dependency_exits_nonzero(self, capsys):
        with mock.patch('skmer.selftest.which', return_value=None):
            with pytest.raises(SystemExit) as e:
                selftest.selftest(argparse.Namespace(quick=True, keep=False, p=1))
        assert e.value.code == 1
        assert 'FAIL' in capsys.readouterr().out

    @pytest.mark.integration
    def test_full_run_passes(self, capsys):
        selftest.selftest(argparse.Namespace(quick=False, keep=False, p=2))
        assert '0 failed' in capsys.readouterr().out
