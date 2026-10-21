import os
import argparse
import pandas as pd
import pytest
from unittest import mock

from skmer import skmer_functions as sf
from conftest import FakePool, write_hist, write_fastq

names = ['s1', 's2', 's3']

def fake_dist(label):
    '''distance function that records its name; distance is the index difference of the sample names'''
    def dist(s1, s2, lib1, lib2, *args):
        return s1, s2, 0.0 if s1 == s2 else abs(int(s1[1:]) - int(s2[1:])) / 10.0
    dist.label = label
    return dist

@pytest.fixture
def patched(monkeypatch):
    '''replaces process pools and external-tool steps with in-process fakes'''
    calls = {'dist': [], 'sketch': [], 'cov': []}
    monkeypatch.setattr(sf.mp, 'Pool', FakePool)

    def estimate_cov(seq, lib, k, e, nth, skmer_ver, ref_hist, theta):
        sample = os.path.basename(seq).rsplit('.f', 1)[0]
        calls['cov'].append((sample, skmer_ver, ref_hist, theta))
        os.makedirs(os.path.join(lib, sample), exist_ok=True)
        return sample, 10.0, 1000, 0.01, 150, 0.004
    monkeypatch.setattr(sf, 'estimate_cov', estimate_cov)
    monkeypatch.setattr(sf, 'sketch', lambda *a, **kw: calls['sketch'].append(a))
    for label in ['estimate_skmer_dist', 'estimate_reskmer_dist', 'estimate_dipskmer_dist']:
        fxn = fake_dist(label)
        def wrapper(*a, _fxn=fxn, _label=label):
            calls['dist'].append(_label)
            return _fxn(*a)
        monkeypatch.setattr(sf, label, wrapper)
    monkeypatch.setattr(sf, 'parse_reference', lambda r, k, nth, lib, ver: pd.DataFrame({0: [1.0], 'x': [5]}) if r != 'assemblies' else None)
    return calls

def ref_args(tmp_path, **kwargs):
    inp = tmp_path / 'inp'
    inp.mkdir(exist_ok=True)
    for n in names:
        (inp / (n + '.fastq')).write_text('x')
    args = dict(input_dir=str(inp), l=str(tmp_path / 'lib'), o=str(tmp_path / 'out'), k=21, s=1000, S=42, e=None, t=False, p=2, r=None, d=False, theta=None)
    args.update(kwargs)
    return argparse.Namespace(**args)

class TestReference(object):
    def test_writes_distance_matrix(self, tmp_path, patched):
        sf.reference(ref_args(tmp_path))
        mat = pd.read_csv(str(tmp_path / 'out.txt'), sep='\t', index_col=0)
        assert sorted(mat.index) == names
        assert mat.loc['s1', 's3'] == pytest.approx(0.2)
        assert (mat.values.diagonal() == 0).all()
        assert set(patched['dist']) == {'estimate_skmer_dist'}

    def test_writes_config(self, tmp_path, patched):
        sf.reference(ref_args(tmp_path))
        assert (tmp_path / 'lib' / 'CONFIG').read_text() == 'kmer_length\t21\nsketch_size\t1000\nsketching_seed\t42\n'

    def test_reskmer_uses_reskmer_distances(self, tmp_path, patched):
        sf.reference(ref_args(tmp_path, r='spectra.hist'))
        assert set(patched['dist']) == {'estimate_reskmer_dist'}
        assert patched['cov'][0][1] == 'reskmer' and patched['cov'][0][2] is not None

    def test_dipskmer_uses_diploid_distances(self, tmp_path, patched):
        sf.reference(ref_args(tmp_path, d=True, theta=0.004))
        assert set(patched['dist']) == {'estimate_dipskmer_dist'}
        assert patched['cov'][0][3] == 0.004

    def test_assemblies_keyword_builds_spectra_from_each_assembly(self, tmp_path, patched, monkeypatch):
        monkeypatch.setattr(sf, 'estimate_cov', lambda seq, lib, *a: (os.path.basename(seq).rsplit('.f', 1)[0], 'NA', 1000, 0, 'NA', None))
        built = {}
        monkeypatch.setattr(sf, 'build_assembly_spectra', lambda sample_libs: built.update(sample_libs) or 'spectra')
        sf.reference(ref_args(tmp_path, r='assemblies'))
        assert built == {n: str(tmp_path / 'lib') for n in names}
        assert set(patched['dist']) == {'estimate_reskmer_dist'}

    def test_assemblies_keyword_rejects_reads(self, tmp_path, patched):
        with pytest.raises(ValueError, match='assemblies'):
            sf.reference(ref_args(tmp_path, r='assemblies'))

def make_library(tmp_path, theta=False, assemblies=()):
    lib = tmp_path / 'library'
    lib.mkdir()
    (lib / 'CONFIG').write_text('kmer_length\t21\nsketch_size\t1000\nsketching_seed\t5\n')
    for n in names:
        (lib / n).mkdir()
        if n in assemblies:
            dat = 'coverage\tNA\ngenome_length\t5000\nerror_rate\t0\nread_length\tNA\n'
        else:
            dat = 'coverage\t10.0\ngenome_length\t1000\nerror_rate\t0.01\nread_length\t150\n'
        if theta:
            dat += 'theta\t0.0035\n'
        (lib / n / (n + '.dat')).write_text(dat)
    return str(lib)

class TestDistance(object):
    def args(self, lib, tmp_path, **kwargs):
        a = dict(library=lib, o=str(tmp_path / 'out'), t=False, p=2, r=None, d=False)
        a.update(kwargs)
        return argparse.Namespace(**a)

    def test_writes_matrix_for_library(self, tmp_path, patched):
        sf.distance(self.args(make_library(tmp_path), tmp_path))
        mat = pd.read_csv(str(tmp_path / 'out.txt'), sep='\t', index_col=0)
        assert sorted(mat.index) == names and set(patched['dist']) == {'estimate_skmer_dist'}

    def test_diploid_reads_theta_from_dat(self, tmp_path, patched, monkeypatch):
        seen = {}
        def dip(s1, s2, l1, l2, ce, le, ee, rl, k, ct, tran, theta):
            seen.update(theta)
            return s1, s2, 0.0
        monkeypatch.setattr(sf, 'estimate_dipskmer_dist', dip)
        sf.distance(self.args(make_library(tmp_path, theta=True), tmp_path, d=True))
        assert seen == {n: 0.0035 for n in names}

    def test_reskmer_with_spectra(self, tmp_path, patched):
        sf.distance(self.args(make_library(tmp_path), tmp_path, r='x.hist'))
        assert set(patched['dist']) == {'estimate_reskmer_dist'}

    def test_assemblies_keyword_requires_assembly_library(self, tmp_path, patched):
        with pytest.raises(ValueError, match='assemblies'):
            sf.distance(self.args(make_library(tmp_path), tmp_path, r='assemblies'))

    def test_assemblies_keyword_with_assembly_library(self, tmp_path, patched, monkeypatch):
        built = {}
        monkeypatch.setattr(sf, 'build_assembly_spectra', lambda sample_libs: built.update(sample_libs) or 'spectra')
        lib = make_library(tmp_path, assemblies=names)
        sf.distance(self.args(lib, tmp_path, r='assemblies'))
        assert built == {n: lib for n in names}

class TestQuery(object):
    def args(self, lib, qry, **kwargs):
        a = dict(input=qry, library=lib, a=False, o='dist', e=None, t=False, p=2, r=None, d=False, theta=None)
        a.update(kwargs)
        return argparse.Namespace(**a)

    @pytest.fixture
    def cwd(self, tmp_path, monkeypatch):
        work = tmp_path / 'work'
        work.mkdir()
        monkeypatch.chdir(work)
        return work

    def test_writes_sorted_distances(self, tmp_path, patched, cwd):
        lib = make_library(tmp_path)
        sf.query(self.args(lib, '/x/s0.fastq'))
        out = pd.read_csv(str(cwd / 'dist-s0.txt'), sep='\t', header=0, index_col=0)
        assert out.index.tolist() == ['s1', 's2', 's3'] and out.iloc[:, 0].is_monotonic_increasing
        assert not (cwd / 's0').exists()

    def test_query_with_same_name_as_reference_raises(self, tmp_path, patched, cwd):
        with pytest.raises(ValueError, match='same name'):
            sf.query(self.args(make_library(tmp_path), '/x/s1.fastq'))

    def test_add_query_to_library(self, tmp_path, patched, cwd):
        lib = make_library(tmp_path)
        sf.query(self.args(lib, '/x/s0.fastq', a=True))
        assert os.path.isdir(os.path.join(lib, 's0'))

    def test_diploid_theta_is_read_as_float(self, tmp_path, patched, cwd, monkeypatch):
        seen = {}
        def dip(s1, s2, l1, l2, ce, le, ee, rl, k, ct, tran, theta):
            seen.update(theta)
            return s1, s2, 0.0
        monkeypatch.setattr(sf, 'estimate_dipskmer_dist', dip)
        sf.query(self.args(make_library(tmp_path, theta=True), '/x/s0.fastq', d=True))
        assert seen['s1'] == 0.0035 and seen['s0'] == 0.004

    def test_spectrum_query_sketches_once(self, tmp_path, patched, cwd):
        sf.query(self.args(make_library(tmp_path), '/x/s0.fastq', r='x.hist'))
        assert len(patched['sketch']) == 1 and patched['sketch'][0][-1] is True
        assert set(patched['dist']) == {'estimate_reskmer_dist'}

    def test_assemblies_keyword_includes_query_in_spectra(self, tmp_path, patched, cwd, monkeypatch):
        monkeypatch.setattr(sf, 'estimate_cov', lambda *a: ('s0', 'NA', 1000, 0, 'NA', None))
        built = {}
        monkeypatch.setattr(sf, 'build_assembly_spectra', lambda sample_libs: built.update(sample_libs) or 'spectra')
        lib = make_library(tmp_path, assemblies=names)
        sf.query(self.args(lib, '/x/s0.fna', r='assemblies'))
        assert built == dict({n: lib for n in names}, s0=str(cwd))

    def test_assemblies_keyword_rejects_read_query(self, tmp_path, patched, cwd):
        lib = make_library(tmp_path, assemblies=names)
        with pytest.raises(ValueError, match='s0'):
            sf.query(self.args(lib, '/x/s0.fastq', r='assemblies'))
