import os
import sys
import random
import shutil
import tempfile
import importlib
import importlib.util
import subprocess
import numpy as np
import pandas as pd
from shutil import which

from skmer import __version__
from skmer.popgen import read_matrix, read_annotations, assign_populations, calculate_popgen

python_modules = ['numpy', 'pandas', 'scipy', 'pyfastx']
executables = {'jellyfish': ['--version'], 'mash': ['--version'], 'seqkit': ['version']}

def find_data_dir():
    '''looks for the bundled test data (skmer/data, or data/ in a source checkout)'''
    package_dir = os.path.dirname(os.path.abspath(__file__))
    for data_dir in [os.path.join(package_dir, 'data'), os.path.join(os.path.dirname(package_dir), 'data')]:
        if os.path.exists(os.path.join(data_dir, 'ref_dir', 'ref_1.fastq')):
            return data_dir
    return None

def check_dependencies():
    '''returns [(name, passed, details)] for python modules and external tools'''
    checks = []
    for module in python_modules:
        try:
            version = getattr(importlib.import_module(module), '__version__', '')
            checks.append(('python module: ' + module, True, str(version)))
        except ImportError:
            checks.append(('python module: ' + module, False, 'not installed (pip install {0})'.format(module)))
    for exe, version_args in executables.items():
        if which(exe) is None:
            checks.append(('tool: ' + exe, False, 'not found in PATH (conda install -c bioconda {0})'.format('kmer-jellyfish' if exe == 'jellyfish' else exe)))
            continue
        out = subprocess.run([exe] + version_args, capture_output=True, text=True)
        checks.append(('tool: ' + exe, True, (out.stdout + out.stderr).strip().split('\n')[0]))
    return checks

def run_skmer(args, cwd):
    '''runs a skmer command as a subprocess, raising an error with its output if it fails'''
    out = subprocess.run([sys.executable, '-m', 'skmer', '--debug'] + [str(a) for a in args], cwd=cwd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError('skmer {0} exited with {1}:\n{2}'.format(' '.join(str(a) for a in args), out.returncode, out.stderr[-1500:]))
    return out

def check_distance_matrix(path, n_samples):
    '''a distance matrix is square, has a zero diagonal, and has finite, positive, (nearly) symmetric distances'''
    df = read_matrix(path)
    if df.shape != (n_samples, n_samples):
        raise AssertionError('{0} has shape {1}, expected {2}'.format(path, df.shape, (n_samples, n_samples)))
    mat = df.values.astype(float)
    if not np.all(np.isfinite(mat)):
        raise AssertionError('{0} has missing or infinite distances'.format(path))
    if not np.allclose(np.diag(mat), 0):
        raise AssertionError('{0} has a non-zero diagonal'.format(path))
    off_diag = mat[~np.eye(n_samples, dtype=bool)]
    if not np.all(off_diag > 0):
        raise AssertionError('{0} has non-positive distances between different samples'.format(path))
    if not np.allclose(mat, mat.T, atol=0.05):
        raise AssertionError('{0} is not symmetric'.format(path))
    return df

def mutate_assembly(fasta_in, fasta_out, rate, seed):
    '''writes a copy of an assembly with a fraction (rate) of its bases substituted'''
    rng = random.Random(seed)
    swap = {'A': 'C', 'C': 'G', 'G': 'T', 'T': 'A'}
    with open(fasta_in) as fin, open(fasta_out, 'w') as fout:
        for line in fin:
            if not line.startswith('>'):
                line = ''.join(swap.get(b, b) if (rng.random() < rate and b in swap) else b for b in line.rstrip('\n')) + '\n'
            fout.write(line)

# =================================
# tests (each takes the data directory, a shared working directory and the number of processors;
# later tests reuse the libraries of earlier ones)
# =================================
def test_reference_reads(data, work, nth):
    run_skmer(['reference', os.path.join(data, 'ref_dir'), '-l', 'lib_reads', '-o', 'reads-dist', '-p', nth], work)
    check_distance_matrix(os.path.join(work, 'reads-dist.txt'), 3)
    return '3 x 3 distance matrix'

def test_query_reads(data, work, nth):
    run_skmer(['query', os.path.join(data, 'qry.fastq'), 'lib_reads', '-o', 'qry-dist', '-p', nth], work)
    out = pd.read_csv(os.path.join(work, 'qry-dist-qry.txt'), sep='\t', header=0, index_col=0)
    if out.shape[0] != 3 or not np.all(out.values > 0):
        raise AssertionError('expected 3 positive distances from the query, got {0}'.format(out.values.ravel()))
    return 'closest reference: {0}'.format(out.index[0])

def test_distance_command(data, work, nth):
    run_skmer(['distance', 'lib_reads', '-o', 'reads-dist2', '-p', nth], work)
    first, second = read_matrix(os.path.join(work, 'reads-dist.txt')), check_distance_matrix(os.path.join(work, 'reads-dist2.txt'), 3)
    if not np.allclose(first.values.astype(float), second.loc[first.index, first.columns].values.astype(float), atol=1e-3):
        raise AssertionError('distance command did not reproduce the reference distances')
    return 'matches reference command'

def test_reskmer_spectra(data, work, nth):
    run_skmer(['reference', os.path.join(data, 'ref_dir'), '-l', 'lib_reskmer', '-o', 'reskmer-dist', '-r', os.path.join(data, 'ref_dir-spectra.txt'), '-p', nth], work)
    check_distance_matrix(os.path.join(work, 'reskmer-dist.txt'), 3)
    return '3 x 3 repeat-aware distance matrix'

def test_reskmer_assemblies(data, work, nth):
    asm_dir = os.path.join(work, 'assemblies')
    os.makedirs(asm_dir)
    mutate_assembly(os.path.join(data, 'ref_4.fna'), os.path.join(asm_dir, 'asm_a.fna'), 0.0, 1)
    mutate_assembly(os.path.join(data, 'ref_4.fna'), os.path.join(asm_dir, 'asm_b.fna'), 0.01, 2)
    run_skmer(['reference', asm_dir, '-l', 'lib_asm', '-o', 'asm-dist', '-r', 'assemblies', '-p', nth], work)
    df = check_distance_matrix(os.path.join(work, 'asm-dist.txt'), 2)
    d = float(df.loc['asm_a', 'asm_b'])
    # 1% of bases were substituted, so the distance should be near 0.01
    if not 0.005 <= d <= 0.02:
        raise AssertionError('distance between assemblies with 1% substitutions was {0}, expected ~0.01'.format(d))
    return 'd = {0} (expected ~0.01)'.format(d)

def test_query_assemblies(data, work, nth):
    mutate_assembly(os.path.join(data, 'ref_4.fna'), os.path.join(work, 'asm_c.fna'), 0.02, 3)
    run_skmer(['query', 'asm_c.fna', 'lib_asm', '-o', 'asm-qry', '-r', 'assemblies', '-p', nth], work)
    out = pd.read_csv(os.path.join(work, 'asm-qry-asm_c.txt'), sep='\t', header=0, index_col=0)
    d = float(out.loc['asm_a'].iloc[0])
    if not 0.01 <= d <= 0.04:
        raise AssertionError('distance to the assembly with 2% substitutions was {0}, expected ~0.02'.format(d))
    return 'd = {0} (expected ~0.02)'.format(d)

def test_subsample_assemblies(data, work, nth):
    run_skmer(['subsample', os.path.join(work, 'assemblies'), '-sub', 'subsample_asm', '-b', 2, '-r', 'assemblies', '-p', nth], work)
    for rep in ['rep0', 'rep1']:
        check_distance_matrix(os.path.join(work, 'subsample_asm', rep, 'dimtrx_rep.txt'), 2)
    return '2 replicates'

def test_subsample(data, work, nth):
    run_skmer(['subsample', os.path.join(data, 'ref_dir'), '-sub', 'subsample', '-b', 2, '-p', nth], work)
    for rep in ['rep0', 'rep1']:
        check_distance_matrix(os.path.join(work, 'subsample', rep, 'dimtrx_rep.txt'), 3)
    return '2 replicates'

def test_dipskmer(data, work, nth):
    run_skmer(['reference', os.path.join(data, 'dip_ref'), '-l', 'lib_dip', '-o', 'dip-dist', '-d', '-theta', 0.005, '-p', nth], work)
    check_distance_matrix(os.path.join(work, 'dip-dist.txt'), 2)
    return '2 x 2 diploid distance matrix'

def test_pop(data, work, nth):
    samples = ['a1', 'a2', 'b1', 'b2']
    # within A = 0.1, within B = 0.2, between = 0.5
    mat = pd.DataFrame([[0, .1, .5, .5], [.1, 0, .5, .5], [.5, .5, 0, .2], [.5, .5, .2, 0]], index=samples, columns=samples)
    mat.index.name = 'sample'
    mat.to_csv(os.path.join(work, 'pop-matrix.txt'), sep='\t')
    with open(os.path.join(work, 'pop-annotations.tsv'), 'w') as f:
        f.write('sample\tpopulation\na1\tA\na2\tA\nb1\tB\nb2\tB\n')
    run_skmer(['pop', os.path.join(work, 'pop-matrix.txt'), os.path.join(work, 'pop-annotations.tsv'), '-o', os.path.join(work, 'pop-out')], work)
    expected = {'WeirCockerhamFst': 0.7, 'SlatkinFst': 0.53846, 'NeiFst': 0.6087, 'Dxy': 0.5}
    for name, value in expected.items():
        mat = pd.read_csv(os.path.join(work, 'pop-out-{0}.txt'.format(name)), sep='\t', index_col=0)
        if abs(float(mat.loc['A', 'B']) - value) > 1e-4:
            raise AssertionError('{0} was {1}, expected {2}'.format(name, mat.loc['A', 'B'], value))
    thetas = pd.read_csv(os.path.join(work, 'pop-out-thetas.txt'), sep='\t', header=None, index_col=0)
    if abs(float(thetas.loc['A', 1]) - 0.1) > 1e-6 or abs(float(thetas.loc['B', 1]) - 0.2) > 1e-6:
        raise AssertionError('unexpected thetas: {0}'.format(thetas.values.ravel()))
    return 'Fst(WC) = 0.7, theta = 0.1 / 0.2'

def test_unit_tests(data, work, nth):
    tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests')
    if not os.path.isdir(tests_dir):
        return None, 'unit tests are only available in a source checkout'
    if importlib.util.find_spec('pytest') is None:
        return None, 'pytest is not installed (pip install pytest)'
    if os.environ.get('SKMER_SELFTEST'):
        return None, 'already running inside skmer test'
    # integration tests are covered by running each command on the test data
    out = subprocess.run([sys.executable, '-m', 'pytest', tests_dir, '-q', '-x', '-p', 'no:cacheprovider', '-m', 'not integration'],
                         capture_output=True, text=True, env=dict(os.environ, SKMER_SELFTEST='1'))
    if out.returncode != 0:
        raise AssertionError(out.stdout[-1500:])
    return out.stdout.strip().split('\n')[-1]

# (name, function, needs external tools, needs data)
integration_tests = [('reference (reads)', test_reference_reads), ('query (reads)', test_query_reads),
                     ('distance', test_distance_command), ('reference -r spectra (ReSkmer)', test_reskmer_spectra),
                     ('reference -r assemblies', test_reskmer_assemblies), ('query -r assemblies', test_query_assemblies),
                     ('subsample', test_subsample), ('subsample -r assemblies', test_subsample_assemblies),
                     ('reference -d (DipSkmer)', test_dipskmer), ('pop', test_pop)]

def selftest(args):
    '''checks the installation and runs each skmer command on the bundled test data'''
    results = []
    sys.stderr.write('[test] skmer {0}\n'.format(__version__))
    sys.stderr.write('[test] Checking dependencies...\n')
    deps = check_dependencies()
    results += [(name, 'PASS' if ok else 'FAIL', details) for name, ok, details in deps]
    tools_ok = all(ok for _, ok, _ in deps)

    sys.stderr.write('[test] Running unit tests...\n')
    try:
        out = test_unit_tests(None, None, None)
        results.append(('unit tests', 'SKIP', out[1]) if isinstance(out, tuple) else ('unit tests', 'PASS', out))
    except Exception as e:
        results.append(('unit tests', 'FAIL', str(e)))

    data = find_data_dir()
    if args.quick:
        results.append(('commands on test data', 'SKIP', '--quick'))
    elif data is None:
        results.append(('commands on test data', 'FAIL', 'bundled test data not found'))
    elif not tools_ok:
        results.append(('commands on test data', 'SKIP', 'missing dependencies'))
    else:
        work_root = tempfile.mkdtemp(prefix='skmer-test-')
        for name, test in integration_tests:
            sys.stderr.write('[test] Running {0}...\n'.format(name))
            try:
                results.append((name, 'PASS', test(data, work_root, args.p)))
            except Exception as e:
                results.append((name, 'FAIL', str(e)))
        if args.keep:
            sys.stderr.write('[test] Outputs kept in {0}\n'.format(work_root))
        else:
            shutil.rmtree(work_root, ignore_errors=True)

    width = max(len(name) for name, _, _ in results)
    for name, status, details in results:
        print('{0}  {1}  {2}'.format(name.ljust(width), status, details.split('\n')[0] if status != 'FAIL' else details))
    failed = [name for name, status, _ in results if status == 'FAIL']
    print('\n{0} passed, {1} failed, {2} skipped'.format(sum(s == 'PASS' for _, s, _ in results), len(failed), sum(s == 'SKIP' for _, s, _ in results)))
    if failed:
        sys.exit(1)
