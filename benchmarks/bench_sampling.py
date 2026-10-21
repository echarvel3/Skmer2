#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Benchmarks methods for subsampling reads from a fastq file (as done by `skmer subsample`).

Each method runs in a fresh process, and its wall time, CPU time, peak memory and block IO are
measured with os.wait4 (resource usage of the child process). Output size and number of reads are checked.

Methods:
  python-two-pass   previous skmer method (count reads, then write a random set of read indices)
  pyfastx-index     index the fastq with pyfastx, then write randomly chosen reads
  seqkit-n-2pass    seqkit sample -n N -2   (exactly N reads, two passes, low memory) [new default]
  seqkit-n-1pass    seqkit sample -n N      (exactly N reads, keeps sampled reads in memory)
  seqkit-p          seqkit sample -p P      (approximately N reads, single pass)
  seqtk             seqtk sample            (exactly N reads, keeps sampled reads in memory)

Methods for sampling R replicates of the same file (--replicates R, as skmer subsample does):
  seqkit-n-2pass-xR  R runs of seqkit-n-2pass, which reads the input 2R times
  multi-pass-xR      skmer's sample_reads_multi, which reads the input once and writes all R replicates

Example: python benchmarks/bench_sampling.py --reads 100000 1000000 --reps 3 -o bench_results
'''
import os
import sys
import time
import gzip
import random
import argparse
import shutil
import platform
import subprocess
from shutil import which

methods = ['python-two-pass', 'pyfastx-index', 'seqkit-n-2pass', 'seqkit-n-1pass', 'seqkit-p', 'seqtk']
# sampling several replicates of the same file (as skmer subsample does): one pass per replicate or one pass for all
replicate_methods = ['seqkit-n-2pass-xR', 'multi-pass-xR']

def make_fastq(path, n_reads, read_len=150, seed=1):
    '''writes a synthetic fastq (gzipped if path ends with .gz)'''
    rng = random.Random(seed)
    opener = gzip.open if path.endswith('.gz') else open
    qual = 'I' * read_len
    with opener(path, 'wt') as f:
        for i in range(n_reads):
            seq = ''.join(rng.choices('ACGT', k=read_len))
            f.write('@read{0}\n{1}\n+\n{2}\n'.format(i, seq, qual))

def count_reads(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        return sum(1 for _ in f) // 4

def worker(method, path, n, seed, out, total=0, replicates=1):
    '''runs a python sampling method (called in a child process)'''
    if method == 'seqkit-n-2pass-xR':
        for r in range(replicates):
            subprocess.run(command('seqkit-n-2pass', path, n, seed + r, out + str(r), total), check=True, stderr=subprocess.DEVNULL)
    elif method == 'multi-pass-xR':
        from skmer.subsample.subsample import sample_reads_multi
        sample_reads_multi(path, [seed + r for r in range(replicates)], n, total, [out + str(r) for r in range(replicates)])
    if method == 'python-two-pass':
        from skmer.subsample.subsample import sample_reads_python
        sample_reads_python(path, seed, n, out)
    elif method == 'pyfastx-index':
        import pyfastx
        fq = pyfastx.Fastq(path, build_index=True)
        chosen = sorted(random.Random(seed).sample(range(len(fq)), min(n, len(fq))))
        with open(out, 'w') as fout:
            for i in chosen:
                fout.write(fq[i].raw)

def command(method, path, n, seed, out, total, replicates=1):
    if method == 'seqkit-n-2pass':
        return ['seqkit', 'sample', '-n', str(n), '-s', str(seed), '-2', '-j', '1', '-o', out, path]
    if method == 'seqkit-n-1pass':
        return ['seqkit', 'sample', '-n', str(n), '-s', str(seed), '-j', '1', '-o', out, path]
    if method == 'seqkit-p':
        return ['seqkit', 'sample', '-p', str(min(1.0, float(n) / total)), '-s', str(seed), '-j', '1', '-o', out, path]
    if method == 'seqtk':
        return ['seqtk', 'sample', '-s', str(seed), path, str(n)]
    return [sys.executable, os.path.abspath(__file__), '--worker', method, path, str(n), str(seed), out, str(total), str(replicates)]

def measure(cmd, out):
    '''runs a command, returning wall time (s), cpu time (s), peak memory (MB), block reads and writes'''
    start = time.perf_counter()
    stdout = open(out, 'w') if cmd[0] == 'seqtk' else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.DEVNULL)
    _, status, usage = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - start
    if cmd[0] == 'seqtk':
        stdout.close()
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError('command failed: ' + ' '.join(cmd))
    # ru_maxrss is in bytes on macOS and kilobytes on linux
    peak_mb = usage.ru_maxrss / (1024.0 ** 2 if platform.system() == 'Darwin' else 1024.0)
    return wall, usage.ru_utime + usage.ru_stime, peak_mb, usage.ru_inblock, usage.ru_oublock

def main():
    parser = argparse.ArgumentParser(description='Benchmark read sampling methods')
    parser.add_argument('--reads', type=int, nargs='+', default=[100000, 1000000], help='Number of reads in each synthetic fastq')
    parser.add_argument('--read-len', type=int, default=150)
    parser.add_argument('--exponent', type=float, default=0.9, help='Reads sampled = reads ** exponent (as in skmer subsample -c)')
    parser.add_argument('--reps', type=int, default=3, help='Replicates per method')
    parser.add_argument('--methods', nargs='+', default=methods, choices=methods + replicate_methods)
    parser.add_argument('--replicates', type=int, default=10, help='Replicates sampled by the xR methods')
    parser.add_argument('--pyfastx-gz', action='store_true', help='Also run pyfastx-index on gzipped files (very slow, since it must index the whole file)')
    parser.add_argument('--fastq', nargs='*', default=[], help='Existing fastq(.gz) files to benchmark in addition to synthetic ones')
    parser.add_argument('-o', default='bench_results', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.o, exist_ok=True)
    inputs = []
    for n_reads in args.reads:
        for ext in ['fastq', 'fastq.gz']:
            path = os.path.join(args.o, 'synthetic_{0}.{1}'.format(n_reads, ext))
            if not os.path.exists(path):
                sys.stderr.write('Generating {0}...\n'.format(path))
                make_fastq(path, n_reads, args.read_len)
            inputs.append(path)
    inputs += args.fastq

    rows = []
    for path in inputs:
        total = count_reads(path)
        n = round(total ** args.exponent)
        size_mb = os.path.getsize(path) / 1024.0 ** 2
        for method in args.methods:
            exe = 'seqkit' if method.startswith('seqkit') else ('seqtk' if method == 'seqtk' else None)
            if exe and which(exe) is None:
                sys.stderr.write('Skipping {0} ({1} not installed)\n'.format(method, exe))
                continue
            if method == 'pyfastx-index' and path.endswith('.gz') and not args.pyfastx_gz:
                sys.stderr.write('Skipping pyfastx-index on {0} (gzip indexing is very slow, see --pyfastx-gz)\n'.format(os.path.basename(path)))
                continue
            for rep in range(args.reps):
                # pyfastx caches its index next to the input: remove it so that every replicate pays for indexing
                if method == 'pyfastx-index' and os.path.exists(path + '.fxi'):
                    os.remove(path + '.fxi')
                out = os.path.join(args.o, 'sampled.fastq')
                wall, cpu, peak, blk_in, blk_out = measure(command(method, path, n, rep + 1, out, total, args.replicates), out)
                sampled = os.path.join(out + '0', os.path.basename(path)[:-3] if path.endswith('.gz') else os.path.basename(path)) if method == 'multi-pass-xR' else (out + '0' if method.endswith('xR') else out)
                n_out = count_reads(sampled)
                rows.append([os.path.basename(path), total, '{0:.1f}'.format(size_mb), method, rep, n, n_out,
                             '{0:.3f}'.format(wall), '{0:.3f}'.format(cpu), '{0:.1f}'.format(peak), blk_in, blk_out,
                             '{0:.1f}'.format(os.path.getsize(sampled) / 1024.0 ** 2)])
                sys.stderr.write('{0}\t{1}\trep{2}\twall={3:.2f}s\tpeak={4:.0f}MB\treads={5}/{6}\n'.format(
                    os.path.basename(path), method, rep, wall, peak, n_out, n))
                if method == 'multi-pass-xR':
                    for r in range(args.replicates):
                        shutil.rmtree(out + str(r))
                elif method.endswith('xR'):
                    for r in range(args.replicates):
                        os.remove(out + str(r))
                else:
                    os.remove(out)

    header = ['input', 'n_reads', 'input_mb', 'method', 'rep', 'n_requested', 'n_sampled', 'wall_s', 'cpu_s', 'peak_mem_mb',
              'block_reads', 'block_writes', 'output_mb']
    with open(os.path.join(args.o, 'sampling_benchmark.tsv'), 'w') as f:
        f.write('\t'.join(header) + '\n')
        for row in rows:
            f.write('\t'.join(str(x) for x in row) + '\n')

    # summary: median wall time and peak memory per input and method
    import pandas as pd
    df = pd.read_csv(os.path.join(args.o, 'sampling_benchmark.tsv'), sep='\t')
    summary = df.groupby(['input', 'method'], sort=False)[['wall_s', 'cpu_s', 'peak_mem_mb', 'n_sampled']].median().round(2)
    summary.to_csv(os.path.join(args.o, 'sampling_summary.tsv'), sep='\t')
    print(summary.to_string())

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], int(sys.argv[7]), int(sys.argv[8]))
    else:
        main()
