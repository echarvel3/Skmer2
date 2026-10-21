# Changelog

All notable changes to ReSkmer are documented here. Versions follow [semantic versioning](https://semver.org/):
new backwards-compatible features bump the minor version, bug fixes bump the patch version.

## [0.3.0]
### Added
- `skmer pop`: theta (within-population diversity), Dxy, and Nei / Slatkin / Weir-Cockerham Fst from a distance matrix and annotation file. Theta, Dxy, Slatkin and Weir-Cockerham Fst reproduce `calculate_fst.py`; Nei Fst pools every pair of samples once (the script counted within-population pairs twice relative to between-population pairs).
- `skmer test`: checks dependencies and runs a smoke test of every command on the bundled data.
- `-r assemblies`: when all inputs are assemblies, each assembly's own jellyfish repeat spectrum is used.
- `--version` flag.
- Unit test suite (`tests/`) covering every function.
- `benchmarks/bench_sampling.py` to compare read-sampling methods.
### Changed
- `subsample` samples the reads of up to `-chunk` (default 10) replicates with a single pass over each input file (`sample_reads_multi`), instead of reading each input once or twice per replicate. Replicates are reproducible by seed and independent of `-chunk`. Fasta reads are sampled with `seqkit sample` (seeded) instead of a custom python script.
- Version is declared once, in `skmer/__init__.py`; `setup.py` reads it from there.
### Fixed
- `gzip` was never imported for gzipped subsample inputs, subsampled gzipped inputs were looked up under the wrong name, and subsample seeds were ignored.
- `query -d` parsed theta as an integer.
- `query -r` sketched the query twice.
- `correct` used `DataFrame.append`, removed in pandas 2.
- `estimate_diploid_cov` used `newton` without importing it.
- `reskmer + dipskmer` samples with an out-of-range error rate now get `theta = NA`.
- Removed dead code (`get_theta_arr`).

## [0.2.0]
- Merged DipSkmer (diploid-aware distances, `-d`, `-theta`) into ReSkmer.

## [0.1.0]
- First ReSkmer release: repeat-aware distances (`-r`) on top of Skmer 4.4.0.
