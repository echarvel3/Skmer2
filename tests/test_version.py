import os
import re
import subprocess
import sys

import skmer
from conftest import repo_dir

semver = re.compile(r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$')

def test_version_is_semantic():
    assert semver.match(skmer.__version__)

def test_setup_reads_the_package_version():
    out = subprocess.run([sys.executable, 'setup.py', '--version'], cwd=repo_dir, capture_output=True, text=True)
    assert out.stdout.strip().split('\n')[-1] == skmer.__version__

def test_changelog_documents_current_version_first():
    changelog = open(os.path.join(repo_dir, 'CHANGELOG.md')).read()
    versions = re.findall(r'^## \[(\d+\.\d+\.\d+)\]', changelog, re.M)
    assert versions[0] == skmer.__version__

def test_changelog_versions_are_unique_and_descending():
    changelog = open(os.path.join(repo_dir, 'CHANGELOG.md')).read()
    versions = [tuple(map(int, v.split('.'))) for v in re.findall(r'^## \[(\d+\.\d+\.\d+)\]', changelog, re.M)]
    assert versions == sorted(set(versions), reverse=True)

def test_no_other_hardcoded_version_in_setup():
    assert not re.search(r"version\s*=\s*['\"]\d", open(os.path.join(repo_dir, 'setup.py')).read())
