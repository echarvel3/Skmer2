import sys
import pytest
from unittest import mock

from skmer import __main__ as cli
from skmer import __version__

def parse(monkeypatch, argv):
    '''runs main() with argv, returning the (command function name, parsed args) it dispatches to'''
    seen = {}
    for name in ['reference', 'distance', 'query', 'subsample', 'correction', 'pop', 'selftest']:
        monkeypatch.setattr(cli, name, (lambda n: lambda args: seen.update(name=n, args=args))(name))
    monkeypatch.setattr(sys, 'argv', ['skmer'] + argv)
    hook = sys.excepthook
    try:
        cli.main()
    finally:
        sys.excepthook = hook
    return seen['name'], seen['args']

class TestCommands(object):
    def test_reference_defaults(self, monkeypatch):
        name, args = parse(monkeypatch, ['reference', 'dir'])
        assert name == 'reference'
        assert (args.input_dir, args.k, args.s, args.S, args.e, args.t, args.r, args.d, args.theta, args.o) == \
               ('dir', 31, 100000, 42, None, False, None, False, None, 'ref-dist-mat')

    def test_reference_options(self, monkeypatch):
        _, args = parse(monkeypatch, ['reference', 'dir', '-k', '21', '-s', '500', '-S', '3', '-e', '0.01', '-t', '-p', '1', '-r', 'assemblies', '-d', '-theta', '0.004', '-l', 'lib', '-o', 'out'])
        assert (args.k, args.s, args.S, args.e, args.t, args.p, args.r, args.d, args.theta, args.l, args.o) == \
               (21, 500, 3, 0.01, True, 1, 'assemblies', True, 0.004, 'lib', 'out')

    def test_invalid_kmer_length(self, monkeypatch):
        with pytest.raises(SystemExit):
            parse(monkeypatch, ['reference', 'dir', '-k', '32'])

    def test_distance(self, monkeypatch):
        name, args = parse(monkeypatch, ['distance', 'lib', '-t', '-r', 'x.hist'])
        assert name == 'distance' and args.library == 'lib' and args.t and args.r == 'x.hist' and not args.d

    def test_query(self, monkeypatch):
        name, args = parse(monkeypatch, ['query', 'q.fastq', 'lib', '-a', '-o', 'res'])
        assert name == 'query' and (args.input, args.library, args.a, args.o) == ('q.fastq', 'lib', True, 'res')

    def test_subsample_defaults(self, monkeypatch):
        name, args = parse(monkeypatch, ['subsample', 'dir'])
        assert name == 'subsample' and (args.i, args.b, args.c, args.fa, args.msh, args.chunk) == (0, 100, 0.9, True, True, 10)

    def test_subsample_keep_flags(self, monkeypatch):
        _, args = parse(monkeypatch, ['subsample', 'dir', '-fa', '-msh', '-b', '2', '-i', '5', '-c', '0.8'])
        assert (args.fa, args.msh, args.b, args.i, args.c) == (False, False, 2, 5, 0.8)

    def test_subsample_chunk_is_limited_to_32(self, monkeypatch):
        with pytest.raises(SystemExit):
            parse(monkeypatch, ['subsample', 'dir', '-chunk', '33'])

    def test_correct(self, monkeypatch):
        name, args = parse(monkeypatch, ['correct', '-main', 'm.txt', '-sub', 'subs'])
        assert name == 'correction' and (args.main, args.sub) == ('m.txt', 'subs')

    def test_pop(self, monkeypatch):
        name, args = parse(monkeypatch, ['pop', 'm.txt', 'a.tsv', '-o', 'prefix'])
        assert name == 'pop' and (args.matrix, args.annotation, args.o) == ('m.txt', 'a.tsv', 'prefix')

    def test_test(self, monkeypatch):
        name, args = parse(monkeypatch, ['test', '--quick', '--keep'])
        assert name == 'selftest' and args.quick and args.keep

    def test_command_is_required(self, monkeypatch):
        with pytest.raises(SystemExit):
            parse(monkeypatch, [])

    def test_unknown_command(self, monkeypatch):
        with pytest.raises(SystemExit):
            parse(monkeypatch, ['frobnicate'])

def test_version_flag(monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        parse(monkeypatch, ['--version'])
    assert e.value.code == 0
    assert 'skmer ' + __version__ in capsys.readouterr().out

def test_exceptions_print_message_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli, 'reference', mock.Mock(side_effect=ValueError('bad input')))
    monkeypatch.setattr(sys, 'argv', ['skmer', 'reference', 'dir'])
    hook = sys.excepthook
    try:
        with pytest.raises(ValueError):
            cli.main()
        # the handler installed by main is called on an uncaught exception
        sys.excepthook(ValueError, ValueError('bad input'), None)
    finally:
        sys.excepthook = hook
    assert 'ValueError: bad input' in capsys.readouterr().out
