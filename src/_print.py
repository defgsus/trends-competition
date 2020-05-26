import sys


def printe(*args, **kwargs):
    """Print to stderr and flush"""
    print(*args, **kwargs, file=sys.stderr)
    sys.stderr.flush()
