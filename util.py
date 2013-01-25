"""
Utility definitions for the py-gammon package.
"""
__all__ = ['tmp_path', 'freshmaker']

import os, tempfile
from contextlib import contextmanager

@contextmanager
def tmp_path():
    """
    Yield a path to a temporary file and remove it afterwards.
    """
    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        yield path
    except:
        raise
    finally:
        if os.path.exists(path):
            os.remove(path)


@contextmanager
def freshmaker(onerror=None):
    """
    A simple way to register undo methods in case an unhandled
    exception is raised in this block.
    """
    l = set()
    try:
        yield l
    except Exception as e:
        for f in l:
            f()
        if onerror == freshmaker.PRINT:
            print(e)
        else:
            raise e
freshmaker.PRINT = 'print'
freshmaker.RAISE = 'raise'
