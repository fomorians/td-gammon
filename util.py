"""
Utility definitions for the py-gammon package.
"""
__all__ = ['tmp_path', 'freshmaker']

import os, tempfile
from contextlib import contextmanager


class KeyedMixin:
    """
    Interface for objects to be hashable, comparable, and sortable.
    Implementations need to define a `key` property that uniquely
    distinguishes objects from eachother.
    """

    def __lt__(I, other):
        return I.key < other.key

    def __gt__(I, other):
        return I.key > other.key

    def __le__(I, other):
        return I.key <= other.key

    def __ge__(I, other):
        return I.key >= other.key

    def __eq__(I, other):
        return I.key == other.key

    def __ne__(I, other):
        return I.key != other.key

    def __hash__(I):
        return hash(I.key)


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
