import os, tempfile
from contextlib import contextmanager

class KeyedMixin:
    """
    Interface for objects to be hashable, comparable, and sortable.
    Implementations need to define a `key` property that uniquely
    distinguishes objects from eachother.
    """

    def __lt__(self, other):
        return self.key < other.key

    def __gt__(self, other):
        return self.key > other.key

    def __le__(self, other):
        return self.key <= other.key

    def __ge__(self, other):
        return self.key >= other.key

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return self.key != other.key

    def __hash__(self):
        return hash(self.key)
