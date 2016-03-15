import random

class Roll(object):
    """
    A Roll of two dies.
    """
    @property
    def dies(self):
        'Collection of unused dies.'
        return self._dies

    def __init__(self, d1=None, d2=None):
        if d1 is None:
            d1 = random.choice(range(1, 7))
        if d2 is None:
            d2 = random.choice(range(1, 7))
        assert d1 >= 1 and d1 <= 6, "invalid roll: {}".format(d1)
        assert d2 >= 1 and d2 <= 6, "invalid roll: {}".format(d2)
        # Preserve original roll.
        self.d1, self.d2 = d1, d2
        # Capture number of unused dies/moves.
        if d1 != d2:
            self._dies = (d1, d2)
        else:
            self._dies = (d1,d1,d1,d1)

    def __repr__(self):
        return "{}x{}".format(self.d1, self.d2)

    def __hash__(self):
        return (10 * self.d1) + self.d2

    def __eq__(self, other):
        return self.d1 == other.d1 and self.d2 == other.d2

    @staticmethod
    def from_str(s):
        return Roll(*[int(i) for i in s.split('x')])

    def copy(self):
        """
        Return a deep copy of this Roll.
        """
        new = Roll(self.d1, self.d2)
        new._dies = tuple(self.dies)
        return new

    def use(self, move):
        """
        Mark die(s) as used to satisfy given move.
        """
        working = list(self.dies)
        if move in working:
            # NOTE: list.remove() will only remove one matching entry,
            # which works out well for us since we don't want to
            # remove multiple dies when doubles are rolled.
            working.remove(move)
        else:
            while working and move >= max(working):
                # Consume dies until move is satisfied.  We don't care
                # about the order since will either be doubles or both
                # dies will be needed when not doubles.
                move -= working.pop()
            if move != 0:
                raise ValueError('impossible move')
        self._dies = tuple(working)

    def unuse(self, move):
        """
        Mark dies as unused for given move - useful for undo or automated tests.
        """
        # assert die in (self.d1, self.d2), 'die not part of this roll'
        working = list(self.dies)
        if move in (self.d1, self.d2):
            if move == self.d2:
                working.append(move)
            else:
                working.insert(0, move)
        else:
            # Whether doubles or not, will need at least two dies to satisfy this move.
            working.extend([self.d1, self.d2])
            move -= self.d1 + self.d2
            while move > 0:
                # Should only get here when there are doubles.  Unuse
                # a die until the move is satisfied
                working.append(self.d1)
                move -= self.d1
            if move != 0 or len(working) > 4:
                raise ValueError('impossible to unuse')
        self._dies = tuple(working)
