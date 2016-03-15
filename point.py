from keyed_mixin import KeyedMixin

class Point(KeyedMixin, object):
    """
    A Point represents a position on the Board.
    """

    @property
    def pieces(self):
        """
        The read-only collection of Pieces that are at this Point.
        Points can be modified via push() & pop().
        """
        return self._pieces

    def __init__(self, num):
        self._pieces = ()
        self.num = num
        self.key = num # Satisfy KeyedMixin

    def __str__(self):
        """
        A version of repr(Point) with whitespace to make str(Board) nicer.
        """
        s = "{:2d}".format(self.num)
        if self.pieces:
            s += ":{}{}".format(self.color, len(self.pieces))
        else:
            s += '   '
        return s

    def __repr__(self):
        color = 'NA'
        if self.pieces:
            color = "{}{}".format(self.color, len(self.pieces))
        return "{}:{}".format(self.num, color)

    def copy(self):
        """
        Return a deep copy of this Point.
        """
        new = Point(self.num)
        new._pieces = tuple(p.copy() for p in self.pieces)
        return new

    def push(self, piece):
        """
        Add given Piece to this Point.
        """
        if piece not in self.pieces:
            self._pieces += (piece,)
            if self.num not in (0,25): # Making exception for jail/home.
                assert set(i.color for i in self.pieces) == set([piece.color]), \
                    'only pieces of same color allowed in a point'

    def pop(self):
        """
        Remove top Piece and return it.
        """
        assert self.pieces, 'no pieces at this point'
        piece = self.pieces[-1]
        self._pieces = self.pieces[:-1]
        return piece

    def blocked(self, color):
        """
        True if this Point contains more than one opposing Piece,
        excluding home/jail since they should never be blocked.
        """
        return self.num not in (0,25) and color != self.color and len(self.pieces) > 1

    @property
    def color(self):
        """
        None if there are no Pieces here.  Otherwise, the color of one
        of the Pieces here.  The special jail & home Points are
        treated specially in that only the jail's color is ever considered.
        """
        val = None
        if self.pieces:
            colors = set(i.color for i in self.pieces)
            if self.num == 0:
                if WHITE in colors:
                    val = WHITE
            elif self.num == 25:
                if BLACK in colors:
                    val = BLACK
            elif len(colors) == 1:
                val = self.pieces[0].color
            else:
                raise ValueError("multiple colors occupy same point: {}".format(self))
        return val
