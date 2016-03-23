from player import Player

class Point(object):
    """
    A Point represents a position on the Board.
    """

    def __init__(self, num):
        self._pieces = ()
        self.num = num

    def __lt__(self, other):
        return self.num < other.num

    def __gt__(self, other):
        return self.num > other.num

    def __le__(self, other):
        return self.num <= other.num

    def __ge__(self, other):
        return self.num >= other.num

    def __eq__(self, other):
        return self.num == other.num

    def __ne__(self, other):
        return self.num != other.num

    def __hash__(self):
        return hash(self.num)

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
        color = 'N/A'
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
        if piece in self.pieces:
            return

        self._pieces += (piece,)

        if self.num not in (0, 25): # Making exception for jail/home.
            a = str(set(i.color for i in self.pieces))
            b = str(set([piece.color]))
            assert a == b, "only pieces of same color allowed in a point {0} {1}".format(a, b)

    def pop(self, color):
        """
        Remove top Piece and return it.
        """
        assert self.pieces, 'no pieces at this point'
        assert color, 'no specified color'

        index = 0
        for piece in self.pieces:
            if piece.color == color:
                self._pieces = self.pieces[:index] + self.pieces[(index+1):]
                return piece
            index += 1

        assert False, 'no pieces of this color {0} at {1}'.format(str(color), str(self.num))

    def blocked(self, color):
        """
        True if this Point contains more than one opposing Piece,
        excluding home/jail since they should never be blocked.
        """
        return self.num not in (0, 25) and color != self.color and len(self.pieces) > 1

    def count(self, color=None):
        if color is not None:
            return len(list(piece for piece in self.pieces if piece.color == color))
        else:
            return len(self.pieces)

    @property
    def pieces(self):
        """
        The read-only collection of Pieces that are at this Point.
        Points can be modified via push() & pop().
        """
        return self._pieces

    @property
    def color(self):
        """
        None if there are no Pieces here.  Otherwise, the color of one
        of the Pieces here.  The special jail & home Points are
        treated specially in that only the jail's color is ever considered.
        """
        val = None

        if not self.pieces:
            return val

        colors = set(i.color for i in self.pieces)

        if self.num == 0:
            if Player.WHITE in colors:
                val = Player.WHITE
        elif self.num == 25:
            if Player.BLACK in colors:
                val = Player.BLACK
        elif len(colors) == 1:
            val = self.pieces[0].color
        else:
            raise ValueError("multiple colors occupy same point: {}".format(self))

        return val
