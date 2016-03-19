import numpy as np
import operator

from player import Player
from point import Point
from piece import Piece

class Board(object):
    """
    A Board has 24 Points with at most 15 Pieces in play at a time.
    Each Point can have 0 or more Pieces.

    White's home is lower-right and can only move up the array.
    Black's home is upper-right and can only move down the array.

    Points are positioned in the array like this:

    [ 12 | 11 | 10 |  9 |  8 |  7 ][  6 |  5 |  4 |  3 |  2 |  1 ] [ white jail / black home ]
    [ 13 | 14 | 15 | 16 | 17 | 18 ][ 19 | 20 | 21 | 22 | 23 | 24 ] [ black jail / white home ]

    Point 0 is both the jail for white and the home for black.
    Point 25 is both the jail for black and home for white.
    A representation of a new Board:

    [ 12:W5 | 11    | 10    |  9    |  8:B3 |  7    ] [  6:B5 |  5    |  4    |  3    |  2    |  1:W2 ] [  0:W0:B0 ]
    [ 13:B5 | 14    | 15    | 16    | 17:W3 | 18    ] [ 19:W5 | 20    | 21    | 22    | 23    | 24:B2 ] [ 25:B0:W0 ]
    """

    @property
    def points(self):
        """
        The read-only collection of Point instances that make up a
        Board.  While this collection is read-only, the Point
        instances are not since Pieces can be pushed & popped.
        """
        return self._points

    def __init__(self):
        """
        Create board initialized for a new game.
        """
        self._points = tuple(Point(i) for i in range(26))
        num = 0
        for pt, count in ((1,2), (12,5), (17,3), (19,5)):
            for i in range(count):
                self.points[pt].push(Piece(Player.WHITE, num))
                num += 1
        num = 0
        for pt, count in ((24,2), (13,5), (8,3), (6,5)):
            for i in range(count):
                self.points[pt].push(Piece(Player.BLACK, num))
                num += 1

    def last_checkers_position(self, color):
        positions = range(0, 25) if color == Player.WHITE else range(25, 0, -1)
        for position in positions:
            if self.points[position].color == color:
                return position
        return 25 if color == Player.WHITE else 0

    def __str__(self):
        arr = []
        arr.append('[ ')
        for i in range(12, 0, -1):
            if i == 6:
                arr.append(' ] [ ')
            arr.append(str(self.points[i]))
            if i not in (7, 1):
                arr.append(' | ')
        homed = len(self.homed(Player.BLACK))
        jailed = len(self.jailed(Player.WHITE))
        arr.append(" ] [  0:W{}:B{} ]\n[ ".format(jailed, homed))
        for i in range(13, 25):
            if i == 19:
                arr.append(' ] [ ')
            arr.append(str(self.points[i]))
            if i not in (18, 24):
                arr.append(' | ')
        homed = len(self.homed(Player.WHITE))
        jailed = len(self.jailed(Player.BLACK))
        arr.append(" ] [ 25:B{}:W{} ]".format(jailed, homed))
        return ''.join(arr)

    def to_array(self):
        num_points = len(self.points) + 2

        pieces = np.zeros(num_points, dtype='uint')
        colors = np.zeros(num_points, dtype='uint')
        i = 0
        for point in self.points:
            # expand home/jail into separate rows
            if point.num in (0, 25):
                pieces[i] = point.count(Player.BLACK)
                colors[i] = 0
                i += 1

                pieces[i] = point.count(Player.WHITE)
                colors[i] = 1
                i += 1
            else:
                pieces[i] = point.count()
                colors[i] = 1 if point.color == Player.WHITE else 0
                i += 1

        # one-hot encoding
        max_pieces = 15
        ones = np.zeros((num_points, max_pieces + 1)) # +1 to include "empty"
        ones[np.arange(num_points), pieces] = 1

        ones_colors = np.zeros((num_points, max_pieces + 2)) # +1 for empty, +1 for color
        ones_colors[:,0] = colors
        ones_colors[:,1:] = ones

        return ones_colors.reshape(1, ones_colors.size)

    def copy(self):
        """
        Return a deep copy of this Board.
        """
        new = Board()
        new._points = tuple(pt.copy() for pt in self.points)
        return new

    def move(self, src, dst):
        """
        Return new Board with a piece from src moved to dst.
          * src: a Point instance or position
          * dts: a Point instance or position
        """
        board = self.copy()

        if dst < 0:
            dst = 0
        elif dst > 25:
            dst = 25

        assert src >= 0 and src <= 25, 'valid points are [0..25]'
        assert dst >= 0 and dst <= 25, 'valid points are [0..25]'

        dst = board.points[dst]
        src = board.points[src]

        sharing_allowed = dst in (board.home(Player.WHITE), board.home(Player.BLACK))
        if not sharing_allowed:
            assert not dst.blocked(src.color), 'cannot move to a blocked point'

        color = Player.WHITE if dst.num > src.num else Player.BLACK
        if dst.pieces and src.color != dst.color and not sharing_allowed:
            # Move exposed piece to jail.
            board.jail(dst.color).push(dst.pop(Player.WHITE if color == Player.BLACK else Player.BLACK))

        dst.push(src.pop(color))
        return board

    def can_go_home(self, color):
        """
        True if there are no pieces outside given color's home quadrant.
        """
        points = range(7, 26) if color == Player.BLACK else range(19)
        for point in points:
            if color == self.points[point].color:
                return False
        return True

    def finished(self):
        """
        True once all pieces for a color are home.
        """
        return len(self.homed(Player.WHITE)) == 15 \
            or len(self.homed(Player.BLACK)) == 15

    def jail(self, color):
        """
        Return Point that represents jail for given color.
        """
        return self.points[0 if color == Player.WHITE else 25]

    def jailed(self, color):
        """
        List of pieces in jail for given color.
        """
        return tuple(i for i in self.jail(color).pieces if i.color == color)

    def home(self, color):
        """
        Return Point that represents home for given color.
        """
        return self.points[0 if color == Player.BLACK else 25]

    def homed(self, color):
        """
        List of pieces that made it home.
        """
        return tuple(i for i in self.home(color).pieces if i.color == color)

    def strongholds(self, color):
        """
        List of points with two or more pieces for given color.
        """
        return [pt for pt in self.points if pt.color == color and len(pt.pieces) > 1]

    def safe(self, color):
        """
        List of points past last enemy piece.
        """
        if color == Player.WHITE:
            enemy = Player.BLACK
            behind = operator.gt
            enemy_line = self.points[max(i for i in range(25, 1, -1) if self.points[i].color == enemy)]
        else:
            enemy = Player.WHITE
            behind = operator.lt
            enemy_line = self.points[min(i for i in range(0, 24, 1) if self.points[i].color == enemy)]
        return [pt for pt in self.points if behind(pt, enemy_line) and pt.pieces]

    def exposed(self, color):
        """
        List of points for given color that contain 1 piece that are not safe.
        """
        safe = self.safe(color)
        jail = self.jail(color)
        return [pt for pt in self.points if pt.color == color and len(pt.pieces) == 1 and pt not in safe and pt != jail]
