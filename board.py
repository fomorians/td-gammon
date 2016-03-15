import operator

class Board(object):
    """
    A Board has 24 Points with at most 15 Pieces in play at a time.
    Each Point can have 0 or more Pieces.

    White's home is lower-right and can only move up the array.
    Black's home is upper-right and can only move down the array.

    Points are positioned in the array like this:

    [ 12 | 11 | 10 |  9 |  8 |  7 ][  6 |  5 |  4 |  3 |  2 |  1 ] [ white jail / black home ]
    [ 13 | 14 | 15 | 16 | 17 | 18 ][ 19 | 20 | 21 | 22 | 23 | 24 ] [ black jail / white home ]

    Point 0 is both the jail for white and the home for black.  Point
    25 is both the jail for black and home for white.  A
    representation of a new Board:

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
                self.points[pt].push(Piece(WHITE, num))
                num += 1
        num = 0
        for pt, count in ((24,2), (13,5), (8,3), (6,5)):
            for i in range(count):
                self.points[pt].push(Piece(BLACK, num))
                num += 1

    @staticmethod
    def from_str(s):
        """
        Return a new Board from string representation of a Board.
        """
        is_digit = ('0','1','2','3','4','5','6','7','8','9').__contains__
        brd = Board()
        for pt in brd.points:
            while pt.pieces:
                pt.pop()
        counts = {WHITE: 0, BLACK: 0}
        for line in s.split('\n'):
            for i in line.split():
                if is_digit(i[0]):
                    l = i.split(':')
                    if len(l) > 1:
                        pt = int(l[0])
                        for pieces in l[1:]:
                            color = pieces[0]
                            count = int(pieces[1:])
                            for j in range(count):
                                brd.points[pt].push(Piece(color, counts[color]))
                                counts[color] += 1
        return brd

    def __str__(self):
        l = []
        out = l.append
        out('[ ')
        for i in range(12, 0, -1):
            if i == 6:
                out(' ] [ ')
            out(str(self.points[i]))
            if i not in (7, 1):
                out(' | ')
        homed = len(self.homed(BLACK))
        jailed = len(self.jailed(WHITE))
        out(" ] [  0:W{}:B{} ]\n[ ".format(jailed, homed))
        for i in range(13, 25):
            if i == 19:
                out(' ] [ ')
            out(str(self.points[i]))
            if i not in (18,24):
                out(' | ')
        homed = len(self.homed(WHITE))
        jailed = len(self.jailed(BLACK))
        out(" ] [ 25:B{}:W{} ]".format(jailed, homed))
        return ''.join(l)

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
        new = self.copy()
        if not isinstance(dst, int):
            dst = dst.num
        assert dst >= 0 and dst <= 25, 'valid points are [0..25]'
        dst = new.points[dst]
        if not isinstance(src, int):
            src = src.num
        assert src >= 0 and src <= 25, 'valid points are [0..25]'
        src = new.points[src]
        sharing_allowed = dst in (new.home(WHITE), new.home(BLACK))
        if not sharing_allowed:
            assert not dst.blocked(src.color), 'cannot move to a blocked point'
        if dst.pieces and src.color != dst.color and not sharing_allowed:
            # Move exposed piece to jail.
            new.jail(dst.color).push(dst.pop())
        dst.push(src.pop())
        return new

    def possible_moves(self, roll, point):
        """
        Return list of available Points to move to, accounting for all
        combinations of unused dies.
        """
        if isinstance(point, int):
            assert point >= 0 and point <= 25, 'valid points are [0..25]'
            point = self.points[point]
        assert point.pieces, 'there are no pieces on this point'
        piece = point.pieces[0]
        direction = 1 if piece.color == WHITE else -1
        dies = roll.dies
        if not dies:
            return []
        if len(dies) == 1:
            paths = [[dies[0]]]
        elif dies[0] == dies[1]:
            paths = [len(dies) * [dies[0]]]
        else:
            paths = [(dies[0], dies[1]), (dies[1], dies[0])]
        multiple_jailed = len(self.jailed(piece.color)) > 1
        moves = []
        min_point = 1
        max_point = 24
        if self.can_go_home(piece.color):
            if piece.color == BLACK:
                min_point -= 1
            else:
                max_point += 1
        for hops in paths:
            if multiple_jailed:
                hops = hops[:1]
            num = point.num
            for hop in hops:
                num += direction * hop
                if num < min_point or num > max_point or self.points[num].blocked(piece.color):
                    break
                if num not in moves:
                    moves.append(num)
        return sorted(moves)

    def can_go_home(self, color):
        """
        True if there are no pieces outside given color's home quadrant.
        """
        points = range(7, 26) if color == BLACK else range(19)
        for point in points:
            if color == self.points[point].color:
                return False
        return True

    def finished(self):
        """
        True once all pieces for a color are home.
        """
        return 15 == len([i for i in self.home(WHITE).pieces if i.color == WHITE]) or \
            15 == len([i for i in self.home(BLACK).pieces if i.color == BLACK])

    def jail(self, color):
        """
        Return Point that represents jail for given color.
        """
        return self.points[0 if color == WHITE else 25]

    def jailed(self, color):
        """
        List of pieces in jail for given color.
        """
        return tuple(i for i in self.jail(color).pieces if i.color == color)

    def home(self, color):
        """
        Return Point that represents home for given color.
        """
        return self.points[0 if color == BLACK else 25]

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
        if color == WHITE:
            enemy = BLACK
            behind = operator.gt
            enemy_line = self.points[max(i for i in range(25, 1, -1) if self.points[i].color == enemy)]
        else:
            enemy = WHITE
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
