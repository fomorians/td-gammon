
import sys, random
from functools import partial
from contextlib import contextmanager

WHITE = 'W'
BLACK = 'B'


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


class Game(object):

    def __init__(I):
        I.board = Board()
        I.roll = Roll()
        I.color = WHITE if I.roll.d1 > I.roll.d2 else BLACK

    def play(I):
        while True:
            while I.roll.dies:
                print()
                print(I.board)
                print('Current roll for {}: {} {}'.format(I.color, I.roll, I.roll.dies))
                print('Possible moves:')
                can_move = False
                possible_points = [I.board.jail[I.color]]
                if not possible_points[0].pieces:
                    # No pieces are jailed, so consider entire board.
                    possible_points = [I.board.points[i] for i in range(26)]
                for point in possible_points:
                    if point.pieces and point.color == I.color:
                        moves = I.board.possible_moves(I.roll, point)
                        if moves:
                            can_move = True
                        print('  {} {{{}}}'.format(point, ','.join(str(i) for i in moves)))
                if not can_move:
                    print('No possible moves left')
                    break
                move = input('Next move? ')
                if move == 'stop':
                    sys.exit('Good-bye')
                try:
                    start, end = [int(i) for i in move.split()]
                    dies = abs(end - start)
                except:
                    print('Invalid move.  Use "<start-pos> <end-pos>".')
                    continue
                with freshmaker(onerror=freshmaker.PRINT) as undo:
                    I.roll.use(dies)
                    undo.add(partial(I.roll.unuse, dies))
                    I.board.move(start, end)
            I.roll = Roll()
            I.color = BLACK if I.color == WHITE else WHITE


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

    [ 12:W5 | 11    | 10    |  9    |  8:B3 |  7    ] [  6:B5 |  5    |  4    |  3    |  2    |  1:W2 ] [ 0 ]
    [ 13:B5 | 14    | 15    | 16    | 17:W3 | 18    ] [ 19:W5 | 20    | 21    | 22    | 23    | 24:B2 ] [ 0 ]    
    """

    def __init__(I):
        """
        Create board initialized for a new game.
        """
        I.points = [Point(i) for i in range(26)]
        white = [Piece(WHITE, i) for i in range(15)]
        black = [Piece(BLACK, i) for i in range(15)]
        for i in range(0,2):
            I.move(white[i], 1)
            I.move(black[i], 24)
        for i in range(2,7):
            I.move(white[i], 19)
            I.move(black[i], 6)
        for i in range(7,10):
            I.move(white[i], 17)
            I.move(black[i], 8)
        for i in range(10,15):
            I.move(white[i], 12)
            I.move(black[i], 13)
        I.jail = {WHITE: I.points[0], BLACK: I.points[25]}
        I.home = {BLACK: I.points[0], WHITE: I.points[25]}

    def __str__(I):
        l = []
        out = l.append
        out('[ ')
        for i in range(12, 0, -1):
            if i == 6:
                out(' ] [ ')
            out(str(I.points[i]))
            if i not in (7, 1):
                out(' | ')
        out(" ] [ {} ]\n[ ".format(len(I.jail[WHITE].pieces)))
        for i in range(13, 25):
            if i == 19:
                out(' ] [ ')
            out(str(I.points[i]))
            if i not in (18,24):
                out(' | ')
        out(" ] [ {} ]".format(len(I.jail[BLACK].pieces)))
        return ''.join(l)

    def move(I, piece, point):
        """
        Move piece to destination (point).
        """
        if isinstance(point, int):
            assert point >= 0 and point <= 26, 'valid points are [0..26]'
            point = I.points[point]
        if isinstance(piece, int):
            assert piece >= 0 and piece <= 26, 'valid points are [0..26]'
            piece = I.points[piece].pieces[0]
        assert not point.blocked(piece), 'cannot move to a blocked point'
        if piece.point is not None:
            piece.point.remove(piece)
        if point.pieces:
            other = point.pieces[0]
            if other.color != piece.color:
                I.move(other, I.jail[other.color])
        point.add(piece)

    def possible_moves(I, roll, point):
        """
        Return list of available Points to move to, accounting for all
        combinations of unused dies.
        """
        if isinstance(point, int):
            assert point >= 0 and point <= 26, 'valid points are [0..26]'
            point = I.points[point]
        assert point.pieces, 'there are no pieces on this point'
        piece = point.pieces[0]
        direction = 1 if piece.color == WHITE else -1
        dies = roll.unused()
        if not dies:
            return []
        if len(dies) == 1:
            paths = [[dies[0]]]
        elif dies[0] == dies[1]:
            paths = [len(dies) * [dies[0]]]
        else:
            paths = [(dies[0], dies[1]), (dies[1], dies[0])]
        multiple_jailed = len(I.jail[piece.color].pieces) > 1
        moves = []
        min_point = 1
        max_point = 24
        if I.can_go_home(piece.color):
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
                if num < min_point or num > max_point or I.points[num].blocked(piece):
                    break
                if num not in moves:
                    moves.append(num)
        return sorted(moves)

    def can_go_home(I, color):
        """
        True if there are no pieces outside given color's home quadrant.
        """
        points = range(7, 26) if color == BLACK else range(19)
        for point in points:
            if color == I.points[point].color:
                return False
        return True
        

class Point(object):
    """
    A Point represents a position on the Board.  It contains zero or
    more Pieces.
    """

    def __init__(I, num):
        I.pieces = []
        I.num = num

    def __str__(I):
        s = "{:2d}".format(I.num)
        if I.pieces:
            s += ":{}{}".format(I.color, len(I.pieces))
        else:
            s += '   '
        return s
    __repr__ = __str__

    def add(I, piece):
        if piece not in I.pieces:
            I.pieces.append(piece)
            piece.point = I
            assert set(i.color for i in I.pieces) == set([piece.color]), \
                'only pieces of same color allowed in a point'

    def remove(I, piece):
        if piece in I.pieces:
            I.pieces.remove(piece)

    def blocked(I, piece):
        """
        True if this Point contains more than one opposite Piece.
        """
        return len(I.pieces) > 1 and piece.color != I.color

    @property
    def color(I):
        return I.pieces[0].color if I.pieces else None


class Piece(object):
    """
    A game piece that knows which Point on the Board it currently is.
    """

    def __init__(I, color, num):
        assert num >= 0 and num <= 15, \
            "number out of range [0,15]: {}".format(num)
        assert color in (WHITE, BLACK), \
            "color must be '{}' or '{}': {}".format(WHITE, BLACK, color)
        I.color = color
        I.num = num
        I.point = None

    def __str__(I):
        return "{}:{}".format(I.color, I.num)
    __repr__ = __str__

    def __hash__(I):
        return ord(I.color) + I.num


class Roll(object):
    """
    A Roll that can track which moves have been used.
    """

    def __init__(I, d1=None, d2=None):
        if d1 is None:
            d1 = random.choice(range(1, 7))
        if d2 is None:
            d2 = random.choice(range(1, 7))
        assert d1 >= 1 and d1 <= 6, "invalid roll: {}".format(d1)
        assert d2 >= 1 and d2 <= 6, "invalid roll: {}".format(d2)
        # Preserve original roll.
        I.d1, I.d2 = d1, d2
        # Capture number of unused dies/moves.
        I.dies = []
        if d1 != d2:
            I.dies = [d1, d2]
        else:
            I.dies = [d1,d1,d1,d1]

    def __str__(I):
        return "{}x{}".format(I.d1, I.d2)

    def __hash__(I):
        return (10 * I.d1) + I.d2

    def use(I, move):
        """
        Mark die(s) as used to satisfy given move.
        """
        if move in I.dies:
            # NOTE: list.remove() will only remove one matching entry,
            # which works out well for us since we don't want to
            # remove multiple dies when doubles are rolled.
            I.dies.remove(move)
        else:
            orig = I.dies.copy()
            while I.dies and move >= max(I.dies):
                # Consume dies until move is satisfied.  We don't care
                # about the order since will either be doubles or both
                # dies will be needed when not doubles.
                move -= I.dies.pop()
            if move != 0:
                I.dies = orig
                raise ValueError('impossible move')

    def unuse(I, move):
        """
        Mark dies as unused for given move - useful for undo or automated tests.
        """
        # assert die in (I.d1, I.d2), 'die not part of this roll'
        if move in (I.d1, I.d2):
            if move == I.d2:
                I.dies.append(move)
            else:
                I.dies.insert(0, move)
        else:
            # Whether doubles or not, will need at least two dies to satisfy this move.
            I.dies.extend([I.d1, I.d2])
            move -= I.d1 + I.d2
            while move > 0:
                # Should only get here when there are doubles.  Unuse
                # a die until the move is satisfied
                I.dies.append(I.d1)
                move -= I.d1
            if move != 0 or len(I.dies) > 4:
                raise ValueError('impossible to unuse')

    def unused(I):
        """
        Return list of unused dies.
        """
        return tuple(I.dies)


if __name__ == '__main__':
    Game().play()
