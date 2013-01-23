
from .model import Board, Roll, WHITE, BLACK


NEW_BOARD_MOVES = { # All possible moves for all possible rolls from these points.
    1: {
        11: [2, 3, 4, 5], 12: [2, 3, 4], 13: [2, 4, 5], 14: [2, 5], 15: [2, 7], 16: [2, 7],
        21: [2, 3, 4], 22: [3, 5, 7, 9], 23: [3, 4], 24: [3, 5, 7], 25: [3], 26: [3, 7, 9],
        31: [2, 4, 5], 32: [3, 4], 33: [4, 7, 10], 34: [4, 5], 35: [4, 9], 36: [4, 7, 10],
        41: [2, 5], 42: [3, 5, 7], 43: [4, 5], 44: [5, 9], 45: [5, 10], 46: [5, 7, 11],
        51: [2, 7], 52: [3], 53: [4, 9], 54: [5, 10], 55: [], 56: [7, 12],
        61: [2, 7], 62: [3, 7, 9], 63: [4, 7, 10], 64: [5, 7, 11], 65: [7, 12], 66: [7],
    },
    24: {
        11: [20, 21, 22, 23], 12: [21, 22, 23], 13: [20, 21, 23], 14: [20, 23], 15: [18, 23], 16: [18, 23],
        21: [21, 22, 23], 22: [16, 18, 20, 22], 23: [21, 22], 24: [18, 20, 22], 25: [22], 26: [16, 18, 22],
        31: [20, 21, 23], 32: [21, 22], 33: [15, 18, 21], 34: [20, 21], 35: [16, 21], 36: [15, 18, 21],
        41: [20, 23], 42: [18, 20, 22], 43: [20, 21], 44: [16, 20], 45: [15, 20], 46: [14, 18, 20],
        51: [18, 23], 52: [22], 53: [16, 21], 54: [15, 20], 55: [], 56: [13, 18],
        61: [18, 23], 62: [16, 18, 22], 63: [15, 18, 21], 64: [14, 18, 20], 65: [13, 18], 66: [18],
    },
}
NEW_BOARD_MOVES_AFTER_1_DIE = { # All possible moves for all possible rolls after using first die.
    1: {
        11: [3, 4, 5], 12: [4], 13: [5], 14: [], 15: [7], 16: [],
        21: [4], 22: [5, 7, 9], 23: [], 24: [7], 25: [], 26: [9],
        31: [5], 32: [], 33: [7, 10], 34: [], 35: [9], 36: [10],
        41: [], 42: [7], 43: [], 44: [9], 45: [10], 46: [11],
        51: [], 52: [], 53: [], 54: [], 55: [], 56: [],
        61: [], 62: [9], 63: [10], 64: [11], 65: [12], 66: [],
    },
    24: {
        11: [20, 21, 22], 12: [21], 13: [20], 14: [], 15: [18], 16: [],
        21: [21], 22: [16, 18, 20], 23: [], 24: [18], 25: [], 26: [16],
        31: [20], 32: [], 33: [15, 18], 34: [], 35: [16], 36: [15],
        41: [], 42: [18], 43: [], 44: [16], 45: [15], 46: [14],
        51: [], 52: [], 53: [], 54: [], 55: [], 56: [],
        61: [], 62: [16], 63: [15], 64: [14], 65: [13], 66: [],
    },
}

def _moves(brd, roll, point, expected):
    'Assert possible moves for given board, roll, & point equals expected moves.'
    print(brd)
    print(roll, point, expected, brd.possible_moves(roll, point))
    assert expected == brd.possible_moves(roll, point)

def test_new_board_possible_moves():
    brd = Board()
    for point in (1, 24):
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                roll = Roll(d1, d2)
                expected = NEW_BOARD_MOVES[point][hash(roll)]
                _moves.description = "new board - point {:2d} - roll {}".format(point, roll)
                yield _moves, brd, roll, point, expected

def test_new_board_possible_moves_after_first_die():
    brd = Board()
    for point in (1, 24):
        for d1 in range(1, 7):
            for d2 in range(1, 7):
                roll = Roll(d1, d2)
                expected = NEW_BOARD_MOVES_AFTER_1_DIE[point][hash(roll)]
                _moves.description = "new board - point {} - roll {} - first die used".format(point, roll)
                piece = brd.points[point].pieces[0]
                if roll.d1 != roll.d2:
                    # Trick possible_moves() call below into thinking only the first die is available.
                    roll.use(roll.d2)
                moves = brd.possible_moves(roll, point)
                if not moves:
                    first_move = point
                else:
                    # Use the first die, moving the the first possible move.
                    if piece.color == WHITE:
                        first_move = moves[0] # White moves up.
                    else:
                        first_move = moves[-1] # Black moves down.
                    brd.move(piece, first_move)
                    roll.use(roll.d1)
                    if roll.d1 != roll.d2:
                        # Make it so the second die is available now.
                        roll.unuse(roll.d2)
                yield _moves, brd, roll, first_move, expected
                # Move back to original point to test other rolls.
                brd.move(piece, point)

def test_possible_moves_home1():
    'Home considered a possible move once all pieces in home quadrant.'
    brd = Board()
    # Arbitrarily move all black pieces to home quadrant.
    for i in range(2):
        brd.move(24, 2)
    for i in range(5):
        brd.move(13, 5)
    for i in range(3):
        brd.move(8, 4)
    print(brd)
    roll = Roll(5, 3)
    expected = {1: [], 4: [], 5: [0,2], 6: [3]}
    for point in (1, 4, 5, 6):
        _moves.description = "black is home - point {} - roll {}".format(point, roll)
        yield _moves, brd, roll, point, expected[point]


def test_jail1():
    'Moving a piece to a point with only one opposite piece sends him to jail.'
    brd = Board()
    white1 = brd.points[1].pieces[0]
    white2 = brd.points[1].pieces[1]
    black = brd.points[6].pieces[0]
    brd.move(white1, 3)
    assert not brd.jail[WHITE].pieces, 'jail should be empty'
    brd.move(black, 3)
    assert [white1] == brd.jail[WHITE].pieces, 'not in jail'
    brd.move(black, 1)
    assert [white1, white2] == brd.jail[WHITE].pieces, 'not in jail'
    assert [3] == brd.possible_moves(Roll(6, 3), 0)

def test_jail2():
    'When multiple pieces jailed, possible moves should be limited to exit points only.'
    brd = Board()
    for i in range(2):
        brd.move(brd.points[1].pieces[0], 0)
    assert len(brd.jail[WHITE].pieces) == 2
    # Should only be allowed to move to position 5, not 11 (5+6)
    assert [5] == brd.possible_moves(Roll(6, 5), 0)


def _use(roll, move, expected):
    'Assert that using given move produces correct unused dies.'
    roll.use(move)
    print(roll, move, expected, roll.unused())
    assert expected == roll.unused()

def test_use1():
    'Can use a single die or multiple dies at once.'
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            # Test all combinations when using the first die.
            roll = Roll(d1, d2)
            _use.description = "roll {} - used {}".format(roll, d1)
            if d1 != d2:
                expected = (d2,)
            else:
                expected = (d2,d2,d2)
            yield _use, roll, d1, expected
            # Test all combinations when using the first two dies.
            roll = Roll(d1, d2)
            if d1 != d2:
                expected = ()
            else:
                expected = (d2,d2)
            _use.description = "roll {} - used {}".format(roll, d1 + d2)
            yield _use, roll, d1 + d2, expected

def test_use3():
    'Roll.use() raises exception when given invalid move.'
    roll = Roll(4, 6)
    msg = None
    try:
        roll.use(3)
    except Exception as e:
        msg = str(e)
    print('MSG:', msg)
    assert msg == 'impossible move'

def test_use4():
    'Roll.use() is read-only if given invalid move.'
    roll = Roll(4, 6)
    msg = None
    try:
        roll.use(13)
    except Exception as e:
        msg = str(e)
    print('MSG:', msg)
    assert msg == 'impossible move'
    print('UNUSED:', roll.unused())
    assert (4,6) == roll.unused()


def test_unuse1():
    'Can Roll.unuse() a single die.'
    roll = Roll(4, 6)
    roll.use(4)
    assert roll.unused() == (6,)
    roll.unuse(4)
    assert roll.unused() == (4, 6)

def test_unuse2():
    'Can Roll.unuse() multiple dies at once.'
    roll = Roll(4, 6)
    roll.use(10)
    assert roll.unused() == ()
    roll.unuse(10)
    assert roll.unused() == (4, 6)

def test_unuse3():
    'Can Roll.unuse() doubles.'
    roll = Roll(4, 4)
    roll.use(12)
    assert roll.unused() == (4,)
    roll.unuse(12)
    assert roll.unused() == (4, 4, 4, 4)

def test_unuse4():
    'Roll.unuse() raises exception when given invalid move.'
    roll = Roll(5, 5)
    roll.use(15)
    msg = None
    try:
        roll.unuse(20)
    except Exception as e:
        msg = str(e)
    print('MSG:', msg)
    assert msg == 'impossible to unuse'
