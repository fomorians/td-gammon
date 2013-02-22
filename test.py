"""
Automated tests for the py-gammon package.  Intended to be ran with
the nosetests utility.
"""

import json
from .util import tmp_path
from .model import Board, Roll, Turn, WHITE, BLACK
from .game import Game
from . import strategy


def equals(expected, actual, msg=None):
    """
    Wrap `assert` to provide better msg when fails.
    """
    if msg is None:
         msg = "{} != {}".format(expected, actual)
    assert expected == actual, msg

def gt(val1, val2, msg=None):
    """
    Wrap `assert` to provide better msg when fails.
    """
    if msg is None:
         msg = "{} <= {}".format(val1, val2)
    assert val1 > val2, msg


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
    equals(expected, brd.possible_moves(roll, point))

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
                    roll.use(roll.d1)
                    if roll.d1 != roll.d2:
                        # Make it so the second die is available now.
                        roll.unuse(roll.d2)
                yield _moves, brd.move(point, first_move), roll, first_move, expected

def test_possible_moves_home1():
    'Home considered a possible move once all pieces in home quadrant.'
    brd = Board()
    # Arbitrarily move all black pieces to home quadrant.
    for i in range(2):
        brd = brd.move(24, 2)
    for i in range(5):
        brd = brd.move(13, 5)
    for i in range(3):
        brd = brd.move(8, 4)
    print(brd)
    roll = Roll(5, 3)
    expected = {1: [], 4: [], 5: [0,2], 6: [3]}
    for point in (1, 4, 5, 6):
        _moves.description = "black is home - point {} - roll {}".format(point, roll)
        yield _moves, brd, roll, point, expected[point]


def test_jail1():
    'Moving a piece to a point with only one opposite piece sends him to jail.'
    brd = Board()
    brd = brd.move(1, 3)
    assert not brd.jail(WHITE).pieces, 'jail should be empty'
    brd = brd.move(6, 3)
    equals('(W:1,)', str(brd.jail(WHITE).pieces))
    brd = brd.move(3, 1)
    equals('(W:1, W:0)', str(brd.jail(WHITE).pieces))
    equals([3], brd.possible_moves(Roll(6, 3), 0))

def test_jail2():
    'When multiple pieces jailed, possible moves should be limited to exit points only.'
    brd = Board()
    for i in range(2):
        brd = brd.move(1, 0)
    equals(2, len(brd.jail(WHITE).pieces))
    # Should only be allowed to move to position 5, not 11 (5+6)
    equals([5], brd.possible_moves(Roll(6, 5), 0))

def test_jail3():
    'The color of the shared home/jail point is considered the jailed color.'
    brd = Board()
    equals(None, brd.points[0].color) # No pieces at a point has no color.
    equals(None, brd.points[3].color)
    brd = brd.move(6, 0)
    equals(None, brd.points[0].color) # Still no color since nothing jailed yet.
    brd = brd.move(1, 0)
    equals(WHITE, brd.points[0].color)


def _use(roll, move, expected):
    'Assert that using given move produces correct unused dies.'
    roll.use(move)
    print(roll, move, expected, roll.dies)
    equals(expected, roll.dies)

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
    equals('impossible move', msg)

def test_use4():
    'Roll.use() is read-only if given invalid move.'
    roll = Roll(4, 6)
    msg = None
    try:
        roll.use(13)
    except Exception as e:
        msg = str(e)
    equals('impossible move', msg)
    equals((4,6), roll.dies)


def test_unuse1():
    'Can Roll.unuse() a single die.'
    roll = Roll(4, 6)
    roll.use(4)
    equals((6,), roll.dies)
    roll.unuse(4)
    equals((4,6), roll.dies)

def test_unuse2():
    'Can Roll.unuse() multiple dies at once.'
    roll = Roll(4, 6)
    roll.use(10)
    equals((), roll.dies)
    roll.unuse(10)
    equals((4,6), roll.dies)

def test_unuse3():
    'Can Roll.unuse() doubles.'
    roll = Roll(4, 4)
    roll.use(12)
    equals((4,), roll.dies)
    roll.unuse(12)
    equals((4,4,4,4), roll.dies)

def test_unuse4():
    'Roll.unuse() raises exception when given invalid move.'
    roll = Roll(5, 5)
    roll.use(15)
    msg = None
    try:
        roll.unuse(20)
    except Exception as e:
        msg = str(e)
    equals('impossible to unuse', msg)


def test_json1():
    'Can [de]serialize Turn objects.'
    l = [
        Turn(Roll(), [(1,2), (3,4)]),
        Turn(Roll(), []),
    ]
    s = json.dumps(l, default=Turn.to_json)
    print('TO JSON:', s)
    x = json.loads(s, object_hook=Turn.from_json)
    print('FROM JSON:', x)
    for i in x:
        print('ACTUAL:', type(i), i)
    for i in l:
        print('EXPECTED:', type(i), i)
    equals(l, x)

def test_save1():
    'Can save & load games.'
    game = Game()
    game.roll_dice(Roll(2, 6))
    game.move(1, 3)
    game.move(12, 18)
    game.roll_dice(Roll(1, 1))
    game.move(8, 7)
    game.move(8, 7)
    game.draw()
    print(game)
    loaded_game = Game()
    assert game != loaded_game
    with tmp_path() as path:
        game.save(path)
        loaded_game.load(path)
        loaded_game.draw()
        print(loaded_game)
        equals(game, loaded_game)

def test_safe1():
    'Can get list of safe points.'
    # Nothing is safe for initial board.
    brd = Board()
    print(brd)
    equals([], brd.safe(WHITE))
    equals([], brd.safe(BLACK))
    # Black past white.
    brd = brd.move(1, 3)
    brd = brd.move(1, 3)
    brd = brd.move(6, 2)
    print(brd)
    equals([], brd.safe(WHITE))
    equals([brd.points[i] for i in [2]], brd.safe(BLACK))
    # White past black.
    brd = brd.move(24, 22)
    brd = brd.move(19, 23)
    brd = brd.move(24, 22)
    print(brd)
    equals([brd.points[i] for i in [23]], brd.safe(WHITE))

def test_exposed1():
    'Can get list of exposed points.'
    # Nothing is exposed for initial board.
    brd = Board()
    print(brd)
    equals([], brd.exposed(WHITE))
    equals([], brd.exposed(BLACK))
    # Common cases.
    brd = brd.move(1, 7)
    print(brd)
    equals([brd.points[i] for i in [1, 7]], brd.exposed(WHITE))
    brd = brd.move(6, 2)
    print(brd)
    equals([brd.points[i] for i in [2]], brd.exposed(BLACK))
    # Not considered exposed if behind enemy lines.
    brd = brd.move(1, 3)
    print(brd)
    equals([], brd.exposed(BLACK))
    # Jailed pieces not considered exposed, but...
    brd = brd.move(3, 0)
    print(brd)
    equals([brd.points[i] for i in [7]], brd.exposed(WHITE))
    # ...opposing pieces should still be considered exposed. 
    equals([brd.points[i] for i in [2]], brd.exposed(BLACK))

def test_jailed1():
    'Can get list of pieces in jail.'
    # Nothing is jailed for initial board.
    brd = Board()
    print(brd)
    equals((), brd.jailed(WHITE))
    equals((), brd.jailed(BLACK))
    # Common case.
    brd = brd.move(1, 0)
    print(brd)
    equals('(W:1,)', str(brd.jailed(WHITE)))
    # Multiple pieces in jail.
    brd = brd.move(24, 25)
    brd = brd.move(13, 25)
    print(brd)
    equals('(B:1, B:6)', str(brd.jailed(BLACK)))

def test_finished1():
    'A board is finished once all pieces are home for a color.'
    # Initial boards are not finished.
    brd = Board()
    print(brd)
    equals(False, brd.finished())
    # Should be finished after all white pieces are home.
    for point, count in ((1,2), (12,5), (17,3), (19,5)):
        for i in range(count):
            brd = brd.move(point, 25)
    print(brd)
    equals(True, brd.finished())
    # Make sure only counting appropriate color since jail & home points are shared.
    brd = brd.move(25, 23)
    brd = brd.move(24, 25)
    print(brd)
    equals(False, brd.finished())


def test_move1():
    'Board.move() accepts Point instances as well as integers.'
    brd = Board()
    equals(' 1:W2', str(brd.points[1]))
    brd = brd.move(brd.points[1], 3)
    equals(' 1:W1', str(brd.points[1]))
    equals(' 3:W1', str(brd.points[3]))


def test_all_choices_roll21():
    'Game.all_choices() returns all possible paths from all possible points for current turn.'
    game = Game()
    game.roll_dice(Roll(2, 1))
    equals(set(), set(game.all_choices()).symmetric_difference([((1, 2), (1, 3)), ((1, 2), (2, 4)), ((1, 2), (12, 14)), ((1, 2), (17, 19)), ((1, 2), (19, 21)), ((1, 3), (1, 2)), ((1, 3), (3, 4)), ((1, 3), (17, 18)), ((1, 3), (19, 20)), ((12, 14), (1, 2)), ((12, 14), (14, 15)), ((12, 14), (17, 18)), ((12, 14), (19, 20)), ((17, 18), (1, 3)), ((17, 18), (12, 14)), ((17, 18), (17, 19)), ((17, 18), (18, 20)), ((17, 18), (19, 21)), ((17, 19), (1, 2)), ((17, 19), (17, 18)), ((17, 19), (19, 20)), ((19, 20), (1, 3)), ((19, 20), (12, 14)), ((19, 20), (17, 19)), ((19, 20), (19, 21)), ((19, 20), (20, 22)), ((19, 21), (1, 2)), ((19, 21), (17, 18)), ((19, 21), (19, 20)), ((19, 21), (21, 22))]))


def test_weighted_strategy1():
    'The "safe" strategy will avoid capturing pieces if will expose its pieces.'
    # With this board and a roll of 4x1, the move [(12,13), (13,17)] captures 13:B1 but exposes 20:W1.
    # [ 12:W3 | 11    | 10    |  9    |  8:B2 |  7:B4 ] [  6:B4 |  5    |  4:B2 |  3    |  2    |  1:W2 ] [  0:W0:B0 ]
    # [ 13:B1 | 14    | 15    | 16    | 17:W2 | 18:B2 ] [ 19:W2 | 20:W1 | 21:W3 | 22    | 23:W2 | 24    ] [ 25:B0:W0 ]''')
    b1 = Board.from_str('''[ 12:W2 | 11    | 10    |  9    |  8:B2 |  7:B4 ] [  6:B4 |  5    |  4:B2 |  3    |  2    |  1:W2 ] [  0:W0:B0 ]
                           [ 13:B1 | 14    | 15    | 16    | 17:W3 | 18:B2 ] [ 19:W2 | 20:W1 | 21:W3 | 22    | 23:W2 | 24    ] [ 25:B0:W0 ]''')
    b2 = Board.from_str('''[ 12:W2 | 11    | 10    |  9    |  8:B2 |  7:B4 ] [  6:B4 |  5    |  4:B2 |  3    |  2    |  1:W2 ] [  0:W0:B0 ]
                           [ 13    | 14    | 15    | 16    | 17:W3 | 18:B2 ] [ 19:W2 | 20:W1 | 21:W3 | 22    | 23:W2 | 24    ] [ 25:B1:W0 ]''')
    gt(strategy.safe(WHITE, b1), strategy.safe(WHITE, b2))


def test_blocked1():
    'Home should never be considered blocked.'
    brd = Board.from_str('''[ 12:W3 | 11    | 10    |  9    |  8    |  7    ] [  6:B6 |  5:B3 |  4:B2 |  3:B4 |  2    |  1:W1 ] [  0:W1:B0 ]
                            [ 13    | 14    | 15    | 16    | 17:W3 | 18    ] [ 19:W2 | 20:W1 | 21:W3 | 22    | 23:W2 | 24    ] [ 25:B0:W0 ]''')
    assert not brd.points[0].blocked(BLACK)
    brd = brd.move(12, 0)
    assert not brd.points[0].blocked(BLACK)


def test_can_home_when_enemy_jailed1():
    'Should be able to move home even with enemies jailed.'
    brd = Board.from_str('''[ 12:W2 | 11    | 10    |  9    |  8    |  7    ] [  6:B6 |  5:B3 |  4:B2 |  3:B4 |  2    |  1:W1 ] [  0:W2:B0 ]
                            [ 13    | 14    | 15    | 16    | 17:W3 | 18    ] [ 19:W2 | 20:W1 | 21:W3 | 22    | 23:W2 | 24    ] [ 25:B0:W0 ]''')
    print( brd.possible_moves(Roll(3, 2), 3) )
    assert 0 in brd.possible_moves(Roll(3, 2), 3), 'cannot move home'


    # for path in game.all_choices():
    #     x = reduce(lambda brd,move: brd.move(*move), path, game.board)
    #     print()
    #     print("SCORE:", strategy.safe(game.color, x), "     PATH:", path)
    #     print(x)
    # assert False

