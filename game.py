"""
A way to "play" a game.  Not much fun right now but it's a start.
"""

import sys
from functools import partial
from contextlib import contextmanager

# I'm an idiot and I can't figure out how to do a simple relative import.
# from .model import Board, Roll, WHITE, BLACK
from model import Board, Roll, WHITE, BLACK


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


if __name__ == '__main__':
    Game().play()
