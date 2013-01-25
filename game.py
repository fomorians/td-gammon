"""
A way to "play" a game of backgammon.  Not much fun right now but it's a start.
"""
__all__ = ['Game']

import sys, json
from functools import partial

# I'm an idiot and I can't figure out how to do relative imports.
# from .model import Board, Roll, WHITE, BLACK
from model import Board, Roll, Turn, WHITE, BLACK
from util import freshmaker

if sys.version_info[0] == 2:
    input = raw_input


class Game(object):
    """
    A game is a Board and a history of Turns.
    """

    def __init__(I):
        I.board = Board()
        I.history = []

    def __str__(I):
        return '\n'.join(str(i) for i in I.history)

    def __eq__(I, other):
        return I.history == other.history

    @property
    def roll(I):
        """
        The current Roll.
        """
        return I.history[-1].roll

    @property
    def color(I):
        """
        The current color.  White always starts.
        """
        return BLACK if len(I.history) % 2 == 0 else WHITE

    @property
    def moves(I):
        """
        The moves used for the current Turn (mutable).
        """
        return I.history[-1].moves

    def play(I):
        """
        The main game loop.
        """
        while True:
            I.roll_dice()
            while I.roll.dies:
                I.draw()
                try:
                    cmd = I.get_command()
                except Exception as e:
                    print('Invalid command: {}'.format(e))
                    print(' - to make a move: <start-position> <end-position>')
                    print(' - to stop the game: stop')
                    print(' - to save the game: save <path>')
                    print(' - to load a saved game: load <path>')
                    continue
                cmd()

    def roll_dice(I, roll=None):
        """
        Record a new Roll.  If none specified, then use a random one.
        """
        I.history.append(Turn(roll or Roll(), []))

    def draw(I):
        """
        Print the following to stdout:
          * current roll
          * current color
          * unused dies for current roll
          * current board
          * possible moves
        """
        print()
        print(I.board)
        print('Current roll for {}: {} {}'.format(I.color, I.roll, I.roll.dies))
        print('Possible moves:')
        cant_move = True
        possible_points = [I.board.jail[I.color]]
        if not possible_points[0].pieces:
            # No pieces are jailed, so consider entire board.
            possible_points = [I.board.points[i] for i in range(26)]
        for point in possible_points:
            if point.pieces and point.color == I.color:
                moves = I.board.possible_moves(I.roll, point)
                if moves:
                    cant_move = False
                print('  {} -> {}'.format(point, moves)) # ','.join(str(i) for i in moves)))
        if cant_move:
            print('  No possible moves left')

    def get_command(I):
        """
        Prompt user for the next command and return it as a callable
        method.  Any exceptions that occur from incorrectly formatted
        commands will bubble up.
        """
        try:
            cmd = input('> ')
        except EOFError:
            cmd = 'stop'
        if cmd.startswith('stop'):
            return I.stop
        elif cmd.startswith('save'):
            l = cmd.split()
            return partial(I.save, l[1])
        elif cmd.startswith('load'):
            l = cmd.split()
            return partial(I.load, l[1])
        else:
            start, end = [int(i) for i in cmd.split()]
            return partial(I.move, start, end)

    def stop(I):
        sys.exit('Good-bye')

    def move(I, start, end):
        """
        * Update the board for given move.
        * Mark the move as used in the roll.
        * Capture move in this game's history.
        """
        dies = abs(end - start)
        with freshmaker(onerror=freshmaker.PRINT) as undo:
            I.roll.use(dies)
            undo.add(partial(I.roll.unuse, dies))
            I.board.move(start, end)
            I.moves.append((start, end))

    def save(I, path):
        """
        Serialize the history to given path.
        """
        with open(path, mode='w', encoding='utf-8') as f:
            json.dump(I.history, f, default=Turn.to_json)

    def load(I, path):
        """
        Reset the board and load history from given path.
        """
        with open(path, mode='r', encoding='utf-8') as f:
            history = json.load(f, object_hook=Turn.from_json)
        board = Board()
        for turn in history:
            for start, end in turn.moves:
                board.move(start, end)
        I.board = board
        I.history = history


if __name__ == '__main__':
    Game().play()
