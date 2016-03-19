import sys
import json
import numpy as np
import itertools

from player import Player
from board import Board
from point import Point
from roll import Roll
from turn import Turn

class Game(object):
    """
    A game is a Board and a history of Turns.
    """

    def __init__(self, white, black):
        self.board = Board()
        self.history = []
        self.black = black
        self.white = white

    def __str__(self):
        return '\n'.join(str(i) for i in self.history)

    def __eq__(self, other):
        return self.history == other.history

    @property
    def roll(self):
        """
        The current Roll.
        """
        return self.history[-1].roll

    @property
    def color(self):
        """
        The current color.  White always starts.
        """
        return Player.BLACK if len(self.history) % 2 == 0 else Player.WHITE

    @property
    def moves(self):
        """
        The moves used for the current Turn (mutable).
        """
        return self.history[-1].moves

    @property
    def winner(self):
        return Player.WHITE if len(self.board.homed(Player.WHITE)) == 15 else Player.BLACK

    def play(self, draw_board=True):
        """
        The main game loop.
        """
        while not self.board.finished():
            self.next(draw_board)

    def next(self, draw_board=True):
        self.roll_dice()
        player = self.white if self.color == Player.WHITE else self.black
        player.interact(self, draw_board=draw_board)

    def stop(self):
        sys.exit('Game stopped')

    def roll_dice(self, roll=None):
        """
        Record a new Roll.  If none specified, then use a random one.
        """
        self.history.append(Turn(roll or Roll(), []))

    def move(self, src, dst):
        """
        * Update the board for given move.
        * Mark the move as used in the roll.
        * Capture move in this game's history.
        """
        dies = abs(dst - src)

        if dst < 0:
            dst = 0
        elif dst > 25:
            dst = 25

        new = self.board.move(src, dst)
        self.roll.use(dies)
        self.moves.append((src, dst))
        self.board = new

    def draw(self):
        """
        Print the following to stdout:
          * current roll
          * current color
          * unused dies for current roll
          * current board
          * possible moves
        """
        print(self.board)
        print('{}: {}'.format(self.color, self.roll))

    def to_outcome_array(self):
        homed_white = len(self.board.homed(Player.WHITE))
        homed_black = len(self.board.homed(Player.BLACK))
        if homed_white == 15 and homed_black == 0: # gammon white
            return np.array([[0, 0, 1, 0]], dtype='float')
        elif homed_black == 15 and homed_white == 0: # gammon black
            return np.array([[0, 0, 0, 1]], dtype='float')
        elif homed_white == 15:
            return np.array([[1, 0, 0, 0]], dtype='float') # win white
        elif homed_black == 15: # gammon
            return np.array([[0, 1, 0, 0]], dtype='float') # win black

    @staticmethod
    def _all_choices(board, roll, color, path):
        direction = 1 if color == Player.WHITE else -1
        min_point = 1
        max_point = 24
        last_checkers_position = board.last_checkers_position(color)
        biggest_distance_to_home = last_checkers_position if color == Player.BLACK else (25 - last_checkers_position)

        if biggest_distance_to_home == 25:
            points = [board.jail(color)]
        else:
            points = filter(lambda pt: pt.color == color, board.points)
            if biggest_distance_to_home <= 6:
                if color == Player.BLACK:
                    min_point -= 1
                else:
                    max_point += 1

        for src in [pt.num for pt in points]:
            moves = set()
            for die in sorted(set(roll.dies)):
                dst = src + (direction * die)
                if min_point <= dst <= max_point and not board.points[dst].blocked(color):
                    moves.add((dst,die))
            if not moves and roll.dies and \
                            biggest_distance_to_home <= 6 and \
                            max(roll.dies) > biggest_distance_to_home and \
                            src == last_checkers_position:
                moves.add((src + (direction * max(roll.dies)), max(roll.dies)))

            if not moves:
                yield path

            for dst, die in moves:
                used_roll = roll.copy()
                used_roll.use(die)
                try:
                    next_board = board.move(src, dst)
                except AssertionError as e:
                    print(src, dst)
                    print(board.__str__())
                    raise e
                if next_board.finished():
                    yield path + ((src,dst),)
                else:
                    for i in Game._all_choices(next_board, used_roll, color, path + ((src,dst),)):
                        yield i

    def all_choices(self):
        """
        Return all possible paths from all points for current turn.
        """
        min_moves = 0
        paths = set()
        for path in Game._all_choices(self.board, self.roll, self.color, ()):
            if len(path) > min_moves:
                min_moves = len(path)
            if any(p in paths for p in itertools.permutations(path, min_moves)):
                continue
            paths.add(path)
        return filter(lambda p : len(p) == min_moves, paths)
