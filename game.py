import sys
import json

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

    def play(self):
        """
        The main game loop.
        """
        while not self.board.finished():
            self.next()

    def next(self):
        self.roll_dice()
        player = self.white if self.color == Player.WHITE else self.black
        player.interact(self)

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
        assert(src >= 0 and src <= 25)
        assert(dst >= 0 and dst <= 25)

        dies = abs(dst - src)
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

    # TODO: this might be wrong
    @staticmethod
    def _all_choices(brd, roll, color, path):
        direction = 1 if color == Player.WHITE else -1
        min_point = 1
        max_point = 24

        if brd.jailed(color):
            points = [brd.jail(color)]
        else:
            points = filter(lambda pt: pt.color == color and pt.pieces, brd.points)
            if brd.can_go_home(color):
                if color == Player.BLACK:
                    min_point -= 1
                else:
                    max_point += 1

        for src in [pt.num for pt in points]:
            moves = []

            for hop in sorted(set(roll.dies)):
                dst = src + (direction * hop)
                if dst >= min_point and dst <= max_point and not brd.points[dst].blocked(color):
                    moves.append(dst)

            if not moves:
                yield path

            for dst in moves:
                used_roll = roll.copy()
                used_roll.use(abs(dst - src))

                for i in Game._all_choices(brd.move(src, dst), used_roll, color, path + ((src,dst),)):
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
            paths.add(path)
        return sorted(i for i in paths if len(i) >= min_moves)
