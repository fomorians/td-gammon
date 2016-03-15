import json

from player import Player
from player_ai import PlayerAI
from player_human import PlayerHuman

from board import Board
from point import Point
from roll import Roll
from turn import Turn

import strategy

class Game(object):
    """
    A game is a Board and a history of Turns.
    """

    def __init__(self, white=None, black=None):
        self.board = Board()
        self.history = []
        self.black = black or PlayerHuman()
        self.white = white or PlayerAI(Player.WHITE, strategy.safe)

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

    def play(self):
        """
        The main game loop.
        """
        while True:
            self.roll_dice()
            player = self.white if self.color == Player.WHITE else self.black
            player.interact(self)

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
        if isinstance(src, Point):
            src = src.num
        if isinstance(dst, Point):
            dst = dst.num
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
        print('Current roll for {}: {} {}'.format(self.color, self.roll, self.roll.dies))
        print('Possible moves:')
        cannot_move = True
        possible_points = [self.board.jail(self.color)]
        if not possible_points[0].pieces:
            # No pieces are jailed, so consider entire board.
            possible_points = [self.board.points[i] for i in range(26)]
        for point in possible_points:
            if point.pieces and point.color == self.color:
                moves = self.board.possible_moves(self.roll, point)
                if moves:
                    cannot_move = False
                print('  {} -> {}'.format(point, moves)) # ','.join(str(i) for i in moves)))
        if cannot_move:
            print('  No possible moves left')

    def save(self, path):
        """
        Serialize the history to given path.
        """
        with open(path, mode='w', encoding='utf-8') as f:
            json.dump(self.history, f, default=Turn.to_json)

    def load(self, path):
        """
        Reset the board and load history from given path.
        """
        with open(path, mode='r', encoding='utf-8') as f:
            history = json.load(f, object_hook=Turn.from_json)

        board = Board()
        for turn in history:
            for src, dst in turn.moves:
                board = board.move(src, dst)

        self.board = board
        self.history = history

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
                if color == BLACK:
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
