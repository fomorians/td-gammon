"""
A way to "play" a game of backgammon.  Not much fun right now but it's a start.
"""
__all__ = ['Game']

import sys, json
from functools import partial, reduce

# I'm an idiot and I can't figure out how to do relative imports.
# from .model import Board, Roll, WHITE, BLACK
from model import Board, Point, Roll, Turn, WHITE, BLACK
from util import freshmaker
import strategy

if sys.version_info[0] == 2:
    input = raw_input


class Player(object):
    """
    Interact with a board given a color & strategy.
    """
    def interact(I, board, roll):
        raise ValueError("need to implement interact(): {}".format(I))


class ComputerPlayer(Player):
    """
    An artificial player using given strategy to make moves.
    """

    def __init__(I, color, strategy):
        I.color = color
        I.score = partial(strategy, color)

    def interact(I, game):
        """
        Using strategy make most optimal move by enumerating all possible board positions.
        """
        high_score = -9999
        best_moves = []
        for moves in game.all_choices():
            score = I.score(reduce(lambda brd,move: brd.move(*move), moves, game.board))
            # print("SCORE: {:5}     PATH: {}".format(score, moves))
            if score > high_score:
                high_score = score
                best_moves = moves
        for move in best_moves:
            # print("MOVE:", move)
            game.draw()
            game.move(*move)


class ConsolePlayer(Player):
    """
    A human player using text-based console to interact with the game.
    """

    def interact(I, game):
        while game.roll.dies:
            game.draw()
            try:
                cmd = I.get_command(game)
                cmd()
            except Exception as e:
                print('Invalid command: {}'.format(e))
                print(' - to make a move: <start-position> <end-position>')
                print(' - to stop the game: stop')
                print(' - to save the game: save <path>')
                print(' - to load a saved game: load <path>')

    def get_command(I, game):
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
            return partial(game.save, l[1])
        elif cmd.startswith('load'):
            l = cmd.split()
            return partial(game.load, l[1])
        else:
            start, end = [int(i) for i in cmd.split()]
            return partial(game.move, start, end)

    def stop(I):
        sys.exit('Good-bye')


class Game(object):
    """
    A game is a Board and a history of Turns.
    """

    def __init__(I, white=None, black=None):
        I.board = Board()
        I.history = []
        I.black = black or ConsolePlayer()
        I.white = white or ComputerPlayer(WHITE, strategy.safe)

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
            player = I.white if I.color == WHITE else I.black
            player.interact(I)

    def roll_dice(I, roll=None):
        """
        Record a new Roll.  If none specified, then use a random one.
        """
        I.history.append(Turn(roll or Roll(), []))

    def move(I, src, dst):
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
        new = I.board.move(src, dst)
        I.roll.use(dies)
        I.moves.append((src, dst))
        I.board = new

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
        possible_points = [I.board.jail(I.color)]
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
            for src, dst in turn.moves:
                board = board.move(src, dst)
        I.board = board
        I.history = history

    @staticmethod
    def _all_choices(brd, roll, color, path):
        direction = 1 if color == WHITE else -1
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
                # print("SRC: {:<10} DST: {:<10} DIES: {:<10} PATH: {}".format(src, dst, used_roll.dies, path))
                for i in Game._all_choices(brd.move(src, dst), used_roll, color, path + ((src,dst),)):
                    yield i

    def all_choices(I):
        """
        Return all possible paths from all points for current turn.
        """
        min_moves = 0
        paths = set()
        for path in Game._all_choices(I.board, I.roll, I.color, ()):
            if len(path) > min_moves:
                min_moves = len(path)
            paths.add(path)
        return sorted(i for i in paths if len(i) >= min_moves)


if __name__ == '__main__':
    Game().play()
