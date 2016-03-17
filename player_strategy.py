from functools import partial, reduce

from player import Player

class PlayerStrategy(Player):
    """
    An artificial player using given strategy to make moves.
    """

    def __init__(self, color, strategy):
        self.color = color
        self.score = partial(strategy, color)

    def interact(self, game):
        """
        Make optimal move by enumerating all possible board positions.
        """
        high_score = None
        best_moves = []

        for moves in game.all_choices():
            score = self.score(reduce(lambda board, move: board.move(*move), moves, game.board))
            if high_score is None or score > high_score:
                high_score = score
                best_moves = moves

        for move in best_moves:
            game.draw()
            game.move(*move)
