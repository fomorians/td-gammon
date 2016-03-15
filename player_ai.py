class PlayerAI(Player):
    """
    An artificial player using given strategy to make moves.
    """

    def __init__(self, color, strategy):
        self.color = color
        self.score = partial(strategy, color)

    def interact(self, game):
        """
        Using strategy make most optimal move by enumerating all possible board positions.
        """
        high_score = -9999
        best_moves = []

        for moves in game.all_choices():
            score = self.score(reduce(lambda brd,move: brd.move(*move), moves, game.board))
            if score > high_score:
                high_score = score
                best_moves = moves

        for move in best_moves:
            game.draw()
            game.move(*move)
