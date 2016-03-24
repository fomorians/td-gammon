class Agent:
    def __init__(self, player):
        self.player = player

    def getAction(self, moves, game=None):
        raise NotImplementedError()
