class Player(object):
    WHITE = 'W'
    BLACK = 'B'

    """
    Interact with a board given a color & strategy.
    """
    def interact(self, game):
        raise ValueError("need to implement interact(): {}".format(self))
