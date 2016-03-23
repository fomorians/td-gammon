class Turn(object):
    """
    A Turn captures the Roll and the moves made by the player.
    """

    def __init__(self, roll, moves):
        self.roll = roll
        self.moves = moves

    def __str__(self):
        return "{}: {}".format(self.roll, self.moves)

    def __eq__(self, other):
        return self.roll == other.roll and self.moves == other.moves
