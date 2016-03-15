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

    @staticmethod
    def to_json(obj):
        """
        Hook for json.dump() & json.dumps().
        """
        if isinstance(obj, Turn):
            return dict(roll=str(obj.roll), moves=obj.moves)
        raise TypeError("not json-serializable: {}<{}>".format(type(obj), obj))

    @staticmethod
    def from_json(obj):
        """
        Hook for json.load() & json.loads().
        """
        if isinstance(obj, dict) and 'roll' in obj:
            return Turn(Roll.from_str(obj['roll']), [tuple(i) for i in obj['moves']])
        return obj

    def __eq__(self, other):
        return self.roll == other.roll and self.moves == other.moves
