import random
from agent import Agent

class RandomAgent(Agent):

    def __init__(self, player):
        self.player = player
        self.name = 'Random'

    def getAction(self, moves, game=None):
        return random.choice(list(moves)) if moves else None
