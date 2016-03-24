import random
from agent import Agent

class RandomAgent(Agent):
    def getAction(self, moves, game=None):
        return random.choice(list(moves)) if moves else None
