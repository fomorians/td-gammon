import numpy as np
from agent import Agent

class TDAgent(Agent, object):

    def __init__(self, player, model):
        self.player = player
        self.model = model
        self.name = 'TD-Gammon'

    def getAction(self, actions, game):
        """
        Return best action according to self.evaluationFunction,
        with no lookahead.
        """
        v_best = 0
        a_best = None

        for a in actions:
            ateList = game.take_action(a, self.player)
            features = game.extract_features(game.opponent(self.player))
            v = self.model.get_output(features)
            if v > v_best:
                v_best = v
                a_best = a
            game.undo_action(a, self.player, ateList)

        return a_best
