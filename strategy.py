import numpy as np

from functools import partial
from collections import namedtuple

from player import Player
from board import Board

def td_gammon_strategy(model, color, board, game):
    scores = model.get_output(game)
    score = scores[0]
    if color == Player.WHITE:
        return score # max(scores[0][0], scores[0][2])
    else:
        return 1. - score # max(scores[0][1], scores[0][3])

class Weights(namedtuple('Weights', ['jailed', 'homed', 'exposed', 'stronghold', 'safe'])):
    """
    Weights to apply to these data points:
      * jailed: number of opponent pieces in jail
      * homed: number of pieces at home
      * exposed number of pieces exposed
      * points: number of strongholds made
      * safe: number of pieces safe (past all enemy pieces)
    """
    pass

def simply_weighted(weights, color, board):
    """
    A generic strategy that computes the quality of a board from given weights.
    """
    enemy = Player.WHITE if color == Player.BLACK else Player.BLACK
    score = 0
    if board.can_go_home(color):
        score = weights.homed * len(board.homed(color))
    else:
        score += weights.jailed * len(board.jailed(enemy))
        score += weights.exposed * len(board.exposed(color))
        score += weights.stronghold * len(board.strongholds(color))
    return score

def aggressive_strategy(color, board):
    """
    A strategy that highly favors boards with opponent pieces in jail.
    """
    return simply_weighted(aggressive_strategy.weights, color, board)

aggressive_strategy.weights = Weights(
    jailed = 5,
    homed = 1,
    exposed = -2,
    stronghold = 2,
    safe = 0,
)

def safe_strategy(color, board):
    """
    A strategy that highly penalizes exposed pieces.
    """
    return simply_weighted(safe_strategy.weights, color, board)

safe_strategy.weights = Weights(
    jailed = 1,
    homed = 1,
    exposed = -5,
    stronghold = 2,
    safe = 0,
)

def random_strategy(color, board):
    """
    A strategy that returns a random score for any board.
    """
    scores = np.random.random([1, 4])
    if color == Player.WHITE:
        return max(scores[0][0], scores[0][2])
    else:
        return max(scores[0][1], scores[0][3])
