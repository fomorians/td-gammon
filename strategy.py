"""
Algorithms for evaluating how good a Board is for a given color.
"""
__all__ = ['Weights', 'simply_weighted', 'aggressive', 'safe']

import random
from functools import partial
from collections import namedtuple
from model import Board, WHITE, BLACK


class Weights(namedtuple('Weights', ['jailed', 'homed', 'exposed', 'stronghold', 'safe'])):
    """
    Weights to apply to these data points:
      * jailed: number of opponent pieces in jail
      * homed: number of pieces at home
      * exposed number of pieces exposed
      * points: number of strongholds made
      * safe: number of pieces safe (past all enemy pieces)
    """


def simply_weighted(weights, color, board):
    """
    A generic strategy that computes the quality of a board from given weights.
    """
    enemy = WHITE if color == BLACK else BLACK
    score = 0
    if board.can_go_home(color):
        score = weights.homed * len(board.homed(color))
    else:
        score += weights.jailed * len(board.jailed(enemy))
        score += weights.exposed * len(board.exposed(color))
        score += weights.stronghold * len(board.strongholds(color))
    return score


def aggressive(color, board):
    """
    A strategy that highly favors boards with opponent pieces in jail.
    """
    return simply_weighted(aggressive.weights, color, board)

aggressive.weights = Weights(
    jailed = 5,
    homed = 1,
    exposed = -2,
    stronghold = 2,
    safe = 0,
)


def safe(color, board):
    """
    A strategy that highly penalizes exposed pieces.
    """
    return simply_weighted(safe.weights, color, board)

safe.weights = Weights(
    jailed = 1,
    homed = 1,
    exposed = -5,
    stronghold = 2,
    safe = 0,
)

def random(color, board):
    """
    A strategy that returns a random score for any board.
    """
    return random.randint(0, 9)


if __name__ == '__main__':
    pass
