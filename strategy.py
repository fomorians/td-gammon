from backgammon.player import Player

def td_gammon_strategy(model, color, board, game):
    scores = model.get_output(game)
    score = scores[0]
    if color == Player.WHITE:
        return score # max(scores[0][0], scores[0][2])
    else:
        return 1. - score # max(scores[0][1], scores[0][3])
