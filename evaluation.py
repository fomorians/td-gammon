import random

from backgammon.game import Game

def test(players, episodes=100):
    winners = [0, 0]
    for episode in range(episodes):
        game = Game.new()

        while not game.is_over():
            game.next_step(players[game.player_num], draw=False)

        winner = game.winner()
        winners[not winner] += 1

        print("[Episode %d] Player %s \t (%s) %d/%d" % (episode, players[0].name, players[0].player, winners[0], sum(winners)))
        print("[Episode %d] Player %s \t (%s) %d/%d" % (episode, players[1].name, players[1].player, winners[1], sum(winners)))

def play(players):
    game = Game.new()
    while not game.is_over():
        game.next_step(players[game.player_num], draw=True)
