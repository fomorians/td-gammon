import os
import random

from backgammon.game import Game

def test(players, episodes=100):
    winners = [0, 0]
    for episode in range(episodes):
        game = Game()
        game.reset()

        player_num = random.randint(0, 1)
        while not game.is_over():
            game.next_step(players[player_num], player_num, draw=False)
            player_num = (player_num + 1) % 2

        winner = game.winner()
        winners[not winner] += 1

        os.system('clear')
        print("[Episode %d] Player %s \t (%s) %d/%d" % (episode, players[0].name, players[0].player, winners[0], sum(winners)))
        print("[Episode %d] Player %s \t (%s) %d/%d" % (episode, players[1].name, players[1].player, winners[1], sum(winners)))

def play(players):
    game = Game()
    game.reset()

    player_num = random.randint(0, 1)
    while not game.is_over():
        game.next_step(players[player_num], player_num, draw=True)
        player_num = (player_num + 1) % 2
