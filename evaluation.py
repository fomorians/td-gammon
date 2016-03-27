import time
import random
import numpy as np

from backgammon.game import Game
from backgammon.agents.random_agent import RandomAgent
from backgammon.agents.td_gammon_agent import TDAgent

def play(players):
    game = Game.new()
    game.play(players, draw=True)

def test(players, episodes=100, draw=False):
    winners = [0, 0]
    for episode in range(episodes):
        game = Game.new()

        winner = game.play(players, draw=draw)
        winners[winner] += 1

        print("[Episode %d] Player %s \t (%s) %d/%d" % (episode, players[0].name, players[0].player, winners[0], sum(winners)))
        print("[Episode %d] Player %s \t (%s) %d/%d" % (episode, players[1].name, players[1].player, winners[1], sum(winners)))

def train(sess):
    tf.train.write_graph(sess.graph_def, model_path, 'td_gammon.pb', as_text=False)
    summary_writer = tf.train.SummaryWriter('{0}{1}'.format(summary_path, int(time.time()), sess.graph_def))

    model = Model()

    # the agent plays against itself, making the best move for each player
    players = [TDAgent(Game.TOKENS[0], model), TDAgent(Game.TOKENS[1], model)]
    players_test = [TDAgent(Game.TOKENS[0], model), RandomAgent(Game.TOKENS[1])]

    validation_interval = 1000
    episodes = 2000

    for episode in range(episodes):
        if episode != 0 and episode % validation_interval == 0:
            test(players_test, episodes=100)

        game = Game.new()
        player_num = random.randint(0, 1)

        x = game.extract_features(players[player_num].player)

        game_step = 0
        while not game.is_over():
            game.next_step(players[player_num], player_num)
            player_num = (player_num + 1) % 2

            x_next = game.extract_features(players[player_num].player)
            V_next = model.get_output(x_next)
            sess.run(model.train_op, feed_dict={ model.x: x, model.V_next: V_next })

            x = x_next
            game_step += 1

        winner = game.winner()

        _, global_step, summaries, _ = sess.run([
            model.train_op,
            model.global_step,
            model.summaries_op,
            model.reset_op
        ], feed_dict={ model.x: x, model.V_next: np.array([[winner]], dtype='float') })
        summary_writer.add_summary(summaries, global_step=global_step)

        print("Game %d/%d (Winner: %s) in %d turns" % (episode, episodes, players[winner].player, game_step))
        model.saver.save(sess, checkpoint_path + 'checkpoint', global_step=global_step)

    summary_writer.close()

    test(players_test, episodes=1000)
