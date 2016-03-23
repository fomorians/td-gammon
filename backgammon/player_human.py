from functools import partial

from player import Player

class PlayerHuman(Player):
    """
    A human player using text-based console to interact with the game.
    """

    def interact(self, game, draw_board=True):
        print('move <start-position> <end-position>\tmake a move')
        print('save <path>\t\t\t\tsave the game')
        print('load <path>\t\t\t\tload a saved game')
        print('stop\t\t\t\t\tstop the game')

        while game.roll.dies:
            if draw_board:
                game.draw()

            try:
                cmd = self.get_command(game)
                cmd()
            except Exception as e:
                print('Invalid command: {}'.format(e))

    def get_command(self, game):
        """
        Prompt user for the next command and return it as a callable
        method.  Any exceptions that occur from incorrectly formatted
        commands will bubble up.
        """
        try:
            cmd = input('> ')
        except EOFError:
            cmd = 'stop'

        if cmd.startswith('stop'):
            return game.stop

        if cmd.startswith('save'):
            cmd_name, path = cmd.split()
            return partial(game.save, path)

        if cmd.startswith('load'):
            cmd_name, path = cmd.split()
            return partial(game.load, path)

        if cmd.startswith('move'):
            cmd_name, start, end = cmd.split()
            start, end = int(start), int(end)
            return partial(game.move, start, end)
