class PlayerHuman(Player):
    """
    A human player using text-based console to interact with the game.
    """

    def interact(self, game):
        while game.roll.dies:
            game.draw()
            try:
                cmd = self.get_command(game)
                cmd()
            except Exception as e:
                print('Invalid command: {}'.format(e))
                print(' - to make a move: <start-position> <end-position>')
                print(' - to stop the game: stop')
                print(' - to save the game: save <path>')
                print(' - to load a saved game: load <path>')

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
            return self.stop
        elif cmd.startswith('save'):
            l = cmd.split()
            return partial(game.save, l[1])
        elif cmd.startswith('load'):
            l = cmd.split()
            return partial(game.load, l[1])
        else:
            start, end = [int(i) for i in cmd.split()]
            return partial(game.move, start, end)

    def stop(self):
        sys.exit('Good-bye')
