from ..game import Game

class HumanAgent(object):
    def __init__(self, player):
        self.player = player
        self.name = 'Human'

    def get_action(self, moves, game=None):
        if not moves:
            raw_input("No moves for you...(hit enter)")
            return None

        while True:
            while True:
                mv1 = raw_input('Please enter a move "<location start>,<location end>" ("%s" for on the board, "%s" for off the board): ' % (Game.ON, Game.OFF))
                mv1 = self.get_formatted_move(mv1)
                if not mv1:
                    print 'Bad format enter e.g. "3,4"'
                else:
                    break

            while True:
                mv2 = raw_input('Please enter a second move (enter to skip): ')
                if mv2 == '':
                    mv2 = None
                    break
                mv2 = self.get_formatted_move(mv2)
                if not mv2:
                    print 'Bad format enter e.g. "3,4"'
                else:
                    break

            if mv2:
                move = (mv1, mv2)
            else:
                move = (mv1,)

            if move in moves:
                return move
            elif move[::-1] in moves:
                move = move[::-1]
                return move
            else:
                print "You can't play that move"

        return None

    def get_formatted_move(self, move):
        try:
            start, end = move.split(",")
            if start == Game.ON:
                return (start, int(end))
            if end == Game.OFF:
                return (int(start), end)
            return (int(start), int(end))
        except:
            return False
