import os
import copy
import random
import numpy as np

class Game:

    LAYOUT = "0-2-o,5-5-x,7-3-x,11-5-o,12-5-x,16-3-o,18-5-o,23-2-x"
    NUMCOLS = 24
    QUAD = 6
    OFF = 'off'
    ON = 'on'
    TOKENS = ['o', 'x']

    def __init__(self,layout=None,grid=None,offPieces=None,
                 barPieces=None,numPieces=None,gPlayers=None):
        """
        Define a new game object
        """
        self.die = Game.QUAD
        self.layout = layout
        if grid:
            self.grid = copy.deepcopy(grid)
            self.offPieces = copy.deepcopy(offPieces)
            self.barPieces = copy.deepcopy(barPieces)
            self.numPieces = copy.deepcopy(numPieces)
            self.players = gPlayers
            return
        self.players = Game.TOKENS
        self.grid = [[] for _ in range(Game.NUMCOLS)]
        self.offPieces = {}
        self.barPieces = {}
        self.numPieces = {}
        for t in self.players:
            self.barPieces[t] = []
            self.offPieces[t] = []
            self.numPieces[t] = 0

    def extract_features(self, player):
        features = []
        for p in self.players:
            for col in self.grid:
                feats = [0.] * 6
                if len(col) > 0 and col[0] == p:
                    for i in range(len(col)):
                        feats[min(i, 5)] += 1
                features += feats
            features.append(float(len(self.barPieces[p])) / 2.)
            features.append(float(len(self.offPieces[p])) / self.numPieces[p])
        if player == self.players[0]:
            features += [1., 0.]
        else:
            features += [0., 1.]
        return np.array(features).reshape(1, -1)

    def roll_dice(self):
        return (random.randint(1, self.die), random.randint(1, self.die))

    def next_step(self, player, player_num, draw=False):
        roll = self.roll_dice()
        if draw:
            self.draw()
            print "Player %s rolled <%d,%d>." % (player.player, roll[0], roll[1])

        if player_num:
            self.reverse()

        moves = self.getActions(roll, self.players[0], nodups=True)
        move = player.getAction(moves, self) if moves else None

        if move:
            self.takeAction(move, self.players[0])

        if player_num:
            self.reverse()

    def clone(self):
        """
        Return an exact copy of the game. Changes can be made
        to the cloned version without affecting the original.
        """
        return Game(None,self.grid,self.offPieces,
                    self.barPieces,self.numPieces,self.players)

    def takeAction(self,action,token):
        """
        Makes given move for player, assumes move is valid,
        will remove pieces from play
        """
        ateList = [0]*4
        for i,(s,e) in enumerate(action):
            if s==Game.ON:
                piece = self.barPieces[token].pop()
            else:
                piece = self.grid[s].pop()
            if e==Game.OFF:
                self.offPieces[token].append(piece)
                continue
            if len(self.grid[e])>0 and self.grid[e][0] != token:
                bar_piece = self.grid[e].pop()
                self.barPieces[bar_piece].append(bar_piece)
                ateList[i] = 1
            self.grid[e].append(piece)
        return ateList

    def undoAction(self,action,player,ateList):
        """
        Reverses given move for player, assumes move is valid,
        will remove pieces from play
        """
        for i,(s,e) in enumerate(reversed(action)):
            if e==Game.OFF:
                piece = self.offPieces[player].pop()
            else:
                piece = self.grid[e].pop()
                if ateList[len(action)-1-i]:
                    bar_piece = self.barPieces[self.opponent(player)].pop()
                    self.grid[e].append(bar_piece)
            if s==Game.ON:
                self.barPieces[player].append(piece)
            else:
                self.grid[s].append(piece)


    def getActions(self,roll,player,nodups=False):
        """
        Get set of all possible move tuples
        """
        moves = set()
        if nodups:
            start=0
        else:
            start=None

        r1,r2 = roll
        if r1 == r2: # doubles
            i = 4
            # keep trying until we find some moves
            while not moves and i>0:
                self.findMoves(tuple([r1]*i),player,(),moves,start)
                i -= 1
        else:
            self.findMoves(roll,player,(),moves,start)
            self.findMoves((r2,r1),player,(),moves,start)
            # has no moves, try moving only one piece
            if not moves:
                for r in roll:
                    self.findMoves((r,),player,(),moves,start)

        return moves

    def findMoves(self,rs,player,move,moves,start=None):
        if len(rs)==0:
            moves.add(move)
            return
        r,rs = rs[0],rs[1:]
        # see if we can remove a piece from the bar
        if self.barPieces[player]:
            if self.can_onboard(player,r):
                piece = self.barPieces[player].pop()
                bar_piece = None
                if len(self.grid[r-1])==1 and self.grid[r-1][-1]!=player:
                    bar_piece = self.grid[r-1].pop()

                self.grid[r-1].append(piece)

                self.findMoves(rs,player,move+((Game.ON,r-1),),moves,start)
                self.grid[r-1].pop()
                self.barPieces[player].append(piece)
                if bar_piece:
                    self.grid[r-1].append(bar_piece)

            return

        # otherwise check each grid location for valid move using r
        offboarding = self.can_offboard(player)

        for i in range(len(self.grid)):
            if start is not None:
                start = i
            if self.is_valid_move(i,i+r,player):

                piece = self.grid[i].pop()
                bar_piece = None
                if len(self.grid[i+r])==1 and self.grid[i+r][-1]!=player:
                    bar_piece = self.grid[i+r].pop()
                self.grid[i+r].append(piece)
                self.findMoves(rs,player,move+((i,i+r),),moves,start)
                self.grid[i+r].pop()
                self.grid[i].append(piece)
                if bar_piece:
                    self.grid[i+r].append(bar_piece)

            # If we can't move on the board can we take the piece off?
            if offboarding and self.remove_piece(player,i,r):
                piece = self.grid[i].pop()
                self.offPieces[player].append(piece)
                self.findMoves(rs,player,move+((i,Game.OFF),),moves,start)
                self.offPieces[player].pop()
                self.grid[i].append(piece)

    def opponent(self,token):
        """
        Retrieve opponent players token for a given players token.
        """
        for t in self.players:
            if t!= token: return t

    def isWon(self,player):
        """
        If game is over and player won, return True, else return False
        """
        return self.is_over() and player==self.players[self.winner()]

    def isLost(self,player):
        """
        If game is over and player lost, return True, else return False
        """
        return self.is_over() and player!=self.players[self.winner()]


    def reverse(self):
        """
        Reverses a game allowing it to be seen by the opponent
        from the same perspective
        """
        self.grid.reverse()
        self.players.reverse()

    def new_game(self):
        """
        Resets game to original layout.
        """
        for col in self.layout.split(','):
            loc,num,token = col.split('-')
            self.grid[int(loc)] = [token for _ in range(int(num))]
        for col in self.grid:
            for piece in col:
                self.numPieces[piece] += 1

    def winner(self):
        """
        Get winner.
        """
        return len(self.offPieces[self.players[0]])==self.numPieces[self.players[0]]

    def is_over(self):
        """
        Checks if the game is over.
        """
        for t in self.players:
            if len(self.offPieces[t])==self.numPieces[t]:
                return True
        return False

    def can_offboard(self,player):
        count = 0
        for i in range(Game.NUMCOLS-self.die,Game.NUMCOLS):
            if len(self.grid[i])>0 and self.grid[i][0]==player:
                count += len(self.grid[i])
        if count+len(self.offPieces[player]) == self.numPieces[player]:
            return True
        return False

    def can_onboard(self,player,r):
        """
        Can we take a players piece on the bar to a position
        on the grid given by roll-1?
        """
        if len(self.grid[r-1])<=1 or self.grid[r-1][0]==player:
            return True
        else:
            return False

    def remove_piece(self,player,start,r):
        """
        Can we remove a piece from location start with roll r ?
        In this function we assume we are cool to offboard,
        i.e. no pieces on the bar and all are in the home quadrant.
        """
        if start < Game.NUMCOLS - self.die:
            return False
        if len(self.grid[start]) == 0 or self.grid[start][0] != player:
            return False
        if start+r == Game.NUMCOLS:
            return True
        if start+r > Game.NUMCOLS:
            for i in range(start-1,Game.NUMCOLS-self.die-1,-1):
                if len(self.grid[i]) != 0 and self.grid[i][0]==self.players[0]:
                    return False
            return True
        return False

    def is_valid_move(self,start,end,token):
        if len(self.grid[start]) > 0 and self.grid[start][0] == token:
            if end < 0 or end >= len(self.grid):
                return False
            if len(self.grid[end]) <= 1:
                return True
            if len(self.grid[end])>1 and self.grid[end][-1] == token:
                return True
        return False


    def draw_col(self,i,col):
        print "|",
        if i==-2:
            if col<10:
                print "",
            print str(col),
        elif i==-1:
            print "--",
        elif len(self.grid[col])>i:
            print " "+self.grid[col][i],
        else:
            print "  ",

    def draw(self):
        os.system('clear')
        largest = max([len(self.grid[i]) for i in range(len(self.grid)/2,len(self.grid))])
        for i in range(-2,largest):
            for col in range(len(self.grid)/2,len(self.grid)):
                self.draw_col(i,col)
            print "|"
        print
        print
        largest = max([len(self.grid[i]) for i in range(len(self.grid)/2)])
        for i in range(largest-1,-3,-1):
            for col in range(len(self.grid)/2-1,-1,-1):
                self.draw_col(i,col)
            print "|"
        for t in self.players:
            print "<Player %s>  Off Board : "%(t),
            for piece in self.offPieces[t]:
                print t+'',
            print "   Bar : ",
            for piece in self.barPieces[t]:
                print t+'',
            print

    def gridLocFromPos(self,pos,player):
        tx,ty = self.tokIms['x'].get_rect().size

        def onPiece(pieceLoc,pos,sizex,sizey):
            px,py = pieceLoc
            tx,ty = pos
            if tx < px+sizex and tx > px:
                if ty < py+sizey and ty > py:
                    return True
            return False

        # find out if we are on the grid
        for i,col in enumerate(self.grid):
            for loc in self.gridLocs[23-i]:
                if onPiece(loc,pos,tx,ty):
                    return i

        # find out if we are on the bar
        for i,bp in enumerate(self.barPieces[player]):
            if onPiece(self.barLocs[player][i],pos,tx,ty):
                return Game.ON

        # find out if we are removing pieces
        offBase = self.offLocs['o'][0] if player=='o' else self.offLocs['x'][-1]
        offHeight = 200
        offWidth,_ = self.offIms['x'].get_rect().size
        if onPiece(offBase,pos,offWidth,offHeight):
            return Game.OFF

        return None

if __name__=='__main__':
    g = Game(LAYOUT)
    g.new_game()
    g.draw()
