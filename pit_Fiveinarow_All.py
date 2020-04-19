import Arena
from MCTS import MCTS
from fiveinarow.FiveinarowGame import FiveinarowGame
from fiveinarow.FiveinarowPlayers import *
from fiveinarow.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

# mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = False

g = FiveinarowGame(10, 10)

# if mini_othello:
#     g = OthelloGame(6)
# else:
#     g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
# gp = GreedyOthelloPlayer(g).play
hp = HumanFiveinarowPlayer(g).play
op = OneStepLookaheadFiveinarowPlayer(g).play

# nnet + onestepfoward player

# nnet players
# n3 = NNet(g)
# n3.load_checkpoint('./temp','best.pth.tar')
# args3 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
# mcts3 = MCTS(g, n3, args3)
# n3p = lambda x: np.argmax(mcts3.getActionProb(x, temp=0))

class NNetPlusOneStepPlayer():
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""
    def __init__(self, game, verbose=True):
        self.game = game
        self.player_num = 1
        self.verbose = verbose

        self.n3 = NNet(g)
        self.n3.load_checkpoint('./temp/best','best28.pth.tar')
        self.args3 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
        self.mcts3 = MCTS(g, self.n3, self.args3)
        self.n3p = lambda x: np.argmax(self.mcts3.getActionProb(x, temp=0))

    def play(self, board):
        valid_moves = self.game.getValidMoves(board, self.player_num)
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()
        for move, valid in enumerate(valid_moves):
            if not valid: continue
            if self.player_num == self.game.getGameEnded(*self.game.getNextState(board, self.player_num, move)):
                win_move_set.add(move)
            if -self.player_num == self.game.getGameEnded(*self.game.getNextState(board, -self.player_num, move)):
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose: print('Playing winning action %s from %s' % (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose: print('Playing loss stopping action %s from %s' % (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            # ret_move = np.random.choice(list(fallback_move_set))
            ret_move = self.n3p(board)
            if self.verbose: print('Playing nnet action %s from %s' % (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % game.stringRepresentation(board))

        return ret_move

nop1 = NNetPlusOneStepPlayer(g).play
nop2 = NNetPlusOneStepPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./temp/best', 'best1.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, rp, g, display=FiveinarowGame.display)

print(arena.playGames(50, verbose=False))
