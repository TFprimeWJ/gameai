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



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp','best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./temp/best', 'best.pth1.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, n2p, g, display=FiveinarowGame.display)

print(arena.playGames(40, verbose=True))
