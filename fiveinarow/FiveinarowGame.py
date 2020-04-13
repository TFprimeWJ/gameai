import sys
import numpy as np

sys.path.append('..')
from Game import Game
from .FiveinarowLogic import Board


class FiveinarowGame(Game):
    """
    Connect4 Game class implementing the alpha-zero-general Game interface.
    """
    square_content = {
        -1: "◉",
        +0: "-",
        +1: "◯"
    }

    def __init__(self, height=None, width=None, win_length=None, np_pieces=None):
        Game.__init__(self)
        self._base_board = Board(height, width, win_length, np_pieces)

    def getInitBoard(self):
        return self._base_board.np_pieces

    def getBoardSize(self):
        return (self._base_board.height, self._base_board.width)

    def getActionSize(self):
        return self._base_board.width * self._base_board.height

    def getNextState(self, board, player, action):
        """Returns a copy of the board with updated move, original board is unmodified."""
        b = self._base_board.with_np_pieces(np_pieces=np.copy(board))
        move = (int(action/self._base_board.width), action%self._base_board.height)
        b.add_stone(move, player)
        return (b.np_pieces, -player)

    def getValidMoves(self, board, player):
        """Any zero value in the board is a valid move"""
        return self._base_board.with_np_pieces(np_pieces=board).get_valid_moves().reshape([1,-1]).squeeze()

    def getGameEnded(self, board, player):
        b = self._base_board.with_np_pieces(np_pieces=board)
        winstate = b.get_win_state()
        if winstate.is_ended:
            if winstate.winner is None:
                # draw has very little value.
                return 1e-4
            elif winstate.winner == player:
                return +1
            elif winstate.winner == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        else:
            # 0 used to represent unfinished game.
            return 0

    def getCanonicalForm(self, board, player):
        # Flip player from 1 to -1
        return board * player

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self._base_board.width**2)  # 1 for pass
        pi_board = np.reshape(pi, (self._base_board.width, self._base_board.width))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        return str(self._base_board.with_np_pieces(np_pieces=board))


    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(format(y, '<2'), end="")
            print("|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(FiveinarowGame.square_content[piece], end=" ")
            print("|")
        print("-----------------------")
