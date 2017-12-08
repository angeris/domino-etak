import numpy as np

class FeatureAgent:
    def __init__(self):
        pass

    def get_agent_move(self):
        pass

    def save_weights(self):
        pass

    def last_k_pip(self, game, player, move):
        remaining_dominoes = 7
        curr_domino = move[0]
        for d in game.board:
            if move.fits_val(d):
                remaining_dominoes -= 1

        for d in game.get_player_hand(player):
            if d == move:
                continue
            if move.fits_val(d):
                remaining_dominoes -= 1

        return remaining_dominoes == 1
