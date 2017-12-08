import numpy as np

class FeatureAgent:
    def __init__(self):
        pass

    def get_agent_move(self, game):
        curr_player = game.curr_player
        curr_player_hand = game.get_player_hand(curr_player)
        poss_actions = game.get_possible_actions()
        for poss_a in poss_actions:
            domino, side = poss_a
            

        pass

    def save_weights(self):
        pass

    '''
        Whether action will match previous move of opponent. Expect negative score because
        opens up same pips for opponent teammate to play.
    '''
    def matches_opp_last_move(self, game, player, move):
        domino, side = move
        new_end = domino[0] if domino[0] == game.ends[side] else domino[1]
        if new_end == game.ends[0] or new_end == game.ends[1]:
            return 1
        return 0


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
