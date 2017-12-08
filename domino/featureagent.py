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
        Whether action will match teammate's last move
    '''
    def matches_teammate_last_move(self, board, curr_player_hand, curr_player, action):
        domino, side = action
        reveal_pip = domino[reveal_side]
        last_action = board[-2]
        if last_action:
            last_dom, last_side = last_action
            last_reveal_side = 0 if last_side == 1 else 1
            if reveal_pip == last_reveal_side:
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
