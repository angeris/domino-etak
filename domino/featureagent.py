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
            # 

        pass

    def save_weights(self):
        pass

