from game import DominosGame
from domino import Domino
import numpy as np

def is_greedy_move(self, game, hand, player, action):
    poss_actions = game.get_possible_actions()
    best_a = None
    max_pip_domino = Domino(0,0)
    if poss_actions[0] is not None:
        for action in poss_actions:
            if action[0] >= max_pip_domino:
                max_pip_domino = action[0]
                best_a = action
    if best_a == action:
        return True
    return False

def num_dom_inhand_matches(self, game, hand, player, action):
    board_pip = self.ends[action[1]]
    if action[0].value[0] == board_pip:
        face_out_pip = action[0].value[0]
    else:
        face_out_pip = action[0].value[1]

    num_matches = 0
    curr_hand = game.get_player_hand(curr_player)
    poss_actions = game.get_possible_actions()
    if curr_hand[0] is not None:
        for dom in curr_hand:
            if face_out_pip == dom[0] or face_out_pip == dom[1]:
                num_matches += 1
    num_matches_list = [0,0,0,0,0,0,0]
    num_matches_list[num_matches] = 1
    return num_matches_list

def total_pip(self, game, hand, player, action):
    return action[0][0] + action[0][1]