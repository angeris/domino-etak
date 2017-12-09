import numpy as np
from collections import deque

class FeatureAgent:
    def __init__(self, q_maxlen=10000):
        self.memory = deque(maxlen=q_maxlen)
        self.discount = .99
        self.learning_rate = 0.01
        self.dimension = 100
        self.weights = np.array(self.dimension)

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

    def last_k_pip(self, game, player, move):
        remaining_dominoes = 7
        curr_domino = move[0]
        for d in game.board:
            if move.fits(d):
                remaining_dominoes -= 1

        for d in game.get_player_hand(player):
            if d == move:
                continue
            if move.fits(d):
                remaining_dominoes -= 1

        return remaining_dominoes == 1

    def to_one_hot(self, game, player, move):
        continue

    def train_on_memory(self):
        for m in self.memory:
            game, player, move, reward, next_game, next_move = m
            sa = to_one_hot(game, player, move)
            spap = to_one_hot(next_game, player, next_move)

            self.weights += self.learning_rate*(reward + self.weights @ (spap - sa))*sa

