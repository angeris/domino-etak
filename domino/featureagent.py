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
        pass

    def save_weights(self):
        pass

    def selfplay(self):
        agent0Wins = 0
        for i in range(num_games): # play multiple games
            game = DominosGame()
            is_end_state = game.is_end_state()
            while(not is_end_state):    # play game
                board = copy(game.board)
                curr_player = game.curr_player
                curr_player_hand = game.get_player_hand(curr_player)
                best_a = self.getAgentMove(game, self.total_games)    # pass in total games played so far which is updated when testing against greedy?
                game.move(best_a)
                is_end_state = game.is_end_state()
                scores = []
                if is_end_state:
                    for player_idx in range(4):
                        scores.append(game.get_score(player_idx))
                    # print('scores', scores)
                    # back propogate scores and end state
                    self.memory[-1][3] = scores
                    self.memory[-1][2] = True
                    self.memory[-2][3] = scores
                    self.memory[-2][2] = True
                    self.memory[-3][3] = scores
                    self.memory[-3][2] = True

                    if scores[0] >= scores[1]:
                        agent0Wins +=1
                # s', a, is_end, scores, hand, curr_player
                sa = [board, best_a, is_end_state, scores, curr_player_hand, curr_player]
                self.memory.append(sa)
        print('Agent 0 wins', float(agent0Wins)/num_games)


    '''
        Whether action will match previous move of opponent. Expect negative weight.
        Opponent: 1-3
        Me:       3-3
        Return 1
    '''
    def matches_opp_last_move(self, game, player, move):
        domino, side = move
        new_end_val = domino[0] if domino[0] == game.ends[side] else domino[1]
        if new_end_val == game.ends[0] or new_end_val == game.ends[1]:
            return 1
        return 0

    '''
        Whether action will match what teammate opened up for you. Expect positive weight.
        Teammate: 2-1
        Opponent: 1-3
        Me:       3-2
        Returns 1
    '''
    def matches_team_last_move(self, game, player, move):
        domino, side = move
        new_end_val = domino[0] if domino[0] == game.ends[side] else domino[1]
        last_team_move = board[-2]
        if last_team_move:
            last_dom = last_team_move[0]
            if last_dom.fits_val(new_end_val)
                return 1
        return 0

    '''
        Whether action will match next player's (oponent) move. Expect negative weight.
        Opponent_Next: 4-2
        Teammate: 2-1
        Opponent: 1-3
        Me:       3-4
        Returns 1
    '''
    def matches_next_player_last_move(self, game, player, move):
        domino, side = move
        new_end_val = domino[0] if domino[0] == game.ends[side] else domino[1]
        last_team_move = board[-3]
        if last_team_move:
            last_dom = last_team_move[0]
            if last_dom.fits_val(new_end_val)
                return 1
        return 0

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

    def to_one_hot(self, game, player, move):
        continue

    def train_on_memory(self):
        for m in self.memory:
            game, player, move, reward, next_game, next_move = m
            sa = to_one_hot(game, player, move)
            spap = to_one_hot(next_game, player, next_move)

            self.weights += self.learning_rate*(reward + self.weights @ (spap - sa))*sa

