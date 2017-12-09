import numpy as np
from collections import deque
from game import DominosGame
from domino import Domino
from copy import deepcopy
import random
from copy import copy
import os
import pickle as pk

class FeatureAgent:
    def __init__(self, q_maxlen=10000):
        self.memory = deque(maxlen=q_maxlen)
        self.discount = .9
        self.learning_rate = 1e-2
        self.dimension = 38
        self.weights = np.zeros(self.dimension)
        self.weights[0] = 0
        self.total_games = 0
        self.EPSILON_THRESHOLD = 1
        self.eps_discount = .9
        self.min_eps = .05
        self.won_games = 0
        self.all_games = []
        self.epsilon = 1.0
        self.num_iters = 2


    def get_agent_move(self, game, total_games):
        poss_actions = game.get_possible_actions()
        max_score = float('-inf')
        max_action = None

        player = game.curr_player
        if poss_actions[0] is not None:
            if random.random() < self.epsilon:
                return(random.choice(poss_actions))

        for action in poss_actions:
            state_vec = self.to_one_hot(game, player-1, action)
            score = self.weights @ state_vec

            if score > max_score:
                max_score = score
                max_action = action

        return action

    def getGreedyMove(self, game):
        poss_actions = game.get_possible_actions()
        best_a = None
        max_pip_domino = Domino(0,0)
        if poss_actions[0] is not None:
            for action in poss_actions:
                if action[0] >= max_pip_domino:
                    max_pip_domino = action[0]
                    best_a = action
        return best_a

    def save_weights(self, file_name='output'):
        np.save(open('{}.npz'.format(file_name), 'wb'), self.weights)

    def selfplay(self, num_games):
        for i in range(num_games): # play multiple games
            game = DominosGame()
            is_end_state = game.is_end_state()
            while(not is_end_state): 
                curr_game = deepcopy(game)
                curr_player = game.curr_player
                curr_move = self.get_agent_move(game, self.total_games)
                print(curr_move)
                game.move(curr_move)
                is_end_state = game.is_end_state()
                reward = []
                if is_end_state:
                    for player_idx in range(4):
                        reward.append(game.get_score(player_idx))
                    # back propagate is_end_state=True and reward to last moves of past 3 players
                    self.memory[-1][4] = reward
                    self.memory[-1][3] = True
                    self.memory[-2][4] = reward
                    self.memory[-2][3] = True
                    self.memory[-3][4] = reward
                    self.memory[-3][3] = True

                    # print('Reward', reward)

                sar = [curr_game, curr_player, curr_move, is_end_state, reward]
                self.memory.append(sar)

            
    '''
        Test against greedy (shows stats no save to memory)
        Can also test greedy against random by passing random_flag = True
    '''
    def play_greedy(self, num_games, random_flag=False):
        print('Play agent against Greedy')
        agent_total = 0
        greedy_total = 0
        for i in range(num_games): # play multiple games
            # print('Game', i)

            if random.random() < 0.5:   # init starting player
                greedyTurn = True
                greedyPlayer = 0
            else:
                greedyTurn = False
                greedyPlayer = 1

            game = DominosGame(0)
            is_end_state = game.is_end_state()
           
            while(not is_end_state):    # play game
                if greedyTurn:
                    best_a = self.getGreedyMove(game)
                else:   # regular agent turn
                    if random_flag:
                        best_a = self.getRandomMove(game)
                    else:
                        best_a = self.get_agent_move(game, self.total_games)
                game.move(best_a)
                is_end_state = game.is_end_state()
                if is_end_state:
                    scores = []
                    for player_idx in range(4):
                        scores.append(game.get_score(player_idx))
                    print('scores', scores, 'greedyTeam', scores[greedyPlayer], 'agentTeam', scores[greedyPlayer+1])
                    greedy_total += scores[greedyPlayer]
                    agent_total += scores[greedyPlayer+1]
                greedyTurn = not greedyTurn
        
        print('Agent total: {} | Greedy total: {}'.format(agent_total, greedy_total))
        self.total_games += 1
        if self.total_games % self.EPSILON_THRESHOLD == 0:
            self.epsilon *= self.eps_discount
            self.epsilon = max(self.min_eps, self.epsilon)
        self.won_games += agent_total > greedy_total
        self.all_games.append(agent_total > greedy_total)
        if len(self.all_games) % 100 == 0:
            pk.dump({'all_games':self.all_games}, open('all_games_{}'.format(len(self.all_games)), 'wb'))
        last_idx = min(100, len(self.all_games))
        print('Current proportion of games won : {}'.format(float(self.won_games)/self.total_games))
        print('Proportion of last {} games won: {}'.format(last_idx, sum(self.all_games[-last_idx:])/last_idx))

    '''
        Whether action will match previous move of opponent. Expect negative weight.
        Opponent: 1-3
        Me:       3-3
        Return 1
    '''
    def matches_opp_last_move(self, game, player, move):
        if move is None: return 0
        domino = move[0]
        side = move[1]
        new_end_val = domino[0] if domino[1] == game.ends[side] else domino[1]
        if new_end_val == game.ends[0] or new_end_val == game.ends[1]:
            return 1
        return 0

    def matches_opp_pass(self, game, player, move):
        if move is None: return 0
        domino = move[0]
        side = move[1]
        new_end_val = domino[0] if domino[1] == game.ends[side] else domino[1]
        opp_hand = game.get_player_hand((player+1)%4)
        for d in opp_hand:
            if d.fits_val(new_end_val):
                return 0
        return 1

    def matches_opp_2_pass(self, game, player, move):
        if move is None: return 0
        domino = move[0]
        side = move[1]
        new_end_val = domino[0] if domino[1] == game.ends[side] else domino[1]
        opp_hand = game.get_player_hand((player-1)%4)
        for d in opp_hand:
            if d.fits_val(new_end_val):
                return 0
        return 1

    def matches_team_pass(self, game, player, move):
        if move is None: return 0
        domino = move[0]
        side = move[1]
        new_end_val = domino[0] if domino[1] == game.ends[side] else domino[1]
        team_hand = game.get_player_hand((player-2)%4)
        for d in team_hand:
            if d.fits_val(new_end_val):
                return 0
        return 1

    '''
        Whether action will match what teammate opened up for you. Expect positive weight.
        Teammate: 2-1
        Opponent: 1-3
        Me:       3-2
        Returns 1
    '''
    def matches_team_last_move(self, game, player, move):
        if move is None: return 0
        domino = move[0]
        side = move[1]
        new_end_val = domino[0] if domino[1] == game.ends[side] else domino[1]
        if len(game.board) > 2:
            last_team_move = game.board[-2]
            if last_team_move:
                last_dom = last_team_move[0]
                if last_dom.fits_val(new_end_val):
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
        if move is None: return 0
        domino = move[0]
        side = move[1]
        new_end_val = domino[0] if domino[1] == game.ends[side] else domino[1]
        if len(game.board) > 3:
            last_team_move = game.board[-3]
            if last_team_move:
                last_dom = last_team_move[0]
                if last_dom.fits_val(new_end_val):
                    return 1
        return 0

    def last_k_pip(self, game, player, move):
        remaining_dominoes = 7
        if move is None: return 0
        curr_domino = move[0]
        for d in game.board:
            if d is None:
                continue
            if curr_domino.fits(d[0]):
                remaining_dominoes -= 1

        for d in game.get_player_hand(player):
            if d == move[0]:
                continue
            if move[0].fits(d):
                remaining_dominoes -= 1

        return remaining_dominoes == 1

    def num_dom_remaining_leftopp(self,game,player,move):
        player_opp = (player - 1) % 4
        curr_hand = game.get_player_hand(player_opp)
        curr_hand_len = [0,0,0,0,0,0,0]
        curr_hand_len[len(curr_hand) -1] = 1
        return curr_hand_len

    def num_dom_remaining_rightopp(self,game,player,move):
        player_opp = (player + 1) % 4
        curr_hand = game.get_player_hand(player_opp)
        curr_hand_len = [0,0,0,0,0,0,0]
        curr_hand_len[len(curr_hand)-1] = 1
        return curr_hand_len

    def num_dom_remaining_teammate(self,game,player,move):
        teammate = (player + 2) % 4
        curr_hand = game.get_player_hand(teammate)
        curr_hand_len = [0,0,0,0,0,0,0]
        curr_hand_len[len(curr_hand)-1] = 1
        return curr_hand_len

    def is_greedy_move(self, game, player, move):
        poss_actions = game.get_possible_actions()
        best_a = None
        max_pip_domino = Domino(0,0)
        if poss_actions[0] is not None:
            for action in poss_actions:
                if action[0] >= max_pip_domino:
                    max_pip_domino = action[0]
                    best_a = action
            if best_a == move:
                return True
        return False

    def num_dom_inhand_matches(self, game, player, move):
        if move is None: return [0,0,0,0,0,0,0]
        board_pip = game.ends[move[1]]
        if move[0].value[0] == board_pip:
            face_out_pip = move[0].value[0]
        else:
            face_out_pip = move[0].value[1]

        num_matches = 0
        curr_hand = game.get_player_hand(player)
        poss_actions = game.get_possible_actions()
        if len(curr_hand) > 0 and curr_hand[0] is not None:
            for dom in curr_hand:
                if face_out_pip == dom[0] or face_out_pip == dom[1]:
                    num_matches += 1
        num_matches_list = [0,0,0,0,0,0,0]
        num_matches_list[num_matches] = 1
        return num_matches_list

    def total_pip(self, game, player, move):
        if move is None: return 0
        return move[0].pip_val

    def to_one_hot(self, game, player, move):
        opp_move = self.matches_opp_last_move(game, player, move)
        team_move = self.matches_team_last_move(game, player, move)
        n_player_move = self.matches_next_player_last_move(game, player, move)
        last_k_pip = self.last_k_pip(game, player, move)
        is_greedy_move = self.is_greedy_move(game, player, move)
        num_match = self.num_dom_inhand_matches(game, player, move)
        t_pip = self.total_pip(game, player, move)
        num_dom_leftopp = self.num_dom_remaining_leftopp(game, player, move)
        num_dom_rightopp = self.num_dom_remaining_rightopp(game, player, move)
        num_dom_remaining_teammate = self.num_dom_remaining_teammate(game, player,move)
        matches_opp_pass = self.matches_opp_pass(game, player, move)
        matches_opp2_pass = self.matches_opp_2_pass(game, player, move)
        matches_team_pass = self.matches_team_pass(game, player, move)
        return np.r_[is_greedy_move, team_move, n_player_move, last_k_pip,
                     opp_move, num_match, t_pip, num_dom_leftopp, num_dom_rightopp, num_dom_remaining_teammate, 
                     matches_opp_pass, matches_opp2_pass, matches_team_pass, 1]

    def train_on_memory(self):
        for it in range(self.num_iters):
            for perspective_player in range(4):
                if perspective_player % 2 == 1:
                    continue
                curr_mem = None
                for m in self.memory:
                    [game, player, move, is_end, reward] = m
                    if player == perspective_player:
                        # print('next', m)
                        if curr_mem is None:   # first move of game (s,a)
                            curr_mem = m
                        else:
                            sa = self.to_one_hot(curr_mem[0], curr_mem[1], curr_mem[2]) # game, player, move
                            curr_end = curr_mem[3]
                            # print('curr', curr_mem)
                            if not curr_end: # not end state
                                spap = self.to_one_hot(game, player, move)
                                self.weights += self.learning_rate*(self.weights @ (self.discount * spap - sa))*sa
                                curr_mem = m

                            else:
                                # print('Reward of sa', curr_mem[4])
                                self.weights += self.learning_rate*(curr_mem[4][player] - self.weights @ sa) # do not consider spap
                                curr_mem = m
                if curr_mem[3]: # case of last sar in memory
                    # print('curr', curr_mem)
                    # print('Reward of sa', curr_mem[4])
                    self.weights += self.learning_rate*(curr_mem[4][player] - self.weights @ sa) # do not consider spap
        
        print('Weights:')
        print('Is greedy move')
        print(self.weights[0])
        print('Matches team last move')
        print(self.weights[1])
        print('Matches next player last move')
        print(self.weights[2])
        print('Do I have the last set of this pip?')
        print(self.weights[3])
        print('Matches next opponent\'s last move')
        print(self.weights[4])
        print('Number of matches in current hand')
        print(self.weights[5:12])
        print('Number of total pips of current domino')
        print(self.weights[12])
        print('Number of dominoes left in left opponent\'s hand')
        print(self.weights[13:20])
        print('Number of dominoes left in right opponent\'s hand')
        print(self.weights[20:27])
        print('Number of dominoes in teammate\'s hand')
        print(self.weights[27:34])
        print('Does it match the pass of the next opponent?')
        print(self.weights[34])
        print('Does it match the pass of the previous opponent?')
        print(self.weights[35])
        print('Does it match the pass of my teammate?')
        print(self.weights[36])
        print('Constant offset:')
        print(self.weights[37])

    '''
        Save agent to memory as play against greedy and print states
    '''
    def selfplay_greedy(self, num_games):
        print('Play agent against Greedy + Save to memory')
        print('Epsilon', self.epsilon)
        agent_total = 0
        greedy_total = 0
        agent_won_games = 0
        for i in range(num_games): # play multiple games
            # if random.random() < 0.5:   # init starting player
                # greedyTurn = True
                # greedyPlayer = 0
            # else:
                # greedyTurn = False
                # greedyPlayer = 1
            greedyTurn = False
            greedyPlayer = 1

            game = DominosGame(0)
            is_end_state = game.is_end_state()
           
            while(not is_end_state):    # play game
                game = deepcopy(game)
                curr_player = game.curr_player
                
                if greedyTurn:
                    best_a = self.getGreedyMove(game)
                else:   # regular agent turn
                    best_a = self.get_agent_move(game, self.total_games)
                game.move(best_a)
                
                is_end_state = game.is_end_state()
                scores = []
                if is_end_state:
                    for player_idx in range(4):
                        scores.append(game.get_score(player_idx))

                    # back propogate scores and end state
                    self.memory[-1][4] = scores
                    self.memory[-1][3] = True

                    # print('scores', scores, 'greedyTeam', scores[greedyPlayer], 'agentTeam', scores[greedyPlayer+1])
                    greedy_total += scores[greedyPlayer]
                    agent_total += scores[greedyPlayer+1]
                    agent_won_games += scores[greedyPlayer+1] > scores[greedyPlayer]

                # save agent's plays to memory
                if not greedyTurn:
                    sar = [game, curr_player, best_a, is_end_state, scores]
                    self.memory.append(sar)
                greedyTurn = not greedyTurn
        
        print('Agent pip total: {} | Greedy pip total: {}'.format(agent_total, greedy_total))
        self.total_games += 1
        if self.total_games % self.EPSILON_THRESHOLD == 0:
            self.epsilon *= self.eps_discount
            self.epsilon = max(self.min_eps, self.epsilon)
        self.won_games += agent_total > greedy_total
        self.all_games.append(agent_total > greedy_total)
        if len(self.all_games) % 100 == 0:
            pk.dump({'all_games':self.all_games}, open('all_games_{}'.format(len(self.all_games)), 'wb'))
        last_idx = min(100, len(self.all_games))
        print('Current proportion of games won : {}'.format(float(self.won_games)/self.total_games))
        print('Proportion of last {} games won: {}'.format(last_idx, sum(self.all_games[-last_idx:])/last_idx))
        print('Proportion of indiv games won:', agent_won_games/num_games)
