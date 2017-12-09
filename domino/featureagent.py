import numpy as np
from collections import deque
from game import DominosGame
from domino import Domino
from copy import deepcopy
import random

class FeatureAgent:
    def __init__(self, q_maxlen=10000):
        self.memory = deque(maxlen=q_maxlen)
        self.discount = .99
        self.learning_rate = 0.01
        self.dimension = 14
        self.weights = np.zeros(self.dimension)
        self.weights[0] = 1
        self.total_games = 0
        self.EPSILON_THRESHOLD = 100
        self.won_games = 0
        self.all_games = []
        self.epsilon = 1.0
        self.num_iters = 10


    def get_agent_move(self, game, total_games):
        poss_actions = game.get_possible_actions()
        max_score = float('-inf')
        max_action = None

        player = game.curr_player

        for action in poss_actions:
            state_vec = self.to_one_hot(game, player, action)
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

                    print('Reward', reward)

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
            self.epsilon *= 0.5
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
        if not move: return 0
        domino = move[0]
        side = move[1]
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
        if not move: return 0
        domino = move[0]
        side = move[1]
        new_end_val = domino[0] if domino[0] == game.ends[side] else domino[1]
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
        if not move: return 0
        domino = move[0]
        side = move[1]
        new_end_val = domino[0] if domino[0] == game.ends[side] else domino[1]
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
        if curr_hand[0] is not None:
            for dom in curr_hand:
                if face_out_pip == dom[0] or face_out_pip == dom[1]:
                    num_matches += 1
        num_matches_list = [0,0,0,0,0,0,0]
        num_matches_list[num_matches] = 1
        return num_matches_list

    def total_pip(self, game, hand, player, move):
        if move is None: return 0
        return move[0].pip_val

    def to_one_hot(self, game, player, move):
        opp_move = self.matches_opp_last_move(game, player, move)
        team_move = self.matches_team_last_move(game, player, move)
        n_player_move = self.matches_next_player_last_move(game, player, move)
        last_k_pip = self.last_k_pip(game, player, move)
        is_greedy_move = self.is_greedy_move(game, player, move)
        num_match = self.num_dom_inhand_matches(game, player, move)
        hand = game.get_player_hand
        t_pip = self.total_pip(game, hand, player, move)

        return np.r_[is_greedy_move, team_move, n_player_move, last_k_pip,
                     opp_move, num_match, t_pip, 1]

    def train_on_memory(self):

        for it in range(self.num_iters):
            for perspective_player in range(4):
                for m in self.memory:
                    [game, player, move, is_end, reward] = m
                    print(m)
                    
                    curr_mem = [None, None, None, None, None]

                    if player == perspective_player:
                        if curr_game is None:   # first move of game (s,a)
                            curr_mem = m
                        else:
                            sa = self.to_one_hot(curr_mem[0], curr_mem[1], curr_mem[2]) # game, player, move
                            if not curr_mem[3]:
                                spap = self.to_one_hot(game, player, move)
                                curr_mem = m

                            if curr_end:
                                self.weights += self.learning_rate*(reward[player] - self.weights @ sa)
                                curr_game = None
                                curr_player = None
                                curr_end = None
                                curr_move = None
                                continue

                        self.weights += self.learning_rate*(self.weights @ (spap - sa))*sa

