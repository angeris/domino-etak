from collections import deque
from random import shuffle
from domino import Domino

class DominosGame:

    def __init__(self, initial_player=None, player_dominoes=None):
        self.board = [] # Using list, since it's too small for a deque to matter
        self.domino_set = set()
        self.player_set = [[] for _ in range(4)]

        if player_dominoes is None:
            # Generate usual double six set and randomly assign to players
            all_dominoes = [Domino(a, b) for a in range(1, 7) for b in range(1, 7)]
            shuffle(all_dominoes)
            for i in range(4):
                self.player_set[i] = all_dominoes[i*7:(i+1)*7]


        if initial_player is not None:
            self.curr_player = initial_player
            self.initial_player = initial_player
            return

        for player, dominos in enumerate(self.player_set):
            if (6,6) in dominos:
                self.curr_player = player
                self.initial_player = player
                break


    # To do RL, we require the following: a tentative_move (state s, action a,
    # current_player) which returns a reward, a given_move such that the move
    # is performed in the game and gives the reward, and an is_end_state where
    # we can query if the current game has ended, and a method to get our current
    # possible actions.

    def tentative_move(self, action):
        """Act as if curr_player is about to put down domino action

        Args:
            action (tuple): A (domino, side_int) pair, where side_int is the side
                on which the domino should be played.

        """
        pass

    def is_end_state(self):
        pass

    

