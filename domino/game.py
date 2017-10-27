from collections import deque
from random import shuffle
from domino.domino import Domino

class DominosGame:
    # TODO: Implement arbitrary domino scenarios
    def __init__(self, initial_player=None, player_dominoes=None):
        self.board = []
        self.domino_set = set()
        self.player_set = [None for _ in range(4)]
        self.ends = [None, None]

        if player_dominoes is None:
            # Generate usual double six set and randomly assign to players
            all_dominoes = [Domino(a, b) for a in range(7) for b in range(a, 7)]
            self.domino_set = set(all_dominoes)
            shuffle(all_dominoes)
            for i in range(4):
                self.player_set[i] = set(all_dominoes[i*7:(i+1)*7])

        if initial_player is not None:
            self.curr_player = initial_player
            self.initial_player = initial_player
            return

        for player, dominos in enumerate(self.player_set):
            if (6,6) in dominos:
                self.curr_player = (player+1) % 4
                self.initial_player = player
                self.ends = [6,6]
                self.player_set[player].remove(Domino(6,6))
                return

        assert False

    # To do RL, we require the following: a tentative_move (state s, action a,
    # current_player) which returns a reward, a given_move such that the move
    # is performed in the game and gives the reward, and an is_end_state where
    # we can query if the current game has ended, and a method to get our current
    # possible actions.

    def tentative_move(self, action):
        """Act as if curr_player is about to put down domino action; return reward

        Args:
            action (tuple): A (domino, side_int) pair, where side_int is the side
                on which the domino should be played.

        """
        prev_ends = self.ends
        prev_player = self.curr_player
        self.move(action)
        curr_score = 0 if not self.is_end_state() else self.get_score(prev_player)

        # Undo move
        self.board.pop()
        self.ends = prev_ends
        self.curr_player = prev_player
        self.player_set[self.curr_player].add(domino)

        return curr_score


    def move(self, action):
        """Checks if the move is possible and performs it
        """
        if action is None:
            self.board.append(action)
            self.curr_player = (self.curr_player + 1) % 4

        domino, side = action
        assert 0 <= side <= 1

        # We can probably remove this later, but for debugging this might be useful
        if domino not in self.player_set[self.curr_player]:
            assert False
        if domino.fits_val(self.ends[side]):
            self.board.append(action)
            self.ends[side] = domino[0] if self.ends[side] == domino[1] else domino[1]
            self.player_set[self.curr_player].remove(domino)
            self.curr_player = (self.curr_player + 1) % 4
            return

        assert False

    def get_possible_actions(self):
        """Returns a list of all possible actions
        """
        possible_actions = []

        for side in [0,1]:
            for domino in self.player_set[self.curr_player]:
                if domino.fits_val(self.ends[side]):
                    possible_actions.append((domino, side))

        return possible_actions if possible_actions else [None]

    def get_player_hand(self, player=None):
        if player is None:
            return list(self.player_set[self.curr_player])
        return list(self.player_set[player])

    def is_end_state(self):
        """Checks if we're in an ending state
        """
        return self._end_player() or self._end_block()
        
    
    def _end_player(self):
        """Checks if a player has no more dominoes
        """
        return any(not d_list for d_list in self.player_set)

    def _which_end_player(self):
        """Same as _end_player, except returns the player who won or None
        """
        for player, d_list in enumerate(self.player_set):
            if not d_list:
                return player
        return None

    def _end_block(self):
        """Checks if the game ends with a block (given that all players have >= 1 domino)
        """
        for player in self.player_set:
            if any((domino.fits_val(self.ends[0]) or
                    domino.fits_val(self.ends[1])) for domino in player):
                return False
        
        return True

    def get_score(self, player):
        """Get the score from `player`'s perspective, given that the round has ended.
        """
        assert 0 <= player <= 3

        player_end = self._which_end_player()

        if player_end is not None:
            if (player_end == player or 
                player_end == (player+2) % 4):
                return (self._get_player_score((player+1) % 4) + 
                        self._get_player_score((player+3) % 4))
            return 0

        return max(self._get_player_score((player+1) % 4) +
                   self._get_player_score((player+3) % 4) -
                   self._get_player_score(player) - 
                   self._get_player_score((player+2) % 4),
                   0)


    def _get_player_score(self, player):
        """Get the total number of pips of the current player
        """
        return sum(d.pip_val for d in self.player_set[player])