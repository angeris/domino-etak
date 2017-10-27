from domino.game import DominosGame
from random import seed, choice

seed(1) # Set seed for now

curr_game = DominosGame()

print('Initial board')
for i in range(4):
    print(curr_game.get_player_hand(i))

print('Performing some random (valid) moves')
print('Initial ends of the board: {}'.format(curr_game.ends))
for _ in range(5):
    print('Current player : {}'.format(curr_game.curr_player))
    possible_moves = curr_game.get_possible_actions()
    print('Possible actions : {}'.format(possible_moves))
    chosen_action = choice(possible_moves)
    print('Chosen action : {}'.format(chosen_action))
    curr_game.move(chosen_action)
    print('New ends : {}'.format(curr_game.ends))


print('Playing until the end')
while True:
    curr_game.move(choice(curr_game.get_possible_actions()))
    