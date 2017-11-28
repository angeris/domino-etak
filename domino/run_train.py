from agent import Agent
import time

NUM_PLAYS = 1

def main():
    agent = Agent()

    num_games = 50
    total_games = 0
    
    start = time.time()
    for curr_iter in range(NUM_PLAYS):
        agent.selfplay(num_games)
        agent.train()
        total_games += num_games

    end = time.time()


    print('time to train', end - start)
    agent.play_greedy(num_games)

if __name__ == '__main__':
    main()
