from agent import Agent
import time

def main():
    agent = Agent()

    num_games = 50
    start = time.time()
    agent.selfplay(num_games)
    agent.train()
    end = time.time()
    print('time to train', end - start)
    agent.playGreedy(num_games)

if __name__ == '__main__':
    main()
