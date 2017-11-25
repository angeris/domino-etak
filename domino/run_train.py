from agent import Agent

def main():
    agent = Agent()

    num_games = 200
    agent.selfplay(num_games)
    agent.train()
    agent.playGreedy(num_games)

if __name__ == '__main__':
    main()
