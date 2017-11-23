from agent import Agent

def main():
    agent = Agent()

    num_games = 10
    agent.selfplay(num_games)
    agent.train()

if __name__ == '__main__':
    main()
