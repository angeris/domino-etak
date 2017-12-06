from agent import Agent
import time

NUM_PLAYS = 20000

def main():


    agent = Agent()

    num_games = 10
    total_games = 0
    
    start = time.time()

    
    
    for curr_iter in range(NUM_PLAYS):
        print('Curr_iter', curr_iter)
        # agent.selfplay(num_games)
        agent.selfplay_greedy(num_games)    # adds agents moves to memory
        agent.train()                       # trains on self.memory
        total_games += num_games
        # agent.play_greedy(2*num_games)    # testing against greedy
        if (curr_iter + 1)%200==0:
            agent.save_curr_network('iter_{}'.format(curr_iter+1))
            print('total number of games played so far : {}'.format(total_games))
    
    end = time.time()
    agent.save_curr_network('output')


    print('time to train', end - start)
    agent.play_greedy(2*num_games)
    
if __name__ == '__main__':
    main()
