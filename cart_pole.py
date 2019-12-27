import numpy as np
import gym



"""
    Global variables 
"""
nval = 6 # number of values for each observation variable
N = nval ** 4  # total number of states
A = (0,1) # actions
episodes = 100 #total number of episodes to play
env = gym.make('CartPole-v1') # create cart pole environment
gam = 0.5
epl = 0.5
'''
  The presentation of the different states 
'''
def discretise(x,mini,maxi):
    # discretise x
    # return an integer between 0 and nval - 1
    if x<mini: x=mini
    if x>maxi: x=maxi
    return int(np.floor((x-mini)*nval/(maxi-mini+0.0001)))

def observation_vers_etat(observation):
    pos = discretise(observation[0],mini=-1,maxi=1)
    vel = discretise(observation[1],mini=-1,maxi=1)
    angle = discretise(observation[2],mini=-1,maxi=1)
    pos2 = discretise(observation[3],mini=-1,maxi=1)
    return pos + vel*nval + angle*nval*nval + pos2*nval*nval*nval

'''
    Monte Carlo algorithm 
'''


def MonteCarlo_OnPolicy_FirstVisit():
    """
    Monte carlo algorithm with On-policy - First visit
    :return: the average number of iterations per episode
    """
    '''
        initialize Occ(s,a) : how many times we visited (s,a) (first iteration)
        initialize Q(s,a): the value function for (s,a)
        initialize Acc(s,a): reward function
        initialize p(s,a): politic 
        initialize episode : visited states 
        
    '''
    Occ, Q, Acc, p = {}, {}, {}, {}
    for i in range(episodes):
        '''
            play an episode 
        '''
        episode = []
        nbIt = 0
        done = False
        observation = env.reset() # reset all variables to the initial state
        while not done:
            s = observation_vers_etat(observation)
            if (s,0) not in p:
                p[(s, 0)], p[(s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
            alt = np.random.rand(1)[0]
            a = 0 if alt <= p[(s,0)] else 1
            observation, reward, done, info = env.step(a)
            episode.append((s,a))
            env.render()
            nbIt += 1
        print("Episode ", i, " terminé après itérations".format(nbIt))
        state_visited = set()
        for t in range(len(episode)):
            item = episode[t]
            if item not in state_visited:
                state_visited.add(item)
                G = (1- gam ** (nbIt - t)) / (1 - gam)
                Acc[item] = G if item not in Acc else Acc[item] + G
                Occ[item] = 1 if item not in Occ else Occ[item] + 1
                Q[item] = Acc[item] / Occ[item]
        for state, action in state_visited:
            best_a_state = 0 if Q[(state,0)] > Q[(state,1)] else 1
            for action in A:
                p[(state,action)] = 1 - epl + epl / 2 if best_a_state == action else epl / 2













nbIt=0
done=False
while not done:
    observation, reward, done, info = env.step(np.random.randint(2))
    env.render()
    nbIt+=1
print("Episode terminé après itérations".format(nbIt))
env.close()