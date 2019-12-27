import numpy as np
import gym

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







nval = 6 # number of values for each observation variable
N= nval ** 4  # total number of states
env = gym.make('CartPole-v1')
env.reset()
nbIt=0
done=False
while not done:
    observation, reward, done, info = env.step(np.random.randint(2))
    env.render()
    nbIt+=1
print("Episode terminé après itérations".format(nbIt))
env.close()