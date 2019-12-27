import numpy as np
import gym

nval = 6 # nombre de valeurs discrètes qu’une variable peut prendre
N= nval ** 4  # Puisqu’il y a 4 variables, la taille de l’espace est nval4
def discretise(x,mini,maxi):
    # discretise x
    # renvoie un entier entre 0 et nval-1
    if x<mini: x=mini
    if x>maxi: x=maxi
    return int(np.floor((x-mini)*nval/(maxi-mini+0.0001)))

def observation_vers_etat(observation):
    pos = discretise(observation[0],mini=-1,maxi=1)
    vel = discretise(observation[1],mini=-1,maxi=1)
    angle = discretise(observation[2],mini=-1,maxi=1)
    pos2 = discretise(observation[3],mini=-1,maxi=1)
    return pos + vel*nval + angle*nval*nval + pos2*nval*nval*nval


