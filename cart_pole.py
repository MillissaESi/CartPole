import numpy as np
import gym


class cartPole():
    def __init__(self, nval,  episodes):
        self.nval = nval # number of values for each observation variable
        self.N = self.nval ** 4
        self.episodes = episodes #total number of episodes to play
        self.A = (0,1)  # actions
        self.env = gym.make('CartPole-v1')

    '''
      The presentation of the different states 
    '''

    def discretise(self, x, mini, maxi):
        # discretise x
        # return an integer between 0 and nval - 1
        if x < mini: x = mini
        if x > maxi: x = maxi
        return int(np.floor((x - mini) * nval / (maxi - mini + 0.0001)))

    def observation_vers_etat(self, observation):
        pos = self.discretise(observation[0], mini=-1, maxi=1)
        vel = self.discretise(observation[1], mini=-1, maxi=1)
        angle = self.discretise(observation[2], mini=-1, maxi=1)
        pos2 = self.discretise(observation[3], mini=-1, maxi=1)
        return pos + vel * nval + angle * nval * nval + pos2 * nval * nval * nval

    '''
        Monte Carlo algorithms 
    '''

    def MonteCarlo_OnPolicy_FirstVisit(self,gam, epl):
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
        for i in range(self.episodes):
            '''
                play an episode 
            '''
            episode = []
            nbIt = 0
            done = False
            observation = self.env.reset()  # reset all variables to the initial state
            while not done:
                s = self.observation_vers_etat(observation)
                if (s, 0) not in p:
                    p[(s, 0)], p[(s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
                alt = np.random.rand(1)[0]
                a = 0 if alt <= p[(s, 0)] else 1
                observation, reward, done, info = self.env.step(a)
                episode.append((s, a))
                self.env.render()
                nbIt += 1
            print("Episode ", i, " terminé après itérations ", nbIt)
            state_visited = set()
            for t in range(len(episode)):
                item = episode[t]
                if item not in state_visited:
                    state_visited.add(item)
                    G = (1 - gam ** (nbIt - t)) / (1 - gam)
                    Acc[item] = G if item not in Acc else Acc[item] + G
                    Occ[item] = 1 if item not in Occ else Occ[item] + 1
                    Q[item] = Acc[item] / Occ[item]
            for state, action in state_visited:
                best_a_state = 0 if ((state, 1) not in Q or ((state, 0) in Q and Q[(state, 0)] > Q[(state, 1)])) else 1
                for action in self.A:
                    p[(state, action)] = 1 - epl + epl / 2 if best_a_state == action else epl / 2

    def MonteCarlo_OnPolicy_EveryVisit(self, gam, epl):
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
        for i in range(self.episodes):
            '''
                play an episode 
            '''
            episode = []
            nbIt = 0
            done = False
            observation = self.env.reset()  # reset all variables to the initial state
            while not done:
                s = self.observation_vers_etat(observation)
                if (s, 0) not in p:
                    p[(s, 0)], p[(s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
                alt = np.random.rand(1)[0]
                a = 0 if alt <= p[(s, 0)] else 1
                observation, reward, done, info = self.env.step(a)
                episode.append((s, a))
                self.env.render()
                nbIt += 1
            print("Episode ", i, " terminé après itérations ", nbIt)
            state_visited = set()
            for t in range(len(episode)):
                item = episode[t]
                if item not in state_visited:
                    state_visited.add(item)
                G = (1 - gam ** (nbIt - t)) / (1 - gam)
                Acc[item] = G if item not in Acc else Acc[item] + G
                Occ[item] = 1 if item not in Occ else Occ[item] + 1
                Q[item] = Acc[item] / Occ[item]
            for state, action in state_visited:
                best_a_state = 0 if ((state, 1) not in Q or ((state, 0) in Q and Q[(state, 0)] > Q[(state, 1)])) else 1
                for action in [0, 1]:
                    p[(state, action)] = 1 - epl + epl / 2 if best_a_state == action else epl / 2

    def sarsa_OnPolicy(self, alpha, gam, epl):
        Q = {}
        for i in range(self.episodes):
            done = False
            observation = self.env.reset()  # reset all variables to the initial state
            s = self.observation_vers_etat(observation)
            if (s, 0) not in Q:
                Q[(s, 0)], Q[(s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
            alt = np.random.rand(1)[0]
            best_a = 0 if Q[(s, 0)] > Q[(s, 1)] else 1
            a = best_a if alt > epl else np.random.randint(2)
            nbInt = 0
            while not done:
                # play (s,a)
                observation, reward, done, info = self.env.step(a)
                next_s = self.observation_vers_etat(observation)
                if (next_s, 0) not in Q:
                    Q[(next_s, 0)], Q[(next_s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
                alt = np.random.rand(1)[0]
                best_a = 0 if Q[(next_s, 0)] > Q[(next_s, 1)] else 1
                next_a = best_a if alt > epl else np.random.randint(2)
                Q[(s,a)] += alpha * (reward + gam * Q[(next_s,next_a)] - Q[(s,a)])
                s = next_s
                a = next_a
                nbInt += 1
                #self.env.render()
            print("Episode ", i, " terminé après itérations ", nbInt)
    def QLearning(self, alpha, gam, epl):
        Q = {}
        for i in range(self.episodes):
            done = False
            observation = self.env.reset()  # reset all variables to the initial state
            s = self.observation_vers_etat(observation)
            if (s, 0) not in Q:
                Q[(s, 0)], Q[(s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
            alt = np.random.rand(1)[0]
            best_a = 0 if Q[(s, 0)] > Q[(s, 1)] else 1
            a = best_a if alt > epl else np.random.randint(2)
            nbInt = 0
            while not done:
                # play (s,a)
                observation, reward, done, info = self.env.step(a)
                next_s = self.observation_vers_etat(observation)
                if (next_s, 0) not in Q:
                    Q[(next_s, 0)], Q[(next_s, 1)] = np.random.rand(1)[0], 1 - np.random.rand(1)[0]
                alt = np.random.rand(1)[0]
                best_a = 0 if Q[(next_s, 0)] > Q[(next_s, 1)] else 1
                next_a = best_a if alt > epl else np.random.randint(2)
                Q[(s,a)] += alpha * (reward + gam * Q[(next_s, best_a)] - Q[(s,a)])
                s = next_s
                a = next_a
                nbInt += 1
                #self.env.render()
            print("Episode ", i, " terminé après itérations ", nbInt)







nval = 6
N = nval ** 4
episodes = 1000000
gam = 0.1
alpha = 0.1
epl = 0.1
cartPole = cartPole(nval,episodes)
#cartPole.MonteCarlo_OnPolicy_EveryVisit(epl,gam)
#cartPole.sarsa_OnPolicy(alpha, gam, epl)
cartPole.QLearning(alpha,gam,epl)
cartPole.env.close()












