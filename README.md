# entabular cart pole

The purpose of this project is to implement an entabular cart pole solution using some reinforcement learning algorithms.
First step is to represent the environment as a decisional markovian process: 
* States S: the state is represented by an integer that describes four main elements of the problem ( the position of cart, velocity, angle, the position of the pole). We need to discretize the different observation’s elements in order to obtain a finite number of states. To improve the accuracy of the results, it’s better to set “nval” to a great value. For this solution, I chose to set nval = 22. This has shown better performances in different RL algorithms comparing to nval = 6. This is because each time we increment the value of nval, we have more states that describe precisely the right position of the pole. 
* Actions: there are two actions based on the cart movement( 0 : move right, 1: move left) 
* r : if the pole angle of the beam vertically is less than 12°, the reward = 1 else 0
* T :  unknown  
* ε :  used for exploring different states in the beginning of the episode. The value of ε decrease while playing an episode. The initial value of ε = 0.1 

 #### 1. Monte carlo “on-policy”:

* I implemented both “First visit” and “Every visit” version, the plot below compares between the two algorithms by showing the average number of iterations (per 100 episodes) in a total of 100 K episodes. 

![](/FirstVsEvery_100.png)
 
Blue : First visit
Orange: Every visit 
The discount factor is equal to 1

We notice that the average number of iterations for “Every visit” and “First visit” algorithm increases quickly in the beginning. However, it takes a considerable time to converge to the optimum average number (v*). At the end, “First visit” algorithm shows better performances than “Every visit” version of Monte Carlo. It converges to an average of 160 iterations in 100 episodes. 

* The second plot compares between different results of Monte Carlo “first visit” for different values of the discount factor. the discount factor is a measure of how far ahead in time the algorithm looks. To prioritise rewards in the distant future, we keep the value closer to one. 

![](/gam.png)

Blue: 0.6 -> purple -> red : 0.8  -> Orange -> Green: 1

The figure above compares between the average number of iterations for different values of the discount factors: 0,6; 0,7; …; 1. When the value of the discount factor is closer to 1, the average number of iterations increase quickly compared to smaller values. After running thousands of episodes, the different results converge to almost the same numbers. The discount factor allows to prioritize rewards that can be obtained in the future. When its value is closer to 1, the valuation function add weight to the possible future rewards which explains the increasing curves in the beginning of the experience. After the agent learns more about its environment, this discount factor does not influence the learning process, and the agent is able to maximize the results and converge to an average number of iterations. 

#### 2. SARSA and Q-Learning:

In this part, we will compare between two TD algorithms: SARSA and Q-Learning. In Q-learning, the agent starts out in state S, performs action A, sees what the highest possible reward is for taking any action from its new state, and updates its value for the state S-action A pair based on this new highest possible value. In SARSA, the agent starts in state S, takes action A and gets a reward, then moves to state S’, takes action B and gets a reward, and then goes back to update the value for S-A based on the actual value of the reward it received from taking action B.

 * α  is the learning rate that gives a higher importance to the recent rewards compared to the old ones. 
 
 ![](/sarsavsQlearning.png)
 
Orange: Q-Learning
Blue: SARSA

Clearly, Q-learning algorithms presents better results compared to SARSA implementation. The average iterations number for Q-Learning is 350 , contrary to SARSA that converges to almost 280 per 100 episodes. 



