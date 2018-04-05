# Reinforcement-Learning

This is my solution to the Pacman reinforcement learning project - Included in Berkley's CS188 Artificial Intelligence materials.

In this project I implemented the Q-Learning reinforcement learning algorithm. 

In Q-Learning, the agent aims to learn all the Q-Values. A Q-Value is the value of a state, action pair.

The update rule for Q-Learning is: <br> 
**Q(S,a) = Q(S,a) + alpha * (R(S) + gamma * (Max(a)Q(s’,a’) – Q(S,a))**

Where alpha is the learning rate, gamma is the discount factor, R(S) is the reward for state s, and s’ is the subsequent state.


### Performance:
The main metric to evaluate performance is the winrate. <br>
After just 100 training runs, I found the average winrate to be 81%.

After 500 games, Pacman will never lose a game. 

This is a better result than the standard requirements for the project, which is that Pacman wins 80% of games after 2000 training runs.

### How does reinforcement learning work?
Reinforcement learning is the branch of machine learning which focuses on how agents can learn how to better perform a specific activity overtime, through many repeated training runs.

The key idea behind reinforcement learing is the use of 'rewards' and 'punishments' for certain outcomes. The agents goal is to figure out how to generate the maximum reward. Over many repeated runs, it begins to learn which sequences of events and which actions, generate more rewards, and which generate poor rewards. It then carries out the actions which lead to better rewards more often. This is the basic idea behind how an agent learns the best strategy overtime. 


## My functions:

Storage: My Q-Values are stored in self.Q_Table. This is a dictionary. The format is
{state: {‘North’: Q, ‘East’: Q, ‘South’: Q, ‘West’:Q}}

### ComputeReward(previous_state, current_state): 
This function computes the reward for the previous state.
Quite simply, this is the difference between the current score and the previous score. And if it is the first move of the game, then the reward is just the current score. 
I found this proved to be a fair reward system for incentivising Pacman’s playstyle. The rewards based on score are as follows:

-500 for losing (hitting a ghost) <br>
+500 for winning (getting the last food) <br>
+9 for collecting food <br>
-1 for anything else

### setPreviousState(state, action)
A key feature of Q-Learning is that we update the Q-Value after moving to the next state.
Therefore, we need to keep a track of what the previous state and action pair was.
This function enables this by setting self.previous_state and self.previous_action to the values passed in as arguments. 

### UpdateQValue(self, previous_state, previous_action, current_state)

This function is responsible for updating the Q-value for the previous state/action pair. 
It gets the values for all the necessary components; alpha, gamma, the old Q(s,a) value, the next best Q value for the next state Q(s’,a’), and the reward R(S).
It then applies the Q-Learning update rule to calculate the new Q(s,a) value. 
Finally it updates this value in the self.Q_Table.

### getQValue(self,state,action):
This function returns the Q-Value for a specific state, action pair.
If the state has not been seen before, it creates an instance of the state in the Q_Table, with all Q-Values initialised to 0.

### getBest(self,state)
This one function can return both the best Action and Q-Value for a state.
It returns a dictionary {‘action’: bestAction, ‘Q_Value’: bestQ} so you can choose the output. 
It only returns the best Q Value/Action for legal moves. 
If the state is a terminal state, It returns ‘None’ and 0.


### DecreaseAlpha(self), DecreaseEpsilon(self)
In theory it is ideal to gradually reduce the learning rate and epsilon over time. I decided to implement this into my code. These functions set Alpha to 100/(400 + N), and Epsilon to (50/1000 + N), where N is how many training runs have occurred, and the functions are called at the end of the game in the final() function.

### Exploration(self)
This function implements the epsilon greedy method for determining whether if Pacman is going to explore or exploit. 
It achieves this by getting a random x from a uniform distribution (0,1). If x is less than epsilon, it returns True.  Therefore, Pacman explores with epsilon probability as required. 


### FoodLocations(state):
This function parses the default representation of food locations from state.getFood(). It exploits the fact that we know there are only 2 foods in the same locations each game. My code outputs a Boolean tuple which indicates which foods are still left to eat in the game. 

### ParseState(self,state):

This function converts any state into a readable representation of the features that I want to include in my state-space. This reduced, readable format of state as the state in the Q_Table. When deciding on the state-space, I decided that the best features to include were PacmanLocation, GhostLocation, and FoodLocations. 

