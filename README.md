# Reinforcement-Learning

This is my solution to my MSc Coursework in reinforcement learning. You can find the original source materials in Berkley's CS188 Artificial Intelligence course. 

In this project I implemented the Q-Learning reinforcement learning algorithm from scratch.

**Performance:** <br>
This code enables Pacman to achieve a 100% winrate on the 'smallgrid' map. <br>
This map is trivial for a human to play, but relatively demanding for a reinforcement learning agent. <br>
From my own testing, I have found that after 100 training runs, the average win rate is 81%.
After 500 training runs, the win rate is 100% everytime. 

This is a better result than the standard requirements for the project, which is that Pacman wins 80% of games after 2000 training runs.

## How to run the code yourself

- You will need to copy my file mlLearningAgents.py aswell as all the files in 'Other required files',
place them into **one** folder.
- In your Terminal, cd to the required folder.
- Make sure your python environment is 2.7, as the code only runs in Python 2.7.
- Then type the command below into Terminal:<br>
**python pacman.py -p QLearnAgent -x 500 -n 510 -l smallGrid**

This will run 500 training runs, and then demonstrate Pacman playing 10 live games on the map.

You should see a 100% win rate. 

### The algorithm

In Q-Learning the aim is to learn 'Q-Values'. A Q-Value is the value of a state, action pair. 
Once the learning is complete, the agent will then play by picking the action with the highest Q-Value in each state. 

**How are Q-Values learnt?**
These Q-Values are learnt by playing repeated training runs and updating the Q-Values after each move:
The update rule for Q-Learning is: <br> 
**Q(S,a) = Q(S,a) + alpha * (R(S) + gamma * (Max(a)Q(s’,a’) – Q(S,a))**

Where alpha is the learning rate, gamma is the discount factor, R(S) is the reward for state s, and s’ is the subsequent state.




## An Explanation of the functions I have written and what they do:

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

