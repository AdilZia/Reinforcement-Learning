# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np


# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.25, epsilon=0.05, gamma=0.8,numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Initialise empty dictionary to store Q_table 
        self.Q_Table = {}
        # Initialise 'previous state' and 'action' to None
        self.previous_state = None
        self.previous_action = None
    
    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts
    
    # Function to set previous state and action values 
    def setPreviousState(self,state,action):
        self.previous_state = state
        self.previous_action = action        

    # Every game, this function is called to reduce the alpha value 
    def DecreaseAlpha(self):
        value = 100.00 / (400.00 + self.episodesSoFar)
        self.setAlpha(value)
        
    def DecreaseEpsilon(self):
        value = 50.00/ (1000.00 + self.episodesSoFar)
        self.setEpsilon(value)


# GAME FUNCTIONS
        
    # Gets the Q_Value for a specific state and action pair
    def getQValue(self,state,action):
        readable_state = self.ParseState(state)
        # Gets Q_Value if the state/action exist in the Q_Table
        if readable_state in self.Q_Table.keys() and action in self.Q_Table[readable_state]:
            Q_Value = self.Q_Table[readable_state][action] 
        else:
        # State is unseen, so initialise the Q_Values as 0 in the Q_Table
            self.Q_Table[readable_state] = {'North':0,'East':0,'South':0,'West':0}
            Q_Value = 0
            
        return Q_Value 
       

    # This function returns the best action and best Q_Value for a state                    
    def getBest(self,state):
        QList = [] # To store a temporary list of QValues
        Action_Dict = {} # To store a temporary list of action:QValues
        
        # Append LEGAL QValues/Actions to the QList & Action_Dict
        legal = state.getLegalPacmanActions()
        for action in ['North','East','South','West']:
            if action in legal:
                Q_Value = self.getQValue(state,action)
                QList.append(Q_Value)
                Action_Dict[action] = Q_Value
        
        # If QList empty, we know it's a terminal state
        # Terminal states do not have actions/Q_Values
        if len(QList) == 0:
            bestQ = 0 
            bestAction = None
        else:
        # Otherwise, get the bestQ and bestAction from QList and Action_Dict
            bestQ = max(QList)
            bestAction = max(Action_Dict, key=lambda key: Action_Dict[key])
        
        # You can select if you want the action or Q_value output
        return {'action':bestAction,'Q_Value': bestQ}

     
    # Function to enable epsilon-greedy exploration
    # Returns True with probability = Epsilon
    def Exploration(self):
        x = random.uniform(0,1)
        if x <= self.epsilon:
            return True
        else:
            return False

    
    # This function returns the difference between current score and previous score
    # As the reward for the previous state
    def ComputeReward(self,previous_state,current_state):
        current_score = current_state.getScore()
        # If this is the first move of the game, just return current score
        if previous_state == None:
            return current_score
        else: # It's not the first move
        # So return difference in current score and previous score
            previous_score = previous_state.getScore()
            return current_score - previous_score
            
        
    # This function acquires the food locations given the state
    def FoodLocations(self,state):
        food = state.getFood()
        # Convert the food state into an indexable array
        food = [i for i in food]
        food = np.array(food)
        # Index the array to extract information about food1, food2
        # Exploiting that we know the food is always in the same locations
        food1 = food[1,1]
        food2 = food[3,3]
        # Return a tuple of True/False indicating if food1, food2 are in the game
        return (food1,food2)
    
    # Convert the state to a readable format, which is used to index the Q_Table
    def ParseState(self,state):
        PacmanLocation = state.getPacmanPosition()
        GhostLocation = state.getGhostPositions()[0]
        FoodLocations = self.FoodLocations(state)
        return (PacmanLocation, GhostLocation,FoodLocations)
    
    # This function updates the Q Value for the (previous state, previous action) pair. 
    def UpdateQValue(self,previous_state,previous_action,current_state):
        # Use ParseState function to convert state to readable format
        readable_state = self.ParseState(previous_state)
        
        previous_reward = self.ComputeReward(previous_state,current_state)

        # Get current Alpha, Gamma
        alpha = self.getAlpha()
        gamma = self.getGamma()
        
        # Get Q Values for previous state, action and the best next state value
        Old_Q = self.getQValue(previous_state,previous_action)
        Next_BestQ = self.getBest(current_state)['Q_Value']
        
        # Compute New Q Value
        TD = Next_BestQ - Old_Q # The Temporal Difference
        Update = alpha * (previous_reward + (gamma * TD))
        New_Q = Old_Q + Update
        
        # Update the Q_Table with the new Q_Value
        self.Q_Table[readable_state][previous_action] = New_Q
        
        
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        
         # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
         # Update the QValue of the previous state
         # If Alpha = 0, training runs are over, so don't update.
        if self.previous_state != None and self.getAlpha() != 0: 
            self.UpdateQValue(self.previous_state,self.previous_action,state)

         # The best action to take in the current state 
        BestAction = self.getBest(state)['action']
        
         # Decide whether to Exploit or Explore
        Exploration = self.Exploration()
        if Exploration == False: #Exploit
            action = BestAction
            
        elif Exploration == True: #Explore 
            action = random.choice(legal) 
            
         # Finally, set this current state and action as the previous state
        self.setPreviousState(state,action)

        return action
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        
        print "A game just ended! Count:", self.episodesSoFar

         # Only update Q Value if alpha is not 0
        if self.alpha != 0:
            self.UpdateQValue(self.previous_state,self.previous_action,state)
               
         # Start of a new game, so reset the 'previous state'
        self.setPreviousState(None,None)

        self.incrementEpisodesSoFar()
        # Decrease alpha and epsilon after every training run
        if self.alpha != 0:
            self.DecreaseAlpha()
            self.DecreaseEpsilon()
            
         # Keep track of the number of games played, and set learning
         # parameters to zero when we are done with the pre-set number
         # of training episodes
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            print self.Q_Table
            self.setAlpha(0)
            self.setEpsilon(0)


