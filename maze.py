#!/usr/bin/env python3
import argparse
import math
import numpy as np

class qlearning(object):
	''' given n actions and n_s states, use e-greedy to train Q'''
	def __init__(self,n = 5, n_s = 42, e = 0.1):
		self.n = n # number of actions (<, ^, >, v, F)
		self.e = e # epsilon
		self.n_s = n_s # number of states
		self.Q = np.zeros([n_s,n]) # the expected reward of choose action a at state b. shape states number by action number, initialized with all zeros.

	def eAction(self,s):
		'''given a state s, using epsilon-greedy to give the next action.'''
		if np.random.random() < self.e:
			a = np.random.randint(0,self.n)
		else:
			a = np.argmax(self.Q[s])
		return a

	def update(self,s,a,r,s_new, gamma=.95, alpha=.1):
		'''Estimate the parameters of the agent using Bellman Equation.Q(s,a) = Expectation of ( r + gamma * max(Q(s_new,a')) )Update the Q parameters using gradient descent on the Bellman Equation.'''
		# s is the old state
		# a is the action
		# r is the reward
		# s_new is the new state
		# gamma is the discount factor
		# alpha is learning rate
		self.Q[s,a] += alpha * (r + gamma * max(self.Q[s_new])-self.Q[s,a])

	def play(self, gridworld, n_episodes, gamma=.95, alpha=.1):
		''' given the gridworld by class GridWord, play n_episodes times of the game'''
		total_rewards = []
		# play multiple episodes
		for _ in range(n_episodes):
			reward = 0
			s = gridworld.reset() # initialize the episode
			done = False
			# play the game until the episode is done
			while not done:
				a = self.eAction(s)
				# agent selects an action
				s_new, r, done= gridworld.attempt(a)
				# game return a reward and new state
				self.update(s,a,r,s_new,gamma,alpha)
				# agent update the parameters
				s = s_new
				reward += r # assuming the reward of the step is r
			total_rewards.append(reward)
		return total_rewards

	def render(self, gridworld):
		Qrange = [list(range(self.n)) for i in range(self.n_s)]
		Qscore = gridworld.map.copy()
		Qstep = gridworld.map.copy()
		actioncode = {0:"<",1:'^',2:'>',3:'v',4:'T'}
		for i,r in enumerate(Qrange):
			# first column cannot move left
			if i % gridworld.ncol == 0:
				r.remove(0)
			# first line cannot move up
			if i < gridworld.ncol:
				r.remove(1)
			# the most right column cannot move right
			if i % gridworld.ncol == gridworld.ncol - 1:
				r.remove(2)
			# the bottom line cannot move down
			if i > gridworld.ncol * (gridworld.nrow - 1) -1:
				r.remove(3)
		# decide Qscore and the recommend step
		for i in gridworld.normal:
			Qscore[i] = np.max(self.Q[[i],Qrange[i]])
			step = Qrange[i][np.argmax(self.Q[[i],Qrange[i]])]
			Qstep[i] = actioncode[step]
		# render out
		print("The recommend actions are:")
		for i in range(gridworld.nrow):
			# number of row = 6
			s = ''
			for q in Qstep[gridworld.ncol*i:gridworld.ncol*i+gridworld.ncol]: #number of col = 7
				s += q
				s += '\t'
			print(s)
		print("The future expected reward is:")
		for i in range(gridworld.nrow):
			# number of row = 6
			s = ''
			for q in Qscore[gridworld.ncol*i:gridworld.ncol*i+gridworld.ncol]:
				s += str(q)[:5]
				if len(str(q)) == 1:
					s += '    '
				s += '\t'
			print(s) #number of col = 7

class GridWorld(object):
	''' the goal of the agent is to receive the most reward it can for each trial.
		- The agent can attempt to move up, down, right, or left.  Movement carries a cost (see below).  Movement is not deterministic, and behaves similarly to the example we discussed in class.  If an agent attempts to move in a direction, there is a:
			70% chance the move occurs as expected
			10% chance the agent moves 90 degrees to the right
			10% chance the agent moves 90 degrees to the left
			10% chance the agent moves forward 2 squares.If the agent encounters a goal or a pit on the first move, the trial terminates.If there is a wall after the first move, the agent moves one square forward.
		- The agent can give up on the trial and end the trial immediately.  The agent receives the reward received so far for the trial, plus a penalty for taking this action (see below)
		---------
    	Action code
        	0 : "<",
        	1 : "^",
        	2 : ">",
        	3 : "v",
        	4 : "T" giveup'''
	def __init__(self,r_goal = 5, r_pit = -2, move_cost = -0.1, giveup_cost = -3):
		self.r_goal = r_goal # the reward for reaching the goal state
		self.r_pit = r_pit # the reward for falling into a pit
		self.move_cost = move_cost # each move cost
		self.giveup_cost = giveup_cost # giving up cost
		self.map = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','P','P','-','-','-','-','P','G','-','-','P','-','-','-','P','P','P','-','-','-','-','-','-','-','-','-'] # 7 by 6
		#self.answermap = None
		self.ncol = 7
		self.nrow = 6
		self.normal = []
		self.pits = []
		self.goal = []
		#self.actioncode = {0:"<",1:'^',2:'>',3:'v',4:'T'}
		self.s = None # current state
		self.initiatelist()

	def initiatelist(self):
		for i,c in enumerate(self.map):
			if c == '-':
				self.normal.append(i)
			elif c == 'P':
				self.pits.append(i)
			elif c == 'G':
				self.goal.append(i)

	def render(self):
		for i in range(self.nrow):
			# number of row = 6
			print(self.map[self.ncol*i:self.ncol*i+self.ncol]) #number of col = 7

	def reset(self):
		self.s = np.random.choice(self.normal)
		#self.answermap = self.map.copy()#renew the answermap
		# The start state is selected at random. The agent can start at any state that is neither a goal nor a pit.
		return self.s

	def step(self,a):
		''' folow a direction and go one step, return new state, reward and if the game is terminate'''
		# move left
		if a == 0:
			# if current state is on the first column
			if self.s % self.ncol == 0:
				return self.s, 0, False# do nothing, as the action is move to the edge
			else:
				self.s -= 1 # move to the left
		# move up
		elif a == 1:
			# if current state is on the first row
			if self.s < self.ncol:
				return self.s, 0, False
			else:
				self.s -= self.ncol # move to the up
		# move right
		elif a == 2:
			# if current state is on the most right column
			if self.s % self.ncol == self.ncol - 1:
				return self.s, 0, False
			else:
				self.s += 1 # move to the right
		# move down
		elif a == 3:
			# if current state is on the bottom
			if self.s > self.ncol * (self.nrow - 1) -1:
				return self.s, 0, False
			else:
				self.s += self.ncol # move down
		# check if it move to a pit
		if self.s in self.pits:
			# terminate game
			return self.s, self.r_pit, True
		# check if it move to a goal
		elif self.s in self.goal:
			return self.s, self.r_goal, True
		else:
			return self.s, self.move_cost, False

	def attempt(self,a):
		'''given an expected action, return new state, reward, if the game is terminated'''
		if a < 4: # not terminate
			p = np.random.random()
			if p < 0.7:
				_,reward, done = self.step(a)

			elif p < 0.8:
				a += 1 # move 90 degrees to the right
				a %= 4 # avoid overrange
				_,reward, done = self.step(a)

			elif p < 0.9:
				a -= 1 # move 90 degrees to the left
				a %= 4
				_, reward, done = self.step(a)

			else:
				stp = 2 # move 2 square
				_, reward, done = self.step(a)
				# if first step not terminate the game
				if not done:
					old_s = self.s # update old_s
					_, nreward, done = self.step(a)
					reward += nreward
		else:
			reward = self.giveup_cost
			done = True
			real_action = (a)
		return self.s, reward, done



# Run in command
if  __name__ == '__main__':
	#parser = argparse.ArgumentParser(description = "Final")
	#parser.add_argument('r_goal',metavar='r_goal',nargs=1, type=float, default=5, help='''Reward for reaching the goal. Default is 5.''')
	#parser.add_argument('r_pit',metavar='r_pit',nargs=1, type=float, default=-2, help='''Reward for falling into a pit. Default is -2.''')
	#parser.add_argument('move_cost',metavar='move_cost',nargs=1,type=float,default=-0.1, help = '''Reward for moving. Default is -0.1.''')
	#parser.add_argument('giveup_cost',metavar='giveup_cost',nargs=1,type=float,default=-3, help = '''Reward for giving up. Default is -3.''')
	#parser.add_argument('n_episodes',metavar='n_episodes',nargs=1,type=int,default=10000, help = '''Number of episodes to train agent for. Default is 10000.''')
	#parser.add_argument('epsilon',metavar='epsilon',nargs=1,type=float,default=0.1, help = '''Epsilon, for e-greedy exploration. Default is 0.1.''')
	#args = parser.parse_args()
	#gridworld = GridWorld(args.r_goal[0],args.r_pit[0],args.move_cost[0],args.giveup_cost[0])
	gridworld = GridWorld()
	#ql = qlearning(e = args.epsilon[0])
	ql = qlearning()
	#ql.play(gridworld,args.n_episodes[0])
	ql.play(gridworld,10000)
	ql.render(gridworld)









