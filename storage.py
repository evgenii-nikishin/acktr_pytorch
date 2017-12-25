import numpy as np


class ACStorage(object):
    """
        Class to effeciently store and operate with actor-critic model's data.
    """

    def __init__(self, n_steps, states_shape):
        """
        Class constructor

        Keyword arguments:
        n_steps      -- number of agent steps in one batch
        states_shape -- shape of variable describing state (observation)
        """

        self.n_steps = n_steps
        
        self.states  = np.zeros((self.n_steps, ) + states_shape)
        self.actions = np.zeros(self.n_steps)
        self.rewards = np.zeros(self.n_steps)
        
        self.last_step = -1
        
    def insert(self, state, action, reward):
        """
        Inserts new triple <state,action,reward> into storage

        Keyword arguments:
        state  -- new state
        action -- new action
        reward -- new reward
        """

        self.last_step += 1
        assert self.last_step < self.n_steps, 'storage capacity exceeded'
        
        self.states [self.last_step] = state
        self.actions[self.last_step] = action
        self.rewards[self.last_step] = reward
        
    def get_states(self):
        """
        Get all states from storage
        """

        return self.states[:self.last_step+1, ...]
    
    def get_actions(self):
        """
        Get all actions from storage
        """

        return self.actions[:self.last_step+1]
    
    def get_rewards(self):
        """
        Get all rewards from storage
        """

        return self.rewards[:self.last_step+1]
    
    def clear(self):
        """
        Clear storage
        """

        self.last_step = -1
    
    def calc_G_0(self, gamma):
        """
        Calculates G_0 = \sum_{r=0}^{storage_size} [gamma^r * reward_r]

        Keyword arguments:
        gamma -- discount factor, float
        """

        n = self.last_step+1
        g_0 = self.rewards[n-1] 
        for i in range(n-2, -1, -1):
            g_0 = self.rewards[i] + g_0 * gamma
        return g_0
    
    def calc_returns(self, gamma):
        """
        Calculates G_t for every time stamp

        Keyword arguments:
        gamma -- discount factor, float
        """

        n = self.last_step+1
        returns = np.zeros((n))
        
        returns[-1] = self.rewards[n-1]
        for i in range(n-2, -1, -1):
            returns[i] = self.rewards[i] + returns[i+1] * gamma
        return returns
    
    def calc_gae(self, values, next_value, gamma, k=5.0):
        """
        Calculates GAE (https://arxiv.org/pdf/1602.01783.pdf) for every time stamp

        Keyword arguments:
        values     -- values of critic on every time stamp
        next_value -- value of critic for next state after current batch
        gamma      -- discount factor, float
        k          -- bootstrap size, float (default 5.0)
        """

        n = self.last_step+1
        assert n == values.data.shape[0], 'shapes mismatch'
        
        k = min(n, k)
        gae = np.zeros((n))  

        gae[n-1] = self.rewards[n-1]
        for i in range(n-2, -1, -1): 
            gae[i] = self.rewards[i] + gamma * gae[i+1]
        
        if k < n:
            gae[:-k] += gamma**k * (values[k:].cpu().data.numpy().ravel() - gae[k:])

        gamma_buf = 1.0
        for i in range(1, k+1):
            gamma_buf *= gamma
            gae[-i] += gamma_buf * next_value

        return gae
    