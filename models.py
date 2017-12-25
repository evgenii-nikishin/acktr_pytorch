import numpy as np
import copy
import gym
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch import optim

from storage import ACStorage
from ac_nets import OneDimensionalACNets, KDimensionalACNets

"""
    Implementations of algorithmn for RL
"""


class AgentA3C(nn.Module):
    """
        Implementation of A3C algorithm
        (https://arxiv.org/abs/1602.01783)
    """


    def __init__(self, envs, gamma=0.99, optimizer=None, lr=1e-4, save_returns=True):
        """
        Constructor

        Arguments:
            envs         --  list of game environments
            gamma        --  discount factor, float
            optimizer    --  optimizer for learning proc., torch.Optimizer (default Adam)
            lr           --  learning rate, float (default 1e-4)
            save_returns --  whether save returns on each steps, bool
        """

        super(AgentA3C, self).__init__()
        self.envs = envs
        self.gamma = gamma
        self.save_returns = save_returns

        self.n_envs = len(envs)
        self.n_actions = int(envs[0].action_space.n)
        self.is_cuda = False
        

        if (type(envs[0].observation_space) == gym.spaces.Discrete) or (
            type(envs[0].observation_space) == gym.spaces.Box and 
             len(envs[0].observation_space.shape) == 1):

            if type(envs[0].observation_space) == gym.spaces.Discrete:
                self.states_shape = (envs[0].observation_space.n, )
                self.nets = OneDimensionalACNets(
                    self.states_shape[0], self.n_actions, need_encode=True
                )
            else:
                self.states_shape = (envs[0].observation_space.shape[0], )
                self.nets = OneDimensionalACNets(
                    self.states_shape[0], self.n_actions, need_encode=False
                )
                     
        elif (type(envs[0].observation_space) == gym.spaces.Box and 
             len(envs[0].observation_space.shape) == 3):
            self.states_shape = envs[0].observation_space.shape
            self.nets = KDimensionalACNets(self.states_shape, self.n_actions) 


        """
        # define type of environment and init AC NNs       
        if (type(envs[0].observation_space) == gym.spaces.Discrete) or (
            type(envs[0].observation_space) == gym.spaces.Box and 
                len(envs[0].observation_space.shape) == 1):

            if type(envs[0].observation_space) == gym.spaces.Discrete:
                self.states_shape = (envs[0].observation_space.n, )
                self.nets = OneDimensionalACNets(
                    self.states_shape[0], self.n_actions, need_encode=True
                )
            else:
                self.states_shape = (envs[0].observation_space.shape[0], )
                self.nets = OneDimensionalACNets(
                    self.states_shape[0], self.n_actions, need_encode=False
                )
            
        elif (type(envs[0].observation_space) == gym.spaces.Box 
            and len(envs[0].observation_space.shape) == 3): 
            self.states_shape = envs[0].observation_space.shape
            self.nets = KDimensionalACNets(self.states_shape, self.n_actions) 
        """

        # set optimizer of create default
        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = copy.deepcopy(optimizer)

    def cuda(self):
        """
        Move model to GPU
        """

        self.is_cuda = True
        self.nets.cuda()
        return super(AgentA3C, self).cuda()
    
    def cpu(self):
        """
        Move nets to CPU
        """

        self.is_cuda = False
        self.nets.cpu()
        return super(AgentA3C, self).cpu()
        
    def get_distr(self, state):
        """
        Logits as Variable -> Distribution

        Arguments:
            state   -- state for which actions distr. need to be computed
        """

        state_enc = Variable(self.nets.encode_state(state), volatile=True)
        return F.softmax(self.nets.actor_nn(state_enc), dim=-1)
    
    def act(self, state):
        """
        Choose action according to probabilities

        Arguments:
            state   -- state from which agent acts
        """

        return torch.multinomial(self.get_distr(state), 1).data[0]
    
    def get_policy(self):
        """
        Returns distributions for all possible states
        """

        if type(self.envs[0].observation_space) != gym.spaces.Discrete:
            raise ValueError('Avaliable only for discrete state spaces')

        all_states = np.arange(self.states_shape[0])
        states = Variable(self.nets.encode_states(all_states), volatile=True)
        return F.softmax(self.nets.actor_nn(states), dim=-1).cpu().data.numpy()
    
    def learn(self, n_agents, n_episodes, n_steps=10, k_bootstrap=5, entr_C=1e-3, verbosity=0):
        """
        Learn model via Actor-Critic interaction

        Arguments:
            n_agents        --  number of async. agents, int
            n_episodes      --  number of full interaction emulations, int
            n_steps         --  number of steps for each parameters update, int
            k_bootstrap     --  number of rewards used to compute GAE, int
            entr_C          --  entropy loss multiplier, float
            verbosity       --  numbeg of episodes to print debug inf, int (default 0, don't print)
        """

        self.train()
        self.nets.train()
        
        storages = np.array([
            ACStorage(n_steps, self.states_shape) for _ in range(n_agents)
        ])    
        states = np.array([env.reset() for env in self.envs])
        dones   = np.zeros((n_agents), dtype=bool)
        batch_n = np.zeros((n_agents))
        episode_rewards = np.zeros((n_agents))

        if self.save_returns:
            self.returns_over_episodes = [] 

        if self.save_returns or verbosity > 0:
            sum_rewards_e = 0.0
            if verbosity > 0: 
                sum_loss_actor = 0.0
                sum_loss_critic = 0.0
                sum_entropy = 0.0
                batches = 0
                timer_st = time.time()

        for i_episode in range(n_episodes):
            for i_agent in np.where(dones == True)[0]:
                batch_n[i_agent] = 0
                episode_rewards[i_agent] = 0
                states[i_agent] = self.envs[i_agent].reset()
            for i_agent in range(n_agents):
                storages[i_agent].clear()
                
            for i_agent in range(n_agents):
                for step in range(n_steps):
                    act = self.act(states[i_agent])
                    s_new, rew, dones[i_agent], _ = self.envs[i_agent].step(act)
                    storages[i_agent].insert(states[i_agent], act, rew)

                    states[i_agent] = s_new
                    if dones[i_agent]:
                        break
            dones_agents  = np.where(dones == True)[0] 
            active_agents = np.where(dones == False)[0] 
            
            #DEBUG
            if self.save_returns or verbosity > 0:
                episode_rewards += self.gamma**(batch_n*n_steps) * \
                    np.array([s.calc_G_0(self.gamma) for s in storages])
                
            batch_n += 1
                
            next_values = np.zeros(n_agents)
            if active_agents.shape[0] > 0:
                s_enc = self.nets.encode_states(np.array([s for s in states[active_agents]]))
                next_values[active_agents] = self.nets.critic_nn(
                    Variable(s_enc, volatile=True)
                ).data[0]
                
            states_all = [s.get_states() for s in storages]
            indices = np.cumsum([0] + [s.shape[0] for s in states_all])
                
            states_var = Variable(self.nets.encode_states(
                np.concatenate(states_all)), requires_grad=False
            )
                
            values_var = self.nets.critic_nn(states_var) 
                
            actions_all = [s.get_actions() for s in storages]
            aprobs_var = F.log_softmax(self.nets.actor_nn(states_var), dim=1)
            aprobs_var = aprobs_var[np.arange(aprobs_var.size(0)), np.concatenate(actions_all)]

            gae = np.concatenate(
                [storages[i].calc_gae(
                    values_var[indices[i]:indices[i+1]], 
                    next_values[i], self.gamma, k=k_bootstrap)
                for i in range(n_agents)]
            )
            gae_var = Variable(torch.FloatTensor(gae), requires_grad=False)
            if self.is_cuda:
                gae_var = gae_var.cuda()
                
            advantages = gae_var - values_var

            entropy      = -(aprobs_var * torch.exp(aprobs_var)).sum() / n_agents
            actor_gain   = (advantages * aprobs_var).sum() / n_agents
            critic_loss  = 0.5*advantages.pow(2).sum() / n_agents

            self.optimizer.zero_grad()
            (-actor_gain + critic_loss - entr_C*entropy).backward() 
            self.optimizer.step()
                   
            if verbosity > 0:
                sum_entropy     += entropy.data[0] / n_agents
                sum_loss_actor  += actor_gain.data[0] / n_agents
                sum_loss_critic += critic_loss.data[0] / n_agents
             
            if self.save_returns or verbosity > 0:
                episode_reward = 0.0
                if dones_agents.shape[0] > 0:
                    episode_reward = episode_rewards[dones_agents].sum() / dones_agents.shape[0]
                elif len(self.returns_over_episodes) > 0:
                    episode_reward = self.returns_over_episodes[-1]

                if self.save_returns:
                    self.returns_over_episodes.append(episode_reward)

                sum_rewards_e += episode_reward

            if verbosity > 0 and (i_episode+1) % verbosity == 0:
                print(('episode {:6} [act: {:.4f}, crt: {:.4f}, ent: {:.4f} ' + 
                    'rew_e: {:.6f}], {:.1f} ms/ep').format(
                    i_episode+1, sum_loss_actor/verbosity, sum_loss_critic/verbosity, 
                    sum_entropy/verbosity, sum_rewards_e/verbosity, 
                    (time.time()-timer_st)*1000/verbosity )
                )
                batches = sum_rewards_e = sum_loss_actor = sum_loss_critic = sum_entropy = 0.0
                timer_st = time.time()
        
        self.nets.eval()
        self.eval()


class AgentA2C(nn.Module):
    """
        Implementation of A2C algorithm
        (https://arxiv.org/abs/1602.01783)
    """


    def __init__(self, env, gamma=0.99, optimizer=None, lr=1e-4, save_returns=True):
        """
        Constructor

        Arguments:
            env          --  game environments
            gamma        --  discount factor, float
            optimizer    --  optimizer for learning proc., torch.Optimizer (default Adam)
            lr           --  learning rate, float (default 1e-4)
            save_returns --  whether save returns on each steps, bool
        """

        super(AgentA2C, self).__init__()
        self.env = env
        self.gamma = gamma
        self.save_returns = save_returns
        
        self.is_cuda = False
        self.n_actions = int(env.action_space.n)

        # define type of environment and init AC NNs  
        if (type(env.observation_space) == gym.spaces.Discrete) or (
            type(env.observation_space) == gym.spaces.Box and 
             len(env.observation_space.shape) == 1):

            if type(env.observation_space) == gym.spaces.Discrete:
                self.states_shape = (env.observation_space.n, )
                self.nets = OneDimensionalACNets(
                    self.states_shape[0], self.n_actions, need_encode=True
                )
            else:
                self.states_shape = (env.observation_space.shape[0], )
                self.nets = OneDimensionalACNets(
                    self.states_shape[0], self.n_actions, need_encode=False
                )
                     
        elif (type(env.observation_space) == gym.spaces.Box and 
             len(env.observation_space.shape) == 3):
            self.states_shape = env.observation_space.shape
            self.nets = KDimensionalACNets(self.states_shape, self.n_actions) 
        
        if optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = copy.deepcopy(optimizer)
    
    def cuda(self):
        """
        Move model to GPU
        """
        self.is_cuda = True
        self.nets.cuda()
        return super(AgentA2C, self).cuda()
    
    def cpu(self):
        """
        Move model to CPU
        """
        self.is_cuda = False
        self.nets.cpu()
        return super(AgentA2C, self).cpu()
        
    def get_distr(self, state):
        """
        Logits as Variable -> Distribution

        Arguments:
            state   -- state for which actions distr. need to be computed
        """

        state_enc = Variable(self.nets.encode_state(state), volatile=True)
        return F.softmax(self.nets.actor_nn(state_enc), dim=-1)
    
    def act(self, state):
        """
        Choose action according to probabilities

        Arguments:
            state   -- state from which agent acts
        """

        return torch.multinomial(self.get_distr(state), 1).data[0]
    
    
    def get_policy(self):
        """
        Returns distributions for all possible states
        """

        if type(self.env.observation_space) != gym.spaces.Discrete:
            raise ValueError('Avaliable only for discrete state spaces')

        all_states = np.arange(self.states_shape[0])
        states = Variable(self.nets.encode_states(all_states), volatile=True)
        return F.softmax(self.nets.actor_nn(states), dim=-1).cpu().data.numpy()
    
    def learn(self, n_episodes, n_steps=10, k_bootstrap=5, entr_C=1e-3, verbosity=0):
        """
        Learn model via Actor-Critic interaction

        Arguments:
            n_episodes      --  number of full interaction emulations, int
            n_steps         --  number of steps for each parameters update, int
            k_bootstrap     --  number of rewards used to compute GAE, int
            entr_C          --  entropy loss multiplier, float
            verbosity       --  numbeg of episodes to print debug inf, int (default 0, don't print)
        """

        self.train()
        self.nets.train()

        if self.save_returns:
            self.returns_over_episodes = [] 

        if self.save_returns or verbosity > 0:
            sum_rewards_e = 0.0
            if verbosity > 0: 
                sum_loss_actor = 0.0
                sum_loss_critic = 0.0
                sum_entropy = 0.0
                timer_st = time.time()
        
        batches = 0
        storage = ACStorage(n_steps, self.states_shape)
        for i_episode in range(n_episodes):
            state = self.env.reset()
            done = False
            
            batch_n = 0
            episode_reward = 0.0
            while not done:
                storage.clear()
                for i_step in range(n_steps):
                    action = self.act(state)
                    state_new, reward, done, _ = self.env.step(action)
                    storage.insert(state, action, reward)
                    
                    state = state_new
                    if done:
                        break
                
                if self.save_returns or verbosity > 0:
                    episode_reward += self.gamma**(batch_n*n_steps) * \
                                        storage.calc_G_0(self.gamma)
                
                batch_n += 1
                
                state_enc = self.nets.encode_state(state)
                next_value = 0.0 
                if not done:
                    next_value = self.nets.critic_nn(Variable(state_enc, volatile=True)).data[0]
                
                states = self.nets.encode_states(storage.get_states())
                states_var = Variable(states, requires_grad=False)
                
                values_var = self.nets.critic_nn(states_var) 
                
                aprobs_var = F.log_softmax(self.nets.actor_nn(states_var), dim=1)
                aprobs_var = aprobs_var[np.arange(aprobs_var.size(0)), storage.get_actions()]
                
                gae = storage.calc_gae(values_var, next_value, self.gamma, k=k_bootstrap)
                gae_var = Variable(torch.FloatTensor(gae), requires_grad=False)
                if self.is_cuda:
                    gae_var = gae_var.cuda()
                
                advantages = gae_var - values_var

                entropy      = -(aprobs_var * torch.exp(aprobs_var)).sum()
                actor_gain   = (gae_var * aprobs_var).sum()
                critic_loss  = 0.5*advantages.pow(2).sum()

                self.optimizer.zero_grad()
                (-actor_gain + critic_loss - entr_C*entropy).backward() 
                self.optimizer.step()
                
                if verbosity > 0:
                    sum_entropy     += entropy.data[0]
                    sum_loss_actor  += actor_gain.data[0]
                    sum_loss_critic += critic_loss.data[0]
             
            if self.save_returns or verbosity > 0:
                if self.save_returns:
                    self.returns_over_episodes.append(episode_reward)
                sum_rewards_e += episode_reward

            batches += batch_n+1 

            if verbosity > 0 and (i_episode+1) % verbosity == 0:
                print(('episode {:6} [act: {:.4f}, crt: {:.4f}, ent: {:.4f} ' + 
                    'rew_e: {:.6f}], {:.1f} ms/ep').format(
                    i_episode+1, sum_loss_actor/verbosity, sum_loss_critic/verbosity, 
                    sum_entropy/verbosity, sum_rewards_e/verbosity, 
                    (time.time()-timer_st)*1000/verbosity )
                )
                batches = sum_rewards_e = sum_loss_actor = sum_loss_critic = sum_entropy = 0.0
                timer_st = time.time()
        
        self.nets.eval()
        self.eval()