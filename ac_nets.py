import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from skimage.transform import resize
from utils import ReshapingLayer

"""
    Wrappers for Actor and Critic NNs
"""


class OneDimensionalACNets(nn.Module):
    """
        Wrapper for Actor and Critic neural networks.
        Use this in games with one-dimensional environment.
    """


    def __init__(self, n_states, n_actions, need_encode=True):
        """
        Constructor

        Arguments:
            n_states    - number of possible states, int 
            n_actions   - number of possible actions, int 
            need_encode - if we need to one-hot encode state, bool 
        """

        super(OneDimensionalACNets, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.need_encode = need_encode
        
        self.is_cuda = False

        # nets initialization
        self._init_actor()
        self._init_critic()
        
    def _init_actor(self):
        """
        Initialize actor neural network
        """

        self.actor_nn  = nn.Sequential(
            nn.Linear(self.n_states, 64),
            nn.Tanh(),
            nn.Linear(64, self.n_actions),
        )
        # random weights and biases
        self.actor_nn[0].weight.data[0].normal_()
        self.actor_nn[0].bias.data[0] = 0
        self.actor_nn[2].weight.data[0].normal_()
        self.actor_nn[2].bias.data[0] = 0

    def _init_critic(self):
        """
        Initialize critic neural network
        """

        self.critic_nn = nn.Sequential(
            nn.Linear(self.n_states, 64),
            nn.Tanh(),

            nn.Linear(64, 1),
        )
        #random weights and biasees
        self.critic_nn[0].weight.data[0].normal_()
        self.critic_nn[0].bias.data[0] = 0
        self.critic_nn[2].weight.data[0].normal_()
        self.critic_nn[2].bias.data[0] = 0
    
    def cuda(self):
        """
        Move nets to GPU
        """

        self.is_cuda = True
        return super(OneDimensionalACNets, self).cuda()
    
    def cpu(self):
        """
        Move nets to CPU
        """

        self.is_cuda = False
        return super(OneDimensionalACNets, self).cpu()
    
    def encode_state(self, state):
        """
        One-hot encode state

        Arguments:
            state   -- state to encode
        """

        if self.need_encode:
            s_onehot = torch.zeros(self.n_states)
            s_onehot[state] = 1.0
        else:
            s_onehot = torch.FloatTensor(state)
        
        if self.is_cuda:
            s_onehot = s_onehot.cuda()
        return s_onehot
    
    def encode_states(self, states):
        """
        One-hot encode group of states

        Arguments:
            states  -- group of states to encode
        """

        if self.need_encode:
            n = len(states)
            encoded = torch.zeros(n, self.n_states)
            indices = np.array(
                states[:,0] if len(states.shape) > 1 else states
            )
            encoded[np.arange(n), indices] = 1.0
        else:
            encoded = torch.FloatTensor(states)
            
        if self.is_cuda:
            encoded = encoded.cuda()
        return encoded


class KDimensionalACNets(nn.Module):
    """
        Wrapper for Actor and Critic neural networks.
        Use this in games with K-dimensional environment, K > 1
    """


    def __init__(self, n_states, n_actions, inner_state_shape=(84,84)):
        """
        Constructor

        Arguments:
            n_states            -- number of possible states, int 
            n_actions           -- number of possible actions, int 
            inner_state_shape   -- shape of state represesentation in NNs, tuple
        """

        super(KDimensionalACNets, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.inner_state_shape = inner_state_shape

        self.is_cuda = False
        # manual parameter, depends on architecture
        self.hidden_dim = 32 * 7 * 7

        self._init_actor()
        self._init_critic()    
        
    def _init_actor(self):
        """
        Initialize actor neural network
        """

        self.actor_nn = nn.Sequential(
            nn.Conv2d(self.n_states[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            ReshapingLayer(-1, self.hidden_dim),
            
            nn.Linear(self.hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, self.n_actions)
        )
        self.actor_nn[7].weight.data[0].normal_()
        self.actor_nn[7].bias.data[0] = 0
        self.actor_nn[9].weight.data[0].normal_()
        self.actor_nn[9].bias.data[0] = 0

    def _init_critic(self):
        """
        Initialize critic neural network
        """

        self.critic_nn = nn.Sequential(
            nn.Conv2d(self.n_states[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            ReshapingLayer(-1, self.hidden_dim),
            
            nn.Linear(self.hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.critic_nn[7].weight.data[0].normal_()
        self.critic_nn[7].bias.data[0] = 0
        self.critic_nn[9].weight.data[0].normal_()
        self.critic_nn[9].bias.data[0] = 0
    
    def cuda(self):
        """
        Move nets to GPU
        """

        self.is_cuda = True
        return super(KDimensionalACNets, self).cuda()
    
    def cpu(self):
        """
        Move nets to CPU
        """

        self.is_cuda = False
        return super(KDimensionalACNets, self).cpu()
    
    def preprocess_img(self, imgs):
        """
        Resize image for universal representation of env. state

        Arguments:
            imgs    -- array of images to resize 
        """

        return resize(imgs, self.inner_state_shape, mode='reflect')
    
    def encode_state(self, s):
        """
        One-hot encode state

        Arguments:
            state   -- state to encode
        """

        enc = torch.FloatTensor(self.preprocess_img(s)).view(
            (-1, self.n_states[2]) + self.inner_state_shape
        )
        if self.is_cuda:
            enc = enc.cuda()
        return enc
    
    def encode_states(self, states):
        """
        One-hot encode group of states

        Arguments:
            states  -- group of states to encode
        """
        # NOTE: can be optimized

        return torch.cat([self.encode_state(s) for s in states])
    
