import numpy as np

import torch
from torch import nn

"""
    Utilities for RF models implementation
"""

class ReshapingLayer(nn.Module):
    """
        Wrapper for 'reshape' 
         to embed this operation in nn.Sequential(...)
    """

    def __init__(self, *args):
        """
        Constructor

        Arguments:
            *args  -- new shape dimensions

        Example:
            ReshapingLayer(10, 20, -1)
        """

        super(ReshapingLayer, self).__init__()
        self.shape = args

    def forward(self, x):
        """
        Reshape input w.r.t. class parameters

        Arguments:
            x   --  input data, tensor
        """

        return x.view(self.shape)