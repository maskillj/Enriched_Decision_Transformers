#from offlinerllib.module.actor import (
    # SquashedDeterministicActor, 
    # SquashedGaussianActor, 
    # CategoricalActor

import torch
import torch.nn as nn
import numpy as np

from common.mlp import MLP, EnsembleMLP
from torch.distributions import Categorical, Normal

from abc import ABC, abstractmethod


class BaseActor(nn.Module):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, obs: torch.Tensor, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, obs: torch.Tensor, *args, **kwargs):
        """Sampling procedure.
        
        Parameters
        ----------
        obs :  The observation, shoule be torch.Tensor.
        """
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, obs, action, *args, **kwargs):
        """Evaluate the log_prob of the action. 
        
        obs :  The observation, shoule be torch.Tensor.
        action :  The action for evaluation, shoule be torch.Tensor with the sample size as `obs`.
        """
        raise NotImplementedError

class DeterministicActor(BaseActor):
    """
    Deterministic Actor, which maps the given obs to a deterministic action. 
    
    Notes
    -----
    All actors creates an extra post-processing module which maps the output of `backend` to
      the real final output. You can pass in any arguments for `MLP` or `EnsembleMLP` to 
      further customize the post-processing module. This is useful when you hope to, for example, 
      create an ensemble-style actor: just designating `ensemble_size`>1 when instantiaing by designating `ensemble_size`.
    
    Parameters
    ----------
    backend :  The preprocessing backend of the actor, which is used to extract vectorized features from the raw input. 
    input_dim :  The dimensions of input (the output of backend module). 
    output_dim :  The dimension of actor's output. 
    device :  The device which the model runs on. Default is cpu. 
    ***(any args of MLP or EnsembleMLP)
    """
    
    def __init__(
        self, 
        backend, 
        input_dim, 
        output_dim, 
        device ="cpu", 
        *, 
        ensemble_size = 1, 
        hidden_dims = [], 
        norm_layer = None, 
        activation = nn.ReLU, 
        dropout = None, 
        share_hidden_layer = False, 
    ) -> None:
        super().__init__()
        self.actor_type = "DeterministicActor"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims.copy()
        self.device = device
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if ensemble_size == 1:
            self.output_layer = MLP(
                input_dim = input_dim, 
                output_dim = output_dim, 
                hidden_dims = hidden_dims, 
                norm_layer = norm_layer, 
                activation = activation, 
                dropout = dropout, 
                device = device
            )
        elif isinstance(ensemble_size, int) and ensemble_size > 1:
            self.output_layer = EnsembleMLP(
                input_dim = input_dim, 
                output_dim = output_dim, 
                hidden_dims = hidden_dims, 
                norm_layer = norm_layer, 
                activation = activation, 
                dropout = dropout, 
                device = device, 
                ensemble_size = ensemble_size, 
                share_hidden_layer = share_hidden_layer
            )
        else:
            raise ValueError(f"ensemble size should be int >= 1.")
    
    def forward(self, input: torch.Tensor):
        return self.output_layer(self.backend(input))
    
    def sample(self, obs: torch.Tensor, *args, **kwargs):
        """Sampling procedure, note that in DeterministicActor we don't do any operation on the sample. 

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.  
        
        Returns
        -------
        (torch.Tensor, torch.Tensor, Dict) :  The sampled action, logprob and info dict. 
        """
        return self(obs), None, {}
    
    def evaluate(self, *args, **kwargs):
        """
        Evaluate the log_prob of the action. Note that this actor does not support evaluation.  
        """
        raise NotImplementedError("Evaluation shouldn't be called for DeterministicActor.")
        
        
class SquashedDeterministicActor(DeterministicActor):
    """
    Squashed Deterministic Actor, which maps the given obs to a deterministic action squashed into [-1, 1] by tanh. 
    
    Notes
    -----
    1. The output of this actor is [-1, 1] by default. 
    2. All actors creates an extra post-processing module which maps the output of `backend` to
        the real final output. You can pass in any arguments for `MLP` or `EnsembleMLP` to 
        further customize the post-processing module. This is useful when you hope to, for example, 
        create an ensemble-style actor: just designating `ensemble_size`>1 when instantiaing by designating `ensemble_size`.
    
    Parameters
    ----------
    backend :  The preprocessing backend of the actor, which is used to extract vectorized features from the raw input. 
    input_dim :  The dimensions of input (the output of backend module). 
    output_dim :  The dimension of actor's output. 
    device :  The device which the model runs on. Default is cpu. 
    ***(any args of MLP or EnsembleMLP)
    """
    def __init__(
        self,
        backend: nn.Module, 
        input_dim, 
        output_dim, 
        device="cpu", 
        *, 
        ensemble_size= 1, 
        hidden_dims = [],
        norm_layer = None, 
        activation = nn.ReLU, 
        dropout = None, 
        share_hidden_layer = False, 
    ) -> None:
        super().__init__(backend, input_dim, output_dim, device, ensemble_size=ensemble_size, hidden_dims=hidden_dims, norm_layer=norm_layer, activation=activation, dropout=dropout, share_hidden_layer=share_hidden_layer)
        self.actor_type = "SqushedDeterministicActor"
        
    def sample(self, obs, *args, **kwargs):
        """Sampling procedure. The action is squashed into [-1, 1] by tanh.

        Parameters
        ----------
        obs :  The observation, should be torch.Tensor.  
        
        Returns
        -------
        (torch.Tensor, torch.Tensor, Dict) :  The sampled action, logprob and info dict. 
        """
        action_prev_tanh = super().forward(obs)
        return torch.tanh(action_prev_tanh), None, {}