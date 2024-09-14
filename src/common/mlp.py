import torch
import torch.nn as nn
import math 



def miniblock(
    input_dim,
    output_dim= 0,
    norm_layer= None,
    activation= None,
    dropout= None, 
    linear_layer= nn.Linear,
    *args, 
    **kwargs):
    """
    Construct a miniblock with given input and output. It is possible to specify norm layer, activation, and dropout for the constructed miniblock.
    
    Parameters
    ----------
    input_dim :  Number of input features..
    output_dim :  Number of output features. Default is 0.
    norm_layer :  Module to use for normalization. When not specified or set to True, nn.LayerNorm is used.
    activation :  Module to use for activation. When not specified or set to True, nn.ReLU is used.
    dropout :  Dropout rate. Default is None.
    linear_layer :  Module to use for linear layer. Default is nn.Linear.
    
    Returns
    -------
    List of modules for miniblock.
    """

    layers= [linear_layer(input_dim, output_dim, *args, **kwargs)]
    if norm_layer is not None:
        if isinstance(norm_layer, nn.Module):
            layers += [norm_layer(output_dim)]
        else:
            layers += [nn.LayerNorm(output_dim)]
    if activation is not None:
        layers += [activation()]
    if dropout is not None and dropout > 0:
        layers += [nn.Dropout(dropout)]
    return layers


class EnsembleLinear(nn.Module):
    """
    An linear module for concurrent forwarding, which can be used for ensemble purpose.

    Parameters
    ----------
    in_features :  Number of input features.
    out_features :  Number of output features.
    ensemble_size :  Ensemble size. Default is 1.
    bias :  Whether to add bias or not. Default is True.
    device :  Device to use for parameters.
    dtype :  Data type to use for parameter.
    """
    def __init__(
        self,
        in_features,
        out_features,
        ensemble_size= 1,
        share_input=True,
        bias= True,
        device = None,
        dtype = None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.share_input = share_input
        self.add_bias = bias
        self.register_parameter("weight", torch.nn.Parameter(torch.empty([in_features, out_features, ensemble_size], **factory_kwargs)))
        if bias:
            self.register_parameter("bias", torch.nn.Parameter(torch.empty([out_features, ensemble_size], **factory_kwargs)))
        else:
            self.register_buffer("bias", torch.zeros([out_features, ensemble_size], **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        # Naively adapting the torch default initialization to EnsembleMLP results in
        # bad performance and strange initial output.
        # So we used the initialization strategy by https://github.com/jhejna/cpl/blob/a644e8bbcc1f32f0d4e1615c5db4f6077d6d2605/research/networks/common.py#L75
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        if self.add_bias:
            nn.init.uniform_(self.bias, -std, std)

    def forward(self, input: torch.Tensor):
        if self.share_input:
            res = torch.einsum('...j,jkb->...kb', input, self.weight) + self.bias
        else:
            res = torch.einsum('b...j,jkb->...kb', input, self.weight) + self.bias
        return torch.einsum('...b->b...', res)

    def __repr__(self):
        return f"EnsembleLinear(in_features={self.in_features}, out_features={self.out_features}, bias={self.add_bias})"


class NoisyLinear(nn.Linear):
    """
    An linear module which supports for noisy parameters.
    
    Parameters
    ----------
    in_features :  Number of input features.
    out_features :  Number of output features.
    std_init :  Standard deviation of the weight and noise initialization.
    bias :  Whether to add bias. 
    device :  Device to use for parameters.
    dtype :  Data type to use for parameter.
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        std_init: float=0.5, 
        bias=True, 
        device=None, 
        dtype=None
    ):
        super().__init__(in_features, out_features, bias, device, dtype) # this will do the initialization

        self.std_init = std_init
        self.register_parameter("weight_std", torch.nn.Parameter(torch.empty(out_features, in_features)))
        self.weight_std.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.register_buffer("weight_noise", torch.empty_like(self.weight))
        if bias:
            self.register_parameter("bias_std", torch.nn.Parameter(torch.empty(out_features)))
            self.bias_std.data.fill_(self.std_init / math.sqrt(self.out_features))
            self.register_buffer("bias_noise", torch.empty_like(self.bias))
        else:
            self.register_parameter("bias_std", None)
            self.register_buffer("bias_noise", None)
        
        self.reset_noise()

    @staticmethod
    def scaled_noise(size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor, reset_noise=False):
        if self.training:
            if reset_noise:
                self.reset_noise()
            if self.bias is not None:
                return torch.nn.functional.linear(
                        x, 
                        self.weight + self.weight_std * self.weight_noise, 
                        self.bias + self.bias_std * self.bias_noise
                    )
            else:
                return torch.nn.functional.linear(
                    x, 
                    self.weight + self.weight_std * self.weight_noise, 
                    None
                )
        else:
            return torch.nn.functional.linear(x, self.weight, self.bias)
        
    def reset_noise(self):
        """
        Reset the noise to the noise matrix .
        """
        device = self.weight.data.device
        epsilon_in = self.scaled_noise(self.in_features)
        epsilon_out = self.scaled_noise(self.out_features)
        self.weight_noise.data = torch.matmul(
            torch.unsqueeze(epsilon_out, -1), other=torch.unsqueeze(epsilon_in, 0)
        ).to(device)
        if self.bias is not None:
            self.bias_noise.data = epsilon_out.to(device)
        


class MLP(nn.Module):
    """
    Creates an MLP module. 
    
    Parameters
    ----------
    input_dim :  The number of input dimensions.
    output_dim :  The number of output dimensions. The value of 0 indicates a cascade model, and the output is \
                activated; while other non-negative values indicate a standalone module, and the output is not activated. Default to 0.
    hidden_dims :  The list of numbers of hidden dimensions. Default is [].
    norm_layer :  List_or_Single[bool, nn.Module(), None]. When `norm_layer` is a List, its length should be equal to the num of layers
                in the created MLP, and each element specifies the operation on each layer. When `norm_layer` is a single element, it will be broadcast 
                to a List as long as the module. When `norm_layer` is `None` or `False`, no normalization will be added; when `True`, we will use `nn.LayerNorm`; 
                otherwise `norm_layer()` will be used. 
    activation :  List_or_Single[bool, nn.Module(), None]. When `activation` is a List, its length should be equal to the num of layers
                in the created MLP, and each element specifies the operation on each layer. When `activation` is a single element, it will be broadcast 
                to a List as long as the module. When `activation is `None` or `False`, no activation will be added; when `True`, we will use `nn.ReLU`; 
                otherwise `norm_layer()` will be used. 
    dropout : List_or_Single[bool, float, None]. When `dropout` is a List, its length should be equal to the num of layers
                in the created MLP, and each element specifies the operation on each layer. When `dropout` is a single element, it will be broadcast 
                to a List as long as the module. When `dropout is `None` or `False` or 0, no dropout will be added; otherwise a layer of `nn.Dropout(dropout)`
                will be added at the end of the layer. 
    device :  The device to run on. Default is " cpu ".
    linear_layer :  The linear layer module. Default to nn.Linear.
    """
    def __init__(
        self, 
        input_dim, 
        output_dim = 0, 
        hidden_dims = [], 
        norm_layer = None, 
        activation = nn.ReLU, 
        dropout = None, 
        device = "cpu", 
        linear_layer: nn.Module=nn.Linear
    ) -> None:
        super().__init__()
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_dims)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_dims))]
        else:
            norm_layer_list = [None]*len(hidden_dims)
        
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_dims)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_dims))]
        else:
            activation_list = [None]*len(hidden_dims)
        
        if dropout:
            if isinstance(dropout, list):
                assert len(dropout) == len(hidden_dims)
                dropout_list = dropout
            else:
                dropout_list = [dropout for _ in range(len(hidden_dims))]
        else:
            dropout_list = [None]*len(hidden_dims)
                        
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim, norm, activ, dropout in zip(
            hidden_dims[:-1], hidden_dims[1:], norm_layer_list, activation_list, dropout_list
        ):
            model += miniblock(in_dim, out_dim, norm, activ, dropout, device=device, linear_layer=linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_dims[-1], output_dim, device=device)]
        self.output_dim = output_dim or hidden_dims[-1]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)        # do we need to flatten x staring at dim=1 ?


class EnsembleMLP(nn.Module):
    """
    Creates MLP module with model ensemble.

    Parameters
    ----------
    input_dim :  The number of input dimensions.
    output_dim :  The number of output dimensions. The value of 0 indicates a cascade model, and the output is \
                activated; while other non-negative values indicate a standalone module, and the output is not activated. Default to 0.
    ensemble_size :  The number of models to ensemble. Default is 1. 
    hidden_dims :  The list of numbers of hidden dimensions. Default is [].
    norm_layer :  List_or_Single[bool, nn.Module(), None]. When `norm_layer` is a List, its length should be equal to the num of layers
                in the created MLP, and each element specifies the operation on each layer. When `norm_layer` is a single element, it will be broadcast 
                to a List as long as the module. When `norm_layer` is `None` or `False`, no normalization will be added; when `True`, we will use `nn.LayerNorm`; 
                otherwise `norm_layer()` will be used. 
    activation :  List_or_Single[bool, nn.Module(), None]. When `activation` is a List, its length should be equal to the num of layers
                in the created MLP, and each element specifies the operation on each layer. When `activation` is a single element, it will be broadcast 
                to a List as long as the module. When `activation is `None` or `False`, no activation will be added; when `True`, we will use `nn.ReLU`; 
                otherwise `norm_layer()` will be used. 
    dropout : List_or_Single[bool, float, None]. When `dropout` is a List, its length should be equal to the num of layers
                in the created MLP, and each element specifies the operation on each layer. When `dropout` is a single element, it will be broadcast 
                to a List as long as the module. When `dropout is `None` or `False` or 0, no dropout will be added; otherwise a layer of `nn.Dropout(dropout)`
                will be added at the end of the layer. 
    share_hidden_layer :  List_of_Single[bool]. The list of indicators of whether each layer should be shared or not. Single values will be broadcast to as long as the lengths of the layers. 
    device :  The device to run on. Default is " cpu ".
    """
    def __init__(
        self, 
        input_dim, 
        output_dim= 0, 
        ensemble_size= 1, 
        hidden_dims= [], 
        norm_layer = None, 
        activation = nn.ReLU, 
        dropout= None, 
        share_hidden_layer= False, 
        device= "cpu", 
    ) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_dims)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_dims))]
        else:
            norm_layer_list = [None]*len(hidden_dims)
        
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_dims)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_dims))]
        else:
            activation_list = [None]*len(hidden_dims)
            
        if dropout:
            if isinstance(dropout, list):
                assert len(dropout) == len(hidden_dims)
                dropout_list = dropout
            else:
                dropout_list = [dropout for _ in range(len(hidden_dims))]
        else:
            dropout_list = [None]*len(hidden_dims)
            
        if share_hidden_layer:
            if isinstance(share_hidden_layer, list):
                assert len(share_hidden_layer) == len(hidden_dims)
                share_hidden_layer_list = share_hidden_layer
            else:
                share_hidden_layer_list = [share_hidden_layer for _ in len(hidden_dims)]
        else:
            share_hidden_layer_list = [False]*len(hidden_dims)
                
        
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        share_input = True
        for in_dim, out_dim, norm, activ, dropout, share_layer in zip(
            hidden_dims[:-1], hidden_dims[1:], norm_layer_list, activation_list,dropout_list, share_hidden_layer_list
        ):
            if share_layer:      
                model += miniblock(in_dim, out_dim, norm, activ, dropout, linear_layer=nn.Linear, device=device)
            else:
                model += miniblock(in_dim, out_dim, norm, activ, dropout, linear_layer=EnsembleLinear, ensemble_size=ensemble_size, device=device, share_input=share_input)
                share_input = False  # The first EnsembleLinear shares the input and produce branched outputs, while the subsequent EnsembleLinear do not. 
        if output_dim > 0:
            model += [EnsembleLinear(hidden_dims[-1], output_dim, ensemble_size, device=device, share_input=share_input)]
        self.output_dim = output_dim or hidden_dims[-1]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input: torch.Tensor):
        return self.model(input)