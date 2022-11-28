import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import List 
import numpy as np 

def get_conv_output_size(initial_input_size, number_blocks, max_pool=True):
    """ The conv blocks we use keep the same size but use max pooling, so the output of all convolution blocks will be of length input_size / 2"""
    if max_pool==False:
        return initial_input_size
    out_size = initial_input_size
    for i in range(number_blocks):
        out_size = int(out_size / 2)
    return out_size

def get_trans_output_size(input_size, stride, padding, kernel_size):
    """ A function to get the output length of a vector of length input_size after a tranposed convolution layer"""
    return (input_size -1)*stride - 2*padding + (kernel_size - 1) +1

def get_final_output(initial_input_size, number_blocks, number_trans_per_block, stride, padding, kernel_size):
    """A function to get the final output size after tranposed convolution blocks"""
    out_size = initial_input_size
    for i in range(number_blocks):
        for k in range(number_trans_per_block):
            out_size = get_trans_output_size(out_size, stride, padding, kernel_size)
    return out_size

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class UnFlatten(nn.Module):
    def __init__(self, size, length):
        super(UnFlatten, self).__init__()
        self.size = size
        self.length = length

    def forward(self, input):
        out = input.view(input.size(0), self.size, self.length)
        return out


class AUXREG(nn.Module):
    """Module block for predicing mps from the latent dimensions"""
    def __init__(self, in_dim: int,  out_dim: int, hidden_dims: List[int], ): 
        super().__init__()
        self.in_dim = in_dim # Latent space size
        self.out_dim = out_dim # Machine parameter size
        self.hidden_dims = hidden_dims

        self.block = nn.ModuleList()

        last_dim = in_dim 
        for dim in self.hidden_dims: 
            self.block.append(nn.Linear(last_dim, dim))
            self.block.append(nn.ReLU())
            last_dim = dim 
        self.block.append(nn.Linear(last_dim, out_dim))
    
    def forward(self, z): 
        for lay in self.block: 
            z = lay(z)
        return z 

class PRIORREG(nn.Module):
    """Module block for conditional prior via machine paramteers"""
    def __init__(self, in_dim: int,  hidden_dims: List[int], out_dim: int = None, make_prior=False): 
        super().__init__()
        self.in_dim = in_dim # Machine parameter size 
        self.out_dim = out_dim # Latent dimension size 
        self.hidden_dims = hidden_dims
        self.make_prior = make_prior
        
        self.block = nn.ModuleList()

        last_dim = in_dim 
        for dim in self.hidden_dims: 
            self.block.append(nn.Linear(last_dim, dim))
            self.block.append(nn.ReLU())
            last_dim = dim 
        # self.block.append(nn.Linear(last_dim, out_dim))
        if self.make_prior: 
            self.fc_mu = nn.Linear(last_dim, self.out_dim)
            self.fc_var = nn.Linear(last_dim, self.out_dim)
        else: 
            self.block.append(nn.Linear(last_dim, self.out_dim))

    def forward(self, z): 
        for lay in self.block: 
            z = lay(z)

        if self.make_prior: 
            mu, var = self.fc_mu(z), self.fc_var(z)
            return mu, var 
        else: 
            return z


class ENCODER(nn.Module): 
    def __init__(self, filter_sizes: List[int], in_length: int = 75, in_ch: int = 2): 
        super().__init__()
        hidden_channels = filter_sizes
        self.kernel_size = 5
        self.padding = 0
        self.stride = 1
        self.pool_padding = 0
        self.pool_dilation = 1
        self.pool_kernel_size = 2
        self.pool_stride = self.pool_kernel_size
        
        self.block = nn.ModuleList()
        self.end_conv_size = in_length # [(W - K + 2P) / S] + 1
        for dim in hidden_channels: 
            self.block.append(nn.Conv1d(in_ch, dim, kernel_size=self.kernel_size, padding=self.padding))
            self.end_conv_size = ((self.end_conv_size - self.kernel_size + 2*self.padding) / self.stride) + 1
            self.block.append(nn.ReLU())
            self.block.append(nn.MaxPool1d(self.pool_kernel_size, padding=self.pool_padding, dilation=self.pool_dilation, ))
            self.end_conv_size = ((self.end_conv_size + 2*self.pool_padding - self.pool_dilation*(self.pool_kernel_size -1)-1) / self.pool_stride) + 1
            
            in_ch = dim
        self.block.append(Flatten())

        self.end_conv_size = int(self.end_conv_size)
    def forward(self, x): 
        # print('Encoder in shape', x.shape)
        for lay in self.block: 
            x = lay(x)
            # print(x.shape)
        # print('encoder out shape', x.shape)
        return x

class DECODER(nn.Module): 
    def __init__(self, filter_sizes: List[int], end_conv_size: int, clamping_zero_tensor: torch.Tensor): 
        super().__init__()
        in_ch = 2
        self.hidden_channels = filter_sizes
        self.end_conv_size = end_conv_size
        self.num_trans_conv_blocks = 1
        self.trans_stride = 1
        self.trans_padding = 0
        self.trans_kernel_size = 2

        self.block = nn.ModuleList()
        if self.hidden_channels[-1] != in_ch: 
            self.hidden_channels.append(in_ch)
        self.block.append(UnFlatten(size=self.hidden_channels[0], length=self.end_conv_size))
        # Needs trasnpose kernel instead 
        for i in range(len(self.hidden_channels) - 1):
            self.block.append(nn.ConvTranspose1d(self.hidden_channels[i], self.hidden_channels[i+1], kernel_size=self.trans_kernel_size))
            self.block.append(nn.ReLU())
        self.final_size = get_final_output(end_conv_size, len(self.hidden_channels) -1, self.num_trans_conv_blocks, self.trans_stride, self.trans_padding, self.trans_kernel_size)

        self.clamping_zero_tensor = clamping_zero_tensor
        if self.clamping_zero_tensor is not None: 
            print('Applying a clamping tensor to reconstruct only non-negative profiles!')
    def forward(self, x): 
        # print('Decoder in shape', x.shape)
        for lay in self.block: 
            x = lay(x)

        if self.clamping_zero_tensor is not None: 
            torch.clamp(x, self.clamping_zero_tensor)
        # print('Decoder out shape', x.shape)
        return x
