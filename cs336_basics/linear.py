# define class / functions to support deep learning training
from torch.nn import Module
from torch import nn
import torch 

class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # define init
        super().__init__()
        # self.in_features = in_features
        # self.out_features = out_features 
        # self.device = device 
        # self.dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device = device, dtype = dtype)
            )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t())

# layer = Linear(10, 5)
# input_tensor = torch.rand(3,10)
# output = layer(input_tensor)
# print(input_tensor, layer, output)

# from torchviz import make_dot
# output = layer(input_tensor)
# make_dot(output)