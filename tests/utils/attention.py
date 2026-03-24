import torch
from torch import nn 
from torch.nn import Module 
import math

from jaxtyping import Bool, Float, Int
from torch import Tensor


class Attention(Module):
    ''' Deliverable: fImplement the SwiGLU feed-forward network, 
    composed of a SiLU activation unction and a GLU.
    
    Note: in this particular case, you should feel free to use 
    torch.sigmoid in your implementation for numerical stability.
    
    You should set d_ff to approximately 8/3 × dmodel in your implementation,
    while ensuring that the dimensionality of the inner feed-forward layer is
    a multiple of 64 to make good use of your hardware. To test 
    your implementation against our provided tests, you will need to implement
    the test adapter at [adapters.run_swiglu].
    '''
    def __init__(self, device=None, dtype=None ):
        """
        Given key (K), query (Q), and value (V) tensors, return
        the output of your scaled dot product attention implementation.

        Args:
            Q (Float[Tensor, " ... queries d_k"]): Query tensor
            K (Float[Tensor, " ... keys d_k"]): Key tensor
            V (Float[Tensor, " ... values d_v"]): Values tensor
            mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
        Returns:
            Float[Tensor, " ... queries d_v"]: Output of SDPA
        """
        super().__init__()
        # self.Q = Q
        # self.K = K 
        # self.V = V 
        # self.mask = mask 
        

    def softmax(self, in_features: Float[Tensor, " ..."], dim: int): # -> Float[Tensor, " ..."]:
        """
        Given a tensor of inputs, return the output of softmaxing the given `dim`
        of the input.

        Args:
            in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
            dim (int): Dimension of the `in_features` to apply softmax to.

        Returns:
            Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
            softmax normalizing the specified `dim`.
        """
        in_features_max = in_features.max(dim=dim, keepdim = True).values
        exp = torch.exp(in_features - in_features_max)
        
        return exp / exp.sum(dim, keepdim=True)
    

    def forward(self, Q, K, V, mask): # -> Float[Tensor, " ... queries d_v"]:
        self.Q = Q
        self.K = K 
        self.V = V 
        self.mask = mask 
        matQK = torch.matmul(self.Q, self.K.transpose(-2,-1))
        if self.mask is not None:
            matQK[~self.mask] -= math.inf
        dk = self.Q.shape[-1]
        # dim = len(matQK.shape)
        # print('here-->', dim)
        
        matA = self.softmax(matQK/math.sqrt(dk), -1) 
        return torch.matmul(matA,self.V)









    # run uv run pytest -k test_swiglu
    