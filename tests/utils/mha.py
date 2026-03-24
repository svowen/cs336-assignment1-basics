import torch
from torch import nn 
from torch.nn import Module 
import math

from jaxtyping import Bool, Float, Int
from torch import Tensor


class MHA(Module):
    ''' 
    # Problem (multihead_self_attention): Implement causal multi-head self-attention (5# points)
    # Deliverable: Implement causal multi-head self-attention as a torch.nn.Module. Your implemen-
    # tation should accept (at least) the following parameters:
    # d_model: int Dimensionality of the Transformer block inputs.
    # num_heads: int Number of heads to use in multi-head self-attention.
    # Folllowing Vaswani et al. [2017], set dk = dv = dmodel/h. To test your implementation against our
    # provided tests, implement the test adapter at [adapters.run_multihead_self_attention]. Then,
    # run uv run pytest -k test_multihead_self_attention to test your implementation.
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
    

    def forward(self, d_model, num_heads, Q=None, K=None, V=None, mask=None): # -> Float[Tensor, " ... queries d_v"]:

        # step1 combining Q/K/V
        dk = d_model // num_heads
        if Q is None: # not Q: 
            Q = torch.rand(dk, dk) 
        if K is None: # not Q: 
            K = torch.rand(dk, dk)
        if V is None: # not Q: 
            V = torch.rand(dk, dk)

        # self.Q, self.K, self.V, self.mask = torch.block_diag(*([Q] * num_heads)), \
        #     torch.block_diag(*([K] * num_heads)), \
        #     torch.block_diag(*([V] * num_heads)), \
        #     torch.block_diag(*([mask] * num_heads))

        #step2 x of size dk
        matQK = torch.matmul(Q, K.transpose(-2,-1))
        if mask is not None:
            matQK[~mask] -= math.inf
        dk = Q.shape[-1]
        # dim = len(matQK.shape)
        # print('here-->', dim)
        
        matA = self.softmax(matQK/math.sqrt(dk), -1) 
        return torch.matmul(matA, V)







