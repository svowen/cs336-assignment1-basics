import torch
from torch import nn
from torch.nn import Module 
import math

class Rope(Module):
    # Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input
    # tensor. The following interface is recommended:

    # def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=torch.float32):
    #     super().__init__()
    #     assert d_k % 2 == 0
    #     half = d_k // 2
    #     j = torch.arange(half, device=device, dtype=torch.float64)
    #     inv_freq = torch.exp(-math.log(theta) * (2.0 * j / d_k))   # theta^(-2j/d)
    #     pos = torch.arange(max_seq_len, device=device, dtype=torch.float64)
    #     angles = torch.outer(pos, inv_freq)                        # [S, half]
    #     self.register_buffer("cos", angles.cos().to(dtype=dtype), persistent=False)  # [S,half]
    #     self.register_buffer("sin", angles.sin().to(dtype=dtype), persistent=False)  # [S,half]
    # def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    #     # x: [..., S, D], token_positions: [..., S]
    #     cos = self.cos[token_positions]
    #     sin = self.sin[token_positions]
    #     x_even = x[..., 0::2]
    #     x_odd  = x[..., 1::2]
    #     out_even = x_even * cos - x_odd * sin
    #     out_odd  = x_even * sin + x_odd * cos
    #     out = torch.empty_like(x)
    #     out[..., 0::2] = out_even
    #     out[..., 1::2] = out_odd
    #     return out


    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
    # Construct the RoPE module and create buffers if needed.
    # theta: float Θ value for the RoPE
    # d_k: int dimension of query and key vectors
    # max_seq_len: int Maximum sequence length that will be inputted
    # device: torch.device | None = None Device to store the buffer on
        super().__init__()
        def theta_ik(i,k): # k is the index with range(d_k//2)
            return i / (theta ** ((2*k)/d_k) )
        def r_ik(i, k):
            val = theta_ik(i,k)
            return torch.tensor([[math.cos(val), -math.sin(val)], 
                             [math.sin(val), math.cos(val)]])

        def r_i(i):
            r_i = torch.zeros(d_k, d_k, device = device)
            for k in range(d_k//2):
                r_i[2*k:2*k+2, 2*k:2*k+2] = r_ik(i, k) # tricky here
            return r_i
        
        r = torch.stack([r_i(i) for i in range(max_seq_len)], dim=0)        
        self.register_buffer('r', r, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    # Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
    # Note that you should tolerate x with an arbitrary number of batch dimensions. You should
    # assume that the token positions are a tensor of shape (..., seq_len) specifying the token
    # positions of x along the sequence dimension.
    # You should use the token positions to slice your (possibly precomputed) cos and sin tensors
    # along the sequence dimension.
        # torch.matmul()
        token_r = self.r[token_positions]
        print('shape', token_r.shape, x.shape)
        # ret = torch.matmul(token_r, x) # ... dk, dk x dk
        ret = torch.einsum('hij, bhj -> bhi', token_r, x)
        print(ret)
        return ret


        


# if __name__ == '__main__':
#     # To test your implementation, complete [adapters.run_rope] and make sure it passes uv run
#     # pytest -k test_rope.
#     rope = Rope(1,4,4)
#     rope.forward(torch.tensor([1,2,3,4]), torch.tensor([1,2,3,4])) 