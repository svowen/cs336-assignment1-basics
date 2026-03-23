import torch
from torch import nn 
from torch.nn import Module 

class Swiglu(Module):
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
    def __init__(self, d_model: int, d_ff: int, w1, w2, w3, device=None, dtype=None ):
        super().__init__()
        # self.d_model = d_model 
        # d_ff = int(d_model *  8 / 3)

        # w1 = torch.empty(d_ff, d_model, dtype = dtype, device = device)
        # w2 = torch.empty(d_model, d_ff, device = device, dtype = dtype)
        # w3 = torch.empty(d_ff, d_model, device = device, dtype = dtype)
        counter = {'w1':w1, 'w2':w2, 'w3':w3}
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)
        # nn.init.trunc_normal_(self.w1)
        # nn.init.trunc_normal_(self.w2)
        # nn.init.trunc_normal_(self.w3)
        # print(self.paras)
    
    def forward(self, x):
        # x.shape = (d_model, )
        # print(self.paras)
        x_w1 = torch.matmul(x, self.w1.T)   # ... * d_moodel, d_model * d_ff -> ... * d_ff
#         >       x_w1 = torch.matmul(x, self.w1)
#                ^^^^^^^^^^^^^^^^^^^^^^^^
# E       RuntimeError: mat1 and mat2 shapes cannot be multiplied (48x64 and 128x64)
        silu = x_w1 * torch.sigmoid(x_w1)   # ... * d_ff, ... * d_ff --> ... * d_ff
        x_w3 = torch.matmul(x, self.w3.T)   # ... * d_model, d_model * d_ff -> ... * d_ff
        print('silu, x_w3 shape', silu.shape, x_w3.shape)
        silu_x = silu * x_w3                 # (... * d_ff), ... * d_ff --> ... * d_ff
        output = torch.matmul(silu_x, self.w2.T) # (... * d_ff, d_ff * d_model) --> ... * d_model

        return output



if __name__ == '__main__':
    a = Swiglu(3)
    x = torch.tensor([1.,2.,3.])
    ret = a.forward(x)
    print(ret)





    # run uv run pytest -k test_swiglu
    