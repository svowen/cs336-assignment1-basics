import torch 
from torch import nn 
from torch.nn import Module 
import math

class Rmsnorm(Module):
    # Deliverable: Implement RMSNorm as a torch.nn.Module. 
    # We recommend the following interface:

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    # Construct the RMSNorm module. This function should accept the following parameters:
    # d_model: int Hidden dimension of the model
    # eps: float = 1e-5 Epsilon value for numerical stability
    # device: torch.device | None = None Device to store the parameters on
    # dtype: torch.dtype | None = None Data type of the parameters
        super().__init__()
        self.d_model = d_model
        self.eps = eps 



    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Process an input tensor of shape
    # (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        in_dtype = x.dtype 
        x = x.to(torch.float32)

        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        out = x / rms 
        # out = out * self.weight.view(*([1] * (out.ndim - 1)), -1)

        # result = x.copy()
        # for i in range(len(result)):
        #     for j in range(len(result[i])):
        #         a_dmodel = result[i][j]
        #         rms_a = math.sqrt(a_dmodel)
        #         a_dmodel_new = torch.tensor([ai / rms_a for ai in a_dmodel])
        #         result[i][j] = a_dmodel_new


        return out.to(in_dtype)


if __name__ == '__main__':
    # Note: Remember to upcast your input to torch.float32 before performing the normalization (and
    # later downcast to the original dtype), as described above.
    # To test your implementation, implement the test adapter at [adapters.run_rmsnorm]. Then, run uv
    # run pytest -k test_rmsnorm.
    print('ok')