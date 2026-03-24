from rmsnorm import Rmsnorm
from mha import MHA
from linear import Linear

class transformer_block:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
    
    def layer_mha(self, x):
        y = Rmsnorm(self.d_model).forward(x)
        heads = MHA()
        y = heads(self.d_model, self.num_heads)
        y += x

        return y 

    def layer_ff(self, x):
        y = Rmsnorm(self.d_model).forward(x)
        Linear(self.d_model, self.d_ff).forward(x)
        y = heads(self.d_model, self.num_heads)
        y += x

        return y 





# y= x + MultiHeadSelfAttention(RMSNorm(x)). (15)
# Problem (transformer_block): Implement the Transformer block (3 points)
# Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. Your
# Transformer block should accept (at least) the following parameters.
# d_model: int Dimensionality of the Transformer block inputs.
# num_heads: int Number of heads to use in multi-head self-attention.
# d_ff: int Dimensionality of the position-wise feed-forward inner layer.
# 26
# To test your implementation, implement the adapter [adapters.run_transformer_block]. Then
# run uv run pytest -k test_transformer_block to test your implementation.
# Deliverable: Transformer block code that passes the provided tests.