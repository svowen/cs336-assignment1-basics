import torch 
from torch import nn
from torch.nn import Module

class Embedding(Module):
    # Construct an embedding module. This function should accept the following parameters:
    # num_embeddings: int Size of the vocabulary
    # embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
    # device: torch.device | None = None Device to store the parameters on
    # dtype: torch.dtype | None = None Data type of the parameters

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype)
        )
        nn.init.trunc_normal_(self.embedding_matrix, mean=0.0, std=0.02)



    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup the embedding vectors for the given token IDs.
        # The forward method should select the embedding
        # vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a
        # torch.LongTensor of token IDs with shape (batch_size, sequence_length)
        # batch_size, sequence_length = token_ids.shape 
        # embedding_output = torch.empty(batch_size, sequence_length, self.embedding_dim)
        # for i in range(batch_size):
        #     for j in range(sequence_length):
        #         token_id = token_ids[i][j]
        #         embedding_output[i][j] = self.embedding_matrix[token_id]
        # return embedding_output

        # ensure integer indices on same device as embedding matrix
        token_ids = token_ids.long().to(self.embedding_matrix.device)
        # advanced indexing returns shape (B, L, D)
        return self.embedding_matrix[token_ids]








if __name__ == '__main__':
    # Make sure to:
    # • subclass nn.Module
    # • call the superclass constructor
    # • initialize your embedding matrix as a nn.Parameter
    # • store the embedding matrix with the d_model being the final dimension
    # • of course, don’t use nn.Embedding or nn.functional.embedding
    # Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ to
    # initialize the weights.
    # To test your implementation, implement the test adapter at [adapters.run_embedding]. Then, run
    # uv run pytest -k test_embedding.
    print('here')
    token_ids = torch.tensor([[1,2,3],[4,5,6]])        # shape (2,3)
    print(Embedding(7,3).embedding_matrix)
    print(Embedding(7,3).forward(token_ids))
    print(token_ids)
