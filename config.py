import torch
import tiktoken

# config
batch_size = 8  # seq processed in parallel
context_length = 32  # max length of predictions
max_training_steps = 5000
evaluation_frequency = 500
evaluation_iterations = 100  # iterations for evaluation metrics
learning_rate = 3e-4  # sweet spot for Adam optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"

# hyperparameters
embedding_dimension = 128  # dimensionality of token embeddings
n_attention_heads = 8  # n of attention heads
n_transformer_layers = 6  # n of layers
dropout_rate = 0.1  # rate for regularization -> 0.0 no dropout

# registery number of specific starship
torch.manual_seed(1701)

# used by openai:
# gpt-4-turbo, gpt-4, gpt-3.5-turbo, text-embedding-ada-002,
# text-embedding-3-small, text-embedding-3-large
enc = tiktoken.get_encoding("cl100k_base")

# vocabulary size
vocab_size = enc.n_vocab