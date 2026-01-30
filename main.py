"""
todo:
loss func missing
class gptmodel no ready
-> init weights forward return
"""

import torch
from config import *
from model import *

# tokenizer -> https://github.com/google/sentencepiece
# import sentencepiece as sp

with open("data.txt", "r", encoding="utf-8") as file:
    dataset = file.read()

# dataset to tensor
tensor_data = torch.tensor(enc.encode(dataset))

# 80% training, 20% validation
split = int(0.8 * len(tensor_data))
train_data = tensor_data[:split]
val_data = tensor_data[split:]


# def get_batch(split):
#     """batch of data."""
# 
#     data = train_data if split == "train" else val_data
#     random_indices = torch.randint(len(data) - context_length, (batch_size,))
#     # print(random_indices)
# 
#     inputs = torch.stack([data[i : i + context_length] for i in random_indices])
#     # print(inputs)
# 
#     targets = torch.stack(
#         [data[i + 1 : i + context_length + 1] for i in random_indices]
#     )
#     # print(targets)
# 
#     inputs, targets = inputs.to(device), targets.to(device)
# 
#     return inputs, targets


# inputs, targets = get_batch("train")
# print("inputs shapes:", inputs.shape)
# print("targets:", targets.shape)
# print("inputs:", inputs)
# print("targets:", targets)

# inputs: tensor([[16820,    11,  5568,  2357,   308, 16820,  1980, 99682],
#         [12203, 60166,   597,  2319, 31824,  7643, 14635,   477],
#         [   71, 15492, 51890, 16373, 14360,  2357, 93464, 22243],
#         [ 1441,  2357,   267,  2357, 39004, 12949,    11, 73730]])
# targets: tensor([[   11,  5568,  2357,   308, 16820,  1980, 99682,  1937],
#         [60166,   597,  2319, 31824,  7643, 14635,   477,   300],
#         [15492, 51890, 16373, 14360,  2357, 93464, 22243, 15492],
#         [ 2357,   267,  2357, 39004, 12949,    11, 73730, 10248]])


def estimate_loss(logits, targets):
    """ """
    pass

dummy = torch.randn(batch_size, context_length, embedding_dimension)
# head = Head(head_size=16)
# output = head(dummy)
# print("attentionhead:", output.shape)
# print("attentionhead:", output)

# mh = MultiHead(
#     num_heads=n_attention_heads, head_size=embedding_dimension // n_attention_heads
# )
# output = mh(dummy)
# print("multihead.shape", output.shape)
# print("multihead", output)

# ff = FF(embedding_dimension)
# output = ff(dummy)
# print("feedforward:", output.shape)
# print("ff", output)

# block = Block(embedding_dimension, n_attention_heads)
# output = block(dummy)
# print("block:", output)
# print("block_shape", output.shape)


def main():
    # init
    model = GPTModel().to("cpu")
    print(model)

if __name__ == "__main__":
    main()