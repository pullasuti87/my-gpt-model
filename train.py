import torch
from config import enc, context_length, batch_size, device
from model import GPTModel

with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# tiktoken cl100k_base
data = torch.tensor(enc.encode(text), dtype=torch.long)

# 80/20
i = int(0.8 * len(data))
train_data = data[:i]
val_data = data[i:]


def get_batch(dataset):

    # random starting point
    start_point = torch.randint(len(dataset) - context_length, (batch_size,))

    t1 = []
    t2 = []

    for i in start_point:
        # input batch
        t1.append(dataset[i: i + context_length])

        # new batch with one token to the right
        t2.append(dataset[i+1: i + context_length+1])

    # single tensor
    x = torch.stack(t1)
    y = torch.stack(t2)

    # if gpu exists
    return x.to(device), y.to(device)


x, y = get_batch(train_data)
print("inputs shape:", x.shape)
print("targets shape:", y.shape)
print("input id:", x[0].tolist())
print("target id:", y[0].tolist())
