import torch
import torch.nn
from torch.nn import functional as F
from config import *


class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = torch.nn.Linear(embedding_dimension, head_size, bias=False)
        self.query = torch.nn.Linear(
            embedding_dimension, head_size, bias=False)
        self.value = torch.nn.Linear(
            embedding_dimension, head_size, bias=False)

        # mask
        self.register_buffer('tril', torch.tril(
            torch.ones(context_length, context_length)))

    def forward(self, x):
        _, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores * (C ** -0.5)

        # use mask
        mask = self.tril[:T, :T]
        scores = scores.masked_fill(mask == 0, float('-inf'))

        #  scores to probabilities
        weights = F.softmax(scores, dim=-1)

        # combine values
        result = torch.matmul(weights, v)

        return result


class MultiHead(torch.nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        # list of heads
        self.heads = torch.nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(Head(head_size))

        n_embd = num_heads * head_size

        # final projection
        self.proj = torch.nn.Linear(n_embd, n_embd)

    def forward(self, x):

        t = []
        for h in self.heads:
            a = h(x)
            t.append(a)

        combine = torch.cat(t, dim=-1)

        # linear layer
        result = self.proj(combine)

        return result


class FeedForward(torch.nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            # 4 orginal paper value
            torch.nn.Linear(embedding_dim, 4 * embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class Block(torch.nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = int(n_embd / n_head)

        # self attention
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
