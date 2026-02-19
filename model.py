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

    def __init__(self, n, size):
        super().__init__()

        # list of heads
        self.heads = torch.nn.ModuleList()
        for _ in range(n):
            self.heads.append(Head(size))

        # final projection
        self.proj = torch.nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x):

        t = []
        for i in self.heads:
            a = i(x)
            t.append(a)

        combine = torch.cat(t, dim=-1)

        # linear layer
        result = self.proj(combine)

        return result
