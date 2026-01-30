import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *


class Head(nn.Module):
    """attentionhead"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dimension, head_size, bias=False)
        self.query = nn.Linear(embedding_dimension, head_size, bias=False)
        self.value = nn.Linear(embedding_dimension, head_size, bias=False)
        self.register_buffer(
            "causal_mask", torch.tril(
                torch.ones(context_length, context_length))
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        # (batch_size, seq_length, head_size)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # print("k", k,q,v)

        # attention scores
        scores = (
            q @ k.transpose(-2, -1) * (embedding_dimension**-0.5)
        )  # @ -> batch-wise matrix multiplication
        # print(scores)
        # apply causal mask
        scores = scores.masked_fill(
            self.causal_mask[:seq_length, :seq_length] == 0, float("-inf")
        )

        # (batch_size, seq_length, seq_length)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # weighted values
        output = weights @ v  # (batch_size, seq_length, head_size)
        return output


class MultiHead(nn.Module):
    """attentionheads in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # connect heads
        # print(out)
        out = self.dropout(self.proj(out))  # linear projection and dropout
        # print(out)
        return out


class FF(nn.Module):
    """position-wise ffn, feedforward_network"""

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # print(self.net(x))
        return self.net(x)


class Block(nn.Module):
    """communication computation"""

    def __init__(self, num_embd, num_head):
        super().__init__()
        head_size = num_embd // num_head
        self.sa = MultiHead(num_head, head_size)
        self.ffwd = FF(num_embd)
        self.ln1 = nn.LayerNorm(num_embd)
        self.ln2 = nn.LayerNorm(num_embd)

    def forward(self, x):
        # residual connection -> gradient flow
        x = x + self.sa(self.ln1(x))
        # training stability
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """main class"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, embedding_dimension)
        self.position_embedding_table = nn.Embedding(
            context_length, embedding_dimension
        )
        self.blocks = nn.Sequential(
            *[
                Block(embedding_dimension, n_attention_heads)
                for _ in range(n_transformer_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embedding_dimension)
        self.lm_head = nn.Linear(embedding_dimension, vocab_size)

      # init weights
        self.apply(self._init_weights)

    # weights of linear and embedding layers
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        # transformer blocks
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)

        # head
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # calculate loss
        if targets is None:
            loss_val = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss_val = F.cross_entropy(logits, targets)

        return logits, loss_val

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]

            # predictions
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
