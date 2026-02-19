

class FeedForward(nn.Module):
    """ Simple Feed Forward Network (Computation layer) """

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),  # Expand
            nn.ReLU(),                                   # Activation function
            nn.Linear(4 * embedding_dim, embedding_dim),  # Contract back
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ One Transformer Block: Communication + Computation """

    def __init__(self, num_embd, num_head):
        super().__init__()
        head_size = num_embd // num_head
        # Communication (Self-Attention)
        self.sa = MultiHead(num_head, head_size)
        self.ffwd = FeedForward(num_embd)        # Computation (Feed-Forward)
        self.ln1 = nn.LayerNorm(num_embd)        # Layer Norm 1
        self.ln2 = nn.LayerNorm(num_embd)        # Layer Norm 2

    def forward(self, x):
        # Residual connections (x + ...) help keep the signal strong
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """ The Main GPT Model """

    def __init__(self):
        super().__init__()
        # 1. Token embeddings (content)
        self.token_embedding_table = nn.Embedding(
            vocab_size, embedding_dimension)
        # 2. Position embeddings (location)
        self.position_embedding_table = nn.Embedding(
            context_length, embedding_dimension)

        # 3. Stack of Transformer Blocks
        self.blocks = nn.Sequential(*[
            Block(embedding_dimension, n_attention_heads) for _ in range(n_transformer_layers)
        ])

        # 4. Final LayerNorm and projection to vocabulary size
        self.ln_f = nn.LayerNorm(embedding_dimension)
        self.lm_head = nn.Linear(embedding_dimension, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Create embeddings (token + position)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        # Pass through transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # Final predictions (logits)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # Calculate loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Function to generate new text
        for _ in range(max_new_tokens):
            # Crop input to the context_length
            idx_cond = idx[:, -context_length:]

            # Get predictions
            logits, loss = self(idx_cond)

            # Focus only on the last time step
            logits = logits[:, -1, :]

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
