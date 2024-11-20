import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionBlock(nn.Module):
    """Multi-head Attention Block with optional attention masking.

    This implementation follows the scaled dot-product attention mechanism described
    in "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        embed_dim (int): Total dimension of the model embeddings.
        num_heads (int): Number of parallel attention heads. Note: embed_dim must be divisible by num_heads.
        dropout (float, optional): Dropout probability for attention weights. Defaults to 0.1.

    Attributes:
        q_proj (nn.Linear): Linear projection for queries.
        k_proj (nn.Linear): Linear projection for keys.
        v_proj (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Linear projection for output.
        scale (float): Scaling factor for dot product attention (1/√d_k).

    Shape:
        - query: (batch_size, seq_length, embed_dim)
        - key: (batch_size, seq_length, embed_dim)
        - value: (batch_size, seq_length, embed_dim)
        - attn_mask: (batch_size, seq_length, seq_length) or (seq_length, seq_length)
        - Output: tuple(
            attention_output: (batch_size, seq_length, embed_dim),
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
          )
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """Initialize the AttentionBlock.

        Args:
            embed_dim (int): Embedding dimension
            num_heads (int): Number of attention heads
            dropout (float, optional): Dropout probability. Defaults to 0.1
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear layers for queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layer
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Scaling factor for dot product attention
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, attn_mask=None):
        """Forward pass for the attention block.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_length, embed_dim)
            key (torch.Tensor): Key tensor of shape (batch_size, seq_length, embed_dim)
            value (torch.Tensor): Value tensor of shape (batch_size, seq_length, embed_dim)
            attn_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length, seq_length)
                or (seq_length, seq_length). Defaults to None.

        Returns:
            tuple:
                - output (torch.Tensor): Attention output of shape (batch_size, seq_length, embed_dim)
                - attention_weights (torch.Tensor): Attention weights of shape
                  (batch_size, num_heads, seq_length, seq_length)
        """

        batch_size = query.size(0)

        # Project queries, keys, and values
        q = self.proj_dropout(self.q_proj(query))
        k = self.proj_dropout(self.k_proj(key))
        v = self.proj_dropout(self.v_proj(value))

        # Reshape for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply attention mask if provided
        if attn_mask is not None:
            # Expand mask for multiple heads
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            if attn_mask.dim() == 3:  # [batch_size, seq_length, seq_length]
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Calculate attention output
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(batch_size, -1, self.embed_dim)
        out = self.output_dropout(self.out_proj(out))

        return out, attn_weights


if __name__ == '__main__':
    """Example usage of AttentionBlock with GPU support."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Example usage
    batch_size = 32
    seq_length = 100
    embed_dim = 256
    num_heads = 8

    # Initialize attention block
    attention = AttentionBlock(embed_dim=embed_dim, num_heads=num_heads).to(device)

    # Create sample inputs
    query = torch.randn(batch_size, seq_length, embed_dim).to(device)
    key = torch.randn(batch_size, seq_length, embed_dim).to(device)
    value = torch.randn(batch_size, seq_length, embed_dim).to(device)

    # Optional attention mask (1 for attend, 0 for mask)
    mask = torch.randn(batch_size, seq_length, seq_length).to(device) > 0.5

    attention.eval()
    with torch.no_grad():
        # Forward pass
        output, attention_weights = attention(query, key, value, mask)

    # Print diagnostic information
    print(f"\nModel device: {next(attention.parameters()).device}")
    print(f"Input shapes:")
    print(f"Query: {query.shape} on {query.device}")
    print(f"Key: {key.shape} on {key.device}")
    print(f"Value: {value.shape} on {value.device}")
    print(f"Mask: {mask.shape} on {mask.device}")
    print(f"\nOutput shapes:")
    print(f"Output: {output.shape} on {output.device}")
    print(f"Attention weights: {attention_weights.shape} on {attention_weights.device}")

    # Optional: Test memory usage
    print("\nMemory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved(device)/1024**2:.2f} MB")

    # write another test case
