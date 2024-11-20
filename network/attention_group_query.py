import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupQueryAttention(nn.Module):
    """
    Group Query Attention (GQA) implementation.

    GQA uses fewer query heads than key-value heads to reduce computation while maintaining model quality.
    As described in the paper: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"

    Args:
        embed_dim (int): Total embedding dimension
        num_query_heads (int): Number of query heads
        num_kv_heads (int): Number of key/value heads (should be less than or equal to num_query_heads)
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim, num_query_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = embed_dim // num_query_heads

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, attn_mask=None):
        """
        Forward pass for GQA.

        Args:
            query: (batch_size, seq_length_q, embed_dim)
            key: (batch_size, seq_length_k, embed_dim)
            value: (batch_size, seq_length_k, embed_dim)
            attn_mask: Optional mask of shape (seq_length_q, seq_length_k) or (batch_size, seq_length_q, seq_length_k)

        Returns:
            output: (batch_size, seq_length_q, embed_dim)
            attention_weights: (batch_size, num_query_heads, seq_length_q, seq_length_k)
        """
        batch_size = query.size(0)
        seq_length_q = query.size(1)
        seq_length_k = key.size(1)

        # Project inputs
        q = self.q_proj(query)  # (batch_size, seq_length_q, embed_dim)
        k = self.k_proj(key)    # (batch_size, seq_length_k, num_kv_heads * head_dim)
        v = self.v_proj(value)  # (batch_size, seq_length_k, num_kv_heads * head_dim)

        # Reshape q, k, v for multi-head attention
        q = q.view(batch_size, seq_length_q, self.num_query_heads, self.head_dim)
        k = k.view(batch_size, seq_length_k, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_length_k, self.num_kv_heads, self.head_dim)

        # Repeat k,v for each query head in the group
        k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Transpose for attention calculation
        q = q.transpose(1, 2)  # (batch_size, num_query_heads, seq_length_q, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_query_heads, seq_length_k, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_query_heads, seq_length_k, head_dim)

        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)

        # Calculate output
        out = torch.matmul(attn_weights, v)  # (batch_size, num_query_heads, seq_length_q, head_dim)

        # Reshape output
        out = out.transpose(1, 2).contiguous()  # (batch_size, seq_length_q, num_query_heads, head_dim)
        out = out.view(batch_size, seq_length_q, self.embed_dim)

        # Final projection and dropout
        out = self.out_proj(out)
        out = self.output_dropout(out)

        return out, attn_weights

# Test the implementation
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model parameters
    batch_size = 8
    seq_length_q = 128
    seq_length_k = 256
    embed_dim = 512
    num_query_heads = 8
    num_kv_heads = 2  # Fewer KV heads than query heads

    # Initialize model
    gqa = GroupQueryAttention(
        embed_dim=embed_dim,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads
    ).to(device)

    # Create sample inputs
    query = torch.randn(batch_size, seq_length_q, embed_dim).to(device)
    key = torch.randn(batch_size, seq_length_k, embed_dim).to(device)
    value = torch.randn(batch_size, seq_length_k, embed_dim).to(device)

    # Create attention mask (optional)
    mask = torch.ones(seq_length_q, seq_length_k).to(device)

    # Test in both training and evaluation modes
    gqa.train()
    output_train, weights_train = gqa(query, key, value, mask)

    gqa.eval()
    with torch.no_grad():
        output_eval, weights_eval = gqa(query, key, value, mask)

    # Print shapes and statistics
    print("\nModel Configuration:")
    print(f"Query heads: {num_query_heads}")
    print(f"KV heads: {num_kv_heads}")
    print(f"Queries per KV: {num_query_heads // num_kv_heads}")

    print("\nOutput Shapes:")
    print(f"Output: {output_train.shape}")
    print(f"Attention Weights: {weights_train.shape}")

    print("\nMemory Usage:")
    print(f"Parameters: {sum(p.numel() for p in gqa.parameters())/1e6:.2f}M")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
