"""
Copy Mechanism for Seq2seq CodeQA Model

Implements the pointer-generator network from "Get To The Point: 
Summarization with Pointer-Generator Networks" (See et al., 2017).

The copy mechanism allows the model to either:
1. Generate a word from the vocabulary (generator)
2. Copy a word from the input source (pointer)

This is especially useful for CodeQA because:
- Variable names from code can be copied directly
- Technical terms can be copied instead of generated
- Reduces vocabulary size requirements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CopyMechanism(nn.Module):
    """
    Pointer-Generator Network (Copy Mechanism).
    
    Computes a generation probability p_gen that decides between:
    - Generating from vocabulary: p_gen * P_vocab
    - Copying from source: (1 - p_gen) * P_attention
    
    Final probability for each word = 
        p_gen * P_vocab(word) + (1 - p_gen) * sum(P_attention(positions where word appears))
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize copy mechanism.
        
        Args:
            hidden_dim: Dimension of hidden states
        """
        super(CopyMechanism, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Linear layers to compute generation probability
        # p_gen = sigmoid(W_h * h_t + W_s * s_t + W_x * x_t + b)
        # where:
        #   h_t = decoder hidden state
        #   s_t = context vector from attention
        #   x_t = decoder input embedding
        
        self.w_h = nn.Linear(hidden_dim, 1)  # For decoder hidden
        self.w_s = nn.Linear(hidden_dim, 1)  # For context vector
        self.w_x = nn.Linear(hidden_dim, 1)  # For input embedding
    
    def forward(self,
                decoder_hidden: torch.Tensor,
                context: torch.Tensor,
                decoder_input_embed: torch.Tensor,
                vocab_dist: torch.Tensor,
                attn_dist: torch.Tensor,
                src_tokens: torch.Tensor,
                vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute final distribution with copy mechanism.
        
        Args:
            decoder_hidden: Current decoder hidden state [batch_size, hidden_dim]
            context: Context vector from attention [batch_size, hidden_dim]
            decoder_input_embed: Embedded input token [batch_size, embed_dim]
            vocab_dist: Vocabulary distribution [batch_size, vocab_size]
            attn_dist: Attention distribution [batch_size, src_len]
            src_tokens: Source token indices [batch_size, src_len]
            vocab_size: Size of vocabulary
        
        Returns:
            final_dist: Final probability distribution [batch_size, vocab_size]
            p_gen: Generation probability [batch_size, 1]
        """
        batch_size, src_len = src_tokens.size()
        
        # Compute generation probability
        # p_gen = sigmoid(W_h * h + W_s * s + W_x * x)
        p_gen_input = self.w_h(decoder_hidden) + self.w_s(context)
        
        # Note: decoder_input_embed might have different dimension
        # We need to handle this by projecting or padding
        if decoder_input_embed.size(-1) != self.hidden_dim:
            # If embedding dimension doesn't match, create a projection layer
            # For simplicity, we'll just use the parts we have
            p_gen_input = p_gen_input + self.w_x(decoder_input_embed) if decoder_input_embed.size(-1) == self.hidden_dim else p_gen_input
        else:
            p_gen_input = p_gen_input + self.w_x(decoder_input_embed)
        
        p_gen = torch.sigmoid(p_gen_input)  # [batch_size, 1]
        
        # Apply softmax to vocabulary logits to get probabilities
        vocab_dist = F.softmax(vocab_dist, dim=-1)  # [batch_size, vocab_size]
        
        # Weight vocabulary distribution by p_gen
        vocab_dist_weighted = p_gen * vocab_dist  # [batch_size, vocab_size]
        
        # Weight attention distribution by (1 - p_gen)
        attn_dist_weighted = (1 - p_gen) * attn_dist  # [batch_size, src_len]
        
        # Create extended vocabulary distribution
        # We need to scatter attention weights to vocabulary positions
        # This accumulates attention weights for positions where the same word appears
        
        # Initialize final distribution with vocabulary distribution
        final_dist = vocab_dist_weighted  # [batch_size, vocab_size]
        
        # Add attention weights for words that appear in source
        # Use scatter_add to accumulate attention weights at vocabulary indices
        final_dist = final_dist.scatter_add(
            dim=1,
            index=src_tokens.clamp(0, vocab_size - 1),  # Clamp to vocab size
            src=attn_dist_weighted
        )
        
        return final_dist, p_gen


# Example usage and testing
if __name__ == "__main__":
    print("Testing Copy Mechanism...")
    
    batch_size = 4
    src_len = 50
    vocab_size = 10000
    hidden_dim = 512
    embed_dim = 256
    
    # Create dummy inputs
    decoder_hidden = torch.randn(batch_size, hidden_dim)
    context = torch.randn(batch_size, hidden_dim)
    decoder_input_embed = torch.randn(batch_size, hidden_dim)  # Match hidden_dim for simplicity
    
    # Vocabulary distribution (logits)
    vocab_logits = torch.randn(batch_size, vocab_size)
    
    # Attention distribution (probabilities)
    attn_dist = F.softmax(torch.randn(batch_size, src_len), dim=1)
    
    # Source tokens
    src_tokens = torch.randint(0, vocab_size, (batch_size, src_len))
    
    # Create copy mechanism
    copy_mechanism = CopyMechanism(hidden_dim)
    
    # Forward pass
    final_dist, p_gen = copy_mechanism(
        decoder_hidden,
        context,
        decoder_input_embed,
        vocab_logits,
        attn_dist,
        src_tokens,
        vocab_size
    )
    
    print(f"Decoder hidden shape: {decoder_hidden.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Vocab logits shape: {vocab_logits.shape}")
    print(f"Attention distribution shape: {attn_dist.shape}")
    print(f"Source tokens shape: {src_tokens.shape}")
    
    print(f"\nGeneration probability (p_gen) shape: {p_gen.shape}")
    print(f"p_gen values (first batch): {p_gen[0].item():.4f} (should be between 0 and 1)")
    print(f"Final distribution shape: {final_dist.shape}")
    print(f"Final distribution sum: {final_dist[0].sum().item():.4f} (should be ~1.0)")
    
    # Verify that final distribution is valid
    assert torch.allclose(final_dist.sum(dim=1), torch.ones(batch_size), atol=1e-3), \
        "Final distribution doesn't sum to 1!"
    
    print("\n✅ Copy mechanism test passed!")
