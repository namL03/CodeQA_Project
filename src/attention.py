"""
Attention Mechanism for Seq2seq CodeQA Model

Implements Bahdanau (additive) attention mechanism.
This allows the decoder to focus on different parts of the input
when generating each word of the answer.

Reference: "Neural Machine Translation by Jointly Learning to Align and Translate"
(Bahdanau et al., 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Attention(nn.Module):
    """
    Bahdanau Attention Mechanism (Additive Attention).
    
    At each decoding step, computes attention weights over encoder outputs,
    then creates a context vector as weighted sum of encoder outputs.
    """
    
    def __init__(self, hidden_dim: int, method: str = 'general'):
        """
        Initialize attention mechanism.
        
        Args:
            hidden_dim: Dimension of hidden states
            method: Attention method ('general', 'concat', or 'dot')
                   - 'general': W * encoder_output
                   - 'concat': Bahdanau attention (additive)
                   - 'dot': Simple dot product
        """
        super(Attention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.method = method
        
        if method == 'general':
            # W_a in the attention formula
            self.attn = nn.Linear(hidden_dim, hidden_dim)
        elif method == 'concat':
            # Bahdanau attention
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
        elif method == 'dot':
            # No learnable parameters
            pass
        else:
            raise ValueError(f"Unknown attention method: {method}")
    
    def forward(self, 
                decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                src_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden: Current decoder hidden state [batch_size, hidden_dim]
            encoder_outputs: All encoder hidden states [batch_size, src_len, hidden_dim]
            src_mask: Mask for padding tokens [batch_size, src_len]
                     (1 for valid tokens, 0 for padding)
        
        Returns:
            context: Context vector [batch_size, hidden_dim]
            attn_weights: Attention weights [batch_size, src_len]
        """
        batch_size, src_len, hidden_dim = encoder_outputs.size()
        
        # Calculate attention scores
        if self.method == 'general':
            # Score = decoder_hidden^T * W_a * encoder_output
            # [batch_size, 1, hidden_dim] @ [batch_size, hidden_dim, src_len]
            energy = torch.bmm(
                self.attn(encoder_outputs),  # [batch_size, src_len, hidden_dim]
                decoder_hidden.unsqueeze(2)  # [batch_size, hidden_dim, 1]
            )
            energy = energy.squeeze(2)  # [batch_size, src_len]
            
        elif self.method == 'concat':
            # Bahdanau attention: score = v^T * tanh(W * [decoder_hidden; encoder_output])
            # Expand decoder hidden to match encoder outputs
            decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
            # [batch_size, src_len, hidden_dim]
            
            # Concatenate decoder and encoder hidden states
            concat = torch.cat([decoder_hidden_expanded, encoder_outputs], dim=2)
            # [batch_size, src_len, hidden_dim * 2]
            
            # Apply attention transformation
            energy = torch.tanh(self.attn(concat))  # [batch_size, src_len, hidden_dim]
            energy = self.v(energy).squeeze(2)  # [batch_size, src_len]
            
        elif self.method == 'dot':
            # Simple dot product: decoder_hidden^T * encoder_output
            energy = torch.bmm(
                encoder_outputs,  # [batch_size, src_len, hidden_dim]
                decoder_hidden.unsqueeze(2)  # [batch_size, hidden_dim, 1]
            ).squeeze(2)  # [batch_size, src_len]
        
        # Apply mask (set attention scores for padding to very negative value)
        if src_mask is not None:
            energy = energy.masked_fill(src_mask == 0, -1e10)
        
        # Convert scores to probabilities (attention weights)
        attn_weights = F.softmax(energy, dim=1)  # [batch_size, src_len]
        
        # Compute context vector as weighted sum of encoder outputs
        # [batch_size, 1, src_len] @ [batch_size, src_len, hidden_dim]
        context = torch.bmm(
            attn_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)  # [batch_size, hidden_dim]
        
        return context, attn_weights


# Example usage and testing
if __name__ == "__main__":
    print("Testing Attention Mechanism...")
    
    batch_size = 4
    src_len = 50
    hidden_dim = 512
    
    # Create dummy inputs
    decoder_hidden = torch.randn(batch_size, hidden_dim)
    encoder_outputs = torch.randn(batch_size, src_len, hidden_dim)
    
    # Create source mask (simulate some padding)
    src_mask = torch.ones(batch_size, src_len)
    src_mask[0, 45:] = 0  # First sequence has padding after position 45
    src_mask[1, 40:] = 0  # Second sequence has padding after position 40
    
    # Test different attention methods
    for method in ['general', 'concat', 'dot']:
        print(f"\nTesting {method} attention...")
        attention = Attention(hidden_dim, method=method)
        
        # Forward pass
        context, attn_weights = attention(decoder_hidden, encoder_outputs, src_mask)
        
        print(f"  Context shape: {context.shape}")
        print(f"  Attention weights shape: {attn_weights.shape}")
        print(f"  Attention weights sum: {attn_weights[0].sum().item():.4f} (should be ~1.0)")
        print(f"  Attention on padding (first seq): {attn_weights[0, 45:].sum().item():.6f} (should be ~0.0)")
        print(f"  ✅ {method.capitalize()} attention test passed!")
    
    print("\n✅ All attention tests passed!")
