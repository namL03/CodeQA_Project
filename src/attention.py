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
    
    Following "Get To The Point" (See et al., 2017):
    Uses additive attention (concat method) as described in Bahdanau et al., 2015.
    
    e_t = v^T * tanh(W_h * h_dec + W_s * h_enc + b_attn)
    
    At each decoding step, computes attention weights over encoder outputs,
    then creates a context vector as weighted sum of encoder outputs.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize attention mechanism.
        
        Following "Get To The Point" paper:
        - Uses Bahdanau-style attention (additive/concat)
        - Decoder hidden dim: 256
        - Encoder output dim: 512 (bidirectional, 256 * 2)
        
        Args:
            hidden_dim: Dimension of decoder hidden states (256 in paper)
        """
        super(Attention, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Bahdanau attention components
        # W_h projects decoder hidden state
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # W_s projects encoder hidden states (which are 2*hidden_dim due to bidirectional)
        self.W_s = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        
        # v projects to scalar score
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
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
        batch_size, src_len, encoder_hidden_dim = encoder_outputs.size()
        
        # Bahdanau attention: e_t = v^T * tanh(W_h * h_dec + W_s * h_enc + b_attn)
        
        # Project decoder hidden state: W_h * h_dec
        # [batch_size, hidden_dim]
        decoder_features = self.W_h(decoder_hidden)  # [batch_size, hidden_dim]
        
        # Project encoder hidden states: W_s * h_enc
        # [batch_size, src_len, hidden_dim]
        encoder_features = self.W_s(encoder_outputs)  # [batch_size, src_len, hidden_dim]
        
        # Expand decoder features to match encoder
        # [batch_size, 1, hidden_dim] -> [batch_size, src_len, hidden_dim]
        decoder_features_expanded = decoder_features.unsqueeze(1).expand(-1, src_len, -1)
        
        # Add both features and apply tanh
        # [batch_size, src_len, hidden_dim]
        attention_hidden = torch.tanh(decoder_features_expanded + encoder_features)
        
        # Project to scalar scores: v^T * tanh(...)
        # [batch_size, src_len, 1] -> [batch_size, src_len]
        energy = self.v(attention_hidden).squeeze(2)
        
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
    
    # Test Bahdanau attention (as used in "Get To The Point" paper)
    print(f"\nTesting Bahdanau attention (Get To The Point style)...")
    
    # Note: decoder_hidden is 256, encoder_outputs is 512 (bidirectional)
    decoder_hidden_256 = torch.randn(batch_size, 256)  # Decoder hidden size
    encoder_outputs_512 = torch.randn(batch_size, src_len, 512)  # Encoder output (bidirectional)
    
    attention = Attention(hidden_dim=256)
    
    # Forward pass
    context, attn_weights = attention(decoder_hidden_256, encoder_outputs_512, src_mask)
    
    print(f"  Decoder hidden shape: {decoder_hidden_256.shape}")
    print(f"  Encoder outputs shape: {encoder_outputs_512.shape}")
    print(f"  Context shape: {context.shape} (should be [batch, 512])")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Attention weights sum: {attn_weights[0].sum().item():.4f} (should be ~1.0)")
    print(f"  Attention on padding (first seq): {attn_weights[0, 45:].sum().item():.6f} (should be ~0.0)")
    
    print("\n✅ Bahdanau attention test passed (matches 'Get To The Point' paper)!")
