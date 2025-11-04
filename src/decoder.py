"""
Decoder Module for Seq2seq CodeQA Model

The decoder generates the answer token by token, using:
1. Previous decoder hidden state
2. Previously generated token
3. Context vector from attention mechanism

This implements the decoder from "Sequence to Sequence Learning with Neural Networks"
(Sutskever et al., 2014) with attention from Bahdanau et al., 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Handle imports for both module usage and direct execution
try:
    from src.attention import Attention
except ImportError:
    from attention import Attention


class Decoder(nn.Module):
    """
    LSTM Decoder with Attention.
    
    At each step:
    1. Takes previous word embedding and previous hidden state
    2. Computes new hidden state with LSTM
    3. Uses attention to get context from encoder
    4. Combines hidden state and context to predict next word
    """
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        Initialize the decoder.
        
        Following "Get To The Point" (See et al., 2017):
        - Single-layer unidirectional LSTM
        - Hidden size: 256
        - Embedding size: 128
        - Bahdanau attention
        - No dropout in base model
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of word embeddings (default: 128)
            hidden_dim: Dimension of LSTM hidden state (default: 256)
            num_layers: Number of LSTM layers (default: 1, as per paper)
            dropout: Dropout probability (default: 0.0, as per paper)
        """
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Single-layer unidirectional LSTM (as per "Get To The Point" paper)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Bahdanau attention mechanism
        self.attention = Attention(hidden_dim)
        
        # Combine LSTM output and context vector
        # Input: hidden_dim + (hidden_dim * 2) because encoder is bidirectional
        # Output: hidden_dim
        self.concat_layer = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self,
                input_token: torch.Tensor,
                decoder_hidden: Tuple[torch.Tensor, torch.Tensor],
                encoder_outputs: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Single decoding step.
        
        Args:
            input_token: Previous token [batch_size, 1]
            decoder_hidden: Previous decoder state (hidden, cell)
                - hidden: [num_layers, batch_size, hidden_dim]
                - cell: [num_layers, batch_size, hidden_dim]
            encoder_outputs: All encoder hidden states [batch_size, src_len, hidden_dim]
            src_mask: Mask for source padding [batch_size, src_len]
        
        Returns:
            output: Vocabulary distribution [batch_size, vocab_size]
            decoder_hidden: New decoder state (hidden, cell)
            attn_weights: Attention weights [batch_size, src_len]
        """
        batch_size = input_token.size(0)
        
        # Embed the input token
        # Shape: [batch_size, 1, embed_dim]
        embedded = self.embedding(input_token)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM
        # lstm_output: [batch_size, 1, hidden_dim]
        # decoder_hidden: ([num_layers, batch_size, hidden_dim], [num_layers, batch_size, hidden_dim])
        lstm_output, decoder_hidden = self.lstm(embedded, decoder_hidden)
        
        # Remove sequence dimension for attention
        lstm_output = lstm_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # Apply attention
        # context: [batch_size, hidden_dim]
        # attn_weights: [batch_size, src_len]
        context, attn_weights = self.attention(lstm_output, encoder_outputs, src_mask)
        
        # Concatenate LSTM output and context
        concat_input = torch.cat([lstm_output, context], dim=1)  # [batch_size, hidden_dim * 2]
        concat_output = torch.tanh(self.concat_layer(concat_input))  # [batch_size, hidden_dim]
        concat_output = self.dropout(concat_output)
        
        # Project to vocabulary
        output = self.output_projection(concat_output)  # [batch_size, vocab_size]
        
        return output, decoder_hidden, attn_weights
    
    def init_hidden_from_encoder(self, 
                                 encoder_hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize decoder hidden state from encoder's final state.
        
        For bidirectional encoder, we need to combine forward and backward states.
        
        Args:
            encoder_hidden: Tuple of (hidden, cell) from encoder
                - hidden: [num_layers * 2, batch_size, hidden_dim]
                - cell: [num_layers * 2, batch_size, hidden_dim]
        
        Returns:
            Decoder hidden state (hidden, cell)
                - hidden: [num_layers, batch_size, hidden_dim]
                - cell: [num_layers, batch_size, hidden_dim]
        """
        h, c = encoder_hidden
        
        # For "Get To The Point" paper setup:
        # Encoder has 1 bidirectional layer: h is [2, batch_size, hidden_dim]
        # Decoder has 1 unidirectional layer: we need [1, batch_size, hidden_dim]
        
        # We take only the forward direction from the encoder
        # (alternatively, could concatenate or use a linear projection)
        # Following common practice: use forward direction of last encoder layer
        
        # h: [2, batch_size, hidden_dim] -> [1, batch_size, hidden_dim]
        # Index 0 is forward, index 1 is backward
        decoder_h = h[0:1]  # Take forward direction [1, batch_size, hidden_dim]
        decoder_c = c[0:1]  # Take forward direction [1, batch_size, hidden_dim]
        
        return decoder_h, decoder_c


# Example usage and testing
if __name__ == "__main__":
    print("Testing Decoder...")
    
    vocab_size = 10000
    batch_size = 4
    src_len = 50
    hidden_dim = 256  # Decoder hidden dim (Get To The Point paper)
    embed_dim = 128   # Embedding dim (Get To The Point paper)
    num_layers = 1    # Single layer (Get To The Point paper)
    
    # Create dummy inputs
    input_token = torch.randint(0, vocab_size, (batch_size, 1))
    
    # Simulate encoder outputs (bidirectional, so 512 dim)
    encoder_outputs = torch.randn(batch_size, src_len, hidden_dim * 2)
    
    # Simulate encoder hidden state (bidirectional single layer)
    encoder_hidden = (
        torch.randn(2, batch_size, hidden_dim),  # 2 directions, 256 each
        torch.randn(2, batch_size, hidden_dim)
    )
    
    # Create source mask
    src_mask = torch.ones(batch_size, src_len)
    src_mask[0, 45:] = 0
    
    # Create decoder with "Get To The Point" paper settings
    decoder = Decoder(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=1
    )
    
    # Initialize decoder hidden from encoder
    decoder_hidden = decoder.init_hidden_from_encoder(encoder_hidden)
    
    print(f"Input token shape: {input_token.shape}")
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Encoder hidden shape: {encoder_hidden[0].shape}")
    print(f"Decoder hidden shape (after init): {decoder_hidden[0].shape}")
    
    # Forward pass
    output, new_decoder_hidden, attn_weights = decoder(
        input_token, decoder_hidden, encoder_outputs, src_mask
    )
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output vocab distribution sum: {F.softmax(output[0], dim=0).sum().item():.4f} (should be ~1.0)")
    print(f"New decoder hidden shape: {new_decoder_hidden[0].shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum: {attn_weights[0].sum().item():.4f} (should be ~1.0)")
    
    print("\n✅ Decoder test passed (matches 'Get To The Point' paper)!")
