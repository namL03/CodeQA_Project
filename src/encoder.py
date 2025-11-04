"""
Encoder Module for Seq2seq CodeQA Model

The encoder reads the input sequence ([CLS] Question [SEP] Code) and 
produces context vectors that the decoder will use to generate answers.

We use a Bidirectional LSTM as specified in the Seq2seq paper.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    
    Takes token indices as input and produces:
    1. Encoder hidden states for each input token (for attention)
    2. Final encoder state (to initialize decoder)
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 512,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        """
        Initialize the encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(Encoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bidirectional LSTM
        # Note: hidden_dim will be split between forward and backward
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,  # Divide by 2 because bidirectional
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                src_tokens: torch.Tensor, 
                src_lengths: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the encoder.
        
        Args:
            src_tokens: Input token indices [batch_size, max_src_len]
            src_lengths: Actual lengths of each sequence [batch_size]
            
        Returns:
            encoder_outputs: Hidden states for all timesteps [batch_size, max_src_len, hidden_dim]
            encoder_hidden: Tuple of (hidden, cell) states
                - hidden: [num_layers * 2, batch_size, hidden_dim // 2]
                - cell: [num_layers * 2, batch_size, hidden_dim // 2]
        """
        batch_size, max_len = src_tokens.size()
        
        # Embed the input tokens
        # Shape: [batch_size, max_src_len, embed_dim]
        embedded = self.embedding(src_tokens)
        embedded = self.dropout(embedded)
        
        # Pack padded sequences for efficient LSTM processing
        # This tells LSTM to ignore padding tokens
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            src_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Pass through LSTM
        # encoder_outputs: [batch_size, max_src_len, hidden_dim]
        # encoder_hidden: ([num_layers*2, batch_size, hidden_dim//2], [num_layers*2, batch_size, hidden_dim//2])
        packed_outputs, encoder_hidden = self.lstm(packed_embedded)
        
        # Unpack the sequence
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, 
            batch_first=True,
            total_length=max_len
        )
        
        # encoder_outputs: [batch_size, max_src_len, hidden_dim]
        # encoder_hidden: (h, c) where each is [num_layers*2, batch_size, hidden_dim//2]
        return encoder_outputs, encoder_hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state (not typically needed as LSTM does this automatically).
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (hidden, cell) states
        """
        hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2, device=device)
        cell = torch.zeros(self.num_layers * 2, batch_size, self.hidden_dim // 2, device=device)
        return hidden, cell


# Example usage and testing
if __name__ == "__main__":
    # Test the encoder
    print("Testing Encoder...")
    
    vocab_size = 10000
    batch_size = 4
    max_src_len = 50
    
    # Create dummy input
    src_tokens = torch.randint(0, vocab_size, (batch_size, max_src_len))
    src_lengths = torch.tensor([50, 45, 40, 35])
    
    # Create encoder
    encoder = Encoder(vocab_size=vocab_size, embed_dim=256, hidden_dim=512, num_layers=2)
    
    # Forward pass
    encoder_outputs, encoder_hidden = encoder(src_tokens, src_lengths)
    
    print(f"Input shape: {src_tokens.shape}")
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Encoder hidden state shape: {encoder_hidden[0].shape}")
    print(f"Encoder cell state shape: {encoder_hidden[1].shape}")
    print("\n✅ Encoder test passed!")
