"""
Complete Seq2seq Model with Attention and Copy Mechanism for CodeQA

This module combines all components into a complete model:
- Encoder (Bi-LSTM)
- Decoder (LSTM with Attention)
- Copy Mechanism (Pointer-Generator)

Following the architecture from:
- "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)
- "Get To The Point: Summarization with Pointer-Generator Networks" (See et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

try:
    from src.encoder import Encoder
    from src.decoder import Decoder
    from src.copy_mechanism import CopyMechanism
except ImportError:
    from encoder import Encoder
    from decoder import Decoder
    from copy_mechanism import CopyMechanism


class Seq2SeqWithCopy(nn.Module):
    """
    Complete Seq2seq model with attention and copy mechanism for CodeQA.
    
    Input format: [CLS] Question [SEP] Code
    Output: Answer tokens
    """
    
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 512,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 attention_method: str = 'general',
                 use_copy: bool = True):
        """
        Initialize the Seq2seq model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            attention_method: Type of attention ('general', 'concat', or 'dot')
            use_copy: Whether to use copy mechanism
        """
        super(Seq2SeqWithCopy, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_copy = use_copy
        
        # Encoder
        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_method=attention_method
        )
        
        # Copy mechanism (optional)
        if use_copy:
            self.copy_mechanism = CopyMechanism(hidden_dim)
    
    def forward(self,
                src_tokens: torch.Tensor,
                src_lengths: torch.Tensor,
                tgt_tokens: Optional[torch.Tensor] = None,
                max_decode_len: int = 50,
                teacher_forcing_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            src_tokens: Source token indices [batch_size, src_len]
            src_lengths: Source sequence lengths [batch_size]
            tgt_tokens: Target token indices [batch_size, tgt_len] (for training)
            max_decode_len: Maximum decoding length (for inference)
            teacher_forcing_ratio: Probability of using teacher forcing during training
        
        Returns:
            Dictionary containing:
            - logits: Output logits [batch_size, tgt_len, vocab_size]
            - predictions: Predicted token indices [batch_size, tgt_len]
            - attn_weights: Attention weights [batch_size, tgt_len, src_len]
            - p_gens: Generation probabilities (if using copy) [batch_size, tgt_len]
        """
        batch_size = src_tokens.size(0)
        device = src_tokens.device
        
        # Create source mask (1 for valid tokens, 0 for padding)
        src_mask = (src_tokens != 0).float()  # Assuming 0 is PAD_TOKEN
        
        # Encode the source
        encoder_outputs, encoder_hidden = self.encoder(src_tokens, src_lengths)
        
        # Initialize decoder hidden state from encoder
        decoder_hidden = self.decoder.init_hidden_from_encoder(encoder_hidden)
        
        # Determine decoding length
        if tgt_tokens is not None:
            decode_len = tgt_tokens.size(1)
            use_teacher_forcing = True
        else:
            decode_len = max_decode_len
            use_teacher_forcing = False
        
        # Storage for outputs
        outputs = []
        predictions = []
        all_attn_weights = []
        all_p_gens = []
        
        # First decoder input is SOS token (assumed to be index 4)
        SOS_TOKEN = 4
        decoder_input = torch.full((batch_size, 1), SOS_TOKEN, dtype=torch.long, device=device)
        
        # Decode step by step
        for t in range(decode_len):
            # Get input embedding for copy mechanism
            decoder_input_embed = self.decoder.embedding(decoder_input)
            
            # Decoder forward step
            output, decoder_hidden, attn_weights = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs,
                src_mask
            )
            
            # Apply copy mechanism if enabled
            if self.use_copy and self.training:
                # Get context from attention weights
                context = torch.bmm(
                    attn_weights.unsqueeze(1),
                    encoder_outputs
                ).squeeze(1)
                
                # Compute final distribution with copy mechanism
                final_dist, p_gen = self.copy_mechanism(
                    decoder_hidden[0][-1],  # Last layer's hidden state
                    context,
                    decoder_input_embed.squeeze(1),
                    output,
                    attn_weights,
                    src_tokens,
                    self.vocab_size
                )
                
                # Convert back to logits for loss computation
                output = torch.log(final_dist + 1e-10)
                all_p_gens.append(p_gen)
            
            outputs.append(output)
            all_attn_weights.append(attn_weights)
            
            # Get prediction
            pred = output.argmax(dim=1, keepdim=True)
            predictions.append(pred)
            
            # Determine next input
            if use_teacher_forcing and torch.rand(1).item() < teacher_forcing_ratio:
                # Use ground truth as next input
                decoder_input = tgt_tokens[:, t].unsqueeze(1)
            else:
                # Use model's prediction as next input
                decoder_input = pred
        
        # Stack outputs
        logits = torch.stack(outputs, dim=1)  # [batch_size, decode_len, vocab_size]
        predictions = torch.cat(predictions, dim=1)  # [batch_size, decode_len]
        attn_weights = torch.stack(all_attn_weights, dim=1)  # [batch_size, decode_len, src_len]
        
        result = {
            'logits': logits,
            'predictions': predictions,
            'attn_weights': attn_weights
        }
        
        if self.use_copy and all_p_gens:
            result['p_gens'] = torch.stack(all_p_gens, dim=1)
        
        return result
    
    def generate(self,
                 src_tokens: torch.Tensor,
                 src_lengths: torch.Tensor,
                 max_len: int = 50,
                 sos_token: int = 4,
                 eos_token: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate answer for given source (greedy decoding).
        
        Args:
            src_tokens: Source tokens [batch_size, src_len]
            src_lengths: Source lengths [batch_size]
            max_len: Maximum generation length
            sos_token: Start of sequence token index
            eos_token: End of sequence token index
        
        Returns:
            predictions: Generated token indices [batch_size, gen_len]
            attn_weights: Attention weights [batch_size, gen_len, src_len]
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                tgt_tokens=None,
                max_decode_len=max_len,
                teacher_forcing_ratio=0.0
            )
        
        return result['predictions'], result['attn_weights']


# Example usage and testing
if __name__ == "__main__":
    print("Testing Complete Seq2seq Model with Copy Mechanism...")
    
    vocab_size = 10000
    batch_size = 4
    src_len = 50
    tgt_len = 20
    
    # Create dummy data
    src_tokens = torch.randint(1, vocab_size, (batch_size, src_len))  # Start from 1 (avoid PAD)
    src_lengths = torch.tensor([50, 45, 40, 35])
    tgt_tokens = torch.randint(1, vocab_size, (batch_size, tgt_len))
    
    # Create model
    model = Seq2SeqWithCopy(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        attention_method='general',
        use_copy=True
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training mode forward pass
    print("\n--- Training Mode (with teacher forcing) ---")
    model.train()
    result = model(
        src_tokens=src_tokens,
        src_lengths=src_lengths,
        tgt_tokens=tgt_tokens,
        teacher_forcing_ratio=0.5
    )
    
    print(f"Output logits shape: {result['logits'].shape}")
    print(f"Predictions shape: {result['predictions'].shape}")
    print(f"Attention weights shape: {result['attn_weights'].shape}")
    if 'p_gens' in result:
        print(f"Generation probabilities shape: {result['p_gens'].shape}")
        print(f"Sample p_gen values: {result['p_gens'][0, :5].squeeze()}")
    
    # Inference mode
    print("\n--- Inference Mode (greedy decoding) ---")
    predictions, attn_weights = model.generate(
        src_tokens=src_tokens,
        src_lengths=src_lengths,
        max_len=20
    )
    
    print(f"Generated predictions shape: {predictions.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Sample prediction: {predictions[0, :10]}")
    
    print("\n✅ Complete Seq2seq model test passed!")
