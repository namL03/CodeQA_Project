"""
Training Script for Seq2seq CodeQA Model

Following the training setup from:
1. "Get To The Point: Summarization with Pointer-Generator Networks" (See et al., 2017)
2. "CodeQA: A Question Answering Dataset for Source Code Comprehension"

Training Configuration:
- Optimizer: Adagrad (as per Get To The Point paper)
- Learning rate: 0.15 with decay (as per paper)
- Gradient clipping: 2.0 (as per paper)
- Batch size: 16 (adjustable based on GPU memory)
- Coverage loss: Optional (can be added later)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import time
import argparse
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.seq2seq_model import Seq2SeqWithCopy
from src.vocabulary import Vocabulary
from src.data_loader import CodeQADataLoader


class CodeQADataset(torch.utils.data.Dataset):
    """PyTorch Dataset for CodeQA."""
    
    def __init__(self, examples, vocab, max_src_len=400, max_tgt_len=50):
        """
        Initialize dataset.
        
        Following the paper:
        - Max source length: 400 tokens (question + code)
        - Max target length: 50 tokens (answer)
        
        Args:
            examples: List of examples from data_loader
            vocab: Vocabulary object
            max_src_len: Maximum source sequence length
            max_tgt_len: Maximum target sequence length
        """
        self.examples = examples
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get a single example."""
        example = self.examples[idx]
        
        # Tokenize
        question_tokens = example['question'].lower().split()
        code_tokens = example['code'].split()
        answer_tokens = example['answer'].lower().split()
        
        # Create source: [CLS] Question [SEP] Code
        src_tokens = ([self.vocab.CLS_TOKEN] + 
                     question_tokens + 
                     [self.vocab.SEP_TOKEN] + 
                     code_tokens)
        
        # Truncate if too long
        if len(src_tokens) > self.max_src_len:
            src_tokens = src_tokens[:self.max_src_len]
        
        # Create target: [SOS] Answer [EOS]
        tgt_input = [self.vocab.SOS_TOKEN] + answer_tokens
        tgt_output = answer_tokens + [self.vocab.EOS_TOKEN]
        
        # Truncate target
        if len(tgt_input) > self.max_tgt_len:
            tgt_input = tgt_input[:self.max_tgt_len]
            tgt_output = tgt_output[:self.max_tgt_len]
        
        # Encode to indices
        src_indices = torch.tensor(self.vocab.encode(src_tokens), dtype=torch.long)
        tgt_input_indices = torch.tensor(self.vocab.encode(tgt_input), dtype=torch.long)
        tgt_output_indices = torch.tensor(self.vocab.encode(tgt_output), dtype=torch.long)
        
        return {
            'src': src_indices,
            'tgt_input': tgt_input_indices,
            'tgt_output': tgt_output_indices,
            'src_len': len(src_indices),
            'tgt_len': len(tgt_input_indices)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Pads sequences to the same length within a batch.
    """
    src_seqs = [item['src'] for item in batch]
    tgt_input_seqs = [item['tgt_input'] for item in batch]
    tgt_output_seqs = [item['tgt_output'] for item in batch]
    src_lengths = torch.tensor([item['src_len'] for item in batch], dtype=torch.long)
    tgt_lengths = torch.tensor([item['tgt_len'] for item in batch], dtype=torch.long)
    
    # Pad sequences (pad_sequence pads with 0 by default, which is our PAD token)
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_input_padded = pad_sequence(tgt_input_seqs, batch_first=True, padding_value=0)
    tgt_output_padded = pad_sequence(tgt_output_seqs, batch_first=True, padding_value=0)
    
    return {
        'src': src_padded,
        'tgt_input': tgt_input_padded,
        'tgt_output': tgt_output_padded,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths
    }


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=2.0, teacher_forcing_ratio=1.0):
    """
    Train for one epoch.
    
    Following "Get To The Point" paper:
    - Gradient clipping: 2.0
    - Teacher forcing ratio: Start with 1.0, can decay over time
    
    Args:
        model: Seq2seq model
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        grad_clip: Gradient clipping threshold
        teacher_forcing_ratio: Probability of using teacher forcing
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move to device
        src = batch['src'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        src_lengths = batch['src_lengths'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        result = model(
            src_tokens=src,
            src_lengths=src_lengths,
            tgt_tokens=tgt_input,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # Get logits
        logits = result['logits']  # [batch_size, tgt_len, vocab_size]
        
        # Reshape for loss calculation
        batch_size, tgt_len, vocab_size = logits.size()
        logits_flat = logits.reshape(-1, vocab_size)  # [batch_size * tgt_len, vocab_size]
        tgt_flat = tgt_output.reshape(-1)  # [batch_size * tgt_len]
        
        # Calculate loss (ignore padding tokens)
        loss = criterion(logits_flat, tgt_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (as per "Get To The Point" paper)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update parameters
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on validation/test set.
    
    Args:
        model: Seq2seq model
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            # Move to device
            src = batch['src'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            src_lengths = batch['src_lengths'].to(device)
            
            # Forward pass (no teacher forcing during evaluation)
            result = model(
                src_tokens=src,
                src_lengths=src_lengths,
                tgt_tokens=tgt_input,
                teacher_forcing_ratio=0.0
            )
            
            # Get logits
            logits = result['logits']
            
            # Reshape for loss calculation
            batch_size, tgt_len, vocab_size = logits.size()
            logits_flat = logits.reshape(-1, vocab_size)
            tgt_flat = tgt_output.reshape(-1)
            
            # Calculate loss
            loss = criterion(logits_flat, tgt_flat)
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint.get('train_loss', 0)
    val_loss = checkpoint.get('val_loss', 0)
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
    
    return epoch, train_loss, val_loss


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Seq2seq CodeQA model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing the data')
    parser.add_argument('--language', type=str, default='python', choices=['python', 'java'],
                       help='Programming language to train on')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file (default: saved_models/vocab_{language}.pkl)')
    
    # Model arguments (following "Get To The Point" paper)
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension (default: 128 as per paper)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension (default: 256 as per paper)')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of LSTM layers (default: 1 as per paper)')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability (default: 0.0 as per paper)')
    parser.add_argument('--use_copy', action='store_true', default=True,
                       help='Use copy mechanism (pointer-generator)')
    
    # Training arguments (following "Get To The Point" paper)
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16, adjust based on GPU memory)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.15,
                       help='Learning rate (default: 0.15 as per paper)')
    parser.add_argument('--grad_clip', type=float, default=2.0,
                       help='Gradient clipping threshold (default: 2.0 as per paper)')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0,
                       help='Teacher forcing ratio (default: 1.0)')
    
    # Data processing arguments
    parser.add_argument('--max_src_len', type=int, default=400,
                       help='Maximum source sequence length')
    parser.add_argument('--max_tgt_len', type=int, default=50,
                       help='Maximum target sequence length')
    
    # Checkpoint arguments
    parser.add_argument('--save_dir', type=str, default='saved_models',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set default vocab path if not provided
    if args.vocab_path is None:
        args.vocab_path = f'saved_models/vocab_{args.language}.pkl'
    
    print("=" * 80)
    print("Training Seq2seq Model for CodeQA")
    print("Following 'Get To The Point' (See et al., 2017) architecture")
    print("=" * 80)
    print(f"Language: {args.language}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("=" * 80)
    
    # Load vocabulary
    print(f"\nLoading vocabulary from {args.vocab_path}...")
    vocab = Vocabulary.load(args.vocab_path)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    data_loader = CodeQADataLoader(args.data_dir, language=args.language)
    
    train_examples = data_loader.load_split('train')
    dev_examples = data_loader.load_split('dev')
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(dev_examples)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CodeQADataset(train_examples, vocab, args.max_src_len, args.max_tgt_len)
    dev_dataset = CodeQADataset(dev_examples, vocab, args.max_src_len, args.max_tgt_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\nCreating model...")
    model = Seq2SeqWithCopy(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_copy=args.use_copy
    )
    
    model = model.to(args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer (Adagrad as per "Get To The Point" paper)
    print("\nSetting up optimizer...")
    print("Using Adagrad optimizer (as per 'Get To The Point' paper)")
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=0.1)
    
    # Loss function (CrossEntropyLoss, ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is PAD token
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch, _, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, args.device,
            grad_clip=args.grad_clip,
            teacher_forcing_ratio=args.teacher_forcing_ratio
        )
        
        # Evaluate
        val_loss = evaluate(model, dev_loader, criterion, args.device)
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.save_dir,
                f'checkpoint_{args.language}_epoch{epoch + 1}.pt'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(
                args.save_dir,
                f'best_model_{args.language}.pt'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, best_model_path)
            print(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
