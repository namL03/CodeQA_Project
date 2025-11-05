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
from collections import Counter
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.seq2seq_model import Seq2SeqWithCopy
from src.vocabulary import Vocabulary
from src.data_loader import CodeQADataLoader
from src.dataset import CodeQADataset, collate_fn


def compute_bleu(reference_tokens, hypothesis_tokens, max_n=4):
    """
    Compute BLEU score for a single reference-hypothesis pair.
    
    Args:
        reference_tokens: List of reference tokens
        hypothesis_tokens: List of hypothesis tokens
        max_n: Maximum n-gram order (default: 4 for BLEU-4)
    
    Returns:
        BLEU score (0-100)
    """
    # Remove special tokens
    special_tokens = {'[PAD]', '[UNK]', '[CLS]', '[SEP]', '[SOS]', '[EOS]'}
    reference_tokens = [t for t in reference_tokens if t not in special_tokens]
    hypothesis_tokens = [t for t in hypothesis_tokens if t not in special_tokens]
    
    # Handle empty sequences
    if len(hypothesis_tokens) == 0 or len(reference_tokens) == 0:
        return 0.0
    
    # Compute n-gram precisions (up to min of max_n and hypothesis length)
    effective_max_n = min(max_n, len(hypothesis_tokens))
    precisions = []
    for n in range(1, effective_max_n + 1):
        # Get n-grams
        ref_ngrams = Counter([tuple(reference_tokens[i:i+n]) for i in range(len(reference_tokens) - n + 1)])
        hyp_ngrams = Counter([tuple(hypothesis_tokens[i:i+n]) for i in range(len(hypothesis_tokens) - n + 1)])
        
        # Count matches (clipped)
        matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        
        if total == 0:
            precisions.append(0.0)
        else:
            # Apply add-1 smoothing only for higher order n-grams with no matches
            # This is similar to SacreBLEU smoothing
            if matches == 0 and n > 1:
                # Add epsilon smoothing for n-grams > 1
                precisions.append(1.0 / (2.0 * total))
            else:
                precisions.append(matches / total if matches > 0 else 0.0)
    
    # Handle case where all precisions are 0
    if all(p == 0 for p in precisions):
        return 0.0
    
    # Geometric mean of precisions
    log_precisions = [math.log(p) if p > 0 else math.log(1e-10) for p in precisions]
    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    
    # Brevity penalty
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)
    
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    # BLEU score (0-100)
    return 100 * bp * geo_mean


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
        src = batch['src_tokens'].to(device)
        tgt_input = batch['tgt_tokens'].to(device)
        tgt_output = batch['labels'].to(device)
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


def evaluate(model, dataloader, criterion, device, vocab, beam_size=4, compute_metrics=True):
    """
    Evaluate the model on validation/test set.
    
    Following "Get To The Point" paper: beam size = 4 for evaluation.
    
    Args:
        model: Seq2seq model
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        vocab: Vocabulary object for decoding tokens
        beam_size: Beam size for beam search (default: 4 as per paper)
        compute_metrics: Whether to compute BLEU/EM (slow). Set False for faster validation.
    
    Returns:
        Dictionary with 'loss', 'bleu', and 'exact_match' scores
        (bleu and exact_match are None if compute_metrics=False)
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Metrics
    total_bleu = 0
    exact_matches = 0
    total_examples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating")
        
        for batch in progress_bar:
            # Move to device
            src = batch['src_tokens'].to(device)
            tgt_input = batch['tgt_tokens'].to(device)
            tgt_output = batch['labels'].to(device)
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
            
            # Compute BLEU and Exact Match (optional, can be slow)
            if compute_metrics:
                # Get predictions using beam search (as per "Get To The Point" paper)
                predictions = model.beam_search(
                    src_tokens=src,
                    src_lengths=src_lengths,
                    beam_size=beam_size,
                    max_len=50
                )  # [batch_size, pred_len]
                
                # Compute BLEU and Exact Match for each example in batch
                for i in range(batch_size):
                    # Get reference (ground truth)
                    ref_ids = tgt_output[i].cpu().tolist()
                    ref_tokens = vocab.decode(ref_ids)
                    
                    # Get hypothesis (prediction)
                    hyp_ids = predictions[i].cpu().tolist()
                    hyp_tokens = vocab.decode(hyp_ids)
                    
                    # Compute BLEU
                    bleu_score = compute_bleu(ref_tokens, hyp_tokens)
                    total_bleu += bleu_score
                    
                    # Compute Exact Match (ignoring special tokens and case)
                    special_tokens = {'[PAD]', '[UNK]', '[CLS]', '[SEP]', '[SOS]', '[EOS]'}
                    ref_clean = [t.lower() for t in ref_tokens if t not in special_tokens]
                    hyp_clean = [t.lower() for t in hyp_tokens if t not in special_tokens]
                    
                    if ref_clean == hyp_clean:
                        exact_matches += 1
                    
                    total_examples += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    
    if compute_metrics:
        avg_bleu = total_bleu / total_examples if total_examples > 0 else 0.0
        exact_match_pct = 100 * exact_matches / total_examples if total_examples > 0 else 0.0
    else:
        avg_bleu = None
        exact_match_pct = None
    
    return {
        'loss': avg_loss,
        'bleu': avg_bleu,
        'exact_match': exact_match_pct
    }


def save_checkpoint(model, optimizer, epoch, train_loss, val_metrics, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_metrics['loss'],
        'val_bleu': val_metrics.get('bleu', None),  # May be None if not computed
        'val_exact_match': val_metrics.get('exact_match', None)  # May be None if not computed
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
    
    # Data processing arguments (optimized based on data analysis)
    parser.add_argument('--max_src_len', type=int, default=200,
                       help='Maximum source sequence length (default: 200, covers 100%% of examples)')
    parser.add_argument('--max_tgt_len', type=int, default=30,
                       help='Maximum target sequence length (default: 30, covers 99.8%% of examples)')
    
    # Beam search arguments (following "Get To The Point" paper)
    parser.add_argument('--beam_size', type=int, default=4,
                       help='Beam size for evaluation (default: 4 as per paper)')
    
    # Early stopping arguments
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience: stop if no improvement for N epochs (default: 5)')
    
    # Evaluation arguments
    parser.add_argument('--compute_metrics_every', type=int, default=1,
                       help='Compute BLEU/EM every N epochs (default: 1). Set to 0 to only compute at end.')
    
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
    print(f"Early stopping patience: {args.early_stopping_patience} epochs")
    print(f"Beam size for evaluation: {args.beam_size}")
    print("=" * 80)
    
    best_val_loss = float('inf')
    best_bleu = None
    epochs_without_improvement = 0
    
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
        
        # Decide whether to compute metrics this epoch
        compute_metrics = (args.compute_metrics_every > 0 and 
                          (epoch + 1) % args.compute_metrics_every == 0)
        
        # Evaluate
        val_metrics = evaluate(
            model, dev_loader, criterion, args.device, vocab, 
            beam_size=args.beam_size,
            compute_metrics=compute_metrics
        )
        val_loss = val_metrics['loss']
        val_bleu = val_metrics['bleu']
        val_exact_match = val_metrics['exact_match']
        
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}")
        if val_bleu is not None:
            print(f"Val Loss: {val_loss:.4f} | BLEU: {val_bleu:.2f} | Exact Match: {val_exact_match:.2f}%")
        else:
            print(f"Val Loss: {val_loss:.4f} (metrics not computed this epoch)")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.save_dir,
                f'checkpoint_{args.language}_epoch{epoch + 1}.pt'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, val_metrics, checkpoint_path)
        
        # Save best model and track early stopping (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_bleu = val_bleu if val_bleu is not None else best_bleu
            epochs_without_improvement = 0
            best_model_path = os.path.join(
                args.save_dir,
                f'best_model_{args.language}.pt'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, val_metrics, best_model_path)
            if val_bleu is not None:
                print(f"✅ New best model saved! Val Loss: {val_loss:.4f} | BLEU: {val_bleu:.2f} | Exact Match: {val_exact_match:.2f}%")
            else:
                print(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"⚠️  No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if best_bleu is not None:
        print(f"Best BLEU score: {best_bleu:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
