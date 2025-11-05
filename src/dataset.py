"""
Dataset and DataLoader for CodeQA Seq2seq Model

This module prepares the data for training by:
1. Loading questions, code, and answers
2. Tokenizing and encoding them using vocabulary
3. Creating batches with proper padding
4. Formatting input as "[CLS] Question [SEP] Code"
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple
import random


class CodeQADataset(Dataset):
    """
    PyTorch Dataset for CodeQA.
    
    Each example contains:
    - source: [CLS] + question tokens + [SEP] + code tokens
    - target: [SOS] + answer tokens + [EOS]
    - Extended vocab positions for copy mechanism
    """
    
    def __init__(self, examples: List[Dict[str, str]], vocab, max_src_len: int = 256, 
                 max_tgt_len: int = 30):
        """
        Initialize the dataset.
        
        Based on data analysis:
        - Max source length: 256 tokens (covers 100% of examples, 99th %tile = 155)
        - Max target length: 30 tokens (covers 99.8% of examples, 99th %tile = 19)
        
        Args:
            examples: List of dicts with 'question', 'code', 'answer' keys
            vocab: Vocabulary object
            max_src_len: Maximum source sequence length (question + code)
            max_tgt_len: Maximum target sequence length (answer)
        """
        self.examples = examples
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        print(f"Dataset initialized with {len(examples)} examples")
        print(f"Max source length: {max_src_len}, Max target length: {max_tgt_len}")
    
    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Returns:
            Dictionary containing:
            - src_tokens: Source token indices
            - src_len: Source sequence length
            - tgt_tokens: Target token indices (with SOS)
            - tgt_len: Target sequence length
            - labels: Target token indices (with EOS) for loss calculation
            - src_vocab_mask: Mask for copy mechanism (which source positions are in vocab)
            - src_text: Original source tokens (for copy mechanism)
        """
        example = self.examples[idx]
        
        # Tokenize question and code
        question_tokens = example['question'].lower().split()
        code_tokens = example['code'].split()  # Keep code case-sensitive
        answer_tokens = example['answer'].lower().split()
        
        # Create source sequence: [CLS] Question [SEP] Code
        src_tokens = ([self.vocab.CLS_TOKEN] + 
                     question_tokens + 
                     [self.vocab.SEP_TOKEN] + 
                     code_tokens)
        
        # Truncate source if too long
        if len(src_tokens) > self.max_src_len:
            # Keep [CLS] and try to keep [SEP] by truncating code
            cls_and_question = [self.vocab.CLS_TOKEN] + question_tokens + [self.vocab.SEP_TOKEN]
            if len(cls_and_question) < self.max_src_len:
                # Truncate code portion
                code_space = self.max_src_len - len(cls_and_question)
                src_tokens = cls_and_question + code_tokens[:code_space]
            else:
                # Question itself is too long
                src_tokens = src_tokens[:self.max_src_len]
        
        # Create target sequence: [SOS] Answer tokens [EOS]
        tgt_tokens_with_sos = [self.vocab.SOS_TOKEN] + answer_tokens
        tgt_tokens_with_eos = answer_tokens + [self.vocab.EOS_TOKEN]
        
        # Truncate target if too long
        if len(tgt_tokens_with_sos) > self.max_tgt_len:
            tgt_tokens_with_sos = tgt_tokens_with_sos[:self.max_tgt_len]
            tgt_tokens_with_eos = tgt_tokens_with_eos[:self.max_tgt_len]
        
        # Encode tokens to indices
        src_indices = self.vocab.encode(src_tokens, add_eos=False)
        tgt_indices = self.vocab.encode(tgt_tokens_with_sos, add_eos=False)
        label_indices = self.vocab.encode(tgt_tokens_with_eos, add_eos=False)
        
        # For copy mechanism: create extended vocabulary mapping
        # Keep track of which source tokens are in vocabulary
        src_vocab_mask = []
        unk_idx = self.vocab.word2idx[self.vocab.UNK_TOKEN]
        for idx in src_indices:
            src_vocab_mask.append(1 if idx != unk_idx else 0)
        
        return {
            'src_tokens': torch.tensor(src_indices, dtype=torch.long),
            'src_len': len(src_indices),
            'tgt_tokens': torch.tensor(tgt_indices, dtype=torch.long),
            'tgt_len': len(tgt_indices),
            'labels': torch.tensor(label_indices, dtype=torch.long),
            'src_vocab_mask': torch.tensor(src_vocab_mask, dtype=torch.float),
            'src_text': src_tokens  # Keep original tokens for copy mechanism
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to create batches with padding.
    
    Args:
        batch: List of examples from __getitem__
        
    Returns:
        Batched and padded tensors
    """
    # Separate components
    src_tokens = [item['src_tokens'] for item in batch]
    tgt_tokens = [item['tgt_tokens'] for item in batch]
    labels = [item['labels'] for item in batch]
    src_vocab_masks = [item['src_vocab_mask'] for item in batch]
    src_texts = [item['src_text'] for item in batch]
    
    # Get lengths
    src_lengths = torch.tensor([item['src_len'] for item in batch], dtype=torch.long)
    tgt_lengths = torch.tensor([item['tgt_len'] for item in batch], dtype=torch.long)
    
    # Pad sequences (pad_sequence pads with 0, which is our PAD token)
    src_padded = pad_sequence(src_tokens, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_tokens, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    src_vocab_masks_padded = pad_sequence(src_vocab_masks, batch_first=True, padding_value=0)
    
    # Create attention masks (1 for real tokens, 0 for padding)
    src_mask = (src_padded != 0).float()  # Shape: (batch_size, max_src_len)
    tgt_mask = (tgt_padded != 0).float()  # Shape: (batch_size, max_tgt_len)
    
    return {
        'src_tokens': src_padded,          # (batch_size, max_src_len)
        'src_lengths': src_lengths,         # (batch_size,)
        'src_mask': src_mask,               # (batch_size, max_src_len)
        'tgt_tokens': tgt_padded,           # (batch_size, max_tgt_len)
        'tgt_lengths': tgt_lengths,         # (batch_size,)
        'tgt_mask': tgt_mask,               # (batch_size, max_tgt_len)
        'labels': labels_padded,            # (batch_size, max_tgt_len)
        'src_vocab_mask': src_vocab_masks_padded,  # (batch_size, max_src_len)
        'src_text': src_texts               # List of token lists
    }


def create_dataloaders(train_data: List[Dict], dev_data: List[Dict], 
                       vocab, batch_size: int = 32, 
                       max_src_len: int = 256, max_tgt_len: int = 30,
                       num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_data: Training examples
        dev_data: Validation examples
        vocab: Vocabulary object
        batch_size: Batch size
        max_src_len: Maximum source length
        max_tgt_len: Maximum target length
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, dev_loader)
    """
    # Create datasets
    train_dataset = CodeQADataset(train_data, vocab, max_src_len, max_tgt_len)
    dev_dataset = CodeQADataset(dev_data, vocab, max_src_len, max_tgt_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Dev batches: {len(dev_loader)}")
    
    return train_loader, dev_loader


# Testing
if __name__ == "__main__":
    from data_loader import CodeQADataLoader
    from vocabulary import Vocabulary
    
    print("=" * 80)
    print("Testing CodeQA Dataset and DataLoader")
    print("=" * 80)
    
    # Load data
    DATA_DIR = "c:/Users/namlh/Code_QA_Project/data"
    print("\nStep 1: Loading data...")
    loader = CodeQADataLoader(DATA_DIR, language='python')
    train_data = loader.load_split('train')[:1000]  # Use subset for testing
    dev_data = loader.load_split('dev')[:100]
    
    # Build or load vocabulary
    print("\nStep 2: Building vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.build_from_examples(train_data)
    
    # Create datasets
    print("\nStep 3: Creating datasets...")
    train_dataset = CodeQADataset(train_data, vocab, max_src_len=256, max_tgt_len=30)
    
    # Test single example
    print("\n" + "=" * 80)
    print("Testing Single Example")
    print("=" * 80)
    example = train_dataset[0]
    print(f"Source shape: {example['src_tokens'].shape}")
    print(f"Source length: {example['src_len']}")
    print(f"Target shape: {example['tgt_tokens'].shape}")
    print(f"Target length: {example['tgt_len']}")
    print(f"Labels shape: {example['labels'].shape}")
    
    # Decode and show
    src_decoded = vocab.decode(example['src_tokens'].tolist(), skip_special=False)
    tgt_decoded = vocab.decode(example['tgt_tokens'].tolist(), skip_special=False)
    print(f"\nSource tokens (first 20): {src_decoded[:20]}")
    print(f"Target tokens: {tgt_decoded}")
    
    # Create dataloaders
    print("\n" + "=" * 80)
    print("Testing DataLoader with Batching")
    print("=" * 80)
    train_loader, dev_loader = create_dataloaders(
        train_data, dev_data, vocab, 
        batch_size=4, 
        max_src_len=256, 
        max_tgt_len=30
    )
    
    # Test batch
    print("\nFetching first batch...")
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  src_tokens shape: {batch['src_tokens'].shape}")
    print(f"  src_lengths: {batch['src_lengths']}")
    print(f"  src_mask shape: {batch['src_mask'].shape}")
    print(f"  tgt_tokens shape: {batch['tgt_tokens'].shape}")
    print(f"  tgt_lengths: {batch['tgt_lengths']}")
    print(f"  labels shape: {batch['labels'].shape}")
    
    # Show first example from batch
    print(f"\n" + "=" * 80)
    print("First Example in Batch (Decoded)")
    print("=" * 80)
    first_src = batch['src_tokens'][0]
    first_tgt = batch['tgt_tokens'][0]
    first_label = batch['labels'][0]
    
    src_text = vocab.decode(first_src[first_src != 0].tolist(), skip_special=False)
    tgt_text = vocab.decode(first_tgt[first_tgt != 0].tolist(), skip_special=False)
    label_text = vocab.decode(first_label[first_label != 0].tolist(), skip_special=False)
    
    print(f"Source (truncated): {' '.join(src_text[:30])}...")
    print(f"Target: {' '.join(tgt_text)}")
    print(f"Labels: {' '.join(label_text)}")
    
    print("\n" + "=" * 80)
    print("Dataset test complete! ✓")
    print("=" * 80)
