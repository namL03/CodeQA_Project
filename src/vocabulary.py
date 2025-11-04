"""
Vocabulary Builder for CodeQA Dataset

This module creates word-to-index and index-to-word mappings
from the training data. This is essential for converting text
into numerical representations that the neural network can process.
"""

from typing import List, Dict, Set
from collections import Counter
import pickle
import os


class Vocabulary:
    """
    Builds and manages vocabulary for the CodeQA task.
    
    Special tokens:
    - [PAD]: Padding token (for making sequences the same length)
    - [UNK]: Unknown token (for words not in vocabulary)
    - [CLS]: Classification token (marks start of input)
    - [SEP]: Separator token (separates question from code)
    - [SOS]: Start of sequence (for decoder)
    - [EOS]: End of sequence (marks end of answer)
    """
    
    # Define special tokens
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'
    CLS_TOKEN = '[CLS]'
    SEP_TOKEN = '[SEP]'
    SOS_TOKEN = '[SOS]'
    EOS_TOKEN = '[EOS]'
    
    def __init__(self, min_freq: int = 2):
        """
        Initialize the vocabulary.
        
        Args:
            min_freq: Minimum frequency for a word to be included in vocabulary.
                     Words appearing less than this will be mapped to [UNK].
        """
        self.min_freq = min_freq
        
        # Word to index mapping
        self.word2idx = {}
        
        # Index to word mapping
        self.idx2word = {}
        
        # Word frequencies
        self.word_freq = Counter()
        
        # Initialize with special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Add special tokens to the vocabulary."""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN
        ]
        
        for token in special_tokens:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    def build_from_examples(self, examples: List[Dict[str, str]]):
        """
        Build vocabulary from a list of examples.
        
        Args:
            examples: List of dictionaries with 'question', 'code', and 'answer' keys
        """
        print(f"Building vocabulary from {len(examples)} examples...")
        
        # Count word frequencies
        for example in examples:
            # Count words in questions (convert to lowercase)
            question_tokens = example['question'].lower().split()
            self.word_freq.update(question_tokens)
            
            # Count words in code (keep original case for code)
            code_tokens = example['code'].split()
            self.word_freq.update(code_tokens)
            
            # Count words in answers (convert to lowercase)
            answer_tokens = example['answer'].lower().split()
            self.word_freq.update(answer_tokens)
        
        # Add words that meet minimum frequency requirement
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Vocabulary built with {len(self.word2idx)} unique tokens")
        print(f"Total words seen: {len(self.word_freq)}")
        print(f"Words filtered out (freq < {self.min_freq}): {len(self.word_freq) - len(self.word2idx) + 6}")
    
    def encode(self, tokens: List[str], add_eos: bool = False) -> List[int]:
        """
        Convert a list of tokens to a list of indices.
        
        Args:
            tokens: List of word tokens
            add_eos: Whether to add [EOS] token at the end
            
        Returns:
            List of integer indices
        """
        indices = []
        for token in tokens:
            # Look up the token in vocabulary, use [UNK] if not found
            idx = self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
            indices.append(idx)
        
        if add_eos:
            indices.append(self.word2idx[self.EOS_TOKEN])
        
        return indices
    
    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """
        Convert a list of indices back to tokens.
        
        Args:
            indices: List of integer indices
            skip_special: Whether to skip special tokens in output
            
        Returns:
            List of word tokens
        """
        special_tokens = {
            self.word2idx[self.PAD_TOKEN],
            self.word2idx[self.SOS_TOKEN],
            self.word2idx[self.EOS_TOKEN]
        }
        
        tokens = []
        for idx in indices:
            # Skip special tokens if requested
            if skip_special and idx in special_tokens:
                continue
            
            # Convert index to word
            token = self.idx2word.get(idx, self.UNK_TOKEN)
            tokens.append(token)
        
        return tokens
    
    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """
        Save vocabulary to a file.
        
        Args:
            filepath: Path to save the vocabulary
        """

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Vocabulary saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load vocabulary from a file.
        
        Args:
            filepath: Path to load the vocabulary from
            
        Returns:
            Vocabulary object
        """
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        # Handle both old format (Vocabulary object) and new format (dict)
        if isinstance(vocab_data, cls):
            # Old format: directly loaded Vocabulary object
            return vocab_data
        else:
            # New format: dictionary with vocab data
            vocab = cls(min_freq=vocab_data['min_freq'])
            vocab.word2idx = vocab_data['word2idx']
            vocab.idx2word = vocab_data['idx2word']
            vocab.word_freq = vocab_data['word_freq']
        
        print(f"Vocabulary loaded from {filepath}")
        print(f"Vocabulary size: {len(vocab)}")
        
        return vocab