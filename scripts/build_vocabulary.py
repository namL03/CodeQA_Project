"""
Script to build vocabulary for the Seq2seq CodeQA model.

Following the CodeQA paper methodology:
---------------------------------------
1. Build SEPARATE vocabularies for Python and Java
2. Use TRAINING DATA ONLY (proper ML practice)
3. Filter rare words (min_freq threshold)
4. This matches the original paper's experimental setup

VOCABULARY BUILDING STRATEGY:
------------------------------
We use TRAINING DATA ONLY to build vocabulary (proper ML practice).

Why this matters:
1. ✅ Follows ML principles - no information from test set
2. ✅ Simulates real-world deployment (new/unseen words will appear)
3. ✅ Forces model to handle <UNK> tokens properly
4. ✅ More rigorous evaluation
5. ✅ Matches the CodeQA paper methodology
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vocabulary import Vocabulary
from src.data_loader import CodeQADataLoader
import pickle

def build_vocabulary_for_language(language: str, min_freq: int = 2):
    """
    Build vocabulary from TRAINING SET ONLY for a specific language.
    
    Args:
        language: 'python' or 'java'
        min_freq: Minimum frequency for a word to be included (default=2)
    
    This follows the standard machine learning principle:
    - The model should only learn from training data
    - Unknown words in dev/test will be mapped to <UNK>
    - This simulates real-world scenarios where new words may appear
    """
    print("=" * 70)
    print(f"Building Vocabulary for {language.upper()}")
    print("Using TRAINING SET ONLY (following CodeQA paper methodology)")
    print("=" * 70)
    
    # Load training data
    print(f"\n📚 Loading {language} training data...")
    data_loader = CodeQADataLoader(data_dir='data', language=language)
    train_examples = data_loader.load_split('train')
    print(f"Loaded {len(train_examples)} training examples")
    
    # Build vocabulary from training examples
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_from_examples(train_examples)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Vocabulary Statistics:")
    print("=" * 70)
    print(f"Total vocabulary size: {len(vocab)} tokens")
    print(f"Total unique words seen: {len(vocab.word_freq)}")
    print(f"Words filtered (freq < {min_freq}): {len(vocab.word_freq) - len(vocab) + 6}")
    print(f"Coverage: {len(vocab) / len(vocab.word_freq) * 100:.2f}%")
    
    # Show some sample tokens
    print("\n📝 Sample tokens from vocabulary (non-special):")
    non_special_tokens = [word for word in vocab.word2idx.keys() 
                         if not word.startswith('[')]
    print(f"Sample (20 words): {non_special_tokens[:20]}")
    
    # Save vocabulary
    vocab_path = f'saved_models/vocab_{language}.pkl'
    os.makedirs('saved_models', exist_ok=True)
    
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    print(f"\n✅ Vocabulary saved to: {vocab_path}")
    return vocab

if __name__ == "__main__":
    # Build vocabularies for BOTH Python and Java
    # Following the CodeQA paper: separate models for each language
    
    languages = ['python', 'java']
    min_freq = 2  # Filter words appearing less than 2 times
    
    vocabs = {}
    
    for lang in languages:
        print("\n\n")
        vocabs[lang] = build_vocabulary_for_language(lang, min_freq=min_freq)
    
    # Test encoding/decoding with Python vocabulary
    print("\n\n" + "=" * 70)
    print("Testing Encoding/Decoding (Python vocabulary):")
    print("=" * 70)
    
    vocab = vocabs['python']
    
    test_sentence = "What does this function do ?"
    test_tokens = test_sentence.lower().split()
    
    # Encode
    indices = vocab.encode(test_tokens, add_eos=True)
    
    print(f"\nOriginal: {test_sentence}")
    print(f"Tokens: {test_tokens}")
    print(f"Encoded: {indices}")
    print(f"Decoded: {' '.join(vocab.decode(indices))}")
    
    # Test with code
    test_code = "def add ( a , b ) :"
    code_tokens = test_code.split()
    code_indices = vocab.encode(code_tokens)
    
    print(f"\nCode: {test_code}")
    print(f"Tokens: {code_tokens}")
    print(f"Encoded: {code_indices}")
    print(f"Decoded: {' '.join(vocab.decode(code_indices))}")
    
    print("\n" + "=" * 70)
    print("✅ Vocabularies for both Python and Java have been built!")
    print("   - saved_models/vocab_python.pkl")
    print("   - saved_models/vocab_java.pkl")
    print("=" * 70)
