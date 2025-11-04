"""
Quick script to check our current vocabulary statistics
"""
import pickle
import sys
sys.path.append('.')

# Load Python vocabulary
with open('saved_models/vocab_python.pkl', 'rb') as f:
    vocab = pickle.load(f)

print("=" * 80)
print("CURRENT VOCABULARY STATISTICS")
print("=" * 80)
print(f"Vocabulary size: {len(vocab.word2idx):,}")
print(f"Min frequency threshold: {vocab.min_freq}")
print(f"Total unique words seen: {len(vocab.word_freq):,}")
print(f"Coverage: {len(vocab.word2idx) / len(vocab.word_freq) * 100:.2f}%")

# Check some statistics
print(f"\nSpecial tokens: {list(vocab.word2idx.keys())[:6]}")
print(f"Sample words: {list(vocab.word2idx.keys())[6:26]}")

# Check frequency distribution
freq_values = list(vocab.word_freq.values())
freq_values.sort(reverse=True)
print(f"\nTop 10 most frequent words:")
for word, freq in sorted(vocab.word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  '{word}': {freq:,}")

print(f"\nWords appearing exactly once: {sum(1 for f in vocab.word_freq.values() if f == 1):,}")
print(f"Words appearing 2+ times (in vocab): {sum(1 for f in vocab.word_freq.values() if f >= 2):,}")
