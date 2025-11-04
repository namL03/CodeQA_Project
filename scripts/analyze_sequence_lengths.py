"""
Analyze sequence lengths in the CodeQA dataset to validate max_src_len and max_tgt_len.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import CodeQADataLoader
from src.vocabulary import Vocabulary
import argparse


def analyze_lengths(examples, vocab, split_name):
    """Analyze sequence lengths for a dataset split."""
    src_lengths = []
    tgt_lengths = []
    
    for ex in examples:
        # Tokenize source (question + code)
        question_tokens = ex['question'].split()
        code_tokens = ex['code'].split()
        src_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + code_tokens
        src_length = len(src_tokens)
        
        # Tokenize target (answer)
        answer_tokens = ex['answer'].split()
        tgt_length = len(answer_tokens)
        
        src_lengths.append(src_length)
        tgt_lengths.append(tgt_length)
    
    # Statistics
    src_lengths.sort()
    tgt_lengths.sort()
    
    n = len(src_lengths)
    
    print(f"\n{'='*80}")
    print(f"{split_name.upper()} SET - {n} examples")
    print(f"{'='*80}")
    
    print(f"\n📊 SOURCE (Question + Code) Length Statistics:")
    print(f"   Min:     {min(src_lengths):,} tokens")
    print(f"   Max:     {max(src_lengths):,} tokens")
    print(f"   Mean:    {sum(src_lengths)/n:.1f} tokens")
    print(f"   Median:  {src_lengths[n//2]:,} tokens")
    print(f"   95th %:  {src_lengths[int(n*0.95)]:,} tokens")
    print(f"   99th %:  {src_lengths[int(n*0.99)]:,} tokens")
    
    print(f"\n📊 TARGET (Answer) Length Statistics:")
    print(f"   Min:     {min(tgt_lengths):,} tokens")
    print(f"   Max:     {max(tgt_lengths):,} tokens")
    print(f"   Mean:    {sum(tgt_lengths)/n:.1f} tokens")
    print(f"   Median:  {tgt_lengths[n//2]:,} tokens")
    print(f"   95th %:  {tgt_lengths[int(n*0.95)]:,} tokens")
    print(f"   99th %:  {tgt_lengths[int(n*0.99)]:,} tokens")
    
    # Coverage analysis
    src_400_coverage = sum(1 for l in src_lengths if l <= 400) / n * 100
    src_300_coverage = sum(1 for l in src_lengths if l <= 300) / n * 100
    src_500_coverage = sum(1 for l in src_lengths if l <= 500) / n * 100
    
    tgt_50_coverage = sum(1 for l in tgt_lengths if l <= 50) / n * 100
    tgt_30_coverage = sum(1 for l in tgt_lengths if l <= 30) / n * 100
    tgt_100_coverage = sum(1 for l in tgt_lengths if l <= 100) / n * 100
    
    print(f"\n📈 Coverage Analysis:")
    print(f"\n   Source (Question + Code):")
    print(f"      max_src_len = 300: {src_300_coverage:.1f}% of examples")
    print(f"      max_src_len = 400: {src_400_coverage:.1f}% of examples ⭐ (current)")
    print(f"      max_src_len = 500: {src_500_coverage:.1f}% of examples")
    
    print(f"\n   Target (Answer):")
    print(f"      max_tgt_len = 30:  {tgt_30_coverage:.1f}% of examples")
    print(f"      max_tgt_len = 50:  {tgt_50_coverage:.1f}% of examples ⭐ (current)")
    print(f"      max_tgt_len = 100: {tgt_100_coverage:.1f}% of examples")
    
    # Warnings
    if src_400_coverage < 95:
        print(f"\n⚠️  WARNING: max_src_len=400 only covers {src_400_coverage:.1f}% of examples!")
        print(f"   Consider increasing to 500 for {src_500_coverage:.1f}% coverage")
    
    if tgt_50_coverage < 95:
        print(f"\n⚠️  WARNING: max_tgt_len=50 only covers {tgt_50_coverage:.1f}% of examples!")
        print(f"   Consider increasing to 100 for {tgt_100_coverage:.1f}% coverage")
    
    return src_lengths, tgt_lengths


def main():
    parser = argparse.ArgumentParser(description='Analyze sequence lengths in CodeQA dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing the data')
    parser.add_argument('--language', type=str, default='python', choices=['python', 'java'],
                       help='Programming language')
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to vocabulary file')
    
    args = parser.parse_args()
    
    if args.vocab_path is None:
        args.vocab_path = f'saved_models/vocab_{args.language}.pkl'
    
    print("="*80)
    print("CodeQA Sequence Length Analysis")
    print("="*80)
    print(f"Language: {args.language}")
    print(f"Data directory: {args.data_dir}")
    
    # Load vocabulary
    print(f"\nLoading vocabulary from {args.vocab_path}...")
    vocab = Vocabulary.load(args.vocab_path)
    print(f"Vocabulary size: {len(vocab):,}")
    
    # Load data
    print(f"\nLoading data...")
    data_loader = CodeQADataLoader(args.data_dir, language=args.language)
    
    train_examples = data_loader.load_split('train')
    dev_examples = data_loader.load_split('dev')
    
    # Analyze each split
    analyze_lengths(train_examples, vocab, 'train')
    analyze_lengths(dev_examples, vocab, 'dev')
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATION:")
    print("="*80)
    print("Based on the statistics above:")
    print("- If 95%+ examples fit in current limits → Keep defaults")
    print("- If <95% coverage → Consider increasing max_src_len or max_tgt_len")
    print("- Balance between coverage and computational efficiency")
    print("="*80)


if __name__ == "__main__":
    main()
