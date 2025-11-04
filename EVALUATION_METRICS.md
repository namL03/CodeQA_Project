# Evaluation Metrics

This document explains the evaluation metrics implemented in the training script.

## Metrics Implemented

### 1. **BLEU Score** (0-100)
- **What it measures**: Quality of generated text compared to reference
- **How it works**: Computes n-gram precision (unigrams, bigrams, trigrams, 4-grams) with brevity penalty
- **Interpretation**:
  - 0-10: Very poor quality
  - 10-20: Poor quality (baseline models often start here)
  - 20-30: Acceptable quality
  - 30-40: Good quality
  - 40+: Very good quality
- **Used in paper**: Standard metric for code generation and summarization tasks

### 2. **Exact Match** (0-100%)
- **What it measures**: Percentage of predictions that exactly match the reference
- **How it works**: Compares cleaned predictions with references (ignoring special tokens and case)
- **Interpretation**:
  - 0-5%: Model struggling to learn
  - 5-15%: Model learning patterns
  - 15-25%: Good performance
  - 25%+: Excellent performance
- **Important**: Strict metric - only counts perfect matches

### 3. **Loss** (lower is better)
- **What it measures**: Cross-entropy loss on validation set
- **Interpretation**:
  - High loss (>5): Model not learning well
  - Medium loss (3-5): Model learning
  - Low loss (<3): Model fitting well
- **Note**: Loss alone doesn't tell you if answers are meaningful

## Training Output Example

```
Epoch 1/20: Train Loss: 4.23 | Val Loss: 3.88 | BLEU: 12.5 | Exact Match: 3.2%
Epoch 2/20: Train Loss: 3.85 | Val Loss: 3.52 | BLEU: 15.8 | Exact Match: 5.1%
...
Epoch 10/20: Train Loss: 2.41 | Val Loss: 2.89 | BLEU: 24.3 | Exact Match: 12.7%
```

## What to Watch For

### Good Signs ✅
- BLEU score increasing over epochs
- Exact Match percentage increasing
- Loss decreasing
- All three metrics improving together

### Warning Signs ⚠️
- Loss decreasing but BLEU not improving → overfitting or memorization
- BLEU very low (<5) after many epochs → model not learning
- Exact Match stuck at 0% → model not generating valid answers

## Checkpoints Saved With Metrics

Each checkpoint now includes:
- `epoch`: Training epoch number
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `val_bleu`: Validation BLEU score
- `val_exact_match`: Validation Exact Match percentage

## Next Steps

After training with these basic metrics, you can:
1. Implement full evaluation script with:
   - ROUGE-L (for longer sequences)
   - F1 score (token-level precision/recall)
   - Per-example analysis
2. Hyperparameter tuning based on metrics
3. Error analysis on low-BLEU examples

## References

- BLEU: "BLEU: a Method for Automatic Evaluation of Machine Translation" (Papineni et al., 2002)
- Used in: "Get To The Point: Summarization with Pointer-Generator Networks" (See et al., 2017)
- Used in: "CodeQA: A Question Answering Dataset for Source Code Comprehension" (Liu et al., 2021)
