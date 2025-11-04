# Training Quick Start Guide

## Prerequisites

1. ✅ Virtual environment activated
2. ✅ Dependencies installed (`requirements.txt`)
3. ✅ Data in `data/` folder
4. ✅ Vocabularies built (`scripts/build_vocabulary.py`)

## Quick Start - Train Python Model

```bash
# Basic training (uses default settings from config.yaml)
python scripts/train.py --language python

# Training with custom batch size (if GPU memory is limited)
python scripts/train.py --language python --batch_size 8

# Training for more epochs
python scripts/train.py --language python --epochs 30

# Training on CPU (if no GPU)
python scripts/train.py --language python --device cpu
```

## Quick Start - Train Java Model

```bash
python scripts/train.py --language java
```

## Monitoring Training

The script will show:
- Progress bar for each epoch
- Training loss (real-time)
- Validation loss (after each epoch)
- Time per epoch
- Best model indicator (✅)

Example output:
```
Epoch 1/20
Training: 100%|████████| 3505/3505 [05:23<00:00, loss=4.2341]
Evaluating: 100%|████████| 439/439 [00:32<00:00, loss=3.8765]

Epoch 1 completed in 355.67s
Train Loss: 4.2341
Val Loss: 3.8765
✅ New best model saved! Val Loss: 3.8765
```

## Output Files

Training creates:
```
saved_models/
├── checkpoint_python_epoch1.pt
├── checkpoint_python_epoch2.pt
├── ...
└── best_model_python.pt  ← Best model based on validation loss
```

## Resume Training

If training is interrupted, resume from checkpoint:
```bash
python scripts/train.py --language python --resume saved_models/checkpoint_python_epoch10.pt
```

## Advanced Options

### Adjust Hyperparameters

```bash
# Lower learning rate
python scripts/train.py --language python --lr 0.1

# Increase gradient clipping
python scripts/train.py --language python --grad_clip 5.0

# Reduce teacher forcing (more challenging)
python scripts/train.py --language python --teacher_forcing_ratio 0.5
```

### Adjust Sequence Lengths

```bash
# Longer source sequences
python scripts/train.py --language python --max_src_len 500

# Longer answer sequences
python scripts/train.py --language python --max_tgt_len 100
```

### Disable Copy Mechanism

```bash
# Train without copy mechanism (for comparison)
python scripts/train.py --language python --use_copy false
```

## Expected Training Time

**With GPU (NVIDIA RTX 3080):**
- Python (~56K examples): ~6-8 minutes per epoch
- Java (~96K examples): ~10-12 minutes per epoch

**With CPU:**
- Much slower, expect 10-20x longer

## Tips

1. **Start with small number of epochs** to test: `--epochs 2`
2. **Monitor GPU memory**: Reduce `--batch_size` if out of memory
3. **Validation loss should decrease**: If not, adjust learning rate
4. **Save checkpoints frequently**: Default saves every epoch
5. **Best model is automatically saved**: Based on validation loss

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python scripts/train.py --language python --batch_size 4

# Reduce sequence lengths
python scripts/train.py --language python --max_src_len 300 --max_tgt_len 30
```

### Slow Training
```bash
# Reduce number of workers
python scripts/train.py --language python --num_workers 0

# Use smaller dataset (for testing)
# Edit data_loader.py to load subset
```

### Loss Not Decreasing
```bash
# Try lower learning rate
python scripts/train.py --language python --lr 0.01

# Try smaller gradient clipping
python scripts/train.py --language python --grad_clip 1.0
```

## Next Steps

After training:
1. Evaluate model: `python scripts/evaluate.py`
2. Generate predictions: `python scripts/inference.py`
3. Analyze results: `python scripts/analyze_results.py`

## Configuration Details

All default settings match the "Get To The Point" paper:
- **Optimizer**: Adagrad
- **Learning Rate**: 0.15
- **Gradient Clipping**: 2.0
- **Architecture**: Single-layer Bi-LSTM encoder, single-layer LSTM decoder
- **Embedding**: 128 dimensions
- **Hidden**: 256 dimensions
- **Attention**: Bahdanau (additive)
- **Copy Mechanism**: Pointer-Generator

See `config.yaml` for full configuration.
