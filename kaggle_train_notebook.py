"""
Kaggle Training Notebook for CodeQA Seq2seq Model
==================================================

This notebook trains a Seq2seq model with attention and copy mechanism
for code question answering, following the "Get To The Point" paper.

Instructions:
1. Upload this notebook to Kaggle
2. Enable GPU (Settings → Accelerator → GPU T4 x2)
3. Add your dataset (codeqa-python-dataset)
4. Run all cells
"""

# ============================================================================
# CELL 1: Setup and Install Dependencies
# ============================================================================

print("=" * 80)
print("CodeQA Seq2seq Training on Kaggle")
print("Following: Get To The Point (See et al., 2017)")
print("=" * 80)

# Install required packages
print("\n📦 Installing dependencies...")
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q tqdm pyyaml

# Check GPU availability
import torch
print(f"\n🔧 PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  WARNING: No GPU detected! Training will be slow.")


# ============================================================================
# CELL 2: Setup File Structure
# ============================================================================

import os
import sys
import shutil

print("\n📁 Setting up file structure...")

# Create necessary directories
os.makedirs('/kaggle/working/src', exist_ok=True)
os.makedirs('/kaggle/working/scripts', exist_ok=True)
os.makedirs('/kaggle/working/saved_models', exist_ok=True)

# Copy source files (assumes you uploaded them to Kaggle dataset)
# Adjust paths based on your dataset structure
dataset_path = '/kaggle/input/codeqa-code'  # Change this to your dataset name

# If you uploaded code as a dataset
if os.path.exists(dataset_path):
    print(f"✅ Found code dataset at: {dataset_path}")
    
    # Copy Python source files
    if os.path.exists(f"{dataset_path}/src"):
        shutil.copytree(f"{dataset_path}/src", '/kaggle/working/src', dirs_exist_ok=True)
        print("✅ Copied src/")
    
    if os.path.exists(f"{dataset_path}/scripts"):
        shutil.copytree(f"{dataset_path}/scripts", '/kaggle/working/scripts', dirs_exist_ok=True)
        print("✅ Copied scripts/")
else:
    print("⚠️  Code dataset not found. You need to upload your source code.")
    print("   Create a dataset with: src/, scripts/, config.yaml")

# Add to Python path
sys.path.append('/kaggle/working')
print(f"✅ Added to Python path: /kaggle/working")

# List available datasets
print("\n📊 Available input datasets:")
!ls /kaggle/input/


# ============================================================================
# CELL 3: Verify Data Files
# ============================================================================

print("\n🔍 Verifying data files...")

# Check for data
data_path = '/kaggle/input/codeqa-python-dataset/data'  # Change this to your data dataset name
vocab_path = '/kaggle/input/codeqa-python-dataset/saved_models/vocab_python.pkl'

if os.path.exists(data_path):
    print(f"✅ Data found at: {data_path}")
    !ls {data_path}/python/train/ | head -n 5
    !ls {data_path}/python/dev/ | head -n 5
else:
    print(f"❌ Data NOT found at: {data_path}")
    print("   You need to upload your data/ directory as a Kaggle dataset")

if os.path.exists(vocab_path):
    print(f"✅ Vocabulary found at: {vocab_path}")
else:
    print(f"❌ Vocabulary NOT found at: {vocab_path}")
    print("   You need to upload saved_models/vocab_python.pkl")


# ============================================================================
# CELL 4: Configuration
# ============================================================================

print("\n⚙️  Training Configuration:")
print("=" * 80)

# Kaggle-optimized settings
config = {
    # Data paths
    'data_dir': '/kaggle/input/codeqa-python-dataset/data',
    'vocab_path': '/kaggle/input/codeqa-python-dataset/saved_models/vocab_python.pkl',
    'save_dir': '/kaggle/working/saved_models',
    'language': 'python',
    
    # Model architecture (following "Get To The Point" paper)
    'embed_dim': 128,
    'hidden_dim': 256,
    'num_layers': 1,
    'dropout': 0.0,
    'use_copy': True,
    
    # Training hyperparameters (following paper)
    'batch_size': 32,  # Increased for GPU
    'epochs': 20,
    'lr': 0.15,
    'grad_clip': 2.0,
    'teacher_forcing_ratio': 1.0,
    
    # Data parameters (optimized)
    'max_src_len': 256,
    'max_tgt_len': 30,
    
    # Beam search (following paper)
    'beam_size': 4,
    
    # Early stopping
    'early_stopping_patience': 5,
    
    # Performance optimization for Kaggle
    'compute_metrics_every': 2,  # Compute BLEU every 2 epochs (saves time)
    'save_every': 1,
    'num_workers': 2,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

for key, value in config.items():
    print(f"  {key:25s}: {value}")

print("=" * 80)


# ============================================================================
# CELL 5: Build Training Command
# ============================================================================

print("\n🏋️  Preparing training command...")

# Build command
cmd = f"""python scripts/train.py \
    --data_dir {config['data_dir']} \
    --vocab_path {config['vocab_path']} \
    --save_dir {config['save_dir']} \
    --language {config['language']} \
    --embed_dim {config['embed_dim']} \
    --hidden_dim {config['hidden_dim']} \
    --num_layers {config['num_layers']} \
    --dropout {config['dropout']} \
    --batch_size {config['batch_size']} \
    --epochs {config['epochs']} \
    --lr {config['lr']} \
    --grad_clip {config['grad_clip']} \
    --teacher_forcing_ratio {config['teacher_forcing_ratio']} \
    --max_src_len {config['max_src_len']} \
    --max_tgt_len {config['max_tgt_len']} \
    --beam_size {config['beam_size']} \
    --early_stopping_patience {config['early_stopping_patience']} \
    --compute_metrics_every {config['compute_metrics_every']} \
    --save_every {config['save_every']} \
    --num_workers {config['num_workers']} \
    --device {config['device']}"""

print("Command to execute:")
print(cmd)


# ============================================================================
# CELL 6: Quick Test (Optional - Run 1 Epoch)
# ============================================================================

print("\n🧪 Quick test with 1 epoch...")
print("=" * 80)

# Test command (1 epoch, fast metrics)
test_cmd = f"""python scripts/train.py \
    --data_dir {config['data_dir']} \
    --vocab_path {config['vocab_path']} \
    --save_dir {config['save_dir']}/test \
    --language {config['language']} \
    --batch_size {config['batch_size']} \
    --epochs 1 \
    --compute_metrics_every 0 \
    --device {config['device']}"""

print("Running 1 epoch test to verify setup...")
!{test_cmd}

print("\n✅ Test completed! If no errors, proceed to full training.")


# ============================================================================
# CELL 7: Full Training
# ============================================================================

print("\n" + "=" * 80)
print("🚀 STARTING FULL TRAINING")
print("=" * 80)
print("\n⏰ Estimated time on GPU T4: ~2-3 hours for 20 epochs")
print("💡 Tip: Monitor GPU usage in the right panel")
print("\n" + "=" * 80 + "\n")

# Run full training
!{cmd}

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETED!")
print("=" * 80)


# ============================================================================
# CELL 8: View Results
# ============================================================================

print("\n📊 Training Results:")
print("=" * 80)

# List saved models
print("\n📁 Saved checkpoints:")
!ls -lh {config['save_dir']}/

# Load and display best model info
import torch
import pickle

best_model_path = f"{config['save_dir']}/best_model_python.pt"
if os.path.exists(best_model_path):
    print(f"\n✅ Best model found: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location='cpu')
    
    print(f"\n📈 Best Model Metrics:")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    if checkpoint.get('val_bleu') is not None:
        print(f"  Val BLEU: {checkpoint['val_bleu']:.2f}")
    if checkpoint.get('val_exact_match') is not None:
        print(f"  Val Exact Match: {checkpoint['val_exact_match']:.2f}%")
else:
    print(f"❌ Best model not found at: {best_model_path}")


# ============================================================================
# CELL 9: Plot Training Curves (Optional)
# ============================================================================

print("\n📉 Plotting training curves...")

import matplotlib.pyplot as plt
import glob

# Try to load all checkpoints
checkpoint_files = sorted(glob.glob(f"{config['save_dir']}/checkpoint_python_epoch*.pt"))

if len(checkpoint_files) > 0:
    epochs = []
    train_losses = []
    val_losses = []
    val_bleus = []
    
    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location='cpu')
        epochs.append(ckpt['epoch'] + 1)
        train_losses.append(ckpt['train_loss'])
        val_losses.append(ckpt['val_loss'])
        if ckpt.get('val_bleu') is not None:
            val_bleus.append(ckpt['val_bleu'])
    
    # Plot losses
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # BLEU plot
    if val_bleus:
        axes[1].plot([epochs[i] for i in range(len(val_bleus))], val_bleus, 'g-o', label='Val BLEU', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('BLEU Score', fontsize=12)
        axes[1].set_title('Validation BLEU Score', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'BLEU scores not computed\n(compute_metrics_every=0)', 
                    ha='center', va='center', fontsize=12)
        axes[1].set_title('BLEU Score', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Training curves saved to: /kaggle/working/training_curves.png")
else:
    print("⚠️  No checkpoint files found for plotting")


# ============================================================================
# CELL 10: Download Instructions
# ============================================================================

print("\n" + "=" * 80)
print("💾 DOWNLOAD YOUR TRAINED MODEL")
print("=" * 80)
print("""
To download your trained model:

1. Click on the 'Output' tab (right side of the screen)
2. Look for 'saved_models/' directory
3. Download these files:
   - best_model_python.pt (your best model)
   - checkpoint_python_epoch*.pt (all checkpoints)
   - training_curves.png (visualization)

4. You can also download the entire /kaggle/working directory

To resume training in a new session:
1. Upload your checkpoint to a new dataset
2. Use --resume flag:
   --resume /kaggle/input/your-checkpoint/checkpoint_python_epoch10.pt

""")

print("=" * 80)
print("🎉 ALL DONE!")
print("=" * 80)
print(f"""
Training Summary:
- Model: Seq2seq with Attention + Copy Mechanism
- Architecture: Following "Get To The Point" (See et al., 2017)
- Dataset: CodeQA Python
- Device: {config['device'].upper()}
- Epochs: {config['epochs']}
- Batch Size: {config['batch_size']}

Check the Output tab to download your trained model!
""")
