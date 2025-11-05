# Training CodeQA Model on Kaggle

This guide will help you train your Seq2seq model on Kaggle with free GPU access.

## 🚀 Quick Start

### Step 1: Prepare Your Files

You need to upload these files to Kaggle:

#### Required Core Files:
```
src/
  ├── __init__.py
  ├── encoder.py
  ├── decoder.py
  ├── attention.py
  ├── copy_mechanism.py
  ├── seq2seq_model.py
  ├── vocabulary.py
  ├── data_loader.py
  └── dataset.py

scripts/
  └── train.py

config.yaml
requirements.txt
```

#### Required Data Files:
```
data/
  └── python/
      ├── train/
      │   ├── train.answer
      │   ├── train.code
      │   └── train.question
      └── dev/
          ├── dev.answer
          ├── dev.code
          └── dev.question

saved_models/
  └── vocab_python.pkl
```

### Step 2: Create Kaggle Dataset

1. **Go to Kaggle**: https://www.kaggle.com/datasets
2. **Click "New Dataset"**
3. **Upload your files**:
   - Create a zip file with your data and vocabulary:
     ```bash
     # On Windows PowerShell
     Compress-Archive -Path data, saved_models -DestinationPath codeqa_data.zip
     ```
   - Upload `codeqa_data.zip`
4. **Name it**: `codeqa-python-dataset`
5. **Make it public or private** (your choice)

### Step 3: Create Kaggle Notebook

1. **Go to**: https://www.kaggle.com/code
2. **Click "New Notebook"**
3. **Enable GPU**:
   - Click Settings (right panel)
   - Accelerator → Select **GPU T4 x2** (free)
   - Internet → Turn **ON** (to install packages)
4. **Add your dataset**:
   - Click "+ Add Data" (right panel)
   - Search for your dataset `codeqa-python-dataset`
   - Click "Add"

### Step 4: Upload Code to Notebook

Copy the code from `kaggle_train_notebook.py` (see below) into your Kaggle notebook.

## 📊 Kaggle Advantages

### ✅ Benefits:
- **Free GPU**: NVIDIA T4 or P100 (30 hours/week)
- **Faster training**: ~5-10 minutes per epoch vs 65 minutes on CPU
- **Easy sharing**: Share your notebook and results
- **Auto-save**: Progress is saved automatically
- **Visualization**: Built-in plotting tools

### ⚠️ Limitations:
- **Session time**: 12 hours max per session
- **Weekly quota**: 30 GPU hours/week
- **Internet required**: For installing packages
- **Storage**: Need to save checkpoints to Kaggle Datasets

## 📝 Training on Kaggle

### Recommended Settings for Kaggle GPU:

```python
# Increase batch size for GPU (more memory available)
batch_size = 32  # vs 16 on CPU

# Compute metrics less frequently to save time
compute_metrics_every = 2  # Every 2 epochs instead of every epoch

# More workers for data loading
num_workers = 2
```

### Expected Training Time:
- **CPU (your PC)**: ~65 min/epoch → 20 hours for 20 epochs
- **Kaggle GPU T4**: ~5-8 min/epoch → 2-3 hours for 20 epochs
- **Kaggle GPU P100**: ~3-5 min/epoch → 1-2 hours for 20 epochs

## 💾 Saving Checkpoints

Kaggle sessions can disconnect. To save your progress:

### Option 1: Save to Kaggle Output (Automatic)
```python
# Checkpoints are automatically saved to /kaggle/working/
# After notebook runs, download from "Output" tab
```

### Option 2: Save to Kaggle Dataset (Manual)
```python
# Create a new dataset version with your checkpoints
# This allows you to resume training in a new session
```

### Option 3: Mount Google Drive (Advanced)
```python
# Not recommended - Kaggle doesn't support Google Drive mounting
# Use Option 1 or 2 instead
```

## 🔄 Resuming Training

If your session times out:

1. **Download checkpoints** from previous run (Output tab)
2. **Upload to new dataset** or keep in notebook files
3. **Resume training**:
   ```python
   python train.py --resume /kaggle/input/checkpoints/checkpoint_python_epoch5.pt
   ```

## 📈 Monitoring Training

Kaggle provides:
- **Real-time logs**: Watch training progress
- **Resource usage**: GPU/RAM monitoring
- **Versions**: Each run is saved as a version
- **Comments**: Share results with the community

## 🎯 Full Training Example

```python
# Install dependencies
!pip install torch tqdm pyyaml

# Run training with Kaggle-optimized settings
!python scripts/train.py \
    --language python \
    --epochs 20 \
    --batch_size 32 \
    --compute_metrics_every 2 \
    --num_workers 2 \
    --data_dir /kaggle/input/codeqa-python-dataset/data \
    --vocab_path /kaggle/input/codeqa-python-dataset/saved_models/vocab_python.pkl \
    --save_dir /kaggle/working/saved_models
```

## 🐛 Troubleshooting

### Problem: "CUDA out of memory"
**Solution**: Reduce batch size
```python
--batch_size 16  # or even 8
```

### Problem: "Session timed out"
**Solution**: 
- Save checkpoints more frequently (`--save_every 1`)
- Download checkpoints from Output tab
- Resume in new session with `--resume`

### Problem: "Module not found"
**Solution**: Install missing packages
```python
!pip install torch tqdm pyyaml
```

### Problem: "Data not found"
**Solution**: Check data paths
```python
# List files in input directory
!ls /kaggle/input/
!ls /kaggle/input/codeqa-python-dataset/
```

## 📊 After Training

1. **Download best model**:
   - Go to "Output" tab
   - Download `best_model_python.pt`

2. **View metrics**:
   - Training logs show BLEU and Exact Match
   - Copy metrics for comparison

3. **Share results**:
   - Make notebook public
   - Share link with others

## 🔗 Useful Links

- **Kaggle Notebooks**: https://www.kaggle.com/code
- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **GPU Quota**: https://www.kaggle.com/settings
- **Kaggle Documentation**: https://www.kaggle.com/docs

## 💡 Tips for Success

1. **Test first**: Run 1-2 epochs to verify everything works
2. **Monitor early**: Watch first epoch closely for errors
3. **Save often**: Use `--save_every 1`
4. **Use metrics wisely**: Set `--compute_metrics_every 2` to save time
5. **Check GPU usage**: Make sure GPU is actually being used
6. **Download checkpoints**: Don't lose your work!

Good luck with training! 🚀
