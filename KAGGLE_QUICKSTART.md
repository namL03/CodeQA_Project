# Kaggle Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Prepare Your Files (2 minutes)

Run this command in your project directory:

```powershell
python prepare_kaggle_upload.py
```

This creates two zip files in `kaggle_upload/`:
- `codeqa_code.zip` - Your source code
- `codeqa_data.zip` - Your data and vocabulary

### Step 2: Upload to Kaggle (2 minutes)

#### Upload Code Dataset:
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Drag and drop `codeqa_code.zip`
4. Title: **codeqa-code**
5. Click "Create"

#### Upload Data Dataset:
1. Click "New Dataset" again
2. Drag and drop `codeqa_data.zip`
3. Title: **codeqa-python-dataset**
4. Click "Create"

### Step 3: Create & Run Notebook (1 minute)

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. **Enable GPU**:
   - Settings (right panel) → Accelerator → **GPU T4 x2**
   - Internet → **ON**
4. **Add your datasets**:
   - Click "+ Add Data" → Search "codeqa-code" → Add
   - Click "+ Add Data" → Search "codeqa-python-dataset" → Add
5. **Copy notebook code**:
   - Open `kaggle_train_notebook.py` in your project
   - Copy ALL the code
   - Paste into Kaggle notebook cells (split by "# CELL N:" comments)
6. **Run All**!

## ⏰ Training Time

- **CPU (your PC)**: ~20 hours for 20 epochs
- **Kaggle GPU T4**: ~2-3 hours for 20 epochs
- **Kaggle GPU P100**: ~1-2 hours for 20 epochs

## 💾 Download Your Model

After training completes:

1. Click **"Output"** tab (right side)
2. Download `best_model_python.pt`
3. Download `training_curves.png` (visualization)

## ⚙️ Recommended Settings for Kaggle

The notebook uses these optimized settings:

```yaml
batch_size: 32              # Increased from 16 (more GPU memory)
compute_metrics_every: 2    # Every 2 epochs (saves time)
num_workers: 2              # Faster data loading
max_src_len: 256           # Optimized for data
max_tgt_len: 30            # Optimized for data
```

## 🐛 Common Issues

### "CUDA out of memory"
**Fix**: Reduce batch size
```python
config['batch_size'] = 16  # or even 8
```

### "Dataset not found"
**Fix**: Update dataset names in Cell 3:
```python
dataset_path = '/kaggle/input/YOUR-DATASET-NAME'
data_path = '/kaggle/input/YOUR-DATA-NAME/data'
```

### "Session timed out after 6 hours"
**Fix**: Download checkpoints from Output tab, then resume:
```python
# Add to training command:
--resume /kaggle/input/your-checkpoint/checkpoint_python_epoch10.pt
```

## 📊 Monitor Training

Watch for these patterns:

### ✅ Good Signs:
- Train loss: 5.0 → 4.0 → 3.0 → 2.5 (decreasing)
- Val loss: 4.8 → 3.8 → 3.2 → 2.8 (decreasing)
- BLEU: 0 → 10 → 20 → 25+ (increasing)
- GPU usage: 80-100% (check right panel)

### ⚠️ Warning Signs:
- Val loss increasing (overfitting)
- BLEU stuck at 0 after 5 epochs (check data)
- GPU usage <10% (not using GPU!)

## 💡 Pro Tips

1. **Test first**: Run Cell 6 (1 epoch test) before full training
2. **Monitor early**: Watch first epoch closely
3. **Check GPU**: Should see "GPU T4" or "GPU P100" in Cell 1
4. **Save often**: Checkpoints saved every epoch by default
5. **Use fast settings**: `compute_metrics_every: 2` or even `5` for speed

## 📈 Expected Results

After 20 epochs, you should see:

- **BLEU Score**: 20-30 (good), 30-40 (very good)
- **Exact Match**: 10-20% (good), 20%+ (very good)
- **Val Loss**: 2.0-3.0 (good), <2.0 (excellent)

## 🔗 Quick Links

- **Kaggle Notebooks**: https://www.kaggle.com/code
- **Your Datasets**: https://www.kaggle.com/[your-username]/datasets
- **GPU Quota**: https://www.kaggle.com/settings (check remaining hours)

## ❓ Need Help?

1. Check `KAGGLE_SETUP.md` for detailed guide
2. Read notebook comments (each cell is documented)
3. Check Kaggle discussion forums
4. Review training logs for errors

---

**Ready to start?** Run `python prepare_kaggle_upload.py` now! 🚀
