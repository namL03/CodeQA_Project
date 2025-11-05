# Kaggle Upload Package

This folder contains everything you need to train your CodeQA model on Kaggle.

## 📦 Contents

### 1. `codeqa_code.zip` (27 KB)
Contains your source code:
- `src/` - All model files (encoder, decoder, attention, etc.)
- `scripts/train.py` - Training script
- `config.yaml` - Configuration
- `requirements.txt` - Dependencies

### 2. `codeqa_data.zip` (6.5 MB)
Contains your data:
- `data/python/train/` - Training data (56,085 examples)
- `data/python/dev/` - Validation data (7,000 examples)
- `saved_models/vocab_python.pkl` - Vocabulary (79,071 tokens)

## 🚀 Upload Instructions

### Quick Method (5 minutes):

1. **Go to Kaggle Datasets**: https://www.kaggle.com/datasets

2. **Upload Code** (Dataset 1):
   - Click "New Dataset"
   - Drag `codeqa_code.zip`
   - Title: `codeqa-code`
   - Click "Create"

3. **Upload Data** (Dataset 2):
   - Click "New Dataset"
   - Drag `codeqa_data.zip`
   - Title: `codeqa-python-dataset`
   - Click "Create"

4. **Create Notebook**: https://www.kaggle.com/code
   - Click "New Notebook"
   - Settings → GPU T4 x2 → ON
   - Settings → Internet → ON
   - Add Data → `codeqa-code`
   - Add Data → `codeqa-python-dataset`

5. **Copy Notebook Code**:
   - Open `../kaggle_train_notebook.py`
   - Copy all code
   - Paste into Kaggle cells
   - Run all!

## 📊 Expected Results

After uploading and training on Kaggle:
- **Training time**: 2-3 hours (vs 20+ hours on CPU)
- **BLEU Score**: 20-30+ (target)
- **Exact Match**: 10-20%
- **Model size**: ~26 MB

## 📚 Documentation

Detailed guides in parent directory:
- `KAGGLE_QUICKSTART.md` - 5-minute setup guide
- `KAGGLE_SETUP.md` - Detailed instructions
- `KAGGLE_WORKFLOW.txt` - Visual workflow
- `kaggle_train_notebook.py` - Notebook code to copy

## ✅ What's Included

All necessary files for training:
- ✅ Source code (all 9 Python files)
- ✅ Training script
- ✅ Configuration
- ✅ Training data (56K examples)
- ✅ Validation data (7K examples)
- ✅ Pre-built vocabulary

## 🎯 Quick Start

```bash
# Already done - files are ready!
# Just upload the 2 zip files to Kaggle
```

Then follow: `../KAGGLE_QUICKSTART.md`

## 💡 Tips

1. **Test first**: Run 1 epoch before full training
2. **Monitor GPU**: Should show 80-100% usage
3. **Save checkpoints**: Auto-saved every epoch
4. **Download results**: From "Output" tab after training
5. **Check BLEU**: Should increase from 0 → 20-30

## 🔗 Quick Links

- **Upload datasets**: https://www.kaggle.com/datasets
- **Create notebook**: https://www.kaggle.com/code
- **Check GPU quota**: https://www.kaggle.com/settings

## ❓ Questions?

See the full documentation in:
- `../KAGGLE_QUICKSTART.md` - Start here!
- `../KAGGLE_SETUP.md` - Detailed guide
- `../KAGGLE_WORKFLOW.txt` - Visual overview

Ready to train? Upload these 2 files to Kaggle! 🚀
