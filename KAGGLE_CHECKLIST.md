# 📋 Kaggle Training Checklist

Print this or keep it open while setting up Kaggle!

---

## ✅ Pre-Upload Checklist

- [x] Ran `python prepare_kaggle_upload.py`
- [x] Found `kaggle_upload/codeqa_code.zip` (27 KB)
- [x] Found `kaggle_upload/codeqa_data.zip` (6.5 MB)
- [x] Read `KAGGLE_QUICKSTART.md`

---

## 📤 Upload Checklist (2 minutes)

### Upload Dataset 1: Code
- [ ] Go to: https://www.kaggle.com/datasets
- [ ] Click "New Dataset"
- [ ] Drag & drop: `codeqa_code.zip`
- [ ] Title: `codeqa-code`
- [ ] Click "Create"
- [ ] Wait for processing (~30 seconds)
- [ ] ✅ Dataset 1 ready!

### Upload Dataset 2: Data
- [ ] Click "New Dataset" again
- [ ] Drag & drop: `codeqa_data.zip`
- [ ] Title: `codeqa-python-dataset`
- [ ] Click "Create"
- [ ] Wait for processing (~1 minute)
- [ ] ✅ Dataset 2 ready!

---

## 📓 Notebook Setup (2 minutes)

### Create Notebook
- [ ] Go to: https://www.kaggle.com/code
- [ ] Click "New Notebook"
- [ ] ✅ Notebook created!

### Configure Settings (RIGHT PANEL)
- [ ] Click ⚙️ Settings
- [ ] Accelerator → Select **GPU T4 x2** ✅
- [ ] Internet → Turn **ON** ✅
- [ ] Click "Save" or close settings

### Add Datasets (RIGHT PANEL)
- [ ] Click "+ Add Data"
- [ ] Search: `codeqa-code`
- [ ] Click "Add" ✅
- [ ] Click "+ Add Data" again
- [ ] Search: `codeqa-python-dataset`
- [ ] Click "Add" ✅

---

## 📝 Copy Notebook Code (1 minute)

### Prepare Code
- [ ] Open `kaggle_train_notebook.py` in your editor
- [ ] Select All (Ctrl+A)
- [ ] Copy (Ctrl+C)

### Paste into Kaggle
- [ ] In Kaggle notebook, delete default cell
- [ ] Paste code (Ctrl+V)
- [ ] Split into cells by "# CELL N:" comments
- [ ] Or just paste as one big cell (works too!)

---

## 🧪 Test Run (2 minutes)

### Run Test Cell
- [ ] Find "CELL 6: Quick Test"
- [ ] Click ▶️ Run Cell
- [ ] Watch output for errors
- [ ] Should complete in ~2 minutes
- [ ] ✅ If no errors, proceed!

---

## 🚀 Full Training (2-3 hours)

### Start Training
- [ ] Find "CELL 7: Full Training"
- [ ] Click ▶️ Run Cell
- [ ] Or click "Run All" at top

### Monitor Progress
- [ ] Check output shows GPU T4 detected
- [ ] Watch epoch progress bars
- [ ] Monitor GPU usage (right panel): 80-100% ✅
- [ ] Check metrics:
  - [ ] Train Loss decreasing
  - [ ] Val Loss decreasing
  - [ ] BLEU increasing (if computed)

### Expected Timeline
- [ ] Epoch 1: ~5-8 minutes
- [ ] Epoch 10: Should see BLEU ~15-25
- [ ] Epoch 20: Should see BLEU ~25-35
- [ ] Total time: ~2-3 hours

---

## 📊 During Training (Check Every 30 mins)

- [ ] Training still running (no errors)
- [ ] GPU usage: 80-100%
- [ ] Val Loss decreasing
- [ ] Session time remaining: >30 mins
  - If <30 mins, download checkpoints!

---

## 💾 After Training

### Download Model
- [ ] Go to "Output" tab (top right)
- [ ] Find `saved_models/` folder
- [ ] Download `best_model_python.pt` (26 MB)
- [ ] Download `training_curves.png` (85 KB)
- [ ] (Optional) Download all checkpoints

### Check Results
- [ ] BLEU Score: 20-30+ ✅
- [ ] Exact Match: 10-20% ✅
- [ ] Val Loss: 2.0-3.0 ✅

---

## 🎉 Success Criteria

Training is successful if:
- [x] Completed 20 epochs (or early stopped)
- [x] BLEU score: 20+
- [x] Val Loss: <3.5
- [x] Downloaded best_model_python.pt
- [x] No errors in training log

---

## 🐛 Troubleshooting

### Problem: "CUDA out of memory"
- [ ] Go to Cell 4
- [ ] Change: `config['batch_size'] = 16`
- [ ] Restart kernel & run again

### Problem: "Dataset not found"
- [ ] Check Cell 3 paths
- [ ] Update to match your dataset names:
  ```python
  dataset_path = '/kaggle/input/YOUR-CODE-DATASET'
  data_path = '/kaggle/input/YOUR-DATA-DATASET/data'
  ```

### Problem: "No GPU detected"
- [ ] Check Settings → Accelerator → GPU T4 x2
- [ ] Turn OFF then ON again
- [ ] Save settings
- [ ] Restart notebook

### Problem: "BLEU = 0 after many epochs"
- [ ] Check Cell 3 output: data files found?
- [ ] Check Cell 3 output: vocabulary found?
- [ ] Verify dataset uploaded correctly

### Problem: "Session timed out"
- [ ] Download checkpoints from Output tab
- [ ] Create new notebook
- [ ] Add `--resume` flag in training command:
  ```python
  --resume /kaggle/input/checkpoints/checkpoint_python_epoch10.pt
  ```

---

## 📱 Quick Reference

### Important URLs
```
Datasets:  https://www.kaggle.com/datasets
Notebooks: https://www.kaggle.com/code
Settings:  https://www.kaggle.com/settings
Quota:     Check "GPU" hours remaining
```

### Key Settings
```
GPU:        GPU T4 x2
Internet:   ON
Batch Size: 32 (reduce to 16 if OOM)
Epochs:     20
```

### Expected Metrics
```
Epoch 1:  Loss ~4.0, BLEU ~0-5
Epoch 5:  Loss ~3.0, BLEU ~10-15
Epoch 10: Loss ~2.5, BLEU ~20-25
Epoch 20: Loss ~2.2, BLEU ~25-35
```

---

## 📞 Help Resources

If stuck:
1. [ ] Read `KAGGLE_QUICKSTART.md`
2. [ ] Check `KAGGLE_SETUP.md` (detailed)
3. [ ] Review notebook comments
4. [ ] Check Kaggle discussion forums
5. [ ] Review training logs for errors

---

## 🎯 Final Checklist

Before closing:
- [ ] Training completed successfully
- [ ] Downloaded best_model_python.pt
- [ ] Downloaded training_curves.png
- [ ] Checked BLEU score (20+)
- [ ] Noted final metrics
- [ ] Saved/copied training logs

---

## ✨ Completion

```
✅ Model trained on Kaggle GPU
✅ BLEU Score: ____ (fill in)
✅ Training Time: ____ hours
✅ Model downloaded
✅ Ready for evaluation/inference

🎉 Congratulations!
```

---

**Start here**: Open `KAGGLE_QUICKSTART.md` and begin! 🚀

Date started: ___________
Date completed: ___________
Total time: ___________ (should be ~2-3 hours)
