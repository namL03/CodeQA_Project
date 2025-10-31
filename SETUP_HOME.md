# Quick Start Guide - Home Laptop Setup

## ✅ What You Need at Home
- Python 3.8+
- Your CodeQA dataset (already have it)
- Git installed

## 🚀 First-Time Setup (Do Once)

### 1. Clone the Repository
```bash
# After you push from work and create GitHub repo
git clone https://github.com/YOUR_USERNAME/CodeQA_Project.git
cd CodeQA_Project
```

### 2. Create Virtual Environment
```bash
# Windows PowerShell:
python -m venv codeqa_venv
codeqa_venv\Scripts\activate

# Mac/Linux:
python3 -m venv codeqa_venv
source codeqa_venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Copy Your Data
Make sure your data is at: `Code_QA_Project/data/`
```
data/
├── python/
│   ├── train/ (train.question, train.code, train.answer)
│   ├── dev/
│   └── test/
└── java/
    ├── train/
    ├── dev/
    └── test/
```

### 5. Run Setup Script
```bash
python setup.py
```

This will:
- Check your data structure
- Create necessary directories
- Build vocabularies (vocab_python.pkl, vocab_java.pkl)

**Done! 🎉**

---

## 🔄 Daily Workflow

### At Work (End of Day)
```bash
git add .
git commit -m "Describe what you did"
git push
```

### At Home (Start of Work)
```bash
git pull
```

### At Home (End of Work)
```bash
git add .
git commit -m "Describe what you did"
git push
```

### At Work (Next Day)
```bash
git pull
```

---

## 📝 Quick Commands

```bash
# Activate environment
codeqa_venv\Scripts\activate          # Windows
source codeqa_venv/bin/activate       # Mac/Linux

# Rebuild vocabularies (if needed)
python scripts/build_vocabulary.py

# Check git status
git status

# See what changed
git diff

# View commit history
git log --oneline
```

---

## ⚠️ Important Notes

1. **Data is NOT in git** - Make sure you have it locally
2. **Vocabularies are NOT in git** - Run `setup.py` to rebuild them
3. **Always pull before starting work** - Prevents merge conflicts
4. **Commit often** - Small commits are better than big ones

---

## 🆘 Troubleshooting

### "No module named 'src'"
```bash
# Make sure you're in the project root directory
cd Code_QA_Project
```

### "Data directory not found"
```bash
# Check if data is in the right location
ls data/python/train
```

### "Merge conflict"
```bash
# This happens if you forgot to pull
# Fix conflicts in the file, then:
git add .
git commit -m "Resolved conflicts"
git push
```

---

## 🔗 Next Steps

After setup is complete:
1. Continue building the Seq2seq model
2. Implement training script
3. Run experiments

**Happy coding! 🚀**
