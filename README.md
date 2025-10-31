# CodeQA: Seq2seq Model for Source Code Comprehension

A PyTorch implementation of a Seq2seq model with attention and copy mechanism for answering natural language questions about source code.

## 📋 Project Overview

This project implements a baseline Seq2seq model following the methodology from the paper:
**"CodeQA: A Question Answering Dataset for Source Code Comprehension"**

### Task
Given a code snippet and a free-form question about that code, generate an answer to the question.

**Example:**
```
Question: What does this function do?
Code: def add ( a , b ) : return a + b
Answer: adds two numbers
```

## 🏗️ Project Structure

```
Code_QA_Project/
├── data/                          # Dataset (not in git - setup locally)
│   ├── python/
│   │   ├── train/
│   │   │   ├── train.question
│   │   │   ├── train.code
│   │   │   └── train.answer
│   │   ├── dev/
│   │   └── test/
│   └── java/
│       ├── train/
│       ├── dev/
│       └── test/
├── src/                           # Core library code
│   ├── __init__.py
│   ├── data_loader.py            # Dataset loading
│   ├── vocabulary.py             # Vocabulary builder
│   └── dataset.py                # PyTorch Dataset (TODO)
├── scripts/                       # Executable scripts
│   └── build_vocabulary.py       # Build vocab from training data
├── saved_models/                  # Trained models & vocabularies
│   ├── vocab_python.pkl          # (generated, not in git)
│   └── vocab_java.pkl            # (generated, not in git)
├── notebooks/                     # Jupyter notebooks for exploration
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Setup Instructions

### On Your Company Computer (Already Done ✅)
You've already set up the project and built vocabularies.

### On Your Home Laptop (First Time Setup)

#### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/CodeQA_Project.git
cd CodeQA_Project
```

#### 2. Set up Python environment
```bash
# Create virtual environment
python -m venv codeqa_venv

# Activate it
# On Windows:
codeqa_venv\Scripts\activate
# On Mac/Linux:
source codeqa_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Set up data
Since you already have the data at home, just make sure it's in the correct location:
```
data/
├── python/
│   ├── train/
│   ├── dev/
│   └── test/
└── java/
    ├── train/
    ├── dev/
    └── test/
```

#### 4. Build vocabularies
```bash
python scripts/build_vocabulary.py
```

This will create:
- `saved_models/vocab_python.pkl`
- `saved_models/vocab_java.pkl`

## 📝 Model Architecture (TODO)

Following the Seq2seq baseline from the CodeQA paper:

- **Encoder**: Bi-LSTM to read `[CLS] Question [SEP] Code`
- **Attention Mechanism**: Allows decoder to focus on relevant parts
- **Copy Mechanism**: Enables copying words from input
- **Decoder**: LSTM to generate the answer

## 🔧 Usage

### Build Vocabulary (Preprocessing)
```bash
python scripts/build_vocabulary.py
```

### Train Model (TODO)
```bash
python scripts/train.py --language python --epochs 20
```

### Evaluate Model (TODO)
```bash
python scripts/evaluate.py --model saved_models/model_python.pt
```

## 📊 Dataset Statistics

**Python:**
- Training: 56,085 examples
- Vocabulary: 79,071 tokens

**Java:**
- Training: 95,778 examples
- Vocabulary: 32,908 tokens

## 🎯 Key Design Decisions

1. **Separate models for Python and Java** - Following the original paper
2. **Training data only for vocabulary** - Proper ML practice, no test leakage
3. **Min frequency threshold = 2** - Filter rare words to reduce vocabulary size
4. **Tokenized code** - Using code tokens (not AST) as per paper findings

## 📚 References

- CodeQA Paper: "CodeQA: A Question Answering Dataset for Source Code Comprehension"
- Seq2seq: Sutskever et al., 2014
- Copy Mechanism: "Get To The Point" (See et al., 2017)

## 🔄 Git Workflow

### At work (after making changes):
```bash
git add .
git commit -m "Description of changes"
git push
```

### At home (before starting work):
```bash
git pull
```

## ⚠️ Important Notes

- The `data/` folder is NOT in git (too large)
- Vocabulary `.pkl` files are NOT in git (regenerate locally)
- Always rebuild vocabularies after cloning: `python scripts/build_vocabulary.py`

## 📧 Contact

Your Name - Your Email

## 📄 License

MIT License (or your choice)
