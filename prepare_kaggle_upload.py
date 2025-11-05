# Prepare Files for Kaggle Upload
# This script helps you package your code and data for Kaggle

import os
import shutil
import zipfile
from pathlib import Path

print("=" * 80)
print("Preparing Files for Kaggle Upload")
print("=" * 80)

# Create output directory
output_dir = Path("kaggle_upload")
output_dir.mkdir(exist_ok=True)

print(f"\n📁 Output directory: {output_dir.absolute()}")

# ============================================================================
# Package 1: Source Code
# ============================================================================

print("\n📦 Creating source code package...")

code_dir = output_dir / "codeqa_code"
code_dir.mkdir(exist_ok=True)

# Copy source files
files_to_copy = {
    'src/': ['__init__.py', 'encoder.py', 'decoder.py', 'attention.py', 
             'copy_mechanism.py', 'seq2seq_model.py', 'vocabulary.py', 
             'data_loader.py', 'dataset.py'],
    'scripts/': ['train.py'],
    './': ['config.yaml', 'requirements.txt']
}

for source_dir, files in files_to_copy.items():
    for file in files:
        src_path = Path(source_dir) / file if source_dir != './' else Path(file)
        if src_path.exists():
            # Create destination directory
            dest_dir = code_dir / source_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            dest_path = dest_dir / file
            shutil.copy2(src_path, dest_path)
            print(f"  ✅ Copied: {src_path} → {dest_path}")
        else:
            print(f"  ⚠️  Not found: {src_path}")

# Create zip for code
code_zip_path = output_dir / "codeqa_code.zip"
print(f"\n📦 Creating zip: {code_zip_path}")
shutil.make_archive(str(code_zip_path.with_suffix('')), 'zip', code_dir)
print(f"  ✅ Created: {code_zip_path}")

# ============================================================================
# Package 2: Data and Vocabulary
# ============================================================================

print("\n📦 Creating data package...")

data_zip_path = output_dir / "codeqa_data.zip"

print(f"📦 Creating zip: {data_zip_path}")
print("  Including: data/python/ and saved_models/vocab_python.pkl")

# Create zip manually to include only necessary files
with zipfile.ZipFile(data_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    # Add vocabulary
    vocab_path = Path('saved_models/vocab_python.pkl')
    if vocab_path.exists():
        zipf.write(vocab_path, 'saved_models/vocab_python.pkl')
        print(f"  ✅ Added: {vocab_path}")
    else:
        print(f"  ⚠️  Not found: {vocab_path}")
    
    # Add data files
    data_base = Path('data/python')
    if data_base.exists():
        for split in ['train', 'dev']:
            split_dir = data_base / split
            if split_dir.exists():
                for file in split_dir.iterdir():
                    if file.is_file():
                        arcname = f'data/python/{split}/{file.name}'
                        zipf.write(file, arcname)
                        print(f"  ✅ Added: {file} → {arcname}")
            else:
                print(f"  ⚠️  Not found: {split_dir}")
    else:
        print(f"  ⚠️  Not found: {data_base}")

print(f"  ✅ Created: {data_zip_path}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("✅ PACKAGING COMPLETE")
print("=" * 80)

print("\nCreated files:")
print(f"  1. {code_zip_path} ({code_zip_path.stat().st_size / 1024:.1f} KB)")
print(f"  2. {data_zip_path} ({data_zip_path.stat().st_size / (1024*1024):.1f} MB)")

print("\n" + "=" * 80)
print("📤 NEXT STEPS: Upload to Kaggle")
print("=" * 80)

print("""
Step 1: Upload Code Dataset
   1. Go to: https://www.kaggle.com/datasets
   2. Click "New Dataset"
   3. Upload: codeqa_code.zip
   4. Name: codeqa-code
   5. Click "Create"

Step 2: Upload Data Dataset
   1. Go to: https://www.kaggle.com/datasets
   2. Click "New Dataset"
   3. Upload: codeqa_data.zip
   4. Name: codeqa-python-dataset
   5. Click "Create"

Step 3: Create Kaggle Notebook
   1. Go to: https://www.kaggle.com/code
   2. Click "New Notebook"
   3. Settings → Accelerator → GPU T4 x2
   4. Settings → Internet → ON
   5. Add Data → Search "codeqa-code" → Add
   6. Add Data → Search "codeqa-python-dataset" → Add
   7. Copy code from kaggle_train_notebook.py
   8. Update dataset names in notebook (lines with /kaggle/input/)
   9. Run all cells!

See KAGGLE_SETUP.md for detailed instructions.
""")

print("=" * 80)
