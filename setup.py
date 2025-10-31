"""
Setup script for home laptop
Run this after cloning the repository to set everything up
"""

import os
import sys

def check_data_directory():
    """Check if data directory exists with required structure."""
    print("=" * 70)
    print("Step 1: Checking data directory...")
    print("=" * 70)
    
    required_paths = [
        'data/python/train',
        'data/python/dev',
        'data/python/test',
        'data/java/train',
        'data/java/dev',
        'data/java/test'
    ]
    
    missing = []
    for path in required_paths:
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print("❌ Missing data directories:")
        for path in missing:
            print(f"   - {path}")
        print("\n⚠️  Please ensure your data is in the correct location!")
        print("   Expected structure:")
        print("   data/")
        print("   ├── python/")
        print("   │   ├── train/")
        print("   │   ├── dev/")
        print("   │   └── test/")
        print("   └── java/")
        print("       ├── train/")
        print("       ├── dev/")
        print("       └── test/")
        return False
    
    print("✅ Data directory structure looks good!")
    return True

def create_directories():
    """Create necessary directories."""
    print("\n" + "=" * 70)
    print("Step 2: Creating necessary directories...")
    print("=" * 70)
    
    dirs = ['saved_models', 'notebooks', 'logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created/verified: {dir_name}/")

def build_vocabularies():
    """Build vocabularies from training data."""
    print("\n" + "=" * 70)
    print("Step 3: Building vocabularies...")
    print("=" * 70)
    print("This may take a few minutes...\n")
    
    # Run the vocabulary builder
    exit_code = os.system("python scripts/build_vocabulary.py")
    
    if exit_code == 0:
        print("\n✅ Vocabularies built successfully!")
        return True
    else:
        print("\n❌ Failed to build vocabularies!")
        return False

def main():
    print("\n" + "=" * 70)
    print("CodeQA Project Setup Script")
    print("=" * 70)
    print("\nThis script will:")
    print("1. Check if data directory exists")
    print("2. Create necessary directories")
    print("3. Build vocabularies from training data")
    print("\n" + "=" * 70 + "\n")
    
    # Check data
    if not check_data_directory():
        print("\n⚠️  Setup incomplete: Please fix data directory first")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Build vocabularies
    if not build_vocabularies():
        print("\n⚠️  Setup incomplete: Vocabulary building failed")
        sys.exit(1)
    
    # Success!
    print("\n" + "=" * 70)
    print("🎉 Setup Complete!")
    print("=" * 70)
    print("\nYou're ready to start working!")
    print("\nNext steps:")
    print("  - Train a model: python scripts/train.py")
    print("  - Explore data: jupyter notebook")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
