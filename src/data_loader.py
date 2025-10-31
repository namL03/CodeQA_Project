"""
Data Loader Module for CodeQA Dataset

This module handles loading the CodeQA dataset from text files.
Each example consists of three components:
- Question: A natural language question about code
- Code: The source code snippet (already tokenized)
- Answer: The expected answer to the question
"""

import os
from typing import List, Tuple, Dict


class CodeQADataLoader:
    """
    Loads the CodeQA dataset from text files.
    
    The data is organized as:
    - {language}/train/{split}.{question/answer/code}
    - {language}/dev/{split}.{question/answer/code}
    - {language}/test/{split}.{question/answer/code}
    
    Each line in the three files corresponds to one example.
    """
    
    def __init__(self, data_dir: str, language: str = 'python'):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory (e.g., 'c:/Users/.../data')
            language: Programming language ('python' or 'java')
        """
        self.data_dir = data_dir
        self.language = language
        self.lang_dir = os.path.join(data_dir, language)
        
        # Verify the directory exists
        if not os.path.exists(self.lang_dir):
            raise ValueError(f"Language directory not found: {self.lang_dir}")
    
    def load_split(self, split: str) -> List[Dict[str, str]]:
        """
        Load a specific data split (train, dev, or test).
        
        Args:
            split: One of 'train', 'dev', or 'test'
            
        Returns:
            A list of dictionaries, each containing:
                - 'question': The question text
                - 'code': The code snippet (tokenized)
                - 'answer': The answer text
        """
        # Construct file paths
        split_dir = os.path.join(self.lang_dir, split)
        question_file = os.path.join(split_dir, f"{split}.question")
        code_file = os.path.join(split_dir, f"{split}.code")
        answer_file = os.path.join(split_dir, f"{split}.answer")
        
        # Check if all files exist
        for file_path in [question_file, code_file, answer_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Read all three files
        print(f"Loading {split} split from {split_dir}...")
        with open(question_file, 'r', encoding='utf-8') as f:
            questions = f.readlines()
        
        with open(code_file, 'r', encoding='utf-8') as f:
            codes = f.readlines()
        
        with open(answer_file, 'r', encoding='utf-8') as f:
            answers = f.readlines()
        
        # Verify all files have the same number of lines
        assert len(questions) == len(codes) == len(answers), \
            f"Mismatch in number of lines: {len(questions)} questions, {len(codes)} codes, {len(answers)} answers"
        
        # Combine into a list of examples
        examples = []
        for idx, (question, code, answer) in enumerate(zip(questions, codes, answers)):
            examples.append({
                'question': question.strip(),
                'code': code.strip(),
                'answer': answer.strip(),
                'id': f"{split}_{idx}"
            })
        
        print(f"Loaded {len(examples)} examples from {split} split")
        return examples
    
    def load_all_splits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load all data splits (train, dev, test).
        
        Returns:
            A tuple of (train_data, dev_data, test_data)
        """
        train_data = self.load_split('train')
        dev_data = self.load_split('dev')
        test_data = self.load_split('test')
        
        return train_data, dev_data, test_data


def preprocess_example(example: Dict[str, str], 
                       add_special_tokens: bool = True) -> Dict[str, List[str]]:
    """
    Preprocess a single example by tokenizing the text.
    
    The input format for the model is: "[CLS] Question [SEP] Code"
    The output is the answer.
    
    Args:
        example: Dictionary with 'question', 'code', and 'answer'
        add_special_tokens: Whether to add [CLS] and [SEP] tokens
        
    Returns:
        Dictionary with tokenized 'input' and 'output'
    """
    # Tokenize question (split by spaces)
    question_tokens = example['question'].lower().split()
    
    # Tokenize code (already tokenized, just split by spaces)
    code_tokens = example['code'].split()
    
    # Tokenize answer
    answer_tokens = example['answer'].lower().split()
    
    # Create input sequence: [CLS] Question [SEP] Code
    if add_special_tokens:
        input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + code_tokens
    else:
        input_tokens = question_tokens + code_tokens
    
    return {
        'input': input_tokens,
        'output': answer_tokens,
        'question': question_tokens,
        'code': code_tokens,
        'id': example.get('id', '')
    }


# Example usage and testing
if __name__ == "__main__":
    # This code runs when you execute this file directly
    # It's useful for testing!
    
    # Update this path to your data directory
    DATA_DIR = "c:/Users/namlh/Code_QA_Project/data"
    
    print("=" * 60)
    print("Testing CodeQA Data Loader")
    print("=" * 60)
    
    # Create data loader
    loader = CodeQADataLoader(DATA_DIR, language='python')
    
    # Load training data
    train_data = loader.load_split('train')
    
    # Show first 3 examples
    print("\nFirst 3 examples from training set:")
    print("-" * 60)
    for i in range(min(3, len(train_data))):
        example = train_data[i]
        print(f"\nExample {i+1}:")
        print(f"Question: {example['question'][:100]}...")
        print(f"Answer: {example['answer']}")
        print(f"Code (first 50 chars): {example['code'][:50]}...")
    
    # Test preprocessing
    print("\n" + "=" * 60)
    print("Testing Preprocessing")
    print("=" * 60)
    processed = preprocess_example(train_data[0])
    print(f"\nOriginal question: {train_data[0]['question']}")
    print(f"Processed input (first 20 tokens): {' '.join(processed['input'][:20])}")
    print(f"Processed output: {' '.join(processed['output'])}")
