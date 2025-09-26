import os
import json
import datasets
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass
from healthbench_reward import RubricItem

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def make_map_fn(split: str):
    """Construct data mapping function"""
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract prompt
        prompt = example['prompt']
        
        # Extract rubrics
        rubrics = [RubricItem.from_dict(r) for r in example['rubrics']]
        
        # Construct reward_model field
        reward_model = {
            "style": "rubric",
            "rubrics": [r.to_dict() for r in rubrics],
            "ground_truth": ""  # Use empty string
        }
        
        # Construct data format required by verl
        data = {
            "data_source": "healthbench",
            "prompt": prompt,  # Keep outer prompt
            "ability": "medical_chat",
            "reward_model": reward_model,  # Keep outer reward_model
            "extra_info": {
                "prompt": prompt,  # Also put prompt in extra_info
                "reward_model": reward_model  # Also put reward_model in extra_info
            }
        }
        return data
    
    return process_fn

def process_dataset(data_list: List[Dict[str, Any]], split: str) -> datasets.Dataset:
    """Process dataset"""
    dataset = datasets.Dataset.from_list(data_list)
    processed_dataset = dataset.map(
        function=make_map_fn(split),
        with_indices=True,
        remove_columns=dataset.column_names  # Remove all original columns
    )
    # Shuffle the data
    
    shuffled_dataset = processed_dataset.shuffle(seed=42)
    return shuffled_dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data_group')
    parser.add_argument('--output_dir', default='data/health_bench')
    parser.add_argument('--hdfs_dir', default=None)
    
    args = parser.parse_args()
    
    # # Load training data
    # train_data = []
    # for i in range(1, 4):  # group1-3
    #     file_path = os.path.join(args.local_dir, f'healthbench_group{i}.jsonl')
    #     train_data.extend(load_jsonl(file_path))

    # Load training data
    train_file = os.path.join(args.local_dir, 'healthbench_train.jsonl')
    train_data = load_jsonl(train_file)
    
    # Load validation data
    val_file = os.path.join(args.local_dir, 'healthbench_eval.jsonl')
    val_data = load_jsonl(val_file)
    # Only keep first 100 validation data "validation set"
    val_data = val_data
    
    # Process training and validation sets
    train_dataset = process_dataset(train_data, 'train')
    val_dataset = process_dataset(val_data, 'val')
    
    # Save as parquet format
    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_train.parquet'))
    val_dataset.to_parquet(os.path.join(args.output_dir, 'healthbench_val.parquet'))
    
    # Print dataset information
    print("\nDataset information:")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Print data sample examples
    print("\nTraining set sample example:")
    print(json.dumps(train_dataset[0], indent=2, ensure_ascii=False))
    print("\nValidation set sample example:")
    print(json.dumps(val_dataset[0], indent=2, ensure_ascii=False))
    
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.output_dir, dst=args.hdfs_dir)

if __name__ == '__main__':
    main()