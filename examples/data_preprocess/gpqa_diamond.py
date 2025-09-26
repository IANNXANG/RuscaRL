# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets
import random

from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/angel/lsy/huggingface/datasets/verl/gpqa_diamond')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = '/angel/lsy/huggingface/datasets/Idavidrein/gpqa'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, 'gpqa_diamond', trust_remote_code=True)

    test_dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer as a single option letter (A, B, C, or D) within \\boxed{}."
    GPQA_TEMPLATE = "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            choices = [example.pop("Incorrect Answer 1"), example.pop("Incorrect Answer 2"), example.pop("Incorrect Answer 3")]
            random.shuffle(choices)
            gold_index = random.randint(0, 3)
            choices.insert(gold_index, example.pop("Correct Answer"))
            question = GPQA_TEMPLATE.format(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example.pop("Question")
            )

            question = question + '\n\n' + instruction_following

            solution = "ABCD"[gold_index]
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
