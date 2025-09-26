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

import re

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

# Patterns for GPQA multiple choice questions
ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
BOXED_PATTERN_MULTICHOICE = r"\\boxed\{([A-D])\}"
JSON_ANSWER_PATTERN_MULTICHOICE = r'"answer"\s*:\s*"([A-D])"'


def _extract_choice_answer(model_output: str) -> str:
    """Extract choice answer (A, B, C, D) from model output."""
    # First try to match Answer: format
    match = re.search(ANSWER_PATTERN_MULTICHOICE, model_output)
    if match:
        return match.group(1)
    
    # Try to match \boxed{} format
    match = re.search(BOXED_PATTERN_MULTICHOICE, model_output)
    if match:
        return match.group(1)
    
    # Try to match JSON answer format: "answer": "C"
    match = re.search(JSON_ANSWER_PATTERN_MULTICHOICE, model_output)
    if match:
        return match.group(1)
    
    return None


def _is_choice_question(ground_truth: str) -> bool:
    """Check if the ground truth is a single choice (A, B, C, D)."""
    return ground_truth.strip() in ['A', 'B', 'C', 'D']


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    # First check if this is a multiple choice question
    if _is_choice_question(ground_truth):
        # Try to extract choice answer first
        extracted_choice = _extract_choice_answer(model_output)
        if extracted_choice == ground_truth.strip():
            return 1.0
        
        # If choice matching fails (score is 0), fall back to math verification
        # This handles cases where the answer might be expressed mathematically
        # but the ground truth is still a choice letter
        pass
    
    # Perform math verification for non-choice questions or when choice matching fails
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score
