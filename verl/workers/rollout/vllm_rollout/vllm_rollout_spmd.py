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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import math
import os
import pickle
import socket
import threading
import random
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Union
from dataclasses import dataclass

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: List[str]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=d.get("tags", [])
        )

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags
        }

def _generate_rubric_system_message(rubric_items: List[RubricItem], rubric_ratio: float, include_evaluation_warning: bool = True, rule_type: str = "linear") -> tuple[str, int]:
    """Generate system message with rubric information for open-book evaluation.
    
    Returns:
        tuple: (system_message, actual_rubric_count) - System prompt and actual number of displayed rubrics
    """
    if not rubric_items:
        return "You are a helpful medical assistant.", 0
    
    # Select criteria to display based on different rules
    if rule_type == "math_linear":
        # math_linear rule: Always show the last item (format requirement), show second-to-last item (final answer) when rubric_ratio=1, others shown in linear order
        total_criteria = len(rubric_items)
        selected_rubric_items = []
        
        # Always include the second-to-last item (format requirement)
        if total_criteria > 0:
            selected_rubric_items.append(rubric_items[-2])
        
        # When rubric_ratio=1, include the final item (final answer)
        if rubric_ratio >= 1.0 and total_criteria > 1:
            selected_rubric_items.insert(0, rubric_items[-1])  # Insert at the front to maintain order
        
        # For other items, display in linear order
        if total_criteria > 2:  # There are other items besides the last two
            other_items = rubric_items[:-2]  # Other items excluding the last two
            num_other_to_show = round(len(other_items) * rubric_ratio)
            
            # Select in order (not random)
            if num_other_to_show > 0:
                selected_other_items = other_items[:num_other_to_show]
                # Insert other items at the front, maintaining original order
                selected_rubric_items = selected_other_items + selected_rubric_items
    elif rubric_ratio < 1.0:
        # Original random selection logic (applicable to linear and binary rules)
        total_criteria = len(rubric_items)
        num_to_show = round(total_criteria * rubric_ratio)  # Round to nearest integer
        
        # 🔧 Modification: When calculated rubric count is 0, don't force display of any criterion
        # Remove original forced display logic: if num_to_show == 0 and total_criteria > 0 and rubric_ratio > 0: num_to_show = 1
        
        # Randomly select criteria to display
        if num_to_show > 0:
            selected_rubric_items = random.sample(rubric_items, num_to_show)
        else:
            selected_rubric_items = []
    else:
        selected_rubric_items = rubric_items
    
    # 🔧 If no rubric is selected, return base system prompt and 0
    if not selected_rubric_items:
        return "You are a helpful medical assistant.", 0
    
    positive_points = []
    negative_points = []
    
    pos_count = 1
    neg_count = 1
    
    for rubric_item in selected_rubric_items:
        criterion = rubric_item.criterion
        points = rubric_item.points
        
        if points > 0:
            positive_points.append(f"Criterion {pos_count}: {criterion} (worth {points} points)")
            pos_count += 1
        elif points < 0:
            negative_points.append(f"Criterion {neg_count}: {criterion} (penalty: {abs(points)} points)")
            neg_count += 1
    
    rubric_message = "You are a helpful assistant. For this question, please consider the following evaluation criteria:\n\n"
    
    if positive_points:
        rubric_message += "IMPORTANT POINTS TO INCLUDE (you should aim to address these):\n"
        rubric_message += "\n".join(positive_points)
        rubric_message += "\n\n"
    
    if negative_points:
        rubric_message += "IMPORTANT POINTS TO AVOID (you should not do these):\n"
        rubric_message += "\n".join(negative_points)
        rubric_message += "\n\n"
    
    rubric_message += "Please provide a comprehensive and helpful response that addresses the user's concerns while following the above guidelines.\n\n"
    
    if include_evaluation_warning:
        rubric_message += "IMPORTANT: Do not mention or reference these evaluation criteria in your response. Do not indicate that you have seen any scoring rubric or evaluation guidelines. Your response should appear natural and spontaneous. Revealing that you have access to evaluation criteria would be considered cheating and is strictly prohibited."
    
    return rubric_message, len(selected_rubric_items)


def _calculate_rubric_ratio(sample_idx: int, total_samples: int, rule_type: str = "linear", full_reveal_count: int = 1, global_steps: int = None, total_training_steps: int = None, step_power_n: float = 1.0, step_sigmoid_start_point: float = 0.3, step_sigmoid_steepness: float = 10.0, group_decay_type: str = "linear") -> float:
    """Calculate rubric ratio based on different rules.
    
    Args:
        sample_idx: Current sample index (0-based)
        total_samples: Total number of samples
        rule_type: Rule type ("linear", "binary", "math_linear", "step_linear", "step_power", or "step_sigmoid")
        full_reveal_count: Number of full reveals under binary rule (only valid when rule_type is "binary")
        global_steps: Current training step (only valid when rule_type is "step_linear", "step_power" or "step_sigmoid")
        total_training_steps: Total training steps (only valid when rule_type is "step_linear", "step_power" or "step_sigmoid")
        step_power_n: Exponent of power function (only valid when rule_type is "step_power", default 1.0)
        step_sigmoid_start_point: Start timing of sigmoid decay (only valid when rule_type is "step_sigmoid", range 0-1, default 0.3)
        step_sigmoid_steepness: Steepness of sigmoid decay (only valid when rule_type is "step_sigmoid", higher values mean faster decay, default 10.0)
        group_decay_type: In-group decay method ("linear" or "binary", only valid in step-level rules, default "linear")
    
    Returns:
        float: Rubric reveal ratio [0.0, 1.0]
    """
    if total_samples == 1:
        return 1.0
    
    if rule_type == "linear":
        # Linear change rule: 1st time: 1.0, 2nd time: (n-2)/(n-1), ..., nth time: 0.0
        return max(0.0, (total_samples - 1 - sample_idx) / (total_samples - 1))
    
    elif rule_type == "binary":
        # Binary rule: First x samples fully revealed (1.0), remaining n-x samples not revealed (0.0)
        full_reveal_count = min(full_reveal_count, total_samples)  # Ensure not exceeding total count
        return 1.0 if sample_idx < full_reveal_count else 0.0
    
    elif rule_type == "math_linear":
        # Math linear rule: Same linear change as linear, but with special handling in _generate_rubric_system_message
        # 1st time: 1.0, 2nd time: (n-2)/(n-1), ..., nth time: 0.0
        return max(0.0, (total_samples - 1 - sample_idx) / (total_samples - 1))
    
    elif rule_type == "step_linear":
        # Step linear rule: In-group decay + step linear
        # In-group decay: Choose linear or binary based on group_decay_type
        if group_decay_type == "binary":
            # In-group binary: First x samples fully revealed (1.0), remaining n-x samples not revealed (0.0)
            group_ratio = 1.0 if sample_idx < full_reveal_count else 0.0
        else:
            # In-group linear: 1st time: 1.0, 2nd time: (n-2)/(n-1), ..., nth time: 0.0
            group_ratio = max(0.0, (total_samples - 1 - sample_idx) / (total_samples - 1))
        
        # Step linear: From step 1 fully revealed (1.0) to last step not revealed (0.0)
        if global_steps is None or total_training_steps is None or total_training_steps <= 1:
            # If no step information provided or total steps <= 1, use only in-group decay
            step_ratio = 1.0
        else:
            # Step linear decay: step1=1.0, step_last=0.0
            step_ratio = max(0.0, (total_training_steps - global_steps) / (total_training_steps - 1))
        
        # Combine in-group factor and step factor
        combined_ratio = group_ratio * step_ratio
        return combined_ratio
    
    elif rule_type == "step_power":
        # Step power function rule: In-group decay + step power function decay
        # In-group decay: Choose linear or binary based on group_decay_type
        if group_decay_type == "binary":
            # In-group binary: First x samples fully revealed (1.0), remaining n-x samples not revealed (0.0)
            group_ratio = 1.0 if sample_idx < full_reveal_count else 0.0
        else:
            # In-group linear: 1st time: 1.0, 2nd time: (n-2)/(n-1), ..., nth time: 0.0
            group_ratio = max(0.0, (total_samples - 1 - sample_idx) / (total_samples - 1))
        
        # Step power function decay: y = (1-x)^n, where x ranges from 0 to 1
        if global_steps is None or total_training_steps is None or total_training_steps <= 1:
            # If no step information provided or total steps <= 1, use only in-group decay
            step_ratio = 1.0
        else:
            # Calculate step progress: x = (global_steps - 1) / (total_training_steps - 1)
            # So that x=0 at step 1, x=1 at last step
            x = min(1.0, max(0.0, (global_steps - 1) / (total_training_steps - 1)))
            # Power function decay: y = (1-x)^n
            step_ratio = max(0.0, (1.0 - x) ** step_power_n)
        
        # Combine in-group factor and power function step factor
        combined_ratio = group_ratio * step_ratio
        return combined_ratio
    
    elif rule_type == "step_sigmoid":
        # Step sigmoid rule: In-group decay + step sigmoid decay
        # In-group decay: Choose linear or binary based on group_decay_type
        if group_decay_type == "binary":
            # In-group binary: First x samples fully revealed (1.0), remaining n-x samples not revealed (0.0)
            group_ratio = 1.0 if sample_idx < full_reveal_count else 0.0
        else:
            # In-group linear: 1st time: 1.0, 2nd time: (n-2)/(n-1), ..., nth time: 0.0
            group_ratio = max(0.0, (total_samples - 1 - sample_idx) / (total_samples - 1))
        
        # Step sigmoid decay: y = 1 / (1 + exp((x - start_point) * steepness))
        if global_steps is None or total_training_steps is None or total_training_steps <= 1:
            # If no step information provided or total steps <= 1, use only in-group decay
            step_ratio = 1.0
        else:
            # Calculate step progress: x = (global_steps - 1) / (total_training_steps - 1)
            # So that x=0 at step 1, x=1 at last step
            x = min(1.0, max(0.0, (global_steps - 1) / (total_training_steps - 1)))
            # Sigmoid decay: Shift and scale x coordinate to start decay at specified point
            shifted_x = (x - step_sigmoid_start_point) * step_sigmoid_steepness
            # Use inverse sigmoid form to achieve decay from 1 to 0
            step_ratio = max(0.0, min(1.0, 1.0 / (1.0 + np.exp(shifted_x))))
        
        # Combine in-group factor and sigmoid step factor
        combined_ratio = group_ratio * step_ratio
        return combined_ratio
    
    else:
        # Default to linear rule
        return max(0.0, (total_samples - 1 - sample_idx) / (total_samples - 1))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(model_hf_config.llm_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(model_hf_config.text_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = {} if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer  # Store tokenizer for later use
        self.model_path = model_path  # Store model path for model type detection

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array([_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")] * batch_size

        # Graded system prompt functionality
        enable_graded_system_prompt = getattr(self.config, 'enable_graded_system_prompt', False)
        participate_training = getattr(self.config, 'graded_system_prompt_participate_training', False)
        original_n = self.sampling_params.n
        original_vllm_inputs = vllm_inputs.copy()
        original_lora_requests = lora_requests.copy() if lora_requests else None
        
        if enable_graded_system_prompt and do_sample and not is_validate and original_n > 1:
            
            # Get rubric information from meta_info (additional data channel)
            rubric_info_available = False
            rubric_items_list = []
            
            # Prioritize getting reward_model information from meta_info
            if 'graded_system_prompt_reward_models' in prompts.meta_info:
                reward_models = prompts.meta_info['graded_system_prompt_reward_models']
                
                for i in range(batch_size):
                    if i < len(reward_models):
                        reward_model = reward_models[i]
                        if isinstance(reward_model, dict) and 'rubrics' in reward_model:
                            rubrics = reward_model['rubrics']
                            rubric_items = [RubricItem.from_dict(r) for r in rubrics]
                            rubric_items_list.append(rubric_items)
                            rubric_info_available = True
                        else:
                            rubric_items_list.append([])
                    else:
                        rubric_items_list.append([])
            
            # Fallback: Get rubric information from non_tensor_batch
            elif 'reward_model' in non_tensor_batch:
                for i in range(batch_size):
                    reward_model = non_tensor_batch['reward_model'][i]
                    if isinstance(reward_model, dict) and 'rubrics' in reward_model:
                        rubrics = reward_model['rubrics']
                        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
                        rubric_items_list.append(rubric_items)
                        rubric_info_available = True
                    else:
                        rubric_items_list.append([])
            else:
                # If no rubric information found, fill with empty lists
                rubric_items_list = [[] for _ in range(batch_size)]
            
            # If not found in non_tensor_batch, try to get from other places
            if not rubric_info_available and hasattr(prompts, 'non_tensor_batch'):
                # Check extra_info
                if 'extra_info' in prompts.non_tensor_batch:
                    for i in range(batch_size):
                        extra_info = prompts.non_tensor_batch['extra_info'][i]
                        if isinstance(extra_info, dict) and 'reward_model' in extra_info:
                            reward_model = extra_info['reward_model']
                            if isinstance(reward_model, dict) and 'rubrics' in reward_model:
                                rubrics = reward_model['rubrics']
                                rubric_items = [RubricItem.from_dict(r) for r in rubrics]
                                rubric_items_list.append(rubric_items)
                                rubric_info_available = True
                            else:
                                rubric_items_list.append([])
                        else:
                            rubric_items_list.append([])
            
            # Get rule type and related parameters
            rule_type = getattr(self.config, 'graded_system_prompt_rule', 'linear')
            full_reveal_count = getattr(self.config, 'graded_system_prompt_full_reveal_count', 1)
            # Control whether to add base system prompt when rubric_ratio is 0, default True
            add_base_system_prompt_when_zero = getattr(self.config, 'graded_system_prompt_add_base_when_zero', True)
            # Control whether to add evaluation criteria warning text in rubric_message, default True
            include_evaluation_warning = getattr(self.config, 'graded_system_prompt_include_evaluation_warning', True)
            # Get power function exponent parameter for step_power rule, default 1
            step_power_n = getattr(self.config, 'graded_system_prompt_step_power_n', 1)
            # Get parameters for step_sigmoid rule
            step_sigmoid_start_point = getattr(self.config, 'graded_system_prompt_step_sigmoid_start_point', 0.3)
            step_sigmoid_steepness = getattr(self.config, 'graded_system_prompt_step_sigmoid_steepness', 10.0)
            # Get in-group decay method parameter, default linear
            group_decay_type = getattr(self.config, 'graded_system_prompt_group_decay_type', 'linear')
            
            # Get training step information (for step_linear rule)
            global_steps = prompts.meta_info.get('global_steps', None)
            total_training_steps = prompts.meta_info.get('total_training_steps', None)
            
            # Simplified startup information
            if rubric_info_available:
                total_rubrics = sum(len(items) for items in rubric_items_list)
                print(f"[GRADED SYSTEM PROMPT] Graded system prompt functionality enabled: n={original_n}, total rubrics={total_rubrics}, rule type={rule_type}")
                if rule_type == "binary":
                    print(f"[GRADED SYSTEM PROMPT] Binary rule: First {full_reveal_count} fully revealed, remaining {original_n - full_reveal_count} not revealed")
                elif rule_type == "step_linear":
                    if global_steps is not None and total_training_steps is not None:
                        step_ratio = max(0.0, (total_training_steps - global_steps) / (total_training_steps - 1)) if total_training_steps > 1 else 1.0
                        print(f"[GRADED SYSTEM PROMPT] Step linear rule: Current step={global_steps}/{total_training_steps}, step decay factor={step_ratio:.3f}, in-group decay method={group_decay_type}")
                    else:
                        print(f"[GRADED SYSTEM PROMPT] Step linear rule: No step information obtained, using only in-group decay, in-group decay method={group_decay_type}")
                elif rule_type == "step_power":
                    if global_steps is not None and total_training_steps is not None:
                        step_progress = global_steps / (total_training_steps - 1) if total_training_steps > 1 else 0.0
                        step_ratio = (1 - step_progress) ** step_power_n
                        print(f"[GRADED SYSTEM PROMPT] Step power function rule: Current step={global_steps}/{total_training_steps}, step progress={step_progress:.3f}, power exponent n={step_power_n}, step decay factor={step_ratio:.3f}, in-group decay method={group_decay_type}")
                    else:
                        print(f"[GRADED SYSTEM PROMPT] Step power function rule: No step information obtained, using only in-group decay, power exponent n={step_power_n}, in-group decay method={group_decay_type}")
                elif rule_type == "step_sigmoid":
                    if global_steps is not None and total_training_steps is not None:
                        x = min(1.0, max(0.0, (global_steps - 1) / (total_training_steps - 1))) if total_training_steps > 1 else 0.0
                        shifted_x = (x - step_sigmoid_start_point) * step_sigmoid_steepness
                        step_ratio = max(0.0, 1 / (1 + math.exp(shifted_x)))
                        print(f"[GRADED SYSTEM PROMPT] Step Sigmoid rule: Current step={global_steps}/{total_training_steps}, step progress={x:.3f}, start point={step_sigmoid_start_point}, steepness={step_sigmoid_steepness}, step decay factor={step_ratio:.3f}, in-group decay method={group_decay_type}")
                    else:
                        print(f"[GRADED SYSTEM PROMPT] Step Sigmoid rule: No step information obtained, using only in-group decay, start point={step_sigmoid_start_point}, steepness={step_sigmoid_steepness}, in-group decay method={group_decay_type}")

            else:
                print(f"[GRADED SYSTEM PROMPT] Graded system prompt functionality enabled: n={original_n}, no rubric information found, rule type={rule_type}")
                if rule_type == "step_linear":
                    if global_steps is not None and total_training_steps is not None:
                        step_ratio = max(0.0, (total_training_steps - global_steps) / (total_training_steps - 1)) if total_training_steps > 1 else 1.0
                        print(f"[GRADED SYSTEM PROMPT] Step linear rule: Current step={global_steps}/{total_training_steps}, step decay factor={step_ratio:.3f}")
                    else:
                        print(f"[GRADED SYSTEM PROMPT] Step linear rule: No step information obtained, using only in-group linear")
                elif rule_type == "step_power":
                    if global_steps is not None and total_training_steps is not None:
                        step_progress = global_steps / (total_training_steps - 1) if total_training_steps > 1 else 0.0
                        step_ratio = (1 - step_progress) ** step_power_n
                        print(f"[GRADED SYSTEM PROMPT] Step power function rule: Current step={global_steps}/{total_training_steps}, step progress={step_progress:.3f}, power exponent n={step_power_n}, step decay factor={step_ratio:.3f}")
                    else:
                        print(f"[GRADED SYSTEM PROMPT] Step power function rule: No step information obtained, using only in-group linear, power exponent n={step_power_n}")
            
            # Generate n different proportions of system prompts
            expanded_vllm_inputs = []
            expanded_lora_requests = []
            
            # Collect graded system prompt information for logging
            graded_prompt_info = []
            
            for i in range(batch_size):
                rubric_items = rubric_items_list[i]
                original_prompt_ids = original_vllm_inputs[i]["prompt_token_ids"]
                
                for sample_idx in range(original_n):
                    # Use new rules to calculate rubric reveal ratio
                    rubric_ratio = _calculate_rubric_ratio(sample_idx, original_n, rule_type, full_reveal_count, global_steps, total_training_steps, step_power_n, step_sigmoid_start_point, step_sigmoid_steepness, group_decay_type)
                    
                    # Generate system prompt
                    if rubric_items:
                        system_message, actual_rubric_count = _generate_rubric_system_message(rubric_items, rubric_ratio, include_evaluation_warning, rule_type)
                        
                        # 🔧 Decide whether to add system prompt based on actual_rubric_count
                        if actual_rubric_count > 0 or (actual_rubric_count == 0 and add_base_system_prompt_when_zero):
                            # Only add when actual rubric count > 0, or when it's 0 but base prompt addition is allowed
                            # Convert system message to token ids and add to front of prompt
                            # Use tokenizer to encode system message, wrapped in correct chat template
                            tokenizer = self.tokenizer
                            # Choose correct chat template format based on model type
                            model_name_lower = self.model_path.lower()
                            if "llama" in model_name_lower:
                                # Llama models need special handling: system prompt inserted after <|begin_of_text|>
                                formatted_system_message = f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
                                system_tokens = tokenizer.encode(formatted_system_message, add_special_tokens=False)
                                
                                # Check if original prompt starts with <|begin_of_text|>
                                begin_of_text_token = tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)
                                if len(original_prompt_ids) > 0 and original_prompt_ids[:len(begin_of_text_token)] == begin_of_text_token:
                                    # If <|begin_of_text|> exists, insert system prompt after it
                                    combined_prompt_ids = begin_of_text_token + system_tokens + original_prompt_ids[len(begin_of_text_token):]
                                else:
                                    # If no <|begin_of_text|>, assert error
                                    assert False, f"Llama model prompt must start with <|begin_of_text|>, but current prompt does not contain this token. Model path: {self.model_path}"
                            else:
                                # Format used by Qwen models (default)
                                formatted_system_message = f"<|im_start|>system\n{system_message}<|im_end|>\n"
                                system_tokens = tokenizer.encode(formatted_system_message, add_special_tokens=False)
                                
                                # Combine system tokens and original prompt tokens
                                combined_prompt_ids = system_tokens + original_prompt_ids
                            
                            # Decode combined_prompt_ids to text
                            combined_prompt_text = tokenizer.decode(combined_prompt_ids, skip_special_tokens=False)
                            
                            # Collect information for logging
                            graded_info = {
                                "rubric_count": len(rubric_items),  # Total rubric count
                                "actual_rubric_count": actual_rubric_count,  # Actual displayed rubric count
                                "rubric_ratio": rubric_ratio,
                                "system_prompt": system_message,
                                "combined_prompt_text": combined_prompt_text,
                                "original_sample_index": i,
                                "graded_sample_index": sample_idx
                            }
                        else:
                            # 🔧 When actual_rubric_count is 0 and base prompt addition is not allowed, don't add any system prompt
                            combined_prompt_ids = original_prompt_ids
                            
                            # Decode combined_prompt_ids to text
                            combined_prompt_text = self.tokenizer.decode(combined_prompt_ids, skip_special_tokens=False)
                            
                            # Collect information for logging
                            graded_info = {
                                "rubric_count": len(rubric_items),  # Total rubric count
                                "actual_rubric_count": actual_rubric_count,  # Actual displayed rubric count (0)
                                "rubric_ratio": rubric_ratio,
                                "system_prompt": "",
                                "combined_prompt_text": combined_prompt_text,
                                "original_sample_index": i,
                                "graded_sample_index": sample_idx
                            }
                    else:
                        # When no rubric_items, don't add any system prompt
                        combined_prompt_ids = original_prompt_ids
                        
                        # Decode combined_prompt_ids to text
                        combined_prompt_text = self.tokenizer.decode(combined_prompt_ids, skip_special_tokens=False)
                        
                        # Collect information for logging
                        graded_info = {
                            "rubric_count": 0,  # Total rubric count is 0
                            "actual_rubric_count": 0,  # Actual displayed rubric count is 0
                            "rubric_ratio": rubric_ratio,
                            "system_prompt": "",
                            "combined_prompt_text": combined_prompt_text,
                            "original_sample_index": i,
                            "graded_sample_index": sample_idx
                        }
                    
                    graded_prompt_info.append(graded_info)
                    
                    # Create new vllm input
                    new_input = original_vllm_inputs[i].copy()
                    new_input["prompt_token_ids"] = combined_prompt_ids
                    expanded_vllm_inputs.append(new_input)
                    
                    # Add corresponding lora request
                    if original_lora_requests:
                        expanded_lora_requests.append(original_lora_requests[i])
            
            # Update parameters
            vllm_inputs = expanded_vllm_inputs
            lora_requests = expanded_lora_requests if expanded_lora_requests else None
            batch_size = len(vllm_inputs)  # New batch size = original batch_size * n
            
            # Modify sampling parameters, each prompt samples only once
            kwargs_with_n1 = kwargs.copy()
            kwargs_with_n1["n"] = 1
        else:
            kwargs_with_n1 = kwargs

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs_with_n1):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(rollout_log_probs, -1, max_length=self.config.response_length).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], self.sampling_params.n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], self.sampling_params.n)
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], self.sampling_params.n)
                # Handle reward_model field
                if "reward_model" in non_tensor_batch.keys():
                    non_tensor_batch["reward_model"] = _repeat_interleave(non_tensor_batch["reward_model"], self.sampling_params.n)

            # If graded system prompt is used, decide whether to restore original prompt information based on participate_training parameter
            if enable_graded_system_prompt and do_sample and not is_validate and original_n > 1 and not participate_training:
                
                # Save original idx for assertion verification
                original_idx = prompts.batch["input_ids"]  # (bs, prompt_length)
                original_attention_mask = prompts.batch["attention_mask"]
                original_position_ids = prompts.batch["position_ids"]
                
                # Rebuild original idx, attention_mask, position_ids
                original_batch_size = len(original_vllm_inputs)
                
                # Restore original prompt token ids
                restored_prompts = []
                for i in range(original_batch_size):
                    original_prompt_ids = original_vllm_inputs[i]["prompt_token_ids"]
                    # Repeat n times
                    for _ in range(original_n):
                        restored_prompts.append(original_prompt_ids)
                
                # Note: Need to use the same max_len as current idx to ensure size consistency
                current_max_len = idx.shape[1]  # Use the maximum length already in current batch
                restored_idx = []
                restored_attention_mask = []
                
                for prompt_ids in restored_prompts:
                    # Left padding to current_max_len length
                    if len(prompt_ids) > current_max_len:
                        # If original prompt is longer than current length, truncate (left truncation keeping last tokens)
                        padded_ids = prompt_ids[-current_max_len:]
                        attention = [1] * current_max_len
                    else:
                        # Left padding
                        padding_length = current_max_len - len(prompt_ids)
                        padded_ids = [self.pad_token_id] * padding_length + prompt_ids
                        attention = [0] * padding_length + [1] * len(prompt_ids)
                    
                    restored_idx.append(padded_ids)
                    restored_attention_mask.append(attention)
                
                restored_idx = torch.tensor(restored_idx, device=idx.device)
                restored_attention_mask = torch.tensor(restored_attention_mask, device=attention_mask.device)
                
                # Recalculate position_ids
                restored_position_ids = (restored_attention_mask.cumsum(dim=1) - 1) * restored_attention_mask
                
                # Update idx, attention_mask, position_ids to original values
                idx = restored_idx
                attention_mask = restored_attention_mask  # Only prompt part attention_mask
                position_ids = restored_position_ids
                
                # Assert verification that restored prompt_idx matches original
                # Repeat original idx n times for comparison
                expected_idx = _repeat_interleave(original_idx, original_n)
                expected_attention_mask = _repeat_interleave(original_attention_mask, original_n)
                expected_position_ids = _repeat_interleave(original_position_ids, original_n)
                
                # Verify that restored idx matches original repeated idx
                assert idx.shape == expected_idx.shape, f"Restored idx shape mismatch: {idx.shape} vs {expected_idx.shape}"
                assert torch.equal(idx, expected_idx), f"Restored idx content mismatch, original idx and restored idx don't match"
                
                # Verify attention_mask
                assert attention_mask.shape == expected_attention_mask.shape, f"Restored attention_mask shape mismatch: {attention_mask.shape} vs {expected_attention_mask.shape}"
                assert torch.equal(attention_mask, expected_attention_mask), f"Restored attention_mask content mismatch"
                
                # Verify position_ids
                assert position_ids.shape == expected_position_ids.shape, f"Restored position_ids shape mismatch: {position_ids.shape} vs {expected_position_ids.shape}"
                assert torch.equal(position_ids, expected_position_ids), f"Restored position_ids content mismatch"
                
                print(f"[GRADED SYSTEM PROMPT] Assertion verification passed: Restored prompt information completely matches original information (repeated {original_n} times)")
                
                # Also need to repeat data in non_tensor_batch
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], original_n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], original_n)
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], original_n)
                # Handle reward_model field - this is the key fix!
                if "reward_model" in non_tensor_batch.keys():
                    non_tensor_batch["reward_model"] = _repeat_interleave(non_tensor_batch["reward_model"], original_n)
            
            # When participate_training=True, system prompts participate in training, need to rebuild prompt information to match expanded batch
            elif enable_graded_system_prompt and do_sample and not is_validate and original_n > 1 and participate_training:
                print(f"[GRADED SYSTEM PROMPT] System prompt training mode: Rebuilding prompt information to include system prompts")
                
                # Rebuild idx, attention_mask, position_ids to match expanded prompts (including system prompts)
                original_batch_size = len(original_vllm_inputs)
                
                # Rebuild prompt information from expanded vllm_inputs
                expanded_prompts = []
                for vllm_input in vllm_inputs:
                    expanded_prompts.append(vllm_input["prompt_token_ids"])
                
                # Find maximum length for padding
                max_prompt_len = max(len(prompt_ids) for prompt_ids in expanded_prompts)
                
                # Synchronize max_prompt_len in distributed environment to ensure all workers use same padding length
                if torch.distributed.is_initialized():
                    max_prompt_len_tensor = torch.tensor(max_prompt_len, dtype=torch.long, device=idx.device)
                    torch.distributed.all_reduce(max_prompt_len_tensor, op=torch.distributed.ReduceOp.MAX)
                    max_prompt_len = max_prompt_len_tensor.item()
                    print(f"[GRADED SYSTEM PROMPT] Synchronized max_prompt_len: {max_prompt_len}")
                
                # Rebuild idx and attention_mask
                new_idx = []
                new_attention_mask = []
                
                for prompt_ids in expanded_prompts:
                    # Left padding to max_prompt_len length
                    if len(prompt_ids) > max_prompt_len:
                        # If prompt is longer than maximum length, truncate (left truncation keeping last tokens)
                        padded_ids = prompt_ids[-max_prompt_len:]
                        attention = [1] * max_prompt_len
                    else:
                        # Left padding
                        padding_length = max_prompt_len - len(prompt_ids)
                        padded_ids = [self.pad_token_id] * padding_length + prompt_ids
                        attention = [0] * padding_length + [1] * len(prompt_ids)
                    
                    new_idx.append(padded_ids)
                    new_attention_mask.append(attention)
                
                # Convert to tensor
                idx = torch.tensor(new_idx, device=idx.device)
                attention_mask = torch.tensor(new_attention_mask, device=attention_mask.device)
                
                # Recalculate position_ids
                position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask
                
                # Update batch_size
                batch_size = len(vllm_inputs)
                
                print(f"[GRADED SYSTEM PROMPT] Rebuild completed: batch_size={batch_size}, prompt_length={max_prompt_len}")
                
                # Repeat data in non_tensor_batch to match expanded batch_size
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(non_tensor_batch["tools_kwargs"], original_n)
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(non_tensor_batch["interaction_kwargs"], original_n)
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(non_tensor_batch["raw_prompt"], original_n)
                if "reward_model" in non_tensor_batch.keys():
                    non_tensor_batch["reward_model"] = _repeat_interleave(non_tensor_batch["reward_model"], original_n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # If graded system prompt is used, add related information to non_tensor_batch for trainer logging
        if enable_graded_system_prompt and do_sample and not is_validate and original_n > 1:
            # Convert graded system prompt information to numpy array
            graded_prompt_info_array = np.array(graded_prompt_info, dtype=object)
            non_tensor_batch["graded_prompt_info"] = graded_prompt_info_array

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
