import json
import re
import os
import requests
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from openai import OpenAI
from dotenv import load_dotenv
import importlib

# Import math_verify module
try:
    from verl.utils.reward_score.math_verify import compute_score as math_verify_compute_score
except ImportError:
    print("Warning: math_verify module not found. Rule-based verification will be disabled.")
    math_verify_compute_score = None

# Concurrent workers control variable for single URL
MAX_CONCURRENT_WORKERS = 512

# Load .env file
load_dotenv()

def math_verify(model_response: str, parameter: str) -> bool:
    """
    Use math_verify for mathematical answer verification
    
    Args:
        model_response: Model's response
        parameter: Verification parameter (usually ground truth)
        
    Returns:
        bool: Verification result, True for correct, False for incorrect
    """
    if math_verify_compute_score is None:
        print("math_verify module not available, returning False")
        return False
        
    try:
        # Use math_verify's compute_score function for verification
        # This function returns 0.0 or 1.0, we convert it to bool
        score = math_verify_compute_score(model_response, parameter)
        return score > 0.5  # Consider > 0.5 as correct
    except Exception as e:
        print(f"math_verify verification failed: {e}")
        return False

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

@dataclass
class SamplerResponse:
    """Sampler response"""
    response_text: str
    response_metadata: dict
    actual_queried_message_list: List[Dict[str, str]]

class SamplerBase:
    """Base sampler class"""
    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        return {"role": role, "content": content}

class ChatCompletionSampler(SamplerBase):
    """OpenAI API sampler"""
    def __init__(
        self,
        model: str = "gpt-4.1-2025-04-14",
        system_message: str | None = None,
        temperature: float = 0,
        max_tokens: int = 2048,
    ):
        self.client = OpenAI()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message(self.system_message, "system")
            ] + message_list
        
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response, retrying...")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = min(2**trial, 300)  # Exponential backoff, max wait time 300 seconds
                print(
                    f"Rate limit exception, waiting {exception_backoff} seconds before retry {trial}",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1

class VLLMSampler(SamplerBase):
    """Local VLLM service sampler"""
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        enable_thinking: bool = False,
        filter_think_tags: bool = True,
    ):
        # Support multiple URL configuration
        if base_url:
            self.base_urls = [base_url]
        else:
            # Read URL list from environment variable, support comma separation
            url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
            self.base_urls = [url.strip() for url in url_env.split(',') if url.strip()]
        
        # Load balancing related variables
        self.current_url_index = 0  # Index for round-robin
        self.url_loads = {}  # Store load information for each URL
        self.url_weights = {}  # Store weights for each URL
        self.virtual_loads = {}  # Store virtual load for each URL (including allocated requests)
        
        # Initialize load information for each URL
        for url in self.base_urls:
            self.url_loads[url] = {'running': 0, 'waiting': 0, 'total': 0}
            self.url_weights[url] = 1.0  # Initial weights are equal
            self.virtual_loads[url] = 0  # Initial virtual load is 0
        
        # Perform load statistics once during initialization
        self._update_loads_and_weights()
        
        self.model = model or os.getenv("VLLM_MODEL", "8001vllm")
        self.system_message = system_message
        self.temperature = temperature if temperature is not None else float(os.getenv("VLLM_TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("VLLM_MAX_TOKENS", "2048"))
        self.timeout = timeout if timeout is not None else int(os.getenv("VLLM_TIMEOUT", "120"))
        self.enable_thinking = enable_thinking
        self.filter_think_tags = filter_think_tags
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy"
        }
        
        print(f"VLLMSampler initialization completed, configured {len(self.base_urls)} URLs: {self.base_urls}")
        
        # Print load information during initialization
        print("\n=== Initialization Load Information ===")
        available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]
        if available_urls:
             total_running = sum(self.url_loads[url]['running'] for url in available_urls)
             total_waiting = sum(self.url_loads[url]['waiting'] for url in available_urls)
             total_load = sum(self.url_loads[url]['total'] for url in available_urls)
             print(f"Average load - Running: {total_running/len(available_urls):.1f}, Waiting: {total_waiting/len(available_urls):.1f}, Total load: {total_load/len(available_urls):.1f}")
        print(f"Available servers: {len(available_urls)}/{len(self.base_urls)}")
        print("========================\n")

    def _filter_think_tags(self, text: str) -> str:
        """Remove <think></think> tags and their content"""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    def _get_url_load(self, url: str) -> dict:
        """Get load information for a single URL"""
        try:
            # Build metrics endpoint URL
            if url.endswith('/v1'):
                base_url = url[:-3]  # Remove trailing '/v1'
            else:
                base_url = url.rstrip('/')
            metrics_url = f"{base_url}/metrics"
            
            response = requests.get(metrics_url, timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Parse Prometheus format metrics
                running = self._parse_metric_value(metrics_text, 'vllm:num_requests_running')
                waiting = self._parse_metric_value(metrics_text, 'vllm:num_requests_waiting')
                
                return {
                    'running': running,
                    'waiting': waiting,
                    'total': running + waiting,
                    'available': True
                }
        except Exception as e:
            # If retrieval fails, return default values
            pass
        
        return {'running': 0, 'waiting': 0, 'total': 0, 'available': False}
    
    def _parse_metric_value(self, metrics_text: str, metric_name: str) -> int:
        """Parse the value of specified metric from Prometheus format metrics text"""
        try:
            pattern = rf'^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.]+)'
            matches = re.findall(pattern, metrics_text, re.MULTILINE)
            if matches:
                return int(float(matches[0]))
        except Exception:
            pass
        return 0
    
    def _reload_urls_from_env(self):
        """Reload URL configuration from environment variables"""
        
        
        
        # Reload .env file
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)  # Force reload
        except ImportError:
            # If python-dotenv is not available, manually read .env file
            env_file = '.env'
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
        
        # Re-read URL configuration
        url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        new_urls = [url.strip() for url in url_env.split(',') if url.strip()]
        
        # Check if URLs have changed
        old_urls = set(self.base_urls)
        new_urls_set = set(new_urls)
        url_changed = old_urls != new_urls_set
        
        if url_changed:
            print(f"Detected URL configuration change: {self.base_urls} -> {new_urls}")
            
            # Remove URLs that are no longer used
            for url in old_urls - new_urls_set:
                if url in self.url_loads:
                    del self.url_loads[url]
                if url in self.url_weights:
                    del self.url_weights[url]
                if url in self.virtual_loads:
                    del self.virtual_loads[url]
            
            # Add new URLs
            for url in new_urls_set - old_urls:
                self.url_loads[url] = {'running': 0, 'waiting': 0, 'total': 0}
                self.url_weights[url] = 1.0
                self.virtual_loads[url] = 0
            
            # Update URL list
            self.base_urls = new_urls
            print(f"URL configuration updated, current configuration: {self.base_urls}")
        
        return url_changed
    
    def _update_loads_and_weights(self):
        """Update load information for all URLs and recalculate weights"""
        # Get current load for all URLs
        loads = []
        available_urls = []
        
        for url in self.base_urls:
            load_info = self._get_url_load(url)
            self.url_loads[url] = load_info
            
            if load_info['available']:
                loads.append(load_info['total'])
                available_urls.append(url)
        
        if not available_urls:
            return
        
        # Use new weight calculation algorithm
        if loads:
            # Calculate average load
            avg_load = sum(loads) / len(loads)
            # Calculate double average
            double_avg = avg_load * 2
            
            # Calculate weight for each URL: double average minus specific load value, treat negative as 0
            raw_weights = []
            for url in available_urls:
                current_load = self.url_loads[url]['total']
                weight = max(0, double_avg - current_load)
                raw_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(raw_weights)
            if total_weight > 0:
                for i, url in enumerate(available_urls):
                    self.url_weights[url] = raw_weights[i] / total_weight
            else:
                # If all weights are 0, distribute evenly
                for url in available_urls:
                    self.url_weights[url] = 1.0 / len(available_urls)
        else:
            # If no load data, distribute weights evenly
            for url in available_urls:
                self.url_weights[url] = 1.0 / len(available_urls)
        
        # Set weight to 0 for unavailable URLs
        for url in self.base_urls:
            if url not in available_urls:
                self.url_weights[url] = 0.0
    
    def _get_next_url(self) -> str:
        """Select URL with lowest load based on fill-the-gap algorithm"""
        # Get all available URLs
        available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]
        
        if not available_urls:
            # If no available URLs, wait for server recovery
            while True:
                # Reload URL configuration from .env
                self._reload_urls_from_env()
                    # Re-read environment variables and update URL configuration at the beginning of each batch
                self._update_loads_and_weights()
                

                # Re-check available URLs
                available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]
                
                if available_urls:
                    break
                
                time.sleep(10)
        
        # Find URL with lowest virtual load (fill-the-gap algorithm)
        min_virtual_load = float('inf')
        selected_url = available_urls[0]
        
        for url in available_urls:
            # Virtual load = actual load + allocated but unfinished requests
            virtual_load = self.url_loads[url]['total'] + self.virtual_loads[url]
            if virtual_load < min_virtual_load:
                min_virtual_load = virtual_load
                selected_url = url
        
        # After selecting URL, increase its virtual load
        self.virtual_loads[selected_url] += 1
        
        return selected_url

    def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message(self.system_message, "system")
            ] + message_list

        # Use different configurations based on enable_thinking
        if self.enable_thinking:
            payload = {
                "model": self.model,
                "messages": message_list,
                "temperature": 0.6,
                "top_p": 0.95,
                "presence_penalty": 1.5,
                "max_tokens": self.max_tokens,
                "top_k": 20,
                "min_p": 0,
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            }
        else:
            payload = {
                "model": self.model,
                "messages": message_list,
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 1.5,
                "max_tokens": self.max_tokens,
                "top_k": 20,
                "min_p": 0,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }

        trial = 0
        current_url = None
        while True:
            try:
                # Select URL
                current_url = self._get_next_url()
                response = requests.post(
                    f"{current_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                
                if content is None:
                    raise ValueError("VLLM service returned empty response, retrying...")
                
                if self.filter_think_tags:
                    content = self._filter_think_tags(content)
                
                # Request successful, decrease virtual load
                if current_url and current_url in self.virtual_loads:
                    self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)
                
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response_data.get("usage", {})},
                    actual_queried_message_list=message_list,
                )
            
            except requests.exceptions.RequestException as e:
                # Request failed, also decrease virtual load
                if current_url and current_url in self.virtual_loads:
                    self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)
                
                exception_backoff = min(2**trial, 300)  # Max wait time 300 seconds
                print(
                    f"Request failed (URL: {current_url}), retrying in {exception_backoff} seconds, attempt {trial}",
                    str(e),
                )
                if isinstance(e, requests.exceptions.ConnectionError):
                    print(f"Connection error: Please ensure VLLM service is running at {current_url}")
                    # On connection error, try to reload .env file and update URL configuration
                    print("Detected connection error, trying to reload .env file and update URL configuration...")
                    url_changed = self._reload_urls_from_env()
                    if url_changed:
                        print("URL configuration updated, will retry with new URL configuration")
                    else:
                        print("No change in URL configuration")
                    
                    # Re-detect load status of all servers (like initialization)
                    self._update_loads_and_weights()
                    
                    # Print average load information
                    available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]
                    if available_urls:
                         total_running = sum(self.url_loads[url]['running'] for url in available_urls)
                         total_waiting = sum(self.url_loads[url]['waiting'] for url in available_urls)
                         total_load = sum(self.url_loads[url]['total'] for url in available_urls)
                         print(f"Average load - Running: {total_running/len(available_urls):.1f}, Waiting: {total_waiting/len(available_urls):.1f}, Total load: {total_load/len(available_urls):.1f}")
                    
                    print(f"Load detection completed, current available servers: {len(available_urls)}/{len(self.base_urls)}")
                    print("================================\n")
                elif isinstance(e, requests.exceptions.Timeout):
                    print(f"Timeout error: Request exceeded {self.timeout} seconds")
                time.sleep(exception_backoff)
                trial += 1
                # Remove retry limit, implement infinite retry

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

def parse_json_to_dict(json_string: str) -> dict:
    """Parse JSON string, handle markdown format"""
    original_string = json_string
    
    # Method 1: Original matching approach - Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        # JSON parsing failed, but don't print details to avoid log pollution
        pass
        
        # Backup method: more lenient approach - extract any content that looks like JSON
        # New: try to fix double quote escaping issues in JSON
        try:
            # Find the first complete JSON object
            brace_count = 0
            start_idx = -1
            for i, char in enumerate(original_string):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        json_content = original_string[start_idx:i+1]
                        
                        # Try to fix double quote escaping issues
                        # Find all "explanation": "..." and "criteria_met": ... patterns
                        def fix_quotes_in_json(json_str):
                            # Use regex to fix double quotes in explanation field
                            import re
                            
                            # Match "explanation": "..." part
                            explanation_pattern = r'("explanation"\s*:\s*")(.*?)("(?:\s*,|\s*}))'
                            
                            def fix_explanation(match):
                                prefix = match.group(1)
                                content = match.group(2)
                                suffix = match.group(3)
                                
                                # Escape double quotes in content, but preserve already escaped ones
                                # First replace already escaped double quotes with temporary placeholder
                                content = content.replace('\\"', '###ESCAPED_QUOTE###')
                                # Escape unescaped double quotes
                                content = content.replace('"', '\\"')
                                # Restore already escaped double quotes
                                content = content.replace('###ESCAPED_QUOTE###', '\\"')
                                
                                return prefix + content + suffix
                            
                            fixed_json = re.sub(explanation_pattern, fix_explanation, json_str, flags=re.DOTALL)
                            return fixed_json
                        
                        fixed_json = fix_quotes_in_json(json_content)
                        return json.loads(fixed_json)
        except Exception as e:
            # Don't print details to avoid log pollution
            pass
        
        # Directly extract values of explanation and criteria_met fields
        try:
            # Extract explanation field value (content within quotes)
            explanation_pattern = r'"explanation"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
            explanation_match = re.search(explanation_pattern, original_string)
            explanation = explanation_match.group(1) if explanation_match else ""
            
            # Extract criteria_met field value (true or false)
            criteria_pattern = r'"criteria_met"\s*:\s*(true|false)'
            criteria_match = re.search(criteria_pattern, original_string)
            criteria_met = criteria_match.group(1) == 'true' if criteria_match else False
            
            # Construct JSON object
            result = {
                "explanation": explanation,
                "criteria_met": criteria_met
            }
            # Don't print detailed info to avoid log pollution
            return result
        except Exception as e:
            print(f"Field extraction exception: {e}")

        

        print("All JSON parsing methods failed, returning empty dict")
        print("="*80)
        return {}

def calculate_score(rubric_items: List[RubricItem], grading_response_list: List[dict]) -> float:
    """Calculate total score"""
    total_possible_points = sum(
        rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0
    )
    if total_possible_points == 0:
        return 0.0

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(rubric_items, grading_response_list)
        if grading_response["criteria_met"]
    )
    overall_score = achieved_points / total_possible_points
    return overall_score

def grade_single_example(
    prompt: List[Dict[str, str]], 
    response: str,
    rubric_items: List[RubricItem],
    grader_model,
    executor=None,  # New parameter: external thread pool
) -> Tuple[float, str, List[Dict]]:
    """Evaluate a single example

    Args:
        prompt: Conversation history
        response: Model response
        rubric_items: List of grading criteria
        grader_model: Grading model
        executor: External thread pool for global concurrency control

    Returns:
        tuple: (score, detailed explanation, grading results for each criterion)
    """
    # Build complete conversation
    convo_with_response = prompt + [dict(content=response, role="assistant")]
    
    def grade_rubric_item(rubric_item: RubricItem) -> dict:
        # Check if tags contain rule-based verification
        if rubric_item.tags:
            # Check for verifier:rule tag
            verifier_rule = any(tag.startswith("verifier:rule") for tag in rubric_item.tags)
            if verifier_rule:
                # Look for function and parameters tags
                function_name = None
                parameter_value = None
                
                for tag in rubric_item.tags:
                    if tag.startswith("function:"):
                        function_name = tag.split(":", 1)[1]
                    elif tag.startswith("parameters:"):
                        parameter_value = tag.split(":", 1)[1]
                
                # If math_verify function found, use rule-based verification
                if function_name == "math_verify" and parameter_value is not None:
                    try:
                        criteria_met = math_verify(response, parameter_value)
                        return {
                            "criteria_met": criteria_met,
                            "explanation": f"Rule-based verification using math_verify with parameter '{parameter_value}': {'PASS' if criteria_met else 'FAIL'}"
                        }
                    except Exception as e:
                        return {
                            "criteria_met": False,
                            "explanation": f"Rule-based verification failed: {e}"
                        }
        
        # If no rule-based verification or verification failed, use traditional LLM grading
        # Build conversation string, filter out system role messages
        convo_str = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in convo_with_response if m['role'] != 'system']
        )
        # Build grading prompt
        grader_prompt = GRADER_TEMPLATE.replace(
            "<<conversation>>", convo_str
        ).replace("<<rubric_item>>", str(rubric_item))
        messages = [dict(content=grader_prompt, role="user")]
        
        # Call grading model
        retry_count = 0
        max_retries = 10
        while retry_count < max_retries:
            sampler_response = grader_model(messages)
            # Get response_text from SamplerResponse object
            grading_response_dict = parse_json_to_dict(sampler_response.response_text)
            if "criteria_met" in grading_response_dict:
                label = grading_response_dict["criteria_met"]
                if label is True or label is False:
                    break
            # Grading failed, JSON output error, retrying (no detailed info to avoid log pollution)
            pass
            retry_count += 1
            
        # If retry limit reached, return failure result
        if retry_count >= max_retries:
            print(f"Grading failure limit reached ({max_retries} times), returning failure result")
            return {
                "criteria_met": False,
                "explanation": "Too many JSON parsing failures"
            }
            
        return grading_response_dict

    # Evaluate each criterion - only use external thread pool (single-layer global concurrency)
    if executor is not None:
        # Use external thread pool (global concurrency control)
        futures = [executor.submit(grade_rubric_item, rubric_item) for rubric_item in rubric_items]
        grading_response_list = [future.result() for future in futures]
    else:
        # If no external thread pool, execute sequentially (for single example testing)
        print("No external thread pool, executing sequentially")
        grading_response_list = [grade_rubric_item(rubric_item) for rubric_item in rubric_items]

    # Calculate total score
    overall_score = calculate_score(rubric_items, grading_response_list)

    # Generate detailed explanation
    rubric_items_with_grades = []
    readable_explanation_list = []
    for rubric_item, grading_response in zip(rubric_items, grading_response_list):
        explanation = grading_response.get("explanation", "No explanation provided")
        criteria_met = grading_response["criteria_met"]
        readable_explanation = (
            f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
        )
        readable_explanation_list.append(readable_explanation)
        rubric_items_with_grades.append(
            {
                **rubric_item.to_dict(),
                "criteria_met": criteria_met,
                "explanation": explanation,
            }
        )

    # Display in original rubric order
    readable_explanation_str = "\n\n".join(readable_explanation_list)
    readable_explanation_str = f"\n\n{readable_explanation_str}"

    return overall_score, readable_explanation_str, rubric_items_with_grades

def compute_score(data_source: str, solution_str: str, ground_truth: str = None, extra_info: Dict[str, Any] = None) -> float:
    """
    Calculate healthbench reward score

    Args:
        data_source: Dataset name (from DataProto.non_tensor_batch['data_source'])
        solution_str: Model generated response
        ground_truth: Not used
        extra_info: Contains prompt and reward_model information

    Returns:
        float: Reward score [0, 1]
    """
    
    try:
        # Check if extra_info is None
        if extra_info is None:
            return 0.0
            
        # Get data from extra_info
        prompt = extra_info.get("prompt", [])
        reward_model = extra_info.get("reward_model", {})
        rubrics = reward_model.get("rubrics", [])
        
        if not prompt or not rubrics:
            return 0.0
            
        # Rebuild rubrics
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        
        # Use VLLM as grading model
        grader = get_global_grader()  # Use global grader instance
        
        score, _, _ = grade_single_example(prompt, solution_str, rubric_items, grader)
        return score  # Already normalized score [0,1]
        
    except Exception as e:
        print(f"Error calculating reward score: {e}")
        return 0.0

def compute_score_batched(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], max_workers_per_url: int = MAX_CONCURRENT_WORKERS) -> List[float]:
    """
    Batch calculate reward scores for multiple responses

    Args:
        data_sources: List of dataset names
        solution_strs: List of model generated responses
        ground_truths: Not used
        extra_infos: List containing prompt and reward_model information
        max_workers_per_url: Concurrency per URL, defaults to MAX_CONCURRENT_WORKERS

    Returns:
        List[float]: List of reward scores [0, 1]
    """
    batch_data = list(zip(data_sources, solution_strs, ground_truths, extra_infos))
    return batch_compute_scores(batch_data, max_workers_per_url=max_workers_per_url)

# Global grader instance, avoid repeated creation
_global_grader = None

def get_global_grader():
    """Get or create global grader instance"""
    global _global_grader
    if _global_grader is None:
        _global_grader = VLLMSampler(
            max_tokens=2048,
            enable_thinking=False,
            filter_think_tags=True
        )
    return _global_grader

def batch_compute_scores(batch_data: List[Tuple[str, str, str, Dict[str, Any]]], max_workers_per_url: int = MAX_CONCURRENT_WORKERS) -> List[float]:
    """
    Batch calculate reward scores for multiple responses

    New concurrency control mechanism:
    - Expand all rubric grading tasks from all examples into independent tasks
    - Use global thread pool to handle all grading tasks, total concurrency = concurrency per URL × number of URLs
    - Compared to previous 32 example concurrency + 128 rubric concurrency per example,
      new mechanism provides more stable concurrency control, avoiding request fluctuations due to different rubric counts

    Args:
        batch_data: List, each item contains (data_source, solution_str, ground_truth, extra_info)
        max_workers_per_url: Concurrency per URL, defaults to MAX_CONCURRENT_WORKERS. Total requests = max_workers_per_url × number of URLs

    Returns:
        List[float]: List of reward scores
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    # Record start time
    start_time = time.time()
    
    # Statistics
    total_criteria = 0
    rule_based_criteria = 0
    llm_criteria = 0
    
    # Get global grader
    grader = get_global_grader()
    
    # Re-read environment variables from .env and update URL configuration at the start of each batch
    print(f"Batch start: Re-reading .env file and updating URL configuration...")
    url_changed = grader._reload_urls_from_env()
    if url_changed:
        print("URL configuration updated")
    else:
        print("URL configuration unchanged")
    
    # Re-get URL count (after re-reading URLs)
    url_count = len(grader.base_urls)
    
    # Re-update load information and weights
    grader._update_loads_and_weights()
    print(f"Load update complete, current available URL count: {len([url for url in grader.base_urls if grader.url_loads[url].get('available', False)])}")
    
    # Print average load information
    print("\nCurrent load information:")
    available_urls = [url for url in grader.base_urls if grader.url_loads[url].get('available', False)]
    if available_urls:
        total_running = sum(grader.url_loads[url].get('running', 0) for url in available_urls)
        total_waiting = sum(grader.url_loads[url].get('waiting', 0) for url in available_urls)
        total_load = sum(grader.url_loads[url].get('total', 0) for url in available_urls)
        print(f"Average load - Running: {total_running/len(available_urls):.1f}, Waiting: {total_waiting/len(available_urls):.1f}, Total load: {total_load/len(available_urls):.1f}")
    print(f"Available servers: {len(available_urls)}/{len(grader.base_urls)}")
    print()
    
    # Reset virtual load counters
    for url in grader.base_urls:
        grader.virtual_loads[url] = 0
    
    # Calculate actual total concurrency: concurrency per URL × number of URLs
    actual_max_workers = max_workers_per_url * url_count
    print(f"Configured {url_count} URLs, concurrency per URL: {max_workers_per_url}, total concurrency: {actual_max_workers}")
    
    # Collect all tasks that need grading
    all_grading_tasks = []
    task_to_sample_mapping = []  # Record which sample each task belongs to
    
    for sample_idx, (data_source, solution_str, ground_truth, extra_info) in enumerate(batch_data):
        if extra_info is None:
            continue
            
        prompt = extra_info.get("prompt", [])
        reward_model = extra_info.get("reward_model", {})
        rubrics = reward_model.get("rubrics", [])
        
        if not prompt or not rubrics:
            continue
            
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        
        # Create separate grading task for each rubric
        for rubric_idx, rubric_item in enumerate(rubric_items):
            task = {
                'sample_idx': sample_idx,
                'rubric_idx': rubric_idx,
                'prompt': prompt,
                'response': solution_str,
                'rubric_item': rubric_item,
                'rubric_items': rubric_items  # For final score calculation
            }
            all_grading_tasks.append(task)
            task_to_sample_mapping.append(sample_idx)
    
    def grade_single_rubric_task(task):
        """Evaluate single rubric task"""
        try:
            rubric_item = task['rubric_item']
            response = task['response']
            
            # Check if tags contain rule-based verification
            if rubric_item.tags:
                # Check for verifier:rule tag
                verifier_rule = any(tag.startswith("verifier:rule") for tag in rubric_item.tags)
                if verifier_rule:
                    # Look for function and parameters tags
                    function_name = None
                    parameter_value = None
                    
                    for tag in rubric_item.tags:
                        if tag.startswith("function:"):
                            function_name = tag.split(":", 1)[1]
                        elif tag.startswith("parameters:"):
                            parameter_value = tag.split(":", 1)[1]
                    
                    # If math_verify function found, use rule-based verification
                    if function_name == "math_verify" and parameter_value is not None:
                        try:
                            criteria_met = math_verify(response, parameter_value)
                            return {
                                'sample_idx': task['sample_idx'],
                                'rubric_idx': task['rubric_idx'],
                                'result': {
                                    "criteria_met": criteria_met,
                                    "explanation": f"Rule-based verification using math_verify with parameter '{parameter_value}': {'PASS' if criteria_met else 'FAIL'}"
                                },
                                'verification_type': 'rule'
                            }
                        except Exception as e:
                            return {
                                'sample_idx': task['sample_idx'],
                                'rubric_idx': task['rubric_idx'],
                                'result': {
                                    "criteria_met": False,
                                    "explanation": f"Rule-based verification failed: {e}"
                                },
                                'verification_type': 'rule'
                            }
            
            # If no rule-based verification or verification failed, use traditional LLM grading
            grader = get_global_grader()
            
            # Build complete conversation
            convo_with_response = task['prompt'] + [dict(content=task['response'], role="assistant")]
            
            # Build conversation string, filter out system role messages
            convo_str = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response if m['role'] != 'system']
            )
            
            # Build grading prompt
            grader_prompt = GRADER_TEMPLATE.replace(
                "<<conversation>>", convo_str
            ).replace("<<rubric_item>>", str(task['rubric_item']))
            messages = [dict(content=grader_prompt, role="user")]
            
            # Call grading model
            retry_count = 0
            max_retries = 10
            while retry_count < max_retries:
                sampler_response = grader(messages)
                grading_response_dict = parse_json_to_dict(sampler_response.response_text)
                if "criteria_met" in grading_response_dict:
                    label = grading_response_dict["criteria_met"]
                    if label is True or label is False:
                        return {
                            'sample_idx': task['sample_idx'],
                            'rubric_idx': task['rubric_idx'],
                            'result': grading_response_dict,
                            'verification_type': 'llm'
                        }
                # Grading failed, JSON output error, retrying (no detailed info to avoid log pollution)
                pass
                retry_count += 1
                
            # If retry limit reached, return failure result
            print(f"Grading failure limit reached ({max_retries} times), returning failure result")
            return {
                'sample_idx': task['sample_idx'],
                'rubric_idx': task['rubric_idx'],
                'result': {
                    "criteria_met": False,
                    "explanation": "Too many JSON parsing failures"
                }
            }
            
        except Exception as e:
            print(f"Grading task error: {e}")
            return {
                'sample_idx': task['sample_idx'],
                'rubric_idx': task['rubric_idx'],
                'result': {
                    "criteria_met": False,
                    "explanation": f"Grading error: {str(e)}"
                }
            }
    
    # Count total criteria
    total_criteria = len(all_grading_tasks)
    
    # Use global thread pool to handle all grading tasks
    sample_results = {}  # sample_idx -> List[grading_result]
    
    with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(grade_single_rubric_task, task): task for task in all_grading_tasks}
        
        # Collect results
        for future in as_completed(future_to_task):
            result = future.result()
            sample_idx = result['sample_idx']
            if sample_idx not in sample_results:
                sample_results[sample_idx] = {}
            sample_results[sample_idx][result['rubric_idx']] = result['result']
            
            # Count verification types
            verification_type = result.get('verification_type', 'llm')
            if verification_type == 'rule':
                rule_based_criteria += 1
            else:
                llm_criteria += 1
    
    # Calculate final score for each sample
    final_scores = []
    for sample_idx, (data_source, solution_str, ground_truth, extra_info) in enumerate(batch_data):
        try:
            if extra_info is None:
                final_scores.append(0.0)
                continue
                
            prompt = extra_info.get("prompt", [])
            reward_model = extra_info.get("reward_model", {})
            rubrics = reward_model.get("rubrics", [])
            
            if not prompt or not rubrics:
                final_scores.append(0.0)
                continue
                
            rubric_items = [RubricItem.from_dict(r) for r in rubrics]
            
            # Get all grading results for this sample
            if sample_idx in sample_results:
                grading_response_list = []
                for rubric_idx in range(len(rubric_items)):
                    if rubric_idx in sample_results[sample_idx]:
                        grading_response_list.append(sample_results[sample_idx][rubric_idx])
                    else:
                        # If a rubric has no result, use default failure result
                        grading_response_list.append({
                            "criteria_met": False,
                            "explanation": "Grading task not completed"
                        })
                
                # Calculate total score
                overall_score = calculate_score(rubric_items, grading_response_list)
                final_scores.append(overall_score)
            else:
                final_scores.append(0.0)
                
        except Exception as e:
            print(f"Error calculating score for sample {sample_idx}: {e}")
            final_scores.append(0.0)
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print statistics
    print("\n" + "="*60)
    print("Batch grading statistics:")
    print(f"Total criteria count: {total_criteria}")
    print(f"Rule-based verification count: {rule_based_criteria}")
    print(f"LLM as a Judge count: {llm_criteria}")
    print(f"Total time: {total_time:.2f} seconds")
    if total_criteria > 0:
        print(f"Average time per criterion: {total_time/total_criteria:.3f} seconds")
    print("="*60 + "\n")
    
    return final_scores

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Perform healthbench scoring on a single data item")
    
    # Get default grader model from environment variables
    
    #default_grader = "gpt-4.1"
    default_grader = "8001vllm"
    
    parser.add_argument("--grader-model", type=str, default=default_grader,
                      help="Grading model (e.g.: gpt-4.1 or 8001vllm)")
    args = parser.parse_args()

    # Choose appropriate sampler based on model name
    if "vllm" in args.grader_model.lower():
        grader = VLLMSampler(
            model=args.grader_model,
            max_tokens=2048,
            enable_thinking=False,  # Use no-think mode for fast response
            filter_think_tags=True  # Filter think tags to ensure clean JSON output
        )
    else:
        # Get OpenAI configuration from environment variables
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Using OpenAI model requires setting OPENAI_API_KEY environment variable")
            
        grader = ChatCompletionSampler(
            model=openai_model,
            system_message="You are a helpful assistant.",
            max_tokens=2048,
        )

    example_prompt = [
        {"role": "user", "content": "Doctor, I haven't been feeling well for the past two days. I started having a fever yesterday, and my temperature is 38.5 degrees. I also have a headache, sore throat, and runny nose. I'm worried it might be a cold. Do I need to take any medication?"}
    ]
    example_response = "Based on your symptoms - fever of 38.5 degrees, headache, sore throat, and runny nose - it does seem like cold symptoms. I recommend drinking plenty of warm water and getting rest. You can take some ibuprofen for the fever, and if your throat is very sore, you can try throat lozenges. Generally, a cold will improve on its own in 7-10 days. If you still have a high fever after 3 days or if symptoms worsen, you should see a doctor promptly."
    example_rubrics = [
        RubricItem(
            criterion="Correctly identify patient symptoms and provide reasonable preliminary assessment",
            points=8,
            tags=["clinical_assessment"]
        ),
        RubricItem(
            criterion="Provide appropriate self-care advice such as drinking water, rest, etc.",
            points=6,
            tags=["self_care_advice"]
        ),
        RubricItem(
            criterion="Advise when to seek medical care, providing clear timeframes or symptom indicators",
            points=7,
            tags=["safety_guidance"]
        ),
        RubricItem(
            criterion="Ask about symptom duration, severity, or other related symptoms",
            points=4,
            tags=["symptom_assessment"]
        ),
        RubricItem(
            criterion="Suggest monitoring temperature changes or other symptom changes",
            points=3,
            tags=["monitoring"]
        ),
        RubricItem(
            criterion="Ask about underlying conditions or allergy history",
            points=2,
            tags=["medical_history"]
        ),
        RubricItem(
            criterion="Recommend specific prescription medications such as antibiotics, ibuprofen, etc.",
            points=-2,
            tags=["inappropriate_prescription"]
        ),
        RubricItem(
            criterion="Provide clear diagnosis rather than symptom description",
            points=-5,
            tags=["overdiagnosis"]
        ),
        RubricItem(
            criterion="Suggest unnecessary tests or treatments",
            points=-3,
            tags=["overtreatment"]
        )
    ]

    score, explanation, details = grade_single_example(
        example_prompt,
        example_response,
        example_rubrics,
        grader
    )
    
    print("\nGrading results for each criterion:")
    for detail in details:
        print(json.dumps(detail, ensure_ascii=False, indent=2)) 
    print(f"Score: {score:.2f}")
    print("\nDetailed explanation:")
    print(explanation)