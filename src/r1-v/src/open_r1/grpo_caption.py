# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from math import fabs
import os
import re
import glob
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import json
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModel

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerCaption, Qwen2VLGRPOLoraTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import torch
from torchvision import transforms
from PIL import Image
from qwen_vl_utils import process_vision_info
import requests
# Save the original torch.load function
original_load = torch.load

# Create a patched function
def patched_load(f, *args, **kwargs):
    # Check if it is an RNG state file and weights_only=True is used
    if isinstance(f, str) and ('rng_state' in f or 'random_state' in f) and kwargs.get('weights_only', False):
        # Disable weights_only for RNG state files
        kwargs.pop('weights_only')
    return original_load(f, *args, **kwargs)

# Replace the torch.load function
torch.load = patched_load

# _SIGLIP_NAME = "/fs-computility/video/shared/hf_weight/siglip2-so400m-patch14-384"   # SigLIP ckpt path
# _siglip_processor = AutoProcessor.from_pretrained(_SIGLIP_NAME)

# def get_siglip_model():
#     local_rank = int(os.getenv("LOCAL_RANK", 0))
#     device = f"cuda:{local_rank}"
#     model = AutoModel.from_pretrained(_SIGLIP_NAME, torch_dtype=torch.bfloat16).eval().to(device)
#     return model

# _siglip_model = get_siglip_model()

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    semantic: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using semantic reward"},
    )
    visual_knowledge: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using visual knowledge reward"},
    )
    ngram_penalty: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using ngram penalty reward"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    ft_llm: Optional[str] = field(
        default="",
        metadata={"help": "whether finetuning LLM parameters"},
    )
    vision_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Apply LoRA only to vision-related parameters (freeze LLM parameters)"},
    )
    llm_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Apply LoRA only to LLM parameters (freeze vision parameters)"},
    )
    beta_strategy: Optional[str] = field(
        default="default",
        metadata={
            "help": "Strategy for beta scheduling (e.g., 'default', 'cosine')",
            "choices": ["default", "cosine"],
        },
    )
    visual_knowledge_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "Ratio of visual knowledge reward to accuracy reward"},
    )
    
def keywords_reward(content):
    good_words = ["start with", "starts with", "then", "next", "after", 
    "begin with", "begins with", "followed by", "following", 
    "subsequently", "thereafter", "later", "initially", 
    "first", "second", "third", "begin to", "begins to",
    "finally", "lastly", "to begin with", "as a starting point", 
    "in the beginning", "at the outset", "once", "when", 
    "whenever", "meanwhile", "simultaneously"]
    bad_words = ["possibly", "suggesting", "likely", "appears to", "appearing", 
    "designed to", "seems to", "might", "may", "could", 
    "potentially", "presumably", "probably", "perhaps", 
    "allegedly", "reportedly", "supposedly", "apparently", 
    "arguably", "tentatively", "hypothetically", "theoretically", 
    "in theory", "in some cases", "occasionally", "sometimes", 
    "rarely", "unclear", "ambiguous", "vague", "speculative", 
    "indicative of", "hinting at", "implying", "inferring", 
    "open to interpretation", "subject to change", "not necessarily"]

    reward = 0.0
    con_match = re.search(r"<answer>(.*?)(</answer>|$)", content, re.DOTALL)
    Res = con_match.group(1).strip() if con_match else ""
    if Res == "" or Res == " ":
        reward = 0.0
    else:
        for good in good_words:
            if good in Res.lower():
                reward += 0.2
        for bad in bad_words:
            if bad in Res.lower():
                reward = min(reward, 0.0)
                reward -= 0.2
        
        reward = min(reward, 0.4)

    return reward


def accuracy_reward(completions, solution, **kwargs):
    
    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []

    for content, sol in zip(contents, solution):
        try:
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    reward = 0.0
                else:
                    gt_number = normalize_number(gt_ans)
                    out_number = normalize_number(output_ans)
                    if gt_number is None or out_number is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "OCR":
                error_rate = wer(gt_ans, output_ans)
                reward = 1 - error_rate
                reward = max(0.0, min(1.0, reward))
            elif question_type == "free-form":
                score = compute_rouge_score(gt_ans, output_ans)
                reward = max(0.0, min(1.0, score))
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    reward = 0.0
                rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
                rel_diff = min(1.0, max(0.0, rel_diff))
                reward = 1 - rel_diff
            else:
                reward = 0.0
        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0
    
        rewards.append(reward)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
            
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<description>.*?</description>\s*<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"]
                           for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL)
               for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def visual_knowledge_reward(completions, solution, **kwargs):
    rewards = [0.0] * len(completions)  # Initialize rewards list
    contents = [completion[0]["content"] for completion in completions] 
    
    for i, (content, sol) in enumerate(zip(contents, solution)):
        problem = kwargs.get("problem", [""])[0].strip() if kwargs.get("problem") else ""
        description_match = re.search(r'<description>(.*?)</description>', content, re.DOTALL)
        gt_answer_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
        if not gt_answer_match or not description_match:
            rewards[i] = 0.0
            continue
        description_content = description_match.group(1).strip("\n").strip(" ")
        gt_answer = gt_answer_match.group(1).strip("\n").strip(" ")
        
        prompt = f"""
Text description: {description_content}
Question: {problem}
You are provided a text description of a problem and a question. Determine the answer to the question based on the text description. 
Provide only the single option letter (e.g., A, B, C, D, E, etc.) between the <answer> </answer> tags.
The output format should be: <answer> answer here </answer>.
"""
        ##! ToDo: change your own verifier model api
        model_url = "https://sd2t9cj25ni4n75na4290.apigateway-cn-beijing.volceapi.com/v1/chat/completions" #!To change your own model api
        api_key = "2ce8f136-861e-4ea9-8c30-5a57078d2ed8"
        payload = {
            "model": "Qwen2.5-VL-7B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                    # {"type": "image_url", "image_url": {"url": "<image_url>"}},
                    {"type": "text", "text": prompt}
                    ]
                }
            ]
        }
        response = requests.post(
                model_url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"}
        )
        try:
            pred_answer = json.loads(response.content)["choices"][0]["message"]["content"]
            pred_match = re.search(r'<answer>(.*?)</answer>', pred_answer, re.DOTALL)
            if not pred_match:
                rewards[i] = 0.0
                continue
            pred_answer = pred_match.group(1).strip("\n").strip(" ")

            if pred_answer == gt_answer:
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0
        except Exception as e:
            print(f"Error in visual_knowledge_reward: {e}")
            rewards[i] = 0.0

    for i in range(len(rewards)):
        rewards[i] = rewards[i] * kwargs.get("visual_knowledge_ratio", 1.0)
    return rewards

def ngram_penalty_reward(completions, **kwargs):
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    rewards = [0.0] * len(completions)  # Initialize rewards list
    contents = [completion[0]["content"] for completion in completions] 

    for i, content in enumerate(contents):
        if content.strip() == "" or len(content.split()) < kwargs.get("ngram_size", 20):
            rewards[i] = 0.0
            continue
        ngrams = set()
        total = 0
        for ng in zipngram(content, kwargs.get("ngram_size", 20)):
            ngrams.add(ng)
            total += 1
        scaling = 1 - len(ngrams) / total
        reward = scaling * kwargs.get("ngram_max_penalty", -2.0)
        rewards[i] = reward

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "visual_knowledge": visual_knowledge_reward,
    "ngram_penalty": ngram_penalty_reward,
}

SYSTEM_PROMPT = (
    "You are tasked with analyzing an image/video to generate an exhaustive and detailed description. "
    "Your goal is to extract and describe all possible information from the image/video, including but not limited to objects, "
    "numbers, text, and the relationships between these elements. The description should be as fine and detailed as possible, "
    "capturing every nuance. After generating the detailed description, you need to analyze it and provide step-by-step "
    "detailed reasoning for the given question based on the information. Finally, provide a single word or phrase answer "
    "to the question. The description, reasoning process and answer are enclosed within <info> </info>, <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., <info> image/video description here </info> <think> reasoning process here "
    "</think> <answer> answer here </answer>"
)

def find_latest_checkpoint(output_dir):
    """
    Search for checkpoint-* directories in the output path and return the latest one for resuming.
    """
    if not output_dir or not os.path.exists(output_dir):
        return None
    
    # Search all checkpoint-* directories
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # Sort by the numeric suffix in the directory name to find the latest one
    def extract_step_number(checkpoint_path):
        # Extract the step number from the path, e.g., get 1000 from "checkpoint-1000"
        match = re.search(r'checkpoint-(\d+)', checkpoint_path)
        if match:
            return int(match.group(1))
        return 0
    
    # Sort by step number and take the largest (latest) one
    latest_checkpoint = max(checkpoint_dirs, key=extract_step_number)
    
    print(f"Found checkpoint directories: {checkpoint_dirs}")
    print(f"Using the latest checkpoint to resume: {latest_checkpoint}")
    
    return latest_checkpoint


def find_all_checkpoints_desc(output_dir):
    """
    Return every checkpoint-* directory under output_dir sorted by step descending.

    Args:
        output_dir (str): Output directory path

    Returns:
        list[str]: Checkpoint directories ordered from newest to oldest (may be empty)
    """
    if not output_dir or not os.path.exists(output_dir):
        return []

    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    if not checkpoint_dirs:
        return []

    def extract_step_number(checkpoint_path):
        match = re.search(r'checkpoint-(\d+)', checkpoint_path)
        if match:
            return int(match.group(1))
        return 0

    # Sort from largest to smallest step
    checkpoint_dirs.sort(key=extract_step_number, reverse=True)
    return checkpoint_dirs

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    if script_args.semantic: # whether using semantic reward
        reward_funcs.append(reward_funcs_registry["semantic"])
    if script_args.visual_knowledge: # whether using caption reward
        reward_funcs.append(reward_funcs_registry["visual_knowledge"])
    if script_args.ngram_penalty: # whether using ngram penalty reward
        reward_funcs.append(reward_funcs_registry["ngram_penalty"])

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "caption": ""
    }

    QUESTION_TEMPLATE = f"""
You are tasked with analyzing an video to generate a detailed description to help you answer the question. First analyze the video and produce a self-contained description—detailed enough that can lead to the correct answer. Wrap the entire description between <description> </description> tags.
Next, engage in an internal dialogue and include self-reflection or verification in your reasoning process. Provide your detailed, step-by-step reasoning based on the video description information and video, and enclose this part between <think> </think> tags.
Finally, provide only the single option letter (e.g., A, B, C, D, E, etc.) between the <answer> </answer> tags.
The output format should be: <description> video description here </description><think> reasoning process here </think><answer> answer here </answer>.
"""

    def make_conversation_image(example):        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
        
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
        
    def make_conversation_image_and_video(example):
        question = example['problem']
        
        msg ={
            "prompt": 
               [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                        },
                        {
                            "type": "text",
                            "text": question + "\n" + QUESTION_TEMPLATE
                        }
                        ]
                }]
            }
        
        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    
    # Ensure last partial batch is dropped to keep per-device grouping consistent
    try:
        training_args.dataloader_drop_last = True
    except Exception:
        # Be tolerant if the config object doesn’t expose this attribute
        pass

    trainer_cls = Qwen2VLGRPOLoraTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerCaption
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        # Use the explicitly provided checkpoint if available
        checkpoint = training_args.resume_from_checkpoint
        print(f"Using the specified checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        # Try to resume from checkpoints in descending order; start fresh if all attempts fail
        all_ckpts = find_all_checkpoints_desc(training_args.output_dir)
        resumed = False
        if all_ckpts:
            print(f"Found {len(all_ckpts)} checkpoints; attempting to resume from newest to oldest")
            for ckpt in all_ckpts:
                try:
                    print(f"Attempting to resume training from {ckpt}...")
                    trainer.train(resume_from_checkpoint=ckpt)
                    resumed = True
                    print(f"Successfully resumed training from {ckpt}")
                    break
                except Exception as e:
                    print(f"Failed to resume from {ckpt} due to error: {e}. Trying the next older checkpoint...")
        if not resumed:
            print("Could not resume from any checkpoint; starting a new training run")
            trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)