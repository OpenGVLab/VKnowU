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

import os
import re
import glob
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModel

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
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

_SIGLIP_NAME = "/fs-computility/video/shared/hf_weight/siglip2-so400m-patch14-384"   # SigLIP ckpt path
_siglip_processor = AutoProcessor.from_pretrained(_SIGLIP_NAME)

def get_siglip_model():
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    model = AutoModel.from_pretrained(_SIGLIP_NAME, torch_dtype=torch.bfloat16).eval().to(device)
    return model

_siglip_model = get_siglip_model()

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
        default=True,
        metadata={"help": "whether using semantic reward"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    beta_strategy: Optional[str] = field(
        default="default",
        metadata={
            "help": "Strategy for beta scheduling (e.g., 'default', 'cosine')",
            "choices": ["default", "cosine"],
        },
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
            elif question_type in ['DarkEventInfer', 'MixVidQA']:
                question = kwargs["problem"][0].strip()
                response = requests.post(
                    "http://localhost:5000/predict",
                    json={"content": content, # model_genertion
                        "sol": sol, # GT
                        "problem_type": question_type,
                        "problem": question}
                )
                reward = response.json()["output"]
            elif question_type == "caption":
                response = requests.post(
                    "http://localhost:5000/predict",
                    json={"content": content, # model_genertion
                        "sol": sol, # GT
                        "problem_type": question_type}
                )
                recall = response.json()["recall"]
                precision = response.json()["precision"]
                keywords_r = keywords_reward(content)
                reward = recall + 0.5*precision + keywords_r
            elif question_type in ['MER2025']:
                pass
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
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def semantic_reward(completions, data_type, path, w_scale=2.0, **kwargs):
    """Reward function that computes semantic similarity between image/video and text using SigLIP."""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = [0.0] * len(completion_contents)
    pattern = r"<think>(.*?)</think>"
    think_contents = [
        (lambda s: s[s.find('.')+1:].lstrip() if '.' in s else s)
        (re.search(pattern, content, re.DOTALL).group(1)) 
        if re.search(pattern, content, re.DOTALL) else content
        for content in completion_contents
    ]

    if data_type[0] == "image":
        image_path = path[0]
        image = Image.open(image_path).convert("RGB")
        inputs = _siglip_processor(images=[image] * len(think_contents),
                                   text=think_contents,
                                   return_tensors="pt",
                                   truncation=True,
                                   padding="max_length",
                                   max_length=64).to(_siglip_model.device)
        with torch.no_grad():
            outputs = _siglip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            similarities = (image_embeds * text_embeds).sum(dim=-1)
            similarities_rewards = torch.min(torch.ones_like(similarities), w_scale * torch.max(similarities, torch.zeros_like(similarities)))
            rewards = similarities_rewards.tolist()

    elif data_type[0] == "video":
        # try:
        video_path = path[0]
        video_info = {"video": video_path}
        _, video_inputs, _ = process_vision_info([{"content": [video_info]}], return_video_kwargs=True)
        video_frames = video_inputs[0]
        
        all_frame_similarities = []
        for frame in video_frames:
            # Convert frame to PIL Image if it's a tensor
            if isinstance(frame, torch.Tensor):
                frame = transforms.ToPILImage()(frame)

            # Process frame with all texts
            inputs = _siglip_processor(images=[frame] * len(think_contents),
                                        text=think_contents,
                                        return_tensors="pt",
                                        truncation=True,
                                        padding="max_length",
                                        max_length=64).to(_siglip_model.device)
            with torch.no_grad():
                outputs = _siglip_model(**inputs)
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                similarities = (image_embeds * text_embeds).sum(dim=-1)
                similarities_rewards = torch.min(torch.ones_like(similarities), w_scale * torch.max(similarities, torch.zeros_like(similarities)))
                all_frame_similarities.append(similarities_rewards)
                
        all_frame_similarities = torch.stack(all_frame_similarities)
        mean_similarities = all_frame_similarities.mean(dim=0)
        rewards = mean_similarities.tolist()          

        # except Exception as e:
        #     print(f"Error in semantic_reward: {e}")
        #     rewards = [0.0] * len(think_contents)

    return rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "semantic": semantic_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def find_latest_checkpoint(output_dir):
    """
    在输出目录中搜索checkpoint-*文件夹，返回最近的一个用于resume
    """
    if not output_dir or not os.path.exists(output_dir):
        return None
    
    # 搜索所有checkpoint-*文件夹
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    
    if not checkpoint_dirs:
        return None
    
    # 按文件夹名称中的数字排序，找到最新的
    def extract_step_number(checkpoint_path):
        # 从路径中提取step数字，例如从"checkpoint-1000"提取1000
        match = re.search(r'checkpoint-(\d+)', checkpoint_path)
        if match:
            return int(match.group(1))
        return 0
    
    # 按step数字排序，取最大的（最新的）
    latest_checkpoint = max(checkpoint_dirs, key=extract_step_number)
    
    print(f"找到checkpoint文件夹: {checkpoint_dirs}")
    print(f"使用最新的checkpoint进行resume: {latest_checkpoint}")
    
    return latest_checkpoint


def find_all_checkpoints_desc(output_dir):
    """
    返回 output_dir 下所有形如 checkpoint-* 的目录，按 step 从大到小排序。

    Args:
        output_dir (str): 输出目录路径

    Returns:
        list[str]: 按从新到旧排序的 checkpoint 目录列表（可能为空）
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

    # 从大到小排序
    checkpoint_dirs.sort(key=extract_step_number, reverse=True)
    return checkpoint_dirs

def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    if script_args.semantic: # whether using semantic reward
        reward_funcs.append(reward_funcs_registry["semantic"])

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
        "DarkEventInfer": "",
        "MixVidQA": "",
        "MER2025": "",
        "caption": ""
    }

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
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
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

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
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
        # 如果明确指定了checkpoint，使用指定的
        checkpoint = training_args.resume_from_checkpoint
        print(f"使用指定的checkpoint: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        # 自动按从新到旧依次尝试恢复，全部失败则从头训练
        all_ckpts = find_all_checkpoints_desc(training_args.output_dir)
        resumed = False
        if all_ckpts:
            print(f"发现 {len(all_ckpts)} 个checkpoint，按从新到旧依次尝试恢复")
            for ckpt in all_ckpts:
                try:
                    print(f"尝试从 {ckpt} 恢复训练……")
                    trainer.train(resume_from_checkpoint=ckpt)
                    resumed = True
                    print(f"成功从 {ckpt} 恢复训练")
                    break
                except Exception as e:
                    print(f"从 {ckpt} 恢复失败，错误：{e}，尝试下一个较旧的checkpoint…")
        if not resumed:
            print("未能从任何checkpoint恢复，开始新训练")
            trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)