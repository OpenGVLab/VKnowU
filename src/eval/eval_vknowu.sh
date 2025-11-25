#!/bin/bash
model_paths=(
    "your_qwen_model_path"
)

export DECORD_EOF_RETRY_MAX=40960 

for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    model_name=$(basename $model)
    env CUDA_VISIBLE_DEVICES=0,1,2,3 QWENVL_EVAL=True python ./src/eval/eval_vknowu.py --model_path "$model" --mode "SEE_THINK"
done
