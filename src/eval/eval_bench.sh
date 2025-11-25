#!/bin/bash
# run_models.sh

model_paths=(
    "your_qwen_model_path"
)

export DECORD_EOF_RETRY_MAX=40960 

for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    model_name=$(basename $model)
    nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 QWENVL_EVAL=True python ./src/eval/eval_bench.py --model_path "$model" --mode "SEE_THINK" > eval_${model_name}.log 2>&1 &
done
