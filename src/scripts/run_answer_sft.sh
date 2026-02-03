cd src/r1-v
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

export QWENVL_EVAL=False

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_SEE_THINK.py \
    --output_dir "/mnt/shared-storage-user/jiangtianxiang/VKBench/output/Answer-SFT" \
    --model_name_or_path "/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5" \
    --dataset_name /mnt/shared-storage-user/jiangtianxiang/VKBench/VKnowQA/VKnowQA-CS-12K.json \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name Answer-SFT \
    --save_steps 5000 \
    --max_grad_norm 5 \
    --save_only_model true \