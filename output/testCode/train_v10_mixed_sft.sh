#!/bin/bash
#SBATCH -J v10_mixed_sft
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o slurm-v10-%j.out

cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/ms-swift/
export QWEN_OMNI_SKIP_SPK=1
export NPROC_PER_NODE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29502
export RANK=0
export NCCL_DEBUG=INFO
export TORCH_CHECKPOINT_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=5

/home/s2025244189/miniconda3/envs/qwen_echo/bin/python - <<'EOF'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("device count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name =", torch.cuda.get_device_name(0))
    print("memory =", torch.cuda.get_device_properties(0).total_mem / 1024**3, "GiB")
EOF

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v10-mixed-70diverse-30clean-${TIMESTAMP}"

swift sft \
    --model "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/" \
    --train_type lora \
    --dataset "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/eaqa_sft_v10_mixed_70diverse_30clean.jsonl" \
    --torch_dtype float16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-6 \
    --gradient_checkpointing true \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 100 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "${OUTPUT_DIR}" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
