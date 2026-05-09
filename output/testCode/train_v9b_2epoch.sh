#!/bin/bash
#SBATCH -J v9b_2epoch
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o slurm-v9b-2epoch-%j.out

cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/ms-swift/
export QWEN_OMNI_SKIP_SPK=1
export NPROC_PER_NODE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29507
export RANK=0
export NCCL_DEBUG=INFO
export TORCH_CHECKPOINT_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6

SRC_CKPT="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-clean-diverse-cot-20260508-212134/v0-20260508-212211/checkpoint-1539"
DST_CKPT="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-clean-diverse-cot-20260508-212134/v0-20260508-212211/checkpoint-1539-no-optim"

if [ ! -d "$DST_CKPT" ]; then
    cp -a "$SRC_CKPT" "$DST_CKPT"
fi
rm -f "$DST_CKPT"/optimizer.pt
rm -f "$DST_CKPT"/scheduler.pt
rm -f "$DST_CKPT"/scaler.pt
rm -f "$DST_CKPT"/rng_state*.pth

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testResult/v9b-diverse-cot-2epoch-${TIMESTAMP}"

swift sft \
    --model "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/" \
    --train_type lora \
    --dataset "/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/GeneratedData/eaqa_sft_v9_clean_diverse_cot.jsonl" \
    --torch_dtype float16 \
    --num_train_epochs 2 \
    --resume_from_checkpoint "$DST_CKPT" \
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
