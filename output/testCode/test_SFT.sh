#!/bin/bash
#SBATCH -J test_SFT
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o slurm-%j.out

cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/ms-swift/
#这一行意味着跳过正常的tts依赖
export QWEN_OMNI_SKIP_SPK=1
export NPROC_PER_NODE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export NCCL_DEBUG=INFO
export TORCH_CHECKPOINT_SERIALIZATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4   


python - <<EOF
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("device count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name =", torch.cuda.get_device_name(0))
EOF

SRC_CKPT="/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/testResult/v7-20260505-145145/checkpoint-749"
DST_CKPT="/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/testResult/v7-20260505-145145/checkpoint-749-no-optim"

if [ ! -d "$DST_CKPT" ]; then
    cp -a "$SRC_CKPT" "$DST_CKPT"
fi

rm -f "$DST_CKPT"/optimizer.pt
rm -f "$DST_CKPT"/scheduler.pt
rm -f "$DST_CKPT"/scaler.pt
rm -f "$DST_CKPT"/rng_state*.pth

swift sft \
    --model "/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B/" \
    --train_type lora \
    --dataset "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/GeneratedData/eaqa_sft_train_clean_strict.jsonl" \
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
    --output_dir "/hpai/aios3.0/private/user/s2025244189/s2025244265/Echo_Project/output/testResult/" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
