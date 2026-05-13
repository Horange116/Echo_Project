#!/bin/bash
# ============================================================================
# VERL + vLLM GRPO Smoke Test — 单 GPU 最小验证
# ============================================================================
# 目的: 验证单 GPU 是否能绕开 torch 2.9.0 多 GPU 分布式初始化 SIGSEGV
# 完整论文风格训练脚本: script/stage2_multiturn_rl.sh
# 此脚本只做 smoke test，不做真实训练。
#
# 用法:
#   bash script/stage2_multiturn_rl_smoke.sh                          # 默认参数
#   MODEL_PATH=/path/to/model bash script/stage2_multiturn_rl_smoke.sh # 覆盖模型路径
#   末尾可追加 hydra 参数覆盖: bash script/stage2_multiturn_rl_smoke.sh trainer.total_epochs=2
# ============================================================================
set -x

# ---- 路径配置（可按需修改） ----
PROJECT_ROOT="${PROJECT_ROOT:-/home/s2025244189/s2025244265/Projects/Echo_Project}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_ROOT/dataJson/NAQA/EAQA_RL_smoke20.parquet}"
VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/dataJson/NAQA/EAQA_RL_smoke20.parquet}"
MODEL_PATH="${MODEL_PATH:-/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B}"

# ---- 安全检查 ----
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "ERROR: PROJECT_ROOT does not exist: $PROJECT_ROOT"
    exit 1
fi
if [ ! -f "$TRAIN_FILE" ]; then
    echo "ERROR: TRAIN_FILE does not exist: $TRAIN_FILE"
    exit 1
fi
if [ ! -f "$VAL_FILE" ]; then
    echo "ERROR: VAL_FILE does not exist: $VAL_FILE"
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH"
    exit 1
fi

cd "$PROJECT_ROOT"

# ---- 环境变量 ----
export HYDRA_FULL_ERROR=1
export VLLM_MM_INPUT_CACHE_GIB=6
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

# NCCL 稳定: 单 GPU 不需要 P2P/IB
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
unset ROCR_VISIBLE_DEVICES

# ---- 单 GPU Smoke GRPO ----
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.freeze_audio_encoder=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    +actor_rollout_ref.rollout.limit_audios=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_kwargs.id=multiturn_rl_6 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='echo' \
    trainer.experiment_name='smoke' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=999999 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    "$@"
