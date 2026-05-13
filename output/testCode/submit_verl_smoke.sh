#!/bin/bash
# VERL+vLLM single-GPU smoke test
#SBATCH -J verl_smoke
#SBATCH -p A800Z
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=qmultiple9
#SBATCH -o /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/testCode/verl-smoke-%j.out

echo "SCRIPT_START $(date) host=$(hostname)" >&2

cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project || {
    echo "FAIL: cannot cd to project" >&2
    exit 1
}

echo "[$(date)] VERL smoke test on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

export PYTHONPATH="verl:$PYTHONPATH"
unset ROCR_VISIBLE_DEVICES

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export HYDRA_FULL_ERROR=1
export VLLM_MM_INPUT_CACHE_GIB=6

echo "[$(date)] Launching VERL..."
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/dataJson/NAQA/EAQA_RL_smoke20.parquet" \
    data.val_files="/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/dataJson/NAQA/EAQA_RL_smoke20.parquet" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B" \
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
    'trainer.logger=[console]' \
    trainer.project_name=echo \
    trainer.experiment_name=smoke \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=999999 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=1

echo "[$(date)] Done. Exit code: $?"
