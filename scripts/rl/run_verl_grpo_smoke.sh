#!/bin/bash
# VERL GRPO smoke test - run directly on node42

cd /hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project

export QWEN_OMNI_SKIP_SPK=1
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export VLLM_MM_INPUT_CACHE_GIB=6
export CUDA_VISIBLE_DEVICES=5,6
export PYTHONPATH="verl:$PYTHONPATH"

source /home/s2025244189/miniconda3/etc/profile.d/conda.sh
conda activate qwen_echo

echo "[$(date)] VERL GRPO Smoke Test (2 GPUs)"
echo "  Model: output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078/merged_model"
echo "  Data:  dataJson/NAQA/EAQA_RL_smoke20.parquet"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=dataJson/NAQA/EAQA_RL_smoke20.parquet \
    data.val_files=dataJson/NAQA/EAQA_RL_smoke20.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=right \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=output/testResult/v9b-diverse-cot-2epoch-20260509-103234/v0-20260509-103317/checkpoint-3078/merged_model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.freeze_audio_encoder=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.limit_audios=8 \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_kwargs.id=multiturn_rl_6 \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=echo \
    trainer.experiment_name=verl_grpo_smoke \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.default_local_dir=output/verl_grpo_smoke

echo "[$(date)] VERL smoke test exited with code $?"
