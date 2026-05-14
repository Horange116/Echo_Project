# Task 15: vLLM 0.8.5 Singularity Runbook

日期: 2026-05-14

## 目标

把已经验证成功的容器路线固化成可复现实验入口：

- Ubuntu 22.04 / `glibc 2.35`
- `torch 2.6.0+cu124`
- `transformers 4.57.6`
- `huggingface-hub 0.36.0`
- `vllm 0.8.5`
- `Qwen2.5-Omni-7B`

## 已知成功条件

1. 容器镜像：
   - `docker://nvidia/cuda:12.4.0-devel-ubuntu22.04`
   - 本地 SIF：`output/singularity/nvidia_cuda_12.4.sif`
2. 容器内 `glibc`：
   - `2.35`
3. `gpu_memory_utilization`：
   - 默认 `0.6` 会在 KV cache 分配阶段 OOM
   - `0.85` 可通过

## 入口脚本

- [script/test7_vllm_085_singularity_smoke.sh](/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/script/test7_vllm_085_singularity_smoke.sh)

该脚本会：

1. 选择 `singularity` 或 `apptainer`
2. 检查：
   - SIF 是否存在
   - 容器内 Python 根目录是否存在
   - 模型与音频路径是否存在
3. 通过 `--nv` + bind mounts 进入容器
4. 复用：
   - `script/test6_vllm_downgrade_smoke.sh`
   - `scripts/vllm_qwen_omni_downgrade_smoke.py`

## 默认路径

- `PROJECT_ROOT=/home/s2025244189/s2025244265/Projects/Echo_Project`
- `MODEL_PATH=/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B`
- `AUDIO_PATH=/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/output/interleaved_tmp/round1_seg25.wav`
- `SIF_PATH=$PROJECT_ROOT/output/singularity/nvidia_cuda_12.4.sif`
- `CONTAINER_ROOT=$PROJECT_ROOT/output/singularity/miniconda3`
- `GPU_MEMORY_UTILIZATION=0.85`
- `VLLM_VERSION=0.8.5_singularity`

## 运行方式

直接在已分配 GPU 的环境里运行：

```bash
bash script/test7_vllm_085_singularity_smoke.sh
```

如果要用 SLURM 单卡：

```bash
srun -p A800Z --gres=gpu:1 --time=00:10:00 \
  bash script/test7_vllm_085_singularity_smoke.sh
```

## 结果文件

- log：
  - `output/vllm_downgrade_smoke/smoke_0.8.5_singularity.log`
- report：
  - `output/vllm_downgrade_smoke/report_0.8.5_singularity.json`

## 当前建议

后续如果要继续推进 vLLM rollout，不要优先回到宿主机 wheel 路线，优先从这条容器路线继续：

1. 保持同样的容器基础环境
2. 保持 `gpu_memory_utilization=0.85`
3. 在此基础上继续验证更贴近 rollout server 的最小服务形态
