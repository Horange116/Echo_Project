# Task 16: vLLM 0.8.5 Server Next Step

日期: 2026-05-14

## 目的

在已经验证成功的 Singularity 容器路线基础上，补一个更贴近 Echo rollout server 的最小服务入口。

这一步不等于完整 rollout，只是把：

- `import vllm`
- `LLM load Qwen2.5-Omni`
- 单音频 inference

进一步推进到：

- OpenAI-compatible vLLM server 可启动

## 入口脚本

- [script/test8_vllm_085_singularity_server.sh](/hpai/aios3.0/private/user/s2025244189/s2025244265/Projects/Echo_Project/script/test8_vllm_085_singularity_server.sh)

## 默认配置

- 容器镜像：`output/singularity/nvidia_cuda_12.4.sif`
- 模型路径：`/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B`
- 端口：`8000`
- `gpu_memory_utilization=0.85`
- `max_model_len=32768`
- `served_model_name=qwen2_5_omni_7b`

## 运行方式

在已分配 GPU 的环境中直接运行：

```bash
bash script/test8_vllm_085_singularity_server.sh
```

或者通过 SLURM 单卡：

```bash
srun -p A800Z --gres=gpu:1 --time=00:15:00 \
  bash script/test8_vllm_085_singularity_server.sh
```

## 预期结果

如果这一步成功，应该至少能看到：

- vLLM OpenAI API server 监听启动
- 模型成功加载
- 日志稳定，不在 engine 初始化阶段退出

日志文件：

- `output/vllm_server/vllm_085_singularity_server.log`

## 与 rollout 的关系

这一步只是服务启动，不代表：

- 多轮 interleaved controller 已经接上
- 裁剪音频插入逻辑已经服务化
- VERL rollout 已经切换完成

但它是一个非常自然的下一跳，因为：

1. 可以证明容器方案不只是 smoke 可用，而是 server 级别也能稳定加载
2. 后续可以再接：
   - 一个最小 client 请求
   - 一个最小多模态请求
   - 再往上才是 interleaved controller / rollout server
