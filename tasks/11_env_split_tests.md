# 11. 环境拆分测试：训练 vs 推理

## 背景

截至 2026-05-13，所有 VERL + Qwen2.5-Omni-7B 的测试均以 SIGSEGV 失败：

| # | GPU | 框架 | 环境 | Crash 位置 |
|---|-----|------|------|-----------|
| 1-6 | 2 | FSDP (VERL) | torch 2.9 + vLLM 0.12 | `actor_rollout_wg.init_model()` |
| 7-8 | 2 | DeepSpeed ZeRO-2/3 | torch 2.9 + vLLM 0.12 | `deepspeed.initialize()` |
| 9 | 1 | FSDP (VERL) | torch 2.9 + vLLM 0.12 | `ref_policy_wg.init_model()` |

问题已缩小到两个可能来源：

1. **vLLM 0.12 强制绑定 torch 2.9.0** → 环境污染，导致 FSDP/DeepSpeed 的 NCCL/模型包装崩溃
2. **Qwen2.5-Omni 模型本身与 VERL/FSDP 不兼容** → 无论 torch 版本，init_model 阶段都会崩溃

为了区分这两个来源，需要拆分训练环境和推理环境分别测试。

---

## 测试 3: Clean Torch VERL/FSDP Init Smoke

### 目的
在**不安装 vLLM 0.12** 的独立 conda 环境中，验证 VERL/FSDP 是否能成功初始化 Qwen2.5-Omni。

### 环境要求
- 不装 vLLM 0.12
- 推荐 torch 2.5.x 或 2.6.x
- 需要: transformers, peft, ray, hydra-core
- rollout 使用 HF（不用 vLLM），排除 vLLM 干扰
- KL loss 关闭，freeze audio encoder，减少变量

### 文件
- `script/test3_clean_torch_verl_fsdp_init.sh`

### 运行命令
```bash
conda activate <clean_training_env>
cd /home/s2025244189/s2025244265/Projects/Echo_Project
bash script/test3_clean_torch_verl_fsdp_init.sh
```

### 关键配置差异（vs 完整 VERL 脚本）
| 参数 | 测试 3 | 完整 VERL |
|------|--------|----------|
| rollout.name | **hf** | vllm |
| freeze_audio_encoder | **True** | False |
| use_kl_loss | **False** | True |
| gradient_checkpointing | **False** | True |
| max_response_length | **256** | 2048 |
| test_freq | **999999** | 10 |

---

## 测试 4: vLLM-Only Inference Smoke

### 目的
在**当前 vLLM 0.12 + torch 2.9** 环境中，只测试 Qwen2.5-Omni 的 vLLM 推理，不触发 VERL/FSDP/DeepSpeed/训练/backward。

### 环境要求
- vLLM 0.12 + torch 2.9（当前 qwen_echo 环境即可）
- 不需要 VERL、FSDP、DeepSpeed

### 文件
- `script/test4_vllm_only_inference.sh` — bash 包装脚本
- `scripts/vllm_qwen_omni_audio_smoke_minimal.py` — 最小 vLLM 推理 Python 脚本

注意：仓库中已有完整版 `scripts/rl_backends/vllm_qwen_omni_audio_smoke.py`（6 项测试），
但该脚本之前已测试失败（Engine core initialization failed）。test4 使用更简化的 minimal 版本。

### 运行命令
```bash
conda activate qwen_echo
cd /home/s2025244189/s2025244265/Projects/Echo_Project
bash script/test4_vllm_only_inference.sh

# 覆盖模型/音频路径
MODEL_PATH=/path/to/model AUDIO_PATH=/path/to/audio.wav bash script/test4_vllm_only_inference.sh
```

### minimal 脚本功能
- 加载 vLLM LLM 模型（Qwen2.5-Omni）
- 加载单个音频文件
- 使用 `<|audio_bos|><|AUDIO|><|audio_eos|>` 格式构建 multimodal prompt
- temperature=1.0, max_tokens=256, stop=["</seg>", "</answer>"]
- 输出 response 文本和 stop_reason
- 写入报告: `output/vllm_smoke/test4_vllm_only_inference_report.json`

---

## 结果判读表

| 测试 3 (clean train) | 测试 4 (vLLM only) | 结论 |
|---|---|---|
| **通过** | **通过** | 最理想：训练和推理拆环境可行。分别维护 clean torch env + vLLM env。 |
| **通过** | **失败** | VERL/FSDP 可训练 Qwen2.5-Omni，但 vLLM 0.12 不适合该模型 rollout。考虑 custom GRPO + 其他推理后端。 |
| **失败** | **通过** | VERL/FSDP + Qwen2.5-Omni 本身不兼容（与 torch/vLLM 版本无关）。走 custom GRPO + vLLM standalone server 路线。 |
| **失败** | **失败** | 当前版本组合整体不可用。保留 custom HF rollout baseline（Section 30-31），等待上游修复。 |

---

## 注意事项

1. **测试 3 需要创建新的 conda 环境**：当前 `qwen_echo` 环境安装了 vLLM 0.12 + torch 2.9。需要新建一个只装训练依赖的环境（如 `qwen_echo_train`），torch 版本选 2.5.x 或 2.6.x。

2. **测试 4 使用现有环境**：直接在 `qwen_echo` 环境运行，不需要额外环境。

3. **两个测试都是单 GPU**（`n_gpus_per_node=1`），不需要多 GPU 资源。

4. **测试 3 的 MODEL_PATH 默认值可能不存在**：
   - `$PROJECT_ROOT/output/sft_v9b_merged` 不存在时脚本会报错退出
   - 可以用基座模型路径覆盖：`MODEL_PATH=/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B bash script/test3_clean_torch_verl_fsdp_init.sh`

5. **测试 4 的 AUDIO_PATH 会自动搜索**：如果未设置，脚本会在 `output/`、`dataJson/`、`mnt/` 下自动找 wav/flac/mp3 文件。

6. **不删除、不修改已有脚本**：所有新文件都是独立新增的。
