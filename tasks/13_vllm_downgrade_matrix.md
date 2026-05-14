# Task 13: vLLM 降级兼容性矩阵（torch 2.6.0）

日期: 2026-05-13

## 目标

寻找一个与 `torch 2.6.0` 共存、并且能加载 `Qwen2.5-Omni` 的 `vLLM` 版本，用于恢复论文路线中的 vLLM rollout。

## 原因

当前已知结论：

1. `torch 2.6.0 + VERL/FSDP + Qwen2.5-Omni` 已能完成模型加载、FSDP wrap、进入 `trainer.fit()`
2. `vLLM 0.12.0` 强依赖 `torch 2.9.0`
3. `torch 2.9.0 + Qwen2.5-Omni` 在当前环境中会触发 `SIGSEGV`

因此需要反向寻找更低版本的 `vLLM`，确认是否存在可与 `torch 2.6.0` 兼容的 wheel 版本；若 wheel 路线全部失败，再考虑 source build。

## 约束

- 不污染当前可用的 `torch 2.6.0` VERL 训练环境
- 所有测试都在新的 conda env 中进行
- 只做 `vLLM import / registry / LLM load / 单音频 inference`
- 不接 VERL
- 不做训练
- 不做 FSDP
- 不做 DeepSpeed
- 如果某个 `vLLM` wheel 安装会强制升级 `torch`，立即判定该环境失败，不继续在该环境内修补

## 候选版本顺序

优先测试以下版本：

1. `vllm==0.8.5` 或相邻 `0.8.x`
2. `vllm==0.8.4`
3. `vllm==0.8.3`
4. `vllm==0.7.3`
5. `vllm==0.7.2`
6. `vllm==0.6.6`
7. `vllm==0.6.4.post1` 或 `vllm==0.6.4`

暂不从 `0.5.x` 以下开始，因为依赖分歧会显著增加，优先级低于 `0.6.x ~ 0.8.x`。

## 每个版本的记录项

每个版本都需要记录以下字段：

- `python version`
- `torch version`
- `cuda wheel`
- `vllm version`
- `transformers version`
- `import vllm` 是否成功
- `vLLM registry` 检查 `Qwen2.5-Omni` 是否触发 `SIGSEGV`
- `LLM(model=MODEL_PATH, trust_remote_code=True)` 是否成功
- 单音频 inference 是否成功
- 是否支持 `stop=["</seg>", "</answer>"]`
- 是否支持 LoRA
- 失败日志

## 测试阶段定义

### Stage 0: Imports

验证 `torch / transformers / vllm` 是否可导入，并记录版本。

### Stage 1: Registry Check

检查 `vllm.model_executor.models.registry` 是否能正常运行。

重点观察：

- 是否直接在 registry 阶段崩溃
- 是否在 `Qwen2_5OmniModel` 检查阶段触发 `SIGSEGV`

### Stage 2: LLM Load

测试：

```python
from vllm import LLM

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
)
```

### Stage 3: Single Audio Inference

优先测试 text-only prompt 验证 `LLM.load + generate` 基本功能，再单独测试 audio 输入。

注意：

- text-only 成功不等于 audio 成功
- report 中必须分开记录

### Stage 4: Stop Words

验证 `SamplingParams(stop=["</seg>", "</answer>"])` 是否可用，并记录停止行为。

### Stage 5: LoRA Capability Check

只做能力探测，不做真正 LoRA rollout：

- 检查当前 `vLLM` 版本是否暴露 `LoRARequest` 等接口
- 不要求本阶段真正成功加载 adapter

## 判定标准

- 如果某个版本通过 `LLM load + 单音频 inference`，则该版本进入下一步 remote rollout server 设计
- 如果所有 wheel 版本都失败，但 `torch 2.6.0` 主环境本身可用，则下一步才考虑 `vLLM source build`
- 如果某个版本安装时强制升级 `torch` 到 `2.9` 或其他不希望版本，直接标记该环境失败，不继续测试该环境

## 推荐测试顺序

第一优先级：

1. `vllm==0.8.5`
2. `vllm==0.8.4`
3. `vllm==0.8.3`

第二优先级：

1. `vllm==0.7.3`
2. `vllm==0.7.2`

第三优先级：

1. `vllm==0.6.6`
2. `vllm==0.6.4.post1`
3. `vllm==0.6.4`

## 推荐执行方式

### 单版本执行

```bash
VLLM_VERSION=0.8.5 bash script/create_vllm_torch26_env.sh
conda activate echo_vllm_torch26
VLLM_VERSION=0.8.5 bash script/test6_vllm_downgrade_smoke.sh
```

### 手动矩阵循环示例

```bash
for V in 0.8.5 0.8.4 0.8.3 0.7.3 0.7.2 0.6.6 0.6.4.post1 0.6.4; do
  ENV_NAME="echo_vllm_torch26_${V//./_}"
  ENV_NAME="${ENV_NAME//-/_}"
  ENV_NAME="$ENV_NAME" VLLM_VERSION="$V" bash script/create_vllm_torch26_env.sh
  conda activate "$ENV_NAME"
  VLLM_VERSION="$V" bash script/test6_vllm_downgrade_smoke.sh
  conda deactivate
done
```

## 结果表模板

| vLLM | Python | Torch | CUDA | Transformers | import | registry | LLM load | text-only | audio | stop words | LoRA | 结论 |
|------|--------|-------|------|--------------|--------|----------|----------|-----------|-------|------------|------|------|
| 0.8.5 | 3.10 | 2.6.0 | cu124 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 2026-05-14 实测结果

本轮测试的成功标准严格定义为：

- `import vllm` 成功
- `LLM(model=Qwen2.5-Omni-7B, trust_remote_code=True)` 成功
- 单音频 inference 成功

仅 `text-only` 成功，不算 Echo rollout 可用。

### 已测版本摘要

| vLLM | Python | Torch | CUDA | Transformers | import | registry | LLM load | audio inference | 结论 |
|------|--------|-------|------|--------------|--------|----------|----------|-----------------|------|
| 0.8.5 | 3.10.20 | 2.6.0+cu124 | 12.4 | 4.57.6 | ✅ | ❌ timeout (120s) | ❌ | ❌ | 未通过 |
| 0.8.4 | 3.10.20 | 2.6.0+cu124 | 12.4 | 4.57.6 | ✅ | ❌ EOFError | ❌ | ❌ | 未通过 |
| 0.6.4 | 3.10.20 | 2.6.0+cu124 | 12.4 | 4.57.6 | ✅ | 未完成结构化记录 | 未确认 | 未确认 | 未通过当前成功标准 |

## 2026-05-14 Singularity 容器复测结果

本轮容器复测的目标是验证：

- 是否可以用更高 `glibc` 的 Ubuntu 22.04 + CUDA 12.x 环境绕过宿主机的 `GLIBC_2.34` 阻塞
- 若系统 ABI 问题被绕过，`vllm==0.8.5` 是否能真正达到 Echo rollout 可用标准

成功标准仍然严格定义为：

1. `import vllm` 成功
2. `LLM(model=Qwen2.5-Omni-7B, trust_remote_code=True)` 成功
3. 单音频 inference 成功

### 容器环境

- 镜像：`docker://nvidia/cuda:12.4.0-devel-ubuntu22.04`
- 本地 SIF：`output/singularity/nvidia_cuda_12.4.sif`
- 容器内 `glibc`：`2.35`
- Python：`3.10.20`
- `torch==2.6.0+cu124`
- `transformers==4.57.6`
- `huggingface-hub==0.36.0`
- `vllm==0.8.5`

### 关键结果

| 项目 | 结果 |
|------|------|
| `import vllm` | ✅ 成功 |
| `LLM load Qwen2.5-Omni` | ✅ 成功 |
| 单音频 inference | ✅ 成功 |
| stop words | ✅ 成功 |
| LoRA capability check | ✅ 成功 |
| overall_ok | ✅ true |

text-only 结果：

- `"I'm sorry, but I cannot hear audio as I am a text-based AI language model..."`
- 这说明在无音频 token 的 prompt 下，模型行为正常

audio 结果：

- `"The audio contains a sound effect that is described as a 'whoosh'..."`
- 说明单音频输入路径打通，模型可以处理音频内容

### 关键运行指标

- 模型加载：约 `21s`
- 占用显存：约 `16.74 GiB`
- `torch.compile`：约 `9s`（cache hit）
- GPU KV cache：`298,384 tokens`
- graph capture：`26s`
- engine 初始化总耗时：`54.26s`

### 唯一额外注意事项

默认 `gpu_memory_utilization=0.6` 不足以完成 KV cache 分配，首次运行在 cache 分配阶段 OOM。

将 `gpu_memory_utilization` 提升到 `0.85` 后，完整 smoke 通过。

### 产物

- report：`output/vllm_downgrade_smoke/report_0.8.5_singularity.json`
- log：`output/vllm_downgrade_smoke/smoke_0.8.5_singularity.log`

## 更新后的主结论

截至 2026-05-14：

- `vllm==0.8.4`：太旧，Qwen2.5-Omni 原生支持不足
- `vllm==0.8.5` 宿主机 wheel 路线：被宿主机 `glibc/Triton` 阻塞
- `vllm==0.8.5` Singularity 容器路线：**已成功**

因此，当前最明确、已验证的可行路线是：

- **使用 Ubuntu 22.04 / glibc 2.35 的 Singularity 容器**
- 在容器中运行：
  - `torch 2.6.0+cu124`
  - `transformers 4.57.6`
  - `huggingface-hub 0.36.0`
  - `vllm 0.8.5`
- 并将 `gpu_memory_utilization` 调整到 `0.85`

这条路线已经满足 Echo rollout 的最小可用标准：

- `import vllm`
- `LLM load Qwen2.5-Omni`
- 单音频 inference

### 0.8.5 详细结论

测试环境：

- conda env: `echo_vllm_torch26_085`
- `torch==2.6.0+cu124`
- `transformers==4.57.6`
- `huggingface-hub==0.36.0`
- 模型路径：`/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B`
- 音频路径：`output/interleaved_tmp/round1_seg25.wav`

最终 `report` 文件：

- `output/vllm_downgrade_smoke/report_0.8.5.json`

关键结果：

1. `stage_0_imports`：**通过**
   - `import vllm` 成功
   - `vllm==0.8.5`
   - `torch==2.6.0+cu124`
   - `transformers==4.57.6`

2. `stage_1_registry_check`：**失败**
   - `python -m vllm.model_executor.models.registry` 在 120 秒超时
   - 不是 SIGSEGV，但 registry 无法在合理时间内完成

3. `stage_2_llm_load`：**失败**
   - `LLM(...)` 初始化失败
   - `Engine core initialization failed`
   - 深层根因来自 Triton / 系统库：
     `ImportError: /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.34' not found`

4. `stage_3_single_audio_inference`：**失败**
   - 未进入成功推理阶段
   - `text_only_ok=false`
   - `audio_ok=false`

结论：

- `vllm==0.8.5` 虽然已经能在 `torch 2.6.0+cu124` 环境中 import，
  但 **无法完成 Qwen2.5-Omni 的 LLM load，更不用说单音频 inference**
- 因此 **0.8.5 不满足“Echo rollout 可用”的成功标准**

### 0.8.4 详细结论

测试环境：

- conda env: `echo_vllm_torch26_084`
- 安装方式：`pip install vllm==0.8.4`，并约束 `torch==2.6.0+cu124`
- `torch==2.6.0+cu124`
- `transformers==4.57.6`
- 模型路径：`/hpai/aios3.0/private/user/s2025244189/s2025244265/Model_Env/Qwen2.5-Omni-7B`
- 音频路径：`output/interleaved_tmp/round1_seg25.wav`
- 测试脚本：`script/test6_vllm_downgrade_smoke.sh` → `scripts/vllm_qwen_omni_downgrade_smoke.py`

关键结果：

1. `stage_0_imports`：**通过**
   - `import vllm` 成功
   - `vllm==0.8.4`
   - `torch==2.6.0+cu124`
   - `transformers==4.57.6`

2. `stage_1_registry_check`：**失败**
   - 报错：`EOFError: Ran out of input`
   - 不是 `SIGSEGV`
   - 更像是 registry 子进程和父进程之间的 stdio / IPC 通信异常

3. `stage_2_llm_load`：**失败**
   - `LLM(model=..., trust_remote_code=True)` 未能成功初始化
   - 共测试了 3 种变体，全部失败
   - 其中最关键的新发现是：当 SLURM 分配到单卡并成功把 device 初始化推进到更深阶段后，
     `vLLM 0.8.4` 会在 `GPUModelRunner.__init__()` 内进一步失败：

   ```text
   AttributeError: 'Qwen2_5OmniConfig' object has no attribute 'num_attention_heads'
   ```

   完整链路可概括为：

   ```text
   LLM()
     → V1 Engine
     → GPUModelRunner.__init__()
     → model_config.get_num_kv_heads()
     → self.hf_text_config.num_attention_heads
     → Qwen2_5OmniConfig 顶层无该字段
   ```

   这说明问题不只是 worker / GPU 枚举，还有更本质的一层：
   **`vllm 0.8.4` 对 Qwen2.5-Omni 这类多模态 config 结构没有原生适配。**
   Qwen2.5-Omni 的相关字段位于 `config.text_config.*`，而不是顶层 config。

   变体 A：A800 GPU 5 + V1 engine（默认）
   - 报错：`NVMLError_InvalidArgument: Invalid Argument`
   - 现象：`pynvml` 使用 `physical_device_id=5` 查询，但 `CUDA_VISIBLE_DEVICES=5` 映射后，进程内只剩 device 0

   变体 B：A800 GPU 5 + `VLLM_USE_V1=0`
   - 报错：`RuntimeError: No CUDA GPUs are available`
   - 现象：V0 引擎子进程没有正确看到传入的 `CUDA_VISIBLE_DEVICES`

   变体 C：V100 GPU 0 + `VLLM_USE_V1=0`
   - 报错：`ValueError: Bfloat16 is only supported on GPUs with compute capability >= 8.0`
   - 现象：V100（cc 7.0）不支持 bf16，这是硬件限制，不是 Echo 逻辑问题

4. `stage_3_single_audio_inference`：**未到达**
   - `LLM load` 失败，未进入 text-only / audio 生成阶段

5. `stage_4_stop_words` / `stage_5_lora_capability_check_optional`：**未到达**
   - 由于 `LLM load` 未成功，后续阶段无法验证

结论：

- `vllm==0.8.4` 在 `torch 2.6.0+cu124` 环境中可以 import
- 但 **无法完成 `LLM(model=Qwen2.5-Omni-7B, trust_remote_code=True)` 初始化**
- 当前观测到的失败层级至少有两层：
  - `vLLM 0.8.4` 在单卡 `CUDA_VISIBLE_DEVICES` 场景下的 worker / device 映射处理问题
  - V1 路径是 `pynvml` 设备索引混乱
  - V0 路径是 worker 子进程看不到 CUDA
- 同时还有一个更强的结构性信号：
  - `vllm 0.8.4` 期望从顶层 config 读取 `num_attention_heads`
  - 但 `Qwen2_5OmniConfig` 不符合这一预期
  - 这意味着 **0.8.4 很可能发布时尚未原生支持 Qwen2.5-Omni 这类模型结构**
- 因此 **0.8.4 不满足“Echo rollout 可用”的成功标准**

### 0.6.4 当前观察

测试环境：

- conda env: `echo_vllm_torch26_064`
- 基于 `echo_vllm_torch26_085` clone 后将 `vllm` 切换为 `0.6.4`

已确认现象：

- 经过补依赖后，`import transformers, vllm` 可以通过
- `vllm==0.6.4`
- `transformers==4.57.6`

限制：

- 本轮 `0.6.4` smoke 没有产出结构化 `report_0.6.4.json`
- `smoke_0.6.4.log` 为空
- 因此目前只能确认它**达到 import 层**，不能确认它是否能：
  - 成功 `LLM load Qwen2.5-Omni`
  - 成功单音频 inference

按照当前成功标准，`0.6.4` 仍然**不能算降级成功**。

## 当前结论

截至 2026-05-14，宿主机原生 wheel 路线尚未找到一个同时满足以下三项的 `vLLM` 版本：

1. `import vllm`
2. `LLM load Qwen2.5-Omni`
3. 单音频 inference 成功

最接近成功的是 `vllm==0.8.5`，但它在 `LLM load` 阶段撞到了更底层的问题：

- Triton / engine core 初始化失败
- 系统 `glibc` 版本不足：`GLIBC_2.34 not found`

`vllm==0.8.4` 虽然没有撞到同样的 `glibc` 问题，但仍然停在 `LLM load` 阶段，而且暴露出两类问题：

- registry 子进程 `EOFError`
- V1 路径 `pynvml` 设备索引异常
- V0 路径 CUDA 对 worker 子进程不可见
- 更深处还有 `Qwen2_5OmniConfig` 顶层字段不匹配问题

其中最后这一点非常重要：它表明更老版本的 `vLLM` 很可能**根本不认识 Qwen2.5-Omni config 的结构**。

这说明当前阻塞已经不只是 Python 依赖或 `torch` 小版本问题，而是进入了：

- `vLLM wheel` 与当前系统库的 ABI 兼容性
- Triton 编译产物与宿主机 `glibc` 版本的匹配问题
- `vLLM` 不同版本在当前集群单卡 `CUDA_VISIBLE_DEVICES` 场景下的 worker / GPU 枚举兼容性
- `vLLM` 旧版本对 Qwen2.5-Omni 模型结构本身的支持缺失

## 下一步建议

优先级建议如下：

1. 不再把 `0.8.5` 当作“只差一点依赖”的问题看待
   - 它已经进入 `LLM load`，但失败根因是系统库层

2. 若继续 wheel 路线，最多再做少量抽测
   - `0.8.3`
   - 目的是做一次“确认性测试”，验证更老版本是否同样在 Qwen2.5-Omni config 支持层面失败
   - 不建议再机械性扫完整个 `0.7.x / 0.6.x` 列表

3. 如果 `0.8.3` 也复现同类问题，则应把主结论转向：
   - **旧 wheel 版本大概率不具备 Qwen2.5-Omni 原生支持**
   - 即使不撞 `glibc`，也会撞模型 config / model runner 适配问题

4. 后续主线应优先考虑：
   - **Singularity / Apptainer 容器化运行 `vllm 0.8.5`**
   - 或较新 vLLM + source build / 补丁适配
   - 宿主机直接运行不再是优先方案

## 失败后的下一步

如果所有 wheel 版本都满足以下任一条件，则停止 wheel 路线：

- import 失败
- registry 阶段 `SIGSEGV`
- 必须升级到不兼容的 `torch`
- 能 text-only，但音频 inference 始终失败

此时下一步才进入：

1. 调查 `vLLM` 对 `torch 2.6.0` 的历史支持窗口
2. 评估 `source build` 成本
3. 决定是否走自建 wheel / source build 路线
