# Task 12: Test 3 / Test 4 最终结果

## 背景

两个环境隔离测试，用于验证 torch 2.9.0 + vLLM 0.12 + Qwen2.5-Omni 的 SIGSEGV 根因。

- Test 3: torch 2.6.0 (无 vLLM) + VERL/FSDP 初始化 smoke
- Test 4: torch 2.9.0 + vLLM 0.12 + Qwen2.5-Omni inference smoke

---

## Test 4: vLLM-only inference (job 42062) — FAILED

**环境**: qwen_echo conda env, torch 2.9.0+cu128, vLLM 0.12.0, transformers 4.57.6

**结果**: 模型注册阶段 SIGSEGV

```
ERROR [registry.py:735] Error in inspecting model architecture 'Qwen2_5OmniModel'
subprocess.CalledProcessError: Command 'python3 -m vllm.model_executor.models.registry'
died with <Signals.SIGSEGV: 11>
```

**分析**:
- 错误发生在 vLLM 模型注册子进程，与 VERL/FSDP/DeepSpeed 完全无关
- vLLM 0.12.0 自己的 `inspect_model_cls()` 在检查 `Qwen2_5OmniModel` 架构时就崩溃了
- torch 2.9.0 + Qwen2.5-Omni 在任何底层模型操作场景（FSDP wrap、vLLM inspection、DeepSpeed init）都会 SIGSEGV
- 根因在 torch 2.9.0 的 C 层，Python 无法捕获

---

## Test 3: torch 2.6.0 + VERL/FSDP init (job 42065 → srun runs) — PARTIAL SUCCESS

**环境**: qwen3omni conda env, torch 2.6.0+cu124, NO vLLM

**多次迭代过程**:

| 尝试 | 结果 | 原因 |
|------|------|------|
| job 42065 | OOM | GPU 6 已被其他进程占用 28GB |
| srun #1 | 路径错误 | srun 未传入 TRAIN_FILE/MODEL_PATH 环境变量 |
| srun #2 | infer_tp assertion | `rollout.world_size=1 % infer_tp=2 != 0` |
| srun #3 | qwen_vl_utils 缺失 | DataLoader worker 中 `ModuleNotFoundError: qwen_vl_utils` |
| srun #4 (最终) | sequence_length assertion | 见下方 |

**最终通过的阶段** (srun #4 on GPU 5, 完全空闲):

| 阶段 | 结果 |
|------|------|
| Model loading (8.93B params, 5 shards, ~20s) | **PASS** |
| FSDP wrapping (NO_SHARD, world_size=1) | **PASS** |
| HF rollout build (infer_tp=1) | **PASS** |
| DataLoader (20 samples) | **PASS** |
| trainer.fit() entered | **PASS** |
| generate_sequences() | **FAIL** — `assert seq.shape[1] == sequence_length` |

**修复的配置项**:
- `actor_rollout_ref.rollout.tensor_model_parallel_size=1` (原默认 2)
- 安装缺失依赖: qwen_vl_utils, tensordict, codetiming, torchdata 等

**当前阻塞点**: `verl/verl/workers/rollout/hf_rollout.py:137`
```
assert seq.shape[1] == sequence_length
```
HF generate 输出序列长度与 VERL 期望的 `prompt_length + response_length` 不一致。

---

## 结论

1. **torch 2.9.0 + Qwen2.5-Omni 是致命不稳定组合** — 不可用于任何场景（训练或推理）
2. **vLLM 0.12 强依赖 torch 2.9.0** — 当前 vLLM 路线暂停，除非未来 vLLM 支持 torch 2.6 或 Qwen2.5-Omni 支持 torch 2.9
3. **torch 2.6.0 + VERL/FSDP 是可行训练路线** — 模型加载、FSDP wrapping、rollout 构建、DataLoader 全部通过
4. **当前唯一阻塞项**: HF rollout 中 Qwen2.5-Omni 多模态输入/输出序列长度与 VERL 期望不一致

---

## 路线更新

```
旧路线: vLLM rollout server + VERL training (torch 2.9)
         ↓ (torch 2.9 SIGSEGV)
新路线: torch 2.6 + VERL/FSDP + HF rollout
         ↓ (当前阻塞)
         Fix sequence_length assertion in hf_rollout.py
```

- vLLM 路线: **暂停** (除非上游支持 torch 2.6 或 Qwen2.5-Omni 支持 torch 2.9)
- DeepSpeed 路线: **不再尝试** (torch 2.9 SIGSEGV 根因)
- 自定义 GRPO 路线: **保留为 fallback** (fcc5fdf 可跑基线)
- VERL + HF rollout 路线: **当前主线**

---

## 下一步

修复 `verl/verl/workers/rollout/hf_rollout.py:137` 的 sequence_length assertion → Task 13

---

# Task 13: Test 5 — VERL HF Rollout Sequence Length Debug

## 目标

验证 Test 3 中 `seq.shape[1] != sequence_length` 的根因。通过注入 `ECHO_DEBUG_SEQUENCE_LENGTH=1` 环境变量控制的 debug 日志，收集 `generate()` 前后的完整张量信息。

## 复现脚本

[script/test5_verl_hf_rollout_sequence_debug.sh](script/test5_verl_hf_rollout_sequence_debug.sh)

基于 Test 3，关键差异：
- `data.max_response_length=64`（加速）
- `ECHO_DEBUG_SEQUENCE_LENGTH=1`
- 其余参数与 Test 3 一致

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| [verl/verl/workers/rollout/hf_rollout.py](verl/verl/workers/rollout/hf_rollout.py) | 导入 `os`，新增 `_ECHO_DEBUG` 开关；在 `_generate_minibatch` 的两处（输入阶段 + assertion 前）注入 debug 日志 |
| [verl/verl/workers/fsdp_workers.py](verl/verl/workers/fsdp_workers.py) | 在 `generate_sequences` 入口注入 debug 日志 |

## 结果

**Test 5 成功复现 AssertionError**，与 Test 3 完全一致。

### 关键日志

```
=== ECHO_DEBUG: fsdp_workers.generate_sequences entry ===
  prompts.batch keys: ['input_ids', 'attention_mask', 'position_ids']
  prompts.non_tensor_batch keys: ['raw_prompt_ids', 'multi_modal_data', 'raw_prompt', 'tools_kwargs']
  multi_modal_data present, type: <class 'numpy.ndarray'>   ← 不是 dict!

=== ECHO_DEBUG: _generate_minibatch input ===
  prompt_length: 2048
  response_length (config): 64
  expected sequence_length: 2112
  sample[0]: total=2048, pad_tokens=1684, actual_prompt_len=364   ← 实际 prompt 仅 364 tokens
  multi_modal_data keys: <class 'numpy.ndarray'>   ← 无法提取 audio
  multi_modal_inputs: 不存在!
  input_features: 不存在!
  feature_attention_mask: 不存在!

=== ECHO_DEBUG: pre-assertion state ===
  seq.shape: torch.Size([1, 2241])
  sequence_length (expected): 2112
  delta_length: -129   ← output 比预期长 129 tokens!
  output.sequences shape before potential pad: torch.Size([1, 2241])
  sample[0]: total_seq_len=2241, actual_response_len=192   ← 生成了 192 tokens，但 response_length=64!
  seq min/max token_id: 8 / 151648   ← 151648 = audio_end_token_id
  input_ids 末 5: [151645, 198, 151644, 77091, 198]   ← EOS/BOS 可见

AssertionError: assert seq.shape[1] == sequence_length
    seq.shape[1] = 2241
    sequence_length = 2112 (= 2048 + 64)
```

## 根因分析

**Case D — audio features 完全未传入 HF rollout（主因）+ Case B 并发症。**

### 数据流断裂点

```
rl_dataset.py: processor() 产出:
  → input_ids (含 audio placeholder tokens: 151646/151647/151648)
  → multi_modal_inputs = {"input_features": Tensor, "feature_attention_mask": Tensor}

ray_trainer.py: pop gen_batch:
  → batch_keys: ["input_ids", "attention_mask", "position_ids"]
  → non_tensor_batch_keys: ["raw_prompt_ids", "multi_modal_data", ...]
  → multi_modal_inputs 留在原 batch 中，未 pop 到 gen_batch!

hf_rollout.py: generate():
  → 只传入 input_ids/attention_mask/position_ids
  → 没有 input_features/feature_attention_mask
  → input_ids 含 audio placeholder，但无对应音频特征
  → Qwen2.5-Omni generate() 行为异常：
      max_new_tokens=64 → 实际生成 192 tokens (3x)
      seq 长 2241 > 期望 2112
```

### 数值对照

| 量 | 值 | 来源 |
|----|-----|------|
| prompt_length (padded) | 2048 | `idx.size(1)` |
| 实际 prompt 有效 token | 364 | 排除 1684 个 pad token |
| response_length (config) | 64 | `self.config.response_length` |
| sequence_length (期望) | 2112 | `2048 + 64` |
| seq.shape[1] (实际) | 2241 | `output.sequences` |
| 实际生成 token 数 | 193 | `2241 - 2048` |
| 实际 response 长度 (到 EOS) | 192 | EOS 在 response 中的位置 |
| delta | -129 | `2112 - 2241` |

## 修复方向（待确认）

1. **ray_trainer.py**: 将 `multi_modal_inputs` 也 pop 到 `gen_batch`
2. **hf_rollout.py**: 将 `input_features` / `feature_attention_mask` 传给 `self.module.generate()`
3. 修复后重新测量 seq.shape[1]，确认是否等于 sequence_length
4. 增加 delta_length < 0 的容错逻辑（crop 或按实际长度重算）
