# Rollout Reward Dry Run — v9b-2epoch 冒烟测试验证

日期: 2026-05-09

## 验证内容

在 `echo_rl/rollout_rewards.py` 的基础上完成两项验证：

1. **单元测试**: `tests/test_rollout_rewards.py` — 22 个测试，覆盖 penalty/bonus 计算、rollout_total 组合、自定义系数、边界情况
2. **Reward dry run**: 在真实 eval 输出（20 样本）上计算 rollout_reward，对比 base_total vs rollout_total

## Dry Run 结果

### 完整版（带 duplicate guard + finalize）

| 分组 | n | Base avg | Rollout avg | Delta |
|------|---|----------|-------------|-------|
| All | 20 | +0.315 | +0.097 | -0.218 |
| Correct | 6 | +1.208 | +0.908 | -0.300 |
| Wrong | 14 | -0.068 | -0.250 | -0.182 |
| Dup>0 | 19 | +0.258 | +0.037 | -0.221 |
| Dup=0 | 1 | +1.400 | +1.250 | -0.150 |

### 基本版（无 duplicate guard，无 finalize）

| 分组 | n | Base avg | Rollout avg | Delta |
|------|---|----------|-------------|-------|
| All | 20 | -0.305 | -0.395 | -0.090 |
| Wrong | 20 | -0.305 | -0.395 | -0.090 |

### 跨版本对比

| 版本 | Base avg | Rollout avg | Delta |
|------|----------|-------------|-------|
| Full (all) | +0.315 | +0.097 | -0.218 |
| Full correct | +1.208 | +0.908 | -0.300 |
| Full wrong | -0.068 | -0.250 | -0.182 |
| Basic (all) | -0.305 | -0.395 | -0.090 |
| Basic wrong | -0.305 | -0.395 | -0.090 |

## 结论

1. **Rollout_reward 正确区分正确/错误样本**: Correct 组 rollout avg = +0.908 vs Wrong 组 = -0.250（完整版）
2. **重复 seg 惩罚生效**: Dup>0 组 delta = -0.221，正确抑制了重复 seg 行为
3. **基本版无正确样本**: 20 样本全部错误，rollout avg = -0.395，低 accuracy（-0.305）叠加上 round_penalty 和低 segment 得分
4. **完整版优于基本版**: 完整版正确率 30%，rollout avg +0.097 vs 基本版 -0.395
5. **rollout_total = base_total + penalties + bonuses**: 组合逻辑正确，基础 reward 字段（format, consistency, accuracy, segment, total）被完整保留

## 输出文件

- `tests/test_rollout_rewards.py` — 22 个单元测试
- `output/interleaved_eval/reward_dry_run.py` — dry run 脚本
