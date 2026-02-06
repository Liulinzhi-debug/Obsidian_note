## GDPO (Group reward-Decoupled Normalization Policy Optimization)
- 在多奖励（Multi-Reward）场景下（例如同时优化“正确性”和“格式”），GRPO 的做法是先将所有奖励求和 $r_{sum} = r_1 + r_2$，再进行归一化。这会导致 **“奖励信号坍缩” (Reward Signal Collapse)** 。即不同的奖励组合（如 $0+2$ 和 $1+1$）求和后相同，导致归一化后的优势无法区分优劣，丢失了细粒度的训练信号 。
- **核心机制**：GDPO 采用 **“先解耦归一化，再求和，最后 Batch 归一化”** 的策略 。
    1. **解耦 (Decouple)**：对每一个目标（Objective）单独进行组内归一化，保留各自的相对优势 
    2. **求和 (Sum)**：将归一化后的优势相加。
    3. **Batch 归一化 (Batch Normalization)**：对最终的优势和在整个 Batch 维度进行标准化，防止因奖励数量增加导致数值范围膨胀，保证训练稳定性 。
假设有 $n$ 个奖励目标，对于第 $i$ 个问题的第 $j$ 个回答：
![[Pasted image 20260202130518.png]]
![[Pasted image 20260202130436.png]]


```Python
# 基于 verl/trainer/ppo/ray_trainer.py 中的逻辑整理
# 注意：GDPO 通常是在 trainer 循环中处理优势计算，因为它涉及对不同 reward head 的多次调用

def compute_gdpo_advantage(
    data: DataProto, 
    core_algos, # 传入核心算法模块以复用 GRPO 逻辑
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    GDPO 优势计算：解耦归一化 + Batch 归一化
    论文: https://arxiv.org/abs/2601.05242
    """
    # 准备数据：假设数据中已经包含了两个独立的奖励流
    # 实际场景中可能有 n 个，这里以 Correctness 和 Format 为例
    token_level_scores_correctness = data.batch['token_level_scores_correctness']
    token_level_scores_format = data.batch['token_level_scores_format']
    index = data.non_tensor_batch['uid'] # Prompt ID 用于分组

    # ------------------------------------------------------
    # 1. 解耦归一化 (Decoupled Group Normalization)
    # ------------------------------------------------------
    # 分别调用 GRPO 的逻辑计算各自的 Normalized Advantage
    # 关键点：这里是在 Reward 层面就分开了，而不是加完再算
    
    # 计算 Correctness 的相对优势 (r - mu) / sigma
    correctness_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_scores_correctness,
        eos_mask=response_mask,
        index=index
    )

    # 计算 Format 的相对优势 (r - mu) / sigma
    format_normalized_score, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_scores_format,
        eos_mask=response_mask,
        index=index
    )
    
    # ------------------------------------------------------
    # 2. 优势求和 (Summation)
    # ------------------------------------------------------
    # 将独立归一化后的优势相加
    # 此时 (0, 2) 组合的优势会显著高于 (0, 1)，解决了坍缩问题
    new_advantage = correctness_normalized_score + format_normalized_score

    # ------------------------------------------------------
    # 3. Batch 级归一化 (Batch-wise Normalization)
    # ------------------------------------------------------
    # 对整个 Batch 的优势进行白化 (Whiten)，确保数值分布稳定 (Mean=0, Std=1)
    # 如果不做这一步，随着奖励目标数量增加，Advantage 的方差会变大，导致训练不稳定 [cite: 231]
    # masked_whiten 计算的是 (A - BatchMean) / (BatchStd + eps)
    advantages = masked_whiten(new_advantage, response_mask) * response_mask

    return advantages
```

