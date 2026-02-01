## GRPO
- 基于PPO算法，去掉了Critic模型
- 计算优势是$A_t=Q(s_t,a_t)-V(s_t)$，其中减去$V(s_t)$是因为蒙特卡洛$Q(s_t,a_t)$导致的高方差特性，而减去$V(s_t)$能很好的降低方差。
- GRPO使用 **单次样本多次采取求均值减少方差** ，因此去掉了对于 Value Loss 的计算$$A_i = \frac{r_i - \mu_{group}}{\sigma_{group} + \epsilon}$$
    **Group Mean ($\mu_{group}$)**: 同一组回答得分的均值，作为 Baseline。
    **Group Std ($\sigma_{group}$)**: 同一组回答得分的标准差，用于归一化（Standardization）。
    **Dr.GRPO Variant**: 代码支持 `norm_adv_by_std=False`，即只减均值不除标准差 ($A = r - \mu$)，这对应 Dr.GRPO 的改进策略。
```Python
# verl/trainer/ppo/core_algos.py
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray, # 关键输入: 样本所属的 Prompt ID，用于分组
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True, # 控制是否除以 Std
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO 优势计算：基于结果奖励 (Outcome Reward) 进行组内归一化
    """
    # 1. 聚合得分 (Scalar Reward)
    # 将 token 级别的奖励求和，得到整句回答的一个标量总分 r_i
    scores = token_level_rewards.sum(dim=-1)

    # 2. 分组统计 (Group Statistics)
    # 使用 Python 字典在 CPU 上进行分组计算 (因为 group size 通常不大)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        # 根据 index (Prompt ID) 将分数归类
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算每组的均值和方差
        for idx in id2score:
            # 边界情况: 只有 1 个样本时，无法计算有意义的相对优势，设为 0
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor) # Baseline
                id2std[idx] = torch.std(scores_tensor)   # Scale
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        # 3. 计算优势 (Advantage Calculation)
        for i in range(bsz):
            # 标准 GRPO: Z-Score 归一化 (r - mean) / std
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            # Dr.GRPO 变体: 仅中心化 (r - mean)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        
        # 4. 广播回 Token 级别 (Broadcast)
        # GRPO 的优势是整句级别的，需要扩展维度以匹配 token 形状
        # [bsz] -> [bsz, 1] * [bsz, len] -> [bsz, len]
        scores = scores.unsqueeze(-1) * response_mask

    # 在 GRPO 中，Returns 和 Advantages 通常被视为同一值用于 Loss 计算
    return scores, scores
```

---
## REINFORCE ++
- REINFORCE++ 是经典 REINFORCE 算法（Monte Carlo Policy Gradient）在大模型对齐场景下的增强版。与 PPO/GRPO 不同，它回归了最基础的**累计回报 (Cumulative Return)** 概念，直接使用整个序列的折扣回报作为优势函数。
- **核心思想**：如果不使用 Critic 估值，也不使用 Group Normalization，最直接的“好坏”衡量标准就是“这句说完以后一共拿了多少分”。
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{T-t} r_T$$
$$A_t \approx \text{Whiten}(G_t)$$
**Discounted Return ($G_t$)**:
- 含义：从当前时刻 $t$ 开始，一直到结束所获得的所有奖励的折扣和。
- 区别：PPO 使用 GAE (依赖 Critic)，GRPO 使用组内相对分 (依赖 Group)，REINFORCE++ 使用纯蒙特卡洛回报 (依赖完整的轨迹回溯)。
**Whiten (标准化)**:
- 含义：对计算出的回报 $G_t$ 进行减均值除方差 ($\frac{G - \mu}{\sigma}$)，起到类似 Baseline 的作用，大幅降低梯度估计的方差。
```Python
# verl/trainer/ppo/core_algos.py
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, 
    response_mask: torch.Tensor, 
    config: Optional[AlgoConfig] = None, 
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    REINFORCE++ 优势计算：基于蒙特卡洛回报 (Monte Carlo Return)
    论文: https://arxiv.org/abs/2501.03262
    """
    assert config is not None
    gamma = config.gamma
    
    with torch.no_grad():
        # returns 用于存储每个时刻 t 的累计折扣回报 G_t
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        # 1. 逆序回溯 (Reverse Accumulation)
        # 从序列最后一个 token 往前推，计算 Discounted Return
        # 公式: G_t = r_t + gamma * G_{t+1}
        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            
            # [关键] Reset after EOS
            # 如果当前 token 是 padding (mask=0)，说明这一句结束了
            # 为了防止不同样本之间的回报混淆(虽然这里是 batch 维度并行的，
            # 但如果一个序列里有 padding 穿插，需清零)，实际上主要是处理 Padding 区域保持为 0
            running_return = running_return * response_mask[:, t]

        # 2. 优势标准化 (Advantage Whitening)
        # 原始 REINFORCE 的方差极大。
        # REINFORCE++ 的核心改进之一就是对其进行 Whiten (Z-Score 归一化)。
        # 这等价于引入了一个动态的 Baseline (平均回报)。
        advantages = verl_F.masked_whiten(returns, response_mask)
        
        # 再次应用 Mask 确保 Padding 处优势为 0
        advantages = advantages * response_mask

    # 在 REINFORCE 中，优势 A_t 直接由累计回报 G_t 近似
    return advantages, returns
```

---
## RLOO Advantage (Leave-One-Out Baseline)
- RLOO 是对 REINFORCE 的改进，旨在降低梯度的方差。与 GRPO 使用整个组的均值（包含自身）不同，RLOO 为每个样本 $i$ 计算基线时，**只使用同组中其他 $N-1$ 个样本的均值**。
- **核心思想**：通过剔除自身对基线的影响，使得优势估计更加“干净”，避免了因为自身得分极高而拉高基线导致优势被低估的情况。
$$A_i = r_i - \frac{1}{N-1} \sum_{j \neq i} r_j$$
```Python
# verl/trainer/ppo/core_algos.py
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RLOO 优势计算：基于留一法 (Leave-One-Out) 基线
    论文: https://arxiv.org/abs/2402.14740
    """
    # 1. 聚合得分
    scores = token_level_rewards.sum(dim=-1)

    # 2. 分组计算全组均值 (Global Mean)
    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        # 收集每组的分数
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        
        # 计算每组的全量均值 (包含所有样本)
        for idx in id2score:
            if len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                # 只有 1 个样本时无法做 Leave-One-Out，设为 0
                id2mean[idx] = torch.tensor(0.0) 

        # 3. 利用代数技巧计算 LOO 优势
        # 原始定义: A_i = r_i - (sum(all) - r_i) / (N - 1)
        # 代码实现: A_i = (N / (N-1)) * (r_i - mean(all))
        # 这两个公式在数学上是完全等价的，但后者只需计算一次 mean，效率更高。
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = (scores[i] - id2mean[index[i]]) * response_num / (response_num - 1)
        
        # 4. 广播回 Token 级别
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
```
