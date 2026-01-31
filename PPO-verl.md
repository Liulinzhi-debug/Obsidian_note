## PPO
- AC 的 Model-free 策略
- 通过**最小化 Loss** 来等价于**最大化期望收益**。因此，PPO 的目标函数会加上负号转换为 Loss。

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \max \left( -r_t(\theta)\hat{A}_t, \;\; -\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
 **ratio ($r_t(\theta)$)**: 
   - 定义：$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} = \exp(\log\pi - \log\pi_{old})$
   - 含义：新策略相对于旧策略的概率变化比率。
 **Advantage ($\hat{A}_t$)**:
   - 含义：当前动作 $a_t$ 相对于平均水平的优势。$\hat{A}_t > 0$ 表示动作好，$\hat{A}_t < 0$ 表示动作差。
 **Clip ($\epsilon$)**:
   - 含义：允许策略发生变化的最大幅度（通常为 0.1 或 0.2），防止参数更新过大导致模型崩溃。
   
```python
# verl/trainer/ppo/core_algos.py
def compute_policy_loss(old_log_prob, log_prob, advantages, response_mask, 
                        cliprange, cliprange_low, cliprange_high, clip_ratio_c=3.0, ...):
    
    # ---------------------------------------------------------------------
    # 1. 计算概率比率 (Ratio)
    # ---------------------------------------------------------------------
    # 原始公式: ratio = pi_new / pi_old
    # 工程实现: ratio = exp(log_pi_new - log_pi_old)
    # 目的: 避免除法溢出，增强稳定性
    negative_approx_kl = log_prob - old_log_prob
    
    # [稳定性保护] 强制将 log 差值限制在 [-20, 20] 之间，防止 exp 后数值爆炸
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)

    # ---------------------------------------------------------------------
    # 2. 标准 PPO Loss 计算 (Standard PPO Logic)
    # ---------------------------------------------------------------------
    # 对应公式项: -r_t * A_t
    # 含义: 如果不截断，模型应该怎么更新
    pg_losses1 = -advantages * ratio

    # 处理 clip 参数，如果没有指定上下限，就用同一个 cliprange
    if cliprange_low is None: cliprange_low = cliprange
    if cliprange_high is None: cliprange_high = cliprange

    # 对应公式项: -clip(r_t, 1-ε, 1+ε) * A_t
    # 含义: 限制更新幅度后的 Loss
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)

    # [标准 PPO 核心] 取最大值 (对应数学上的取最小值)
    # 解释: 我们要最小化 Loss。
    # 如果 A > 0 (好动作): 我们希望 ratio 变大，loss 变负。clip 限制了 ratio 不能无限大。
    # torch.max 在负数域等价于取"惩罚更重"的那一项。
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    # ---------------------------------------------------------------------
    # 3. Dual-Clip PPO 扩展逻辑 (The Difference)
    # ---------------------------------------------------------------------
    # 这一步是原始 PPO 没有的。
    # 场景: 当 advantage < 0 (坏动作) 时，ratio 会小于 1 (降低概率)。
    # 风险: 如果 ratio 变得非常非常小 (比如 0.0001)，-advantage * ratio 会非常小，但梯度可能会很大或不稳定。
    # 解决: 设置一个下界 clip_ratio_c (默认 3.0)。
    # 逻辑: 当 advantage < 0 时，ratio 导致的 Loss 不能超过 -advantages * 3.0。
    
    # 计算下界 Loss
    pg_losses3 = -advantages * clip_ratio_c
    
    # 应用下界: 如果标准 PPO 的 Loss 超过了下界，就被截断
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)

    # ---------------------------------------------------------------------
    # 4. 最终组合 (Final Combination)
    # ---------------------------------------------------------------------
    # 根据 Advantage 的正负，选择不同的 Loss 计算方式
    # A < 0 (坏动作): 使用 Dual-Clip 逻辑 (clip_pg_losses2) -> 防止把概率降得太离谱
    # A > 0 (好动作): 使用标准 PPO 逻辑 (clip_pg_losses1)   -> 稳步提升概率
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # 聚合 Loss (求平均)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower

```

---
## Generalized Advantage Estimation (GAE)
- **Critic 视角的评价机制**
- 平衡 **方差 (Variance)** 和 **偏差 (Bias)** 的优势估计方法。通过引入 $\lambda$ 参数，在单步 TD (低方差高偏差) 和 蒙特卡洛 (高方差低偏差) 之间寻找平衡点。
$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \text{where} \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
**$\delta_t$ (TD Error)**:
- 含义：单步差分误差。即“真实发生的奖励 + 对未来的预估”与“当前预估”的差值。
    **$\gamma$ (Discount Factor)**:
- 含义：折扣因子（通常 0.99）。决定了模型看多远的未来。
    **$\lambda$ (Smoothing Parameter)**:
- 含义：平滑系数（通常 0.95）。$\lambda=0$ 时退化为单步 TD，$\lambda=1$ 时接近蒙特卡洛。
    **Returns ($R_t$)**:
- 含义：回报。$R_t = A_t + V(s_t)$，用于作为 Critic 模型训练的标签（Target）。
```Python
# verl/trainer/ppo/core_algos.py
def compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam):
    """
    计算 GAE 优势和 Returns
    """
    with torch.no_grad(): # 不需要梯度，这是在生成标签
        nextvalues = 0   # V(t+1)，也就是下一时刻的价值
        lastgaelam = 0   # A(t+1)，也就是下一时刻的优势
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        # ---------------------------------------------------------------------
        # 1. 逆序遍历 (Reverse Loop)
        # ---------------------------------------------------------------------
        # 必须从未来推导现在。因为 A_t 依赖于 A_{t+1}
        for t in reversed(range(gen_len)):
            
            # ---------------------------------------------------------------------
            # 2. 计算 TD Error (delta)
            # ---------------------------------------------------------------------
            # delta = r_t + gamma * V_{t+1} - V_t
            # 代表当前这一步产生的“惊喜”（好于预期还是差于预期）
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            
            # ---------------------------------------------------------------------
            # 3. 递归计算 GAE (Advantage)
            # ---------------------------------------------------------------------
            # A_t = delta + (gamma * lambda) * A_{t+1}
            # lastgaelam_ 是当前步计算出的未处理 Mask 的优势
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # ---------------------------------------------------------------------
            # 4. Mask 处理 (Padding Handling)
            # ---------------------------------------------------------------------
            # 处理序列结束或 Padding 的情况。
            # 如果 response_mask[:, t] 为 0 (padding)，则 nextvalues 和 lastgaelam 保持不变
            # 或者是被重置（取决于具体实现逻辑，这里是穿透逻辑）
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        
        # 翻转回来，变成正常的时间顺序
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # ---------------------------------------------------------------------
        # 5. 计算 Critic 的目标 (Returns)
        # ---------------------------------------------------------------------
        # Returns = Advantage + Value。这是 Critic 需要拟合的真实标签。
        returns = advantages + values
        
        # [标准化] 对优势进行 Whiten (减均值除方差)，这对 PPO 收敛至关重要
        advantages = verl_F.masked_whiten(advantages, response_mask)
        
    return advantages, returns
```

---
## L_value (Critic Loss)
- **Critic 模型的损失函数**
- Critic 的任务是预测状态价值 $V(s)$。损失函数是预测值与真实回报 $R_t$ 之间的 **MSE (均方误差)**。PPO 这里也引入了 **Clip 机制**，防止 Value 更新过猛。
$$L^{VF}_t = \frac{1}{2} \max \left[ (V_\theta(s_t) - R_t)^2, \;\; (\text{clip}(V_\theta(s_t), V_{old}-\epsilon, V_{old}+\epsilon) - R_t)^2 \right]$$
**$V_\theta(s_t)$ (Pred)**:
- 含义：当前 Critic 模型预测的价值。
    **$R_t$ (Target)**:
- 含义：GAE 阶段计算出来的 Returns（真实回报），作为 Ground Truth。
    **$V_{old}$ (Old Pred)**:
- 含义：更新前 Critic 模型的预测值。
    **Clip Range ($\epsilon$)**:
- 含义：允许价值预测偏离旧预测的最大幅度。
```Python
# verl/trainer/ppo/core_algos.py
def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """
    vpreds: 当前 Critic 的预测值
    values: 旧 Critic 的预测值 (Old Value)
    returns: 真实回报 (Target)
    """
    
    # ---------------------------------------------------------------------
    # 1. 价值截断 (Value Clipping)
    # ---------------------------------------------------------------------
    # 强制当前预测值 vpreds 不能偏离旧值 values 太多
    # 范围限制在 [values - epsilon, values + epsilon]
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    
    # ---------------------------------------------------------------------
    # 2. 计算两个 Loss
    # ---------------------------------------------------------------------
    # Loss1: 原始 MSE (预测值 - 真实值)^2
    vf_losses1 = (vpreds - returns)**2
    # Loss2: 截断后的 MSE
    vf_losses2 = (vpredclipped - returns)**2
    
    # ---------------------------------------------------------------------
    # 3. 取最大值 (Pessimistic Bound)
    # ---------------------------------------------------------------------
    # 这里取 max 是为了取由“惩罚更大”的那一项主导的 Loss。
    # 如果预测值跑得太偏（触发 Clip），Loss 会变得很大，强迫模型退回来。
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    
    # 统计有多少比例的数据触发了截断 (用于监控)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    
    return vf_loss, vf_clipfrac
```

---
## KL Estimators (Approximation Methods)
- **KL 散度的具体近似实现**

- 不同的估计器在 **偏差 (Bias)** 和 **方差 (Variance)** 之间做权衡。$$D_{k3} = r - \log r - 1, \quad \text{where } r = \exp(\log \pi_{ref} - \log \pi)$$
**k1 (Standard KL)**:
- 含义：$\log \pi - \log \pi_{ref}$。无偏估计，但单样本方差大，且可能出现负值（违反 KL 定义）。
**k2 (MSE)**:
- 含义：$0.5 (\log \pi - \log \pi_{ref})^2$。有偏估计，但始终非负，方差极低。
**k3 (Low-Var KL)**:
- 含义：利用指数比率构造的非负近似。基于 $x - \log x - 1 \geq 0$ 的性质。在 $\pi \approx \pi_{ref}$ 时，泰勒展开与 KL 一致。
```Python
# verl/trainer/ppo/core_algos.py
def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """具体的 KL 散度近似计算实现"""
    # 1. k1 估计器 (Standard) - 原始定义，方差大
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    # 2. Abs 估计器 - L1 距离
    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    # 3. k2 估计器 (MSE) - 0.5*(log_p - log_q)^2，非负且梯度平滑
    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # 4. k3 估计器 (Low-Variance KL) - 来源: http://joschu.net/blog/kl-approx.html
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20) # [数值稳定性]
        ratio = torch.exp(kl)
        # 公式: r - log(r) - 1 (当 r=1 时为0，且恒 >= 0)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10) # [输出保护]

    raise NotImplementedError
```

---
## Entropy & Critic Loss
- **Entropy ($H(\pi)$)**: 衡量策略随机性的指标。最大化熵（作为 Loss 的负项）可以**鼓励模型探索**，防止策略过早收敛到局部最优（即防止 Collapse）。
- **Critic Loss ($L_{VF}$)**: Critic 网络的训练目标。通过最小化预测值 $V(s)$ 与真实回报 $R_t$ (Returns) 的均方误差 (MSE) 来提升价值估计的准确性。包含 **Clip 机制**以防止 Critic 更新过快破坏稳定性。
**Entropy Formula**:
$$H(\pi(\cdot|s)) = - \sum_{a} \pi(a|s) \log \pi(a|s)$$
**Critic Loss Formula**:
$$L^{VF}_t = \frac{1}{2} \max \left[ (V_\theta - R_t)^2, \;\; (V_{clipped} - R_t)^2 \right]$$
```Python
# verl/trainer/ppo/core_algos.py

def entropy_from_logits(logits):
    """
    从 Logits 直接计算熵，利用 LogSumExp 技巧提高数值稳定性
    公式推导: H(p) = -sum(p * log(p))
             log(p) = logits - logsumexp(logits)
             H(p) = -sum(p * (logits - logsumexp(logits)))
                  = logsumexp(logits) - sum(p * logits)
    """
    pd = torch.nn.functional.softmax(logits, dim=-1)
    # 熵 = LogSumExp(logits) - Expected(logits)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy

def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value, loss_agg_mode="token-mean"):
    """
    计算 PPO 的 Critic Loss (带截断机制)
    """
    # 1. 价值截断 (Value Clipping)
    # 强制当前预测 vpreds 限制在旧预测 values 的 [1-ε, 1+ε] 范围内
    # 目的: 防止 Critic 这一步更新太猛，导致 Value Head "遗忘" 之前的估计
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    
    # 2. 计算两种 MSE Loss
    vf_losses1 = (vpreds - returns) ** 2          # 原始 Loss
    vf_losses2 = (vpredclipped - returns) ** 2    # 截断后 Loss
    
    # 3. 取最大值 (Pessimistic Bound)
    # 类似于 Policy Loss，这里取 max 是为了惩罚 "预测值跑得太远" 的情况
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    
    # 4. 聚合 (Mean)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    
    # 统计截断比例 (监控用)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    
    return vf_loss, vf_clipfrac
```

---
## PPO 总损失函数 
#### **目标 A：Actor 的更新 (对应 `ppo_loss`)**
$$L_{Actor}(\theta) = \underbrace{L^{CLIP}(\theta)}_{Policy} - \underbrace{c_{ent} H(\pi_\theta)}_{Entropy} + \underbrace{c_{kl} D_{KL}(\pi_\theta || \pi_{ref})}_{KL}$$
#### **目标 B：Critic 的更新 (对应 `value_loss`)**

$$L_{Critic}(\phi) = \underbrace{L^{VF}(\phi)}_{Critic}$$
```python
# verl/workers/utils/losses.py

def ppo_loss(config: ActorConfig, model_output, data: TensorDict, ...):
    """
    计算 Actor 的总 Loss: Policy Gradient + Entropy + KL
    """
    # ---------------------------------------------------------------------
    # 1. 策略梯度损失 (Policy Gradient Loss)
    # ---------------------------------------------------------------------
    # 对应公式: L_CLIP (通常包含负号，因为我们要最小化 Loss)
    # 输入: 旧策略概率(old_log_prob), 当前策略概率(log_prob), 优势(advantages)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        ...
    )
    policy_loss = pg_loss

    # ---------------------------------------------------------------------
    # 2. 熵正则项 (Entropy Bonus)
    # ---------------------------------------------------------------------
    # 对应公式: - c_ent * H(π)
    # 逻辑: 熵越大越好(鼓励探索)，所以在 Loss 中是减去熵(最小化负熵)
    if entropy is not None:
        entropy_loss = agg_loss(loss_mat=entropy, ...)
        
        # [关键操作] -= (减号)
        policy_loss -= config.entropy_coeff * entropy_loss

    # ---------------------------------------------------------------------
    # 3. KL 散度惩罚 (KL Penalty)
    # ---------------------------------------------------------------------
    # 对应公式: + c_kl * D_KL(π || π_ref)
    # 逻辑: 这里的 KL 是作为 Loss 的一部分显式加入的 (有些实现是放在 Reward 里)
    # 如果 KL 过大，Loss 变大，梯度会把模型拉回来
    if config.use_kl_loss:
        # 计算 log(pi) - log(ref) 等散度指标
        kld = kl_penalty(logprob=log_prob, ref_logprob=data["ref_log_prob"], ...)
        kl_loss = agg_loss(loss_mat=kld, ...)

        # [关键操作] += (加号)
        policy_loss += kl_loss * config.kl_loss_coef

    return policy_loss, metrics

def value_loss(config: CriticConfig, model_output, data: TensorDict, ...):
    """
    计算 Critic 的 Loss: Value Function Error
    """
    # ---------------------------------------------------------------------
    # 4. 价值损失 (Value Loss)
    # ---------------------------------------------------------------------
    # 对应公式: c_vf * MSE(V_pred, V_target)
    # 通常包含 Clip 机制: max((V - R)^2, (V_clipped - R)^2)
    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        returns=data["returns"], # 真实回报 (Target)
        values=data["values"],   # 旧预测值 (用于 Clip)
        ...
    )
    
    return vf_loss, metrics
```
