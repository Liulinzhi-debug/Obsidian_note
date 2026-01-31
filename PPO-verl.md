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