# PPO: Proximal Policy Optimization 笔记（arXiv:1707.06347）

---

## 1. Introduction（极简）

PPO 的目标：

> 在保证训练稳定性的前提下，提高 policy gradient 的 sample efficiency 和实现简洁性

背景问题：

- Vanilla Policy Gradient：
  - 更新不稳定
  - variance 高
- TRPO：
  - 稳定，但计算复杂（需要二阶优化）

PPO：

> 用一阶方法近似 TRPO 的“trust region”约束 :contentReference[oaicite:0]{index=0}

---

## 2. Notation（符号定义）

---

### 基本变量

- \( s_t \)：状态（state）
- \( a_t \)：动作（action）
- \( \pi_\theta(a_t | s_t) \)：当前策略
- \( \pi_{\theta_{\text{old}}}(a_t | s_t) \)：旧策略
- \( \theta \)：当前参数
- \( \theta_{\text{old}} \)：更新前参数

---

### Advantage

- \( A_t \)：advantage function

表示：

\[
A_t = Q(s_t, a_t) - V(s_t)
\]

---

### Importance Ratio

\[
r_t(\theta) =
\frac{
\pi_\theta(a_t | s_t)
}{
\pi_{\theta_{\text{old}}}(a_t | s_t)
}
\]

含义：

- 当前策略 vs 旧策略 在 action 上的变化

---

## 3. PPO 提出的两个核心方法（论文重点）

PPO 实际上提出的是 **一族方法（family）**：

---

# 🔴 方法1：Clipped Surrogate Objective（最核心）

---

## 3.1 原始 Policy Gradient（问题）

标准目标：

\[
L^{PG}(\theta) =
\mathbb{E}[r_t(\theta) A_t]
\]

问题：

- \( r_t \) 无约束 → 可以非常大
- 导致：
  - 梯度爆炸
  - policy collapse

---

## 3.2 PPO Clipped Objective

\[
L^{CLIP}(\theta) =
\mathbb{E}
\left[
\min \left(
r_t(\theta) A_t,\,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
\]

---

## 3.3 符号说明

- \( \epsilon \)：clipping threshold（通常 0.1~0.2）
- \( \text{clip}(x, a, b) \)：限制在区间 \([a,b]\)

---

## 3.4 核心机制（非常重要）

分两种情况：

---

### 情况1：\( A_t > 0 \)（好的动作）

目标：

\[
\max r_t(\theta)
\]

但 PPO 限制：

\[
r_t(\theta) \leq 1 + \epsilon
\]

👉 超过后不再增加

---

### 情况2：\( A_t < 0 \)（坏的动作）

目标：

\[
\min r_t(\theta)
\]

但 PPO 限制：

\[
r_t(\theta) \geq 1 - \epsilon
\]

---

## 3.5 梯度解释（关键）

当：

\[
r_t(\theta) \notin [1-\epsilon, 1+\epsilon]
\]

梯度变为：

\[
\nabla_\theta L = 0
\]

👉 update 被“截断”

---

## 3.6 本质理解

> PPO 用 clip 构造一个“pessimistic lower bound”

即：

- 防止 policy 更新过大
- 模拟 TRPO 的 trust region

---

## 3.7 解决的问题

| 问题 | PPO 解决方式 |
|------|-------------|
| policy 更新过大 | clipping |
| training instability | 梯度截断 |
| sample inefficiency | 可多次更新 |

---

# 🔴 方法2：KL Penalty Variant（第二种形式）

---

## 3.8 KL-penalty Objective

\[
L^{KL}(\theta) =
\mathbb{E}
\left[
r_t(\theta) A_t
- \beta D_{KL}(\pi_\theta || \pi_{\theta_{\text{old}}})
\right]
\]

---

## 3.9 符号说明

- \( \beta \)：KL penalty coefficient
- \( D_{KL} \)：KL divergence

---

## 3.10 机制

直接惩罚：

\[
\pi_\theta \text{ 偏离 } \pi_{\text{old}}
\]

---

## 3.11 与 TRPO 关系

TRPO：

\[
\max L \quad \text{s.t. } D_{KL} \leq \delta
\]

PPO：

\[
\max L - \beta D_{KL}
\]

👉 从“硬约束”变为“软约束”

---

## 3.12 解决的问题

- 避免复杂二阶优化
- 保留 trust region 思想

---

# 🔴 方法3：Multiple Epochs + Minibatch（工程关键）

---

## 3.13 标准 Policy Gradient

- 每个 sample 只用一次

---

## 3.14 PPO 改进

同一 batch：

\[
\text{update 多次}
\]

---

## 3.15 数学形式

对同一数据：

\[
\theta \leftarrow \theta + \alpha \nabla L
\quad (\text{多次})
\]

---

## 3.16 本质

> 利用 importance sampling 允许重复使用数据 :contentReference[oaicite:1]{index=1}

---

## 3.17 解决的问题

| 问题 | 改善 |
|------|------|
| sample inefficiency | 多次 reuse |
| 训练成本高 | 减少环境交互 |

---

# 🔴 方法4：Generalized Advantage Estimation（实践配套）

（论文中作为关键组件）

---

## 3.18 定义

\[
A_t^{GAE} =
\sum_{l=0}^{\infty}
(\gamma \lambda)^l \delta_{t+l}
\]

---

### TD error

\[
\delta_t =
r_t + \gamma V(s_{t+1}) - V(s_t)
\]

---

## 3.19 符号说明

- \( \gamma \)：discount factor
- \( \lambda \)：bias-variance tradeoff

---

## 3.20 作用

控制：

- bias vs variance

---

## 3.21 解决的问题

- 高 variance
- 不稳定 advantage

---

## 4. PPO Pipeline（必须掌握）

---

### Step 1：采样

\[
(s_t, a_t) \sim \pi_{\theta_{\text{old}}}
\]

---

### Step 2：计算 reward / advantage

\[
A_t \leftarrow \text{GAE}
\]

---

### Step 3：计算 ratio

\[
r_t(\theta)
\]

---

### Step 4：优化目标

使用：

- clip 或 KL

---

### Step 5：多次更新

- 多 epoch
- minibatch SGD

---

### Step 6：更新 old policy

\[
\theta_{\text{old}} \leftarrow \theta
\]

---

## 5. PPO 的核心贡献总结（重点）

---

### 贡献1：Clipped Objective

👉 用简单方法实现 trust region

---

### 贡献2：KL penalty（替代 TRPO）

👉 去掉二阶优化

---

### 贡献3：多次更新（data reuse）

👉 提升 sample efficiency

---

### 贡献4：稳定 + 简单

👉 实用性极强（工业可用）

---

## 6. PPO 改善了什么（必须记住）

---

### 6.1 相比 Policy Gradient

- 降低 variance
- 防止发散

---

### 6.2 相比 TRPO

| 维度 | TRPO | PPO |
|------|------|-----|
| 优化 | 二阶 | 一阶 |
| 实现 | 复杂 | 简单 |
| 稳定性 | 高 | 接近 |
| 速度 | 慢 | 快 |

---

### 6.3 核心本质

> PPO = “用 clipping 模拟 trust region 的 policy gradient”

---

## 7. PPO 的局限（为 GRPO / GSPO 铺垫）

---

### 问题1：token-level ratio（在LLM中）

\[
\prod r_t
\]

→ 高方差

---

### 问题2：依赖 value function

- critic 训练困难

---

### 问题3：credit assignment 粗糙

---

## 8. 一句话总结

> PPO = 用 clipping 限制 policy 更新幅度，实现稳定、高效、可复用数据的 policy gradient 方法