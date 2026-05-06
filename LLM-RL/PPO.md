# PPO: Proximal Policy Optimization 笔记（arXiv:1707.06347）

---

## 1. Introduction

PPO 的目标：

> 在保证训练稳定性的前提下，提高 policy gradient 的 sample efficiency 和实现简洁性

### 背景：为什么需要 PPO？

在 PPO 之前，policy gradient 方法面临一个根本性的矛盾：**想要大步更新以快速学习，但大步更新又容易导致策略崩溃**。

**Vanilla Policy Gradient 的问题：**

标准 policy gradient 的更新公式为 $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A]$。问题在于，每次更新后策略 $\pi_\theta$ 发生变化，之前采集的数据就"过期"了。如果用过期数据做大更新，策略可能跳到一个很差的区域，导致 reward 断崖式下跌——这就是所谓的 **catastrophic policy collapse**。

**TRPO 的解决方案与代价：**

Schulman et al. 在 2015 年提出了 TRPO（Trust Region Policy Optimization），核心思想是约束新旧策略之间的 KL 散度，确保更新不会太大：

$$\max_\theta \; L(\theta) \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta$$

TRPO 有效解决了稳定性问题，但代价是需要计算 Fisher 信息矩阵的逆（二阶优化），实现复杂、计算昂贵，而且不太方便与其他损失函数（如 value loss、entropy bonus）组合使用。

**PPO 的核心洞察：**

> 用一阶方法（简单梯度下降）近似 TRPO 的"trust region"约束，在保持训练稳定性的同时大幅简化实现。

PPO 本质上回答了一个工程问题：**能不能用最简单的方式，获得 TRPO 级别的稳定性？** 答案是 yes——通过 clipping 或 KL penalty 就够了。

---

## 2. Notation（符号定义）

### 基本变量

- $s_t$：时刻 $t$ 的状态（state），代表环境的观测
- $a_t$：时刻 $t$ 的动作（action），由策略采样得到
- $\pi_\theta(a_t | s_t)$：参数为 $\theta$ 的当前策略，输入状态输出动作的概率分布
- $\pi_{\theta_{\text{old}}}(a_t | s_t)$：上一轮更新前的旧策略
- $\theta$：当前策略参数
- $\theta_{\text{old}}$：更新前的策略参数（用于 importance sampling）

### Advantage Function

$$A_t = Q(s_t, a_t) - V(s_t)$$

**直觉理解：** Advantage 回答的是"在状态 $s_t$ 下，采取动作 $a_t$ 比平均水平好多少？"
- $Q(s_t, a_t)$：采取动作 $a_t$ 后的期望累积回报
- $V(s_t)$：在状态 $s_t$ 下的平均期望回报（baseline）
- $A_t > 0$：这个动作比平均好，应该增加其概率
- $A_t < 0$：这个动作比平均差，应该降低其概率

减去 $V(s_t)$ 作为 baseline 的好处是：不改变梯度的期望（无偏），但显著降低方差。

### Importance Ratio（重要性采样比率）

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

**直觉理解：** $r_t(\theta)$ 衡量新策略 $\pi_\theta$ 相对于旧策略 $\pi_{\theta_{\text{old}}}$，对动作 $a_t$ 的"偏好变化程度"。
- $r_t = 1$：新旧策略对这个动作的概率相同
- $r_t > 1$：新策略更倾向于这个动作
- $r_t < 1$：新策略更不倾向于这个动作

**为什么需要 importance ratio？** 因为数据是用旧策略 $\pi_{\theta_{\text{old}}}$ 采集的，但我们想优化新策略 $\pi_\theta$。Importance sampling 允许我们"修正"这个分布差异，从而复用旧数据。

---

## 3. PPO 提出的核心方法

PPO 实际上提出的是**一族方法（family）**，论文中主要讨论了两种变体：Clipped Surrogate Objective 和 KL Penalty。此外，配合 multiple epochs 更新和 GAE，构成完整的 PPO 算法。

---

## 3.1 方法一：Clipped Surrogate Objective（最核心）

### 从 Policy Gradient 到 PPO

标准 policy gradient 的目标函数：

$$L^{PG}(\theta) = \mathbb{E}_t [r_t(\theta) A_t]$$

这里 $r_t(\theta)$ 是 importance ratio。问题是：当 $r_t(\theta)$ 偏离 1 很多时（即新旧策略差异很大），importance sampling 的方差会急剧增大，导致梯度估计不准确，训练不稳定。

**直觉：** 想象你在一个山谷里，importance ratio 告诉你"新位置比旧位置陡了多少"。如果新旧位置差距太大，这个估计就不可靠了——可能把你引向悬崖。

### PPO Clipped Objective

PPO 的核心创新——用 clipping 限制 $r_t(\theta)$ 的范围：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

其中：
- $\epsilon$：clipping 阈值，通常取 0.1 ~ 0.2
- $\text{clip}(x, a, b) = \max(a, \min(x, b))$：将 $x$ 限制在 $[a, b]$ 区间内

### 核心机制：分情况理解

这个公式看似复杂，但可以分两种情况来理解：

**情况 1：$A_t > 0$（好的动作，应该增加概率）**

此时 $r_t(\theta) A_t$ 是正的，我们想最大化它。但 $\min$ 操作意味着：

$$L = \min(r_t A_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t)$$

当 $r_t > 1+\epsilon$ 时，clip 后的值更小，$\min$ 取 clip 值 → 梯度被截断，不再鼓励进一步增大 $r_t$。

**直觉：** 即使某个动作很好，也不要过度增加它的概率——防止策略"过度自信"。

**情况 2：$A_t < 0$（坏的动作，应该降低概率）**

此时 $r_t(\theta) A_t$ 是负的，我们想最大化它（即让 $r_t$ 更小）。但当 $r_t < 1-\epsilon$ 时，clip 后的值更大（更接近 0），$\min$ 取原始值 → 梯度被截断。

**直觉：** 即使某个动作很差，也不要过度降低它的概率——保持一定的探索能力。

### 梯度视角

当 $r_t(\theta) \in [1-\epsilon, 1+\epsilon]$ 时，clipping 没有生效，梯度正常流动。

当 $r_t(\theta) \notin [1-\epsilon, 1+\epsilon]$ 时，$\min$ 操作会选择 clip 后的值，此时梯度为 0——更新被"截断"。

**本质理解：** PPO 用 clip 构造了一个 **pessimistic lower bound**（悲观下界）。它取的是两个目标中的较小值，相当于在说："如果新旧策略差异太大，我就不信任这个梯度估计了。" 这与 TRPO 的 trust region 思想完全一致，只是实现方式简单得多。

### 与 TRPO 的关系

| | TRPO | PPO-Clip |
|---|---|---|
| 约束方式 | 硬约束（KL $\leq \delta$） | 软约束（clipping） |
| 优化方法 | 二阶（共轭梯度 + 线搜索） | 一阶（SGD/Adam） |
| 实现复杂度 | 高 | 低 |
| 能否组合其他 loss | 不方便 | 可以直接加 |

---

## 3.2 方法二：KL Penalty Variant

### 目标函数

PPO 的第二种形式——直接在目标函数中加入 KL 散度惩罚：

$$L^{KL}(\theta) = \mathbb{E}_t \left[ r_t(\theta) A_t - \beta \, D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \right]$$

其中：
- $\beta$：KL penalty 系数，控制约束强度
- $D_{KL}$：KL 散度，衡量两个分布的差异

### 与 TRPO 的关系

TRPO 是硬约束：

$$\max L(\theta) \quad \text{s.t.} \quad D_{KL} \leq \delta$$

PPO-KL 是软约束（拉格朗日对偶形式）：

$$\max L(\theta) - \beta \, D_{KL}$$

直觉：TRPO 说"KL 散度不能超过 $\delta$"，PPO-KL 说"KL 散度越大，惩罚越重"。两者在数学上是等价的（当 $\beta$ 和 $\delta$ 对应时），但 PPO-KL 的实现简单得多。

### 实际使用

论文中提到，KL Penalty 变体在实践中也可以使用自适应的 $\beta$：如果 KL 散度太大就增大 $\beta$，太小就减小 $\beta$。但 Clipped 变体更常用，因为它更简单、效果也更好。

---

## 3.3 方法三：Multiple Epochs + Minibatch 更新

### 标准 Policy Gradient 的低效

标准 policy gradient 中，每个 batch 的数据只用来计算一次梯度就丢弃了。这是因为 policy 更新后，旧数据的 importance ratio 就不准了。

### PPO 的改进

PPO 利用 importance sampling 允许同一 batch 数据被**多次复用**：

$$\theta \leftarrow \theta + \alpha \nabla_\theta L \quad \text{(重复 K 个 epoch)}$$

**为什么可行？** 因为 PPO 的 clipping 机制保证了即使 $r_t(\theta)$ 偏离 1，梯度也会被截断，不会产生过大的更新。这给了我们"安全地多次使用同一批数据"的空间。

**实际流程：**
1. 采集一批 trajectory 数据
2. 将数据分成若干 minibatch
3. 对每个 minibatch 做 K 次梯度更新（通常 K = 3 ~ 10）
4. 采集新数据，重复

### 解决的问题

| 问题 | 改善 |
|------|------|
| sample inefficiency | 同一数据多次复用 |
| 训练成本高 | 减少与环境交互的次数 |
| GPU 利用率低 | 更多计算用于梯度更新 |

---

## 3.4 方法四：Generalized Advantage Estimation (GAE)

GAE 是 PPO 训练中计算 advantage 的标准方法，由同一组作者在 earlier work 中提出。

### 动机：TD error 的 bias-variance tradeoff

**TD error（时序差分误差）：**

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\delta_t$ 衡量的是"实际获得的回报（$r_t + \gamma V(s_{t+1})$）与预期回报（$V(s_t)$）之间的差距"
- $\gamma$：discount factor，通常 0.99

**问题：** 单步 TD error $\delta_t$ 方差低但有 bias（因为 $V$ 是近似的）；多步回报（如 Monte Carlo return）无 bias 但方差高。

### GAE 的定义

GAE 通过指数加权平均，将不同步数的 TD error 平滑地组合起来：

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\lambda \in [0, 1]$ 是控制 bias-variance tradeoff 的超参数。

### $\lambda$ 的作用

- $\lambda = 0$：$A_t^{GAE} = \delta_t$（纯 TD error，低方差、高 bias）
- $\lambda = 1$：$A_t^{GAE}$ 等价于 Monte Carlo return（无 bias、高方差）
- $\lambda = 0.95$（常用值）：在两者之间取得平衡

**直觉：** $\lambda$ 控制"看多远"。$\lambda$ 越大，越依赖实际回报（更准确但波动更大）；$\lambda$ 越小，越依赖 value function 的估计（更稳定但可能不准）。

### 递推计算

GAE 可以从后往前递推计算，非常高效：

$$A_t^{GAE} = \delta_t + \gamma \lambda A_{t+1}^{GAE}$$

---

## 4. PPO 完整 Pipeline

### Step 1：采样

用当前策略 $\pi_{\theta_{\text{old}}}$ 与环境交互，采集 trajectory 数据：

$$(s_t, a_t, r_t, s_{t+1}) \sim \pi_{\theta_{\text{old}}}$$

### Step 2：计算 value 和 advantage

用当前的 value function $V_\phi(s)$ 计算 TD error 和 GAE advantage：

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

### Step 3：计算 importance ratio

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

### Step 4：优化目标

使用 clipped surrogate objective（或 KL penalty）：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]$$

通常还会加上 value function 的 loss 和 entropy bonus：

$$L(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 H(\pi_\theta)$$

其中 $L^{VF} = (V_\theta(s_t) - V_t^{target})^2$，$H$ 是策略的熵（鼓励探索）。

### Step 5：多次更新

对同一 batch 数据做 K 个 epoch 的 minibatch SGD 更新。

### Step 6：更新旧策略

$$\theta_{\text{old}} \leftarrow \theta$$

然后回到 Step 1，采集新数据。

---

## 5. PPO 的核心贡献总结

| 贡献 | 解决的问题 | 实现方式 |
|------|-----------|---------|
| Clipped Objective | policy 更新过大导致崩溃 | 限制 $r_t$ 的范围 |
| KL Penalty | TRPO 太复杂 | 软约束替代硬约束 |
| Multiple Epochs | sample efficiency 低 | importance sampling 允许数据复用 |
| GAE | advantage 估计方差高 | 指数加权多步 TD error |

### 本质

> PPO = 用 clipping（或 KL penalty）模拟 trust region 的 policy gradient，配合 multiple epochs 和 GAE，实现稳定、高效、可复用数据的策略优化方法。

---

## 6. PPO vs TRPO 对比

| 维度 | TRPO | PPO |
|------|------|-----|
| 优化阶数 | 二阶（Fisher 信息矩阵） | 一阶（SGD/Adam） |
| 实现复杂度 | 高（需要共轭梯度、线搜索） | 低（几行代码） |
| 稳定性 | 高 | 接近 TRPO |
| 训练速度 | 慢 | 快 |
| 能否组合其他 loss | 不方便 | 可以直接加 |
| 工业可用性 | 较差 | 极好（被广泛采用） |

---

## 7. PPO 在 LLM 中的局限

PPO 最初是为强化学习环境（如 Atari、机器人控制）设计的。当把它应用到 LLM 的 RLHF 训练时，出现了几个关键问题：

### 问题 1：Token-level ratio 的高方差

在 LLM 中，一个 action 不是单步动作，而是整个输出序列。token-level 的 importance ratio 是：

$$r(\theta) = \prod_{t=1}^{T} \frac{\pi_\theta(y_t | x, y_{\lt t})}{\pi_{\theta_{\text{old}}}(y_t | x, y_{\lt t})}$$

这个连乘会导致：序列越长，方差越大（指数级增长）。即使每个 token 的 ratio 只偏离 1 一点点，长序列的乘积也会偏离很远。

### 问题 2：Value function（Critic）的训练困难

PPO 需要一个 value function $V(s)$ 来计算 advantage。但在 LLM 中：
- 状态空间是所有可能的 token 序列，极其庞大
- Value function 需要对每个中间状态给出准确估计，这本身就是一个很难的学习问题
- Critic 网络（通常是另一个 LLM）的训练增加了内存和计算开销

### 问题 3：Credit assignment 粗糙

当一个序列的最终 reward 很高（低）时，很难判断是哪个 token 贡献了这个结果。PPO 的 token-level advantage 估计在 LLM 场景下不够精确。

这些问题直接催生了后续的 GRPO、DAPO、GSPO 等方法。

---

## 8. 一句话总结

> PPO = 用 clipping 限制 policy 更新幅度，实现稳定、高效、可复用数据的 policy gradient 方法。它是 LLM RLHF 训练的基础算法，但其 token-level ratio 和 value function 的设计在 LLM 场景下存在局限。
