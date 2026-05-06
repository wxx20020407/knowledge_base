# GSPO: Group Sequence Policy Optimization 论文笔记（arXiv:2507.18071）

---

## 1. Introduction

### 问题：Token-level Ratio 的根本缺陷

PPO 和 GRPO 都使用 **token-level importance ratio**：

$$r_t(\theta) = \frac{\pi_\theta(y_t | x, y_{\lt t})}{\pi_{\theta_{\text{old}}}(y_t | x, y_{\lt t})}$$

在更新策略时，需要计算整个序列的 ratio：

$$R(\theta) = \prod_{t=1}^{T} r_t(\theta)$$

这个连乘带来了一个根本性问题：**方差随序列长度指数增长。**

**为什么方差会爆炸？**

假设每个 token 的 ratio 有独立的噪声 $\epsilon_t$，则：

$$\prod_{t=1}^{T} (1 + \epsilon_t) \approx 1 + \sum_{t=1}^{T} \epsilon_t + \text{高阶项}$$

当 $T$ 很大时（LLM 的推理链可能有几千个 token），连乘的方差正比于 $T$。这意味着：
- 短序列的 ratio 还算准确
- 长序列的 ratio 几乎完全是噪声

**更根本的问题：** Reward 是序列级的（"最终答案对不对"），但 ratio 是 token 级的。这种粒度不匹配导致优化信号被 token-level 的噪声淹没。

### GSPO 的核心思想

> 将优化从 token-level 提升到 sequence-level，从根本上解决长序列的高方差问题。

---

## 2. Notation（符号定义）

### 基本变量

- $x$：输入 prompt
- $y = (y_1, y_2, ..., y_T)$：生成的输出序列
- $T$：序列长度（token 数量）
- $y_t$：第 $t$ 个 token
- $y_{\lt t} = (y_1, ..., y_{t-1})$：第 $t$ 步之前的历史 token

### Policy 相关

- $\pi_\theta(y|x)$：当前策略（sequence-level 概率）
- $\pi_{\theta_{\text{old}}}(y|x)$：旧策略

token-level 概率：

$$\pi_\theta(y_t | x, y_{\lt t})$$

sequence-level 概率（token 概率的连乘）：

$$\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t | x, y_{\lt t})$$

### Reward 与 Advantage

- $r(x, y)$：reward function，对整个序列评分
- $r_i$：第 $i$ 个样本的 reward
- $K$：group size（同一 prompt 采样的序列数）

Group-based baseline：

$$\bar{r} = \frac{1}{K} \sum_{j=1}^{K} r_j$$

Advantage：

$$A_i = r_i - \bar{r}$$

---

## 3. PPO 和 GRPO 的问题回顾

### PPO 的 Token-level Ratio

$$r_t(\theta) = \frac{\pi_\theta(y_t | x, y_{\lt t})}{\pi_{\theta_{\text{old}}}(y_t | x, y_{\lt t})}$$

每个 token 有一个 ratio，需要分别计算 advantage 和做 clipping。

### GRPO 的改进与遗留问题

GRPO 去掉了 Critic，用 group relative advantage 替代。但 importance ratio 仍是 token-level 的：

$$\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} = \prod_{t=1}^{T} \frac{\pi_\theta(o_{i,t} | q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,\lt t})}$$

**方差问题依然存在：**

$$\text{Var}(R) \propto T$$

序列越长，ratio 的方差越大。在长推理链（如数学证明）中，这个问题尤为严重。

---

## 4. GSPO 核心方法

### 4.1 Sequence-level Importance Ratio

GSPO 直接在序列级别定义 importance ratio：

$$R(\theta) = \frac{\pi_\theta(y|x)}{\pi_{\theta_{\text{old}}}(y|x)}$$

展开为 token-level 的连乘：

$$R(\theta) = \prod_{t=1}^{T} \frac{\pi_\theta(y_t | x, y_{\lt t})}{\pi_{\theta_{\text{old}}}(y_t | x, y_{\lt t})}$$

这看起来和 GRPO 的连乘一样？关键区别在下一步——length normalization。

### 4.2 Length Normalization（关键创新）

直接使用 $R(\theta)$ 仍然是问题：$T$ 越大，$R(\theta)$ 的方差越大。GSPO 的核心创新是对 $R(\theta)$ 做长度归一化：

$$\tilde{R}(\theta) = R(\theta)^{\frac{1}{T}}$$

**为什么取 $T$ 次方根？**

取对数可以看得更清楚：

$$\log \tilde{R}(\theta) = \frac{1}{T} \log R(\theta) = \frac{1}{T} \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t | x, y_{\lt t})}{\pi_{\theta_{\text{old}}}(y_t | x, y_{\lt t})}$$

这是一个**平均 log-ratio**——把连乘变成了平均。每个 token 的贡献被等权平均，不再受序列长度影响。

**方差分析：**

- 原始 $R(\theta)$：$\text{Var} \propto T$（方差随长度线性增长）
- 归一化后 $\tilde{R}(\theta)$：$\text{Var} \propto \frac{1}{T} \cdot T = 1$（方差与长度无关）

这是一个非常优雅的修复：**用一个简单的 $1/T$ 归一化，就把方差从 $O(T)$ 降到了 $O(1)$。**

### 4.3 Sequence-level PPO Objective

用归一化后的 ratio 替代 PPO 的 token-level ratio：

$$L_{\text{GSPO}} = \mathbb{E} \left[ \min \left( \tilde{R}(\theta) A, \; \text{clip}(\tilde{R}(\theta), 1-\epsilon, 1+\epsilon) A \right) \right]$$

对比 PPO 的目标：

$$L_{\text{PPO}} = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

关键区别：
- PPO：token-level ratio $r_t$ × token-level advantage $A_t$
- GSPO：sequence-level ratio $\tilde{R}$ × sequence-level advantage $A$

**粒度对齐了！** Ratio 和 advantage 都是序列级的，不再有粒度不匹配的问题。

### 4.4 Group-based GSPO（最终形式）

结合 GRPO 的 group relative advantage，GSPO 的最终目标函数：

$$L = \frac{1}{K} \sum_{i=1}^{K} \min \left( \tilde{R}_i(\theta) A_i, \; \text{clip}(\tilde{R}_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right)$$

其中：
- $\tilde{R}_i(\theta) = R_i(\theta)^{1/T_i}$：第 $i$ 个序列的长度归一化 ratio
- $A_i = r_i - \bar{r}$：group relative advantage

### 4.5 梯度形式

$$\nabla_\theta L = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(y|x) \cdot w(y) \right]$$

其中权重函数：

$$w(y) = \begin{cases} A \cdot \tilde{R}(\theta), & \text{未被裁剪} \\ A \cdot \text{clip}(\tilde{R}(\theta)), & \text{被裁剪} \end{cases}$$

**直觉：** 梯度方向由 $\log \pi_\theta(y|x)$ 决定（"如何调整策略让这个输出更可能/更不可能"），梯度大小由 $w(y)$ 控制（"调整多少"）。$\tilde{R}(\theta)$ 和 clipping 共同决定了步长。

---

## 5. GSPO-token（扩展变体）

GSPO 还提出了一种 token-level backprop 的变体：

$$L = \sum_{t=1}^{T} \tilde{R}(\theta) \cdot A$$

即：**所有 token 共用同一个 sequence-level ratio $\tilde{R}(\theta)$**，但保留 token 级别的反向传播。

**为什么这样设计？**

- 直接对 $\tilde{R}(\theta) A$ 做反向传播，梯度只通过 $\tilde{R}(\theta)$ 流动
- 但 $\tilde{R}(\theta)$ 本身就是 token-level 概率的函数，所以梯度自然会分配到每个 token
- 好处是：ratio 是序列级的（低方差），但梯度传播是 token 级的（精确）

---

## 6. Variance Analysis（核心理论）

这是 GSPO 最重要的理论贡献：

### GRPO（Token-level Ratio）

$$\text{Var}(R_{\text{GRPO}}) \propto T$$

原因：$R = \prod_{t=1}^T r_t$，每个 $r_t$ 引入独立噪声，连乘后方差累积。

### GSPO（Length-normalized Ratio）

$$\text{Var}(\tilde{R}_{\text{GSPO}}) \propto 1$$

原因：$\log \tilde{R} = \frac{1}{T} \sum_{t=1}^T \log r_t$，平均化操作将方差除以 $T$，抵消了连乘带来的方差增长。

### 实际影响

| 序列长度 | GRPO ratio 方差 | GSPO ratio 方差 |
|---------|----------------|----------------|
| $T = 100$ | $\propto 100$ | $\propto 1$ |
| $T = 1000$ | $\propto 1000$ | $\propto 1$ |
| $T = 5000$ | $\propto 5000$ | $\propto 1$ |

在长推理链场景下，GSPO 的优势是压倒性的。

---

## 7. 实验结论

### 7.1 收敛速度

GSPO 收敛更快——用更少的训练步数就能达到同等甚至更高的 reward。这是因为梯度信号更准确（方差更低），每次更新都更有效。

### 7.2 稳定性

- **GSPO**：训练曲线平滑，没有 reward collapse（reward 突然下跌）
- **GRPO**：波动大，容易在训练中途崩溃

### 7.3 MoE 训练

GSPO 在 Mixture-of-Experts（MoE）模型上的优势尤其明显：
- 不依赖 routing replay（路由重放）
- 训练更稳定
- MoE 的 routing 策略不容易被 ratio 的噪声干扰

### 7.4 Clipping 行为

GSPO 的 clipping 频率更低——因为 $\tilde{R}(\theta)$ 的方差小，ratio 不容易超出 $[1-\epsilon, 1+\epsilon]$ 的范围。这意味着更多的梯度信号被"有效利用"了，而不是被 clipping 截断。

### 7.5 最终性能

在 reasoning（推理）和 instruction following（指令跟随）任务上全面提升。GSPO 已被用于工业模型（如 Qwen 系列）的训练。

---

## 8. 与其他方法的关系

| 方法 | Ratio 粒度 | Advantage 来源 | 需要 Critic？ | 方差 |
|------|-----------|---------------|-------------|------|
| PPO | Token-level | Critic | 是 | 高 |
| GRPO | Token-level | Group relative | 否 | 高 |
| DAPO | Token-level（修复归一化） | Group relative | 否 | 中 |
| GSPO | **Sequence-level** | Group relative | 否 | **低** |

GSPO 可以看作是 GRPO 的"方差修复版"——保持了 GRPO 去掉 Critic 的简洁性，同时通过 sequence-level ratio 解决了方差问题。

DAPO 和 GSPO 的改进是正交的：
- DAPO 修复的是 loss 归一化、clipping 策略、长度控制
- GSPO 修复的是 ratio 的粒度和方差
- 两者可以组合使用

---

## 9. 一句话总结

> GSPO = 用 sequence-level 的长度归一化 ratio 替代 token-level ratio，将方差从 $O(T)$ 降至 $O(1)$，从根本上解决长序列 RL 的高方差问题。
