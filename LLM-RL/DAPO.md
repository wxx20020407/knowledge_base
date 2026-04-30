# DAPO: LLM Reinforcement Learning System at Scale 笔记（arXiv:2503.14476）

---

## 1. Introduction（极简）

DAPO（Decoupled/Advanced Policy Optimization）是在 GRPO 基础上的系统级改进：

核心目标：

> 解决 GRPO 在大规模 RL 中的三个关键问题：
>
> - entropy collapse（输出退化）
> - 长序列训练不公平
> - exploration 不足

---

## 2. Notation（符号定义）

---

### 基本变量

- \( q \)：输入 prompt
- \( o_i = (o_{i,1}, ..., o_{i,T_i}) \)：第 \( i \) 个生成序列
- \( T_i \)：该序列长度
- \( G \)：group size

---

### Policy

- \( \pi_\theta(o_i | q) \)：当前策略
- \( \pi_{\theta_{\text{old}}}(o_i | q) \)：旧策略

token-level：

\[
\pi_\theta(o_{i,t} | q, o_{i,<t})
\]

---

### Reward

- \( r_i = r(q, o_i) \)

---

### Advantage（继承 GRPO）

\[
A_i = \frac{r_i - \mu}{\sigma}
\]

其中：

\[
\mu = \frac{1}{G} \sum_{j=1}^G r_j,\quad
\sigma = \sqrt{\frac{1}{G} \sum_{j=1}^G (r_j - \mu)^2}
\]

---

## 3. GRPO 基础 Loss（对比基线）

\[
L_{\text{GRPO}} =
\frac{1}{G} \sum_{i=1}^G
\frac{1}{T_i}
\sum_{t=1}^{T_i}
\min\left(
w_{i,t} A_i,\,
\text{clip}(w_{i,t}, 1-\epsilon, 1+\epsilon) A_i
\right)
\]

其中：

\[
w_{i,t} =
\frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}
\]

---

## 4. DAPO 四个核心改进（重点）

---

# 🔴 核心1：Token-level Loss（解决“长度不公平”）

---

### 问题（GRPO）

GRPO 的 loss：

\[
\frac{1}{T_i} \sum_{t=1}^{T_i}
\]

意味着：

> 每个序列被平均 → 长序列中每个 token 的梯度更小

即：

\[
\text{gradient per token} \propto \frac{1}{T_i}
\]

👉 长序列 token 被“稀释”

---

### DAPO 改法

取消 sequence-level 平均，直接 token-level：

\[
L =
\frac{1}{G}
\sum_{i=1}^{G}
\sum_{t=1}^{T_i}
\min\left(
w_{i,t} A_i,\,
\text{clip}(\cdot) A_i
\right)
\]

---

### 本质解释（非常关键）

原来：

\[
\text{每个 sequence 权重相等}
\]

现在：

\[
\text{每个 token 权重相等}
\]

---

### 梯度变化

GRPO：

\[
\nabla \propto \frac{1}{T_i}
\]

DAPO：

\[
\nabla \propto 1
\]

---

### 结论

> DAPO 解决了“长序列 learning signal 被稀释”的问题

---

# 🔴 核心2：Clip-Higher（非对称 clipping）

---

### GRPO clipping

\[
w \in [1-\epsilon,\; 1+\epsilon]
\]

问题：

- 上下对称
- 限制 exploration

---

### DAPO clipping

\[
w \in [1-\epsilon_l,\; 1+\epsilon_h]
\]

其中：

- \( \epsilon_h > \epsilon_l \)

---

### 本质解释

允许：

\[
w > 1 \quad （鼓励好样本）
\]

但限制：

\[
w < 1 \quad （抑制坏样本）
\]

---

### 梯度影响

当 \( A_i > 0 \)：

- 上界更大 → 更强更新

当 \( A_i < 0 \)：

- 下界更严格 → 抑制坏方向

---

### 结论

> 提升 exploitation + exploration 不对称能力

---

# 🔴 核心3：Overlong Filtering（长度过滤）

---

### 问题

RL 会倾向：

\[
\text{生成更长序列} \Rightarrow \text{获得更多 reward机会}
\]

---

### DAPO 策略

定义最大长度 \( T_{\max} \)

过滤：

\[
o_i \text{ 被丢弃 if } T_i > T_{\max}
\]

---

### 本质

这是一个：

> 数据分布约束（不是 loss）

---

### 作用

防止：

- 长输出 bias
- reward hacking

---

### 数学意义

改变 sampling distribution：

\[
P(o|q) \rightarrow P(o|q, T \leq T_{\max})
\]

---

# 🔴 核心4：Soft Overlong Penalty（软惩罚）

---

### 问题

硬过滤会：

- 丢掉太多样本
- 降低 sample efficiency

---

### DAPO 改法

对长序列加 penalty：

\[
r'_i = r_i - \lambda \cdot \max(0, T_i - T_{\text{target}})
\]

---

### 符号说明

- \( \lambda \)：惩罚系数
- \( T_{\text{target}} \)：目标长度

---

### 本质

reward shaping：

\[
r \rightarrow r'
\]

---

### 梯度影响

\[
A_i \rightarrow A'_i
\]

→ 直接影响 policy gradient

---

### 结论

> 用连续惩罚替代 hard filtering

---

# 🔴 核心5：Dynamic Sampling（最关键但最容易被忽略）

---

### 问题

GRPO：

\[
G = \text{constant}
\]

→ 所有 prompt 用相同采样数

---

### DAPO 改法

动态调整：

\[
G(q) \propto \text{difficulty}(q)
\]

---

### 本质

改变：

\[
\mathbb{E}_{o \sim \pi}
\]

的采样策略

---

### 作用

- 简单问题：少采样
- 困难问题：多采样

---

### 结果

> 更高 sample efficiency

---

## 5. DAPO vs GRPO（本质区别）

| 维度 | GRPO | DAPO |
|------|------|------|
| loss | sequence-mean | token-level |
| clipping | symmetric | asymmetric |
| length | 无控制 | filtering + penalty |
| sampling | 固定 | 动态 |

---

## 6. 核心理论 insight

---

### 6.1 entropy collapse 问题

GRPO：

\[
\pi_\theta \rightarrow \text{low entropy}
\]

原因：

- clipping 对称
- exploration 不足

---

### 6.2 DAPO 解决方式

- clip-higher → 增强 exploration
- dynamic sampling → 增强数据多样性

---

### 6.3 长度偏置问题

GRPO：

\[
\mathbb{E}[r] \propto T
\]

DAPO：

- filtering + penalty → 控制长度

---

## 7. 实验结论（核心理解）

---

### 7.1 性能提升来源

不是：

- 模型更强

而是：

\[
\text{更好的优化轨迹}
\]

---

### 7.2 RL 的本质（再次验证）

> RL 改变的是输出分布，而不是模型能力

---

### 7.3 DAPO 优势

- 更稳定
- 更高 entropy
- 更强 exploration

---

## 8. 一句话总结

> DAPO = “从 loss / clipping / sampling / reward 四个维度系统性修复 GRPO”，本质是控制梯度分布与数据分布