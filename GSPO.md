# GSPO: Group Sequence Policy Optimization 论文笔记（全符号定义版）

---

https://arxiv.org/pdf/2507.18071

## 1. Introduction（简要）

现有 RLHF / RLVR 方法（如 PPO / GRPO）存在问题：

- 使用 **token-level importance ratio** → 长序列高方差
- reward 是 sequence-level → 优化粒度不匹配

GSPO 的核心思想：

> 将优化从 token-level 提升到 sequence-level

---

## 2. Notation（符号定义）

### 基本变量

- \( x \)：输入 prompt（文本或上下文）
- \( y = (y_1, y_2, ..., y_T) \)：生成的输出序列
- \( T \)：序列长度（token 数量）
- \( y_t \)：第 \( t \) 个 token
- \( y_{<t} = (y_1, ..., y_{t-1}) \)：第 \( t \) 步之前的历史 token

---

### Policy 相关

- \( \pi_\theta(y|x) \)：当前策略（policy），参数为 \( \theta \)
- \( \pi_{\theta_{\text{old}}}(y|x) \)：旧策略（reference policy）

token-level 概率：

\[
\pi_\theta(y_t | x, y_{<t})
\]

sequence-level 概率：

\[
\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t | x, y_{<t})
\]

---

### Reward 与 Advantage

- \( r(x, y) \)：reward function，对整个序列评分
- \( r_i \)：第 \( i \) 个样本的 reward
- \( K \)：group size（同一个 prompt 采样的序列数）

Group-based baseline：

\[
\bar{r} = \frac{1}{K} \sum_{j=1}^{K} r_j
\]

Advantage：

\[
A_i = r_i - \bar{r}
\]

---

## 3. PPO（回顾）

### Token-level importance ratio

\[
r_t(\theta) =
\frac{
\pi_\theta(y_t | x, y_{<t})
}{
\pi_{\theta_{\text{old}}}(y_t | x, y_{<t})
}
\]

含义：

- 衡量当前策略 vs 旧策略 在 token \( y_t \) 上的变化

---

### PPO 目标函数

\[
L_{\text{PPO}} =
\mathbb{E}
\left[
\min \left(
r_t(\theta) A_t,\,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
\]

符号说明：

- \( A_t \)：token-level advantage
- \( \epsilon \)：clipping 超参数（通常 0.1~0.2）
- \( \text{clip}(\cdot) \)：限制 ratio 范围

---

## 4. GRPO（Group-based）

### Advantage（group 内归一化）

\[
A_i = r_i - \frac{1}{K} \sum_{j=1}^{K} r_j
\]

说明：

- \( i \)：第 \( i \) 个样本
- \( K \)：同一 prompt 的采样数量

---

### 问题

- importance ratio 仍是 token-level
- 长序列时：

\[
\prod_{t=1}^{T} r_t(\theta)
\]

→ variance 指数增长

---

## 5. GSPO 核心方法

---

### 5.1 Sequence-level Importance Ratio

定义：

\[
R(\theta) =
\frac{
\pi_\theta(y|x)
}{
\pi_{\theta_{\text{old}}}(y|x)
}
\]

展开：

\[
R(\theta) =
\prod_{t=1}^{T}
\frac{
\pi_\theta(y_t | x, y_{<t})
}{
\pi_{\theta_{\text{old}}}(y_t | x, y_{<t})
}
\]

---

### 5.2 Length Normalization（关键）

定义：

\[
\tilde{R}(\theta) = R(\theta)^{\frac{1}{T}}
\]

符号说明：

- \( T \)：序列长度
- \( \tilde{R}(\theta) \)：长度归一化后的 importance ratio

作用：

- 避免长序列 ratio 爆炸
- 等价于平均 log-prob：

\[
\log \tilde{R}(\theta)
=
\frac{1}{T}
\sum_{t=1}^{T}
\log
\frac{
\pi_\theta(y_t | x, y_{<t})
}{
\pi_{\theta_{\text{old}}}(y_t | x, y_{<t})
}
\]

---

### 5.3 Sequence-level PPO Objective

\[
L_{\text{GSPO}} =
\mathbb{E}
\left[
\min \left(
\tilde{R}(\theta) A,\,
\text{clip}(\tilde{R}(\theta), 1-\epsilon, 1+\epsilon) A
\right)
\right]
\]

符号说明：

- \( A \)：sequence-level advantage
- \( \epsilon \)：clipping 参数

---

### 5.4 Group-based GSPO（最终形式）

对于 group \( \{y_i\}_{i=1}^{K} \)：

\[
L =
\frac{1}{K}
\sum_{i=1}^{K}
\min \left(
\tilde{R}_i(\theta) A_i,\,
\text{clip}(\tilde{R}_i(\theta), 1-\epsilon, 1+\epsilon) A_i
\right)
\]

符号说明：

- \( i \)：第 \( i \) 个序列
- \( \tilde{R}_i(\theta) \)：该序列的 ratio
- \( A_i \)：该序列的 advantage

---

### 5.5 梯度形式

\[
\nabla_\theta L =
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(y|x)
\cdot w(y)
\right]
\]

其中：

\[
w(y) =
\begin{cases}
A \cdot \tilde{R}(\theta), & \text{未被裁剪} \\
A \cdot \text{clip}(\tilde{R}(\theta)), & \text{被裁剪}
\end{cases}
\]

符号说明：

- \( w(y) \)：权重函数
- 控制梯度大小

---

## 6. GSPO-token（扩展）

定义：

\[
\tilde{R}_t = \tilde{R}(\theta)
\]

即：

- 所有 token 共用同一个 sequence-level ratio

loss：

\[
L =
\sum_{t=1}^{T}
\tilde{R}(\theta) A
\]

符号说明：

- \( t \)：token index
- 保留 token-level backprop

---

## 7. Variance Analysis（核心理论）

### GRPO：

\[
\text{Var} \propto T
\]

原因：

- ratio 是乘积
- 每一步引入噪声

---

### GSPO：

\[
\text{Var} \propto 1
\]

原因：

\[
\log \tilde{R} =
\frac{1}{T} \sum \log r_t
\]

→ 平均化降低方差

---

## 8. Experiments（结果总结）

---

### 8.1 收敛速度

- GSPO 收敛更快
- 更少训练步数达到同等 reward

---

### 8.2 稳定性

GSPO：

- 曲线平滑
- 无 reward collapse

GRPO：

- 波动大
- 容易崩溃

---

### 8.3 MoE 训练

GSPO 优势：

- 不依赖 routing replay
- 训练更稳定

---

### 8.4 Clipping 行为

- GSPO clipping 更少
- 梯度更稳定

---

### 8.5 最终性能

- 在 reasoning / instruction 任务上全面提升
- 已用于工业模型（如 Qwen）

---

## 9. 核心总结

### 方法

- sequence-level ratio
- length normalization
- sequence-level clipping

### 理论

- 降低 variance
- 提升 reward 对齐

### 工程

- 更稳定
- 更易扩展（尤其 MoE）

---

## 10. 一句话总结

> GSPO = 用 sequence-level 的 PPO 替代 token-level PPO，从根本解决长序列 RL 的高方差问题