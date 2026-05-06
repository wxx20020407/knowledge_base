# DAPO: LLM Reinforcement Learning System at Scale 笔记（arXiv:2503.14476）

---

## 1. Introduction

DAPO（Decoupled/Advanced Policy Optimization）是在 GRPO 基础上的**系统级改进**。它不是一个全新的算法，而是对 GRPO 的四个关键组件进行了针对性修复。

### GRPO 在大规模训练中暴露的问题

当把 GRPO 应用到大规模 LLM 训练（如 70B+ 模型）时，出现了三个严重问题：

**问题 1：Entropy Collapse（熵坍塌）**

训练过程中，策略的熵急剧下降，模型的输出变得越来越单一，探索能力丧失。这类似于"过早收敛"——模型过早地锁定了某类输出，丧失了发现更好输出的能力。

**问题 2：长序列训练不公平**

GRPO 的 loss 对每个序列取平均，这意味着长序列中每个 token 的梯度被"稀释"了。一个 1000 token 的好回答和一个 100 token 的好回答，对模型更新的贡献竟然一样——这显然不合理。

**问题 3：Exploration 不足**

GRPO 的对称 clipping 对好样本和坏样本施加相同的约束，限制了模型向好方向更新的幅度。

### DAPO 的解决思路

DAPO 从四个维度系统性地修复 GRPO：

1. **Token-level Loss**：解决长序列梯度稀释
2. **Clip-Higher**：非对称 clipping，增强 exploration
3. **Overlong Filtering / Penalty**：控制长度偏置
4. **Dynamic Sampling**：按难度分配采样资源

---

## 2. Notation（符号定义）

### 基本变量

- $q$：输入 prompt
- $o_i = (o_{i,1}, o_{i,2}, ..., o_{i,T_i})$：第 $i$ 个生成序列
- $T_i$：第 $i$ 个序列的长度
- $G$：group size（同一 prompt 的采样数量）

### Policy

- $\pi_\theta(o_i | q)$：当前策略（sequence-level 概率）
- $\pi_{\theta_{\text{old}}}(o_i | q)$：旧策略

token-level 概率：

$$\pi_\theta(o_{i,t} | q, o_{i,\lt t})$$

### Reward

- $r_i = r(q, o_i)$：第 $i$ 个输出的 reward（序列级）

### Advantage（继承 GRPO）

$$A_i = \frac{r_i - \mu}{\sigma}$$

其中：

$$\mu = \frac{1}{G} \sum_{j=1}^G r_j, \quad \sigma = \sqrt{\frac{1}{G} \sum_{j=1}^G (r_j - \mu)^2}$$

---

## 3. GRPO 基础 Loss（作为对比基线）

GRPO 的 loss 函数：

$$L_{\text{GRPO}} = \frac{1}{G} \sum_{i=1}^G \frac{1}{T_i} \sum_{t=1}^{T_i} \min\left( w_{i,t} A_i, \; \text{clip}(w_{i,t}, 1-\epsilon, 1+\epsilon) A_i \right)$$

其中 token-level importance ratio：

$$w_{i,t} = \frac{\pi_\theta(o_{i,t} | q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,\lt t})}$$

注意这里的 $\frac{1}{T_i}$——这是问题的根源。

---

## 4. DAPO 的四个核心改进

---

## 4.1 改进一：Token-level Loss（解决"长度不公平"）

### 问题分析

GRPO 的 loss 中有 $\frac{1}{T_i}$，即对每个序列的 token loss 取平均。这意味着：

$$\text{每个序列对总 loss 的贡献} = \frac{1}{T_i} \sum_{t=1}^{T_i} L_t$$

无论序列长度如何，每个序列的总贡献是"被平均过的"。具体来说：

- 短序列（$T_i = 100$）：每个 token 的梯度权重为 $\frac{1}{100}$
- 长序列（$T_i = 1000$）：每个 token 的梯度权重为 $\frac{1}{1000}$

**后果：** 长序列中每个 token 的梯度被稀释了 10 倍。如果一个长回答是好回答（$A_i > 0$），模型从中学到的东西比一个短好回答少得多。这在长推理链（long chain-of-thought）的训练中尤为严重——模型倾向于生成短回答以获取更大的梯度信号。

### DAPO 的修复

取消 sequence-level 平均，直接在 token 级别求和：

$$L_{\text{DAPO}} = \frac{1}{G} \sum_{i=1}^G \frac{1}{\sum_{i=1}^G T_i} \sum_{t=1}^{T_i} \min\left( w_{i,t} A_i, \; \text{clip}(w_{i,t}, 1-\epsilon, 1+\epsilon) A_i \right)$$

关键变化：分母从"每个序列各自平均"变为"所有 token 统一归一化"。

### 梯度对比

| | GRPO | DAPO |
|---|---|---|
| 每个 token 的梯度权重 | $\propto \frac{1}{T_i}$ | $\propto 1$（统一） |
| 长序列 vs 短序列 | 长序列 token 梯度更小 | 所有 token 梯度相同 |

**本质：** GRPO 是"每个序列权重相等"，DAPO 是"每个 token 权重相等"。后者更合理，因为每个 token 都是模型需要学习的一个决策点。

---

## 4.2 改进二：Clip-Higher（非对称 Clipping）

### 问题分析

GRPO 使用对称 clipping：

$$w \in [1-\epsilon, \; 1+\epsilon]$$

例如 $\epsilon = 0.2$，则 $w \in [0.8, 1.2]$。

问题：当 $A_i > 0$（好样本）时，$w$ 不能超过 $1+\epsilon$；当 $A_i < 0$（坏样本）时，$w$ 不能低于 $1-\epsilon$。上下界是对称的。

**后果：** 好样本的更新被同样严格地限制了。在 entropy 本来就低的情况下，模型很难增大好样本的概率——探索能力进一步被压制。

### DAPO 的修复

使用非对称 clipping：

$$w \in [1-\epsilon_l, \; 1+\epsilon_h]$$

其中 $\epsilon_h > \epsilon_l$。例如 $\epsilon_l = 0.1, \epsilon_h = 0.3$，则 $w \in [0.9, 1.3]$。

### 机制详解

**当 $A_i > 0$（好样本）：**
- 上界从 $1+\epsilon$ 扩大到 $1+\epsilon_h$
- 允许更大的 $w$ → 好样本的概率可以增加更多
- 增强 exploitation（利用已知的好策略）

**当 $A_i < 0$（坏样本）：**
- 下界从 $1-\epsilon$ 收紧到 $1-\epsilon_l$
- $w$ 的下限更严格 → 坏样本的概率被更积极地降低
- 更强地抑制坏方向

**直觉：** 好样本"放行"得更宽松，坏样本"惩罚"得更严格。这是一种"奖惩不对称"的策略。

### 与 Entropy Collapse 的关系

Entropy collapse 的本质是：策略的输出分布变得过于集中（熵太低），模型丧失了探索新输出的能力。

Clip-Higher 通过允许好样本的概率更大范围地增加，间接维持了策略的多样性——因为好样本被鼓励了，模型有动力去探索更多不同的输出。

---

## 4.3 改进三：Overlong Filtering / Penalty（长度控制）

### 问题分析

在 RL 训练中，模型可能发现一个"hack"：**生成更长的回答，以获取更多 reward 机会。**

直觉：如果 reward 是"最终答案是否正确"，那么生成更长的推理链意味着更多的"尝试机会"——即使中间大部分是废话，只要最后碰对了就行。这导致：
- 输出越来越长
- 训练效率下降（长序列消耗更多计算）
- Reward hacking（用长度换取正确率）

### DAPO 的两种策略

**策略 1：Hard Filtering（硬过滤）**

定义最大长度 $T_{\max}$，直接丢弃超过长度的样本：

$$o_i \text{ 被丢弃 if } T_i > T_{\max}$$

**本质：** 改变了 sampling distribution：

$$P(o|q) \rightarrow P(o|q, T \leq T_{\max})$$

**优点：** 简单直接
**缺点：** 丢掉太多样本，降低 sample efficiency

**策略 2：Soft Penalty（软惩罚）**

对长序列在 reward 上加一个连续惩罚：

$$r'_i = r_i - \lambda \cdot \max(0, T_i - T_{\text{target}})$$

其中：
- $\lambda$：惩罚系数
- $T_{\text{target}}$：目标长度
- 只有当 $T_i > T_{\text{target}}$ 时才触发惩罚

**直觉：** 超过目标长度越多，惩罚越重。这是一种"温和的长度约束"。

### 对 Advantage 的影响

软惩罚改变了 reward，进而改变了 advantage：

$$A'_i = \frac{r'_i - \mu'}{\sigma'}$$

模型会学到：长回答如果不能带来足够高的原始 reward，就会因为长度惩罚而变成"差回答"。

---

## 4.4 改进四：Dynamic Sampling（动态采样）

### 问题分析

GRPO 对所有 prompt 使用相同的 group size $G$。但不同的 prompt 难度不同：

- **简单 prompt**：大部分采样都能答对，$\sigma \approx 0$，学习信号很弱
- **困难 prompt**：只有少数采样答对，$\sigma$ 大，学习信号强

固定 $G$ 意味着：简单 prompt 浪费了大量采样（大部分结果相同），而困难 prompt 可能采样不够。

### DAPO 的修复

根据 prompt 的难度动态调整采样数量：

$$G(q) \propto \text{difficulty}(q)$$

具体实现：如果一个 prompt 的所有采样结果 reward 都相同（$\sigma = 0$），则跳过这个 prompt 或增加采样数；如果 reward 分化明显，保持或减少采样数。

### 作用

- **简单问题**：少采样，节省计算
- **困难问题**：多采样，获取更多学习信号
- **结果**：更高的 sample efficiency，训练资源被更合理地分配

**直觉：** 好比考试复习——简单的知识点不需要做太多题，难题才需要多练。Dynamic sampling 就是"把计算资源花在刀刃上"。

---

## 5. DAPO vs GRPO 对比

| 维度 | GRPO | DAPO |
|------|------|------|
| Loss 归一化 | sequence-mean（$\frac{1}{T_i}$） | token-level（统一归一化） |
| Clipping | 对称（$[1-\epsilon, 1+\epsilon]$） | 非对称（$[1-\epsilon_l, 1+\epsilon_h]$） |
| 长度控制 | 无 | filtering + penalty |
| 采样策略 | 固定 $G$ | 动态 $G(q)$ |
| Entropy | 容易 collapse | 更好维持 |
| 长序列信号 | 被稀释 | 不被稀释 |

---

## 6. 核心理论 Insight

### 6.1 Entropy Collapse 的机制

GRPO 中 entropy 下降的原因链：

1. 对称 clipping 限制了好样本的概率增长
2. 好样本和坏样本被同等对待
3. 模型逐渐"锁死"在某类输出上
4. 策略熵急剧下降
5. 探索能力丧失，训练停滞

DAPO 的 Clip-Higher 打破了这个循环：好样本被更积极地鼓励，维持了输出多样性。

### 6.2 长度偏置的机制

GRPO 中长度偏置的原因链：

1. Loss 对每个序列取平均（$\frac{1}{T_i}$）
2. 长序列中每个 token 的梯度更小
3. 模型发现长回答可以"稀释"梯度，减少对坏结果的惩罚
4. 模型倾向生成长回答
5. 长回答消耗更多计算，且可能 reward hack

DAPO 的 Token-level Loss + Overlong Penalty 同时解决了梯度稀释和长度奖励两个问题。

### 6.3 RL 的本质（再次验证）

> RL 改变的是输出分布，而不是模型能力。

DAPO 的所有改进都是在更好地控制"如何改变输出分布"——让改变更高效、更公平、更不容易走偏。

---

## 7. 实验结论

### 性能提升来源

DAPO 的性能提升不是来自更强的模型架构，而是来自**更好的优化轨迹**——更合理的梯度分配、更强的探索能力、更公平的长度处理。

### 核心优势

- **更稳定**：训练曲线更平滑，不容易崩溃
- **更高 entropy**：维持输出多样性
- **更强 exploration**：发现更多好的输出模式
- **更公平**：长序列和短序列受到同等重视

---

## 8. 一句话总结

> DAPO = 从 loss 归一化、clipping 策略、长度控制、采样分配四个维度系统性修复 GRPO。本质是更好地控制梯度分布与数据分布，使大规模 RL 训练更稳定、更高效。
