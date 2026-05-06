# GRPO: Group Relative Policy Optimization 笔记（DeepSeekMath, arXiv:2402.03300）

---

## 1. Introduction

DeepSeek 提出 GRPO（Group Relative Policy Optimization），用于替代 PPO 进行 LLM 的强化学习训练。

### 核心问题：PPO 中的 Critic 在 LLM 场景下太重了

要理解 GRPO，首先要理解 PPO 的 Actor-Critic 架构在 LLM 场景下的问题。

### Actor 和 Critic：一对搭档

在 PPO 中，有两个核心角色：

**Actor（演员）= Policy Model**

Actor 是我们真正要优化的模型。它的工作是：给定 prompt，生成回答。在 LLM 场景下，Actor 就是那个语言模型本身。

**Critic（评论家）= Value Model**

Critic 的工作是：评估 Actor 在某个状态下的"预期表现"。具体来说，它估计从状态出发，按照当前策略走下去，期望能获得多少累积回报。

**两者如何配合？**

PPO 用 Critic 的估计来计算 advantage：

$$A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- 实际获得的即时 reward + 下一状态的估计价值（"实际回报"）
- 当前状态的估计价值（"预期回报"）
- 两者之差——"这个动作比预期好多少？"

Actor 根据 advantage 来更新：advantage 为正的动作增加概率，advantage 为负的动作降低概率。

**直觉：** Actor 是"做事的人"，Critic 是"打分的人"。Critic 告诉 Actor"你这一步做得好不好"，Actor 根据反馈调整策略。

### Critic 在 LLM 中的四个问题

**问题 1：状态空间极其庞大**

在传统 RL 中（如 Atari 游戏），状态是游戏画面，维度有限。但在 LLM 中，"状态"是到目前为止的所有 token 序列——状态空间是指数级的。Critic 需要对每一种可能的 token 序列都给出准确的价值估计，这极其困难。

**问题 2：Critic 本身就是一个大模型**

为了处理 LLM 的输入，Critic 通常也是一个与 Actor 同等规模的 LLM。这意味着：
- 内存翻倍（需要同时加载 Actor + Critic）
- 计算翻倍（Critic 的前向/反向传播）
- 训练 Critic 本身就是一个困难的优化问题

**问题 3：Critic 的估计不准会伤害 Actor**

如果 Critic 的价值估计不准，计算出的 advantage 就不准，Actor 就会收到错误的学习信号。在 LLM 的长序列中，这种误差会累积。

**问题 4：Token-level credit assignment 粗糙**

PPO 在 LLM 中对每个 token 计算 advantage，但 reward 通常是整个序列级别的（如"最终答案是否正确"）。如何把序列级 reward 分配到每个 token 上，本身就是一个难题。

### GRPO 的核心洞察

> 不用 Critic 来估计 advantage，而是用同一 prompt 下多个采样结果的相对 reward 来代替。

具体来说：对同一个 prompt 生成多个回答，用这些回答的 reward 的均值和标准差做归一化，得到 group-relative advantage。这样就完全不需要 Critic 了。

---

## 2. Notation（符号定义）

### 基本变量

- q：输入问题（query / prompt）
- o：模型生成的输出序列（output）
- o_i：第 i 个采样输出
- G：group size——同一 prompt 的采样数量（通常 8 ~ 64）

### Policy

- 当前策略（Actor），给定 prompt 生成输出的概率
- 旧策略（用于 importance sampling）
- reference model（通常是 SFT 模型，用于 KL regularization）

token-level 概率分解：

$$\pi_\theta(o \mid q) = \prod_{t=1}^{T} \pi_\theta(o_t \mid q, o_{\lt t})$$

### Reward

reward 来源：
- **Rule-based**（规则）：如数学答案是否正确、代码是否通过测试
- **Format-based**（格式）：输出是否符合要求的结构

### Group Statistics

均值（group 内平均 reward）：

$$\mu = \frac{1}{G} \sum_{j=1}^{G} r_j$$

标准差（group 内 reward 的离散程度）：

$$\sigma = \sqrt{\frac{1}{G} \sum_{j=1}^{G} (r_j - \mu)^2}$$

### Advantage（GRPO 的核心创新）

$$A_i = \frac{r_i - \mu}{\sigma}$$

**直觉：** A_i 表示第 i 个回答在 group 内的"相对表现"。
- advantage 为正：比 group 平均好，应该增加这个输出的概率
- advantage 为负：比 group 平均差，应该降低这个输出的概率
- sigma 归一化：使得不同 group 的 advantage 量级一致

**关键：** 这个 advantage 完全不需要 Critic！它纯粹基于采样结果的相对比较。

---

## 3. GRPO vs PPO：Advantage 的来源

这是理解 GRPO 最关键的区别：

| | PPO | GRPO |
|---|---|---|
| **Advantage 来源** | Critic（Value Model）估计 V(s) | Group 内相对 reward |
| **需要 Critic？** | 是（额外一个大模型） | 否 |
| **计算方式** | A = r + γV(s') - V(s) | A = (r - μ) / σ |
| **额外模型** | Actor + Critic + Reference | Actor + Reference |
| **内存开销** | 高 | 低（省去 Critic） |

**PPO 的方式：** 用 Critic 给每个状态打分，计算"比预期好多少"。需要训练一个准确的 value function。

**GRPO 的方式：** 用同一 prompt 的多个采样结果做对比，"比 group 平均好多少"。不需要任何额外模型，只需要多采样几次。

**直觉类比：**
- PPO 的 Critic 像一个"老师"，告诉你"你的水平应该是多少分"
- GRPO 的 group relative 像"同学互评"，通过比较同一道题的不同解答来判断谁做得更好

---

## 4. GRPO 目标函数（核心公式）

### Importance Ratio（与 PPO 相同）

$$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$$

token-level 展开：

$$\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} = \prod_{t=1}^{T} \frac{\pi_\theta(o_{i,t} \mid q, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,\lt t})}$$

**直觉：** 这个 ratio 衡量新策略相对于旧策略，对输出的"偏好变化"。ratio 接近 1 表示变化不大，偏离 1 表示策略发生了显著变化。

### 完整目标函数

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min \left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)} A_i, \; \text{clip}\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]$$

拆解这个公式：

1. **外层期望**：对所有 prompt q 和对应的 group 采样
2. **Group 内平均**：对 group 内每个样本取平均
3. **Clipped surrogate**：与 PPO 相同的 clipping 机制，限制 importance ratio 的范围
4. **KL penalty**：防止策略偏离 reference model 太远

### KL 散度的计算

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{o \sim \pi_\theta} \left[ \log \frac{\pi_\theta(o \mid q)}{\pi_{\text{ref}}(o \mid q)} \right]$$

token-level 近似：

$$D_{\text{KL}} \approx \frac{1}{T} \sum_{t=1}^{T} \log \frac{\pi_\theta(o_t \mid q, o_{\lt t})}{\pi_{\text{ref}}(o_t \mid q, o_{\lt t})}$$

**作用：** 防止模型在 RL 训练中"走偏"——生成高 reward 但不像正常语言的文本。

---

## 5. GRPO 的完整 Pipeline

### Step 1：采样（On-policy）

对每个 prompt q，用当前策略生成 G 个回答：

$$o_1, o_2, ..., o_G \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$$

**为什么是 on-policy？** 因为数据必须是当前策略生成的，否则 importance ratio 的方差会很大。

### Step 2：计算 reward

对每个回答计算 reward：

$$r_i = r(q, o_i)$$

reward 来源可以是：
- **正确性 reward**：数学答案是否正确、代码是否通过测试
- **格式 reward**：是否符合要求的输出格式
- **组合 reward**：正确性 + 格式

### Step 3：计算 group advantage

$$A_i = \frac{r_i - \mu}{\sigma}, \quad \mu = \frac{1}{G}\sum_j r_j, \quad \sigma = \sqrt{\frac{1}{G}\sum_j (r_j - \mu)^2}$$

**直觉：** 如果 group 内所有回答的 reward 都差不多（sigma 接近 0），说明这个 prompt 对当前策略来说"太简单"或"太难"，学习信号很弱。如果 reward 分化明显（sigma 大），说明有明确的好坏之分，学习信号强。

### Step 4：Policy update（类似 PPO）

使用 clipped surrogate + KL penalty 更新策略：

$$\theta \leftarrow \theta + \alpha \nabla_\theta J_{\text{GRPO}}(\theta)$$

### Step 5：迭代（Iterative RL）

重复 Step 1 ~ Step 4：
- 更新参数后，用新策略重新采样
- 计算新的 reward 和 advantage
- 继续更新

---

## 6. 三种 Credit Assignment 方法

论文中讨论了如何将序列级 advantage 分配到 token 级别。这在 GRPO 中是一个重要设计选择。

### 方法 1：全局平均分配

$$A_{i,t} = A_i \quad \text{for all } t$$

同一序列中所有 token 共享同一个 advantage。

**优点：** 简单，实现容易
**缺点：** 粗糙——无法区分哪些 token 对好/坏结果贡献更大

### 方法 2：后向累积

$$A_{i,t} = \sum_{k=t}^{T} r_k$$

后面的 reward 影响前面的 token。

**直觉：** 在序列末尾的 reward 信号"向后传播"到前面的 token。

**缺点：** 在 LLM 中，每个 token 通常没有独立的 reward，所以这种方法不太适用。

### 方法 3：Iterative RL（论文最终采用）

GRPO 最终采用的方式是：**不做 token-level 的 credit assignment，直接在序列级别操作。**

具体来说：
- Advantage 是序列级的（不是 token 级的）
- 但 importance ratio 仍是 token-level 的连乘
- 每次迭代用完整的序列计算 loss

**为什么选择这种方式？**
- 在可验证任务（数学、代码）中，reward 是序列级的（最终答案对不对），强行拆到 token 级别反而引入噪声
- Iterative RL 的多次采样 + 多次更新，已经隐式地完成了 credit assignment

---

## 7. On-policy 采样：为什么实时生成数据更好

论文特别强调：**使用当前策略实时生成数据，比使用固定的离线数据集更有效。**

### On-policy vs Off-policy

| | On-policy | Off-policy |
|---|---|---|
| 数据来源 | 当前策略生成 | 旧数据集 |
| Importance ratio | 接近 1 | 可能偏离很大 |
| 方差 | 低 | 高 |
| 数据新鲜度 | 每次更新后重新采样 | 固定不变 |

### 为什么 off-policy 数据效果差？

如果用 SFT 数据或其他旧策略的数据做 RL：
- Importance ratio 可能偏离 1 很多
- 高方差的 importance ratio 导致梯度估计不准
- 模型可能学到的是"旧数据的分布"而非"最优分布"

### 直觉

> 你想训练一个学生做数学题。与其用去年的考题（可能难度不匹配），不如让他做今年的模拟题——难度、风格都更贴合当前水平。

---

## 8. 过程监督 vs 结果监督

### 结果监督（Outcome Supervision）

$$r = f(\text{final answer})$$

只看最终答案是否正确。

**优点：**
- 简单，易于实现
- 可以自动验证（如数学题对答案、代码跑测试）
- 易于 scale（不需要人工标注中间步骤）

**缺点：**
- 信号稀疏——一个 100 token 的推理过程，只有最后一个答案有 reward
- Credit assignment 困难

### 过程监督（Process Supervision）

$$r_t = f(\text{intermediate reasoning step}_t)$$

对每一步推理都评分。

**优点：**
- 信号密集，学习效率高
- 可以精确指出哪一步出错

**缺点：**
- 需要人工标注每一步（成本极高）
- 难以自动化

### GRPO 的选择

> 主要使用结果监督（verifiable reward）

原因：
- 数学问题可以自动验证答案正确性
- 代码可以自动跑测试
- 无需人工标注，可大规模使用

---

## 9. 奖励函数设计

DeepSeek 在 GRPO 中使用了精心设计的 reward 函数：

### (1) Correctness Reward（正确性）

- 数学：最终答案是否与标准答案一致
- 代码：是否通过所有测试用例

### (2) Format Reward（格式）

- 是否符合要求的推理格式
- 是否遵循输出规范

### (3) Combined Reward

$$r = r_{\text{correctness}} + r_{\text{format}}$$

**直觉：** Correctness reward 告诉模型"答案对不对"，Format reward 告诉模型"格式对不对"。两者缺一不可——格式好但答案错没用，答案对但格式乱也不理想。

---

## 10. RL 的本质作用（论文核心 insight）

### RL 不是让模型"更聪明"

这是一个非常重要的认知：

> RL 并不是让模型获得新的知识或推理能力，而是让模型的输出分布更好。

具体表现：
- **提升 pass@k**：多次采样中，至少有一次答对的概率提高了
- **提升 consistency**：模型更稳定地给出正确答案
- **改善 reasoning style**：输出更结构化、更有条理

### 分布重加权

RL 的数学本质是：

$$\pi_{\text{RL}}(y \mid x) \propto \pi_{\text{SFT}}(y \mid x) \cdot \exp\left(\frac{r(y)}{\beta}\right)$$

高 reward 的输出概率被放大，低 reward 的输出概率被缩小。模型的"能力"（知识、推理）没有变，但"表达方式"变好了。

### Emergent Behavior

随着 RL 训练的进行，模型可能出现涌现行为：
- **Self-reflection**：生成类似"让我重新检查一下"的自我纠错
- **Long chain-of-thought**：生成更长、更详细的推理链

这些行为不是被显式教的，而是 RL 优化过程中自然涌现的。

---

## 11. DeepSeek 模型训练路径

### DeepSeek-Math

1. Base Model → 2. SFT（数学数据）→ 3. GRPO（math reward）

### DeepSeek-R1-Zero

1. Base Model → 2. 直接 GRPO（无 SFT）

**惊人的发现：** 即使没有 SFT 阶段，纯 RL 训练也能让模型学会推理。模型自发地学会了 chain-of-thought。

### DeepSeek-R1

1. Base Model → 2. SFT（cold start data）→ 3. GRPO → 4. Rejection Sampling + SFT → 5. 再次 GRPO

**完整的训练 pipeline**，结合了 SFT 和 RL 的优势。

---

## 12. 总结

### GRPO 的本质

> 用 group relative advantage 替代 Critic 的 PPO。通过同一 prompt 下多个采样结果的相对比较来估计 advantage，省去了训练 Value Model 的开销。

### 关键创新

| 创新 | 说明 |
|------|------|
| 去掉 Critic | 用 group 内相对 reward 替代 value function |
| Group-based advantage | 组内均值归一化 |
| 可验证 reward | 利用数学/代码的自动验证 |
| Iterative RL | 多次采样 + 多次更新 |

### 优点

- **简单**：不需要训练 Critic，减少模型数量
- **高效**：内存和计算开销大幅降低
- **适合 LLM**：利用 LLM 采样便宜的特点，用多次采样替代 Critic 的价值估计

### 局限（为 DAPO、GSPO 铺垫）

- **Token-level ratio 的高方差**：importance ratio 仍是 token-level 连乘，长序列时方差大
- **Credit assignment 仍较粗糙**：序列级 advantage 无法区分 token 的贡献
- **Clipping 对称**：对好样本和坏样本的约束相同，可能限制 exploration

### 一句话总结

> GRPO = 用 group 内相对比较替代 Critic 的 PPO。通过"同学互评"而非"老师打分"来估计 advantage，更适合 LLM 的大规模 RL 训练。
