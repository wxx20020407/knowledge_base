# GRPO: Group Relative Policy Optimization 笔记（DeepSeekMath, arXiv:2402.03300）

---

https://arxiv.org/pdf/2402.03300

## 1. Introduction（极简）

DeepSeek 提出 GRPO（Group Relative Policy Optimization），用于替代 PPO：

核心思想：

> 不再使用 critic（value model），而是用 **group 内相对 reward** 来估计 advantage

优势：

- 无需 value function（降低复杂度）
- 更适合 LLM（采样 cheap） :contentReference[oaicite:0]{index=0}
- 可直接用于可验证任务（math/code）

---

## 2. Notation（全符号定义）

---

### 基本变量

- \( q \)：输入问题（query / prompt）
- \( o \)：模型生成的输出序列（output）
- \( o_i \)：第 \( i \) 个采样输出
- \( G \)：group size（同一 prompt 采样数量）
- \( \{o_i\}_{i=1}^G \)：一个 group 的所有输出

---

### Policy

- \( \pi_\theta(o|q) \)：当前策略（policy）
- \( \pi_{\theta_{\text{old}}}(o|q) \)：旧策略
- \( \pi_{\text{ref}} \)：reference model（用于 KL regularization）

---

### Reward

- \( r_i = r(q, o_i) \)：第 \( i \) 个输出的 reward
- reward 来源：
  - rule-based（正确性）
  - format-based（输出格式） :contentReference[oaicite:1]{index=1}

---

### Group Statistics

均值：

\[
\mu = \frac{1}{G} \sum_{j=1}^{G} r_j
\]

标准差：

\[
\sigma = \sqrt{\frac{1}{G} \sum_{j=1}^{G} (r_j - \mu)^2}
\]

---

### Advantage（核心）

\[
A_i = \frac{r_i - \mu}{\sigma}
\]

含义：

- 相对 group 的表现
- 无需 value function :contentReference[oaicite:2]{index=2}

---

## 3. GRPO Objective（核心公式）

---

### Importance Ratio

\[
\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}
\]

---

### 完整目标函数

\[
J_{\text{GRPO}}(\theta) =
\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\left(
\min \left(
\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i,\,
\text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i
\right)
- \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})
\right)
\right]
\]

---

### 符号说明

- \( \epsilon \)：clipping 参数
- \( \beta \)：KL penalty 系数
- \( D_{\text{KL}} \)：KL divergence
- \( P(Q) \)：prompt 分布

---

## 4. GRPO vs PPO（核心区别）

| 方法 | Advantage来源 | 是否需要critic | ratio粒度 |
|------|-------------|---------------|----------|
| PPO | value function | ✅ | token-level |
| GRPO | group相对reward | ❌ | token-level |

关键区别：

> GRPO 用 sampling 替代 value model :contentReference[oaicite:3]{index=3}

---

## 5. RL Pipeline（必须掌握）

---

### Step 1：采样（On-policy）

对每个 prompt：

\[
o_1, o_2, ..., o_G \sim \pi_{\theta_{\text{old}}}
\]

---

### Step 2：计算 reward

\[
r_i = r(q, o_i)
\]

来源：

- 数学正确性
- 编译执行结果
- 格式约束 :contentReference[oaicite:4]{index=4}

---

### Step 3：计算 group advantage

\[
A_i = \frac{r_i - \mu}{\sigma}
\]

---

### Step 4：policy update（类似 PPO）

使用 clipping + KL penalty：

\[
\min(ratio \cdot A, \text{clip}(\cdot))
\]

---

### Step 5：重复（iterative RL）

- 不断更新 \( \theta \)
- 不断重新采样

---

## 6. 三种 Credit Assignment 方法（重点）

论文中讨论了三种方式：

---

### 方法1：Sequence-level reward 平均到 token

\[
A_t = A
\]

特点：

- 所有 token 同一个 reward
- 简单但 coarse

---

### 方法2：累积 reward（后向传播）

\[
A_t = \sum_{k=t}^{T} r_k
\]

含义：

- 后面的 token reward 影响前面

---

### 方法3：Iterative RL（论文最终采用）

- 每次只优化完整序列
- 不做 token-level credit assignment

特点：

> 更稳定，更适合 LLM

---

## 7. 实时生成数据（On-policy）更好

论文强调：

> 使用当前 policy 生成数据比固定 dataset 更有效

原因：

- 避免 distribution mismatch
- 保证 exploration

即：

\[
o_i \sim \pi_{\theta_{\text{current}}}
\]

而不是：

- SFT dataset

---

## 8. 过程监督 vs 结果监督（关键）

---

### 结果监督（Outcome Supervision）

\[
r = f(\text{final answer})
\]

特点：

- 只看最终答案
- 简单
- 易 scale

---

### 过程监督（Process Supervision）

\[
r_t = f(\text{intermediate reasoning})
\]

特点：

- 对每一步推理评分
- 更精细
- 成本高

---

### GRPO 使用：

> 主要是结果监督（verifiable reward）

原因：

- 数学问题可自动验证
- 无需人工标注

---

## 9. 奖励函数设计（重要）

DeepSeek 使用：

---

### (1) Rule-based reward

- 数学答案是否正确
- 代码是否通过测试

---

### (2) Format reward

- 是否符合 `<think>...</think>` 结构 :contentReference[oaicite:5]{index=5}

---

### (3) Combined reward

\[
r = r_{\text{correctness}} + r_{\text{format}}
\]

---

## 10. 持续更新奖励（implicit）

特点：

- reward function 本身固定
- 但：

\[
\text{reward distribution evolves}
\]

原因：

- policy 改变 → 采样数据变化

---

## 11. 实验结论（非常重要）

---

### 11.1 RL 的本质作用

> RL 并不是让模型“更聪明”，而是让输出分布更好

具体表现：

- 提升 pass@k
- 提升 consistency
- 提升 reasoning style

---

### 11.2 Emergent behavior

随着 RL：

- self-reflection
- long chain-of-thought :contentReference[oaicite:6]{index=6}

---

### 11.3 RL 提升方式

不是：

- 提升 base capability

而是：

\[
\text{reweight output distribution}
\]

---

## 12. DeepSeek 模型训练路径（必须搞清）

---

### 1. Base Model

- 预训练模型（DeepSeek-V3 base）

---

### 2. SFT（Supervised Fine-Tuning）

- instruction / math 数据

---

### 3. RL（GRPO）

---

### DeepSeek-Math

- SFT + GRPO（math reward）

---

### DeepSeek-R1-Zero

- 直接 RL（无 SFT）
- 只用 reward

---

### DeepSeek-R1

- SFT + RL + alignment

---

## 13. 总结（核心理解）

---

### GRPO 本质

> 用 group relative reward 替代 value function 的 PPO

---

### 关键创新

- 无 critic
- group-based advantage
- RL 可扩展

---

### 核心优点

- 简单
- 高效
- 适合 LLM

---

### 核心缺点（为 GSPO 铺垫）

- token-level ratio → 高方差
- credit assignment 粗糙

---

## 14. 一句话总结

> GRPO = “用采样替代 critic 的 PPO”，通过 group 内比较来学习更优输出分布