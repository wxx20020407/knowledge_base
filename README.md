# LLM 强化学习知识库

从 PPO 到 GSPO，系统梳理 LLM 强化学习的核心方法与演进脉络。

## 目录

### 1. [PPO: Proximal Policy Optimization](LLM-RL/PPO.md)

LLM RL 训练的基础算法。用 clipping 模拟 trust region，实现稳定高效的 policy gradient。

核心内容：Clipped Surrogate Objective、KL Penalty、Multiple Epochs、GAE、PPO 在 LLM 中的局限（token-level ratio 高方差、Critic 训练困难）

---

### 2. [SFT + RLHF: 对齐核心范式](LLM-RL/SFT_RLHF.md)

InstructGPT 的三阶段 pipeline：SFT 学格式 → RM 学偏好 → PPO 重分配输出概率。

核心内容：SFT 的作用与局限、Reward Model 的 Pairwise Loss、KL 约束的含义、RLHF 的数学本质（exponential reweighting）

---

### 3. [GRPO: Group Relative Policy Optimization](LLM-RL/GRPO.md)

DeepSeek 提出的 Actor-only 方法。详细讲解 Actor 和 Critic 的关系，以及为什么可以用 group relative advantage 替代 Critic。

核心内容：Actor-Critic 架构详解、Group Relative Advantage、On-policy 采样、过程监督 vs 结果监督、三种 Credit Assignment 方式

---

### 4. [DAPO: 系统级优化](LLM-RL/DAPO.md)

对 GRPO 的四个系统级修复：Token-level Loss、Clip-Higher、Overlong Penalty、Dynamic Sampling。

核心内容：长度不公平问题、非对称 clipping 与 entropy collapse、长度偏置的机制与修复

---

### 5. [GSPO: Sequence-level 优化](LLM-RL/GSPO.md)

将优化从 token-level 提升到 sequence-level，从根本上解决长序列高方差问题。

核心内容：Length Normalization 的方差分析（$O(T) \to O(1)$）、Sequence-level Clipping、与 GRPO/DAPO 的关系

---

## 方法演进脉络

```
PPO (基础)
 ├── RLHF (PPO 的 LLM 应用)
 │    └── 问题：Critic 太重、token-level ratio 高方差
 ├── GRPO (去掉 Critic，用 group relative advantage)
 │    └── 问题：长度不公平、entropy collapse、exploration 不足
 ├── DAPO (系统性修复 GRPO)
 │    └── 问题：token-level ratio 方差仍随长度增长
 └── GSPO (sequence-level ratio，方差 $O(T) \to O(1)$)
```

## 核心共性理解

1. **RL 优化的是输出分布，而不是模型能力** —— 所有方法的本质都是重加权输出概率
2. **Token-level → Sequence-level** —— 从 PPO 到 GSPO 的核心趋势是粒度的提升
3. **Reward 设计决定优化方向** —— 可验证 reward（数学/代码）是 RLVR 成功的关键
4. **Sampling 策略影响训练效果** —— on-policy、dynamic sampling 都在优化数据利用效率
