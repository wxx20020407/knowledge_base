# RLHF（InstructGPT）论文笔记（arXiv:2203.02155）

---

## 1. Introduction

### 核心问题

语言模型的训练目标是 next-token prediction：

$$\max \log P(x) = \max \sum_t \log P(x_t \mid x_{\lt t})$$

但用户期望的是 **helpful（有帮助）、honest（诚实）、harmless（无害）** 的回答。这两个目标之间存在根本性的 mismatch：一个在预测分布上很好的模型，可能生成不听指令、编造事实、或不安全的内容。

**直觉：** 想象一个背了整本百科全书的学生，你问他问题，他可能会流畅地背诵一段相关文字，但不一定在"回答你的问题"。SFT + RLHF 的目标就是教他"怎么当一个好的助手"。

### 解决方案：RLHF

InstructGPT 提出了三阶段 pipeline：

1. **SFT**（Supervised Fine-Tuning）：学习"怎么说话"
2. **RM**（Reward Model）：学习"什么是好话"
3. **RL**（PPO 优化）：学习"多说好话"

---

## 2. Notation（符号定义）

### 基本变量

- $x$：prompt（用户输入的指令）
- $y$：模型输出的完整序列
- $y_t$：序列中第 $t$ 个 token
- $y_{\lt t} = (y_1, ..., y_{t-1})$：第 $t$ 步之前的历史 token
- $T$：序列长度

### Policy

- $\pi_\theta(y\mid x)$：当前模型（policy），参数为 $\theta$，给定 prompt $x$ 生成序列 $y$ 的概率
- $\pi_{\text{SFT}}$：SFT 模型，作为 RL 阶段的初始化策略和 KL 参考

token-level 概率分解：

$$\pi_\theta(y\mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{\lt t})$$

### Reward Model

- $r_\phi(x, y)$：奖励模型，输入 prompt 和回答，输出一个标量 reward
- $\phi$：RM 的参数

### 数据

- $D_{\text{SFT}}$：人工示范数据（prompt + 高质量回答）
- $D_{\text{pref}}$：人类偏好排序数据（同一 prompt 的多个回答的排序）

---

## 3. 总体 Pipeline

### Step 1：SFT（监督微调）

$$\max_\theta \; \mathbb{E}_{(x,y) \sim D_{\text{SFT}}} \left[ \log \pi_\theta(y\mid x) \right]$$

从预训练模型出发，用人工标注的高质量数据做监督微调。

### Step 2：Reward Model（奖励模型训练）

$$\min_\phi \; L_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D_{\text{pref}}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

用人类偏好数据训练一个奖励模型，让它学会区分"好回答"和"差回答"。

### Step 3：RL（PPO 优化）

$$\max_\theta \; \mathbb{E}_{x \sim D, \, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta D_{KL}(\pi_\theta \| \pi_{\text{SFT}}) \right]$$

用 PPO 优化策略，使其生成高 reward 的回答，同时通过 KL 约束防止偏离语言分布。

---

## 4. 核心方法 1：SFT（监督微调）

### 数据

人类标注者为每个 prompt 写出高质量回答：

$$(x, y^*) \in D_{\text{SFT}}$$

其中 $y^*$ 是人工示范的理想回答。

### 目标函数

$$L_{\text{SFT}} = -\mathbb{E}_{(x, y^*)} \left[ \log \pi_\theta(y^* \mid x) \right]$$

这就是标准的语言模型损失——让模型学习生成与人工示范一致的输出。

### 梯度

$$\nabla_\theta L_{\text{SFT}} = -\nabla_\theta \log \pi_\theta(y^* \mid x)$$

直觉：梯度方向是"让 $y^*$ 中每个 token 的概率更高"。

### SFT 的作用

| 问题 | SFT 的作用 |
|------|-----------|
| 预训练模型不会 follow instruction | 学习指令跟随的格式和模式 |
| 输出结构混乱（如纯续写） | 学习对话/问答的模板 |
| 无对齐能力 | 初步的行为对齐 |

### SFT 的局限

- **只能模仿数据**：模型学到的是"标注者会怎么写"，而不是"什么是最好的回答"
- **无法超越示范质量**：如果标注者的回答有瑕疵，模型也会学到这些瑕疵
- **覆盖有限**：标注数据量有限，难以覆盖所有场景

**关键洞察：** SFT 教会了模型"怎么说话"的格式，但没有教会它"什么是好话"。后者需要 reward model。

---

## 5. 核心方法 2：Reward Model（RM）

### 为什么需要 RM？

直接定义"什么是好回答"极其困难——helpful、honest、harmless 这些概念很难用规则编码。InstructGPT 的洞察是：**与其定义规则，不如学习排序。**

### 数据构建

对于同一 prompt，让模型生成多个回答，然后人类标注者进行排序：

$$(x, y_1, y_2, ..., y_K) \quad \text{其中} \quad y_w \succ y_l$$

$y_w$ 是被偏好的（preferred）回答，$y_l$ 是不被偏好的（less preferred）回答。通常 $K \approx 4 \sim 9$。

### Pairwise Loss（核心公式）

$$L_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D_{\text{pref}}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

其中 $\sigma$ 是 sigmoid 函数：$\sigma(z) = \frac{1}{1+e^{-z}}$。

### 为什么用 sigmoid + 差值？

这个 loss 的设计非常精巧：

1. **优化目标**：让 $r_\phi(x, y_w) - r_\phi(x, y_l) > 0$，即好回答的 reward 高于差回答
2. **Sigmoid 的作用**：当差值很大时，梯度趋于 0（已经学好了，不需要再更新）；当差值很小或为负时，梯度很大（还需要继续学习）
3. **Pairwise 而非 pointwise**：不需要绝对 reward 值准确，只需要相对排序正确——这比精确预测 reward 容易得多

### 梯度分析

$$\nabla_\phi L \propto \sigma(-\Delta r) \cdot \nabla_\phi \Delta r$$

其中 $\Delta r = r_\phi(x, y_w) - r_\phi(x, y_l)$。

- 当 $\Delta r$ 很大（正）时：$\sigma(-\Delta r) \approx 0$，梯度很小 → 模型已经能区分，停止学习
- 当 $\Delta r \approx 0$ 时：$\sigma(-\Delta r) \approx 0.5$，梯度最大 → 模型还分不清，需要重点学习
- 当 $\Delta r$ 为负时：$\sigma(-\Delta r) \approx 1$，梯度很大 → 模型完全搞反了，需要大幅修正

### RM 的本质

> RM = 学习 human preference 的 proxy（代理）。它不直接理解"什么是好"，而是学会了"人类会偏好哪个"。

### RM 的局限

| 问题 | 说明 |
|------|------|
| 标注员偏好 ≠ 真实价值 | RM 学到的是特定标注群体的偏好 |
| Reward hacking | 模型可能找到 RM 的漏洞，生成"高 reward 但实际很差"的回答 |
| 多目标冲突 | helpful 和 harmless 可能矛盾，RM 难以平衡 |

---

## 6. 核心方法 3：RLHF（PPO 优化阶段）

### RL 目标函数

$$L_{\text{RL}} = \mathbb{E}_{x \sim D, \, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta D_{KL}(\pi_\theta \| \pi_{\text{SFT}}) \right]$$

这个目标包含两项：

**第一项：$r_\phi(x, y)$**

让模型生成高 reward 的回答。直觉：RM 说"这个回答好"，RL 就让模型多生成类似的回答。

**第二项：$-\beta D_{KL}(\pi_\theta \| \pi_{\text{SFT}})$**

KL 散度惩罚，防止模型偏离 SFT 模型太远。直觉：不能为了追求高 reward 而生成怪异的、不像正常语言的文本。

$\beta$ 控制两者的平衡：
- $\beta$ 太大：模型过于保守，几乎不变
- $\beta$ 太小：模型可能 reward hack，生成高 reward 但质量差的内容

### PPO 在 RLHF 中的具体形式

Token-level importance ratio：

$$r_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{\lt t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{\lt t})}$$

Clipped objective：

$$L = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]$$

### PPO-ptx：加入预训练 loss

InstructGPT 的一个重要改进——在 RL loss 中混入预训练 loss：

$$L = L_{\text{RL}} + \gamma L_{\text{LM}}$$

其中 $L_{\text{LM}} = -\mathbb{E}_{x \sim D_{\text{pretrain}}}[\log \pi_\theta(x)]$。

**为什么？** 纯 RLHF 可能导致模型在 RLHF 数据上过拟合，丧失语言能力（"对齐税"）。混入预训练 loss 可以保持模型的语言建模能力。

### RLHF 的机制

RLHF 优化的本质是：

$$\pi_\theta(y\mid x) \uparrow \quad \text{if} \quad r_\phi(x, y) \text{ is high}$$

但同时：

$$\pi_\theta \approx \pi_{\text{SFT}} \quad \text{(KL constraint)}$$

**直觉：** 在 SFT 模型"附近"，重新分配输出概率——让好回答的概率更高，差回答的概率更低，但不改变整体的语言分布。

---

## 7. 三个模块的分工

| 模块 | 目标 | 学什么 |
|------|------|--------|
| SFT | max log π_θ(y\* | x) | 怎么说话（格式、模式） |
| RM | max log σ(rw - rl) | 什么是好话（偏好排序） |
| RLHF | max rφ - β·D\_KL | 多说好话（重分配概率） |

三者的关系是递进的：SFT 是基础，RM 是评价标准，RLHF 是优化过程。

---

## 8. 数据来源

### SFT 数据

- 人工写答案：标注者直接为 prompt 写出高质量回答
- 量级：通常几千到几万条
- 特点：高质量、多样性、覆盖常见场景

### RM 数据

- 多个回答的排序：同一 prompt，模型生成 K 个回答，人类标注者排序
- 量级：通常几万到几十万条比较对
- 特点：比写回答容易，标注效率更高

### RL 数据

- 当前 policy 生成（on-policy）：用当前策略生成回答，RM 打分
- 特点：数据分布随策略更新而变化

---

## 9. 实验结论

### 核心发现

**1. 小模型 + RLHF > 大模型（无对齐）**

InstructGPT 发现：1.3B 参数的模型经过 RLHF 后，在人类评估中优于 175B 的 GPT-3。这说明**对齐比规模更重要**——至少在"听指令"这件事上。

**2. 行为改善**

- 更少 hallucination（减少编造）
- 更符合 instruction（指令跟随更好）
- 更低 toxicity（减少有害输出）

**3. 最重要的结论**

> RLHF 提升的是"对齐质量"（alignment），而不是"基础能力"（capability）。

模型的知识和推理能力来自预训练，RLHF 只是让它更好地"表达"这些能力——以人类期望的方式。

---

## 10. RLHF 的数学本质

### 输出分布的变化

RLHF 后的策略可以近似表示为：

$$\pi_{\text{RLHF}}(y\mid x) \propto \pi_{\text{SFT}}(y\mid x) \cdot \exp\left(\frac{1}{\beta} r_\phi(x, y)\right)$$

这个结果可以从 KL 约束的优化问题推导出来（对 $\pi$ 求解拉格朗日对偶）。

### 含义

> RLHF = 对输出分布做 exponential reweighting（指数重加权）

- 高 reward 的输出：概率被放大 $\exp(r/\beta)$ 倍
- 低 reward 的输出：概率被缩小
- $\beta$ 控制重加权的幅度

### 为什么是指数？

因为 KL 约束的拉格朗日对偶解就是指数形式。这不是人为选择，而是优化问题的自然结果。

### 结果

- 提升高 reward 输出的概率
- 抑制低质量输出
- 但不会完全消除低概率输出（因为 KL 约束）

---

## 11. RLHF 改善了什么

### 原始问题

| 问题 | 原因 |
|------|------|
| 不听指令 | 预训练目标是续写，不是回答问题 |
| 幻觉 | 没有 truth 约束，只追求流畅 |
| 不安全 | 没有行为约束 |

### RLHF 的改善

| 模块 | 解决什么 |
|------|---------|
| SFT | 学习指令跟随的格式 |
| RM | 编码人类偏好（helpful/honest/harmless） |
| RL | 将偏好转化为输出分布的调整 |

---

## 12. 局限

### 12.1 RM 偏差

RM 学到的是特定标注群体的偏好，不是"客观真理"。不同标注者的偏好可能不同，甚至矛盾。

### 12.2 Reward Hacking

模型可能找到 RM 的漏洞——生成"RM 给高分但人类觉得很差"的回答。例如，模型可能学会使用某些"看起来专业"但实际空洞的措辞来获取高 reward。

### 12.3 Alignment ≠ Truth

RLHF 优化的是"人类觉得好"，不是"客观正确"。如果标注者的判断有误，模型也会学到错误的行为。

### 12.4 训练复杂度

RLHF 阶段需要同时维护 4 个模型：policy model、reference model（KL 约束）、reward model、value model（PPO 的 critic）。内存和计算开销很大。

---

## 13. 一句话总结

> RLHF = SFT 学格式 → RM 学偏好 → PPO 在 KL 约束下重分配输出概率。本质是对语言模型的输出分布做指数重加权，使高偏好的输出概率更高。102041020410204
102041020410204