# RLHF（InstructGPT）论文笔记（arXiv:2203.02155）

---

## 1. Introduction（极简）

问题：

> 语言模型训练目标（next-token prediction） ≠ 用户期望（helpful, honest, harmless）

即：

\[
\max \log P(x) \;\neq\; \max \text{human preference}
\]

解决方案：

> Reinforcement Learning from Human Feedback（RLHF）

核心 pipeline：

1. SFT（监督微调）
2. RM（奖励模型）
3. RL（PPO优化）

---

## 2. Notation（符号定义）

---

### 基本变量

- \( x \)：prompt（输入指令）
- \( y \)：模型输出序列
- \( y_t \)：第 \( t \) 个 token
- \( T \)：序列长度

---

### Policy

- \( \pi_\theta(y|x) \)：当前模型（policy）
- \( \pi_{\text{SFT}} \)：SFT模型（初始化policy）

---

### Reward Model

- \( r_\phi(x, y) \)：奖励模型（预测人类偏好）
- \( \phi \)：RM参数

---

### 数据

- \( D_{\text{SFT}} \)：人工示范数据
- \( D_{\text{pref}} \)：人类偏好排序数据

---

## 3. 总体 Pipeline（核心）

---

### Step 1：SFT（Supervised Fine-Tuning）

\[
\max_\theta \mathbb{E}_{(x,y) \sim D_{\text{SFT}}}
\log \pi_\theta(y|x)
\]

---

### Step 2：Reward Model（RM）

\[
r_\phi(x,y)
\approx \text{human preference}
\]

---

### Step 3：RL（PPO）

\[
\max_\theta \mathbb{E}_{y \sim \pi_\theta}
\left[ r_\phi(x,y) - \beta D_{KL}(\pi_\theta || \pi_{\text{SFT}}) \right]
\]

---

## 4. 核心方法1：SFT（监督微调）

---

### 4.1 数据

人类标注：

\[
(x, y^*)
\]

其中：

- \( y^* \)：高质量人工回答

---

### 4.2 目标函数

\[
L_{\text{SFT}} =
- \mathbb{E}_{(x,y^*)}
\log \pi_\theta(y^* | x)
\]

---

### 4.3 作用机制

本质：

> 学习“什么是一个好回答的模式”

---

### 4.4 梯度

\[
\nabla_\theta L =
- \nabla_\theta \log \pi_\theta(y^* | x)
\]

---

### 4.5 解决问题

| 问题 | SFT作用 |
|------|--------|
| 模型不会follow instruction | 学习格式 |
| 输出结构混乱 | 学习模板 |
| 无对齐能力 | 初步对齐 |

---

### ⚠️ 局限

- 只能模仿数据
- 无法超越示范质量

---

## 5. 核心方法2：Reward Model（RM）

---

## 5.1 数据构建

对于同一 prompt：

\[
(x, y_1, y_2, ..., y_K)
\]

人类排序：

\[
y_w \succ y_l
\]

---

## 5.2 Pairwise Loss（核心公式）

\[
L_{\text{RM}}(\phi) =
- \mathbb{E}_{(x,y_w,y_l)}
\log \sigma \left(
r_\phi(x,y_w) - r_\phi(x,y_l)
\right)
\]

---

### 符号说明

- \( y_w \)：preferred（更好）
- \( y_l \)：less preferred
- \( \sigma \)：sigmoid函数

---

## 5.3 机制解释（非常关键）

优化目标：

\[
r_\phi(x,y_w) > r_\phi(x,y_l)
\]

即：

> 学习一个排序函数（ranking function）

---

## 5.4 梯度解释

\[
\nabla_\phi L \propto
\sigma(-\Delta r) \cdot \nabla_\phi \Delta r
\]

其中：

\[
\Delta r = r_\phi(x,y_w) - r_\phi(x,y_l)
\]

---

## 5.5 本质

> RM = 学习 human preference 的 proxy

---

## 5.6 解决问题

| 问题 | RM解决 |
|------|--------|
| reward 无法定义 | 用人类偏好替代 |
| 多目标（helpful/harmless）难编码 | 学习隐式目标 |

---

## 6. 核心方法3：RLHF（PPO优化）

---

## 6.1 RL目标函数

\[
L_{\text{RL}} =
\mathbb{E}_{y \sim \pi_\theta}
\left[
r_\phi(x,y)
- \beta D_{KL}(\pi_\theta || \pi_{\text{SFT}})
\right]
\]

---

### 符号说明

- \( \beta \)：KL penalty 系数
- \( D_{KL} \)：KL divergence

---

## 6.2 PPO形式

使用 importance ratio：

\[
r_t(\theta) =
\frac{\pi_\theta(y_t | x, y_{<t})}
{\pi_{\text{old}}(y_t | x, y_{<t})}
\]

---

目标：

\[
L =
\mathbb{E}
\left[
\min(
r_t A_t,
\text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t
)
\right]
\]

---

## 6.3 KL约束（关键）

\[
- \beta D_{KL}(\pi_\theta || \pi_{\text{SFT}})
\]

作用：

> 防止模型偏离语言分布

---

## 6.4 PPO-ptx（论文关键改进）

加入预训练loss：

\[
L = L_{\text{RL}} + \gamma L_{\text{LM}}
\]

---

### 符号

- \( \gamma \)：pretraining mixing weight

---

## 6.5 机制解释（重点）

RLHF优化：

\[
\pi_\theta(y|x) \uparrow \text{ if } r_\phi(x,y) \text{ 高}
\]

但同时：

\[
\pi_\theta \approx \pi_{\text{SFT}}
\]

---

## 6.6 本质（必须记住）

> RLHF = 在语言分布附近，重新加权输出概率

---

## 7. 三个模块的分工（核心理解）

---

### SFT

\[
\text{learn } P(y|x)
\]

👉 学“怎么说话”

---

### RM

\[
\text{learn } r(x,y)
\]

👉 学“什么是好话”

---

### RLHF

\[
\text{optimize } \pi(y|x) \text{ w.r.t. } r
\]

👉 学“多说好话”

---

## 8. 数据来源（关键）

---

### SFT数据

- 人工写答案
- 高质量 demonstration

---

### RM数据

- 多个回答排序
- K ≈ 4–9 ([turn0search0])

---

### RL数据

- 当前policy生成（on-policy）

---

## 9. 实验结论（核心 insight）

---

### 9.1 人类偏好显著提升

- 小模型 > 大模型（175B GPT-3） ([turn0search0] :contentReference[oaicite:0]{index=0})

---

### 9.2 行为改善

- 更少 hallucination
- 更符合 instruction
- 更低 toxicity ([turn0search0] :contentReference[oaicite:1]{index=1})

---

### 9.3 核心结论（非常重要）

> RLHF 提升的是“对齐质量”，而不是“基础能力”

---

## 10. RLHF 的本质（必须掌握）

---

### 10.1 数学本质

\[
\pi_{\text{RLHF}}(y|x)
\propto
\pi_{\text{SFT}}(y|x)
\cdot
\exp(r_\phi(x,y))
\]

---

### 10.2 含义

> RLHF = 对输出分布做 exponential reweighting

---

### 10.3 结果

- 提升高reward输出概率
- 抑制低质量输出

---

## 11. 改善了什么（必须写清）

---

### 原问题

| 问题 | 原因 |
|------|------|
| 不听指令 | 训练目标不匹配 |
| 幻觉 | 无truth约束 |
| 不安全 | 无行为约束 |

---

### RLHF 改善

| 模块 | 解决 |
|------|------|
| SFT | instruction |
| RM | preference |
| RL | distribution |

---

## 12. 局限（论文明确指出）

---

### 12.1 reward model 偏差

- 学到的是“标注员偏好”

---

### 12.2 reward hacking

- 模型可能 exploit RM

---

### 12.3 alignment ≠ truth

---

## 13. 一句话总结

> RLHF = 用人类偏好训练一个 reward model，再用 RL 在语言分布附近重新加权输出概率，实现模型行为对齐