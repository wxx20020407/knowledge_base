- LLM-RL
  - PPO（基础优化方法）
    - Clipped Objective（限制policy更新幅度）
    - KL Penalty（近似trust region）
    - Multi-epoch Update（提升sample efficiency）
    - GAE（降低advantage方差）

  - RLHF（对齐核心范式）
    - SFT（学习instruction与输出格式）
    - Reward Model（pairwise preference建模）
    - PPO优化（带KL约束）
    - 本质：distribution reweighting（不是提升能力）

  - GRPO（Group-based RL）
    - Group Relative Advantage（去掉value function）
    - On-policy group sampling
    - 三种credit assignment方式
      - 全局平均
      - 后向累积
      - iterative RL（最终采用）
    - 实时生成数据优于离线数据
    - 结果监督为主（vs 过程监督）

  - DAPO（GRPO系统优化）
    - Token-level Loss（解决长序列梯度稀释）
    - Clip-Higher（非对称clipping增强探索）
    - Overlong Filtering / Penalty（控制长度偏置）
    - Dynamic Sampling（按难度分配采样）

  - GSPO（Sequence-level优化）
    - Sequence-level Importance Ratio
    - Length Normalization（降低variance）
    - Sequence-level Clipping
    - 本质：解决token-level高方差问题

  - 核心共性理解（跨论文总结）
    - RL优化的是输出分布，而不是模型能力
    - Token-level → Sequence-level（方差与稳定性核心问题）
    - Reward设计决定优化方向
    - Sampling策略影响训练效果