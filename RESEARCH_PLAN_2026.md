# 捕获-再捕获范式下的准备金评估：流量三角的替代路径

## Capture-Recapture Reserving: An Alternative Paradigm to the Chain-Ladder Triangle

---

## 一、核心命题（The Big Thesis）

**流量三角（Chain-Ladder）统治精算准备金评估70年，但它存在结构性缺陷：**
- 强依赖"事故年×发展年"的矩形结构，丢失个体级信息
- 假设发展模式跨事故年稳定（reality: 它不稳定）
- 无法自然处理开放系统（保单进出、险种变更）
- IBNR估计本质上是外推，缺乏统计上的总体推断逻辑

**本文提出：保险索赔报告过程本质上是一个被动式捕获-再捕获过程。**

| 保险准备金 | 捕获-再捕获 |
|-----------|-----------|
| 全部索赔（含IBNR） | 总体规模 N |
| 索赔报案 | 个体被"捕获" |
| 同一索赔的后续报告/赔付 | 个体被"再捕获" |
| IBNR未决赔款 | 未观测到的个体 |
| 事故年队列 | 种群队列 |
| 发展模式 | 捕获概率 q_i |
| 赔案结案 | 个体退出系统 |
| 新赔案产生 | 新个体进入 |
| 损失发展因子 | 再捕获率 |

**这不是对Chain-Ladder的修补，而是一条完全不同的推断路径。**

---

## 二、理论框架设计（5层递进）

### Layer 1: 基础映射与封闭系统估计

**设定**：考虑一个事故年队列 $a$，包含 $N_a$ 个索赔（含IBNR），其中 $N_a$ 未知。

在发展期 $d = 1, 2, \ldots, D$ 中，每个索赔 $j$ 在期 $d$ 被"捕获"（报案/赔付/调整）的概率为 $p_{jd}$。

定义捕获历史矩阵 $\mathbf{X} = (x_{jd})$，$x_{jd} = 1$ 表示索赔 $j$ 在发展期 $d$ 被观测到。

**封闭系统假设**：$N_a$ 固定，无新索赔进入、无索赔提前退出。

利用你原始论文的估计量：
$$\hat{N}_a = \frac{\sum_{d=1}^{D} Y_d}{\sum_{d=1}^{D} x_d}$$

其中 $Y_d$ 为第 $d$ 期的累计标记数，$x_d$ 为第 $d$ 期的再捕获数。

**定理 1'（封闭保险系统）**：在有放回抽样框架下，$\hat{N}_a$ 是 $N_a$ 的渐近无偏估计。

### Layer 2: 开放系统 → 跨事故年动态

现实中保险系统是**开放**的：
- 新保单生效 → 新个体进入（$k_d$）
- 保单到期/退保 → 个体退出（$s_d$）
- 赔案结案 → 永久退出

动态方程（你原始论文的核心）：
$$Y_d = Y_{d-1} + 1 - x_d - s_d + k_d$$

**这里需要的理论升级**：
1. 引入**分层捕获概率**：按险种/地区/索赔类型分组，$p_{jd} = p_{g(j),d}$，解决原审稿人"等概率"质疑
2. 引入**时变超参数**：$\delta_d = \ln(\tau_d / \tau_{d-1})$ 的过程建模，用AR(1)或Ornstein-Uhlenbeck过程

### Layer 3: 贝叶斯非参数扩展（核心理论创新）

**传统CR的致命弱点**：假设捕获概率同质或已知分层结构。

**引入 Dirichlet Process Mixture Model (DPMM)**：
$$p_j \sim \text{DP}(\alpha, G_0)$$

其中 $G_0$ 为基分布（如 Beta），$\alpha$ 为集中参数。

这使得：
- 捕获概率自动聚类，无需预先指定分层
- 处理"某些索赔类型报案快、某些慢"的异质性
- 天然回应了审稿人关于"等概率假设"的质疑

**DPMM下的总体估计量**：
$$\hat{N} = n + E\left[\sum_{j=n+1}^{N} \prod_{d=1}^{D}(1 - p_{jd}) \mid \text{data}\right]$$

通过MCMC后验推断得到 $N$ 的后验分布 $\pi(N \mid \mathbf{X})$。

### Layer 4: 从总体规模到准备金估计（连接精算实务）

估计出 $\hat{N}_a$ 后，未决赔款准备金：
$$R_a = \hat{N}_a \cdot \hat{\mu}_a - C_a^{\text{paid}}$$

其中：
- $\hat{\mu}_a$：事故年 $a$ 的平均最终索赔额（可用已结案数据估计）
- $C_a^{\text{paid}}$：已支付赔款

**关键创新**：不需要流量三角！直接从个体级捕获历史推断 $N$，再结合索赔严重度分布得到准备金。

### Layer 5: 与Chain-Ladder的理论对比

**定理（效率比较）**：在以下条件下，CR估计量的MSE严格小于Chain-Ladder估计量：
1. 发展模式跨事故年不稳定（$\text{Var}(\text{LDF}_d)$ 大）
2. 存在显著的报案异质性
3. 观测窗口较短（早期发展年）

**直觉**：Chain-Ladder丢弃了个体级信息（只看加总），而CR保留了每个索赔的完整捕获历史。当异质性大时，加总导致信息损失。

---

## 二（补）、近期方法论进展的融入策略（2026-04-30更新）

基于2020-2025年间CR领域的7项重要方法论进展，制定以下三层融入策略：

### 🔴 核心融入（Core Integration — 写入理论框架）

#### 1. HMM 隐马尔可夫模型 → Layer 2 升级
- **来源**：Labuzzetta et al. (2024)，HMM在CR中的应用
- **融入方式**：将索赔状态建模为隐变量链：未报案(U) → 已报案(R) → 调整中(A) → 已结案(S) → 重开(Re) → 终结(C)。观测到的"捕获事件"是状态转移的排放信号
- **替代内容**：替代Layer 2中简单的进出方程 $Y_i = Y_{i-1} + 1 - x_i - s_i + k_i$，用状态转移矩阵精细表达
- **理论价值**：转移矩阵自然给出每个发展期的报案率、结案率、重开率——精算师最关心的参数

#### 2. 异质检测概率建模 → Layer 3 核心
- **来源**：Moqanaki et al. (2020)，异质检测概率的个体级建模
- **融入方式**：强化DPMM框架，将捕获概率分解为 $\text{logit}(p_{jd}) = \beta_0 + \beta_1 x_{j1} + \cdots + \epsilon_j$，协变量为险种、地区、保额等
- **理论价值**：终极回应原审稿人"等概率假设不满足"的批评。不仅允许异质性，还能估计异质性的来源

#### 3. Monte Carlo GoF 检验 → Layer 5 + 模型诊断
- **来源**：Choo et al. (2024)，Monte Carlo拟合优度检验
- **融入方式**：为CR准备金模型提供专用GoF检验（Chain-Ladder基本没有严格的GoF检验）
- **理论价值**：回答"CR模型是否适合这份数据"——审稿人必问的问题

### 🟡 战略融入（Strategic — 显著增强竞争力）

#### 4. 最小卡方估计量 → Layer 1 替代/补充
- **来源**：Zheng et al. (2023)
- **融入方式**：作为原始 $\hat{\tau} = \sum Y_i / \sum x_i$ 的分布无关替代，在模型设定错误时更稳健
- **论文策略**：模拟实验中同时报告两种估计量表现

#### 5. 开放种群贝叶斯框架 → Layer 2 + Layer 3 基础
- **来源**：Menger et al. (2023)
- **融入方式**：直接处理个体进入/退出的贝叶斯推断，采用数据增强策略（augmenting with "all-zero" capture histories representing unobserved individuals）
- **理论价值**：提供 $N$ 的完整后验分布 $\pi(N|\mathbf{X})$

### 🟢 引用/扩展（Reference — Discussion章节讨论）

#### 6. 空间CR SCR/SECR → Future Work
- **来源**：Efford et al. (2023/2025)
- **映射**：保险"空间" = 产品类型 × 地理区域 × 风险分段
- **留给第二篇论文**：空间结构引入过多复杂性，与异质检测有功能重叠

#### 7. ML + ASCR → 补充材料/Future Work
- **来源**：Wang et al. (2024)
- **映射**：大规模数据下的自动化CR估计，ML自动分类索赔类型
- **留给独立论文**：ML+CR是论文级别工作，放入本文分散核心叙事

### 融入后的升级理论框架

```
Layer 1: 基础估计量（原始τ̂ + 最小卡方替代）  ← Zheng 2023
Layer 2: 开放系统（HMM状态转移 替代 简单进出方程）  ← Labuzzetta 2024 + Menger 2023
Layer 3: 贝叶斯非参数（DPMM + 异质检测概率 + 开放种群贝叶斯）  ← Moqanaki 2020 + Menger 2023
Layer 4: 准备金连接（不变）
Layer 5: 效率对比 + Monte Carlo GoF诊断  ← Choo 2024
Discussion: 空间CR扩展 + ML自动化  ← Efford 2023, Wang 2024
```

---

## 三、数据源方案

> ⚠️ **重要修正（2026-04-30）**：原方案A推荐CAS Loss Reserve Database作为首选数据源，经深入分析后确认**不适合**作为CR方法的输入数据——CAS数据是聚合流量三角（事故年×发展年的累计赔付矩阵），而CR方法需要个体级索赔交易记录。CAS数据恰恰是Chain-Ladder的原生格式，是我们要替代的对象。

### 方案A（首选·模拟实验）：SynthETIC 个体级索赔模拟

- **来源**：`SynthETIC` R包（Avanzi et al., 2020）
- **内容**：生成完整的个体级索赔生命周期——报案时间、每次赔付时间、赔付金额、结案时间、重开事件
- **为什么完美适合CR**：每个索赔的"事件序列"直接就是捕获历史 $x_{jd}$。可以精确控制异质捕获概率、开放系统动态、已知的真实 $N$（ground truth）
- **获取**：`install.packages("SynthETIC")`
- **用途**：7个模拟场景的Monte Carlo实验（S1-S7），比较CR vs CL的Bias/RMSE/Coverage
- **文献**：Avanzi, B., Taylor, G., & Wong, B. (2020). *SynthETIC: An Individual Insurance Claim Simulator with Feature Control*. IME.

### 方案B（主要实证）：French Motor TPL 数据

- **来源**：`CASdatasets` R包中的 `freMTPL2freq` + `freMTPL2sev`
- **内容**：法国车险第三者责任险，约67万张保单，个体级索赔频率和严重度
- **字段**：保单ID、索赔次数、索赔金额、车辆特征（功率、品牌、油气类型）、驾驶人特征（年龄、地区、密度）
- **CR适用性**：同一保单的多次索赔 = 多次捕获；保单级别的协变量可用于异质检测概率建模
- **获取**：`install.packages("CASdatasets", repos="http://cas.uqam.ca/pub/")`
- **学术认可度**：极高，已被数百篇精算论文使用

### 方案C（强力实证·直接对比竞品）：Canadian Auto Insurance 数据

- **来源**：Calcetero-Vanegas et al. (2025) "A Population Sampling Framework for Claim Reserving" 中使用的数据集
- **价值**：与最直接的"竞争者/验证者"使用相同数据，做head-to-head对比——他们的population sampling framework vs 我们的CR framework
- **获取**：需联系作者或查找是否公开附带数据
- **战略意义**：审稿人看到同一数据上的直接对比，说服力最强

### 方案D（跨领域验证）：CMS Medicare Claims

- **来源**：CMS.gov，公开的Medicare claims数据
- **内容**：医疗保险理赔，患者多次就诊 = 多次被动捕获
- **CR适用性**：天然的被动式CR场景——患者不会因为"被观测"而改变就医行为
- **获取**：公开下载（去标识化样本数据）
- **用途**：跨领域验证——展示CR范式不仅适用于财产险，还适用于健康险

### 方案E（仅作基准对比）：CAS Loss Reserve Database

- **来源**：Casualty Actuarial Society，NAIC Schedule P
- **格式**：10×10聚合发展三角
- **⚠️ 仅用途**：计算Chain-Ladder准备金估计值，作为CR方法的对比基准
- **不能用于**：CR模型的输入数据（数据粒度不匹配）
- **获取**：`ChainLadder` R包 / `chainladder-python`

### 推荐策略（修正版）

```
实验数据：SynthETIC（控制模拟实验） + freMTPL2（主实证分析）
对比基准：CAS三角 → Chain-Ladder估计值（作为baseline）
可选加分：Canadian auto（与Calcetero-Vanegas 2025直接对比）
跨领域验证：CMS Medicare（扩大适用性叙事）
```

---

## 四、实验代码框架

### 4.1 项目结构
```
cr_reserving/
├── theory/
│   ├── estimators.py          # 核心CR估计量实现
│   ├── bayesian_cr.py         # 贝叶斯非参数CR (DPMM + MCMC)
│   ├── variance.py            # 方差估计（你原始论文公式(18)的升级版）
│   └── confidence.py          # 置信区间与后验区间
├── baselines/
│   ├── chain_ladder.py        # Chain-Ladder基准
│   ├── bornhuetter_ferguson.py # BF法基准
│   └── state_space.py         # 状态空间模型基准
├── simulation/
│   ├── data_generator.py      # 模拟数据生成（含异质捕获概率）
│   ├── monte_carlo.py         # Monte Carlo实验主循环
│   └── scenarios.py           # 实验场景配置
├── empirical/
│   ├── cas_data_loader.py     # CAS数据加载与预处理
│   ├── reshape_to_cr.py       # 三角数据 → CR格式转换（核心桥梁）
│   └── analysis.py            # 实证分析主程序
├── visualization/
│   ├── convergence_plot.py    # 估计收敛图（类似你原始图1）
│   ├── comparison_plot.py     # CR vs CL对比图
│   └── posterior_plot.py      # 后验分布图
└── paper/
    ├── main.tex               # 论文主文件
    └── figures/
```

### 4.2 核心算法伪代码

#### 被动式CR准备金估计器
```
Algorithm: PassiveCR_Reserve(claims_data, T)
Input: 
  claims_data: 个体级索赔记录 {(claim_id, event_time, event_type)}
  T: 评估时点

1. 构建捕获历史矩阵 X[j,d] for each claim j, period d
2. 计算每期标记总量 Y_d, 再捕获数 x_d, 退出数 s_d, 进入数 k_d
3. 点估计: N_hat = sum(Y_d) / sum(x_d)
4. 方差估计: Var_hat = Formula(18) with layered q_i
5. CI: [N_hat - z * sqrt(Var_hat), N_hat + z * sqrt(Var_hat)]
6. Reserve: R = N_hat * mu_hat - C_paid
Return: (N_hat, Var_hat, R, CI)
```

#### 贝叶斯非参数版本
```
Algorithm: BayesianDPMM_CR(X, n_iter, alpha, G0)
Input:
  X: 捕获历史矩阵 (observed)
  n_iter: MCMC迭代次数

1. Initialize N, {p_j}, cluster assignments {z_j}
2. For t = 1 to n_iter:
   a. Sample N | {p_j}, X  (data augmentation for unobserved)
   b. Sample {p_j} | N, X, {z_j}  (per-cluster Beta posterior)
   c. Sample {z_j} | {p_j}, alpha  (CRP conditional)
   d. Sample alpha | {z_j}  (Gamma posterior)
3. Return posterior samples {N^(t), {p_j}^(t)}
```

### 4.3 模拟实验设计

| 场景 | 描述 | 目的 |
|------|------|------|
| S1 | 封闭系统，同质捕获概率 | 验证基础估计量，CL/CR应等效 |
| S2 | 封闭系统，异质捕获概率（按险种分层） | CR优势场景 |
| S3 | 开放系统，稳定进出 | 验证开放系统理论 |
| S4 | 开放系统，突变（如COVID冲击） | CL崩溃、CR稳健性 |
| S5 | 短尾险种（如车险） | 观测窗口影响 |
| S6 | 长尾险种（如责任险） | 信息累积效应 |
| S7 | 高维异质性（DPMM vs 分层） | 贝叶斯方法价值 |

每场景1000次重复，比较指标：Bias, RMSE, Coverage, Width of CI

---

## 五、目标期刊策略

### 第一梯队（开创性贡献）

| 期刊 | 定位 | 切入角度 |
|------|------|---------|
| **JASA** | 统计学方法论旗舰 | "A capture-recapture framework for loss reserving: theory and inference" |
| **Biometrika** | 统计理论顶刊 | "Asymptotic properties of passive capture-recapture estimators for open populations with applications to insurance" |
| **Annals of Statistics** | 纯理论 | 聚焦DPMM-CR的渐近理论 |

### 第二梯队（精算/保险顶刊）

| 期刊 | 定位 | 切入角度 |
|------|------|---------|
| **Insurance: Mathematics and Economics** | 保险数学旗舰 | 完整的理论+实证，替代Chain-Ladder |
| **ASTIN Bulletin** | IAA官方期刊 | 精算实务导向，强调实用性 |
| **Scandinavian Actuarial Journal** | 北欧精算顶刊 | 理论严谨性 |

### 第三梯队（管理/运筹顶刊，影响力最大）

| 期刊 | 定位 | 切入角度 |
|------|------|---------|
| **Management Science** | 管理科学旗舰 | "Beyond the triangle: A capture-recapture paradigm for claims reserving" |
| **Operations Research** | 运筹学旗舰 | 计算方法+大规模实证 |

### 推荐投稿路径

**路径1（稳妥+高影响力）**：
1. 先投 **Insurance: Mathematics and Economics** 建立领域认知
2. 理论深化后投 **JASA** 或 **Biometrika**

**路径2（直接冲顶）**：
1. 直接投 **JASA**（Theory and Methods section）
2. 若被拒，降级到 **Biometrika** → **IME**

**路径3（跨领域影响力）**：
1. 投 **Management Science** 或 **Operations Research**
2. 强调"范式替代"叙事，辅以大规模实证

---

## 六、论文结构设计

### Title (候选)
1. "Beyond the Run-Off Triangle: A Capture-Recapture Paradigm for Insurance Claims Reserving"
2. "Passive Capture-Recapture Estimation of IBNR Claims: An Alternative to Chain-Ladder"
3. "From Ecology to Actuarial Science: Nonparametric Abundance Estimation for Loss Reserving"

### 结构
1. **Introduction**: 流量三角的70年统治与结构性缺陷；CR的映射直觉
2. **The Structural Isomorphism**: 保险索赔 ↔ 捕获-再捕获的形式化映射
3. **Closed-System CR Reserving**: 封闭系统理论（你原始论文的升级版）
4. **Open-System CR Reserving**: 开放系统理论（核心贡献）
5. **Bayesian Nonparametric Extension**: DPMM处理异质捕获概率
6. **From Abundance to Reserves**: 从总体估计到准备金计算
7. **Asymptotic Theory**: 一致性、渐近正态性、效率比较
8. **Simulation Study**: 7个场景的Monte Carlo对比
9. **Empirical Application**: CAS数据实证
10. **Discussion**: 方法论意义、局限性、未来方向

---

## 七、时间线建议

| 阶段 | 内容 | 时长 |
|------|------|------|
| Phase 1 | 理论框架推导 + LaTeX初稿 | 6-8周 |
| Phase 2 | 代码框架搭建 + 模拟实验 | 4-6周 |
| Phase 3 | CAS数据实证分析 | 3-4周 |
| Phase 4 | 论文撰写与打磨 | 4-6周 |
| Phase 5 | 内部审阅 + 投稿 | 2-4周 |
| **总计** | | **19-28周** |

---

## 八、关键风险与应对

| 风险 | 应对策略 |
|------|---------|
| Schofield式质疑（特例论） | 明确证明CR不是CL的特例：不同的信息集、不同的渐近性质 |
| 审稿人质疑实用性 | 提供R/Python包，展示实操优势 |
| 等概率假设 | DPMM完全回避此假设 |
| 三角数据→CR格式的信息损失 | 论证个体级数据才是自然格式，三角是人为压缩 |
| 与CL结果差异过大 | 模拟实验中展示两者收敛条件 |
