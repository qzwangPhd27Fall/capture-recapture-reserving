# 捕获-再捕获准备金评估：严格理论推导

## Capture-Recapture Loss Reserving: Rigorous Theory

### Phase 1 — 理论推导完整版

---

**摘要：** 本文档提供将捕获-再捕获（Capture-Recapture, CR）方法论系统性地应用于保险索赔准备金评估的严格数学推导。框架分为五个递进层次：封闭系统渐近理论、开放系统隐马尔可夫模型、贝叶斯非参数异质检测概率模型、总体估计到准备金的精算映射、以及与链梯法（Chain-Ladder）的效率比较定理。

---

# 第1章 封闭系统被动式CR估计量的渐近理论

## 1.1 模型设定

考虑一个封闭总体，包含 $N$ 个个体（索赔案件），其中 $N$ 是未知的正整数，也是推断的核心目标。索赔观测过程跨越 $D$ 个发展期，按以下"被动式捕获-再捕获"机制运行。

**定义 1.1（被动式捕获过程）** 在每个离散时点 $t = 1, 2, \ldots$，恰好发生一个捕获事件。令该事件对应一个从总体 $\{1, \ldots, N\}$ 中等概率、有放回抽取的个体。如果该个体此前已被标记（即此前已被观测到），则称该事件为"再捕获"（$x_t = 1$）；否则称该事件为"新捕获"（$x_t = 0$），该个体随即被加入已标记集合。

令 $Y_t$ 表示时点 $t$ 之前已标记的不同个体数量。则：

$$Y_t = \sum_{k=1}^{t} (1 - x_k), \quad Y_0 = 0$$

给定 $Y_{t-1}$，第 $t$ 次捕获事件的再捕获指示变量的条件分布为：

$$x_t \mid Y_{t-1} \sim \text{Bernoulli}\left(\frac{Y_{t-1}}{N}\right)$$

这是因为在均匀有放回抽样下，任意一次抽取恰好选中 $Y_{t-1}$ 个已标记个体之一的概率为 $Y_{t-1} / N$。

设总共观测到 $m$ 次捕获事件。定义：

$$S_Y^{(m)} = \sum_{t=1}^{m} Y_{t-1}, \quad S_x^{(m)} = \sum_{t=1}^{m} x_t$$

## 1.2 似然函数与极大似然估计

### 1.2.1 精确似然

在给定 $Y_0 = 0$ 的条件下，序列 $\{x_t\}_{t=1}^{m}$ 的联合似然函数为：

$$\mathcal{L}_m(N) = \prod_{t=1}^{m} \left(\frac{Y_{t-1}}{N}\right)^{x_t} \left(1 - \frac{Y_{t-1}}{N}\right)^{1 - x_t}$$

对数似然：

$$\ell_m(N) = \sum_{t=1}^{m} \left[x_t \log\left(\frac{Y_{t-1}}{N}\right) + (1 - x_t) \log\left(1 - \frac{Y_{t-1}}{N}\right)\right]$$

### 1.2.2 得分方程

对 $N$ 求导，得分函数为：

$$S_m(N) = \frac{\partial \ell_m}{\partial N} = \sum_{t=1}^{m} \left[-\frac{x_t}{N} + (1 - x_t) \cdot \frac{Y_{t-1}}{N(N - Y_{t-1})}\right]$$

通分整理得：

$$S_m(N) = \sum_{t=1}^{m} \frac{Y_{t-1} - x_t N}{N(N - Y_{t-1})}$$

令 $S_m(\hat{N}_{\text{MLE}}) = 0$，即得 MLE 的得分方程：

$$\sum_{t=1}^{m} \frac{Y_{t-1} - x_t \hat{N}_{\text{MLE}}}{\hat{N}_{\text{MLE}}(\hat{N}_{\text{MLE}} - Y_{t-1})} = 0$$

此方程无显式解，需数值求解（例如 Newton-Raphson 法）。

### 1.2.3 矩估计量

当 $N$ 相对于已标记数 $Y_{t-1}$ 较大时，$N - Y_{t-1} \approx N$，得分方程的一阶近似为：

$$\frac{1}{N^2} \sum_{t=1}^{m} (Y_{t-1} - x_t N) \approx 0 \implies \hat{N}_m = \frac{\sum_{t=1}^{m} Y_{t-1}}{\sum_{t=1}^{m} x_t} = \frac{S_Y^{(m)}}{S_x^{(m)}}$$

这是本文核心的矩估计量。我们强调：**$\hat{N}_m$ 不是 MLE，而是 MLE 的一阶近似**。在 $N$ 大、标记比例适中的情况下，两者的差异是 $O_p(N^{-1})$。

## 1.3 相合性

### 1.3.1 预备引理

**引理 1.1（$Y_t$ 的期望演化）** 在均匀有放回抽样下：

$$E[Y_t \mid Y_{t-1}] = Y_{t-1} + \frac{N - Y_{t-1}}{N} = 1 + \left(1 - \frac{1}{N}\right) Y_{t-1}$$

**证明：** $E[Y_t \mid Y_{t-1}] = Y_{t-1} + E[1 - x_t \mid Y_{t-1}] = Y_{t-1} + (1 - Y_{t-1}/N) = 1 + (1 - 1/N)Y_{t-1}$。$\square$

迭代期望得：

$$E[Y_t] = N\left[1 - \left(1 - \frac{1}{N}\right)^t\right]$$

这是经典的"优惠券收集问题"期望公式：采集 $t$ 次后期望获得的不同优惠券数量。

**引理 1.2（$S_x^{(m)}$ 的期望）**

$$E[S_x^{(m)}] = \frac{1}{N} E[S_Y^{(m)}]$$

**证明：** $E[x_t \mid Y_{t-1}] = Y_{t-1}/N$，由全期望公式立得。$\square$

### 1.3.2 相合性定理

**定理 1.1（弱相合性）** 设 $N$ 固定，令 $m \to \infty$。则 $\hat{N}_m \xrightarrow{p} N$。

**证明：** 当 $m \to \infty$ 时，因为 $Y_t$ 以概率 $1$ 最终达到 $N$（当 $t \to \infty$），且 $\{Y_t\}$ 是吸收态为 $N$ 的 Markov 链，过程满足遍历性。

定义 $S_Y^{(m)} / m$ 和 $S_x^{(m)} / m$ 的几乎必然极限。当 $m \to \infty$：

$$\frac{S_Y^{(m)}}{m} = \frac{1}{m} \sum_{t=1}^{m} Y_{t-1} \to E_{\pi}[Y]$$

其中 $E_{\pi}[Y]$ 是关于过程 $\{Y_t\}$ 在吸收前的拟平稳分布的期望。由于极限下 $Y_t$ 在 $N$ 附近集中，

$$\frac{S_x^{(m)}}{m} \to \frac{E_{\pi}[Y]}{N}$$

因此 $\hat{N}_m = S_Y^{(m)} / S_x^{(m)} \to N$。更严格地，可用 Markov 链大数定律（Meyn & Tweedie, 1993）完成证明。$\square$

**注 1.1（有限 $m$ 下的表现）** 在实践中，$m$（捕获事件数）随总体大小 $N$ 和观测期数 $D$ 一起增长。设 $m = m(N)$ 满足 $m(N) / N \to \infty$，则相合性以 $m(N)^{-1/2}$ 的速率成立。特别地，若 $m / N \to \rho \in (0, \infty)$，则 $\hat{N}_m$ 的渐近偏差为 $O(1)$，需进行有限样本纠偏。

## 1.4 渐近正态性

### 1.4.1 关键挑战

$\{x_t\}$ 序列不独立——$x_t$ 依赖于 $Y_{t-1}$，而 $Y_{t-1}$ 又依赖于之前的 $\{x_1, \ldots, x_{t-1}\}$。因此经典 i.i.d. 中心极限定理不直接适用。我们采用 **Martingale 中心极限定理** 来处理依赖性。

### 1.4.2 Martingale 差分表示

定义过程 $\{M_t\}_{t \geq 1}$：

$$M_t = x_t - \frac{Y_{t-1}}{N}$$

则 $E[M_t \mid \mathcal{F}_{t-1}] = 0$，其中 $\mathcal{F}_{t-1} = \sigma(Y_1, \ldots, Y_{t-1}, x_1, \ldots, x_{t-1})$。因此 $\{M_t\}$ 是关于自然滤子 $\{\mathcal{F}_t\}$ 的鞅差序列（MDS）。

类似地，定义 $\{Z_t\}$：

$$Z_t = (1 - x_t) - \frac{N - Y_{t-1}}{N} = \frac{Y_{t-1}}{N} - x_t = -M_t$$

### 1.4.3 渐近正态性定理

**定理 1.2（渐近正态性）** 设 $m = m(N) \to \infty$ 且 $m/N \to \rho \in (0, \infty)$。令：

$$\sigma_m^2 = \frac{1}{m} \sum_{t=1}^{m} \text{Var}(x_t \mid \mathcal{F}_{t-1}) = \frac{1}{m} \sum_{t=1}^{m} \frac{Y_{t-1}}{N} \left(1 - \frac{Y_{t-1}}{N}\right)$$

若 $\sigma_m^2 \xrightarrow{p} \sigma^2 > 0$，则：

$$\sqrt{m} (\hat{N}_m - N) \xrightarrow{d} \mathcal{N}(0, \Sigma)$$

其中渐近方差 $\Sigma$ 为：

$$\Sigma = \frac{N^2}{\mu_x^2} \cdot \sigma_Y^2 + \frac{N^2 \mu_Y^2}{\mu_x^4} \cdot \sigma_x^2 - 2 \frac{N^2 \mu_Y}{\mu_x^3} \cdot \sigma_{Yx}$$

这里 $\mu_x = \lim_{m \to \infty} E[S_x^{(m)}/m]$，$\mu_Y = \lim_{m \to \infty} E[S_Y^{(m)}/m]$，$\sigma_Y^2$、$\sigma_x^2$、$\sigma_{Yx}$ 为相应的渐近方差和协方差。

**证明概要：** 将 $\hat{N}_m - N$ 写成：

$$\hat{N}_m - N = \frac{S_Y^{(m)} - N S_x^{(m)}}{S_x^{(m)}} = \frac{\sum_{t=1}^{m} (Y_{t-1} - N x_t)}{S_x^{(m)}}$$

定义 $U_t = Y_{t-1} - N x_t$。则 $E[U_t \mid \mathcal{F}_{t-1}] = Y_{t-1} - N \cdot (Y_{t-1}/N) = 0$。因此 $\{U_t\}$ 也是 MDS。

由鞅中心极限定理（Billingsley, 1961），若 Lindeberg 条件成立，则：

$$\frac{1}{\sqrt{m}} \sum_{t=1}^{m} U_t \xrightarrow{d} \mathcal{N}(0, \tau^2)$$

其中 $\tau^2 = \lim_{m \to \infty} \frac{1}{m} \sum_{t=1}^{m} E[U_t^2 \mid \mathcal{F}_{t-1}]$。

再由 $S_x^{(m)} / m \xrightarrow{p} \mu_x$ 和 Slutsky 定理，得：

$$\sqrt{m}(\hat{N}_m - N) = \frac{\frac{1}{\sqrt{m}} \sum U_t}{S_x^{(m)}/m} \xrightarrow{d} \mathcal{N}\left(0, \frac{\tau^2}{\mu_x^2}\right)$$

其中 $\tau^2 = E[(Y - Nx)^2] = E[Y^2] - 2N E[Yx] + N^2 E[x^2]$，在平稳分布下取值。$\square$

### 1.4.4 渐近方差的显式表达式

在 $m \gg N$ 的极限下（即观测到远超 $N$ 次的捕获事件），$Y_t \approx N$ 以高概率成立。此时：

$$E[U_t^2 \mid \mathcal{F}_{t-1}] = \text{Var}(Y_{t-1} - N x_t \mid \mathcal{F}_{t-1}) = N^2 \cdot \frac{Y_{t-1}}{N} \left(1 - \frac{Y_{t-1}}{N}\right) = N Y_{t-1} \left(1 - \frac{Y_{t-1}}{N}\right)$$

当 $Y_{t-1} \approx N$ 时趋于 $0$，当 $Y_{t-1} \approx N/2$ 时最大（等于 $N^2/4$）。

**推论 1.1（方差估计）** $\text{Var}(\hat{N}_m)$ 的相合估计量为：

$$\widehat{\text{Var}}(\hat{N}_m) = \frac{\sum_{t=1}^{m} (Y_{t-1} - \hat{N}_m x_t)^2}{\left(\sum_{t=1}^{m} x_t\right)^2}$$

$N$ 的渐近 $100(1-\alpha)\%$ 置信区间为：

$$\hat{N}_m \pm z_{\alpha/2} \cdot \sqrt{\widehat{\text{Var}}(\hat{N}_m)}$$

## 1.5 可识别性条件：回应 Link (2003)

Link (2003, *Biometrics*) 证明了 CR 文献中一个根本性结果：**在异质检测概率下，仅凭捕获-再捕获数据无法识别总体大小 $N$**。即使样本量任意大，不同的异质性模型也会生成几乎相同的观测数据模式，但给出截然不同的 $N$ 估计值。

### 1.5.1 Link 定理的关键假设

Link (2003) 的不可识别性结论建立在以下设定上：

(i) 每个个体 $j$ 具有固定的捕获概率 $p_j$；

(ii) 各次捕获是独立的；

(iii) 仅观测到 $\{p_j\}$ 在已捕获个体上的分布，而对未捕获个体上的分布一无所知。

在此设定下，$p_j$ 在未观测部分的外推本质上无法被数据约束——这是非参数不可识别性的根源。

### 1.5.2 保险场景为何不同：时间标记论证

在保险 CR 中，我们拥有一个关键的结构性优势：**捕获事件按时间顺序发生**。

每一个捕获事件 $i$ 不仅携带"是否为再捕获"的信息（$x_i$），还携带其发生的**时间标记** $d_i \in \{1, \ldots, D\}$。发展期 $d$ 本身提供了额外的识别约束：

**论证 1（时间标记提供识别信息）** 发展期 $d$ 的结构在保险场景中是可观测的，且具有自然的时间顺序。当我们将捕获概率分解为 $p_{jd} = \text{logit}^{-1}(\beta_{0d} + \gamma_j)$ 时：

- $\beta_{0d}$（基线检测率）由各发展期的事件频率识别；
- $\gamma_j$（个体异质性）由同一索赔在不同发展期的重复观测模式识别；
- 由于 $\beta_{0d}$ 在期数 $D \geq 2$ 时由跨期差异识别，$\gamma_j$ 的不可识别性被打破。

**命题 1.1（保险 CR 的可识别性）** 在保险被动式 CR 模型中，若 $D \geq 3$（至少 3 个发展期），且满足：

(i) $\beta_{0d}$ 随 $d$ 变化（时序变化）；

(ii) 每个发展期 $d$ 内的捕获事件数 $m_d$ 满足 $m_d \to \infty$；

(iii) 异质性参数 $\gamma_j$ 的分布 $G$ 属于某参数族（例如 Beta 混合），

则 $(N, \boldsymbol{\beta}_0, G)$ 是（局部）可识别的。

**证明概要：** 将似然分解为跨发展期的乘积：

$$\mathcal{L}(N, \boldsymbol{\beta}_0, G) = \prod_{d=1}^{D} \prod_{j} \left[p_{jd}^{x_{jd}} (1 - p_{jd})^{1 - x_{jd}}\right]$$

对于每个 $d$，基线 $\beta_{0d}$ 由 $\sum_j x_{jd} / (N \cdot \text{暴露量})$ 识别。给定 $\beta_{0d}$，第 $j$ 个个体跨不同 $d$ 的捕获模式 $\{x_{j1}, \ldots, x_{jD}\}$ 携带关于 $\gamma_j$ 的信息。当 $D \geq 3$ 时，每个个体贡献 $D$ 个二元观测值来推断单一标量参数 $\gamma_j$，提供了足够的重复测量来分离异质性与抽样误差。通过 deconvolution 论证（参见 Efron, 2016, *JASA*），$G$ 在 $D \geq 3$ 且 $m_d$ 充分大时被非参数识别。$\square$

**重要说明：** 此可识别性论证依赖于发展期的时间结构——这是传统生态学 CR 中通常不具备的特征（在生态学中，捕获事件通常不被赋予固有的时间标签，或者时间仅以离散的"捕获场合"出现，而这些场合通常很少（$K = 2, 3, 4$）且缺乏内在的时间梯度）。保险场景中，$D$ 可达到 $10$ 或更多，且发展期具有自然的因果结构（越晚的发展期距离事故发生越远，检测概率应当递减）。这一**时间因果结构**是保险 CR 独有的识别优势。

---

# 第2章 开放系统隐马尔可夫状态转移模型

## 2.1 从静态到动态：为什么需要开放系统

第1章假设总体 $N$ 在观测期间保持不变（封闭系统）。实际保险运营中：

- **新保单生效** $\to$ 新索赔进入系统；
- **保单到期/退保** $\to$ 风险暴露终止；
- **赔案结案** $\to$ 个体退出观测池；
- **结案重开** $\to$ 已退出个体重新出现。

原论文（Xie & Rehman, 2017）用简单差分方程 $Y_t = Y_{t-1} + 1 - x_t - s_t + k_t$ 刻画开放系统动态。本章将其升级为**隐马尔可夫模型（HMM）**，用状态转移矩阵精细表达索赔生命周期的所有可能路径。

## 2.2 状态空间定义

**定义 2.1（索赔状态空间）** 在任意时点 $d$，每个索赔 $j$ 恰好属于以下五个状态之一：

$$\mathcal{S} = \{U, R, A, S, Re\}$$

| 符号 | 状态 | 含义 | 可观测？ |
|------|------|------|---------|
| $U$ | Unreported | 事故已发生但未报案（IBNR） | 否 |
| $R$ | Reported | 已报案，尚未开始赔付 | 部分（报案可观测） |
| $A$ | Active | 正在赔付/调整中 | 是（赔付活动可观测） |
| $S$ | Settled | 已结案 | 是（结案可观测） |
| $Re$ | Reopened | 结案后重新开案 | 是（重开可观测） |

## 2.3 状态转移矩阵

令 $T$ 为 $5 \times 5$ 转移概率矩阵，$T_{kl} = P(S_{d+1} = l \mid S_d = k)$，其中 $S_d$ 为索赔在第 $d$ 期的状态。

状态转移矩阵具有以下结构性约束（反映真实的索赔生命周期）：

$$T = \begin{pmatrix}
T_{UU} & T_{UR} & 0 & 0 & 0 \\
0 & T_{RR} & T_{RA} & T_{RS} & 0 \\
0 & T_{AR} & T_{AA} & T_{AS} & 0 \\
0 & 0 & 0 & T_{SS} & T_{S,Re} \\
0 & T_{Re,R} & 0 & T_{Re,S} & T_{Re,Re}
\end{pmatrix}$$

约束的合理性：
1. **$U$ 只能转移到 $U$ 或 $R$**：未报案案件不能直接开始赔付或结案；
2. **$R$ 可转移到 $A$、$S$ 或留在 $R$**：报案后可开始赔付、直接结案（零赔付案件）、或等待；
3. **$A$ 可转移到 $R$、$A$、$S$**：赔付过程中可能暂停、继续或结案；
4. **$S$ 可转移到 $S$ 或 $Re$**：结案后保持结案或重开；
5. **$Re$ 可转移到 $R$、$S$、$Re$**：重开后可能重新评估、再次结案或持续。

## 2.4 精算参数化

将转移概率映射到精算师最关心的参数：

**定义 2.2（精算参数映射）**

- **报案率：** $\lambda_d = T_{UR}(d) = P(S_{d+1} = R \mid S_d = U)$
- **赔付启动率：** $\alpha_d = T_{RA}(d)$
- **结案率：** $\gamma_d = T_{AS}(d) + T_{RS}(d)$（从活跃状态到结案的加权平均）
- **重开率：** $\rho_d = T_{S,Re}(d)$
- **重新结案率：** $\delta_d = T_{Re,S}(d)$

这些参数可以进一步参数化为发展期 $d$ 的函数，例如：

$$\text{logit}(\lambda_d) = \beta_0^{\lambda} + \beta_1^{\lambda} \cdot d$$

## 2.5 发射分布

HMM 的发射分布将隐状态映射到可观测输出。定义观测向量 $O_d = (O_d^1, O_d^2) \in \{0, 1\}^2$：

- $O_d^1 = 1$：第 $d$ 期发生赔付活动
- $O_d^2 = 1$：第 $d$ 期发生结案事件

发射概率矩阵 $E$（$5 \times 4$，状态 × 观测组合）：

$$E_{k, (o_1, o_2)} = P(O = (o_1, o_2) \mid S = k)$$

其中：
- $E_{U, (0,0)} = 1$（未报案状态无观测）
- $E_{A, (1,0)} = 1 - \gamma_d$，$E_{A, (1,1)} = \gamma_d$（活跃状态以 $\gamma_d$ 概率结案）
- $E_{S, (0,0)} = 1 - \rho_d$，$E_{S, (0,1)} = \rho_d$（结案状态以 $\rho_d$ 概率重开）
- $E_{Re, (1,0)} = 1$（重开状态下必有活动）

## 2.6 EM 算法：Baum-Welch 特化

### 2.6.1 前向-后向递归

对每个索赔 $j$，定义前向概率 $\alpha_{j,d}(k) = P(S_d = k, O_{j,1:d} \mid \theta)$：

$$\alpha_{j,1}(k) = \pi_k \cdot E_{k, O_{j,1}}$$

$$\alpha_{j, d+1}(l) = \left(\sum_{k=1}^{5} \alpha_{j,d}(k) \cdot T_{kl}\right) \cdot E_{l, O_{j,d+1}}$$

后向概率 $\beta_{j,d}(k) = P(O_{j,d+1:D} \mid S_d = k, \theta)$：

$$\beta_{j, D}(k) = 1$$

$$\beta_{j,d}(k) = \sum_{l=1}^{5} T_{kl} \cdot E_{l, O_{j,d+1}} \cdot \beta_{j,d+1}(l)$$

### 2.6.2 E 步：期望充分统计量

状态后验概率：

$$\xi_{j,d}(k) = P(S_d = k \mid O_{j,1:D}, \theta) = \frac{\alpha_{j,d}(k) \cdot \beta_{j,d}(k)}{\sum_{k'} \alpha_{j,d}(k') \cdot \beta_{j,d}(k')}$$

转移后验概率：

$$\zeta_{j,d}(k, l) = P(S_d = k, S_{d+1} = l \mid O_{j,1:D}, \theta) = \frac{\alpha_{j,d}(k) \cdot T_{kl} \cdot E_{l, O_{j,d+1}} \cdot \beta_{j,d+1}(l)}{\sum_{k', l'} \alpha_{j,d}(k') \cdot T_{k'l'} \cdot E_{l', O_{j,d+1}} \cdot \beta_{j,d+1}(l')}$$

期望充分统计量：

$$\hat{T}_{kl} = \frac{\sum_{j=1}^{n_{\text{obs}}} \sum_{d=1}^{D-1} \zeta_{j,d}(k, l)}{\sum_{j=1}^{n_{\text{obs}}} \sum_{d=1}^{D-1} \sum_{l'} \zeta_{j,d}(k, l')}$$

$$\hat{E}_{k, o} = \frac{\sum_{j=1}^{n_{\text{obs}}} \sum_{d=1}^{D} \xi_{j,d}(k) \cdot I(O_{j,d} = o)}{\sum_{j=1}^{n_{\text{obs}}} \sum_{d=1}^{D} \xi_{j,d}(k)}$$

### 2.6.3 M 步：参数更新

在无额外参数化约束时，直接用上述经验频率更新 $T$ 和 $E$。若对转移概率施加了精算参数化（如 $\text{logit}(\lambda_d) = \beta^{\lambda}_0 + \beta^{\lambda}_1 d$），则 M 步变为加权逻辑回归。

## 2.7 总体大小估计

对任意参数 $\theta = (T, E)$，一个个体在整个 $D$ 期中从未被观测到的概率为：

$$p_0(\theta) = P\left(\sum_{d=1}^{D} \sum_{j} I(\text{observed at } d) = 0\right)$$

在 HMM 框架下：

$$p_0(\theta) = \sum_{k=1}^{5} \pi_k \cdot \prod_{d=1}^{D} P(\text{not observed at } d \mid S_d = k)$$

即从未离开"可产生观测"状态路径的概率。

HMM-CR 总体估计量：

$$\hat{N}_{\text{HMM}} = \frac{n_{\text{obs}}}{1 - \hat{p}_0}$$

其中 $\hat{p}_0 = p_0(\hat{\theta}_{\text{BW}})$，$\hat{\theta}_{\text{BW}}$ 为 Baum-Welch 估计。

**定理 2.1（$\hat{N}_{\text{HMM}}$ 的相合性）** 若真实状态转移矩阵 $T^*$ 满足：

(i) 从 $U$ 到可观测状态（$R, A, S, Re$）的转移概率严格为正；

(ii) Markov 链不可约（最终各状态可达）；

(iii) $n_{\text{obs}} \to \infty$ 且 $D$ 充分大使得 $\hat{\theta}_{\text{BW}} \xrightarrow{p} \theta^*$，

则 $\hat{N}_{\text{HMM}} \xrightarrow{p} N$。

**证明概要：** 由 Baum-Welch 算法的相合性（Baum et al., 1970; Leroux, 1992）得 $\hat{\theta}_{\text{BW}} \xrightarrow{p} \theta^*$。由连续映射定理，$\hat{p}_0 \xrightarrow{p} p_0^*$。再由 Slutsky 定理，$\hat{N}_{\text{HMM}} = n_{\text{obs}} / (1 - \hat{p}_0) \xrightarrow{p} N(1 - p_0^*) / (1 - p_0^*) = N$。$\square$

---

# 第3章 DPMM 异质检测概率的贝叶斯非参数推断

## 3.1 问题陈述

第1章的矩估计量 $\hat{N}_m = S_Y / S_x$ 在**同质捕获概率**假设下有良好的渐近性质。然而在实际保险数据中，索赔异质性普遍存在：

- 严重案件比轻微案件更容易被多次记录（报案 → 查勘 → 定损 → 多次赔付）；
- 不同险种/地区的报案率存在系统性差异；
- 律所介入的案件具有不同的"可见度"模式。

**关键困难（Link, 2003）：** 在没有额外约束的条件下，异质检测概率使 $N$ 不可识别。本章论证——保险场景中的**时间维度**提供了这种约束。

## 3.2 Dirichlet Process Mixture Model 设定

### 3.2.1 模型层次

观测模型（第 $j$ 个索赔，第 $d$ 个发展期）：

$$x_{jd} \mid p_{jd} \sim \text{Bernoulli}(p_{jd})$$

$$\text{logit}(p_{jd}) = \beta_{0d} + \gamma_j$$

其中 $\beta_{0d}$ 为第 $d$ 期的基线检测率（在 $d$ 上可变化），$\gamma_j$ 为第 $j$ 个索赔的个体异质性参数。

先验层次：

$$\gamma_j \mid G \sim G, \quad j = 1, \ldots, N$$

$$G \sim \text{DP}(\alpha, G_0)$$

$$G_0 = \mathcal{N}(0, \sigma_\gamma^2)$$

$$\alpha \sim \text{Gamma}(a_\alpha, b_\alpha)$$

$$\beta_{0d} \sim \mathcal{N}(0, \sigma_\beta^2), \quad d = 1, \ldots, D$$

### 3.2.2 DPMM 的 Stick-Breaking 表示

Sethuraman (1994) 给出了 DP 的 stick-breaking 构造：

$$G = \sum_{h=1}^{\infty} w_h \delta_{\gamma_h^*}$$

其中：

$$w_h = v_h \prod_{\ell=1}^{h-1} (1 - v_\ell), \quad v_h \sim \text{Beta}(1, \alpha)$$

$$\gamma_h^* \sim G_0$$

$\delta_x$ 为 $x$ 处的点质量。

### 3.2.3 聚类视角

有限样本中，$\gamma_j$ 值会自然聚类为 $K \leq N$ 个不同值。这对应于 Chinese Restaurant Process (CRP)：

$$P(z_j = h \mid \mathbf{z}_{-j}, \alpha) = \begin{cases}
\frac{n_h^{-j}}{N - 1 + \alpha}, & \text{existing cluster } h \\
\frac{\alpha}{N - 1 + \alpha}, & \text{new cluster}
\end{cases}$$

其中 $z_j$ 为第 $j$ 个索赔的聚类分配，$n_h^{-j}$ 为除去 $j$ 后第 $h$ 类的成员数。

## 3.3 时间标记提供识别约束：正式论证

**命题 3.1（时间维度的识别力量）** 考虑两个总体大小 $N_1 \neq N_2$ 以及两组异质性分布 $G_1 \neq G_2$。在以下保险特有的条件下，$(N_1, G_1)$ 和 $(N_2, G_2)$ 生成的观测分布在 $D \geq 3$ 时是可区分的：

(i) $\beta_{0d}$ 随 $d$ 严格递减（反映了报案-赔付-结案的生命周期衰减）；

(ii) 对于每个 $d$，被观测到的索赔数量 $n_d^{\text{obs}}$ 由其暴露量决定，并随 $N$ 增长；

(iii) 个体的跨期捕获模式携带关于 $\gamma_j$ 的信息。

**证明：** 将观测数据的联合分布分解为：

$$P(\mathbf{X}^{\text{obs}} \mid N, G) = \binom{N}{n_{\text{obs}}} \cdot P(\text{exactly } n_{\text{obs}} \text{ observed}) \cdot P(\mathbf{X}^{\text{obs}} \mid n_{\text{obs}}, G)$$

第一项因子——"恰好 $n_{\text{obs}}$ 个被观测到"的概率——是 $N$ 的函数：

$$P(\text{observed}) = 1 - \int \left[\prod_{d=1}^{D} (1 - \text{logit}^{-1}(\beta_{0d} + \gamma))\right] dG(\gamma)$$

对于固定的 $(G, \boldsymbol{\beta}_0)$，这是 $\gamma$ 的泛函，且对不同的 $N$，由于 $n_{\text{obs}}$ 的二项分布均值 $N \cdot (1 - p_0)$ 随 $N$ 变化，$n_{\text{obs}}$ 的分布提供了（弱的）$N$ 识别。

更关键的是，对于已观测部分，$\gamma_j$ 的分布 $G$ 由跨期捕获模式 $\mathbf{x}_j = (x_{j1}, \ldots, x_{jD})$ 识别。当 $D \geq 3$ 时，每个索赔提供 $D$ 个二元观测来推断单一的 $\gamma_j$。这构成经典的 **deconvolution 问题**。

令 $f_k(\mathbf{x}) = \int \prod_{d=1}^{D} p_d(\gamma)^{x_d} (1 - p_d(\gamma))^{1 - x_d} dG(\gamma)$ 为观测到捕获模式 $\mathbf{x}$ 的概率（在已捕获个体上的条件分布）。通过 Efron (2016) 的 $g$-modeling 论证，$G$ 在 $D \geq 3$ 时由 $f_k(\mathbf{x})$ 非参数识别。进而 $p_0 = \int \prod_{d=1}^{D} (1 - p_d(\gamma)) dG(\gamma)$ 被识别，因此 $N = n_{\text{obs}} / (1 - p_0)$ 被识别。$\square$

**关键洞察：** 生态学 CR 中通常只有 $K = 2$ 或 $3$ 个"捕获场合"且无自然时序，因此 Link (2003) 的不可识别性结论成立。保险场景中的 $D$ 可以大到 $10$ 或以上，且随时间衰减的 $\beta_{0d}$ 提供了函数形式的约束——这在数学上等价于在线性模型中增加工具变量。

## 3.4 后验计算：Neal's Algorithm 8 Gibbs 采样器

采用数据增广策略（Tanner & Wong, 1987）：引入 $M_{\text{aug}}$ 个"全零"捕获历史代表未观测个体。

**算法 3.1（DPMM-CR Gibbs 采样器）**

---

**输入：** 观测捕获历史 $\mathbf{X}^{\text{obs}}$（$n_{\text{obs}} \times D$），增广数 $M_{\text{aug}}$，MCMC 迭代数 $n_{\text{iter}}$，烧入期 $n_{\text{burn}}$

**输出：** $N$ 的后验样本 $\{N^{(t)}\}_{t=n_{\text{burn}}+1}^{n_{\text{iter}}}$

---

**Step 0（初始化）：**
- 设 $M = n_{\text{obs}} + M_{\text{aug}}$
- 构造增广矩阵 $\mathbf{X}$（$M \times D$）——前 $n_{\text{obs}}$ 行为观测值，其余为 $\mathbf{0}$
- $\psi_j = I(j \leq n_{\text{obs}})$：指示第 $j$ 个个体是否属于总体
- $\omega \sim \text{Beta}(1, 1)$（包含概率）
- 为每个个体抽样初始聚类分配 $z_j$（例如全部归入第 1 类）
- 抽样初始 $\gamma_h^* \sim \mathcal{N}(0, \sigma_\gamma^2)$，$\beta_{0d} \sim \mathcal{N}(0, \sigma_\beta^2)$

**对于 $t = 1$ 到 $n_{\text{iter}}$：**

**Step 1（增广个体的包含状态）：** 对于 $j = n_{\text{obs}} + 1$ 到 $M$：

$$p_{jd} = \text{logit}^{-1}(\beta_{0d} + \gamma_{z_j}^*), \quad d = 1, \ldots, D$$

$$p_j^{\text{all-zero}} = \prod_{d=1}^{D} (1 - p_{jd})$$

$$\eta_j = \frac{\omega \cdot p_j^{\text{all-zero}}}{\omega \cdot p_j^{\text{all-zero}} + (1 - \omega)}$$

$$\psi_j \sim \text{Bernoulli}(\eta_j)$$

**Step 2（更新 $\omega$）：**

$$\omega \sim \text{Beta}\left(1 + \sum_{j=1}^{M} \psi_j, \; 1 + M - \sum_{j=1}^{M} \psi_j\right)$$

**Step 3（更新聚类分配 — Neal's Algorithm 8）：** 对于每个 $j$（$\psi_j = 1$）：

- 计算 $j$ 排除后每类的成员数 $n_h^{-j}$
- 对于 $h = 1, \ldots, K$（现有类）：

$$q_h \propto n_h^{-j} \cdot \prod_{d=1}^{D} p_{jd}(\gamma_h^*)^{x_{jd}} (1 - p_{jd}(\gamma_h^*))^{1 - x_{jd}}$$

- 对于新类（$h = K + 1$ 到 $K + m_{\text{aux}}$）：

$$q_h \propto \frac{\alpha}{m_{\text{aux}}} \cdot \int \prod_{d=1}^{D} p_{jd}(\gamma)^{x_{jd}} (1 - p_{jd}(\gamma))^{1 - x_{jd}} dG_0(\gamma)$$

- 按归一化概率从 $\{1, \ldots, K + m_{\text{aux}}\}$ 中抽样新的 $z_j$
- 清理空类，更新 $K$ 和 $\gamma_h^*$ 列表

**Step 4（更新聚类参数 $\gamma_h^*$）：** 对于每个类 $h = 1, \ldots, K$：

- 用 Metropolis-Hastings 步：提议 $\gamma_h^{\text{prop}} \sim \mathcal{N}(\gamma_h^*, \tau_{\text{MH}}^2)$
- 接受概率：

$$\alpha_{\text{MH}} = \min\left(1, \frac{\prod_{j: z_j = h} \prod_{d} p_{jd}(\gamma_h^{\text{prop}})^{x_{jd}} (1 - p_{jd}(\gamma_h^{\text{prop}}))^{1 - x_{jd}} \cdot \phi(\gamma_h^{\text{prop}} / \sigma_\gamma)}{\prod_{j: z_j = h} \prod_{d} p_{jd}(\gamma_h^*)^{x_{jd}} (1 - p_{jd}(\gamma_h^*))^{1 - x_{jd}} \cdot \phi(\gamma_h^* / \sigma_\gamma)}\right)$$

**Step 5（更新 $\beta_{0d}$）：** 对于每个发展期 $d = 1, \ldots, D$，用 MH 步（类似 Step 4）。

**Step 6（更新 $\alpha$ — 可选）：** 若对 DP 集中参数使用超先验（Escobar & West, 1995）：

$$\alpha \mid K, M \sim \text{Gamma}(a_\alpha + K - 1, b_\alpha - \log(\eta))$$

其中 $\eta \sim \text{Beta}(\alpha + 1, M)$ 为辅助变量。

**Step 7（存储）：** 若 $t > n_{\text{burn}}$，存储：

$$N^{(t - n_{\text{burn}})} = \sum_{j=1}^{M} \psi_j$$

---

## 3.5 后验相合性

**定理 3.1（$N$ 的后验相合性）** 在上述 DPMM-CR 模型下，若真实参数 $(N_0, G_0, \boldsymbol{\beta}_0^0)$ 满足命题 3.1 的可识别性条件，且 $n_{\text{obs}} \to \infty$，$D \geq 3$，则 $N$ 的后验分布在 $N_0$ 处相合：

$$\Pi(N \in \cdot \mid \mathbf{X}^{\text{obs}}) \xrightarrow{w} \delta_{N_0}(\cdot), \quad \text{a.s. under } P_{(N_0, G_0, \boldsymbol{\beta}_0^0)}$$

**证明概要：** 由 Schwartz (1965) 定理，需验证：(i) 先验在真实参数处赋予正的 Kullback-Leibler 邻域概率；(ii) 存在相合检验序列。

条件 (i) 由 DP 先验的支撑性质满足——DP 在弱拓扑下具有完全支撑（Ghosal, 2010）。

条件 (ii)：构造检验 $H_0: (N, G) = (N_0, G_0)$ vs $H_1: \|N - N_0\| \geq \varepsilon$。由可识别性，在 $H_1$ 下存在某个检验函数 $\phi(\mathbf{X}^{\text{obs}})$ 具有趋于 $1$ 的功效。具体证明需用到 $g$-modeling 框架下 $G$ 的可估计性（Efron, 2016）。$\square$

---

# 第4章 从总体估计到准备金的精算映射

## 4.1 准备金公式

令 $\hat{N}_a$ 为事故年 $a$ 的估计索赔总数（含 IBNR）。则事故年 $a$ 的未决赔款准备金为：

$$R_a = \hat{N}_a^{\text{IBNR}} \cdot \hat{\mu}_a + \sum_{j \in \text{RBNS}} (\hat{\mu}_{aj} - C_{aj}^{\text{paid}})$$

其中：
- $\hat{N}_a^{\text{IBNR}} = \hat{N}_a - n_a^{\text{obs}}$：估计的 IBNR 案件数；
- $\hat{\mu}_a$：事故年 $a$ 的平均最终索赔额（基于已结案数据估计）；
- $\hat{\mu}_{aj}$：已报案未结案（RBNS）第 $j$ 案的个案估计；
- $C_{aj}^{\text{paid}}$：RBNS 第 $j$ 案的已付赔款。

简化到仅含 IBNR 的情况（RBNS 部分为标准精算操作）：

$$R_a^{\text{IBNR}} = (\hat{N}_a - n_a^{\text{obs}}) \cdot \hat{\mu}_a$$

## 4.2 预测均方误差分解

**定理 4.1（MSEP 分解）** CR 准备金估计量的条件预测均方误差为：

$$\text{MSEP}(R_a^{\text{IBNR}}) = E\left[(R_a^{\text{IBNR}} - R_a^{\text{true}})^2 \mid \mathcal{D}\right]$$

分解为三项：

$$\text{MSEP}(R_a^{\text{IBNR}}) = \underbrace{\mu_a^2 \cdot \text{Var}(\hat{N}_a \mid \mathcal{D})}_{\text{总体估计误差}} + \underbrace{(N_a - n_a^{\text{obs}})^2 \cdot \text{Var}(\hat{\mu}_a \mid \mathcal{D})}_{\text{严重度估计误差}} + \underbrace{\text{Var}(\hat{N}_a \mid \mathcal{D}) \cdot \text{Var}(\hat{\mu}_a \mid \mathcal{D})}_{\text{交叉项}}$$

**证明：** 假设 $\hat{N}_a$ 和 $\hat{\mu}_a$ 条件独立（在给定已付赔款信息后），则：

$$E[(\hat{N}_a^{\text{IBNR}} \hat{\mu}_a - N_a^{\text{IBNR}} \mu_a)^2] = E[(\hat{N}_a^{\text{IBNR}}(\hat{\mu}_a - \mu_a) + \mu_a(\hat{N}_a^{\text{IBNR}} - N_a^{\text{IBNR}}))^2]$$

展开并利用条件独立得上述三项。$\square$

**注 4.1** 第一项占主导——总体大小 $N$ 的估计误差通常大于严重度估计误差，因为 $N$ 的推断面向未观测部分，信息极为稀疏。这凸显了 CR 方法论的价值：更精确的 $N$ 估计直接转化为更精确的准备金。

## 4.3 贝叶斯框架下的准备金分布

在 DPMM-CR 的后验框架（第3章）下，准备金 $R_a^{\text{IBNR}}$ 具有完整的后验分布：

$$R_a^{\text{IBNR}(t)} = (N^{(t)} - n_a^{\text{obs}}) \cdot \mu_a^{(t)}, \quad t = n_{\text{burn}}+1, \ldots, n_{\text{iter}}$$

其中：
- $N^{(t)}$ 为第 $t$ 次 MCMC 迭代的总体大小；
- $\mu_a^{(t)}$ 为相应的严重度后验抽样（例如从 $\mu_a \mid \text{settled claims} \sim \text{LogNormal}(\hat{\nu}_a, \hat{\tau}_a^2)$ 中抽样）。

准备金 $R_a$ 的贝叶斯点估计和区间估计：

$$\hat{R}_a^{\text{Bayes}} = \frac{1}{n_{\text{iter}} - n_{\text{burn}}} \sum_{t} R_a^{\text{IBNR}(t)}$$

$$\text{CrI}_{1-\alpha}(R_a) = \left[Q_{\alpha/2}(\{R_a^{\text{IBNR}(t)}\}), \; Q_{1-\alpha/2}(\{R_a^{\text{IBNR}(t)}\})\right]$$

其中 $Q_p$ 为 $p$-分位数。

## 4.4 与 Chain-Ladder 预测误差的对比

Mack (1993) 给出的 CL 准备金 MSEP：

$$\text{MSEP}_{\text{Mack}}(\hat{R}^{\text{CL}}) = \sum_{i=1}^{I} \left(\text{Var}(\hat{R}_i^{\text{CL}}) + (\text{estimation error})^2\right)$$

CL 的误差来源是**发展因子的外推不确定性**；CR 的误差来源是**总体大小的推断不确定性**。两者在数学结构上是不同的，这为第5章的效率比较提供了基础。

---

# 第5章 CR 与 Chain-Ladder 的效率比较定理

## 5.1 比较框架

考虑一个简化的比较设定：单一事故年，$D$ 个发展期，已知真实的平均严重度 $\mu$。目标是估计未决索赔的**数量** $N^{\text{IBNR}} = N - n_{\text{obs}}$（乘以 $\mu$ 即为 IBNR 准备金）。

两个竞争估计量：

**Chain-Ladder（CL）：** 用发展三角的增长率 $f_d$（LDF）外推未观测的索赔数。实质是对聚合计数三角 $(C_{1,1}, \ldots, C_{1,D})$ 应用乘法模型。

**Capture-Recapture（CR）：** $\hat{N}_{\text{CR}} = S_Y / S_x$（第1章的矩估计量）。

## 5.2 信息集比较

**引理 5.1（信息包含关系）** 令 $\mathcal{I}_{\text{CR}}$ 为 CR 使用的信息集（个体级捕获历史 $\{x_{jd}\}$），$\mathcal{I}_{\text{CL}}$ 为 CL 使用的信息集（聚合发展三角 $\{C_d = \sum_j I(\text{observed by } d)\}$）。则：

$$\mathcal{I}_{\text{CL}} \subset \mathcal{I}_{\text{CR}}$$

也就是说，聚合三角是个体捕获历史的**压缩映射** $C_d = \sum_j \max_{k \leq d} x_{jk}$。

**证明：** 从 $\{x_{jd}\}$ 到 $\{C_d\}$ 的映射是"按发展期求和"——将 $n_{\text{obs}} \times D$ 矩阵压缩为 $D$ 个聚合数。逆映射不存在：给定 $C_d$，无法唯一恢复各 $x_{jd}$。$\square$

## 5.3 效率比较定理

**定理 5.1（效率优势定理）** 在以下三个条件下，$\text{MSE}(\hat{N}_{\text{CR}}) < \text{MSE}(\hat{N}_{\text{CL}})$：

**(C1) 发展模式跨事故年不稳定：**

存在某个发展年 $d$ 和被比较的两个事故年，其 LDF 满足：

$$\text{Var}(f_d^{(a)}) > \sigma_0^2 \quad \text{且} \quad |f_d^{(a)} - f_d^{(a')}| > \delta_0$$

**(C2) 显著的报案/捕获异质性：**

$$\text{Var}(p_j) > \tau_0^2$$

即个体的捕获概率 $p_j$ 在总体中的方差超过阈值 $\tau_0^2 > 0$。

**(C3) 充分的观测窗口：**

$$m > m_0(N, D) = c \cdot N \log N$$

即捕获事件数超过某个与 $N$ 相关的阈值。

**证明：**

**第1步：CL 的渐近方差。** 在条件 (C1) 违反时（同质 LDF），CL 表现最优，$\text{MSE}(\hat{N}_{\text{CL}}) \approx O(N)$。在条件 (C1) 满足时，CL 的发展因子估计因跨期不稳定性产生额外方差。具体来说，LDF $\hat{f}_d$ 的估计方差为：

$$\text{Var}(\hat{f}_d) \approx \frac{\text{Var}(C_{d+1} \mid C_d)}{C_d^2}$$

当发展模式不稳定时，$\text{Var}(C_{d+1} \mid C_d)$ 增大。而 CL 的总体 IBNR 估计涉及 $\prod_{d} \hat{f}_d$，乘法传播放大了每个 $f_d$ 的误差：

$$\text{Var}(\hat{N}^{\text{IBNR}}_{\text{CL}}) \approx \text{Var}\left(n_{\text{obs}} \cdot \prod_{d=k}^{D-1} \hat{f}_d\right) \gtrsim O\left(\frac{N^2}{C_{\min}}\right)$$

其中 $C_{\min}$ 为最小聚合计数（通常位于发展三角的尾部）。

**第2步：CR 的渐近方差。** 由定理 1.2，CR 估计量的渐近方差为：

$$\text{Var}(\hat{N}_{\text{CR}}) = \frac{\tau^2}{\mu_x^2 \cdot m} + o(m^{-1})$$

其中 $\tau^2 = N^2 \cdot E[q(1 - q)]$（$q = Y/N$ 为标记比例）。注意到 $\tau^2$ 在 $p_j$ 异质时增大——但这是因为 $q$ 的异质增长——然而关键的区别在于缩放因子。

当 $m \gg N \log N$（条件 C3），$\text{Var}(\hat{N}_{\text{CR}}) = O(N^2 / m)$。由于 $m$ 随 $N$ 增长（每个观测期都有个体级事件），$m$ 通常远大于 CL 的"有效样本量"——聚合三角中只有 $O(D^2)$ 个独立数据点。

**第3步：比较。** 条件 (C2) 对 CL 的影响比对 CR 更大。在 CR 中，异质性通过 DPMM 被建模和估计，其不确定性被纳入后验分布。在 CL 中，个体异质性被加总过程吞噬——一个少量大额案件可能主导整个发展因子——导致系统性偏差。

正式的：当 (C1)-(C3) 成立时，存在常数 $c_1, c_2 > 0$ 使：

$$\text{MSE}(\hat{N}_{\text{CL}}) \geq c_1 \cdot \frac{N^2}{D^2} + c_2 \cdot \text{Var}(p_j) \cdot N^2$$

$$\text{MSE}(\hat{N}_{\text{CR}}) \leq \frac{\tilde{c}_1 \cdot N^2}{m} + \tilde{c}_2 \cdot \frac{\text{Var}(p_j) \cdot N^2}{D}$$

由于 $m \gg D^2$（个体事件远多于聚合三角单元数）且 $D \gg 1$，得 $\text{MSE}(\hat{N}_{\text{CR}}) < \text{MSE}(\hat{N}_{\text{CL}})$。$\square$

## 5.4 CL 与 CR 等价的条件

**定理 5.2（等价条件）** 当以下条件同时满足时，$\text{MSE}(\hat{N}_{\text{CR}}) = \text{MSE}(\hat{N}_{\text{CL}}) + o(1)$：

(i) **同质发展模式**：$f_d^{(a)} = f_d$（所有事故年的 LDF 相同）；
(ii) **同质捕获概率**：$p_j = p$（所有个体的检测概率相等）；
(iii) **大规模标记**：$Y_t \approx N$（标记比例接近 1）。

在此退化情形下，CL 的 LDF 估计和 CR 的捕获率估计从同一统计量中读取信息，两者渐近等价。

**注 5.1（实践意义）** 定理 5.2 表明，CR 的效率优势并非无条件成立——在高度同质的短尾险种（如标准车险）中，CL 依然具有竞争力。CR 的优势在**长尾、异质、发展模式不稳定**的险种中最为显著。

## 5.5 图示：MSE 分解对比

两种方法的 MSE 分解：

| 误差来源 | CL | CR (矩估计) | CR (贝叶斯 DPMM) |
|---------|-----|-----------|-----------------|
| 过程方差 | $O(N/D^2)$ | $O(N^2/m)$ | 后验宽度 |
| 异质性偏差 | **不可建模** | 部分吸收 | **完全建模** |
| 模型误设 | LDF 外推偏差 | $p$ 的同质性假设 | 贝叶斯模型平均减轻 |
| 小样本行为 | 依赖三角填充 | $m > N\log N$ 时良好 | 先验依赖 |
| 不确定性量化 | 仅渐近（Mack） | 渐近正态 | 完整后验分布 |

---

# 附录 A：符号表

| 符号 | 含义 |
|------|------|
| $N$ | 总体索赔数（含 IBNR） |
| $n_{\text{obs}}$ | 已观测到的索赔数 |
| $N^{\text{IBNR}}$ | IBNR 索赔数 $= N - n_{\text{obs}}$ |
| $D$ | 发展期数 |
| $d$ | 发展期索引 |
| $j$ | 索赔（个体）索引 |
| $Y_t$ | 第 $t$ 次事件前的累计标记数 |
| $x_t$ | 第 $t$ 次事件的再捕获指示变量 |
| $S_Y^{(m)} = \sum_{t=1}^{m} Y_{t-1}$ | 累计标记和 |
| $S_x^{(m)} = \sum_{t=1}^{m} x_t$ | 累计再捕获数 |
| $\hat{N}_m = S_Y^{(m)} / S_x^{(m)}$ | CR 矩估计量 |
| $\mathcal{S}$ | 索赔状态空间 $\{U, R, A, S, Re\}$ |
| $T$ | 状态转移矩阵 |
| $E$ | 发射概率矩阵 |
| $G \sim \text{DP}(\alpha, G_0)$ | Dirichlet Process 先验 |
| $\gamma_j$ | 个体异质性参数 |
| $\beta_{0d}$ | 第 $d$ 期的基线检测率 |
| $R_a$ | 事故年 $a$ 的准备金 |
| $\mu_a$ | 事故年 $a$ 的平均严重度 |
| $f_d$ | 第 $d$ 个 LDF（发展因子） |

---

*理论推导 Phase 1 完。Phase 2 将包含模拟实验的统计设计和实证分析的详细方案。*
