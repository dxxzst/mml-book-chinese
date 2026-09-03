## 练习

### 练习 6.1

考虑两个离散随机变量 $ X $ 和 $ Y $ 的如下二元分布 $ p(x, y) $：

| $ Y \backslash X $ | $ x_1 $ | $ x_2 $ | $ x_3 $ | $ x_4 $ | $ x_5 $ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $ y_3 $ | 0.01 | 0.02 | 0.03 | 0.1 | 0.1 |
| $ y_2 $ | 0.05 | 0.1 | 0.05 | 0.07 | 0.2 |
| $ y_1 $ | 0.1 | 0.05 | 0.03 | 0.05 | 0.04 |

计算：  
a. 边缘分布 $ p(x) $ 和 $ p(y) $。  
b. 条件分布 $ p(x \mid Y = y_1) $ 和 $ p(y \mid X = x_3) $。

---

### 练习 6.2

考虑两个高斯分布的混合（如图 6.4 所示）：
$$
0.4 \mathcal{N}\left( \begin{bmatrix} 10 \\ 2 \end{bmatrix}, \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \right) + 0.6 \mathcal{N}\left( \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 8.4 & 2.0 \\ 2.0 & 1.7 \end{bmatrix} \right)
$$

a. 计算每个维度的边缘分布。  
b. 计算每个边缘分布的均值、众数和中位数。  
c. 计算该二维分布的均值和众数。

---

### 练习 6.3

你编写了一个计算机程序，该程序有时能够编译通过，有时则不能（代码未发生改变）。你决定使用参数为 $ \mu $ 的伯努利分布对编译器的表面随机性（成功与否）$ x $ 进行建模：
$$
p(x \mid \mu) = \mu^x (1 - \mu)^{1-x}, \quad x \in \{0, 1\}
$$
为该伯努利似然选择一个共轭先验，并计算后验分布 $ p(\mu \mid x_1, \ldots, x_N) $。

---

### 练习 6.4

有两个袋子。第一个袋子装有 4 个芒果和 2 个苹果；第二个袋子装有 4 个芒果和 4 个苹果。

我们还有一枚有偏差的硬币，出现“正面”的概率为 0.6，出现“反面”的概率为 0.4。如果硬币显示“正面”，我们从袋子 1 中随机取出一个水果；否则我们从袋子 2 中随机取出一个水果。

你的朋友抛了这枚硬币（你看不到结果），从对应的袋子中随机取出一个水果，并递给你一个芒果。

这个芒果是从袋子 2 中取出的概率是多少？  
提示：使用贝叶斯定理。

---

### 练习 6.5

考虑时间序列模型：
$$
\boldsymbol x_{t+1} = \boldsymbol A \boldsymbol x_t + \boldsymbol w, \quad \boldsymbol w \sim \mathcal{N}(\boldsymbol 0, \boldsymbol Q)
$$
$$
\boldsymbol y_t = \boldsymbol C \boldsymbol x_t + \boldsymbol v, \quad \boldsymbol v \sim \mathcal{N}(\boldsymbol 0, \boldsymbol R)
$$
其中 $ \boldsymbol w, \boldsymbol v $ 是独立同分布（i.i.d.）的高斯噪声变量。此外，假设 $ p(\boldsymbol x_0) = \mathcal{N}(\boldsymbol \mu_0, \boldsymbol \Sigma_0) $。

a. $ p(\boldsymbol x_0, \boldsymbol x_1, \ldots, \boldsymbol x_T) $ 的形式是什么？说明你的理由（无需显式计算联合分布）。  
b. 假设 $ p(\boldsymbol x_t \mid \boldsymbol y_1, \ldots, \boldsymbol y_t) = \mathcal{N}(\boldsymbol \mu_t, \boldsymbol \Sigma_t) $。
1. 计算 $ p(\boldsymbol x_{t+1} \mid \boldsymbol y_1, \ldots, \boldsymbol y_t) $。
2. 计算 $ p(\boldsymbol x_{t+1}, \boldsymbol y_{t+1} \mid \boldsymbol y_1, \ldots, \boldsymbol y_t) $。
3. 在时刻 $ t + 1 $，我们观测到数值 $ \boldsymbol y_{t+1} = \hat{\boldsymbol y} $。计算条件分布 $ p(\boldsymbol x_{t+1} \mid \boldsymbol y_1, \ldots, \boldsymbol y_{t+1}) $。

---

### 练习 6.6

证明式 (6.44) 中的关系，该关系将方差的标准定义与方差的原始分数表达式联系起来。

---

### 练习 6.7

证明式 (6.45) 中的关系，该关系将数据集中样本之间的两两差异与方差的原始分数表达式联系起来。

---

### 练习 6.8

将伯努利分布表示为指数族的自然参数形式，参见式 (6.107)。

---

### 练习 6.9

将二项分布表示为指数族分布。同时将 Beta 分布表示为指数族分布。证明 Beta 分布与二项分布的乘积也是指数族的成员。

---

### 练习 6.10

通过两种方法推导第 6.5.2 节中的关系：  
a. 通过配方法  
b. 通过将高斯分布表示为其指数族形式

两个高斯分布 $ \mathcal{N}(\boldsymbol x \mid \boldsymbol a, \boldsymbol A)\mathcal{N}(\boldsymbol x \mid \boldsymbol b, \boldsymbol B) $ 的乘积是一个未归一化的高斯分布 $ c \mathcal{N}(\boldsymbol x \mid \boldsymbol c, \boldsymbol C) $，其中：
$$
\boldsymbol C = (\boldsymbol A^{-1} + \boldsymbol B^{-1})^{-1}
$$
$$
\boldsymbol c = \boldsymbol C(\boldsymbol A^{-1}\boldsymbol a + \boldsymbol B^{-1}\boldsymbol b)
$$
$$
c = (2\pi)^{-\frac{D}{2}} |\boldsymbol A + \boldsymbol B|^{-\frac{1}{2}} \exp\left( -\frac{1}{2}(\boldsymbol a - \boldsymbol b)^\top (\boldsymbol A + \boldsymbol B)^{-1}(\boldsymbol a - \boldsymbol b) \right)
$$
注意，归一化常数 $ c $ 本身可以被视为关于 $ \boldsymbol a $ 或关于 $ \boldsymbol b $ 的具有“膨胀”协方差矩阵 $ \boldsymbol A + \boldsymbol B $ 的（归一化）高斯分布，即 $ c = \mathcal{N}(\boldsymbol a \mid \boldsymbol b, \boldsymbol A + \boldsymbol B) = \mathcal{N}(\boldsymbol b \mid \boldsymbol a, \boldsymbol A + \boldsymbol B) $。

---

### 练习 6.11

**迭代期望**。  
考虑两个随机变量 $ x, y $，其联合分布为 $ p(x, y) $。证明：
$$
\mathbb{E}_X[x] = \mathbb{E}_Y \left[ \mathbb{E}_X[x \mid y] \right]
$$
这里，$ \mathbb{E}_X[x \mid y] $ 表示在条件分布 $ p(x \mid y) $ 下 $ x $ 的期望值。

---

### 练习 6.12

**高斯随机变量的操作**。  
考虑高斯随机变量 $ \boldsymbol x \sim \mathcal{N}(\boldsymbol x \mid \boldsymbol \mu_x, \boldsymbol \Sigma_x) $，其中 $ \boldsymbol x \in \mathbb{R}^D $。此外，我们有：
$$
\boldsymbol y = \boldsymbol A \boldsymbol x + \boldsymbol b + \boldsymbol w
$$
其中 $ \boldsymbol y \in \mathbb{R}^E $，$ \boldsymbol A \in \mathbb{R}^{E \times D} $，$ \boldsymbol b \in \mathbb{R}^E $，且 $ \boldsymbol w \sim \mathcal{N}(\boldsymbol w \mid \boldsymbol 0, \boldsymbol Q) $ 是独立高斯噪声。“独立”意味着 $ \boldsymbol x $ 和 $ \boldsymbol w $ 是独立的随机变量，且 $ \boldsymbol Q $ 为对角矩阵。

a. 写出似然 $ p(\boldsymbol y \mid \boldsymbol x) $。  
b. 分布 $ p(\boldsymbol y) = \int p(\boldsymbol y \mid \boldsymbol x) p(\boldsymbol x)\,\mathrm{d}\boldsymbol x $ 是高斯分布。计算均值 $ \boldsymbol \mu_y $ 和协方差 $ \boldsymbol \Sigma_y $。详细推导你的结果。  
c. 随机变量 $ \boldsymbol y $ 根据测量映射进行变换：
$$
\boldsymbol z = \boldsymbol C \boldsymbol y + \boldsymbol v
$$
其中 $ \boldsymbol z \in \mathbb{R}^F $，$ \boldsymbol C \in \mathbb{R}^{F \times E} $，且 $ \boldsymbol v \sim \mathcal{N}(\boldsymbol v \mid \boldsymbol 0, \boldsymbol R) $ 是独立高斯（测量）噪声。  
写出 $ p(\boldsymbol z \mid \boldsymbol y) $。  
计算 $ p(\boldsymbol z) $，即均值 $ \boldsymbol \mu_z $ 和协方差 $ \boldsymbol \Sigma_z $。详细推导你的结果。  
d. 现在测量到了一个数值 $ \hat{\boldsymbol y} $。计算后验分布 $ p(\boldsymbol x \mid \hat{\boldsymbol y}) $。  
求解提示：该后验也是高斯分布，即我们只需确定其均值和协方差矩阵。首先显式计算联合高斯分布 $ p(\boldsymbol x, \boldsymbol y) $。这也需要我们计算互协方差 $ \operatorname{Cov}_{\boldsymbol x, \boldsymbol y}[\boldsymbol x, \boldsymbol y] $ 和 $ \operatorname{Cov}_{\boldsymbol y, \boldsymbol x}[\boldsymbol y, \boldsymbol x] $。然后应用高斯条件分布法则。

---

### 练习 6.13

**概率积分变换**。  
已知连续随机变量 $ x $ 具有累积分布函数 $ F_X(x) $，证明随机变量 $ y = F_X(x) $ 服从均匀分布。
