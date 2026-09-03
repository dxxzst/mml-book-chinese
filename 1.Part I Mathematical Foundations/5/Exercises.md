## 练习

### 练习 5.1

计算下列函数的导数 $ f'(x) $：
$$
f(x) = \log(x^4) \sin(x^3)
$$

---

### 练习 5.2

计算 Logistic Sigmoid 函数：
$$
f(x) = \frac{1}{1 + \exp(-x)}
$$
的导数 $ f'(x) $。

---

### 练习 5.3

计算函数：
$$
f(x) = \exp\left(-\frac{1}{2\sigma^2} (x - \mu)^2\right)
$$
的导数 $ f'(x) $，其中 $ \mu, \sigma \in \mathbb{R} $ 为常数。

---

### 练习 5.4

计算函数 $ f(x) = \sin(x) + \cos(x) $ 在 $ x_0 = 0 $ 处的泰勒多项式 $ T_n $（$ n = 0, \ldots, 5 $）。

---

### 练习 5.5

考虑下列函数：
$$
f_1(\boldsymbol x) = \sin(x_1) \cos(x_2), \quad \boldsymbol x \in \mathbb{R}^2
$$
$$
f_2(\boldsymbol x, \boldsymbol y) = \boldsymbol x^\top \boldsymbol y, \quad \boldsymbol x, \boldsymbol y \in \mathbb{R}^n
$$
$$
f_3(\boldsymbol x) = \boldsymbol x \boldsymbol x^\top, \quad \boldsymbol x \in \mathbb{R}^n
$$

a. $ \frac{\partial f_i}{\partial \boldsymbol x} $ 的维度分别是什么？  
b. 计算对应的雅可比矩阵（Jacobians）。

---

### 练习 5.6

求 $ f $ 关于 $ \boldsymbol t $ 的导数以及 $ g $ 关于 $ \boldsymbol X $ 的导数，其中：
$$
f(\boldsymbol t) = \sin(\log(\boldsymbol t^\top \boldsymbol t)), \quad \boldsymbol t \in \mathbb{R}^D
$$
$$
g(\boldsymbol X) = \operatorname{tr}(\boldsymbol A \boldsymbol X \boldsymbol B), \quad \boldsymbol A \in \mathbb{R}^{D \times E}, \; \boldsymbol X \in \mathbb{R}^{E \times F}, \; \boldsymbol B \in \mathbb{R}^{F \times D}
$$
其中 $ \operatorname{tr} $ 表示矩阵的迹。

---

### 练习 5.7

利用链式法则计算下列函数的导数 $ \frac{\mathrm{d}f}{\mathrm{d}\boldsymbol x} $。给出每个偏导数的维度，并详细描述你的推导步骤。

a.
$$
f(z) = \log(1 + z), \quad z = \boldsymbol x^\top \boldsymbol x, \quad \boldsymbol x \in \mathbb{R}^D
$$

b.
$$
f(\boldsymbol z) = \sin(\boldsymbol z), \quad \boldsymbol z = \boldsymbol A \boldsymbol x + \boldsymbol b, \quad \boldsymbol A \in \mathbb{R}^{E \times D}, \; \boldsymbol x \in \mathbb{R}^D, \; \boldsymbol b \in \mathbb{R}^E
$$
其中 $ \sin(\cdot) $ 逐元素作用于 $ \boldsymbol z $。

---

### 练习 5.8

计算下列函数的导数 $ \frac{\mathrm{d}f}{\mathrm{d}\boldsymbol x} $。详细描述你的推导步骤。

a. 使用链式法则。给出每个偏导数的维度。
$$
f(z) = \exp\left(-\frac{1}{2} z\right)
$$
$$
z = g(\boldsymbol y) = \boldsymbol y^\top \boldsymbol S^{-1} \boldsymbol y
$$
$$
\boldsymbol y = h(\boldsymbol x) = \boldsymbol x - \boldsymbol \mu
$$
其中 $ \boldsymbol x, \boldsymbol \mu \in \mathbb{R}^D $，$ \boldsymbol S \in \mathbb{R}^{D \times D} $。

b.
$$
f(\boldsymbol x) = \operatorname{tr}(\boldsymbol x \boldsymbol x^\top + \sigma^2 \boldsymbol I), \quad \boldsymbol x \in \mathbb{R}^D
$$
这里 $ \operatorname{tr}(\boldsymbol A) $ 是 $ \boldsymbol A $ 的迹，即对角线元素之和 $ \sum_i A_{ii} $。提示：显式写出外积。

c. 使用链式法则。给出每个偏导数的维度。不需要显式计算偏导数的乘积。
$$
\boldsymbol f = \tanh(\boldsymbol z) \in \mathbb{R}^M
$$
$$
\boldsymbol z = \boldsymbol A \boldsymbol x + \boldsymbol b, \quad \boldsymbol x \in \mathbb{R}^N, \; \boldsymbol A \in \mathbb{R}^{M \times N}, \; \boldsymbol b \in \mathbb{R}^M
$$
这里，$ \tanh $ 逐分量作用于 $ \boldsymbol z $。

---

### 练习 5.9

对于可微函数 $ p, q, t $，我们定义：
$$
g(z, \nu) := \log p(x, z) - \log q(z, \nu)
$$
$$
z := t(\epsilon, \nu)
$$
利用链式法则，计算梯度：
$$
\frac{\mathrm{d}}{\mathrm{d}\nu} g(z, \nu)
$$
