## 练习

### 练习 7.1

考虑单变量函数：
$$
f(x) = x^3 + 6x^2 - 3x - 5
$$
求其驻点，并指出它们是极大值点、极小值点还是鞍点。

---

### 练习 7.2

考虑随机梯度下降的更新方程（式 (7.15)）。写出当我们使用大小为 1 的小批量（mini-batch）时的更新公式。

---

### 练习 7.3

判断下列陈述的真伪：  
a. 任意两个凸集的交集是凸集。  
b. 任意两个凸集的并集是凸集。  
c. 一个凸集 $ A $ 与另一个凸集 $ B $ 的差集是凸集。

---

### 练习 7.4

判断下列陈述的真伪：  
a. 任意两个凸函数之和是凸函数。  
b. 任意两个凸函数之差是凸函数。  
c. 任意两个凸函数之积是凸函数。  
d. 任意两个凸函数的逐点最大值是凸函数。

---

### 练习 7.5

将以下优化问题用矩阵记号表示为标准线性规划：
$$
\max_{\boldsymbol x \in \mathbb{R}^2,\, \xi \in \mathbb{R}} \boldsymbol p^\top \boldsymbol x + \xi
$$
满足约束条件 $ \xi \geqslant 0 $、$ x_0 \leqslant 0 $ 且 $ x_1 \leqslant 3 $。

---

### 练习 7.6

考虑图 7.9 中所示的线性规划：
$$
\begin{aligned}
\min_{\boldsymbol x \in \mathbb{R}^2} \quad & -\begin{bmatrix} 5 \\ 3 \end{bmatrix}^\top \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \\
\text{subject to} \quad & \begin{bmatrix} 2 & 2 \\ 2 & -4 \\ -2 & 1 \\ 0 & -1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \leqslant \begin{bmatrix} 33 \\ 8 \\ 5 \\ -1 \\ 8 \end{bmatrix}
\end{aligned}
$$
利用拉格朗日对偶性推导其对偶线性规划。

---

### 练习 7.7

考虑图 7.4 中所示的二次规划：
$$
\begin{aligned}
\min_{\boldsymbol x \in \mathbb{R}^2} \quad & \frac{1}{2}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}^\top \begin{bmatrix} 2 & 1 \\ 1 & 4 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} 5 \\ 3 \end{bmatrix}^\top \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \\
\text{subject to} \quad & \begin{bmatrix} 1 & 0 \\ -1 & 0 \\ 0 & 1 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \leqslant \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \end{bmatrix}
\end{aligned}
$$
利用拉格朗日对偶性推导其对偶二次规划。

---

### 练习 7.8

考虑以下凸优化问题：
$$
\begin{aligned}
\min_{\boldsymbol w \in \mathbb{R}^D} \quad & \frac{1}{2}\boldsymbol w^\top \boldsymbol w \\
\text{subject to} \quad & \boldsymbol w^\top \boldsymbol x \geqslant 1
\end{aligned}
$$
通过引入拉格朗日乘子 $ \lambda $，推导其拉格朗日对偶问题。

---

### 练习 7.9

考虑 $ \boldsymbol x \in \mathbb{R}^D $ 的负熵函数：
$$
f(\boldsymbol x) = \sum_{d=1}^D x_d \log x_d
$$
假设使用标准点积，推导其凸共轭函数 $ f^*(\boldsymbol s) $。  
提示：对适当的函数求梯度并令梯度为零。

---

### 练习 7.10

考虑函数：
$$
f(\boldsymbol x) = \frac{1}{2}\boldsymbol x^\top \boldsymbol A \boldsymbol x + \boldsymbol b^\top \boldsymbol x + c
$$
其中 $ \boldsymbol A $ 严格正定，这意味着它是可逆的。推导 $ f(\boldsymbol x) $ 的凸共轭函数。  
提示：对适当的函数求梯度并令梯度为零。

---

### 练习 7.11

合页损失（hinge loss，即支持向量机所使用的损失函数）由下式给出：
$$
L(\alpha) = \max\{0, 1 - \alpha\}
$$
如果我们希望应用诸如 L-BFGS 之类的梯度方法，而不希望诉诸次梯度法，我们需要平滑合页损失中的尖折点。计算合页损失的凸共轭 $ L^*(\beta) $，其中 $ \beta $ 为对偶变量。添加一个 $ \ell_2 $ 近端项，并计算所得函数的共轭：
$$
L^*(\beta) + \frac{\gamma}{2}\beta^2
$$
其中 $ \gamma $ 为给定的超参数。
