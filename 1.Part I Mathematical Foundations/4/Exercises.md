## 练习

### 练习 4.1

对于矩阵
$$
\boldsymbol A = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \\ 0 & 2 & 4 \end{bmatrix}
$$
分别使用拉普拉斯展开（沿第一行）和萨鲁斯法则（Sarrus' rule）计算其行列式。

---

### 练习 4.2

高效计算下列矩阵的行列式：
$$
\begin{bmatrix} 2 & 0 & 1 & 2 & 0 \\ 2 & -1 & 0 & 1 & 1 \\ 0 & 1 & 2 & 1 & 2 \\ -2 & 0 & 2 & -1 & 2 \\ 2 & 0 & 0 & 1 & 1 \end{bmatrix}
$$

---

### 练习 4.3

计算下列矩阵的特征空间：
$$
\begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad \begin{bmatrix} -2 & 2 \\ 2 & 1 \end{bmatrix}
$$

---

### 练习 4.4

计算矩阵
$$
\boldsymbol A = \begin{bmatrix} 0 & -1 & 1 & 1 \\ -1 & 1 & -2 & 3 \\ 2 & -1 & 0 & 0 \\ 1 & -1 & 1 & 0 \end{bmatrix}
$$
的所有特征空间。

---

### 练习 4.5

矩阵的可对角化性与其可逆性无关。判断下列四个矩阵是否可对角化和/或可逆：
$$
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}
$$

---

### 练习 4.6

计算下列变换矩阵的特征空间。它们可对角化吗？

a.
$$
\boldsymbol A = \begin{bmatrix} 2 & 3 & 0 \\ 1 & 4 & 3 \\ 0 & 0 & 1 \end{bmatrix}
$$

b.
$$
\boldsymbol A = \begin{bmatrix} 1 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}
$$

---

### 练习 4.7

下列矩阵是否可对角化？若是，求其对角形式以及变换矩阵关于其呈对角形式的基。若否，给出它们不可对角化的理由。

a.
$$
\boldsymbol A = \begin{bmatrix} 0 & 1 \\ -8 & 4 \end{bmatrix}
$$

b.
$$
\boldsymbol A = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}
$$

c.
$$
\boldsymbol A = \begin{bmatrix} 5 & 4 & 2 & 1 \\ 0 & 1 & -1 & -1 \\ -1 & -1 & 3 & 0 \\ 1 & 1 & -1 & 2 \end{bmatrix}
$$

d.
$$
\boldsymbol A = \begin{bmatrix} 5 & -6 & -6 \\ -1 & 4 & 2 \\ 3 & -6 & -4 \end{bmatrix}
$$

---

### 练习 4.8

求矩阵
$$
\boldsymbol A = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix}
$$
的奇异值分解（SVD）。

---

### 练习 4.9

求矩阵
$$
\boldsymbol A = \begin{bmatrix} 2 & 2 \\ -1 & 1 \end{bmatrix}
$$
的奇异值分解。

---

### 练习 4.10

求矩阵
$$
\boldsymbol A = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix}
$$
的最佳秩 1 近似。

---

### 练习 4.11

证明：对于任意 $ \boldsymbol A \in \mathbb{R}^{m \times n} $，矩阵 $ \boldsymbol A^\top \boldsymbol A $ 与 $ \boldsymbol A \boldsymbol A^\top $ 具有相同的非零特征值。

---

### 练习 4.12

证明对于 $ \boldsymbol x \neq \boldsymbol 0 $ 定理 4.24 成立，即证明：
$$
\max_{\boldsymbol x} \frac{\|\boldsymbol A \boldsymbol x\|_2}{\|\boldsymbol x\|_2} = \sigma_1
$$
其中 $ \sigma_1 $ 是 $ \boldsymbol A \in \mathbb{R}^{m \times n} $ 的最大奇异值。
