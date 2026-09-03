## 练习

### 练习 3.1

证明：对于所有 $ \boldsymbol x = [x_1, x_2]^\top \in \mathbb{R}^2 $ 和 $ \boldsymbol y = [y_1, y_2]^\top \in \mathbb{R}^2 $，由下式定义的 $ \langle\cdot, \cdot\rangle $：
$$
\langle\boldsymbol x, \boldsymbol y\rangle := x_1 y_1 - (x_1 y_2 + x_2 y_1) + 2(x_2 y_2)
$$
是一个内积。

---

### 练习 3.2

考虑 $ \mathbb{R}^2 $，其中对于 $ \mathbb{R}^2 $ 中的所有 $ \boldsymbol x $ 和 $ \boldsymbol y $，定义 $ \langle\cdot, \cdot\rangle $ 为：
$$
\langle\boldsymbol x, \boldsymbol y\rangle := \boldsymbol x^\top \underbrace{\begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}}_{=: \boldsymbol A} \boldsymbol y
$$
$ \langle\cdot, \cdot\rangle $ 是一个内积吗？

---

### 练习 3.3

计算
$$
\boldsymbol x = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad \boldsymbol y = \begin{bmatrix} -1 \\ -1 \\ 0 \end{bmatrix}
$$
之间的距离，分别使用：

a. $ \langle\boldsymbol x, \boldsymbol y\rangle := \boldsymbol x^\top \boldsymbol y $  
b. $ \langle\boldsymbol x, \boldsymbol y\rangle := \boldsymbol x^\top \boldsymbol A \boldsymbol y $，其中
$$
\boldsymbol A := \begin{bmatrix} 2 & 1 & 0 \\ 1 & 3 & -1 \\ 0 & -1 & 2 \end{bmatrix}
$$

---

### 练习 3.4

计算
$$
\boldsymbol x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad \boldsymbol y = \begin{bmatrix} -1 \\ -1 \end{bmatrix}
$$
之间的夹角，分别使用：

a. $ \langle\boldsymbol x, \boldsymbol y\rangle := \boldsymbol x^\top \boldsymbol y $  
b. $ \langle\boldsymbol x, \boldsymbol y\rangle := \boldsymbol x^\top \boldsymbol B \boldsymbol y $，其中
$$
\boldsymbol B := \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}
$$

---

### 练习 3.5

考虑带有点积的欧氏向量空间 $ \mathbb{R}^5 $。子空间 $ U \subseteq \mathbb{R}^5 $ 与向量 $ \boldsymbol x \in \mathbb{R}^5 $ 给定为：
$$
U = \text{span}\left[ \begin{bmatrix} 0 \\ -1 \\ 2 \\ 0 \\ 2 \end{bmatrix}, \begin{bmatrix} 1 \\ -3 \\ 1 \\ -1 \\ 2 \end{bmatrix}, \begin{bmatrix} -3 \\ 4 \\ 1 \\ 2 \\ 1 \end{bmatrix}, \begin{bmatrix} -1 \\ -3 \\ 5 \\ 0 \\ 7 \end{bmatrix} \right], \quad \boldsymbol x = \begin{bmatrix} -1 \\ -9 \\ -1 \\ 4 \\ 1 \end{bmatrix}
$$

a. 确定 $ \boldsymbol x $ 在 $ U $ 上的正交投影 $ \pi_U(\boldsymbol x) $。  
b. 确定距离 $ d(\boldsymbol x, U) $。

---

### 练习 3.6

考虑带有内积的 $ \mathbb{R}^3 $：
$$
\langle\boldsymbol x, \boldsymbol y\rangle := \boldsymbol x^\top \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix} \boldsymbol y
$$
此外，定义 $ \boldsymbol e_1, \boldsymbol e_2, \boldsymbol e_3 $ 为 $ \mathbb{R}^3 $ 中的标准基。

a. 确定 $ \boldsymbol e_2 $ 在
$$
U = \text{span}[\boldsymbol e_1, \boldsymbol e_3]
$$
上的正交投影 $ \pi_U(\boldsymbol e_2) $。  
提示：正交性是通过该内积定义的。  
b. 计算距离 $ d(\boldsymbol e_2, U) $。  
c. 画出该场景示意图：标准基向量以及 $ \pi_U(\boldsymbol e_2) $。

---

### 练习 3.7

设 $ V $ 为一个向量空间，$ \pi $ 为 $ V $ 上的一个自同态（endomorphism）。

a. 证明 $ \pi $ 是投影当且仅当 $ \text{id}_V - \pi $ 是投影，其中 $ \text{id}_V $ 是 $ V $ 上的恒等自同态。  
b. 现在假设 $ \pi $ 是一个投影。计算 $ \text{Im}(\text{id}_V - \pi) $ 和 $ \text{ker}(\text{id}_V - \pi) $ 作为 $ \text{Im}(\pi) $ 和 $ \text{ker}(\pi) $ 的函数（即用 $ \text{Im}(\pi) $ 和 $ \text{ker}(\pi) $ 来表示它们）。

---

### 练习 3.8

使用格拉姆-施密特方法（Gram-Schmidt method），将二维子空间 $ U \subseteq \mathbb{R}^3 $ 的一组基 $ B = (\boldsymbol b_1, \boldsymbol b_2) $ 化为 $ U $ 的一组标准正交基（ONB）$ C = (\boldsymbol c_1, \boldsymbol c_2) $，其中：
$$
\boldsymbol b_1 := \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}, \quad \boldsymbol b_2 := \begin{bmatrix} -1 \\ 2 \\ 0 \end{bmatrix}
$$

---

### 练习 3.9

设 $ n \in \mathbb{N}^* $，并设 $ x_1, \ldots, x_n > 0 $ 为 $ n $ 个正实数，满足 $ x_1 + \cdots + x_n = 1 $。利用柯西-施瓦茨不等式证明：

a.
$$
\sum_{i=1}^n x_i^2 \geqslant \frac{1}{n}
$$

b.
$$
\sum_{i=1}^n \frac{1}{x_i} \geqslant n^2
$$

提示：考虑 $ \mathbb{R}^n $ 上的点积。然后选择特定的向量 $ \boldsymbol x, \boldsymbol y \in \mathbb{R}^n $ 并应用柯西-施瓦茨不等式。

---

### 练习 3.10

将向量
$$
\boldsymbol x_1 := \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad \boldsymbol x_2 := \begin{bmatrix} 0 \\ -1 \end{bmatrix}
$$
旋转 $ 30^\circ $。
