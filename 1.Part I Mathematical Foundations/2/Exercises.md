## 练习

### 练习 2.1

考虑集合与二元运算构成的代数结构 $ (\mathbb{R} \setminus \{-1\}, \star) $，其中：

$$
a \star b := ab + a + b, \quad a, b \in \mathbb{R} \setminus \{-1\} \tag{2.134}
$$

a. 证明 $ (\mathbb{R} \setminus \{-1\}, \star) $ 是一个阿贝尔群（Abelian group）。  
b. 在阿贝尔群 $ (\mathbb{R} \setminus \{-1\}, \star) $ 中求解方程：

$$
3 \star x \star x = 15
$$

其中 $ \star $ 的定义见式 (2.134)。

---

### 练习 2.2

设 $ n \in \mathbb{N} \setminus \{0\} $。设 $ k, x \in \mathbb{Z} $。我们将整数 $ k $ 的同余类（congruence class）$ \bar{k} $ 定义为集合：

$$
\begin{aligned}
\bar{k} &= \{x \in \mathbb{Z} \mid x - k = 0 \pmod n\} \\
&= \{x \in \mathbb{Z} \mid (\exists a \in \mathbb{Z}): (x - k = n \cdot a)\}
\end{aligned}
$$

现在我们将所有模 $ n $ 的同余类构成的集合定义为 $ \mathbb{Z}/n\mathbb{Z} $（有时也记作 $ \mathbb{Z}_n $）。欧几里得除法（带余除法）表明该集合是一个包含 $ n $ 个元素的有限集：

$$
\mathbb{Z}_n = \{\bar{0}, \bar{1}, \ldots, \overline{n - 1}\}
$$

对于所有的 $ \bar{a}, \bar{b} \in \mathbb{Z}_n $，我们定义：

$$
\bar{a} \oplus \bar{b} := \overline{a + b}
$$

a. 证明 $ (\mathbb{Z}_n, \oplus) $ 是一个群。它是否是阿贝尔群？  
b. 现在我们为 $ \mathbb{Z}_n $ 中的所有 $ \bar{a} $ 和 $ \bar{b} $ 定义另一个运算 $ \otimes $：

$$
\bar{a} \otimes \bar{b} = \overline{a \times b} \tag{2.135}
$$

其中 $ a \times b $ 表示 $ \mathbb{Z} $ 中的常规乘法。  
令 $ n = 5 $。画出 $ \mathbb{Z}_5 \setminus \{\bar{0}\} $ 中的元素在 $ \otimes $ 运算下的乘法表（times table），即计算 $ \mathbb{Z}_5 \setminus \{\bar{0}\} $ 中所有 $ \bar{a} $ 和 $ \bar{b} $ 的乘积 $ \bar{a} \otimes \bar{b} $。  
由此证明 $ \mathbb{Z}_5 \setminus \{\bar{0}\} $ 在 $ \otimes $ 运算下是封闭的，并且存在关于 $ \otimes $ 的单位元。列出 $ \mathbb{Z}_5 \setminus \{\bar{0}\} $ 中所有元素在 $ \otimes $ 运算下的逆元。由此得出结论：$ (\mathbb{Z}_5 \setminus \{\bar{0}\}, \otimes) $ 是一个阿贝尔群。  
c. 证明 $ (\mathbb{Z}_8 \setminus \{\bar{0}\}, \otimes) $ 不是一个群。  
d. 回顾裴蜀定理（Bézout's theorem）：两个整数 $ a $ 和 $ b $ 互素（即 $ \gcd(a, b) = 1 $），当且仅当存在两个整数 $ u $ 和 $ v $ 使得 $ au + bv = 1 $。证明 $ (\mathbb{Z}_n \setminus \{\bar{0}\}, \otimes) $ 是一个群当且仅当 $ n \in \mathbb{N} \setminus \{0\} $ 是素数。

---

### 练习 2.3

考虑如下定义的 $ 3 \times 3 $ 矩阵集合 $ G $：

$$
G = \left\{ \begin{bmatrix} 1 & x & z \\ 0 & 1 & y \\ 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 3} \;\middle|\; x, y, z \in \mathbb{R} \right\} \tag{2.136}
$$

我们将 $ \cdot $ 定义为常规的矩阵乘法。  
$ (G, \cdot) $ 是否构成一个群？如果是，它是否是阿贝尔群？请说明理由。

---

### 练习 2.4

若可能，计算下列矩阵的乘积：

a.

$$
\begin{bmatrix} 1 & 2 \\ 4 & 5 \\ 7 & 8 \end{bmatrix} \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix}
$$

b.

$$
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix} \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix}
$$

c.

$$
\begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}
$$

d.

$$
\begin{bmatrix} 1 & 2 & 1 & 2 \\ 4 & 1 & -1 & -4 \end{bmatrix} \begin{bmatrix} 0 & 3 \\ 1 & -1 \\ 2 & 1 \\ 5 & 2 \end{bmatrix}
$$

e.

$$
\begin{bmatrix} 0 & 3 \\ 1 & -1 \\ 2 & 1 \\ 5 & 2 \end{bmatrix} \begin{bmatrix} 1 & 2 & 1 & 2 \\ 4 & 1 & -1 & -4 \end{bmatrix}
$$

---

### 练习 2.5

求下列非齐次线性方程组 $ \boldsymbol A \boldsymbol x = \boldsymbol b $ 关于 $ \boldsymbol x $ 的所有解构成的集合 $ S $，其中 $ \boldsymbol A $ 和 $ \boldsymbol b $ 的定义分别如下：

a.

$$
\boldsymbol A = \begin{bmatrix} 1 & 1 & -1 & -1 \\ 2 & 5 & -7 & -5 \\ 2 & -1 & 1 & 3 \\ 5 & 2 & -4 & 2 \end{bmatrix}, \quad \boldsymbol b = \begin{bmatrix} 1 \\ -2 \\ 4 \\ 6 \end{bmatrix}
$$

b.

$$
\boldsymbol A = \begin{bmatrix} 1 & -1 & 0 & 0 & 1 \\ 1 & 1 & 0 & -3 & 0 \\ 2 & -1 & 0 & 1 & -1 \\ -1 & 2 & 0 & -2 & -1 \end{bmatrix}, \quad \boldsymbol b = \begin{bmatrix} 3 \\ 6 \\ 5 \\ -1 \end{bmatrix}
$$

---

### 练习 2.6

使用高斯消元法，求解下列非齐次线性方程组 $ \boldsymbol A \boldsymbol x = \boldsymbol b $ 的所有解，其中：

$$
\boldsymbol A = \begin{bmatrix} 0 & 1 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 0 & 0 & 1 \end{bmatrix}, \quad \boldsymbol b = \begin{bmatrix} 2 \\ -1 \\ 1 \end{bmatrix}
$$

---

### 练习 2.7

求方程组 $ \boldsymbol A \boldsymbol x = 12 \boldsymbol x $ 在满足 $ \sum_{i=1}^3 x_i = 1 $ 的条件下，所有解 $ \boldsymbol x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \in \mathbb{R}^3 $，其中：

$$
\boldsymbol A = \begin{bmatrix} 6 & 4 & 3 \\ 6 & 0 & 9 \\ 0 & 8 & 0 \end{bmatrix}
$$

---

### 练习 2.8

若可能，求下列矩阵的逆矩阵：

a.

$$
\boldsymbol A = \begin{bmatrix} 2 & 3 & 4 \\ 3 & 4 & 5 \\ 4 & 5 & 6 \end{bmatrix}
$$

b.

$$
\boldsymbol A = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 1 & 1 & 0 \end{bmatrix}
$$

---

### 练习 2.9

下列哪些集合是 $ \mathbb{R}^3 $ 的子空间？

a. $ A = \{(\lambda, \lambda + \mu^3, \lambda - \mu^3) \mid \lambda, \mu \in \mathbb{R}\} $  
b. $ B = \{(\lambda^2, -\lambda^2, 0) \mid \lambda \in \mathbb{R}\} $  
c. 设 $ \gamma \in \mathbb{R} $，

$$
C = \{(\xi_1, \xi_2, \xi_3) \in \mathbb{R}^3 \mid \xi_1 - 2\xi_2 + 3\xi_3 = \gamma\}
$$

d. $ D = \{(\xi_1, \xi_2, \xi_3) \in \mathbb{R}^3 \mid \xi_2 \in \mathbb{Z}\} $

---

### 练习 2.10

下列向量集合是否线性无关？

a.

$$
\boldsymbol x_1 = \begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix}, \quad \boldsymbol x_2 = \begin{bmatrix} 1 \\ 1 \\ -2 \end{bmatrix}, \quad \boldsymbol x_3 = \begin{bmatrix} 3 \\ -3 \\ 8 \end{bmatrix}
$$

b.

$$
\boldsymbol x_1 = \begin{bmatrix} 1 \\ 2 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \quad \boldsymbol x_2 = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 1 \\ 1 \end{bmatrix}, \quad \boldsymbol x_3 = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 1 \\ 1 \end{bmatrix}
$$

---

### 练习 2.11

将向量

$$
\boldsymbol y = \begin{bmatrix} 1 \\ -2 \\ 5 \end{bmatrix}
$$

表示为下列向量的线性组合：

$$
\boldsymbol x_1 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}, \quad \boldsymbol x_2 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad \boldsymbol x_3 = \begin{bmatrix} 2 \\ -1 \\ 1 \end{bmatrix}
$$

---

### 练习 2.12

考虑 $ \mathbb{R}^4 $ 的两个子空间：

$$
U_1 = \operatorname{span}\left[ \begin{bmatrix} 1 \\ 1 \\ -3 \\ 1 \end{bmatrix}, \begin{bmatrix} 2 \\ -1 \\ 0 \\ -1 \end{bmatrix}, \begin{bmatrix} -1 \\ 1 \\ -1 \\ 1 \end{bmatrix} \right], \quad U_2 = \operatorname{span}\left[ \begin{bmatrix} -1 \\ -2 \\ 2 \\ 1 \end{bmatrix}, \begin{bmatrix} 2 \\ -2 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} -3 \\ 6 \\ -2 \\ -1 \end{bmatrix} \right]
$$

求 $ U_1 \cap U_2 $ 的一组基。

---

### 练习 2.13

考虑两个子空间 $ U_1 $ 和 $ U_2 $，其中 $ U_1 $ 是齐次线性方程组 $ \boldsymbol A_1 \boldsymbol x = \boldsymbol 0 $ 的解空间，$ U_2 $ 是齐次线性方程组 $ \boldsymbol A_2 \boldsymbol x = \boldsymbol 0 $ 的解空间，且：

$$
\boldsymbol A_1 = \begin{bmatrix} 1 & 0 & 1 & 1 \\ -2 & -1 & 2 & 1 \\ 3 & 1 & 0 & 1 \end{bmatrix}, \quad \boldsymbol A_2 = \begin{bmatrix} 3 & -3 & 0 & 1 \\ 2 & 3 & 7 & -5 \\ 2 & 3 & -1 & 2 \end{bmatrix}
$$

a. 确定 $ U_1, U_2 $ 的维度。  
b. 确定 $ U_1 $ 和 $ U_2 $ 的基。  
c. 确定 $ U_1 \cap U_2 $ 的一组基。

---

### 练习 2.14

考虑两个子空间 $ U_1 $ 和 $ U_2 $，其中 $ U_1 $ 由 $ \boldsymbol A_1 $ 的列向量张成，$ U_2 $ 由 $ \boldsymbol A_2 $ 的列向量张成，且：

$$
\boldsymbol A_1 = \begin{bmatrix} 1 & 0 & 1 & 1 \\ -2 & -1 & 2 & 1 \\ 3 & 1 & 0 & 1 \end{bmatrix}, \quad \boldsymbol A_2 = \begin{bmatrix} 3 & -3 & 0 & 1 \\ 2 & 3 & 7 & -5 \\ 2 & 3 & -1 & 2 \end{bmatrix}
$$

a. 确定 $ U_1, U_2 $ 的维度。  
b. 确定 $ U_1 $ 和 $ U_2 $ 的基。  
c. 确定 $ U_1 \cap U_2 $ 的一组基。

---

### 练习 2.15

设 $ F = \{(x, y, z) \in \mathbb{R}^3 \mid x + y - z = 0\} $ 且 $ G = \{(a - b, a + b, a - 3b) \mid a, b \in \mathbb{R}\} $。

a. 证明 $ F $ 和 $ G $ 是 $ \mathbb{R}^3 $ 的子空间。  
b. 在不借助任何基向量的前提下，计算 $ F \cap G $。  
c. 分别找出 $ F $ 和 $ G $ 的一组基，利用前面求得的基向量计算 $ F \cap G $，并将结果与上一问进行比对验证。

---

### 练习 2.16

下列映射是否为线性映射？

a. 设 $ a, b \in \mathbb{R} $。

$$
\begin{aligned}
\Phi : L^1([a, b]) &\to \mathbb{R} \\
f &\mapsto \Phi(f) = \int_a^b f(x)\,\mathrm{d}x
\end{aligned}
$$

其中 $ L^1([a, b]) $ 表示 $ [a, b] $ 上的可积函数集合。

b.

$$
\begin{aligned}
\Phi : C^1 &\to C^0 \\
f &\mapsto \Phi(f) = f'
\end{aligned}
$$

其中对 $ k \geqslant 1 $，$ C^k $ 表示 $ k $ 阶连续可微函数集合，而 $ C^0 $ 表示连续函数集合。

c.

$$
\begin{aligned}
\Phi : \mathbb{R} &\to \mathbb{R} \\
x &\mapsto \Phi(x) = \cos(x)
\end{aligned}
$$

d.

$$
\begin{aligned}
\Phi : \mathbb{R}^3 &\to \mathbb{R}^2 \\
\boldsymbol x &\mapsto \begin{bmatrix} 1 & 2 & 3 \\ 1 & 4 & 3 \end{bmatrix} \boldsymbol x
\end{aligned}
$$

e. 设 $ \theta \in [0, 2\pi) $。

$$
\begin{aligned}
\Phi : \mathbb{R}^2 &\to \mathbb{R}^2 \\
\boldsymbol x &\mapsto \begin{bmatrix} \cos(\theta) & \sin(\theta) \\ -\sin(\theta) & \cos(\theta) \end{bmatrix} \boldsymbol x
\end{aligned}
$$

---

### 练习 2.17

考虑线性映射：

$$
\begin{aligned}
\Phi : \mathbb{R}^3 &\to \mathbb{R}^4 \\
\Phi\left(\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}\right) &= \begin{bmatrix} 3x_1 + 2x_2 + x_3 \\ x_1 + x_2 + x_3 \\ x_1 - 3x_2 \\ 2x_1 + 3x_2 + x_3 \end{bmatrix}
\end{aligned}
$$

求变换矩阵 $ \boldsymbol A_{\Phi} $。  
确定 $ \operatorname{rk}(\boldsymbol A_{\Phi}) $。  
计算 $ \Phi $ 的核（kernel，零空间）与像（image）。$ \operatorname{dim}(\operatorname{ker}(\Phi)) $ 和 $ \operatorname{dim}(\operatorname{Im}(\Phi)) $ 分别是多少？

---

### 练习 2.18

设 $ E $ 为一个向量空间。设 $ f $ 和 $ g $ 为 $ E $ 上的两个自同态（automorphisms，注：亦作自同构），满足 $ f \circ g = \operatorname{id}_E $（即 $ f \circ g $ 为恒等映射 $ \operatorname{id}_E $）。证明：

$$
\operatorname{ker}(f) = \operatorname{ker}(g \circ f), \quad \operatorname{Im}(g) = \operatorname{Im}(g \circ f), \quad \operatorname{ker}(f) \cap \operatorname{Im}(g) = \{\boldsymbol 0_E\}
$$

---

### 练习 2.19

考虑一个自同态 $ \Phi : \mathbb{R}^3 \to \mathbb{R}^3 $，其关于 $ \mathbb{R}^3 $ 标准基的变换矩阵为：

$$
\boldsymbol A_{\Phi} = \begin{bmatrix} 1 & 1 & 0 \\ 1 & -1 & 0 \\ 1 & 1 & 1 \end{bmatrix}
$$

1. 确定 $ \operatorname{ker}(\Phi) $ 和 $ \operatorname{Im}(\Phi) $。  
2. 确定 $ \Phi $ 关于基 $ B $ 的变换矩阵 $ \tilde{\boldsymbol A}_{\Phi} $，其中：

$$
B = \left( \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}, \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}, \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} \right)
$$

即执行向新基 $ B $ 的基变换。

---

### 练习 2.20

考虑在 $ \mathbb{R}^2 $ 标准基下表示的 4 个向量：

$$
\boldsymbol b_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \quad \boldsymbol b_2 = \begin{bmatrix} -1 \\ -1 \end{bmatrix}, \quad \boldsymbol b'_1 = \begin{bmatrix} 2 \\ -2 \end{bmatrix}, \quad \boldsymbol b'_2 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

定义 $ \mathbb{R}^2 $ 的两个有序基 $ B = (\boldsymbol b_1, \boldsymbol b_2) $ 和 $ B' = (\boldsymbol b'_1, \boldsymbol b'_2) $。

1. 证明 $ B $ 和 $ B' $ 是 $ \mathbb{R}^2 $ 的两组基，并画出这些基向量。  
2. 计算实现从 $ B' $ 到 $ B $ 基变换的矩阵 $ \boldsymbol P_1 $。  
3. 考虑在 $ \mathbb{R}^3 $ 标准基下定义的 3 个向量：

$$
\boldsymbol c_1 = \begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix}, \quad \boldsymbol c_2 = \begin{bmatrix} 0 \\ -1 \\ 2 \end{bmatrix}, \quad \boldsymbol c_3 = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}
$$

定义 $ C = (\boldsymbol c_1, \boldsymbol c_2, \boldsymbol c_3) $。

a. 证明 $ C $ 是 $ \mathbb{R}^3 $ 的一组基，例如通过计算行列式（参见第 4.1 节）。  
b. 记 $ C' = (\boldsymbol c'_1, \boldsymbol c'_2, \boldsymbol c'_3) $ 为 $ \mathbb{R}^3 $ 的标准基。求实现从 $ C $ 到 $ C' $ 基变换的矩阵 $ \boldsymbol P_2 $。

4. 考虑同态 $ \Phi : \mathbb{R}^2 \to \mathbb{R}^3 $，满足：

$$
\begin{aligned}
\Phi(\boldsymbol b_1 + \boldsymbol b_2) &= \boldsymbol c_2 + \boldsymbol c_3 \\
\Phi(\boldsymbol b_1 - \boldsymbol b_2) &= 2\boldsymbol c_1 - \boldsymbol c_2 + 3\boldsymbol c_3
\end{aligned}
$$

其中 $ B = (\boldsymbol b_1, \boldsymbol b_2) $ 与 $ C = (\boldsymbol c_1, \boldsymbol c_2, \boldsymbol c_3) $ 分别是 $ \mathbb{R}^2 $ 和 $ \mathbb{R}^3 $ 的有序基。  
确定 $ \Phi $ 关于有序基 $ B $ 与 $ C $ 的变换矩阵 $ \boldsymbol A_{\Phi} $。

5. 确定 $ \Phi $ 关于基 $ B' $ 与 $ C' $ 的变换矩阵 $ \boldsymbol A' $。  
6. 考虑在基 $ B' $ 下坐标为 $ [2, 3]^\top $ 的向量 $ \boldsymbol x \in \mathbb{R}^2 $，换言之，$ \boldsymbol x = 2\boldsymbol b'_1 + 3\boldsymbol b'_2 $。

a. 计算 $ \boldsymbol x $ 在基 $ B $ 下的坐标。  
b. 据此计算 $ \Phi(\boldsymbol x) $ 在基 $ C $ 下表示的坐标。  
c. 随后，用 $ \boldsymbol c'_1, \boldsymbol c'_2, \boldsymbol c'_3 $ 表示 $ \Phi(\boldsymbol x) $。  
d. 利用 $ \boldsymbol x $ 在 $ B' $ 下的表示以及矩阵 $ \boldsymbol A' $ 直接求出上述结果。
