# 线性代数

<p align="center">
  <img src="https://raw.githubusercontent.com/dxxzst/mml-book-chinese/main/docs/images/LinearAlgebra.png" alt="LinearA lgebra" title="线性代数 LinearA lgebra" /><br>
</p>

在形式化直观概念时，一种常见的方法是构造一组对象（符号）和一组操作这些对象的规则。 这被称为 _代数（algebra）_。线性代数是研究向量和某些代数规则来操作向量。 我们很多人在学校知道的向量被称为“几何向量”，通常用字母上方的小箭头表示，例如 $ \vec{x} $ 和 $ \vec{y} $ 。 在本书中，我们讨论向量的更一般概念并使用粗体字母表示它们，例如 $ \boldsymbol x $ 和 $ \boldsymbol y $ 。

通常，向量是特殊的对象，可以将它们相加并乘以标量以产生另一个相同类型的对象。 从抽象的数学角度来看，任何满足这两个属性的对象都可以被认为是一个向量。 以下是此类向量对象的一些示例：

1. 几何向量（Geometric vectors）。 这个向量的例子可能在高中数学和物理中很熟悉。 几何向量 — 参见图 2.1(a) — 是有向线段，可以绘制（至少在二维中）。 几何向量可以相加，比如 $ \vec{x} +  \vec{y} = \vec{z} $ 是另外一个几何向量。此外，乘以标量 $ \lambda\vec{x} $，$ \lambda \in \mathbb{R} $，也是一个几何向量。实际上，它是由 $ \lambda $ 缩放的原始向量。 因此，几何向量是前面介绍的向量概念的实例。 将向量解释为几何向量，可以让我们能够利用对方向和大小的直觉来推理数学运算。

<p align="center">
  <img src="https://raw.githubusercontent.com/dxxzst/mml-book-chinese/main/docs/images/Figure2.1.png" alt="不同类型的向量。 向量可以是令人惊讶的对象，包括 (a) 几何向量和 (b) 多项式。" title="不同类型的向量。 向量可以是令人惊讶的对象，包括 (a) 几何向量和 (b) 多项式。" /><br>
  <b>图 2.1 不同类型的向量。 向量可以是令人惊讶的对象，包括 (a) 几何向量和 (b) 多项式。</b><br>
</p>

2. 多项式（Polynomials）也是向量；参见图 2.1(b)：两个多项式可以相加，得到另一个多项式； 并且它们可以乘以一个标量 $ \lambda \in \mathbb{R} $，结果也是一个多项式。 因此，多项式是（相当不寻常的）向量的实例。 请注意，多项式与几何向量非常不同。 几何向量是具体的“图画”，多项式是抽象的概念。 然而，它们都是前述意义上的向量。

3. 音频信号（Audio signals）是向量。 音频信号表示为一系列数字。 我们可以将音频信号加在一起，它们的总和就是一个新的音频信号。 如果我们缩放音频信号，我们也会获得音频信号。 因此，音频信号也是一种向量。

4. $ \mathbb{R}^n $（n 个实数的元组）的元素是向量。  $ \mathbb{R}^n $ 比多项式更抽象，它是我们在本书中关注的概念。例如，
$$ \boldsymbol a = \begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix} \in \mathbb{R}^3 \tag{2.1} $$
是一个三元组数字的例子。将两个向量 $ \boldsymbol a, \boldsymbol b \in \mathbb{R}^n $ 按分量相加得到另一个向量：$ \boldsymbol a + \boldsymbol b = \boldsymbol c \in \mathbb{R}^n $。此外，将 $ \boldsymbol a \in \mathbb{R}^n $ 乘以 $ \lambda \in \mathbb{R} $ 得到一个缩放向量 $ \lambda \boldsymbol a \in \mathbb{R} $。将向量视为 $ \mathbb{R}^n $ 的元素有一个额外的好处，它松散地对应于计算机上的实数数组。 许多编程语言支持数组操作，这允许方便地实现涉及向量操作的算法。

线性代数侧重于这些向量概念之间的相似性。 我们可以将它们加在一起并乘以标量。 我们将主要关注 $ \mathbb{R}^n $ 中的向量，因为线性代数中的大多数算法都是在 $ \mathbb{R}^n $ 中制定的。 我们将在第8章中看到，我们经常将数据表示为 $ \mathbb{R}^n $ 中的向量。 在本书中，我们将关注有限维向量空间，在这种情况下，任何类型的向量和 $ \mathbb{R}^n $ 之间都存在 1:1 的对应关系。 方便时，我们将使用有关几何向量的直觉并考虑基于数组的算法。

数学中的一个主要思想是“闭包（closure）”的思想。 这就是问题：我提议的行动会产生什么样的结果？ 在向量的情况下：从一个小的向量集开始，将它们相加并缩放它们，可以得到什么样的向量集？这就产生了向量空间（第2.4节）。向量空间的概念及其性质是机器学习的基础。本章介绍的概念如图2.2所示。

本章主要基于 Drumm 和 Weil (2001)、Strang (2003)、Hogben (2013)、Liesen 和 Mehrmann (2015) 的讲义和书籍，以及 Pavel Grinfeld 的线性代数系列。 其他优秀资源包括 Gilbert Strang 在麻省理工学院的线性代数课程和 3Blue1Brown 的线性代数系列。

<p align="center">
  <img src="https://raw.githubusercontent.com/dxxzst/mml-book-chinese/main/docs/images/Figure2.2.png" alt="图 2.2 本章介绍的概念的思维导图，以及它们在本书其他部分中的使用位置。" title="图 2.2 本章介绍的概念的思维导图，以及它们在本书其他部分中的使用位置。" /><br>
   <b>图 2.2 本章介绍的概念的思维导图，以及它们在本书其他部分中的使用位置。</b><br>
</p>


线性代数在机器学习和普通数学中扮演着重要的角色。 本章介绍的概念进一步扩展到第3章中的几何概念。在第5章中，我们将讨论向量微积分，其中矩阵运算的基本知识是必不可少的。 在第10章中，我们将使用投影（将在第 3.8 节中介绍）通过主成分分析（PCA）进行降维。 在第9章中，我们将讨论线性回归，其中线性代数在解决最小二乘问题中起着核心作用。

## 2.1 线性方程组

线性方程组是线性代数的核心部分。 许多问题可以表述为线性方程组，而线性代数为我们提供了解决它们的工具。

> 
> **例 2.1**
> 
> 公司生产的产品 $ N_1,...,N_n $ 需要原材料 $ R_1,...,R_m $。为了生产一个单位的产品 $ N_j $，需要 $ a_{ij} $ 个单位的资源 $ R_i $，其中 $ i = 1,...,m $ 且 $ j = 1,...,n $。目标是找到一个最佳生产计划，即如果总共有 $ b_i $ 个单位的资源 $ R_i $ 可用并且（理想情况下）没有剩余资源，则应该生产多少单位 $ x_j $ 的产品 $ N_j $。
> 如果我们生产 $ x_1,...,x_n $ 个单位对应的产品，我们一共需要 $$ a_{i1}x_1 + \cdots + a_{in}x_n \tag{2.2} $$ 个单位的资源 $ R_i $。因此，最优生产计划 $ (x_1,...,x_n) \in \mathbb{R}^n $ 必须满足以下方程组：
$$ 
\begin{array}{clr}
  a_{11}x_1 + & \cdots + a_{1n}x_n = b_1 \\
      & \vdots \\
  a_{m1}x_1 + &  \cdots + a_{mn}x_n = b_m \tag{2.3}
\end{array}
$$
> 其中 $ a_{ij} \in \mathbb{R}^n $ 和 $ b_i \in \mathbb{R}^n $。
> 

方程(2.3)是线性方程组（system of linear equations）的一般形式，$ x_1,...,x_n $ 是这个系统的未知数。 满足(2.3)的每一个n元组 $ (x_1,...,x_n) \in \mathbb{R}^n$ 都是线性方程组的一个解。

> 
> **例 2.2**
>
> 线性方程组:
$$
\begin{array}{clr}
  x_1 + x_2 + x_3 & = 3 &(1)\\
  x_1 - x_2 + 2x_3 & = 2 &(2)\\
  2x_1 + 3x_3 & = 1 &(3) \tag{2.4}
\end{array}
$$
>  _没有解_ ：将前两个方程相加得到 $ 2x_1 + 3x_3 = 5 $ ，这与第三个方程相矛盾。
> 让我们看看线性方程组
$$
\begin{array}{clr}
  x_1 + x_2 + x_3 & = 3 &(1)\\
  x_1 - x_2 + 2x_3 & = 2 &(2)\\
  x_2 + x_3 & = 2 &(3)\tag{2.5}
\end{array}
$$
> 从第一个和第三个等式可以得出 $ x_1 = 1 $ 。 从（1）+（2），我们得到 $ 2x_1 + 3x_3 = 5 $，即 $ x_3 = 1 $。从 (3) 中，我们得到 $ x_2 = 1 $。 因此，(1,1,1) 是唯一可能且 _唯一的解_（通过插入验证（1,1,1）是解）。
> 作为第三个例子，我们考虑
$$
\begin{array}{clr}
  x_1 + x_2 + x_3 & = 3 &(1)\\
  x_1 - x_2 + 2x_3 & = 2 &(2)\\
  2x_1 + 3x_3 & = 5 &(3)\tag{2.6}
\end{array}
$$
> 由于（1）+（2）=（3），我们可以省略第三个方程（冗余）。 从（1）和（2），我们得到 $ 2x_1 = 5 - 3x_3 $ 和 $ 2x_2 = 1 + x_3 $。 我们定义 $ x_3 = a \in \mathbb{R} $ 作为一个自由变量，使得任何三元组
$$
\left( \frac{5}{2} - \frac{3}{2}a, \frac{1}{2} + \frac{1}{2}a, a \right), a \in \mathbb{R} \tag{2.7}
$$
> 都是线性方程组的解，即我们得到一个包含无限多个解的解集。

<p align="center">
  <img src="https://raw.githubusercontent.com/dxxzst/mml-book-chinese/main/docs/images/Figure2.3.png" alt="图 2.3 具有两个变量的两个线性方程组的解空间可以在几何上解释为两条线的交点。 每个线性方程代表一条线。" title="图 2.3 具有两个变量的两个线性方程组的解空间可以在几何上解释为两条线的交点。 每个线性方程代表一条线。" /><br>
   <b>图 2.3 具有两个变量的两个线性方程组的解空间可以在几何上解释为两条线的交点。 每个线性方程代表一条线。</b><br>
</p>

一般而言，对于实值线性方程组，我们要么没有，只有一个，要么有无穷多个解。 当我们无法求解线性方程组时，线性回归（第9章）是解决例2.1的一个版本。

_备注_ （线性方程组的几何解释）。 在具有两个变量 $ x_1, x_2 $ 的线性方程组中，每个线性方程定义 $ x_1 x_2 $ 平面上的一条线。 由于线性方程组的解必须同时满足所有方程，因此解集是这些线的交点。 该交集可以是一条线（如果线性方程描述同一条线）、一个点或空（当这些线平行时）。 图 2.3 给出了方程组
$$
\begin{array}{clr}
  4x_1 + 4x_2 & = 5 \\
  2x_1 - 4x_2 & = 1 \tag{2.8}
\end{array}
$$
的说明，其中解空间是点 $ (x_1, x_2) = (1, \frac{1}{4}) $。 类似地，对于三个变量，每个线性方程确定三维空间中的一个平面。 当我们与这些平面相交时，即同时满足所有线性方程，我们可以得到一个解集，它是一个平面、一条线、一个点或空的（当这些平面没有公共交点时）。

对于求解线性方程组的系统方法，我们将引入一个有用的紧凑符号。 我们将系数 $ a_{ij} $ 收集到向量中并将向量收集到矩阵中。 换句话说，我们按照以下形式编写（2.3）中的系统：
$$
 \begin{bmatrix} a_{11} \\ \vdots \\ a_{m1} \end{bmatrix}x_1 + 
 \begin{bmatrix} a_{12} \\ \vdots \\ a_{m2} \end{bmatrix}x_2 + 
 \cdots + 
 \begin{bmatrix} a_{1n} \\ \vdots \\ a_{mn} \end{bmatrix}x_n = 
 \begin{bmatrix} b_1 \\ \vdots \\ b_m \end{bmatrix} \tag{2.9}
$$
$$
\Longleftrightarrow 
\begin{bmatrix} a_{11} & \cdots & a_{1n} \\ \vdots \\ a_{m1} & \cdots & a_{mn} \end{bmatrix} 
\begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix} = 
\begin{bmatrix} b_1 \\ \vdots \\ b_m \end{bmatrix} \tag{2.10}
$$

下面，我们将仔细研究这些矩阵并定义计算规则。 我们将在2.3节返回求解线性方程。

## 2.2 矩阵

矩阵在线性代数中起着核心作用。 它们可用于紧凑地表示线性方程组，此外它们也表示线性函数（线性映射），我们将在后面的2.7节中看到。 在我们讨论这些有趣的话题之前，让我们首先定义矩阵是什么以及我们可以对矩阵进行什么样的操作。 我们将在第4章看到更多矩阵的性质。

**定义 2.1** （矩阵）。对于 $ m,n \in \mathbb{N} $，实值 $ (m,n) $ _矩阵（matrix）_ $ \boldsymbol A $ 是元素 $ a_{ij} $ 的 $ m·n $ 元组，$ i = 1,...,m , j = 1,...,n $，其由$ m $行和$ n $列组成的矩形排序阵列：

$$
\boldsymbol A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}, a \in \mathbb{R}. \tag{2.11}
$$

通常 $ (1,n) $ - 矩阵称为行，$ (m,1) $ - 矩阵称为列。 这些特殊矩阵也称为 _行/列向量_（row/column vectors）。

$ \mathbb{R}^{m \times n} $ 是所有实值 $ (m,n) $ 矩阵的集合。 $ \boldsymbol A \in \mathbb{R}^{m \times n} $ 可以等价的表示为将矩阵的所有 $ n $ 列堆叠成一个长的向量 $ \boldsymbol a \in \mathbb{R}^{mn} $ ； 见图 2.4。

<p align="center">
  <img src="https://raw.githubusercontent.com/dxxzst/mml-book-chinese/main/docs/images/Figure2.4.png" alt="图 2.4 通过堆叠其列，矩阵 A 可以表示为一个长向量 a。" title="图 2.4 通过堆叠其列，矩阵 A 可以表示为一个长向量 a。" /><br>
   <b>图 2.4 通过堆叠其列，矩阵A可以表示为一个长向量a</b><br>
</p>

### 2.2.1 矩阵加法和乘法

两个矩阵的和 $ \boldsymbol A \in \mathbb{R}^{m \times n}, \boldsymbol B \in \mathbb{R}^{m \times n} $ 被定义为对应元素的和，即，

$$
 \boldsymbol A + \boldsymbol B := \begin{bmatrix} a_{11} + b_{11} & \cdots & a_{1n} + b_{1n} \\ \vdots \\ a_{m1} + b_{m1} & \cdots & a_{mn} + b_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n} \tag{2.12}
$$

对于矩阵 $ \boldsymbol A \in \mathbb{R}^{m \times n}, \boldsymbol B \in \mathbb{R}^{n \times k} $ ，乘积 $ \boldsymbol C = \boldsymbol A \boldsymbol B \in \mathbb{R}^{m \times k} $ 的元素 $ c_{ij} $ 计算自

$$
c_{ij} = \sum_{l=1}^{n}{a_{il}b_{lj}}, i = 1,...,m, \ j = 1,...,k \tag{2.13} 
$$

> 注意矩阵的大小。
> ``` C = np.einsum(’il, lj’, A, B) ```

这意味着，为了计算元素 $ c_{ij} $，我们将 $ \boldsymbol A $ 的第 $ i $ 行的元素与 $ \boldsymbol B $ 的第 $ j $ 列的元素相乘，然后将它们相加。 稍后在第3.2节中，我们将称其为相应行和列的 _点积（dot produc）_ 。 在需要明确表示正在执行乘法的情况下，我们使用符号 $ \boldsymbol A ·\boldsymbol B $  来表示乘法（明确显示“·”）。

> $ \boldsymbol A $ 中有 $ n $ 列，$ \boldsymbol B $ 中有 $ n $ 行，因此我们可以计算 $ a_{il}b_{lj} $，其中 $ l = 1,...,n $。
> 通常，两个向量 $ \boldsymbol a, \boldsymbol b $ 之间的点积表示为 $ \boldsymbol a^\top \boldsymbol b $ 或 $ \langle \boldsymbol a, \boldsymbol b \rangle $。

_备注_ 。 只有当它们的“相邻”维度匹配时，矩阵才能相乘。 例如，一个 $ n \times k $ 矩阵 $ \boldsymbol A $ 可以乘以一个 $ k \times m $ 矩阵 $ \boldsymbol B $ ，只能从左侧相乘：

$$
\underbrace{\boldsymbol A}_{n \times k} \  \underbrace{\boldsymbol B}_{k \times m} = \underbrace{\boldsymbol C}_{n \times m} \tag{2.14}
$$

如果 $ m \not= n $，则乘积  $ \boldsymbol {BA} $ 未定义，因为相邻维度不匹配。

### 2.2.2 逆和转置

### 2.2.3 乘以标量

### 2.2.4 线性方程组的紧凑表示

## 2.3 求解线性方程组 