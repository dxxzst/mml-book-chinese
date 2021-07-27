# 线性代数

<p align="center">
  <img src="https://raw.githubusercontent.com/dxxzst/mml-book-chinese/main/docs/images/LinearAlgebra.png" alt="LinearA lgebra" title="线性代数 LinearA lgebra" /><br>
</p>

在形式化直观概念时，一种常见的方法是构造一组对象（符号）和一组操作这些对象的规则。 这被称为 _代数（algebra）_。线性代数是研究向量和某些代数规则来操作向量。 我们很多人在学校知道的向量被称为“几何向量”，通常用字母上方的小箭头表示，例如 $ \vec{x} $ 和 $ \vec{y} $ 。 在本书中，我们讨论向量的更一般概念并使用粗体字母表示它们，例如 $ \boldsymbol x $ 和 $ \boldsymbol y $ 。

通常，向量是特殊的对象，可以将它们相加并乘以标量以产生另一个相同类型的对象。 从抽象的数学角度来看，任何满足这两个属性的对象都可以被认为是一个向量。 以下是此类向量对象的一些示例：

1. 几何向量（Geometric vectors）。 这个向量的例子可能在高中数学和物理中很熟悉。 几何向量 — 参见图 2.1(a) — 是有向线段，可以绘制（至少在二维中）。 几何向量可以相加，比如 $ \vec{x} +  \vec{y} = \vec{z} $ 是另外一个几何向量。此外，乘以标量 $ \lambda\vec{x} $，$ \lambda \in \mathbb{R} $，也是一个几何向量。实际上，它是由 $ \lambda $ 缩放的原始向量。 因此，几何向量是前面介绍的向量概念的实例。 将向量解释为几何向量使我们能够利用我们对方向和大小的直觉来推理数学运算。

<p align="center">
  <img src="https://raw.githubusercontent.com/dxxzst/mml-book-chinese/main/docs/images/Figure1.2.png" alt="不同类型的向量。 向量可以是令人惊讶的对象，包括 (a) 几何向量和 (b) 多项式。" title="不同类型的向量。 向量可以是令人惊讶的对象，包括 (a) 几何向量和 (b) 多项式。" /><br>
</p>

2. 多项式（Polynomials）也是向量；参见图 2.1(b)：两个多项式可以相加，得到另一个多项式； 并且它们可以乘以一个标量 $ \lambda \in \mathbb{R} $，结果也是一个多项式。 因此，多项式是（相当不寻常的）向量的实例。 请注意，多项式与几何向量非常不同。 几何向量是具体的“图画”，多项式是抽象的概念。 然而，它们都是前述意义上的向量。

3. 音频信号（Audio signals）是向量。 音频信号表示为一系列数字。 我们可以将音频信号加在一起，它们的总和就是一个新的音频信号。 如果我们缩放音频信号，我们也会获得音频信号。 因此，音频信号也是一种向量。

4. $ \mathbb{R}^n $（n 个实数的元组）的元素是向量。  $ \mathbb{R}^n $ 比多项式更抽象，它是我们在本书中关注的概念。例如，
$$ \boldsymbol a = \begin{bmatrix} 1 \\ 2 \\ 3 \\ \end{bmatrix} \in \mathbb{R}^3 $$
是一个三元组数字的例子。将两个向量 $ \boldsymbol a, \boldsymbol b \in \mathbb{R}^n $ 按分量相加得到另一个向量：$ \boldsymbol a + \boldsymbol b = \boldsymbol c \in \mathbb{R}^n $。此外，将 $ \boldsymbol a \in \mathbb{R}^n $ 乘以 $ \lambda \in \mathbb{R} $ 得到一个缩放向量 $ \lambda \boldsymbol a \in \mathbb{R} $。将向量视为 $ \mathbb{R}^n $ 的元素有一个额外的好处，它松散地对应于计算机上的实数数组。 许多编程语言支持数组操作，这允许方便地实现涉及向量操作的算法。

线性代数侧重于这些向量概念之间的相似性。 我们可以将它们加在一起并乘以标量。 我们将主要关注 $ \mathbb{R}^n $ 中的向量，因为线性代数中的大多数算法都是在 $ \mathbb{R}^n $ 中制定的。 我们将在第8章中看到，我们经常将数据表示为 $ \mathbb{R}^n $ 中的向量。 在本书中，我们将关注有限维向量空间，在这种情况下，任何类型的向量和 $ \mathbb{R}^n $ 之间都存在 1:1 的对应关系。 方便时，我们将使用有关几何向量的直觉并考虑基于数组的算法。