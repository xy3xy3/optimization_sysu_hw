# DCS440 最优化理论 第一次作业: 凸集与凸函数

11 月 22 日(星期五) 23:59 前提交

1. 设 $C \subseteq  {\mathbb{R}}^{n}$ 为一个凸集。证明: 对任意 $k$ 个向量 ${x}_{1},\cdots ,{x}_{k} \in  C$ ,以及 ${\theta }_{1},\cdots ,{\theta }_{k} \in  \mathbb{R}$ 满足 ${\theta }_{1} + \cdots  + {\theta }_{k} = 1,{\theta }_{i} \geq  0$ ,都有 ${\theta }_{1}{x}_{1} + \cdots  + {\theta }_{k}{x}_{k} \in  C$ 。(注: 凸集的定义要求此式在 $k = 2$ 时成立,这里需要证明对任意 $k \geq  2$ 都成立)

2. 设 $C \subseteq  {\mathbb{R}}^{n}$ 为线性方程组的解集,即

$$
C = \left\{  {x \in  {\mathbb{R}}^{n} \mid  {Ax} = b}\right\}
$$

其中 $A \in  {\mathbb{R}}^{m \times  n}, b \in  {\mathbb{R}}^{m}$ 。证明: $C$ 是凸集。

3. 设 $C \subseteq  {\mathbb{R}}^{n}$ 为二次不等式的解集,即

$$
C = \left\{  {x \in  {\mathbb{R}}^{n} \mid  {x}^{\top }{Ax} + {b}^{\top }x + c \leq  0}\right\}  ,
$$

其中 $A \in  {\mathbf{S}}^{n}, b \in  {\mathbb{R}}^{n}, c \in  \mathbb{R}$ 。证明: 若 $A \succcurlyeq  0$ (即 $A$ 是半正定矩阵),则 $C$ 是凸集。

4. 确定以下函数的凹凸性:

(a) $f\left( x\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}{x}_{i}\ln {x}_{i},{x}_{i} \in  {\mathbb{R}}_{+ + }, i = 1,\cdots , n$ ;

(b) $f\left( {{x}_{1},{x}_{2}}\right)  = {x}_{1}{x}_{2},\left( {{x}_{1},{x}_{2}}\right)  \in  {\mathbb{R}}_{+ + }^{2}$ ;

(c) $f\left( {{x}_{1},{x}_{2}}\right)  = {x}_{1}/{x}_{2},\left( {{x}_{1},{x}_{2}}\right)  \in  {\mathbb{R}}_{+ + }^{2}$ ;

5. 设 $h : {\mathbb{R}}^{m} \rightarrow  \mathbb{R}, g : {\mathbb{R}}^{n} \rightarrow  {\mathbb{R}}^{m}$ ,则复合函数 $f \mathrel{\text{:=}} h \circ  g : {\mathbb{R}}^{n} \rightarrow  \mathbb{R}$ 定义为

$$
f\left( x\right)  \mathrel{\text{:=}} h\left( {g\left( x\right) }\right)  = h\left( {{g}_{1}\left( x\right) ,\cdots ,{g}_{m}\left( x\right) }\right) ,
$$

$$
\operatorname{dom}f \mathrel{\text{:=}} \{ x \in  \operatorname{dom}g \mid  g\left( x\right)  \in  \operatorname{dom}h\} .
$$

证明: 若 ${g}_{i}$ 是凹函数, $h$ 是凸函数,且 $h$ 关于其每个分量是非增的,则复合函数 $f \mathrel{\text{:=}}$ $h \circ  g$ 是凸函数。