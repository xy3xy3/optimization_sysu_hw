# DCS440 最优化理论 第二次作业: 凸优化问题与对偶理论

## 12 月 13 日(星期五)23:59 前提交

## 1. 推导线性规划问题的对偶问题和KKT条件:

$$
\mathop{\min }\limits_{x}{c}^{\top }x
$$

$$
\text{s.t.}{Gx} \leq  h
$$

$$
{Ax} = b
$$

2. 推导以下问题的对偶问题:

$$
\mathop{\min }\limits_{x}\frac{1}{2}{\begin{Vmatrix}x - {x}_{0}\end{Vmatrix}}_{2}^{2} + \mathop{\sum }\limits_{{i = 1}}^{N}{\begin{Vmatrix}{A}_{i}x + {b}_{i}\end{Vmatrix}}_{2}
$$

其中 ${A}_{i} \in  {\mathbb{R}}^{{m}_{i} \times  n},{b}_{i} \in  {\mathbb{R}}^{{m}_{i}}$ ,且 ${x}_{0} \in  {\mathbb{R}}^{n}$ 。(提示: 引入新的变量 ${y}_{i} \in  {\mathbb{R}}^{{m}_{i}}$ 以及等式约束 ${y}_{i} = {A}_{i}x + {b}_{i}$ ,将原无约束优化问题转化为约束优化问题后,再推导其对偶问题。)