# M-space embedding

Training objective: encoder $f : X \to z$ preserves the multi-scale spectral clustering structure of the input features. Same method for decoder omitted.

## Notation

- $X \in \mathbb{R}^{N \times D}$ : input features, $\quad z = f(X) \in \mathbb{R}^{N \times d} \quad (d \ll D)$
- $W \in \mathbb{R}^{N \times N}$ : RBF affinity, $\quad W_{ij} = \exp(-\|x_i - x_j\|^2 / 2\sigma^2), \quad \sigma$ adaptively chosen by mean degree
- $A = SWS$ : symmetric normalization, $\quad S = \operatorname{diag}(D^{-1/2}), \quad D = \operatorname{diag}(W1)$
- $A = U \Lambda U^T$ : eigenvalues descending, $\quad U_k = U[:, :k]$ (top-$k$ eigenvectors)
- $P_k = U_k U_k^T \in \mathbb{R}^{N \times N}$ : projector onto the top-$k$ eigenspace

## Flag space $P$ stack

For a set of scales $n\_eig\_list = [k_1, \ldots, k_B]$, computed once from $X$:

$$
P_{\mathrm{gt}} = \operatorname{stack}_b(P_{k_b}) \in \mathbb{R}^{B \times N \times N}
$$

Each $k_b$ is one clustering granularity (small $k \to$ coarse, large $k \to$ fine).

# Training Loss

We want to train encoder $f : X \to z$, train the $f$.

## Loss typeA: `z_eigvec`

Repeat the full pipeline on $z$ to get $\hat A, \hat U, \hat \lambda$ and projectors $\hat P_{k_b} = \hat U_{k_b}\hat U_{k_b}^{T}$.

$$
\mathcal{L}^{z\text{-eigvec}}_{\mathrm{flag}}
=
\frac{1}{BN^2}
\sum_{b=1}^{B}
\left\|
P_{k_b}
-
\hat P_{k_b}
\right\|_F^2
$$

- Per-scale alignment: each $k$ requires the top-$k$ eigenspace of $z$ to match that of the input.
- Backward uses the $H/(\lambda_S - \lambda_\perp)$ formula, stable under eigenvalue degeneracy.

## Loss typeB: `z`

No spectral decomposition; use embedding inner products directly:

$$
\hat P = zz^T \in \mathbb{R}^{N \times N},
\qquad
\mathcal{L}^{z}_{\mathrm{flag}}
=
\left\|
\frac{1}{B}\sum_{b}^{B} P_{k_b} - zz^T
\right\|_F^2
$$

$$
\bar P = \frac{1}{B}\sum_b P_{k_b}
\quad \text{(single matrix).}
$$

- Target collapses to the "mean" of multi-scale projectors.
