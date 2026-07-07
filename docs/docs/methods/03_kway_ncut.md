# k-way Ncut

NCut eigenvectors are defined only up to an orthogonal rotation. k-way Ncut finds the rotation that aligns the continuous embedding with a discrete cluster indicator matrix.

Reference: [Yu and Shi, "Multiclass spectral clustering," 2003](https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf).

## Notation

- $Z = [z_1, \dots, z_K] \in \mathbb{R}^{N \times K}$ : NCut eigenvectors
- $R \in \mathbb{R}^{K \times K}, \; R^\top R = I$ : unknown orthogonal rotation
- $\tilde{X}_{i,:} = Z_{i,:} / \|Z_{i,:}\|_2$ : row-normalized, each point on the unit sphere
- $X \in \{0, 1\}^{N \times K}$ : one-hot cluster indicator, $X_{il}=1 \iff l=\arg\max_k (\tilde{X}R)_{ik}$

## Alternating optimization

Initialize $R_0$ from $K$ FPS-selected rows of $\tilde{X}$ (approximately orthogonal), then repeat until convergence:

**1. Assign discrete labels given $R$:**

$$
Y = \tilde{X} R, \qquad
X_{il} = 1 \iff l = \arg\max_k Y_{ik}
$$

**2. Update $R$ given $X$ via SVD:**

$$
M = X^\top \tilde{X}, \qquad
M = U \Omega V^\top, \qquad
R = V U^\top
$$

$R = VU^\top$ is the orthogonal matrix that best aligns $\tilde{X}$ to $X$ in the least-squares sense.

## Convergence

The NCut objective value is tracked as

$$
\mathrm{ncut} = 2\bigl(N - \operatorname{tr}\Omega\bigr)
$$

and the loop stops when $|\mathrm{ncut}_t - \mathrm{ncut}_{t-1}| < \varepsilon$. Typically converges in a few iterations.

## Quick variant: `quick_kway`

`quick_kway` replaces the SVD-based alternating optimization with spherical k-means on the row-normalized eigenvectors $\tilde{X}$.

- $C \in \mathbb{R}^{K \times K}$ : centroids, initialized from $K$ FPS-selected rows of $\tilde{X}$
- Iterate $T$ times:

**1. Assign each point to its nearest centroid by cosine similarity:**

$$
a_i = \arg\max_k (\tilde{X} C^\top)_{ik}
$$

**2. Update centroids as the normalized mean of assigned points:**

$$
C_{k,:} \leftarrow \operatorname{normalize}\!\left(\frac{1}{|\{i : a_i = k\}|} \sum_{i : a_i = k} \tilde{X}_{i,:}\right)
$$

**3. Return rotation:**

$$
R = C^\top
$$

Use `quick_kway` when $N$ is large and a few k-means iterations suffice; use `axis_align` (the SVD alternating optimization above) when accuracy is preferred.
