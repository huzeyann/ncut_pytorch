# k-way Ncut

!!! abstract "TL;DR"
    
    1. **Solve Ncut**: Compute the Ncut eigenvectors $Z \in \mathbb{R}^{N\times K}$.
    2. **Find Rotation**: Iterative optimize rotation $R$ via SVD until convergence.
    3. **Rotate and Discretize**: Apply $R$ to $Z$ and assign discrete clusters.

---

We need k-way Ncut because the Ncut eigenvectors do not directly give discrete labels; k-way Ncut is a principled solution to generate discrete labels from Ncut eigenvectors.

**TODO: add an image shows discrete cluster labels**

## How k-way Ncut Works (short version)

> **Compute Ncut eigenvectors → row-normalize → iterate: assign clusters by argmax → recompute rotation $R$ via SVD ($M = X^\top\tilde{X}$), ($R = \tilde{U}U^\top$).**

> This rotation $R$ aligns normalized cuts with discrete labels.

reference: [Yu and Shi, "Multiclass spectral clustering,"](https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf)

## How k-way Ncut Works (long version)

### **How to Rotate Ncut Eigenvectors Using $R$**

Given the Ncut embedding
$$
Z = [z_1,\dots,z_K] \in \mathbb{R}^{N\times K},
$$
Ncut gives you eigenvectors only up to an **unknown orthogonal rotation**:

$$
Z \sim ZR ,\quad R^\top R = I.
$$

To convert $Z$ into **hard cluster assignments**, [the k-way Ncut paper](https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf) find the rotation $R$ that makes $ZR$ “as close as possible” to a one-hot indicator matrix.

The core idea is:

> **Rotate the continuous embedding so that each data point is closest to a single coordinate axis.**

This is solved via an **alternating optimization**:

### **Step 1 — Row-normalize the eigenvectors**

Normalize each row of $Z$:

$$
\tilde{X}_{i,:} = \frac{Z_{i,:}}{\|Z_{i,:}\|_2},
$$

so each point lies on the unit sphere.

### **Step 2 — Initialize $R$**

Pick $K$ rows of $\tilde{X}$ that are approximately orthogonal.

Form an initial orthogonal matrix $R_0$.

### **Step 3 — Given $R$, assign discrete labels (closest axis)**

Compute:

$$
Y = \tilde{X} R.
$$

For each point $i$, pick the index of its largest coordinate:

$$
X_{il}=1 \iff l=\arg\max_k Y_{ik}.
$$

This converts the rotated embedding into a discrete indicator matrix $X$.

### **Step 4 — Given $X$, update $R$ via SVD**

Compute:

$$
M = X^\top \tilde{X}.
$$

Take the SVD:

$$
M = U \Omega \tilde{U}^\top.
$$

Then the optimal rotation is:

$$
\boxed{R = \tilde{U} U^\top}
$$

This is the orthogonal matrix that best aligns $\tilde{X}$ to the discrete solution in least-squares sense.

### **Step 5 — Repeat until convergence**

Alternate:

1. $R \rightarrow X$  (argmax assignment)
2. $X \rightarrow R$  (SVD update)

Typically converges in a few iterations.


<div style="display: flex; justify-content: space-between; gap: 20px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
  <a href="/methods/02b_nystrom_ncut_quality" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Previous</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">← Nyström Ncut (Quality)</span>
  </a>
  <a href="/methods" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; transition: all 0.2s;">
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">Back to Overview</span>
  </a>
  <a href="/methods/04_mspace_coloring" style="flex: 1; text-decoration: none; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; display: flex; flex-direction: column; align-items: flex-end; text-align: right; transition: all 0.2s;">
    <span style="font-size: 12px; color: #666; margin-bottom: 5px;">Next</span>
    <span style="font-size: 16px; font-weight: bold; color: #007bff;">Mspace Coloring →</span>
  </a>
</div>
