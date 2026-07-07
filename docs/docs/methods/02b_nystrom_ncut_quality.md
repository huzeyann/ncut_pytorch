# Nyström Ncut (Quality)

Nyström NCut solves on a landmark subset and propagates to the full graph. The choice of landmarks changes more than speed — it changes which structure the eigenvectors recover.

## The imbalance problem

On the full affinity $W \in \mathbb{R}^{N \times N}$, natural images are class-imbalanced: a large background region contributes many high-weight edges, small objects few. The degree

$$
D_{ii} = \sum_j W_{ij}
$$

is therefore concentrated in majority regions. Even after symmetric normalization $A = SWS$, $S = \operatorname{diag}(D^{-1/2})$, the leading eigenvectors tend to isolate the dense majority first; minority classes are merged into dominant clusters or fragmented.

## FPS landmarks

Farthest Point Sampling selects landmarks greedily:

$$
i_{t+1} = \arg\max_{i} \min_{j \in \mathcal{S}_t} \|x_i - x_j\|
$$

where $\mathcal{S}_t$ is the set of already-selected landmarks. This maximizes the minimum inter-landmark distance, so landmark density is approximately uniform in feature space, independent of the data density.

<div style="text-align: center;">
<img src="../images/methods_02b_nystrom_ncut_quality/fps_vs_random.png" style="width:100%;">
</div>

## Effect on eigenvectors

Building the Nyström affinity $A$ on FPS landmarks instead of the full $W$:

- Local neighborhood density is roughly constant across landmarks, so $D \approx cI$ in expectation.
- Normalization no longer has to compensate for density imbalance.
- Leading eigenvectors track geometric structure rather than sampling density.

When propagated back to the full graph, this yields sharper boundaries for under-represented classes and higher recall for small objects than NCut on the raw (imbalanced) affinity.
