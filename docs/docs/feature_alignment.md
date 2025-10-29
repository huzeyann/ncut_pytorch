# Feature Alignment


## Motivation

Models (CLIP, DINO, SD) may have learned similar visual concepts, e.g. the concept of dog vs. cat exists in all models. However, each model represent visual concepts in its own language, so computing L2 distance from DINO feature to CLIP feature don’t make sense. What's more, the representation languages of the different layers of the same model will also be different. This document discuss how to align the features into a unified space.

## Results of Feature Alignment

The goal of feature alignment is that corresponding semantic parts become more consistent across images.

First of all, we show the visualization results of SAM model before and after alignment. 

| Before alignment | After alignment |
| --- | --- |
| ![before](../images/02_sam_original_ncut.png) | ![after](../images/02_sam_affinity_ncut.png) |

You should observe that facial regions such as hair, forehead, eyes, nose, and mouth are mapped to more consistent locations/colors after alignment. And it has shown a consistency between layers.

After that, we show the visualization results of DINO V3 model before and after alignment.

| Before alignment | After alignment |
| --- | --- |
| ![before](../images/05_dinov3_original_ncut.png) | ![after](../images/04_dinov3_affinity_ncut.png) |


You should observe that facial regions such as hair, forehead, eyes, nose, and mouth are mapped to more consistent locations/colors after alignment. It has not only shown a consistency between layers, but also a consistency across the models in comparison with SAM Model.

As you can see in the results that, we can see that SAM model pays more attention to the edge information while DINO v3 model pays more attention to the general patterns.

## Didactic Example
Relevent Files:

[n25c0006a_toy_layeralign_rbf.py](../images/n25c0006a_toy_layeralign_rbf.py)

[n25c0006b_toy_layeralign_train.py](../images/n25c0006b_toy_layeralign_train.py)

Given 2 set of points 

$$
set1: X = \{x_1, x_2, \dots,x_n\}, x_i \in \mathbb{R}^{2} \\
set2: Y = \{y_1, y_2, \dots,y_n\}, y_i \in \mathbb{R}^{2}
$$

$X$ and $Y$ are generated from the same prior set $Z = \{z_1, z_2, \dots,z_n\}$, but with different transform function $f$ and $g$.

$$
x_i = f(z_i) \\
y_i = g(z_i)
$$

![image.png](../images/image.png)

The image above shows the Set 1 ($X$), it does not show Set 2 (Y). Dots are labeled from 0 to 30 that match the number in affinity. Affinity is computed as RBF distance of X.

![image.png](../images/image%201.png)

The image above shows both set $X$ (Set 1) and $Y$ (Set 2). Dots are numbered, Set 1 is 0 to 30, Set 2 is 30 to 60.

in this example (eq on right side), $Y$ is scaled by $\frac{1}{4}$ from $X$, rotated 45 degree by the rotation matrix $R$, then shifted by 10.

$$
x_i = f(z_i) \\
y_i = \frac{1}{4} f(z_i) R + [10, 10]^T \\
$$

Before the alignment, the affinity (right side of the image above) shows: there’s 3 cluster in set1, but all dots in set2 belong to one big cluster. 

What we want the alignment to do is showed in the affinity matrix below: it shows there’s 3 cluster in set1 and 3 cluster in set2, and there’s a 1-to-1 pair that each cluster in set1 is paired with one cluster in set2 (off-diagonal strip).

Because we know how the X and Y is generated from Z, we know the true relationship between X and Y, the mutual information between X and Y is 1.

![image.png](../images/image%202.png)

After alignment, Affinity is computed on X_new and Y_new (not plotted on left) but not the original X and Y (plotted on left side)

$$
X_{new} = align(X) \\
Y_{new} = align(Y) \\
$$

where $X, Y \in \mathbb{R}^{n \times 2}$, $X_{new}, Y_{new} \in \mathbb{R}^{n \times d}$. $n$ is number of dots, $d$ is the new dimension after alignment.

## Alignment Methods 1: RBF auto-scaling

Given 2 set of points 

$$
set1: X = \{x_1, x_2, \dots,x_n\}, x_i \in \mathbb{R}^{2} \\
set2: Y = \{y_1, y_2, \dots,y_n\}, y_i \in \mathbb{R}^{2}
$$

The goal of alignment is to transform X and Y to a space that is invariant to shift/rotation/scaling, so that the aligned space can be used to compute distance across X and Y.

```python
def RBF_affinity_from_features(
    features: torch.Tensor,
    features_B: torch.Tensor = None,
    gamma: float = 1.0,
):
    features_B = features if features_B is None else features_B

    d = torch.cdist(features, features_B, p=2)
    A = torch.pow(d, 2)

    sigma = 2 * gamma * features.var(dim=0).sum()
    A = torch.exp(-A / sigma)
    return A
```

The above pair-wise RBF distance, it’s invariant to shift/scale/rotation.

- it’s invariant to shift/rotation because of Euclidean distance; invariant to isotropic scaling because of sigma is multiplied by variance of input. However, it’s not invariant to anisotropic scaling as we will see later

The layer align transform is defined below, it use RBF within a set to compute relative features.

```python
# X and Y is [n, 2]
X_new = RBF_affinity_from_features(X, X)  # [n, n]
Y_new = RBF_affinity_from_features(Y, Y)  # [n, n]
```

X_new converts 2-D absolute coordinates in X into n-D relative distance to other data points in X, Y_new does the same but w.r.t. data points in Y. There’s two critical pieces:

1. X is only contrasting with dots in X, Y in only contrasting with dots in Y, there’s no cross X and Y.
2. the order of x_i and y_i have to be the same.

## Limitation with Solutions

This section first discuss the information loss in alignment caused by RBF transform, loss of information can be solved by more sampling.

Then we study hard cases:

1. The most easy case is when $y_i$ is an **isotropic linear transform** of $x_i$, since RBF is invariant to such transform, we are guaranteed to align them.
2. Hard case 1: **outliers.** a few outliers that break the assumption that x_i and y_i are transformed from the same z_i $x_i = f(z_i), y_i = g(z_i)$
3. Hard case 2: **anisotropic linear transform.** $f$ and $g$ could be anisotrorpic, then a plain RBF is not invariant to the transform.
4. Hard case 3: **anisotropic non-linear transfrom.** A MLP is both anisotropic and non-linear.

---

To help hard cases, we developed **RBF auto-scaling**

---

### 1. Loss of information

In the step

```python
X_new = RBF_affinity_from_features(X, X)  # [n, n]
Y_new = RBF_affinity_from_features(Y, Y)  # [n, n]
```

X_new might loss information compare to X. 

Consider the following 5 sample case, that we sampled the last 5 out of n dots to compute relative distance for X_new and Y_new.

```python
X_new = RBF_affinity_from_features(X, X[-5:])  # [n, 5]
Y_new = RBF_affinity_from_features(Y, Y[-5:])  # [n, 5]
```

![image.png](../images/image%203.png)

In the image above, the 5 sampled dots are in bottom right corner. The affinity shows it incorrectly grouped first 20 dots as one big cluster. It’s because the 5 sample dots are in the bottom left cluster, relative distance to them can not tell the other two cluster apart.

The conclusion is the alignment need to sample diverse enough data points when computing relative distance as new feature, so that the relative distance (X_new) don’t loose information compare to the original (X).

### 2. Outliers

We made the assumption that x_i and y_i are generated from the same z_i. However, there’s could be outliers (e.g., massive activation tokens in CLIP)

![image.png](../images/image%204.png)

In the image above, node 34 and 42 are manually made to be outliers, the affinity on Set 2 only separate out normal vs outlier tokens, which is not what we want.

```python
def RBF_affinity_from_features(
    features: torch.Tensor,
    features_B: torch.Tensor = None,
    gamma: float = 1.0,
):
    features_B = features if features_B is None else features_B

    d = torch.cdist(features, features_B, p=2)
    A = torch.pow(d, 2)

    sigma = 2 * gamma * features.var(dim=0).sum()
    A = torch.exp(-A / sigma)
    return A
```

Recall that there’s a gamma parameter when computing RBF affinity. 

```python
X_new = RBF_affinity_from_features(X, X, gamma=1.0)  # [n, n]
Y_new = RBF_affinity_from_features(Y, Y, gamma=1.0)  # [n, n]
```

The default RBF gamma is 1.0, tuning gamma down make RBF robust to outliers, because it non-linearly scales outliers down.

![image.png](../images/image%205.png)

In the image above, tuning gamma for set 2 down from 1.0 to 0.01 recovered the desired 1-to-1 pair.

---

### RBF auto-scaling

There’s an automatic way to determine what’s the gamma value to use, we want to search for a gamma value such that the mean value of X_new and Y_new equal to a constant $c$. In other word, the average edge weights in X_new and Y_new should be the same constant  $c$

```python
# find gamma1 and gamm2, they should satisfy the following
X_new = RBF_affinity_from_features(X, X, gamma=gamma1)  # [n, n]
Y_new = RBF_affinity_from_features(Y, Y, gamma=gamma2)  # [n, n]
X_new.mean() == c
Y_new.mean() == c
```

what is a good value for constant $c$? The value of c means the average edge weights in affinity within one Set, in this toy example case, we know that the average edge weight should be 1/3 since there’s 3 clusters. 

After setting the constant c = 0.3, we could find by grid search (or binary search since it’s monotonic) that Set 1 gamma=0.186, Set 2 gamma=0.013.

![image.png](../images/image%206.png)

What’s a good constant value c for a real-world case? I would start trying by c=0.05, since it means each pixel is connected to 5% of other pixels. If the affinity is too sparse, then increase c.

---

### 3. Anisotropic scaling

We showed the RBF affinity alignment works for isotropic scaling. But the RBF affinity function is not invariant to anisotropic scaling. 

![image.png](../images/image%207.png)

in the image above, Set 2 dots are squeezed by 1/4 horizontally and stretched 4x vertically. $R$ is 45 degree rotation matrix.

The resulting affinity does not reveal 1-to-1 pair of clusters from Set 1 to Set 2.

$$
x_i = f(z_i) \\
y_i = \left[ f(z_i) R + [10, 10]^T \right] * [0.25, 4]^T \\
$$

![image.png](../images/image%208.png)

After applied RBF auto-scaling (defined in Limitation 2), it does help bring up the 1-to-1 pair.

### 4. non-linear transform

The above cases consider when X and Y is different linear transform, RBF distance is invariant to linear transform, but what about non-linear transform?

![image.png](../images/image%209.png)

The image above, Set 2 is transformed from Set 1 by a randomly initialized 4-layer MLP, $y_i = mlp(x_i)$, the input to MLP is z-score normalized.

![image.png](../images/image%2010.png)

After applying RBF auto-scaling, the 1-to-1 pair is stronger.

## Limitation without Solutions yet

### 1. Anisotropic scaling (strong anisotropicity)

We showed RBF auto-scaling helped anisotropic scaling case, but that was only 1/4 and 4x Anisotropic, what about 1/1000 or 1000x?

![image.png](../images/image%2011.png)

in the image above, Set 2 dots are squeezed by 1/1000 horizontally, $R$ is 45 degree rotation matrix.

$$
x_i = f(z_i) \\
y_i = \left[ f(z_i) R + [10, 10]^T \right] * [0.001, 1]^T \\
$$

![image.png](../images/image%2012.png)

RBF auto-scaling do help bring up 1-to-1 pair, but is this good enough?

### 2. bad non-linear transform

We showed RBF auto-scaling helped MLP, but that was 4 layer MLP, what about 12 layers?

![image.png](../images/image%2013.png)

![image.png](../images/image%2014.png)

This 12-layer MLP even erased the affinity within Set 2, this MLP removed too much information, there’s no way Set 2 could align with Set 1.

## Alignment Methods 2: training MLP

Given 2 set of points 

$$
set1: X = \{x_1, x_2, \dots,x_n\}, x_i \in \mathbb{R}^{2} \\
set2: Y = \{y_1, y_2, \dots,y_n\}, y_i \in \mathbb{R}^{2}
$$

And a label set Z (how to define Z is discussed later), Z is shared for X and Y

$$
label: Z = \{z_1, z_2, \dots, z_n \}, z_i \in \mathbb{R}^1
$$

the half-shared MLP alignment methods trains two models:

$$
model1: z_i = h_{share}(h_x(x_i)) \\
model2: z_i = h_{share}(h_y(y_i))
$$

where $h_{share}$ is a shared linear layer, same weights across model1 and model2. $h_x$ and $h_y$ is not-shared 3-layer MLP for X and Y. 

X and Y is z-score normalized before pass to the MLP

After training model1 and model2, we use the second last layer feature as aligned feature for X and Y:

$$
X_{new} = h_x(X) \\
Y_{new} = h_y(Y)
$$

**Regularization** is added to enforce X_new and X to maintain affinity:

```python
diff = RBF_affinity(X_new, X_new) - RBF_affinity(X, X)
reg = (diff ** 2).mean()
```

Training loss is CrossEntropy or MSE of Z, plus the regularization term. There’s multiply ways to define Z: 

1. Z is ground truth 3 clusters, This works, but not the best, the objective is too easy.
    
    ![image.png](../images/image%2015.png)
    
2. Z is index of each node, 0-29. This works great
    
    ![image.png](../images/image%2016.png)
    
3. Z is randomly assigned value from [0, 1, 2]. This Z does not work
    
    ![image.png](../images/image%2017.png)
    
4. Z is randomly assigned value from 0 to 9. This Z does not work so great, but is better than case3
    
    ![image.png](../images/image%2018.png)
    
5. Z is random permutatoin of 0-29. Similar to case2, this Z works great.
    
    ![image.png](../images/image%2019.png)
    
