

Our new Nystrom NCUT time and space complexity scales **linearly!** w.r.t number of nodes. 


| N images (64x64 pixels per image) | 1 | 10 | 100 | 1,000 | 10,000 | 
| --- | --- | --- | --- | --- | --- |
| **GPU (RTX4090) time** | 0.1 sec | 0.1 sec | 0.4 sec | 3.8 sec | 38.9 sec |
| **CPU (i9-13900K) time** | 0.3 sec | 0.7 sec | 3.5 sec | 42.3 sec | 426.8 sec |

<!-- | n_data | 10K | 100K | 1M | 10M |
| --- | --- | --- | --- | --- |
| **CPU (i9-13900K) time (sec)** | 0.495 | 7.410 | 32.775 | 291.610 |
| **GPU (RTX4090) time (sec)** | 0.049 | 0.443 | 2.092 | 19.226 | -->


```py linenums="1"
from ncut_pytorch import NCUT
import time
import torch

n = 4096

for device in ['cuda:0', 'cpu']:
    for n_data in [1*n, 10*n, 100*n, 1000*n, 10000*n]:

        input_feats = torch.rand(int(n_data), 3)
        input_feats = torch.nn.functional.normalize(input_feats, dim=1)

        start = time.time()
        eigenvectors, eigenvalues = NCUT(
            num_eig=50,
            num_sample=10000,
            knn=10,
            device=device,
            make_orthogonal=False,
            normalize_features=False,
        ).fit_transform(input_feats)
        end = time.time()
        print(device, n_data, "Nyström ncut time: {:.1f}".format(end - start))

```

The limitation is that the quality of Nystrom approximation will degrades as sample ratio goes low, therefore we use FPS sampling for better quality (see [How NCUT Works](how_ncut_works.md)).


---

## Speed-up Tricks

- Start with `num_sample=10000` for NCUT, `num_sample=300` for t-SNE. For larger number and diverse set of images, if the quality gets low, please consider increase t-SNE `num_sample=1000` and increase NCUT `num_sample=25000`.

<div style="text-align:left;">
    <pre><code>
<span style="color: #008000;"># fast, for small-scale</span>
X_3d, rgb = <span style="color: #FF6D00;">rgb_from_tsne_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">300</span>, perplexity=<span style="color: #A020F0;">150</span>)
X_3d, rgb = <span style="color: #FF6D00;">rgb_from_umap_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">300</span>, n_neighbors=<span style="color: #A020F0;">150</span>, min_dist=<span style="color: #A020F0;">0.1</span>)

<span style="color: #008000;"># balanced speed and quality, for medium-scale</span>
X_3d, rgb = <span style="color: #FF6D00;">rgb_from_tsne_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">1000</span>, perplexity=<span style="color: #A020F0;">250</span>)
X_3d, rgb = <span style="color: #FF6D00;">rgb_from_umap_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">1000</span>, n_neighbors=<span style="color: #A020F0;">250</span>, min_dist=<span style="color: #A020F0;">0.1</span>)

<span style="color: #008000;"># extreme quality, much slower</span>
X_3d, rgb = <span style="color: #FF6D00;">rgb_from_tsne_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">10000</span>, perplexity=<span style="color: #A020F0;">500</span>)
X_3d, rgb = <span style="color: #FF6D00;">rgb_from_umap_3d</span>(eigvecs, num_samples=<span style="color: #A020F0;">10000</span>, n_neighbors=<span style="color: #A020F0;">500</span>, min_dist=<span style="color: #A020F0;">0.1</span>)
    </code></pre>
</div>

<!-- ```py
# profile1: balanced quality and fast speed
eigenvectors, eigenvalues = NCUT(num_sample=10000).fit_transform(input_feats) 
X_3d, rgb = rgb_from_tsne_3d(eigvectors, num_sample=300) 
# profile2: good quality and good speed
eigenvectors, eigenvalues = NCUT(num_sample=25000).fit_transform(input_feats) 
X_3d, rgb = rgb_from_tsne_3d(eigvectors, num_sample=1000) 
``` -->

- decrease `num_sample` will **linearly** speed up NCUT and **quadratically** speed up t-SNE.


<!-- - do not apply post-hoc orthogonalization, if there's no need for strict orthogonality.

```py
eigenvectors, eigenvalues = NCUT(make_orthogonal=False).fit_transform(input_feats)
``` -->
