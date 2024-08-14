

Our new Nystrom NCUT time and space complexity scales **linearly** w.r.t number of nodes. 

The limitation is that the quality of Nystrom approximation will degrades as sample ratio goes lower.


| n_data | 10K | 100K | 1M | 10M |
| --- | --- | --- | --- | --- |
| **CPU time (sec)** | 0.495 | 7.410 | 32.775 | 291.610 |
| **GPU time (sec)** | 0.049 | 0.443 | 2.092 | 19.226 |


```py linenums="1"
from ncut_pytorch import NCUT
import time
import torch

for device in ['cuda:0', 'cpu']:
    for n_data in [1e4, 1e5, 1e6, 1e7]:

        input_feats = torch.rand(int(n_data), 3)
        input_feats = torch.nn.functional.normalize(input_feats, dim=1)

        start = time.time()
        eigenvectors, eigenvalues = NCUT(
            num_eig=50,
            num_sample=30000,
            knn=10,
            device=device,
            make_orthogonal=False,
            normalize_features=False,
        ).fit_transform(input_feats)
        end = time.time()
        print(device, n_data, "Nyström ncut time: {:.3f}".format(end - start))

```