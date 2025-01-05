# %%
from ncut_pytorch import NCUT
import time
import torch

n = 4096

# for device in ['cuda:0', 'cpu']:
for device in ['cpu']:
    for n_data in [10000*n]:
    # for n_data in [1*n, 10*n, 100*n, 1000*n]:
# for device in ['cuda:0']:

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

# %%
