# Gradient of Ncut

In our PyTorch implementation of NCUT, gradient is handled by [PyTorch autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html).

---

This example use NCUT **with** Nystrom approximation, and access gradient of eigenvectors.

```py linenums="1"
from ncut_pytorch import NCUT
import torch

features = torch.randn(10000, 768)
features.requires_grad = True
eigvectors, eigvalues = NCUT(num_eig=50, num_sample=1000).fit_transform(features)
loss = eigvectors.sum()
loss.backward()
grad = features.grad
print(grad.shape)
# torch.Size([10000, 768])
```

---

This example use NCUT **without** Nystrom approximation, and access gradient of eigenvectors.

```py linenums="1"
from ncut_pytorch import ncut, affinity_from_features  # use functional API
import torch

features = torch.randn(100, 768)
features.requires_grad = True
affinity = affinity_from_features(features)
eigvectors, eigvalues = ncut(affinity, num_eig=50)
loss = eigvectors.sum()
loss.backward()
grad = features.grad
print(grad.shape)
# torch.Size([100, 768])
```