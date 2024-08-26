## Accessing the Gradient of NCUT with Functional API

In our PyTorch implementation of NCUT, gradient is handled by [PyTorch autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html).

By defaults, the class API `NCUT.fit_transform()` does not keep the gradient. It's possible to access the gradient with [functional API](api_reference.md/#ncut-functional-api). When accessing the gradient, please use the original NCUT without Nystrom approximation. Gradient with Nystrom approximation is not recommended due to the sub-sampling, if you need gradient in the training loop, please use mini-batch to reduce the graph size. 


---

This example use NCUT without Nystrom approximation, and access gradient of eigenvectors.

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