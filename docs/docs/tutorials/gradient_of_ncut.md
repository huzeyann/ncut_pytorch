# Gradient Computation

In our PyTorch implementation of Normalized Cut (NCUT), gradient computation is handled seamlessly by [PyTorch autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html), you can backpropagate gradients from the eigenvectors to the input features.

---

## With Nyström Approximation

This example demonstrates how to compute gradients when using the Nyström approximation. For gradient-sensitive workloads in 3.0.0, enable `exact_gradient=True`.

```python
from ncut_pytorch import Ncut
import torch

# Initialize features with gradient tracking
features = torch.randn(10000, 768)
features.requires_grad = True

# Compute NCUT with the exact eigensolver path for a stable backward pass
ncut = Ncut(n_eig=50, exact_gradient=True)
eigenvectors = ncut.fit_transform(features)
eigenvalues = ncut.eigval

# Define a loss function (e.g., sum of eigenvectors)
loss = eigenvectors.sum()

# Backpropagate
loss.backward()

# Access gradients
grad = features.grad
print(f"Gradient shape: {grad.shape}")
# Output: torch.Size([10000, 768])
```
