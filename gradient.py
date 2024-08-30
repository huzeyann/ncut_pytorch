# %%
from ncut_pytorch import NCUT
import torch

features = torch.randn(10000, 768)
features.requires_grad = True
eigvectors, eigvalues = NCUT(num_eig=50, num_sample=1000).fit_transform(features)
loss = eigvectors.sum()
loss.backward()
grad = features.grad
print(grad.shape)
# torch.Size([100, 768])

# %%
