# %%
from scipy.stats import f
import torch
from ncut_pytorch import Ncut
import time
import matplotlib.pyplot as plt
# %%
# n_datas = [1000, 10000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000]
n_datas = [1e4, 1e5, 1e6, 1e7]
times = []
for n_data in n_datas:
    data = torch.randn(int(n_data), 768)
    start_time = time.time()
    ncut = Ncut(n_eig=100, device='cpu')
    ncut.fit_transform(data)
    end_time = time.time()
    times.append(end_time - start_time)
    print(f'{n_data} data: {end_time - start_time} s')
#%%
plt.plot(n_datas, times)
plt.xlabel('Number of data')
plt.ylabel('Time (s)')
plt.show()
# %%
from sklearn.manifold import SpectralEmbedding
from ncut_pytorch import Ncut
# %%
configs = [
    # n_data, n_eig
    (1e2, 50),
    (1e3, 50),
    (1e4, 50),
    (1e5, 50),
]
def run_speed_test(config, method='ncut'):
    n_data, n_eig = config
    run_times = []
    for _ in range(10):
        data = torch.randn(int(n_data), 768)
        start_time = time.time()
        if method == 'ncut':
            ncut = Ncut(n_eig=n_eig, device='cpu')
            ncut.fit_transform(data)
        if method == 'sklearn':
            spectral_embedding = SpectralEmbedding(n_components=n_eig, affinity='rbf')
            spectral_embedding.fit_transform(data)
        end_time = time.time()
        run_times.append(end_time - start_time)
    mean_time = sum(run_times) / len(run_times)
    std_time = torch.tensor(run_times).std().item()
    return mean_time, std_time
# %%
run_speed_test(configs[0], method='ncut')
# %%
run_speed_test(configs[-1], method='sklearn')
# %%