# %%
from signal import pause
import torch
from ncut_pytorch import Ncut
import time
import matplotlib.pyplot as plt
# %%
# n_datas = [1000, 10000, 100000, 1000000, 2000000, 3000000, 4000000, 5000000]
n_datas = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 100000]
times = []
for n_data in n_datas:
    data = torch.randn(n_data, 768)
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

# %%