

<div style="text-align: center;">
  <img src="./docs/images/ncut.svg" alt="NCUT" style="width: 80%; filter: brightness(60%) grayscale(100%);"/>
</div>

### [🌐Documentation](https://ncut-pytorch.readthedocs.io/) | [🤗HuggingFace Demo](https://huggingface.co/spaces/huzey/ncut-pytorch)


## Nyström Normalized Cut

Normalized Cut and spectral embedding, 100x faster than sklean implementation. 


https://github.com/user-attachments/assets/f0d40b1f-b8a5-4077-ab5f-e405f3ffb70f



<div align="center">
  Video: Ncut spectral embedding eigenvectors, on SAM features.
</div>


---

## Installation

<div style="text-align:">
    <pre><code class="language-shell">pip install -U ncut-pytorch</code></pre>
</div>


## Quick Start: plain Ncut


```py linenums="1"
import torch
from ncut_pytorch import Ncut

features = torch.rand(1960, 768)
eigvecs = Ncut(n_eig=20).fit_tranform(features)
  # (1960, 20)

from ncut_pytorch import kway_ncut
kway_eigvecs = kway_ncut(eigvecs)
cluster_assignment = kway_eigvecs.argmax(1)
cluster_centroids = kway_eigvecs.argmax(0)
```

## Quick Start: Ncut DINOv3 Predictor

```py linenums="1"
from ncut_pytorch.predictor import NcutDinov3Predictor
from PIL import Image

predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
predictor = predictor.to('cuda')

images = [Image.open(f"images/view_{i}.jpg") for i in range(4)]
predictor.set_images(images)

image = predictor.summary(n_segments=[10, 25, 50, 100], draw_border=True)

```

![summary](https://github.com/user-attachments/assets/a5d8a966-990b-4f6d-be10-abb00291bee2)



## Performance

`ncut_pytorch.Ncut` is $O(n)$ time complexity

`sklearn.SpectralEmbedding` is $O(n^2)$ complexity.

#### Setup:

CPU: Intel(R) Core(TM) i9-13900K CPU

RAM: 128 GiB

SYSTEM: Ubuntu 22.04.3 LTS

#### Run benchmark:


```shell
pytest unit_tests/bench_speed.py --benchmark-columns=mean,stddev --benchmark-sort=mean
```

#### Results

```
------------- benchmark 'ncut-pytorch (CPU) vs sklearn': 8 tests ------------
Name (time in ms)                        Mean                StdDev          
-----------------------------------------------------------------------------
test_ncut_cpu_100_data_10_eig          2.5536 (1.0)          0.2782 (1.0)    
test_sklearn_100_data_10_eig           4.0913 (1.60)         1.6749 (6.02)   
test_ncut_cpu_300_data_10_eig          4.9034 (1.92)         1.6575 (5.96)   
test_sklearn_300_data_10_eig          10.1861 (3.99)         3.8870 (13.97)  
test_ncut_cpu_1000_data_10_eig        11.1968 (4.38)         1.7070 (6.13)   
test_ncut_cpu_3000_data_10_eig        38.6101 (15.12)        1.6379 (5.89)   
test_sklearn_1000_data_10_eig        193.5934 (75.81)        8.1933 (29.45)  
test_sklearn_3000_data_10_eig      1,246.4295 (488.11)   1,047.0191 (>1000.0)
-----------------------------------------------------------------------------
```
```
------------- benchmark 'ncut-pytorch (GPU) n_data': 4 tests -------------
Name (time in ms)                         Mean            StdDev          
--------------------------------------------------------------------------
test_ncut_gpu_100_data_10_eig           2.9564 (1.0)      0.1816 (1.0)    
test_ncut_gpu_1000_data_10_eig          4.6938 (1.59)     0.3933 (2.17)   
test_ncut_gpu_100000_data_10_eig      396.9994 (134.29)   3.6202 (19.93)  
test_ncut_gpu_1000000_data_10_eig     798.4598 (270.08)   1.5704 (8.65)   
--------------------------------------------------------------------------
```
```
------------- benchmark 'ncut-pytorch (GPU) n_eig': 3 tests --------------
Name (time in ms)                         Mean            StdDev          
--------------------------------------------------------------------------
test_ncut_gpu_10000_data_10_eig        67.9607 (1.0)      4.0902 (10.76)  
test_ncut_gpu_10000_data_100_eig       74.0033 (1.09)     0.7856 (2.07)   
test_ncut_gpu_10000_data_1000_eig     179.8690 (2.65)     0.3801 (1.0)    
--------------------------------------------------------------------------
```

`ncut-pytorch.Ncut` is $O(1)$ space complexity

#### Run benchmark:

```shell
python unit_tests/bench_memory.py
```

#### Results:

```
+---------------+------------------------+
| Data Points   |   Peak GPU Memory (MB) |
+===============+========================+
| 1,000         |                   8.14 |
+---------------+------------------------+
| 10,000        |                   0.1  |
+---------------+------------------------+
| 100,000       |                   0.39 |
+---------------+------------------------+
| 1,000,000     |                   0.39 |
+---------------+------------------------+
```



## Citation

```
@misc{yang2024alignedcutvisualconceptsdiscovery,
      title={AlignedCut: Visual Concepts Discovery on Brain-Guided Universal Feature Space}, 
      author={Huzheng Yang and James Gee and Jianbo Shi},
      year={2024},
      eprint={2406.18344},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.18344}, 
}
```
