# Hierarchy in NCut

## Understanding Hierarchical Segmentation

The k-way normalized cut algorithm naturally produces hierarchical segmentations by varying the number of clusters `k`. This hierarchical property allows us to control the granularity of segmentation from coarse to fine-grained levels, making it a powerful tool for multi-scale image understanding.

## How to use it in a few lines

``` py
from ncut_pytorch.predictor import NcutDinov3Predictor
from PIL import Image

predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
predictor = predictor.to('cuda')

images = [Image.open(f"images/view_{i}.jpg") for i in range(4)]
predictor.set_images(images)

image = predictor.summary(n_segments=[10, 25, 50, 100], draw_border=True)
display(image)
```


## Interactive Demo (Video)

The video below demonstrates interactive hierarchical NCut segmentation. It shows how the relationships between different regions in different K, and how user interactions can select coherent regions at different hierarchy levels.

<video src="../images/hierarchy_ncut.mp4" controls playsinline muted loop style="width:100%; max-width:960px; height:auto; display:block; margin:0 auto;"></video>

If the video does not render in your browser, you can download it directly: [Download hierarchy_ncut.mp4](../images/hierarchy_ncut.mp4)

## DINO Features Hierarchical Segmentation

![DINO Features K-Way Segmentation](../images/dino_feature.png)


**Key Observation**: DINO features excel at semantic segmentation, maintaining object coherence even at fine granularities. The segmentation respects semantic boundaries rather than just visual edges.

## SAM Features Hierarchical Segmentation

![SAM Features K-Way Segmentation](../images/sam_feature.png)

**Key Observation**: SAM features excel at boundary detection and part-level segmentation. The hierarchy moves from parts to sub-parts to surface details, making it ideal for applications requiring precise spatial localization.

## NCut Hierarchy vs Attention Maps

![NCut vs Attention Comparison](../images/ncut_attention.png)

The comparison above reveals a fundamental difference between NCut hierarchical segmentation and traditional attention map visualization.

Rather than competing methods, NCut hierarchy and attention maps serve **complementary purposes**:

- **Attention maps** are excellent for understanding *how* a neural network makes decisions and *which* features it considers important
- **NCut hierarchy** excels at *what* semantic structures exist in the data and *how* to segment them for practical applications

## Custom Implementations

```python
# Generate hierarchical segmentation
from ncut_pytorch import NCUT

k_values = [16, 32, 64, 128, 256]
hierarchical_segments = {}

for k in k_values:
    eigenvectors, eigenvalues = NCUT(
        num_eig=k,
        num_sample=10000,
    ).fit_transform(features)
    
    hierarchical_segments[k] = eigenvectors

# Access different hierarchy levels
coarse_segmentation = hierarchical_segments[16]
fine_segmentation = hierarchical_segments[256]
```

This hierarchical approach provides a powerful framework for understanding images at multiple scales, enabling applications that require both global context and local precision.
