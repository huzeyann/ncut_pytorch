# Tutorial 3: Hierarchy in NCut

## Understanding Hierarchical Segmentation

The K-way Normalized Cut (NCut) algorithm inherently produces hierarchical segmentations by varying the number of clusters, `k`. This property allows for control over segmentation granularity, ranging from coarse global structures to fine-grained details, making it a powerful tool for multi-scale image understanding.

## Quick Start

You can generate hierarchical segmentations in just a few lines of code using the `NcutDinov3Predictor`.

```python
from ncut_pytorch.predictor import NcutDinov3Predictor
from PIL import Image

# Initialize the predictor
predictor = NcutDinov3Predictor(model_cfg="dinov3_vitl16")
predictor = predictor.to('cuda')

# Load image
images = [Image.open("example.jpg")]
predictor.set_images(images)

# Generate summary showing multiple hierarchy levels
image = predictor.summary(n_segments=[10, 25, 50, 100], draw_border=True)
image.save("summary.jpg")
```

## Interactive Demo

The video below demonstrates interactive hierarchical NCut segmentation. It illustrates how regions relate across different values of `k` and how user interactions can select coherent regions at various hierarchy levels.

<video src="../images/tutorials_03_hierarchy_ncut/hierarchy_ncut.mp4" controls playsinline muted loop style="width:100%; max-width:960px; height:auto; display:block; margin:0 auto;"></video>

If the video does not render in your browser, you can download it directly: [Download hierarchy_ncut.mp4](../images/tutorials_03_hierarchy_ncut/hierarchy_ncut.mp4)

## Feature Comparison: DINO vs. SAM

### DINO Features: Semantic Coherence

![DINO Features K-Way Segmentation](../images/tutorials_03_hierarchy_ncut/dino_feature.png)

**Key Observation**: DINO features excel at semantic segmentation, maintaining object coherence even at fine granularities. The segmentation tends to respect semantic boundaries (e.g., separating a "dog" from "grass") rather than just visual edges.

### SAM Features: Part-Based Segmentation

![SAM Features K-Way Segmentation](../images/tutorials_03_hierarchy_ncut/sam_feature.png)

**Key Observation**: SAM (Segment Anything Model) features excel at boundary detection and part-level segmentation. The hierarchy typically progresses from whole objects to parts, sub-parts, and finally surface details, making it ideal for applications requiring precise spatial localization.

## NCut Hierarchy vs. Attention Maps

![NCut vs Attention Comparison](../images/tutorials_03_hierarchy_ncut/ncut_attention.png)

The comparison above reveals a fundamental difference between NCut hierarchical segmentation and traditional attention map visualization. Rather than being competing methods, they serve **complementary purposes**:

- **Attention maps** are excellent for understanding *how* a neural network makes decisions and *which* features it considers important.
- **NCut hierarchy** excels at identifying *what* semantic structures exist in the data and *how* to segment them for practical applications.

## Custom Implementations

For more control, you can implement the hierarchical segmentation loop manually using the `Ncut` class.

```python
import torch
from ncut_pytorch import Ncut

# Assume 'features' is your input tensor of shape (N, D)
# features = ...

k_values = [16, 32, 64, 128, 256]
hierarchical_segments = {}

for k in k_values:
    # Initialize NCut with k eigenvectors
    ncut = Ncut(
        n_eig=k,
        n_sample=10000,  # Number of samples for Nystrom approximation
    )
    
    # Fit and transform to get eigenvectors
    eigenvectors = ncut.fit_transform(features)
    
    # Store results
    hierarchical_segments[k] = eigenvectors

# Access different hierarchy levels
coarse_segmentation = hierarchical_segments[16]
fine_segmentation = hierarchical_segments[256]
```

This hierarchical approach provides a robust framework for understanding images at multiple scales, enabling applications that require both global context and local precision.
