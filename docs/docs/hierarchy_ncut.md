# Hierarchy in NCut

## Understanding Hierarchical Segmentation

The k-way normalized cut algorithm naturally produces hierarchical segmentations by varying the number of clusters `k`. This hierarchical property allows us to control the granularity of segmentation from coarse to fine-grained levels, making it a powerful tool for multi-scale image understanding.

## Interactive Demo (Video)

The video below demonstrates interactive hierarchical NCut segmentation. It shows how the relationships between different regions in different K, and how user interactions can select coherent regions at different hierarchy levels.

<video src="../images/hierarchy_ncut.mp4" controls playsinline muted loop style="width:100%; max-width:960px; height:auto; display:block; margin:0 auto;"></video>

If the video does not render in your browser, you can download it directly: [Download hierarchy_ncut.mp4](../images/hierarchy_ncut.mp4)

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

'''

## DINO Features Hierarchical Segmentation

![DINO Features K-Way Segmentation](../images/dino_feature.png)

The image above demonstrates hierarchical segmentation using DINO (self-supervised vision transformer) features across different k values:

- **k=16**: Captures the most prominent semantic regions. The segmentation focuses on major objects and background areas. Notice how the person and primary objects are separated from the background with clear boundaries.

- **k=32**: Introduces finer semantic distinctions. Body parts (torso, arms, legs) begin to separate, and background elements start to differentiate (floor, walls, furniture).

- **k=64**: Reveals mid-level details. Clothing items, instruments, and distinct background regions are now segmented separately. The guitar, for instance, shows clearer part-level segmentation.

- **k=128**: Captures fine-grained semantic details. Individual clothing folds, facial features, and instrument components become distinct segments.

- **k=256**: Produces the finest segmentation level. Even subtle texture variations and small objects are segmented, revealing the full hierarchical structure of the scene.

**Key Observation**: DINO features excel at semantic segmentation, maintaining object coherence even at fine granularities. The segmentation respects semantic boundaries rather than just visual edges.

## SAM Features Hierarchical Segmentation

![SAM Features K-Way Segmentation](../images/sam_feature.png)

The SAM (Segment Anything Model) features produce a different hierarchical structure:

- **k=16**: Provides basic part-level segmentation. Unlike DINO, SAM immediately separates objects into more granular parts, focusing on edges and boundaries.

- **k=32**: Shows strong edge-awareness. Small objects and detailed boundaries become visible earlier than with DINO features.

- **k=64**: Captures intricate surface details and texture boundaries. Notice how the guitar strings, keyboard keys, and other fine structures are well-defined.

- **k=128-256**: Produces extremely detailed segmentation maps. Every surface variation, shadow, and texture change gets its own segment.

**Key Observation**: SAM features excel at boundary detection and part-level segmentation. The hierarchy moves from parts to sub-parts to surface details, making it ideal for applications requiring precise spatial localization.

## Comparing DINO vs SAM Hierarchies

| Aspect | DINO Features | SAM Features |
|--------|---------------|--------------|
| **Semantic Understanding** | Strong semantic grouping, respects object identity | Focus on boundaries and parts |
| **Coarse Levels (k=16-32)** | Whole objects and major regions | Object parts and components |
| **Fine Levels (k=128-256)** | Semantic sub-regions | Surface details and textures |
| **Best For** | Object-centric tasks, scene understanding | Edge detection, instance segmentation |

## NCut Hierarchy vs Attention Maps

![NCut vs Attention Comparison](../images/ncut_attention.png)

The comparison above reveals a fundamental difference between NCut hierarchical segmentation and traditional attention map visualization.

### Attention Maps: Point-Based Focus

Attention maps (right column) are designed to answer the question: **"What regions are related to this specific point?"**

- **Behavior**: Given a query point (marked with red cross), the attention mechanism highlights pixels that have high attention weights with respect to that point
- **Information Type**: Provides localized, point-specific relationship information
- **Visualization**: Typically shown as heatmaps where brightness indicates attention strength
- **Limitation**: 
  - Only shows relationships relative to a single query point
  - Requires multiple queries to understand the full scene structure
  - Attention can be diffuse and hard to interpret semantically
  - Does not provide explicit segmentation boundaries

### NCut Hierarchy: Global Semantic Structure

NCut hierarchy (middle column) provides a fundamentally different view: **"What are the complete semantic structures in the scene?"**

- **Behavior**: Automatically discovers and segments all semantically coherent regions simultaneously
- **Information Type**: Provides global, scene-level structural decomposition
- **Visualization**: Clear segmentation masks with distinct colors for each region
- **Advantages**:
  - Reveals the entire scene structure without requiring query points
  - Provides explicit, interpretable semantic boundaries
  - Hierarchical nature allows multi-scale understanding
  - Segments are spatially coherent and semantically meaningful

### When to Use Each Approach

| Task | Attention Maps | NCut Hierarchy |
|------|----------------|----------------|
| **Understanding model behavior** | ✓ Shows what the model "looks at" | ✗ Not directly related to model internals |
| **Semantic segmentation** | ✗ Not designed for this | ✓ Explicit semantic boundaries |
| **Interactive exploration** | ✓ Query-based exploration | ✓ Complete scene structure |
| **Downstream applications** | △ Requires post-processing | ✓ Ready-to-use segmentation masks |
| **Interpretability** | △ Can be noisy and diffuse | ✓ Clear, interpretable regions |
| **Computational efficiency** | ✓ Fast for single query | ✓ One computation for full scene |

### Complementary Use Cases

Rather than competing methods, NCut hierarchy and attention maps serve **complementary purposes**:

- **Attention maps** are excellent for understanding *how* a neural network makes decisions and *which* features it considers important
- **NCut hierarchy** excels at *what* semantic structures exist in the data and *how* to segment them for practical applications

## Implementation Tips

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
