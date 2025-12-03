# Custom Model Integration

You can easily integrate your own vision model into the NCut package. NCut is model-agnostic and works with any backbone that produces spatial feature maps (e.g., ViT, ResNet, SAM, DINO).

To use a custom model, you need to wrap it in a class that follows a specific interface.

## Model Wrapper Requirements

Your custom model wrapper must be a `torch.nn.Module` and implement the `forward` method with the following signature:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: Input image tensor of shape (B, 3, H, W)
    # returns: Feature map of shape (B, C, H', W')
    ...
```

**Key Requirements:**

1.  **Input:** The `forward` method receives a batch of images as a tensor `(B, 3, H, W)`. These images are already transformed by the `transform` function you provide to `NcutVisionPredictor`.
2.  **Output:** The method must return a 4D tensor of shape `(B, C, H', W')`, where:
    -   `B`: Batch size
    -   `C`: Channel dimension (feature dimension)
    -   `H', W'`: Spatial dimensions of the feature map

## Example: Segment Anything Model (SAM)

Below is a complete example of how to wrap the [Segment Anything Model (SAM)](https://segment-anything.com/) for use with NCut.

### 1. Define the Wrapper

First, we define the wrapper class. This class loads the SAM model and extracts features from its image encoder.

```python linenums="1"
import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.modeling.sam import Sam

URL_DICT = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

class SAM(nn.Module):
    def __init__(self, model_cfg='vit_l', **kwargs):
        super().__init__(**kwargs)
        
        # Load SAM weights
        state_dict = torch.hub.load_state_dict_from_url(URL_DICT[model_cfg], map_location='cpu')
        sam: Sam = sam_model_registry[model_cfg]()
        sam.load_state_dict(state_dict)
        
        # Set to eval mode
        sam.eval()
        self.sam = sam

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SAM expects inputs to be 1024x1024
        if x.shape[-2:] != (1024, 1024):
            x = nn.functional.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
            
        # Extract features using the image encoder
        out = self.sam.image_encoder(x)  # (B, 256, 64, 64)
        
        # Normalize features (optional but recommended for cosine similarity)
        out = nn.functional.normalize(out, dim=1)
        
        return out  # (B, C, H, W)
```

### 2. Usage

Now you can use this wrapper with `NcutVisionPredictor`.

```python linenums="1"
from PIL import Image
import matplotlib.pyplot as plt
from ncut_pytorch.predictor.dino.transform import get_input_transform
from ncut_pytorch.predictor.vision_predictor import NcutVisionPredictor

# 1. Prepare input transform (resize to 1024 for SAM)
transform = get_input_transform(resize=1024)

# 2. Initialize the model and predictor
model = SAM(model_cfg='vit_l')
ncut_sam = NcutVisionPredictor(model, transform, batch_size=1)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
ncut_sam = ncut_sam.to(device)

# 3. Load images
# Replace with your image paths
images = [Image.open(f"images/pose/single_{i:04d}.jpg") for i in range(5)]

# 4. Run NCut
ncut_sam.set_images(images)

# 5. Visualize results
image_summary = ncut_sam.summary(n_segments=[10, 25, 50, 100])
plt.figure(figsize=(15, 10))
plt.imshow(image_summary)
plt.axis('off')
plt.show()
```
