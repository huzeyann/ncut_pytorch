# %%
import torch
from ncut_pytorch.predictor import NcutDinoPredictorFeatUp, NcutDinoPredictorSR
from ncut_pytorch.predictor.dino_predictor import NcutDinoPredictor
from PIL import Image
# %%

predictor = NcutDinoPredictorSR(input_size=512, dtype=torch.float16, batch_size=8)
predictor = predictor.to('cuda')

default_images = ['/images/image_0.jpg', '/images/image_1.jpg', '/images/guitar_ego.jpg'] * 1

images = [Image.open(image_path) for image_path in default_images]
predictor.set_images(images)

# %%
segments = predictor.generate(n_segment=32)
color = predictor.color_discrete(segments, draw_border=True)
# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i in range(3):
    axes[0, i].imshow(images[i])
    axes[1, i].imshow(color[i])
    combined = Image.blend(images[i], color[i], 0.5)
    axes[2, i].imshow(combined)

plt.show()
# %%
