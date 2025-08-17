# %%
from ncut_pytorch.predictor import NcutDinov3Predictor
import time
predictor = NcutDinov3Predictor((512, 512), "dinov3_vits16")
# %%
predictor = predictor.to('mps')
from PIL import Image


images = [Image.open("images/view_0.jpg"), Image.open("images/view_1.jpg"), Image.open("images/view_2.jpg")
            , Image.open("images/view_3.jpg"), Image.open("images/view_ego.jpg"), Image.open("images/image2.jpg")]
start_time = time.time()
predictor.set_images(images)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
# %%
predictor.predictor.refresh_color_palette()
# %%
image = predictor.summary(draw_border=True)
# %%
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(image)
ax.axis('off')
ax.set_title('Dinov3 vitl16')
plt.show()
# %%