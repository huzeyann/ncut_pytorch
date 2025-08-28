# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from ncut_pytorch.predictor.dino.dinov3 import Dinov3Backbone
from ncut_pytorch.predictor import NcutDinov3Predictor
from PIL import Image


dinov3_predictor = NcutDinov3Predictor(input_size=(2048, 2048), model_cfg="dinov3_vitl16")
dinov3_predictor = dinov3_predictor.to('cuda')
dinov3_predictor.set_images([Image.open("images/pose/single_0000.jpg")])
# %%

from torch import Tensor

def _save_attn_fn():
    
    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        
        self.attn_q = q
        self.attn_k = k
        self.attn_v = v
        
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])
    
    return compute_attention
# %%

def apply_masked_attention_to_model(model):
    """
    Replace the compute_attention method in all attention layers with the masked version
    """
    new_compute_attention = _save_attn_fn()
    
    # Apply to all transformer blocks
    for block in model.model.blocks:
        if hasattr(block, 'attn'):
            # Replace the compute_attention method
            import types
            block.attn.compute_attention = types.MethodType(new_compute_attention, block.attn)
    
    return model

# %%
dinov3_predictor.model = apply_masked_attention_to_model(dinov3_predictor.model)
# %%
images = [Image.open(f"images/egothink/single_{i:04d}.jpg") for i in range(1)]
dinov3_predictor.set_images(images)
# %%
dinov3_predictor.model.model.blocks[-1].attn.attn_q.shape
# %%
masks = dinov3_predictor.preview([300, 579], 0)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# %%
%matplotlib widget
# %%
def blend_mask(img, mask, alpha=0.8):
    mask = mask.resize((img.width, img.height), resample=Image.Resampling.NEAREST)
    img = np.array(img).astype(np.float32)
    mask = np.array(mask).astype(np.float32)
    blend = img * (1 - alpha) + mask * alpha
    h, w = blend.shape[:2]
    blend = blend.reshape(h*w, 3)
    mask = mask.reshape(h*w, 3)
    img = img.reshape(h*w, 3)
    blend[mask[:, 0] == 255] = img[mask[:, 0] == 255]
    blend = blend.reshape(h, w, 3)
    blend = blend.astype(np.uint8)
    return Image.fromarray(blend)

def blend_mask2(img, mask, alpha=0.8):
    mask = mask.resize((img.width, img.height), resample=Image.Resampling.NEAREST)
    img = np.array(img).astype(np.float32)
    mask = np.array(mask).astype(np.float32)
    blend = img * (1 - alpha) + mask * alpha
    blend = blend.astype(np.uint8)
    return Image.fromarray(blend)

def plot_masks(img, masks, fig, ax):
    color_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    color_values = plt.get_cmap('plasma_r')(color_values)
    color_values = (color_values * 255).astype(np.uint8)

    ax.clear()
    idx = 0 # only one image
    mask = np.ones((*masks[0][0].shape, 3)) * 255
    mask = mask.astype(np.uint8)
    for i in range(len(color_values)):
        mask[masks[i][idx]] = color_values[i][:3]
    mask_img = Image.fromarray(mask)
    ax.imshow(blend_mask(img, mask_img, alpha=0.8))
    ax.set_axis_off()

import cv2
def draw_click_crosshair(image, x, y, size=50, thickness=20):
    img = np.array(image)
    # Convert coordinates to integers for OpenCV
    x_int, y_int = int(round(x)), int(round(y))
    # Draw vertical line
    img = cv2.line(img, (x_int, y_int - size), (x_int, y_int + size), (255, 0, 0), thickness)
    # Draw horizontal line
    img = cv2.line(img, (x_int - size, y_int), (x_int + size, y_int), (255, 0, 0), thickness)
    return Image.fromarray(img)

fig, axes = plt.subplots(1, 3, figsize=(8, 3))
axes = axes.flatten()
plot_masks(images[0], masks, fig, axes[0])


def on_click(event):
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return
    for idx, ax in enumerate(axes[:1]):
        if ax == event.inaxes:
            x, y = event.xdata, event.ydata
            print(x, y)
            # Recompute masks for all images, using the hovered coordinates and image index
            img = draw_click_crosshair(images[0], x, y)
            axes[0].imshow(img)
            masks_new = dinov3_predictor.preview([x, y], idx)
            plot_masks(images[0], masks_new, fig, axes[1])
            
            point_idx = dinov3_predictor._image_xy_to_tensor_index(np.array([images[0].size[1], images[0].size[0]]), np.array([128, 128]), np.array([[x, y]]), np.array([0]))[0]
            print(point_idx)
            num_reg = 5
            q = dinov3_predictor.model.model.blocks[-1].attn.attn_q[0, :, num_reg:]
            k = dinov3_predictor.model.model.blocks[-1].attn.attn_k[0, :, num_reg:]
            v = dinov3_predictor.model.model.blocks[-1].attn.attn_v[0, :, num_reg:]
            i_head = 0
            q = q[i_head, point_idx]
            k = k[i_head, :]
            attn = q @ k.T
            attn = attn / 128 ** 0.5
            attn = attn.softmax(dim=-1)
            attn = attn.reshape(128, 128)
            
            # Create heatmap and blend with original image
            attn_np = attn.cpu().numpy()
            
            # Create a figure for the heatmap (this will be temporary)
            temp_fig = plt.figure(figsize=(8, 8))
            temp_ax = temp_fig.add_subplot(111)
            sns.heatmap(attn_np, cmap='hot', ax=temp_ax, cbar=False, square=True)
            temp_ax.set_axis_off()
            
            # Remove all padding and margins
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            temp_ax.margins(0, 0)
            temp_fig.tight_layout(pad=0)
            
            # Convert the heatmap to an image array
            temp_fig.canvas.draw()
            # Get the RGBA buffer and convert to RGB
            buffer = temp_fig.canvas.buffer_rgba()
            heatmap_data = np.asarray(buffer)
            heatmap_data = heatmap_data[:, :, :3]  # Remove alpha channel
            plt.close(temp_fig)
            
            # Convert to PIL Image and resize to match original image
            heatmap_img = Image.fromarray(heatmap_data)
            heatmap_img = heatmap_img.resize((images[0].width, images[0].height), Image.Resampling.LANCZOS)
            
            # Blend with original image
            axes[2].clear()
            blended = blend_mask2(images[0], heatmap_img, alpha=0.5)
            axes[2].imshow(blended)
            axes[2].set_axis_off()

            fig.canvas.draw_idle()
            break
        
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()

# %%
