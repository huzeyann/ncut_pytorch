

<details>

<summary>
Click to expand full code

``` py
class Llama3(torch.nn.Module):
```

</summary>

```py linenums="1"
# %%
from llama import Llama
from typing import List, Optional
import torch
import os
#%%
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
torch.distributed.init_process_group("nccl")


class Llama3:
    def __init__(self):
        ckpt_dir = "/data/Meta-Llama-3-8B"
        tokenizer_path = "/data/Meta-Llama-3-8B/tokenizer.model"
        max_batch_size = 4
        max_seq_len: int = 2048

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    @torch.no_grad()
    def forward(self, prompts: List[str]):
        
        for prompt in prompts:
            out = self.generator.text_completion(prompts=[prompt], max_gen_len=1)

# %%
llama = Llama3()
# %%
def new_forward(
    self,
    x: torch.Tensor,
    start_pos: int,
    freqs_cis: torch.Tensor,
    mask: Optional[torch.Tensor],
):
    self.saved_attn = self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    h = x + self.saved_attn
    self.saved_ffn = self.feed_forward(self.ffn_norm(h))
    out = h + self.saved_ffn
    self.saved_output = out
    return out
# %%
setattr(llama.generator.model.layers[0].__class__, "forward", new_forward)
# %%
lines = """
1. The majestic giraffe, with its towering height and distinctive long neck, roams the savannas of Africa. These gentle giants use their elongated tongues to pluck leaves from the tallest trees, making them well-adapted to their environment. Their unique coat patterns, much like human fingerprints, are unique to each individual.
2. Penguins, the tuxedoed birds of the Antarctic, are expert swimmers and divers. These flightless seabirds rely on their dense, waterproof feathers and streamlined bodies to propel through icy waters in search of fish, krill, and other marine life. Their huddled colonies and amusing waddles make them a favorite among wildlife enthusiasts.
3. The mighty African elephant, the largest land mammal, is revered for its intelligence and strong family bonds. These gentle giants use their versatile trunks for various tasks, from drinking and feeding to communicating and greeting one another. Their massive ears and wrinkled skin make them an iconic symbol of the African wilderness.
4. The colorful and flamboyant peacock, native to Asia, is known for its stunning iridescent plumage. During mating season, the males fan out their magnificent train of feathers, adorned with intricate eye-like patterns, in an elaborate courtship display to attract potential mates, making them a true spectacle of nature.
5. The sleek and powerful cheetah, the fastest land animal, is built for speed and agility. With its distinctive black tear-like markings and slender body, this feline predator can reach top speeds of up to 70 mph during short bursts, allowing it to chase down its prey with remarkable precision.
6. The playful and intelligent dolphin, a highly social marine mammal, is known for its friendly demeanor and impressive acrobatic abilities. These aquatic creatures use echolocation to navigate and hunt, and their complex communication systems have long fascinated researchers studying their intricate social structures and cognitive abilities.
7. The majestic bald eagle, the national emblem of the United States, soars high above with its distinctive white head and tail feathers. These powerful raptors are skilled hunters, swooping down from great heights to catch fish and other prey with their sharp talons, making them an iconic symbol of strength and freedom.
8. The industrious beaver, nature's skilled engineers, are known for their remarkable ability to construct dams and lodges using their sharp incisors and webbed feet. These semiaquatic rodents play a crucial role in shaping their aquatic ecosystems, creating habitats for numerous other species while demonstrating their ingenuity and perseverance.
9. The vibrant and enchanting hummingbird, one of the smallest bird species, is a true marvel of nature. With their rapidly flapping wings and ability to hover in mid-air, these tiny feathered creatures are expert pollinators, flitting from flower to flower in search of nectar and playing a vital role in plant reproduction.
10. The majestic polar bear, the apex predator of the Arctic, is perfectly adapted to its icy environment. With its thick insulating fur and specialized paws for gripping the ice, this powerful carnivore relies on its exceptional hunting skills and keen senses to locate and capture seals, its primary prey, in the harsh Arctic landscape.
"""

# %%
llama.forward([lines])
# %%
attn_outputs, ffn_outputs, block_outputs = [], [], []
for i, blk in enumerate(llama.generator.model.layers):
    attn_outputs.append(blk.saved_attn)
    ffn_outputs.append(blk.saved_ffn)
    block_outputs.append(blk.saved_output)
attn_outputs = torch.stack(attn_outputs).float().detach().cpu()
ffn_outputs = torch.stack(ffn_outputs).float().detach().cpu()
block_outputs = torch.stack(block_outputs).float().detach().cpu()
print(attn_outputs.shape, ffn_outputs.shape, block_outputs.shape)
# %%
prompt_tokens = llama.generator.tokenizer.encode(lines, bos=True, eos=False)
token_texts = [llama.generator.tokenizer.decode([x]) for x in prompt_tokens]
# %%

# %%
torch.save((attn_outputs, ffn_outputs, block_outputs, token_texts), "/tmp/llama3_features.pt")
exit(0)

# %%
import torch
attn_outputs, ffn_outputs, block_outputs, token_texts = torch.load("/tmp/llama3_features.pt")
from ncut_pytorch import NCUT, rgb_from_tsne_3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_one_ax(token_texts, rgb, ax, fig, title, num_lines=5):
    # Define the colors
    # fill nan with 0
    rgb = np.nan_to_num(rgb)
    colors = [mcolors.rgb2hex(rgb[i]) for i in range(len(token_texts))]

    # Split the sentence into words
    words = token_texts


    y_pos = 0.9
    x_pos = 0.0
    max_word_length = max(len(word) for word in words)
    count = 0
    for word, color in zip(words, colors):
        if word == '<|begin_of_text|>':
            word = '<SoT>'
            y_pos -= 0.05
            x_pos = 0.0

        
        text_color = 'black' if sum(mcolors.hex2color(color)) > 1.3 else 'white'  # Choose text color based on background color
        # text_color = 'black'
        txt = ax.text(x_pos, y_pos, word, color=text_color, fontsize=12, bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2))
        txt_width = txt.get_window_extent().width / (fig.dpi * fig.get_size_inches()[0])  # Calculate the width of the text in inches
        
        x_pos += txt_width * 1.1 + 0.01  # Adjust the spacing between words
        
        if x_pos > 0.97:
            y_pos -= 0.2
            x_pos = 0.0
            count += 1
            if count >= num_lines:
                break
        # break
            
    # Remove the axis ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_title(title, fontsize=14)


for i_layer in range(12):

    attn_eig, _ = NCUT(num_eig=20).fit_transform(
        attn_outputs[i_layer].reshape(-1, attn_outputs[i_layer].shape[-1])
    )
    _, attn_rgb = rgb_from_tsne_3d(attn_eig, seed=42)
    mlp_eig, _ = NCUT(num_eig=20).fit_transform(
        ffn_outputs[i_layer].reshape(-1, ffn_outputs[i_layer].shape[-1])
    )
    _, mlp_rgb = rgb_from_tsne_3d(mlp_eig, seed=42)
    block_eig, _ = NCUT(num_eig=20).fit_transform(
        block_outputs[i_layer].reshape(-1, block_outputs[i_layer].shape[-1])
    )
    _, block_rgb = rgb_from_tsne_3d(block_eig, seed=42)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 5))
    plot_one_ax(token_texts, attn_rgb.numpy(), axs[0], fig, "attention layer output")
    plot_one_ax(token_texts, mlp_rgb.numpy(), axs[1], fig, "MLP layer output")
    plot_one_ax(token_texts, block_rgb.numpy(), axs[2], fig, "sum of residual stream")
    
    plt.suptitle(f"Llama3 layer {i_layer} NCUT spectral-tSNE", fontsize=16)
    plt.tight_layout()
    # plt.show()
    
    save_dir = "/workspace/output/gallery/llama3"
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/llama3_layer_{i_layer}.jpg", bbox_inches="tight")
    plt.close()
    
```

</details>


<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_0.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_1.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_2.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_3.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_4.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_5.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_6.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_7.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_8.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_9.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_10.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_11.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_12.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_13.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_14.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_15.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_16.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_17.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_18.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_19.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_20.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_21.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_22.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_23.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_24.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_25.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_26.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_27.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_28.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_29.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_30.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="/images/gallery_gallery_llama3/llama3_layer_31.jpg" style="width:100%;">
</div>
