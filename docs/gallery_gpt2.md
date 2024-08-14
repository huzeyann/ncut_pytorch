

<details>

<summary>
Click to expand full code

``` py
class GPT2(torch.nn.Module):
```

</summary>

```py linenums="1"

# %%
from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
# %%
from typing import Optional, Tuple, Union
import torch

def new_forward(
    self,
    hidden_states: Optional[Tuple[torch.FloatTensor]],
    layer_past: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = False,
    output_attentions: Optional[bool] = False,
) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        layer_past=layer_past,
        attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
    outputs = attn_outputs[1:]
    # residual connection
    self.attn_output = attn_output.clone()
    hidden_states = attn_output + residual

    if encoder_hidden_states is not None:
        # add one self-attention block for cross-attention
        if not hasattr(self, "crossattention"):
            raise ValueError(
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                "cross-attention layers by setting `config.add_cross_attention=True`"
            )
        residual = hidden_states
        hidden_states = self.ln_cross_attn(hidden_states)
        cross_attn_outputs = self.crossattention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )
        attn_output = cross_attn_outputs[0]
        # residual connection
        hidden_states = residual + attn_output
        outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    # residual connection
    self.mlp_output = feed_forward_hidden_states.clone()
    hidden_states = residual + feed_forward_hidden_states

    if use_cache:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,) + outputs[1:]

    self.block_output = hidden_states.clone()
    return outputs  # hidden_states, present, (attentions, cross_attentions)
# %%
setattr(model.h[0].__class__, "forward", new_forward)
# %%
text = """
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
encoded_input = tokenizer(text, return_tensors='pt')
with torch.no_grad():
    output = model(**encoded_input, output_hidden_states=True)
# %%
attn_outputs, mlp_outputs, block_outputs = [], [], []
for i, blk in enumerate(model.h):
    attn_outputs.append(blk.attn_output)
    mlp_outputs.append(blk.mlp_output)
    block_outputs.append(blk.block_output)
attn_outputs = torch.stack(attn_outputs)
mlp_outputs = torch.stack(mlp_outputs)
block_outputs = torch.stack(block_outputs)
print(attn_outputs.shape, mlp_outputs.shape, block_outputs.shape)
# %%
token_ids = encoded_input['input_ids']
token_texts = [tokenizer.decode([token_id]) for token_id in token_ids[0]]

# %%
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

# %%
from ncut_pytorch import NCUT, rgb_from_tsne_3d

for i_layer in range(12):

    attn_eig, _ = NCUT(num_eig=20, device="cuda:0").fit_transform(
        attn_outputs[i_layer].reshape(-1, attn_outputs[i_layer].shape[-1])
    )
    _, attn_rgb = rgb_from_tsne_3d(attn_eig, device="cuda:0", seed=42)
    mlp_eig, _ = NCUT(num_eig=20, device="cuda:0").fit_transform(
        mlp_outputs[i_layer].reshape(-1, mlp_outputs[i_layer].shape[-1])
    )
    _, mlp_rgb = rgb_from_tsne_3d(mlp_eig, device="cuda:0", seed=42)
    block_eig, _ = NCUT(num_eig=20, device="cuda:0").fit_transform(
        block_outputs[i_layer].reshape(-1, block_outputs[i_layer].shape[-1])
    )
    _, block_rgb = rgb_from_tsne_3d(block_eig, device="cuda:0", seed=42)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 5))
    plot_one_ax(token_texts, attn_rgb.numpy(), axs[0], fig, "attention layer output")
    plot_one_ax(token_texts, mlp_rgb.numpy(), axs[1], fig, "MLP layer output")
    plot_one_ax(token_texts, block_rgb.numpy(), axs[2], fig, "sum of residual stream")
    
    plt.suptitle(f"GPT-2 layer {i_layer} NCUT spectral-tSNE", fontsize=16)
    plt.tight_layout()
    # plt.show()
    
    save_dir = "/workspace/output/gallery/gpt2"
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/gpt2_layer_{i_layer}.jpg", bbox_inches="tight")
    plt.close()
    
# %%
```

</details>


<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_0.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_1.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_2.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_3.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_4.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_5.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_6.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_7.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_8.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_9.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_10.jpg" style="width:100%;">
</div>
<div style="text-align: center;">
    <img src="../images/gallery/gpt2/gpt2_layer_11.jpg" style="width:100%;">
</div>
