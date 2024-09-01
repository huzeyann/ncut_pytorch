# %%
from ncut_pytorch.backbone import load_model
import torch
model = load_model(model_name="SAM(sam_vit_b)").cuda()
images = torch.rand(1, 3, 1024, 1024).cuda()
out_dict = model(images)

for node_name in out_dict.keys():
    print(f"node_name: `{node_name}`, num_layers: {len(out_dict[node_name])}")
    print(f"layer_0 shape: {out_dict[node_name][0].shape}")

# %%
from ncut_pytorch.backbone_text import load_text_model
import torch
model = load_text_model(model_name="gpt2").cuda()
out_dict = model("I know this sky loves you")

# out_dict = {node_name: List[layer_0, layer_1, ...]}

for node_name in out_dict.keys():
    if isinstance(out_dict[node_name][0], torch.Tensor):
        print(f"node_name: `{node_name}`, num_layers: {len(out_dict[node_name])}")
        print(f"layer_0 shape: {out_dict[node_name][0].shape}")
    else:
        print(f"token_texts: {out_dict[node_name]}")
# %%
