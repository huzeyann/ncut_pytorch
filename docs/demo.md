---
hide:
  - navigation
  - toc
---

#

<iframe
	src="https://fun-quetzal-whole.ngrok-free.app/"
	frameborder="0"
	width="100%"
	height="1000"
></iframe>

## This demo is hosted at UPenn

Open this demo at 
<a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">
HuggingFace
</a>
<!-- <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #FF5733; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
🤗 HuggingFace Demo
</a> -->


---

## Hosting this demo software locally

Step 1. Install [Docker](https://www.docker.com/) and [Nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) plugin.

```shell
curl -fsSL https://get.docker.com -o get-docker.sh | sudo sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

Step 2. Run this docker container locally. 

```
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all \
	-e HF_ACCESS_TOKEN="YOUR_VALUE_HERE" \
	-e USE_HUGGINGFACE_ZEROGPU="false" \
	-e DOWNLOAD_ALL_MODELS_DATASETS="false" \
	registry.hf.space/huzey-ncut-pytorch:latest python app.py
```

`HF_ACCESS_TOKEN` is only needed if you need access to restricted models (Llama, SDv3), please see [Backbones](backbones.md).

Step 3. Use the printed out link to access your local demo.

```
...
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://some_link_here.gradio.live
```