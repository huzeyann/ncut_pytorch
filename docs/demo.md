---
hide:
  - navigation
  - toc
---

#

<style>
	.copy-code {
		display: inline-block;
		white-space: nowrap;
		border: 1px solid #ccc;
		padding: 2px 4px;
		cursor: pointer;
		user-select: none;
		position: relative;
	}
	.tooltip {
		position: absolute;
		bottom: 100%;
		left: 50%;
		transform: translateX(-50%);
		padding: 5px;
		border-radius: 4px;
		white-space: nowrap;
		opacity: 0;
		transition: opacity 0.3s;
		pointer-events: none;
		z-index: 1000;
	}
	.tooltip.show {
		opacity: 1;
	}
</style>

This demo is hosted at UPenn 

<a href="https://fun-quetzal-whole.ngrok-free.app/" target="_blank" >Link1</a> 

<a href="https://click-on-the-smile-on-about-page-to-unlock-secret.loca.lt/" target="_blank" >Link2</a> **password is:** 
<span class="copy-code" onclick="copyToClipboard('158.130.50.41')">
	158.130.50.41
	<span class="tooltip" id="tooltip">Copied!</span>
</span>

<a href="http://158.130.50.41:7860/" target="_blank" >Link3</a> (for UPenn internal network)

<a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank">
HuggingFace
</a>

<script>
	function copyToClipboard(text) {
		navigator.clipboard.writeText(text).then(() => {
			const tooltip = document.getElementById('tooltip');
			tooltip.classList.add('show');
			
			setTimeout(() => {
				tooltip.classList.remove('show');
			}, 2000); // Message will be visible for 2 seconds
		}).catch(err => {
			console.error('Failed to copy passcode:', err);
		});
	}
</script>


<iframe
	src="https://fun-quetzal-whole.ngrok-free.app/"
	frameborder="0"
	width="100%"
	height="1600"
></iframe>


<!-- <a href="https://huggingface.co/spaces/huzey/ncut-pytorch" target="_blank" style="width: 30%; text-align: center; background-color: #FF5733; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
🤗 HuggingFace Demo
</a> -->


---

## How to host this demo yourself

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

`HF_ACCESS_TOKEN` can be left blank, only fill it if you need access to restricted models (Llama, SDv3), please see [Backbones](backbones.md).

Step 3. Use the printed out link to access your local demo.

```
...
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://some_link_here.gradio.live
```