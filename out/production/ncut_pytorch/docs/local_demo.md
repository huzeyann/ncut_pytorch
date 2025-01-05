
<!-- <meta http-equiv="refresh" content="0; url=https://fun-quetzal-whole.ngrok-free.app/" /> -->

<!-- <p><a href="https://fun-quetzal-whole.ngrok-free.app/">Redirect</a></p> -->

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

This demo is hosted at UPenn, **password is:** 
<span class="copy-code" onclick="copyToClipboard('158.130.50.41')">
	158.130.50.41
	<span class="tooltip" id="tooltip">Copied!</span>
</span>
<a href="https://click-on-the-smile-on-about-page-to-unlock-secret.loca.lt/" target="_blank" >open demo in new tab</a>

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