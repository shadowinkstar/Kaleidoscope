import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal
import modal.experimental

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("fastapi[standard]==0.115.4")  # install web dependencies
    .pip_install("comfy-cli==1.4.0")  # install comfy-cli
    .run_commands(  # use comfy-cli to install ComfyUI and its dependencies
        "comfy --skip-prompt install --fast-deps --nvidia"
    )
)

# ## Downloading custom nodes

# We'll also use `comfy-cli` to download custom nodes, in this case the popular [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui).

# Use the [ComfyUI Registry](https://registry.comfy.org/) to find the specific custom node name to use with this command.

image = (
    image.run_commands(  # download a custom node
        "comfy node install --fast-deps was-node-suite-comfyui@1.0.2"
    )
    # Add .run_commands(...) calls for any other custom nodes you want to download
)

# See [this post](https://modal.com/blog/comfyui-custom-nodes) for more examples
# on how to install popular custom nodes like ComfyUI Impact Pack and ComfyUI IPAdapter Plus.

# ## Downloading models

# `comfy-cli` also supports downloading models, but we've found it's faster to use
# [`hf_hub_download`](https://huggingface.co/docs/huggingface_hub/en/guides/download#download-a-single-file)
# directly by:

# 1. Enabling [faster downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads)
# 2. Mounting the cache directory to a [Volume](https://modal.com/docs/guide/volumes)

# By persisting the cache to a Volume, you avoid re-downloading the models every time you rebuild your image.


def hf_download():
    from huggingface_hub import hf_hub_download

    flux_model = hf_hub_download(
        repo_id="Comfy-Org/flux1-schnell",
        filename="flux1-schnell-fp8.safetensors",
        cache_dir="/cache",
    )
    # symlink the model to the right ComfyUI directory
    subprocess.run(
        f"ln -s {flux_model} /root/comfy/ComfyUI/models/checkpoints/flux1-schnell-fp8.safetensors",
        shell=True,
        check=True,
    )
    
    t5_base= hf_hub_download(
        repo_id="google-t5/t5-base",
        filename="model.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {t5_base} /root/comfy/ComfyUI/models/text_encoders/t5_base.safetensors",
        shell=True,
        check=True,
    )
    


vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]==0.30.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        hf_download,
        # persist the HF cache to a Modal Volume so future runs don't re-download models
        volumes={"/cache": vol},
    )
)

# Lastly, copy the ComfyUI workflow JSON to the container.
image = image.add_local_file(
    Path(__file__).parent.parent / "comfyui_workflows/flux_img2img.json", "/root/workflow_api.json"
).add_local_file(Path("D:\\SoftWares\\ComfyUI\\ComfyUI\\models\\checkpoints\\stable_audio_open_1.0.safetensors"), "/root/comfy/ComfyUI/models/checkpoints/stable_audio_open_1.0.safetensors")


# ## Running ComfyUI interactively

# Spin up an interactive ComfyUI server by wrapping the `comfy launch` command in a Modal Function
# and serving it as a [web server](https://modal.com/docs/guide/webhooks#non-asgi-web-servers).

app = modal.App(name="example-comfyui", image=image)


@app.function(
    max_containers=1,  # limit interactive session to 1 container
    gpu="L40S",  # good starter GPU for inference
    volumes={"/cache": vol},  # mounts our cached models
)
@modal.concurrent(
    max_inputs=10
)  # required for UI startup process which runs several API calls concurrently
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)