# Kaleidoscope

- 🎥 You can watch the video introduction on [zhihu](https://www.zhihu.com/zvideo/1915566580563157432).
- 📚 You can see the detailed documentation on [GitHub Pages](https://github.com/shadowinkstar/Kaleidoscope).
- 🤗 I deploy a demo project on Huggingface Space [Kaleidoscope](https://huggingface.co/spaces/Agents-MCP-Hackathon/Kaleidoscope)
- 📃 中文文档 [README_zh.md](README_zh.md)

## Project Overview

This project aims to convert traditional text novels into playable visual novels with the help of AI large models. The system parses novel text, extracts key information such as characters and scenes, and uses generative image and audio models to produce the corresponding assets. The final output is a script that can run on engines like Ren'Py.

![](2024-07-07-%E8%AE%BE%E8%AE%A1%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

The project is planned in three phases:
1. **Novel to script**: generate scripts with character sprites, dialog and narration from existing novels.
2. **Interactive experience**: players can act as novel characters and affect the plot through AI-provided options.
3. **Fully original**: with a given theme and settings, the AI generates a complete original visual novel.

## Directory Structure

- `novels/` sample novel texts to convert
- `comfyui_workflows/` ComfyUI workflow configurations for image and audio generation
- `outputs/` generated scripts, character info and the corresponding images and audio
- `scripts/` saved examples of partial results
- main scripts are located in the repository root

## Setup

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies. Python 3.10 or above is recommended.

```bash
# Create and activate the virtual environment
uv venv
source .venv/bin/activate   # Windows users run .venv\Scripts\activate

# Install dependencies
uv sync
```

Before running, you can edit `.env` to configure your API keys and other options.

## Main Scripts

- `base.py`: core script that parses novels and generates the script and character list.
- `img.py`: calls ComfyUI to create character or scene images based on descriptions.
- `audio.py`: calls ComfyUI to generate dialog or music audio.
- `prompt.py`: templates for interacting with the large language model.
- `web_ui.py`: Gradio interface that integrates the entire workflow from document upload to asset generation.

## How to Run

Taking `base.py` as an example, edit it and run the following command to start converting a sample novel:

```bash
uv run base.py
```

The generated script and character information will appear in the `outputs/` directory. If you want to batch-generate images or audio, run `img.py` and `audio.py` respectively, or call them in `base.py` as needed. But use web_ui.py to launch the interface is more convenient.

## Launch the Web Interface

You can experience the full workflow in your browser via `web_ui.py`:

```bash
uv run web_ui.py
```

Within the interface you can upload text or select an example file and fill in model and ComfyUI parameters in the collapsible settings panel.
Once the generation is complete, the results can be viewed in the "Outputs" tab. And you can use the easy-to-use "Copy" feature to copy the outputs to your Ren'Py project.

## Resume from Failure

Each run creates a unique **label** for the output directory (e.g. `example_2024-01-01-12-00-00`). If the generation is interrupted or you want to review previous results, enter this label in the "Resume" field of the interface. The system will continue from the last progress and show the content in the "Outputs" tab.

---

For the Chinese documentation, see [README_zh.md](README_zh.md).
