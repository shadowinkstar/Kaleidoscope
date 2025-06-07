# Kaleidoscope

## 项目简介

本项目旨在将传统文字小说借助 AI 大模型自动转化为可游玩的视觉小说。系统会解析小说文本，提取角色、场景等关键信息，再利用图像和音频生成模型生成相应素材，最终输出一套可在 Ren'Py 等引擎中运行的脚本。

![](2024-07-07-%E8%AE%BE%E8%AE%A1%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

项目规划分为三个阶段：
1. **小说转脚本**：从现有小说出发，生成包含人物立绘、对话与旁白的脚本。
2. **互动式体验**：玩家可扮演小说角色，通过 AI 给出的选项影响剧情走向。
3. **完全原创**：在给定主题和设定的条件下，由 AI 全流程生成原创文字游戏。

## 目录结构

- `novels/` 存放待转换的小说文本示例
- `comfyui_workflows/` ComfyUI 的工作流配置，用于图像/音频生成
- `outputs/` 生成的脚本、人物信息以及对应的图片、音频
- `scripts/` 部分生成结果的示例存档
- 主要脚本位于仓库根目录

## 环境部署

本项目使用 [uv](https://github.com/astral-sh/uv) 管理依赖。建议使用 Python 3.10 以上版本。

```bash
# 创建并激活虚拟环境
uv venv
source .venv/bin/activate   # Windows 用户使用 .venv\Scripts\activate

# 安装依赖
uv sync
```

运行前可根据需要修改 `.env` 中的 API Key 等配置。

## 主要脚本

- `base.py`：核心脚本，负责解析小说并生成脚本、角色列表等信息。
- `img.py`：调用 ComfyUI 根据描述生成人物或场景图片。
- `audio.py`：调用 ComfyUI 生成对白或音乐音频。
- `prompt.py`：存放与大模型交互的提示词模板。
- `web_ui.py`：基于 Gradio 的图形界面，整合从文档上传到素材生成的完整流程。

## 运行方式

以 `base.py` 为例，执行下列命令即可开始转换示例小说：

```bash
uv run base.py
```

生成的脚本和人物信息会存放在 `outputs/` 目录下。若需要批量生成图片或音频，可分别运行 `img.py` 和 `audio.py`，或在 `base.py` 中按需调用。

## 启动 Web 界面

通过 `web_ui.py` 可以在浏览器中体验完整流程：

```bash
uv run web_ui.py
```

界面中可上传文本或选择示例文件，并在折叠的配置面板中填写大模型和 ComfyUI 的参数。

