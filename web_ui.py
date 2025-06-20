import json, time, shutil, subprocess
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import Iterable

from log_config import get_logs

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from base import (
    parse_novel_txt,
    split_chapter,
    generate_person,
    generate_script,
    extract_info_from_script,
    image_generator_agent,
    scene_generator_agent,
    music_gen,
    convert_script,
    concat,
    tag_by_dialogue,
    load_progress,
    save_progress,
    Person,
)

LANG_CONTENT = {
    "en": {
        "intro": """
            <div style='background-color:#f1f5f9;padding:1em;border-radius:8px'>
                <h1 style='color:#6366f1;margin-top:0;'>Kaleidoscope</h1>
                <p>A demo project that converts traditional novels into colorful visual novels.</p>
                <ul>
                    <li>Upload XML text split with <code>&lt;chapter&gt;</code> tags.</li>
                    <li>The system parses chapters and generates characters, scenes and music.</li>
                    <li>Scripts can be exported to popular visual novel engines.</li>
                    <li>The longer the novel, the more tokens the generation process will use. Please be patient.</li>
                    <li>This project is tested using deepseek-v3-0324.</li>
                </ul>
            </div>
        """,
        "upload": "Upload XML text divided by <chapter> tags",
        "run": "Start Conversion",
        "settings": "Settings",
        "resume": "Resume Label",
        "base_url": "LLM Base URL",
        "api_key": "API Key",
        "model_name": "Model Name",
        "comfy_server": "ComfyUI Server",
        "outputs": "Outputs",
        "renpy_title": "Ren'Py Guide",
        "renpy_info": """\
### Install & Create Project
1. Download [Ren'Py](https://www.renpy.org/latest.html) for Windows or Linux and extract it.
2. Run `renpy.exe` on Windows or `./renpy.sh` on Linux.
3. Create a new project from the launcher and remember the folder.
4. Copy generated scripts, images and audio into the `game` directory.
5. Start the project with the launcher.

```
Install Ren'Py -> Create Project -> Open game folder -> Copy assets -> Launch
```
""",
        "renpy_path": "Ren'Py project path",
        "output_label": "Generation label",
        "copy_btn": "Copy & Launch",
        "toggle": "切换到中文",
    },
    "zh": {
        "intro": """
            <div style='background-color:#f1f5f9;padding:1em;border-radius:8px'>
                <h1 style='color:#6366f1;margin-top:0;'>Kaleidoscope</h1>
                <p>这是一个将传统小说自动转化为<span style='color:#f43f5e;'>视觉小说</span>的示例项目。</p>
                <ul>
                    <li>上传带有 <code>&lt;chapter&gt;</code> 标签划分章节的 XML 文本。</li>
                    <li>系统解析章节，生成角色设定、场景图和音乐。</li>
                    <li>最终输出可在视觉小说引擎中使用的脚本。</li>
                    <li>小说篇幅越长，生成流程使用的token就越多，请耐心等待。</li>
                    <li>本项目使用deepseek-v3-0324进行测试。</li>
                </ul>
            </div>
        """,
        "upload": "上传使用 <chapter> 标签划分章节的 XML 文本",
        "run": "开始转换",
        "settings": "配置",
        "resume": "重启标签",
        "base_url": "LLM 接口地址",
        "api_key": "API 密钥",
        "model_name": "模型名称",
        "comfy_server": "ComfyUI 地址",
        "outputs": "查看结果",
        "renpy_title": "Ren'Py 使用说明",
        "renpy_info": """\
### 安装与新建项目
1. 从 [Ren'Py 官网](https://www.renpy.org/latest.html) 下载对应系统版本并解压。
2. Windows 运行 `renpy.exe`，Linux 执行 `./renpy.sh`。
3. 在启动器中新建项目，记录项目目录。
4. 将生成的脚本、图片、音频复制到 `game` 目录。
5. 重新启动项目即可查看效果。

```
安装 Ren'Py -> 创建项目 -> 打开 game 目录 -> 复制素材 -> 启动项目
```
""",
        "renpy_path": "Ren'Py 项目路径",
        "output_label": "生成标签",
        "copy_btn": "复制并启动",
        "toggle": "Switch to English",
    },
}


load_dotenv()

def zh_en(zh: str, en: str) -> dict:
    """Helper to create bilingual messages."""
    return {"zh": zh, "en": en}


def pipeline(
    file: Path,
    base_url: str,
    api_key: str,
    model_name: str,
    comfy_server: str,
    resume_label: str = "",
) -> Iterable[dict]:
    if file is None and not resume_label:
        yield zh_en("请上传小说文件", "Please upload a novel file")
        return

    if base_url == "https://api.studio.nebius.com/v1":
        yield zh_en("感谢[Nebius](https://studio.nebius.com/)提供的免费大模型服务!", "Thanks for [Nebius](https://studio.nebius.com/) free large model service!")
    if comfy_server == "https://shadowinkstar--example-comfyui-ui.modal.run":
        yield zh_en("感谢[Modal](https://modal.com/)提供的免费ComfyUI服务!", "Thanks for [Modal](https://modal.com/) free ComfyUI service!")
    
    if api_key == "":
        llm = ChatOpenAI(
            base_url=base_url,
            model=model_name,
            max_retries=2,
            temperature=0.0,
            max_completion_tokens=8192,
            extra_body={"enable_thinking": False}
        )
    else:
        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            max_retries=2,
            temperature=0.0,
            max_completion_tokens=8192,
            extra_body={"enable_thinking": False}
        )

    if resume_label:
        label = resume_label
        base_dir = Path("outputs") / label
        if not base_dir.exists():
            yield zh_en("指定标签不存在", "Specified label does not exist")
            return
        progress = load_progress(label)
    else:
        yield zh_en("开始解析文档...", "Starting document parsing...")
        chapters = split_chapter(parse_novel_txt(path=file))
        yield zh_en(
            f"共识别到 {len(chapters)} 个章节，共 {sum([len(c.chunks) for c in chapters])} 片段",
            f"Detected {len(chapters)} chapters, {sum([len(c.chunks) for c in chapters])} segments",
        )

        yield zh_en("开始生成人物信息与脚本...", "Generating character info and scripts...")
        start_time = time.time()
        result = ""
        person_list: list[Person] = []
        for chapter in chapters:
            result += f"\n<chapter>{chapter.title}</chapter>\n"
            for chunk in chapter.chunks:
                person_list = generate_person(chunk, llm, person_list)
                script = generate_script(chunk, llm, person_list, previous_script=result)
                result += script + "\n"
                yield zh_en(f"- 已生成第{chapters.index(chapter) + 1}章节第{chapter.chunks.index(chunk) + 1}片段内容", f"- Generated Chapter {chapters.index(chapter) + 1} Segmet {chapter.chunks.index(chunk) + 1}")

        date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        label = Path(file.name).stem + f"_{date}"
        base_dir = Path("outputs") / label
        base_dir.mkdir(parents=True, exist_ok=True)
        script_path = base_dir / "script.txt"
        person_path = base_dir / "person.json"
        script_path.write_text(result, encoding="utf-8")
        person_path.write_text(
            json.dumps([p.model_dump() for p in person_list], ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        progress = {"step": "images", "image_index": 0, "scene_index": 0, "music_index": 0}
        save_progress(label, progress)
        yield zh_en(
            f"脚本与人物生成完成，用时{time.time() - start_time:.2f}秒，保存在{base_dir}。\n"
            f"请记录标签为 `{label}`，它是本次生成的标签，如果出现错误可在“重启标签”输入框中使用，点击按钮重新生成。\n"
            "如果生成结束，可以在上方OutPuts Tab栏中查看对应的生成结果。",
            f"Scripts and characters generated in {time.time() - start_time:.2f}s, saved to {base_dir}.\n"
            f"Please record the label as `{label}`, it is the label of this generation, if there is an error, you can use it in the Resume input box to regenerate."
            "If generation is complete, you can view the corresponding generation results in the OutPuts Tab.",
        )

    script_path = Path("outputs") / label / "script.txt"
    person_path = Path("outputs") / label / "person.json"
    info = extract_info_from_script(script_path, person_path)
    # 把人物信息写进json中
    with person_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(info.persons, ensure_ascii=False, indent=4))

    if progress.get("step") == "images":
        yield zh_en("生成人物立绘...", "Generating character portraits...")
        person_num = len(info.persons)
        labels = sum(len(p["labels"]) for p in info.persons)
        yield zh_en(
            f"角色共 {person_num} 个，标签共 {labels} 个，共计 {person_num + labels} 张图片，预计用时{(person_num + labels) * 20}秒",
            f"{person_num} characters with {labels} labels, {person_num + labels} images total, estimated {(person_num + labels) * 20} seconds",
        )
        start_time = time.time()
        image_generator_agent(
            llm,
            info.persons,
            prefix=label,
            server=comfy_server,
            start_index=progress.get("image_index", 0),
            progress_cb=lambda i: (progress.update({"image_index": i}), save_progress(label, progress)),
        )
        progress["step"] = "scenes"
        save_progress(label, progress)
        yield zh_en(
            f"人物立绘生成完成，用时{time.time() - start_time:.2f}秒，保存在{base_dir}/images/",
            f"Character portraits completed in {time.time() - start_time:.2f}s, saved to {base_dir}/images/",
        )

    if progress.get("step") == "scenes":
        yield zh_en("生成场景图...", "Generating scene images...")
        scene_num = len(info.scenes)
        yield zh_en(
            f"场景共 {scene_num} 个，共 {scene_num} 张图片，预计用时{scene_num * 25}秒",
            f"{scene_num} scenes, {scene_num} images, estimated {scene_num * 25} seconds",
        )
        start_time = time.time()
        scene_generator_agent(
            llm,
            info.scenes,
            prefix=label,
            server=comfy_server,
            start_index=progress.get("scene_index", 0),
            progress_cb=lambda i: (progress.update({"scene_index": i}), save_progress(label, progress)),
        )
        progress["step"] = "music"
        save_progress(label, progress)
        yield zh_en(
            f"场景图生成完成，用时{time.time() - start_time:.2f}秒，保存在{base_dir}/images/",
            f"Scene images completed in {time.time() - start_time:.2f}s, saved to {base_dir}/images/",
        )

    if progress.get("step") == "music":
        yield zh_en("生成音乐...", "Generating music...")
        music_num = len(info.music)
        yield zh_en(
            f"音乐共 {music_num} 个，共 {music_num} 个音乐文件，预计用时{music_num * 15}秒",
            f"{music_num} music pieces, {music_num} files, estimated {music_num * 15} seconds",
        )
        start_time = time.time()
        music_gen(
            llm,
            info.music,
            prefix=label,
            server=comfy_server,
            start_index=progress.get("music_index", 0),
            progress_cb=lambda i: (progress.update({"music_index": i}), save_progress(label, progress)),
        )
        progress["step"] = "done"
        save_progress(label, progress)
        yield zh_en(
            f"音乐生成完成，用时{time.time() - start_time:.2f}秒，保存在{base_dir}/audio/",
            f"Music generation completed in {time.time() - start_time:.2f}s, saved to {base_dir}/audio/",
        )

    output_path = convert_script(script_path)
    tag_by_dialogue(output_path, output_path)
    concat(output_path, Path("head.rpy"), output_path)
    yield zh_en(
        f"全部完成，标签 {label}，如需继续或查看输出，请在“重启标签”输入框填写该标签。",
        f"All done, label {label}. Use it in the Resume box to continue or view outputs.",
    )
    yield zh_en(
        f"最终脚本文件结果保存在 outputs/{label}/script.rpy",
        f"Final script saved to outputs/{label}/script.rpy",
    )


def ui_process(
    file,
    resume,
    base_url,
    api_key,
    model_name,
    comfy_server,
    history: list[dict],
    log_pos: int,
    lang: str,
    raw_history: list[dict],
):
    """执行主流程并将输出追加到聊天记录中，并同步日志."""

    for msg_pair in pipeline(file, base_url, api_key, model_name, comfy_server, resume):
        raw_history.append(msg_pair)
        history.append({"role": "assistant", "content": msg_pair.get(lang, msg_pair.get("zh"))})
        logs = get_logs(log_pos)
        log_pos += len(logs)
        yield history, "\n".join(logs), log_pos, raw_history


def show_file(path: str | list[str] | dict):
    """Show different file types based on extension."""
    if isinstance(path, dict):
        path = path.get("path") if path else ""
    if isinstance(path, list):
        if not path:
            path = ""
        else:
            path = path[0]
    if not path:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".txt", ".json", ".rpy"}:
        return (
            gr.update(value=p.read_text(encoding="utf-8"), visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    if ext in {".png", ".jpg", ".jpeg", ".gif"}:
        return (
            gr.update(visible=False),
            gr.update(value=p.as_posix(), visible=True),
            gr.update(visible=False),
        )
    if ext in {".mp3", ".wav", ".ogg"}:
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=p.as_posix(), visible=True),
        )
    return (
        gr.update(value="Unsupported file type", visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def copy_and_launch(project_path: str, label: str) -> str:
    """Copy generated assets to a Ren'Py project and try to launch it."""
    if not project_path or not label:
        return "missing path or label"
    src = Path("outputs") / label
    if not src.exists():
        return "label not found"
    dest_game = Path(project_path).expanduser()
    if (dest_game / "game").exists():
        dest_game = dest_game / "game"
    dest_game.mkdir(parents=True, exist_ok=True)
    if (src / "script.rpy").exists():
        shutil.copy2(src / "script.rpy", dest_game / "script.rpy")
    for folder in ("images", "audio"):
        d = dest_game / folder
        s = src / folder
        if s.exists():
            shutil.copytree(s, d, dirs_exist_ok=True)
    exe = shutil.which("renpy")
    if exe:
        subprocess.Popen([exe, str(dest_game.parent)])
        return f"Copied to {dest_game} and launched Ren'Py"
    return f"Copied to {dest_game}. Install Ren'Py to launch."


CUSTOM_CSS = """
#mybot [data-testid="user"] {
    background: #8b5cf6 !important;  /* 用户气泡 */
    color: white !important;
}
#mybot [data-testid="assistant"] {
    background: #10b981 !important;  /* 助手气泡 */
    color: white !important;
}
/* 其他角色（system / tool / function）可继续加 */
#toggle-btn {
    margin-left: auto;
}
"""

def build_interface() -> gr.Blocks:
    examples = [p.as_posix() for p in Path("novels").glob("*.txt")]

    theme = gr.themes.Monochrome()

    with gr.Blocks(title="Kaleidoscope", theme=theme, css=CUSTOM_CSS) as demo:
        lang_state = gr.State("en")
        raw_history = gr.State([])
        with gr.Row():
            toggle_btn = gr.Button(LANG_CONTENT["en"]["toggle"], elem_id="toggle-btn")
        with gr.Tabs() as tabs:
            with gr.TabItem("Main"):
                intro = gr.HTML(LANG_CONTENT["en"]["intro"])

                with gr.Accordion(LANG_CONTENT["en"]["settings"], open=False) as cfg:
                    base_url = gr.Textbox(
                        label=LANG_CONTENT["en"]["base_url"],
                        value="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    )
                    api_key = gr.Textbox(label=LANG_CONTENT["en"]["api_key"], type="password")
                    model_name = gr.Textbox(
                        label=LANG_CONTENT["en"]["model_name"], value="deepseek-v3"
                    )
                    comfy_server = gr.Textbox(
                        label=LANG_CONTENT["en"]["comfy_server"],
                        value="https://shadowinkstar--example-comfyui-ui.modal.run",
                    )

                chatbot = gr.Chatbot(
                    [],                           # 初始历史
                    elem_id="mybot",              # 供 CSS 定位
                    height=400,
                    type="messages",
                )
                log_box = gr.Textbox(label="Logs", lines=10, interactive=False)
                log_state = gr.State(0)
                file = gr.File(label=LANG_CONTENT["en"]["upload"])
                resume = gr.Textbox(label=LANG_CONTENT["en"]["resume"], value="")
                gr.Examples(examples=examples, inputs=file)
                run_btn = gr.Button(LANG_CONTENT["en"]["run"])

            with gr.TabItem(LANG_CONTENT["en"]["outputs"]):
                renpy_md = gr.Markdown(LANG_CONTENT["en"]["renpy_info"])
                with gr.Row():
                    with gr.Column():
                        explorer = gr.FileExplorer(
                            root_dir="outputs",
                            height=400,
                            file_count="single",
                            elem_id="output-explorer",
                        )
                        # Refresh button was removed per feedback
                    with gr.Column():
                        text_view = gr.Textbox(label="Text", lines=20, interactive=False, visible=False)
                        image_view = gr.Image(label="Image", visible=False)
                        audio_view = gr.Audio(label="Audio", interactive=False, visible=False)
                event = getattr(explorer, "select", None)
                if callable(event):
                    event(show_file, explorer, [text_view, image_view, audio_view])
                else:
                    explorer.change(show_file, explorer, [text_view, image_view, audio_view])
                with gr.Row():
                    renpy_path = gr.Textbox(label=LANG_CONTENT["en"]["renpy_path"])
                    label_box = gr.Textbox(label=LANG_CONTENT["en"]["output_label"])
                    copy_btn = gr.Button(LANG_CONTENT["en"]["copy_btn"])
                copy_msg = gr.Textbox(interactive=False)
                copy_btn.click(copy_and_launch, [renpy_path, label_box], copy_msg)

        def toggle_language(current_lang, history_raw):
            new_lang = "zh" if current_lang == "en" else "en"
            content = LANG_CONTENT[new_lang]
            translated = [
                {"role": "assistant", "content": msg_pair.get(new_lang, msg_pair.get("zh"))}
                for msg_pair in history_raw
            ]
            return (
                new_lang,
                gr.update(value=content["intro"]),
                gr.update(label=content["upload"]),
                gr.update(label=content["resume"]),
                gr.update(value=content["run"]),
                gr.update(label=content["base_url"]),
                gr.update(label=content["api_key"]),
                gr.update(label=content["model_name"]),
                gr.update(label=content["comfy_server"]),
                gr.update(label=content["settings"]),
                gr.update(value=content["toggle"]),
                gr.update(value=content["renpy_info"]),
                gr.update(label=content["renpy_path"]),
                gr.update(label=content["output_label"]),
                gr.update(value=content["copy_btn"]),
                translated,
            )

        toggle_btn.click(
            toggle_language,
            [lang_state, raw_history],
            [
                lang_state,
                intro,
                file,
                resume,
                run_btn,
                base_url,
                api_key,
                model_name,
                comfy_server,
                cfg,
                toggle_btn,
                renpy_md,
                renpy_path,
                label_box,
                copy_btn,
                chatbot,
            ],
        )

        run_btn.click(
            ui_process,
            [file, resume, base_url, api_key, model_name, comfy_server, chatbot, log_state, lang_state, raw_history],
            [chatbot, log_box, log_state, raw_history],
        )
    return demo
def main():
    demo = build_interface()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
