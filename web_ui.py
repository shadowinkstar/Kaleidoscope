import json
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import Iterable


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
                </ul>
            </div>
        """,
        "upload": "Upload XML text divided by <chapter> tags",
        "run": "Start Conversion",
        "settings": "Settings",
        "base_url": "LLM Base URL",
        "api_key": "API Key",
        "model_name": "Model Name",
        "comfy_server": "ComfyUI Server",
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
                </ul>
            </div>
        """,
        "upload": "上传使用 <chapter> 标签划分章节的 XML 文本",
        "run": "开始转换",
        "settings": "配置",
        "base_url": "LLM 接口地址",
        "api_key": "API 密钥",
        "model_name": "模型名称",
        "comfy_server": "ComfyUI 地址",
        "toggle": "Switch to English",
    },
}

def pipeline(file: Path, base_url: str, api_key: str, model_name: str, comfy_server: str) -> Iterable[str]:
    if file is None:
        yield "请上传小说文件"
        return

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        max_retries=2,
        temperature=0.0,
        max_completion_tokens=8192,
    )

    yield "开始解析文档..."
    chapters = split_chapter(parse_novel_txt(path=file.name))
    yield f"共识别到 {len(chapters)} 个章节"

    result = ""
    person_list: list[Person] = []
    for chapter in chapters:
        result += f"\n<chapter>{chapter.title}</chapter>\n"
        for chunk in chapter.chunks:
            person_list = generate_person(chunk, llm, person_list)
            script = generate_script(chunk, llm, person_list, previous_script=result)
            result += script + "\n"
            yield script

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
    yield "脚本与人物生成完成"

    info = extract_info_from_script(script_path, person_path)
    yield "生成人物立绘..."
    image_generator_agent(info.persons, prefix=label, server=comfy_server)
    yield "人物立绘生成完成"

    yield "生成场景图..."
    scene_generator_agent(info.scenes, prefix=label, server=comfy_server)
    yield "场景图生成完成"

    yield "生成音乐..."
    music_gen(info.music, prefix=label, server=comfy_server)
    yield "音乐生成完成"

    convert_script(script_path)
    yield f"全部完成，结果保存在 outputs/{label}"


def ui_process(
    file, base_url, api_key, model_name, comfy_server, history: list[dict]
):
    """执行主流程并将输出追加到聊天记录中."""

    for message in pipeline(file, base_url, api_key, model_name, comfy_server):
        history.append({"role": "assistant", "content": message})
        yield history





def build_interface() -> gr.Blocks:
    examples = [str(p) for p in Path("novels").glob("*.txt")]

    theme = gr.themes.Soft(
        primary_hue="indigo", secondary_hue="rose", neutral_hue="slate"
    )

    with gr.Blocks(title="Kaleidoscope", theme=theme) as demo:
        lang_state = gr.State("en")
        intro = gr.HTML(LANG_CONTENT["en"]["intro"])
        toggle_btn = gr.Button(LANG_CONTENT["en"]["toggle"])

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
                value="http://127.0.0.1:8188",
            )

        chatbot = gr.Chatbot(
            height=400,
            color_map=["#8b5cf6", "#10b981", "#f97316"],
            value=[],
        )
        file = gr.File(label=LANG_CONTENT["en"]["upload"])
        gr.Examples(examples=examples, inputs=file)
        run_btn = gr.Button(LANG_CONTENT["en"]["run"])

        def toggle_language(current_lang):
            new_lang = "zh" if current_lang == "en" else "en"
            content = LANG_CONTENT[new_lang]
            return (
                new_lang,
                gr.update(value=content["intro"]),
                gr.update(label=content["upload"]),
                gr.update(value=content["run"]),
                gr.update(label=content["base_url"]),
                gr.update(label=content["api_key"]),
                gr.update(label=content["model_name"]),
                gr.update(label=content["comfy_server"]),
                gr.update(label=content["settings"]),
                gr.update(value=content["toggle"]),
            )

        toggle_btn.click(
            toggle_language,
            [lang_state],
            [
                lang_state,
                intro,
                file,
                run_btn,
                base_url,
                api_key,
                model_name,
                comfy_server,
                cfg,
                toggle_btn,
            ],
        )

        run_btn.click(
            ui_process,
            [file, base_url, api_key, model_name, comfy_server, chatbot],
            chatbot,
        )
    return demo
def main():
    demo = build_interface()
    demo.queue()
    demo.launch()


if __name__ == "__main__":
    main()
