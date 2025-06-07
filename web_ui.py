import json
import gradio as gr
from pathlib import Path
from datetime import datetime
from typing import Iterable, Tuple

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


def ui_process(file, base_url, api_key, model_name, comfy_server, history: list[Tuple[str, str]]):
    for message in pipeline(file, base_url, api_key, model_name, comfy_server):
        history = history + [("系统", message)]
        yield history


def build_interface() -> gr.Blocks:
    examples = [str(p) for p in Path("novels").glob("*.txt")]

    with gr.Blocks(title="Kaleidoscope") as demo:
        gr.Markdown(
            """# Kaleidoscope\n将传统小说转化为视觉小说的完整流程示例。"""
        )
        with gr.Accordion("配置", open=False):
            base_url = gr.Textbox(label="LLM Base URL", value="https://dashscope.aliyuncs.com/compatible-mode/v1")
            api_key = gr.Textbox(label="API Key", type="password")
            model_name = gr.Textbox(label="Model Name", value="deepseek-v3")
            comfy_server = gr.Textbox(label="ComfyUI Server", value="http://127.0.0.1:8188")

        chatbot = gr.Chatbot(height=400)
        file = gr.File(label="上传小说文本")
        gr.Examples(examples=examples, inputs=file)
        run_btn = gr.Button("开始转换")

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
