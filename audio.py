import json, uuid, time, requests, pathlib
from typing import Optional
from rich.console import Console
from log_config import logger

console = Console()


def run_audio_workflow(
    server: str = "http://127.0.0.1:8188",
    workflow_path: str = "comfyui_workflows/stable_audio.json",
    prefix: str = "",
    positive: Optional[str] = None,
    negative: Optional[str] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    duration: Optional[float] = None,
    save_node_id: str = "18",
    timeout: int = 300,
) -> Optional[pathlib.Path]:
    """根据提供的 ComfyUI workflow 生成音频

    Parameters
    ----------
    server: ComfyUI 后端地址
    workflow_path: ComfyUI 导出的 workflow JSON 路径
    prefix: 输出目录前缀
    positive: 正面提示词
    negative: 负面提示词
    steps: 采样步数
    seed: 随机种子
    duration: 音频时长，单位秒
    save_node_id: 保存节点 ID（默认18，对应 SaveAudio），用于从 history 中解析文件名
    timeout: 轮询超时时间，单位秒
    """
    wf: dict = json.loads(pathlib.Path(workflow_path).read_text(encoding="utf-8"))

    # 参数注入
    if positive is not None:
        wf["6"]["inputs"]["text"] = positive
    if negative is not None:
        wf["7"]["inputs"]["text"] = negative
    if steps is not None:
        wf["3"]["inputs"]["steps"] = steps
    if seed is not None:
        wf["3"]["inputs"]["seed"] = seed
    if duration is not None:
        wf["11"]["inputs"]["seconds"] = duration

    client_id = str(uuid.uuid4())
    payload = {"prompt": wf, "client_id": client_id}
    try:
        resp = requests.post(f"{server}/prompt", json=payload, timeout=30)
        result = resp.json()
        logger.debug("ComfyUI /prompt 返回：\n{}", json.dumps(result, indent=2, ensure_ascii=False))
        if "error" in result:
            logger.error("ComfyUI /prompt 接口报错，流程终止！")
            return None
        prompt_id = result["prompt_id"]
    except Exception as e:
        logger.error(f"提交 /prompt 失败:{e}")
        return None

    # 轮询生成状态
    t0 = time.time()
    outputs = None
    with console.status("[bold cyan]ComfyUI 正在生成，请稍候…[/]", spinner="dots"):
        while True:
            try:
                hist = requests.get(f"{server}/history/{prompt_id}", timeout=30).json()
                if hist:
                    if "error" in hist:
                        console.print("[red][b]ComfyUI /history 报错，流程终止[/b][/red]")
                        return None
                    if prompt_id in hist and hist[prompt_id].get("status", {}).get("status_str") == "success":
                        outputs = hist[prompt_id].get("outputs")
                        elapsed = time.time() - t0
                        console.print(f"[green][b]ComfyUI 音频生成成功！耗时 {elapsed:.1f} 秒[/b][/green]")
                        break
                    if time.time() - t0 > timeout:
                        console.print("[red]ComfyUI 生成超时[/red]")
                        return None
                time.sleep(0.5)
            except Exception as e:
                console.print(f"[yellow]轮询状态时出错: {e}，继续等待...[/yellow]")
                time.sleep(0.5)

    # 获取音频文件名
    fname = None
    logger.debug(outputs)
    if isinstance(outputs, dict):
        node = outputs.get(str(save_node_id))
        if node:
            if isinstance(node, dict):
                if "audio" in node and node["audio"]:
                    item = node["audio"][0]
                    if isinstance(item, dict) and "filename" in item:
                        fname = item["filename"]
                elif "filename" in node:
                    fname = node["filename"]
    if not fname:
        logger.error("未找到输出文件名，请检查 outputs 字段实际内容！")
        return None

    try:
        audio_bytes = requests.get(f"{server}/view?filename={fname}&type=output&subfolder=audio", timeout=60).content
        out_dir = pathlib.Path("outputs") / prefix / "audio"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / pathlib.Path(fname).name
        out_path.write_bytes(audio_bytes)
        logger.info(f"音频已保存到：{out_path}")
        return out_path
    except Exception as e:
        logger.error(f"音频下载失败：{e}")
        return None


if __name__ == "__main__":
    logger.info("=== 音频生成示例 ===")
    audio_result = run_audio_workflow(
        prefix="demo",
        workflow_path="comfyui_workflows/stable_audio.json",
        positive="heaven church electronic dance music",
        negative="",
        steps=50,
        seed=657172215808422,
        duration=60.0,
        save_node_id="18",
    )

    if audio_result:
        logger.info(f"音频生成完成！结果保存在：{audio_result}")
    else:
        logger.error("音频生成失败！")
