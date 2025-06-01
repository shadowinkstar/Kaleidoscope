import json, uuid, time, requests, pathlib
from typing import Optional
from rich import print

def run_comfy_workflow(
    server: str = "http://127.0.0.1:8188",
    workflow_path: str = "workflow_api.json",
    positive: Optional[str] = None,
    negative: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    save_node_id: str = "9",   # SaveImage节点ID（请根据实际workflow调整）
    timeout: int = 300
) -> Optional[pathlib.Path]:
    # ---------- 1. 加载 ComfyUI API workflow json ----------
    wf: dict = json.loads(pathlib.Path(workflow_path).read_text(encoding="utf-8"))

    # ---------- 2. 参数注入（请对照你的workflow结构和id调整） ----------
    if positive is not None:
        wf["6"]["inputs"]["text"] = positive         # 正面prompt
    if negative is not None:
        wf["7"]["inputs"]["text"] = negative         # 负面prompt
    if width is not None:
        wf["5"]["inputs"]["width"] = width           # 分辨率
    if height is not None:
        wf["5"]["inputs"]["height"] = height
    if steps is not None:
        wf["3"]["inputs"]["steps"] = steps           # 步数
    if seed is not None:
        wf["3"]["inputs"]["seed"] = seed             # 随机种子

    # ---------- 3. 提交到 /prompt ----------
    client_id = str(uuid.uuid4())
    payload = {"prompt": wf, "client_id": client_id}
    try:
        resp = requests.post(f"{server}/prompt", json=payload, timeout=30)
        result = resp.json()
        print("[yellow][b]ComfyUI /prompt 返回：[/b][/yellow]")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if "error" in result:
            print("[red][b]ComfyUI /prompt 接口报错，流程终止！[/b][/red]")
            return None
        prompt_id = result["prompt_id"]
    except Exception as e:
        print(f"[red][b]提交 /prompt 失败:[/b] {e}")
        return None

    # ---------- 4. 轮询 /history ----------
    t0 = time.time()
    outputs = None
    while True:
        hist = requests.get(f"{server}/history/{prompt_id}", timeout=30).json()
        if "error" in hist:
            print("[red][b]ComfyUI /history 报错，流程终止：[/b][/red]")
            return None
        if hist[prompt_id].get("status")["status_str"] == "success":
            print("[green][b]ComfyUI 生成成功！[/b][/green]")
            outputs = hist[prompt_id].get("outputs", None)
            break
        if time.time() - t0 > timeout:
            print("[red]ComfyUI 生成超时[/red]")
            return None
        time.sleep(2)

    # ---------- 5. 解析图片文件名（兼容多种结构） ----------
    fname = None
    print(outputs)
    if isinstance(outputs, dict):
        node = outputs.get(str(save_node_id))
        if node and "images" in node and node["images"]:
            fname = node["images"][0]["filename"]
            print(f"[yellow]从节点 {save_node_id} 获取到图片文件名：{fname}[/yellow]")
    elif isinstance(outputs, list) and outputs:
        img = outputs[0]
        if isinstance(img, dict) and "filename" in img:
            fname = img["filename"]
        elif isinstance(img, str):
            fname = img
    if not fname:
        print(f"[red]未找到输出文件名，请检查 outputs 字段实际内容！[/red]")
        return None

    # ---------- 6. 下载图片 ----------
    try:
        img_bytes = requests.get(f"{server}/view?filename={fname}", timeout=60).content
        out_path = pathlib.Path("outputs") / fname
        out_path.parent.mkdir(exist_ok=True, parents=True)
        out_path.write_bytes(img_bytes)
        print(f"[green]图片已保存到：{out_path}[/green]")
        return out_path
    except Exception as e:
        print(f"[red]图片下载失败：{e}[/red]")
        return None

if __name__ == "__main__":
    img = run_comfy_workflow(
        workflow_path="comfyui_workflows/flux.json",  # 官方API导出的json路径
        positive="a beautiful blonde girl with black eyes, red skirt, long black socks",
        negative="text, watermark",
        width=768, height=1024,
        steps=25, seed=123456789,
        save_node_id="9"  # 保存节点ID（请根据你的workflow实际情况调整）
    )
