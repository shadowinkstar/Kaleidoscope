import json, uuid, time, requests, pathlib
from typing import Optional
from rich import print

def run_comfy_workflow(
    server: str = "http://127.0.0.1:8188",
    workflow_path: str = "comfyui_workflows/flux.json",
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
        if hist:
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

def run_img2img_workflow(
    server: str = "http://127.0.0.1:8188",
    workflow_path: str = "comfyui_workflows/flux_img2img.json",
    input_image: Optional[str] = None,  # 输入图片路径
    positive: Optional[str] = None,
    negative: Optional[str] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    cfg: Optional[float] = None,
    denoise: Optional[float] = None,  # 降噪强度，图生图的关键参数
    sampler_name: Optional[str] = None,
    scheduler: Optional[str] = None,
    save_node_id: str = "3",   # 根据JSON，SaveImage节点ID是3
    timeout: int = 300
) -> Optional[pathlib.Path]:
    """
    图生图工作流函数
    
    Args:
        server: ComfyUI服务器地址
        workflow_path: 工作流JSON文件路径
        input_image: 输入图片文件名（需要在ComfyUI的input目录中）
        positive: 正面提示词
        negative: 负面提示词
        steps: 采样步数
        seed: 随机种子
        cfg: CFG强度
        denoise: 降噪强度 (0.0-1.0，值越小保留原图越多)
        sampler_name: 采样器名称
        scheduler: 调度器名称
        save_node_id: 保存节点ID
        timeout: 超时时间
    
    Returns:
        生成图片的路径，失败返回None
    """
    
    # ---------- 1. 加载 ComfyUI API workflow json ----------
    wf: dict = json.loads(pathlib.Path(workflow_path).read_text(encoding="utf-8"))

    # ---------- 2. 参数注入（根据提供的JSON结构调整） ----------
    
    # 设置输入图片（节点8 - LoadImage）
    if input_image is not None:
        wf["8"]["inputs"]["image"] = input_image
    
    # 设置正面提示词（节点7 - CLIP文本编码）
    if positive is not None:
        wf["7"]["inputs"]["text"] = positive
    
    # 设置负面提示词（节点1 - CLIP文本编码）
    if negative is not None:
        wf["1"]["inputs"]["text"] = negative
    
    # 设置KSampler参数（节点6）
    if seed is not None:
        wf["6"]["inputs"]["seed"] = seed
    if steps is not None:
        wf["6"]["inputs"]["steps"] = steps
    if cfg is not None:
        wf["6"]["inputs"]["cfg"] = cfg
    if denoise is not None:
        wf["6"]["inputs"]["denoise"] = denoise
    if sampler_name is not None:
        wf["6"]["inputs"]["sampler_name"] = sampler_name
    if scheduler is not None:
        wf["6"]["inputs"]["scheduler"] = scheduler

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
        try:
            hist = requests.get(f"{server}/history/{prompt_id}", timeout=30).json()
            if hist:
                if "error" in hist:
                    print("[red][b]ComfyUI /history 报错，流程终止：[/b][/red]")
                    return None
                if prompt_id in hist and hist[prompt_id].get("status", {}).get("status_str") == "success":
                    print("[green][b]ComfyUI 图生图生成成功！[/b][/green]")
                    outputs = hist[prompt_id].get("outputs", None)
                    break
                if time.time() - t0 > timeout:
                    print("[red]ComfyUI 生成超时[/red]")
                    return None
            time.sleep(2)
        except Exception as e:
            print(f"[yellow]轮询状态时出错: {e}, 继续等待...[/yellow]")
            time.sleep(2)

    # ---------- 5. 解析图片文件名 ----------
    fname = None
    print(f"[blue]输出结果：[/blue]")
    print(outputs)
    
    if isinstance(outputs, dict):
        node = outputs.get(str(save_node_id))
        if node and "images" in node and node["images"]:
            fname = node["images"][0]["filename"]
            print(f"[yellow]从节点 {save_node_id} 获取到图片文件名：{fname}[/yellow]")
    
    if not fname:
        print(f"[red]未找到输出文件名，请检查 outputs 字段实际内容！[/red]")
        return None

    # ---------- 6. 下载图片 ----------
    try:
        img_bytes = requests.get(f"{server}/view?filename={fname}", timeout=60).content
        out_path = pathlib.Path("outputs") / fname
        out_path.parent.mkdir(exist_ok=True, parents=True)
        out_path.write_bytes(img_bytes)
        print(f"[green]图生图结果已保存到：{out_path}[/green]")
        return out_path
    except Exception as e:
        print(f"[red]图片下载失败：{e}[/red]")
        return None

# 去除人物图片背景
def remove_background(image_path: str, output_path: str):
    """
    使用rembg库去除图片背景
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
    """
    from rembg import remove
    from PIL import Image

    try:
        input = Image.open(image_path)
        output = remove(input)
        output.save(output_path) # type: ignore
        print(f"[green]背景移除成功，结果保存在：{output_path}[/green]")
    except Exception as e:
        print(f"[red]背景移除失败：{e}[/red]")


if __name__ == "__main__":
    # 原有的文生图示例
    print("[cyan][b]=== 文生图示例 ===[/b][/cyan]")
    img = run_comfy_workflow(
        workflow_path="comfyui_workflows/flux.json",  # 官方API导出的json路径
        positive="a beautiful blonde girl with black eyes, red skirt, long black socks, beautiful",
        negative="text, watermark",
        width=768, height=1024,
        steps=25, seed=123456789,
        save_node_id="9"  # 保存节点ID（请根据你的workflow实际情况调整）
    )
    
    # print("\n" + "="*50 + "\n")
    
    # # 新增的图生图示例
    # print("[cyan][b]=== 图生图示例 ===[/b][/cyan]")
    # img2img_result = run_img2img_workflow(
    #     workflow_path="comfyui_workflows/flux_img2img.json",  # 使用你提供的JSON工作流
    #     input_image="example.png",  # 输入图片（需要在ComfyUI的input目录中）
    #     positive="an beautiful girl who has blonde hair and black eyes, and she has black long socks and red skirts, masterpiece, high quality",
    #     negative="text, watermark, low quality, blurry",
    #     steps=20,
    #     seed=183653249432838,
    #     cfg=1.0,  # CFG强度
    #     denoise=0.7,  # 降噪强度，0.7表示保留30%的原图信息
    #     sampler_name="euler",
    #     scheduler="normal",
    #     save_node_id="3"  # 根据JSON，SaveImage节点ID是3
    # )
    
    # if img2img_result:
    #     print(f"[green][b]图生图处理完成！结果保存在：{img2img_result}[/b][/green]")
    # else:
    #     print("[red][b]图生图处理失败！[/b][/red]")
    
    # 测试背景移除功能
    # print("\n" + "="*50 + "\n")
    # print("[cyan][b]=== 背景移除示例 ===[/b][/cyan]")
    # input_image_path = "outputs/ComfyUI_00009_.png"  # 输入图片路径
    # output_image_path = "outputs/removed_background.png"  # 输出图片路径
    # remove_background(input_image_path, output_image_path)
    # print(f"[green][b]背景移除处理完成！结果保存在：{output_image_path}[/b][/green]")