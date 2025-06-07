from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional, Union, Pattern
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
import time, re, json, base64
from log_config import logger
from prompt import EXTRACT_PERSON_PROMPT, GENERATE_SCRIPT_PROMPT
from img import run_comfy_workflow, run_img2img_workflow, remove_background
from datetime import datetime
from pathlib import Path
from audio import run_audio_workflow
from rich.console import Console
from collections import Counter
import shutil, tempfile


load_dotenv()

llm = ChatOpenAI(
    # model="qwen3-235b-a22b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="deepseek-v3",
    # base_url="https://api.studio.nebius.com/v1/",
    max_retries=2,
    temperature=0.0,
    max_completion_tokens=8192,
    extra_body={"enable_thinking": False}
)

vision_llm = ChatOpenAI(
    model="qwen-vl-plus-0102",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_retries=2,
    temperature=0.0,
    max_completion_tokens=8192
)

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def replace_first(
    text: str,
    pattern: Union[str, Pattern[str]],
    new_char: str
) -> str:
    """
    把 `text` 中第一个匹配 `pattern` 的字符替换成 `new_char`。

    - pattern 可以是普通字符，也可以是正则表达式（字符类、分组都行）
    - 如果找不到匹配，原样返回
    """
    return re.sub(pattern, new_char, text, count=1)

class Chapter(BaseModel):
    """小说章节类"""
    title: str = Field(description="章节标题")
    content: str = Field(description="章节内容")
    chunks: Optional[List[Document]] = Field(default=None, description="使用langchain切分后的章节内容")
    novel_name: str = Field(description="章节来源的小说名称")
    

# 处理文本
def parse_novel_txt(path: str = "", context: str = "") -> List[Chapter]:
    """
    处理小说文本，将小说文本切分成章节，并返回识别的章节列表，这里暂时不涉及到智能的章节划分，
    使用自定义的xml标签进行识别划分
    :param path: 小说文本路径
    :param context: 或者直接传入小说文本内容
    :return: 识别出的小说章节列表
    """
    if context != "" and path != "":
        raise ValueError("path和context不能同时传入")
    # 最终输出列表
    chapters_result = []
    if context:
        novel_text = context
        soup = BeautifulSoup(novel_text, "lxml")
        if soup.find("novel") and soup.find("novel").text: # type: ignore
            novel_name = soup.find("novel").text # type: ignore
        else:
            logger.warning("小说名称未识别，使用默认名称")
            novel_name = f"在时间{time.strftime('%Y-%m-%d %H:%M:%S')}传入小说"
        chapters = soup.find_all("chapter")
        if chapters == []:
            logger.warning("小说章节未识别，使用全文")
            chapters = [soup]
        for chapter in chapters:
            if chapter.find("content") and chapter.find("content").text: # type: ignore
                content = chapter.find("content").text # type: ignore
            else:
                logger.warning("章节内容识别失败，全文作为一整个章节")
                content = chapter.text
            if chapter.find("title") and chapter.find("title").text: # type: ignore
                title = chapter.find("title").text # type: ignore
            else:
                logger.warning("章节标题识别失败，使用该章节前10个字符作为标题")
                title = content[:10].strip()
            chapters_result.append(Chapter(title=title, content=content, novel_name=novel_name)) # type: ignore
    else:
        with open(path, "r", encoding="utf-8") as f:
            novel_name = path.split(".")[0]
            novel_text = f.read()
            soup = BeautifulSoup(novel_text, "lxml")
            if soup.find("novel") and soup.find("novel").text: # type: ignore
                # 优先采用文中标题
                logger.warning("使用文本标题标签内容作为小说标题") 
                novel_name = soup.find("novel").text # type: ignore
            chapters = soup.find_all("chapter")
            for chapter in chapters:
                if chapter.find("content") and chapter.find("content").text: # type: ignore
                    content = chapter.find("content").text # type: ignore
                else:
                    logger.warning("章节内容识别失败，跳过该章节")
                    continue
                if chapter.find("title") and chapter.find("title").text: # type: ignore
                    title = chapter.find("title").text # type: ignore
                else:
                    logger.warning("章节标题识别失败，使用该章节前10个字符作为标题")
                    title = content[:10]
                chapters_result.append(Chapter(title=title, content=content, novel_name=novel_name)) # type: ignore
    return chapters_result

def split_chapter(chapters: List[Chapter], chunk_size: int = 4000, overlap: int = 200) -> List[Chapter]:
    """
    将章节内容切分成Document对象，并返回
    :param chapter: 小说章节对象
    :param chunk_size: 切分大小
    :param overlap: 切分重叠大小
    :return: 增加切分后的章节内容属性的章节对象列表
    """
    # 使用langchain切分，由于要保证每段长度适中，因此需要动态调整
    logger.info("开始切分小说章节内容")
    # 默认切分配置
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap, length_function=len)
    for chapter in chapters:
        length = len(chapter.content)
        if length < chunk_size:
            chapter.chunks = splitter.create_documents(
                texts=[chapter.content],
                metadatas=[{"title": chapter.title, "novel_name": chapter.novel_name}]
                )
        else:
            new_chunk_size = length //((length // chunk_size) + 1)
            logger.info(f"章节内容长度过长，使用动态切分，切分大小为{new_chunk_size}")
            new_splitter = RecursiveCharacterTextSplitter(chunk_size=new_chunk_size, chunk_overlap=overlap, length_function=len)
            chapter.chunks = new_splitter.create_documents(
                texts=[chapter.content],
                metadatas=[{"title": chapter.title, "novel_name": chapter.novel_name}]
                )
    return chapters

class Person(BaseModel):
    """小说人物"""
    name: str = Field(description="对话脚本中使用的人物名称，保证唯一性")
    description: str = Field(description="对于人物形象的全面描述")
    seed: int = Field(default=42, description="文生图的随机种子，保证每次生成图像的结果唯一，默认为神秘数字42")
    novel_name: str = Field(description="人物所属小说名称")
    labels: Optional[List[str]] = Field(default=[], description="人物标签列表，用于标识人物的特征或身份")

def extract_json(message: BaseMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between ```json and ``` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"```json(.*?)```"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL) # type: ignore

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches][0]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

def generate_person(chapter_document: Document, llm: ChatOpenAI, person_list: List[Person]) -> List[Person]:
    """
    根据小说章节内容生成其中出现的人物对象，并返回人物对象列表
    :param chapter_document: 小说章节内容
    :param llm: 使用的LLM模型
    :param person_list: 已经识别的人物列表
    :return: 人物描述列表
    """
    prompt = PromptTemplate.from_template(EXTRACT_PERSON_PROMPT)
    # print(prompt.invoke({"text": chapter_document.page_content}))
    chain = prompt | llm 
    input_person_list = [{"name": person.name, "description": person.description} for person in person_list]
    try:
        result = extract_json(chain.invoke({"text": chapter_document.page_content, "person_list": input_person_list}))
    except Exception as e:
        logger.error(f"提取人物出现错误{e}，直接返回空")
        return []
    logger.info(f"{chapter_document.page_content[:10].strip()}...{chapter_document.page_content[-10:].strip()}提取人物结果：\n{result}")
    result_person_list = person_list.copy()
    # 使用模型输出的result更新人物列表
    if result == []:
        logger.info("人物信息无需更新，跳过")
        return result_person_list
    else:
        logger.info("检测到新人物或人物信息更新，开始更新人物列表")
        for i in result:
            # 检查人物是否已经存在
            if not any(person.name == i["name"] for person in result_person_list):
                # 如果不存在，则添加新人物
                person = Person(**i, novel_name=chapter_document.metadata["novel_name"])
                result_person_list.append(person)
            else:
                # 如果存在，则更新人物信息
                for person in result_person_list:
                    if person.name == i["name"]:
                        person.description = i["description"]
                        person.novel_name = chapter_document.metadata["novel_name"]
                        break
    return result_person_list

def generate_script(chapter_document: Document, llm: ChatOpenAI, person_list: List[Person], previous_script: str) -> str:
    """
    根据小说章节内容生成对话脚本，并返回对话脚本
    :param chapter_document: 小说章节内容
    :param llm: 使用的LLM模型
    :param person_list: 已经识别的人物列表
    :param previous_script: 上一章节的对话脚本内容，用于保证对话脚本的连续性
    :return: 对话脚本
    """
    prompt = PromptTemplate.from_template(GENERATE_SCRIPT_PROMPT)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": chapter_document.page_content, "person_list": [person.name for person in person_list], "previous_script": previous_script})
    logger.info(f"{chapter_document.page_content[:10].strip()}...{chapter_document.page_content[-10:].strip()}提取脚本结果：\n{result}")
    return result


# 使用一个类来管理从文本中提取的内容
class ScriptGenInfo(BaseModel):
    """从脚本中提取的等待生成的内容"""
    persons: List[dict] = Field(default_factory=list, description="提取到的人物列表")
    scenes: List[str] = Field(default_factory=list, description="提取到的场景列表")
    titles: List[str] = Field(default_factory=list, description="提取到的章节标题列表")
    music: List[str] = Field(default_factory=list, description="提取到的章节内容列表")
    

# 提取脚本中人物与场景信息，准备进行图像生成
def extract_info_from_script(script_path: Path, person_path: Path, script: str = "") -> ScriptGenInfo:
    """从脚本文件中提取人物、场景与章节信息，并放在一个对象类中统一返回

    Args:
        script_path (Path): 脚本路径
        person_path (Path): 人物信息路径
        script (str, optional): 直接传入脚本内容. Defaults to "".

    Returns:
        ScriptGenInfo: 包含提取到的人物、场景与章节信息的对象
    """
    result = ScriptGenInfo(
        persons=[],
        scenes=[],
        music=[],
        titles=[]
    )
    if script == "":
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                script = f.read()
        except Exception as e:
            logger.error(f"读取脚本文件失败：{e}")
            return result
    if person_path:
        try:
            with open(person_path, "r", encoding="utf-8") as f:
                person_list = json.load(f)
        except Exception as e:
            logger.error(f"读取人物信息文件失败：{e}")
            return result
    # 使用正则表达式提取<person>标签中的内容
    person_pattern = r"<person>(.*?)</person>"
    persons = re.findall(person_pattern, script)
    logger.debug(f"提取到的人物信息：{persons}")
    conflit_persons = []
    for i in persons:
        name, label = i.split(" ")
        # 检查人物是否已经存在
        if not any(person["name"] == name for person in person_list):
            # 如果不存在，放在待解决冲突人物列表中
            conflit_persons.append({"name": name, "label": label})
        else:
            # 如果存在，则把人物标签放在人物信息中
            for person in person_list:
                if person["name"] == name:
                    if label not in person["labels"]:
                        person["labels"].append(label)
                    break
    logger.debug(f"冲突人物信息：{conflit_persons}")
    logger.debug(f"当前人物信息：{person_list}")
    
    # 处理冲突人物信息，即脚本中出现没有正常提取的人物
    for newbie in conflit_persons:
        if not any(person["name"] == newbie["name"] for person in person_list):
            person_list.append({"name": newbie["name"], "labels": [newbie["label"]], "description": "请参考角色姓名生成合适的立绘", "novel_name": "", "seed": 42})
        else:
            for person in person_list:
                if person["name"] == newbie["name"]:
                    if newbie["label"] not in person["labels"]:
                        person["labels"].append(newbie["label"])
                    break
    result.persons = person_list
    
    # 使用正则表达式提取<scene>标签中的内容
    scene_pattern = r"<scene>(.*?)</scene>"
    scenes = re.findall(scene_pattern, script)
    # 由于场景重复，进行去重
    scenes = list(set(scenes))
    logger.debug(f"提取到的场景信息：{scenes}")
    result.scenes = scenes
    
    # 提取脚本中每一章节的内容
    music_pattern = r"<chapter>(.*?)</chapter>"
    titles = re.findall(music_pattern, script)
    result.titles = titles
    for ch in titles:
        script = script.replace(f"<chapter>{ch}</chapter>", "||||||||")
    result.music = script.split("||||||||")[1:] # TODO: 这里只是暂时去除一个回车带来的多余划分，没有处理更多的错误情况
    
    return result

# 调用大模型生成人物立绘文生图的提示词然后调用文生图工具并查看生成图像进行图生图优化
def image_generator_agent(persons: List[dict], prefix: str, server: str = "http://127.0.0.1:8188") -> None:
    """
    该智能体可以根据人物信息生成对应的立绘
    Args:
        persons (List[dict]): 人物信息列表

    Returns:
        不返回内容，把生成的图片放置在指定路径即可
    """
    logger.debug(f"待生成人物列表: {persons}")
    for person in persons:
        text2img_message = HumanMessage(
            content=[
                {"type": "text", "text": "我会给你提供一个人物的信息，然后你需要结合人物信息生成一个使用FLUX-dev模型的文生图提示词，这个提示词应当具备正面提示词与负面提示词两个部分的内容，以更好地生成符合人物描述的立绘。\
                    请使用简洁清晰的英文短句来构建提示词，注意你可以扩展提供的人物描述，并且结合人物描述中的经历等构造出一个人物形象的描述，不要涉及太多人物性格等描述，并且重心放在人物描述上面，减少对画面背景的描写。请尽量详细描写正面提示词，\
                    尽量从各种角度完善人物形象，控制在10句以上；而负面提示词尽量使用那些有利于图像生成的常见负面提示词，并且针对人物形象做出适应的改变。请你使用如下的\
                    JSON格式返回提示词：```json\n{{'positive': '正面提示词，使用逗号分隔的多个句子', 'negative': '负面提示词，使用逗号分隔的多个句子'}}\n```\n，人物的信息如下{}".format(json.dumps(person))}
            ]
        )
        logger.info(f"正在为人物 {person['name']} 生成提示词...")
        response = llm.invoke([text2img_message])
        logger.debug(f"人物 {person['name']} 的提示词生成结果：{response.content}")
        try:
            result = extract_json(response)
        except:
            logger.info(f"正在为人物 {person['name']} 生成提示词...")
            response = llm.invoke([text2img_message])
            logger.debug(f"人物 {person['name']} 的提示词生成结果：{response.content}")
            try:
                result = extract_json(response)
            except Exception as e:
                logger.error(f"重试后提示词生成结果解析失败: {e}")
                result = {"positive": person["decription"], "negative": ""}
        logger.info(f"正在为人物 {person['name']} 生成立绘...")
        img_path = run_comfy_workflow(server=server, positive=result["positive"], negative=result["negative"], prefix=prefix) # type: ignore
        if img_path:
            person_img_path = img_path.with_name(f"{person['name']}.png")
            img_path.rename(person_img_path)
            # 去除人物背景
            remove_background(person_img_path, person_img_path)
        else:
            raise Exception("图片生成失败")
        logger.info(f"人物 {person['name']} 的立绘生成成功，图片路径为：{person_img_path}")
        # 开始针对不同的label生成一系列立绘
        for label in person["labels"]:
            logger.info(f"正在为人物 {person['name']} 生成标签 {label} 的提示词...")
            text2img_message = HumanMessage(
                content=[
                    {"type": "text", "text": f"我会给你提供一个人物的信息，然后你需要结合提供的人物信息与人物现在的标签生成一个使用FLUX-dev模型的图生图提示词，这个提示词应当具备正面提示词与负面提示词两个部分的内容，以更好地把现有图片立绘修改为符合人物标签描述的立绘。\
                        请使用简洁清晰的英文短句来构建提示词，注意你要根据人物的标签内容（可能很简短），在原有提示词的基础上设计一套新的提示词，重心放在人物如何与标签对应上面，以求更加准确地修改图像；\
                        而负面提示词尽量使用那些有利于图像生成的常见负面提示词，并且针对人物形象做出适应的改变。请你使用如下的\
                        JSON格式返回提示词：```json\n{{'positive': '正面提示词，使用逗号分隔的多个句子', 'negative': '负面提示词，使用逗号分隔的多个句子'}}\n```\n，人物的信息如下{json.dumps(person)}，之前生成立绘的提示词为{json.dumps(result)}，生成的人物立绘需要满足新的标签{label}，请在提示词中充分体现这个标签的内容。"}
                ]
            )
            response = llm.invoke([text2img_message])
            logger.debug(f"人物 {person['name']} 标签 {label} 的提示词生成结果：{response.content}")
            try:
                label_result = extract_json(response)
            except:
                response = llm.invoke([text2img_message])
                logger.debug(f"人物 {person['name']} 标签 {label} 的提示词生成结果：{response.content}")
                try:
                    label_result = extract_json(response)
                except Exception as e:
                    logger.error(f"重新生成提示词失败: {e}")
                    label_result = {"positive": f"{result['positive']}, {label}", "negative": f"{result['negative']}"} # type: ignore
            logger.info(f"正在为人物 {person['name']} 标签 {label} 生成立绘...")
            label_img_path = run_img2img_workflow(server=server, input_image=str(person_img_path.resolve()), positive=label_result["positive"], negative=label_result["negative"], prefix=prefix) # type: ignore
            if label_img_path:
                person_label_img_path = label_img_path.with_name(f"{person['name']} {label}.png")
                label_img_path.rename(person_label_img_path)
                # 背景移除
                remove_background(person_label_img_path, person_label_img_path)
            else:
                raise Exception("图片生成失败")
            logger.info(f"人物 {person['name']} 标签 {label} 的立绘生成成功，图片路径为：{person_label_img_path}")
        # base64_image = encode_image(img_path)
        # img2img_message = HumanMessage(
        #     content=[
        #         {"type": "text", "text": f"下面的图片是一个人物立绘，请查看这张图片是否符合如下人物描述：{json.dumps(person)}\n如果符合，请回复'是'，如果不符合，请根据人物描述修改提示词，使用\
        #         如下的JSON格式返回提示词：```json\n{{'positive': '正面提示词，使用逗号分隔的多个句子', 'negative': '负面提示词，使用逗号分隔的多个句子'}}\n```\n我将使用这个提示词对图片进行修改。"},
        #         {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        #     ]
        # )
        # response = vision_llm.invoke([img2img_message])
        # print(f"人物 {person['name']} 的立绘检查结果：{response.content}")
        # if "是" in response.content:
        #     print(f"人物 {person['name']} 的立绘生成成功，图片路径为：{img_path}")
        # else:
        #     # 重新生成人物立绘
        #     result = extract_json(response)
        #     update_img_path = run_img2img_workflow(input_image=img_path, positive=result["positive"], negative=result["negative"])
        #     print(f"人物 {person['name']} 的立绘修改成功，图片路径为：{update_img_path}")

# 调用大模型生成场景图片，采用1280x720分辨率生成背景
def scene_generator_agent(scenes: List[str], prefix: str, server: str = "http://127.0.0.1:8188") -> None:
    """
    该智能体可以根据场景信息生成对应的背景图
    Args:
        scenes (List[str]): 场景信息列表

    Returns:
        None
    """
    previous_scene = ""
    for scene in scenes:
        text2img_message = HumanMessage(
            content=[
                {"type": "text", "text": f"我会给你提供一个小说描写的场景信息以及这个场景的上一个场景，然后你需要结合当前场景信息以及上一个场景，生成一个使用FLUX-dev模型的文生图提示词，这个提示词应当具备正面提示词与负面提示词两个部分的内容，以更好地生成符合场景描述的背景图。\
                请使用简洁清晰的英文短句来构建提示词，注意你可以扩展提供的场景描述，并且结合场景描述中的细节等构造出一个场景形象的描述，有可能场景描述中有人物的信息，但是你在生成提示词时需要有意去掉人物信息，只生成没有人物的场景。请尽量详细描写正面提示词，\
                尽量从各种角度完善场景形象，控制在10句以上；而负面提示词尽量使用那些有利于场景生成、减少人物生成的常见负面提示词，并且针对场景形象做出适应的改变。请你使用如下的\
                JSON格式返回提示词：```json\n{{'positive': '正面提示词，使用逗号分隔的多个句子', 'negative': '负面提示词，使用逗号分隔的多个句子'}}\n```\n，当前场景的信息如下{scene}，上一场景的信息如下{previous_scene}。请注意，当前场景信息可能会从上一场景变化，当然也可能毫无关系，因此你需要根据当前场景信息和上一场景信息来生成提示词。"}
            ]
        )
        previous_scene = scene
        logger.info(f"正在为场景 {scene} 生成提示词...")
        response = llm.invoke([text2img_message])
        logger.debug(f"场景 {scene} 的提示词生成结果：{response.content}")
        result = extract_json(response)
        logger.info(f"正在为场景 {scene} 生成背景图...")
        img_path = run_comfy_workflow(server=server, positive=result["positive"], negative=result["negative"], width=910, height=512, prefix=prefix) # type: ignore
        # 使用编号为场景名称的图片名称
        if img_path:
            scene_img_path = img_path.with_name(f"bg {scenes.index(scene)}.png")
            img_path.rename(scene_img_path)
        logger.info(f"场景 {scene} 的背景图生成成功，图片路径为：{scene_img_path}")

# 背景音乐生成
def music_gen(musics: List[str], prefix: str, server: str = "http://127.0.0.1:8188") -> None:
    """从每一章节的脚本中生成适合的音乐

    Args:
        musics (List[str]): 需求生成音乐的章节内容
        prefix (str): 最终存储位置
    """
    for index, music in enumerate(musics):
        if music.strip() == "":
            continue  # TODO: 处理空行！
        prompt = f"我会给你提供一篇视觉小说脚本，然后你需要结合提供的脚本内容，生成一个使用stable-audio模型的音乐生成提示词，这个提示词应当具备正面提示词与负面提示词两个部分的内容，以更好地生成适合脚本演绎的背景音乐。\
                请使用简洁清晰的英文短句来构建提示词，一个例子是Soulful Boom Bap Hip Hop instrumental, Solemn effected Piano, SP-1200, low-key swing drums, sine wave bass, Characterful, Peaceful, Interesting, well-arranged composition, 90 BPM，\
                请注意这个背景音乐需要在脚本演绎时播放，因此你需要考虑什么样子的音乐适合，我初步考虑是尽量不要有人声的，负面提示词简要描写即可，要求不多。请你使用如下的\
                JSON格式返回提示词：```json\n{{'positive': '正面提示词，使用逗号分隔的多个句子', 'negative': '负面提示词，使用逗号分隔的多个句子'}}\n```\n，当前脚本章节信息如下{music}"
        response = llm.invoke(prompt)
        logger.debug(f"生成音乐提示词的LLM结果：{response}")
        result = extract_json(response)
        if result:
            music_path = run_audio_workflow(server=server, prefix=prefix, positive=result["positive"], negative=result["negative"], duration=60.0) # type: ignore
            if music_path:
                result_path = music_path.with_name(f"music{index}.mp3")
                music_path.rename(result_path)
            else:
                raise Exception("音乐生成失败")
            logger.info(f"音乐生成完成，结果保存在：{result_path}")
            
# 脚本转化 TODO: 处理脚本中各种语法问题，在生成结束后保证下限
def convert_script(script_path: Path) -> Path:
    """把带有xml标签的脚本转化为renpy格式脚本

    Args:
        script_path (Path): 给定的脚本路径
    
    Returns:
        Path: 转换后的脚本路径
    """
    output_path = script_path.with_name("script").with_suffix(".rpy")
    console = Console()
    with open(script_path, "r", encoding="utf-8") as f:
        script_content = f.read()
        logger.info(f"正在转化脚本：{script_path}")
        with console.status("[bold cyan]正在转化角色名...[/]", spinner="dots"):
            person_pattern = r"<person>(.*?)</person>"
            persons = re.findall(person_pattern, script_content)
            for person in persons:
                script_content = replace_first(script_content, f"<person>{person}</person>", f"show {person}")
            console.print(f"[green]转化角色名完成![/green]")
        with console.status("[bold cyan]正在转化背景描述...[/]", spinner="dots"):
            scene_pattern = r"<scene>(.*?)</scene>"
            scenes = re.findall(scene_pattern, script_content)
            # 创建去重参考列表
            scene_refs = list(set(scenes))
            for scene in scenes:
                script_content = replace_first(script_content, f"<scene>{scene}</scene>", f"scene bg {scene_refs.index(scene)}")
            console.print(f"[green]转化背景描述完成![/green]")
        with console.status("[bold cyan]转化音乐描述中，请稍候…[/]", spinner="dots"):
            title_pattern = r"<chapter>(.*?)</chapter>"
            titles = re.findall(title_pattern, script_content)
            for index, title in enumerate(titles):
                script_content = replace_first(script_content, f"<chapter>{title}</chapter>", f"play music music{index}")
            console.print(f"[green]转化音乐描述完成![/green]")
    logger.info("开始写入输出文档")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("label start:\n    ")
        f.write(script_content.replace("\n", "\n    "))
    logger.info(f"输出文档已保存到：{output_path}")
    return output_path

def tag_by_dialogue(src: Path, dst: Path) -> None:
    order_many = ["middle", "left", "right", "left", "right"]
    lines = src.read_text(encoding="utf-8").splitlines()
    out, i, n = [], 0, len(lines)

    while i < n:
        # ---------- scene 行 ----------
        if re.match(r'\s*scene\b', lines[i]):
            scene_line = f'{lines[i]} at fullscreen_cover'
            out.append(scene_line)
            i += 1
            continue

        # ---------- 一个 scene 区块 ----------
        block = []
        while i < n and not re.match(r'\s*scene\b', lines[i]):
            block.append(lines[i]); i += 1

        # ---------- 统计说话角色 ----------
        talkers = [m.group(1) for ln in block if (m := re.match(r'\s*"(.*?)"\s', ln))]
        cnt = Counter(talkers)
        uniq = list(cnt)
        tag_map = {}

        if len(uniq) == 1:
            tag_map[uniq[0]] = "middle"
        elif len(uniq) == 2:
            a, b = cnt.most_common()
            tag_map[a[0]] = "left"
            tag_map[b[0]] = "right"
        else:
            for idx, (name, _) in enumerate(cnt.most_common()):
                tag_map[name] = order_many[idx % len(order_many)]

        # ---------- 给 show 行加标记 ----------
        for ln in block:
            if ln.strip().startswith("show "):
                parts = ln.split()
                if len(parts) > 1:
                    tag = tag_map.get(parts[1])
                    if tag:
                        ln = f'{ln} at {tag}'
            out.append(ln)

    Path(dst).write_text("\n".join(out), encoding="utf-8")

def concat(dst: Path, *srcs: Path, chunk: int = 1 << 20) -> None:
    """
    顺序把 *srcs 内容写入 dst。
    - 若 dst 也在 srcs：先写入同目录临时文件，再原子替换回 dst。
    - 流式复制，默认块大小 1 MiB，可用 chunk 调整。
    """
    dst = Path(dst).resolve()

    # 统一把 srcs 转成 list[Path]，避免类型警告
    src_paths: list[Path] = [p.resolve() for p in srcs]

    # 如果目标文件也在输入列表，使用临时文件中转
    need_tmp = dst in src_paths
    target = dst

    if need_tmp:
        tmp = tempfile.NamedTemporaryFile(delete=False,
                                          dir=dst.parent,
                                          suffix=".tmp")
        tmp.close()                    # 关闭句柄，交给 Path 使用
        target = Path(tmp.name)

    # 逐块拷贝
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as w:
        for src in src_paths:
            with src.open("rb") as r:
                shutil.copyfileobj(r, w, length=chunk)

    # 如用中转文件，则原子替换回目标
    if need_tmp:
        target.replace(dst)

if __name__ == "__main__":
    console = Console()
    file_path = "novels/乡村教师.txt"
    chapters = split_chapter(parse_novel_txt(path=file_path))
    # print(chapters)
    result = ""
    person_list = []
    # 增加rich加载
    start_time = time.time()
    with console.status("[bold cyan]小说脚本与人物 正在生成，请稍候…[/]", spinner="dots"):
        for chapter in chapters:
            # 在每一章开始时，增加一个标记，用来准备音乐生成
            result += f"\n<chapter>{chapter.title}</chapter>\n"
            for chunk in chapter.chunks: # type: ignore
                person_list = generate_person(chunk, llm, person_list)
                result += generate_script(chunk, llm, person_list, previous_script=result) + "\n"
        console.print(f"最终人物：{person_list}")
        console.print(f"最终脚本：{result[:1000]}...")  # 只打印前1000个字符
        console.print(f"[bold green]小说脚本与人物生成完成！用时{time.time() - start_time:.2f}秒[/]")
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    label = Path(file_path).name.split(".")[0] + f"_{date}"  # 获取文件名加时间作为标签
    # 在这里写一下预期的输出路径结构
    # 脚本输出路径：outputs/{label}/script.txt
    # 人物输出路径：outputs/{label}/person.json
    # 图片输出路径：outputs/{label}/images/xxx.png
    # 音频输出路径：outputs/{label}/audio/xxx.mp3
    base_dir = Path("outputs") / label            # outputs/<label> 目录
    base_dir.mkdir(parents=True, exist_ok=True)   # 若不存在则递归创建

    script_path = base_dir / "script.txt"
    person_path = base_dir / "person.json"

    # —— 1. 追加写入脚本 ——  
    with script_path.open("a", encoding="utf-8") as f:  # append 模式
        f.write(result)

    # —— 2. 覆盖写入人物信息 ——  
    person_json = json.dumps(
        [p.model_dump() for p in person_list],
        ensure_ascii=False,
        indent=4
    )
    person_path.write_text(person_json, encoding="utf-8")
    label = "乡村教师_2025-06-08-01-07-48"
    script_path = Path(f"outputs/{label}/script.txt")
    person_path = Path(f"outputs/{label}/person.json")
    output_path = Path(f"outputs/{label}/script.rpy")
    result = extract_info_from_script(script_path, person_path)
    print(result.music)
    console.print(f"角色共 {len(result.persons)} 个")
    image_generator_agent(result.persons, prefix=label)
    console.print(f"场景共 {len(result.scenes)} 个")
    scene_generator_agent(result.scenes, prefix=label)
    console.print(f"音乐共 {len(result.music)} 个")
    music_gen(result.music, prefix=label)
    output_path = convert_script(script_path)
    # 根据说话人情况调整人物位置
    
    tag_by_dialogue(output_path, output_path)
    concat(output_path, Path("head.rpy"), output_path)