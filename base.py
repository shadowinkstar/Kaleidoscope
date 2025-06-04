from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
import time, re, json, base64
from loguru import logger
from prompt import EXTRACT_PERSON_PROMPT, GENERATE_SCRIPT_PROMPT
from img import run_comfy_workflow, run_img2img_workflow


load_dotenv()

llm = ChatOpenAI(
    model="qwen3-235b-a22b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
    else:
        with open(path, "r", encoding="utf-8") as f:
            novel_name = path.split(".")[0]
            print(path.split("."))
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

def split_chapter(chapters: List[Chapter], chunk_size: int = 4000, overlap: int = 0) -> List[Chapter]:
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
    description: str = Field(description="对于人物形象的全面描述，")
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
    result = extract_json(chain.invoke({"text": chapter_document.page_content, "person_list": input_person_list}))
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

def generate_script(chapter_document: Document, llm: ChatOpenAI, person_list: List[Person]) -> str:
    """
    根据小说章节内容生成对话脚本，并返回对话脚本
    :param chapter_document: 小说章节内容
    :param llm: 使用的LLM模型
    :return: 对话脚本
    """
    prompt = PromptTemplate.from_template(GENERATE_SCRIPT_PROMPT)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"text": chapter_document.page_content, "person_list": [person.name for person in person_list]})
    logger.info(f"{chapter_document.page_content[:10].strip()}...{chapter_document.page_content[-10:].strip()}提取脚本结果：\n{result}")
    return result

# 提取脚本中人物与场景信息，准备进行图像生成
def extract_info_from_script(script_path: str, person_path: str, script: str = "") -> tuple[List[dict], List[str]]:
    """
    从脚本中提取人物和场景信息
    :param script: 对话脚本
    :param script_path: 对话脚本文件路径
    :param person_path: 人物信息文件路径
    :return: 提取的人物和场景信息列表
    """
    if script == "":
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                script = f.read()
        except Exception as e:
            logger.error(f"读取脚本文件失败：{e}")
            return []
    if person_path:
        try:
            with open(person_path, "r", encoding="utf-8") as f:
                person_list = json.load(f)
        except Exception as e:
            logger.error(f"读取人物信息文件失败：{e}")
            return []
    # 使用正则表达式提取<person>标签中的内容
    person_pattern = r"<person>(.*?)</person>"
    persons = re.findall(person_pattern, script)
    print(f"提取到的人物信息：{persons}")
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
                    person["labels"].append(label)
                    break
    print(f"冲突人物信息：{conflit_persons}")
    print(f"当前人物信息：{person_list}")
        
    
    # 使用正则表达式提取<scene>标签中的内容
    scene_pattern = r"<scene>(.*?)</scene>"
    scenes = re.findall(scene_pattern, script)
    print(f"提取到的场景信息：{scenes}")
    
    return person_list, scenes

# 调用大模型生成人物立绘文生图的提示词然后调用文生图工具并查看生成图像进行图生图优化
def image_generator_agent(persons: List[dict]) -> List[str]:
    """
    该智能体可以根据人物信息生成对应的立绘
    Args:
        persons (List[dict]): 人物信息列表

    Returns:
        List[str]: 立绘生成图片路径
    """
    print(persons)
    for person in persons:
        text2img_message = HumanMessage(
            content=[
                {"type": "text", "text": "我会给你提供一个人物的信息，然后你需要结合人物信息生成一个使用FLUX-dev模型的文生图提示词，这个提示词应当具备正面提示词与负面提示词两个部分的内容，以更好地生成符合人物描述的立绘。\
                    请使用简洁清晰的英文短句来构建提示词，注意你可以扩展提供的人物描述，并且结合人物描述中的经历等构造出一个人物形象的描述，不要涉及太多人物性格等描述，并且重心放在人物描述上面，减少对画面背景的描写。请尽量详细描写正面提示词，\
                    尽量从各种角度完善人物形象，控制在10句以上；而负面提示词尽量使用那些有利于图像生成的常见负面提示词，并且针对人物形象做出适应的改变。请你使用如下的\
                    JSON格式返回提示词：```json\n{{'positive': '正面提示词，使用逗号分隔的多个句子', 'negative': '负面提示词，使用逗号分隔的多个句子'}}\n```\n，人物的信息如下{}".format(json.dumps(person))}
            ]
        )
        print(f"正在为人物 {person['name']} 生成提示词...")
        response = vision_llm.invoke([text2img_message])
        print(f"人物 {person['name']} 的提示词生成结果：{response.content}")
        result = extract_json(response)
        print(f"正在为人物 {person['name']} 生成立绘...")
        img_path = run_comfy_workflow(positive=result["positive"], negative=result["negative"])
        base64_image = encode_image(img_path)
        img2img_message = HumanMessage(
            content=[
                {"type": "text", "text": f"下面的图片是一个人物立绘，请查看这张图片是否符合如下人物描述：{json.dumps(person)}\n如果符合，请回复'是'，如果不符合，请根据人物描述修改提示词，使用\
                如下的JSON格式返回提示词：```json\n{{'positive': '正面提示词，使用逗号分隔的多个句子', 'negative': '负面提示词，使用逗号分隔的多个句子'}}\n```\n我将使用这个提示词对图片进行修改。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ]
        )
        response = vision_llm.invoke([img2img_message])
        print(f"人物 {person['name']} 的立绘检查结果：{response.content}")
        if "是" in response.content:
            print(f"人物 {person['name']} 的立绘生成成功，图片路径为：{img_path}")
        else:
            # 重新生成人物立绘
            result = extract_json(response)
            update_img_path = run_img2img_workflow(input_image=img_path, positive=result["positive"], negative=result["negative"])
            print(f"人物 {person['name']} 的立绘修改成功，图片路径为：{update_img_path}")

if __name__ == "__main__":
    # file_path = "朝闻道.txt"
    # chapters = split_chapter(parse_novel_txt(path=file_path))
    # # print(chapters)
    # result = ""
    # person_list = []
    # for chapter in chapters:
    #     for chunk in chapter.chunks:
    #         person_list = generate_person(chunk, llm, person_list)
    #         result += generate_script(chunk, llm, person_list)
    # print(f"最终人物：{person_list}")
    # with open(f"scripts/{file_path.split('.')[0]}_result_{str(int(time.time()))}.txt", "a", encoding="utf-8") as f:
    #     f.write(result)

    # with open(f"scripts/{file_path.split('.')[0]}_person_{str(int(time.time()))}.json", "w", encoding="utf-8") as f:
    #     json.dump([person.model_dump() for person in person_list], f, ensure_ascii=False, indent=4)
    script_path = "scripts/朝闻道_result_1748873516.txt"
    person_path = "scripts/朝闻道_person_1748873516.json"
    persons, scenes = extract_info_from_script(script_path, person_path)
    image_generator_agent(persons[:3])
