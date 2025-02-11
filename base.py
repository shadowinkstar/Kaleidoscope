from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print
from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from bs4 import BeautifulSoup
import time, re, json
from loguru import logger
from prompt import EXTRACT_PERSON_PROMPT, GENERATE_SCRIPRT_PROMPT

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    max_retries=2,
    temperature=0.0,
    max_output_tokens=8192)

class Chapter(BaseModel):
    """小说章节类"""
    title: str = Field(description="章节标题")
    content: str = Field(description="章节内容")
    chunks: List[Document] = Field(default=None, description="使用langchain切分后的章节内容")
    novel_name: str = Field(description="章节来源的小说名称")
    

# 处理文本
def parse_novel_txt(path: str = None, context: str = None) -> List[Chapter]:
    """
    处理小说文本，将小说文本切分成章节，并返回识别的章节列表，这里暂时不涉及到智能的章节划分，
    使用自定义的xml标签进行识别划分
    :param path: 小说文本路径
    :param context: 或者直接传入小说文本内容
    :return: 识别出的小说章节列表
    """
    if context and path:
        raise ValueError("path和context不能同时传入")
    # 最终输出列表
    chapters_result = []
    if context:
        novel_text = context
        soup = BeautifulSoup(novel_text, "lxml")
        if soup.find("novel") and soup.find("novel").text:
            novel_name = soup.find("novel").text
        else:
            logger.warning("小说名称未识别，使用默认名称")
            novel_name = f"在时间{time.strftime('%Y-%m-%d %H:%M:%S')}传入小说"
        chapters = soup.find_all("chapter")
        for chapter in chapters:
            if chapter.find("content") and chapter.find("content").text:
                content = chapter.find("content").text
            else:
                logger.warning("章节内容识别失败，跳过该章节")
                continue
            if chapter.find("title") and chapter.find("title").text:
                title = chapter.find("title").text
            else:
                logger.warning("章节标题识别失败，使用该章节前10个字符作为标题")
                title = content[:10]
            chapters_result.append(Chapter(title=title, content=content, novel_name=novel_name))
    else:
        with open(path, "r", encoding="utf-8") as f:
            novel_name = path.split(".")[0]
            print(path.split("."))
            novel_text = f.read()
            soup = BeautifulSoup(novel_text, "lxml")
            if soup.find("novel") and soup.find("novel").text:
                # 优先采用文中标题
                logger.warning("使用文本标题标签内容作为小说标题")
                novel_name = soup.find("novel").text
            chapters = soup.find_all("chapter")
            for chapter in chapters:
                if chapter.find("content") and chapter.find("content").text:
                    content = chapter.find("content").text
                else:
                    logger.warning("章节内容识别失败，跳过该章节")
                    continue
                if chapter.find("title") and chapter.find("title").text:
                    title = chapter.find("title").text
                else:
                    logger.warning("章节标题识别失败，使用该章节前10个字符作为标题")
                    title = content[:10]
                chapters_result.append(Chapter(title=title, content=content, novel_name=novel_name))
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
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches][0]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

def generate_person(chapter_document: Document, llm: ChatGoogleGenerativeAI) -> List[Person]:
    """
    根据小说章节内容生成其中出现的人物对象，并返回人物对象列表
    :param chapter_document: 小说章节内容
    :param llm: 使用的LLM模型
    :return: 人物描述列表
    """
    prompt = PromptTemplate.from_template(EXTRACT_PERSON_PROMPT)
    # print(prompt.invoke({"text": chapter_document.page_content}))
    chain = prompt | llm 
    result = extract_json(chain.invoke({"text": chapter_document.page_content}))
    logger.info(f"{chapter_document.page_content[:10].strip()}...{chapter_document.page_content[-10:].strip()}提取人物结果：\n{result}")
    person_list = []
    for i in result:
        person = Person(**i, novel_name=chapter_document.metadata["novel_name"])
        person_list.append(person)
    return person_list

def generate_script(chapter_document: Document, llm: ChatGoogleGenerativeAI, person_list: List[Person]) -> str:
    """
    根据小说章节内容生成对话脚本，并返回对话脚本
    :param chapter_document: 小说章节内容
    :param llm: 使用的LLM模型
    :return: 对话脚本
    """
    prompt = PromptTemplate.from_template(GENERATE_SCRIPRT_PROMPT)
    chain = prompt | llm
    result = chain.invoke({"text": chapter_document, "person_list": [person.name for person in person_list]})
    logger.info(f"{chapter_document.page_content[:10].strip()}...{chapter_document.page_content[-10:].strip()}提取脚本结果：\n{result}")
    return result

if __name__ == "__main__":
    chapters = split_chapter(parse_novel_txt(path="朝闻道.txt"))
    # print(chapters)
    result = ""
    for chapter in chapters:
        for chunk in chapter.chunks:
            person_list = generate_person(chunk, llm)
            result += generate_script(chunk, llm, person_list).content
    with open(f"result_{str(int(time.time()))}.txt", "a", encoding="utf-8") as f:
        f.write(result)
        
