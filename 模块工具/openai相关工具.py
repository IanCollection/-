import os
import threading
import time
import requests

from dotenv import load_dotenv
from sqlalchemy import create_engine, func
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from ian_evolution.client_manager import qwen_client
from 模块工具.API调用工具 import record_money

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')


# 配置数据库连接
def create_db_engine(echo=False):
    engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}', echo=echo,
                           isolation_level="READ COMMITTED")
    return engine


def create_db_session(max_retries=10, retry_interval=5):
    retries = 0
    while retries < max_retries:
        try:
            # 尝试创建数据库引擎和会话
            engine = create_db_engine(False)
            Session = sessionmaker(bind=engine)
            session = Session()
            return session, engine
        except OperationalError as e:
            if e.orig.args[0] == 1040:  # pymysql.err.OperationalError: (1040, "Too many connections")
                print(f"Too many connections. Retrying in {retry_interval} seconds...")
                time.sleep(retry_interval)
                retries += 1
            else:
                raise  # 重新抛出非"太多连接"错误
    raise Exception("Failed to create database session after several retries")

def count_gpt_tokens(text, tokenizer):
    if isinstance(text, str):
        # 如果text是字符串，按照目前的运行方式
        output = tokenizer.encode(text)
        return len(output)
    elif isinstance(text, list) and all(isinstance(item, str) for item in text):
        # 如果text是字符串列表，计算每个非空元素的output并相加
        total_length = 0
        for item in text:
            if item.strip():  # 检查是否是非空字符串
                output = tokenizer.encode(item)
                total_length += len(output)
        return total_length
    elif isinstance(text, list) and all(isinstance(item, list) for item in text):
        # 如果text是包含嵌套列表的列表，递归处理嵌套列表中的元素
        total_length = 0
        for item in text:
            total_length += count_gpt_tokens(item, tokenizer)
        return total_length
    elif isinstance(text, list) and all(isinstance(item, tuple) for item in text):
        # 如果text是包含元组的列表，将元组中的非空元素连接为字符串并计算output
        total_length = 0
        for item in text:
            concatenated_item = ' '.join([elem for elem in item if elem.strip()])  # 连接非空元素
            if concatenated_item:  # 检查是否是非空字符串
                output = tokenizer.encode(concatenated_item)
                total_length += len(output)
        return total_length
    elif isinstance(text, dict):
        # 如果text是字典，计算字典中每个value的output并相加
        total_length = 0
        for value in text.values():
            if isinstance(value, str):
                # 如果值是字符串，计算output
                output = tokenizer.encode(value)
                total_length += len(output)
            elif isinstance(value, list):
                # 如果值是列表，递归处理列表中的元素
                total_length += count_gpt_tokens(value, tokenizer)
            else:
                raise ValueError("Dictionary values must be strings, lists, or nested lists of strings")
        return total_length
    else:
        # 如果text的类型不是字符串、字符串列表、包含元组的列表或字典，返回错误或者采取适当的处理方式
        raise ValueError(
            "Unsupported input type: text must be a string, a list of strings, a list of tuples of strings, or a dictionary")

def get_cluster_embeddings(texts, model="text-embedding-v1"):
    """
    获取文本嵌入向量，优化的版本：
    1. 更好的错误处理和重试逻辑
    2. 批量处理更多文本
    3. 更高效的数据库交互
    """
    if not texts:
        return []

    max_retries = 5
    retry_count = 0
    backoff_factor = 1.5  # 指数退避因子

    while retry_count < max_retries:
        try:
            # 添加超时设置
            response = qwen_client.embeddings.create(
                input=texts,
                model=model,
                timeout=300  # 增加超时时间到5分钟
            )

            # 计算使用量和费用
            usage = response.usage.total_tokens
            total_usage = usage / 1000000 * 0.13 * 7.5
            # 提取并返回嵌入结果
            return [embedding.embedding for embedding in response.data]

        except Exception as e:
            # 错误类型识别和处理
            error_str = str(e)
            retry_count += 1

            # 根据错误类型决定如何处理
            if "RateLimitError" in error_str:
                # 对于速率限制错误，使用指数退避策略
                sleep_time = backoff_factor ** retry_count
                print(
                    f"Rate limit reached, retrying in {sleep_time:.1f} seconds... (attempt {retry_count}/{max_retries})")
                time.sleep(sleep_time)
            elif "timeout" in error_str.lower():
                # 超时错误，增加延迟重试
                sleep_time = backoff_factor ** retry_count
                print(f"API timeout, retrying in {sleep_time:.1f} seconds... (attempt {retry_count}/{max_retries})")
                time.sleep(sleep_time)
            else:
                # 其他错误
                if retry_count < max_retries:
                    print(f"API error: {error_str}. Retrying... (attempt {retry_count}/{max_retries})")
                    time.sleep(2)  # 基本重试延迟
                else:
                    print(f"All retries failed. Last error: {error_str}")
                    # 所有重试都失败了，返回空列表或抛出异常
                    return []

    # 超过最大重试次数
    print("Failed to get embeddings after maximum retries")
    return []

def get_cluster_embeddings_local(texts, model="bge-m3:567m"):
    """
    获取文本嵌入向量，使用本地部署的API：
    1. 更好的错误处理和重试逻辑
    2. 批量处理更多文本
    """
    if not texts:
        return []

    max_retries = 5
    retry_count = 0
    backoff_factor = 1.5
    api_url = "http://106.14.88.25:11434/api/embed"

    while retry_count < max_retries:
        try:
            payload = {
                "model": model,
                "input": texts if isinstance(texts, list) else [texts]
            }
            response = requests.post(api_url, json=payload, timeout=300)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            result = response.json()
            return [embedding['embedding'] for embedding in result['embeddings']]

        except requests.exceptions.RequestException as e:
            error_str = str(e)
            retry_count += 1

            if "Connection refused" in error_str or "timeout" in error_str.lower():
                sleep_time = backoff_factor ** retry_count
                print(f"Local API connection error or timeout, retrying in {sleep_time:.1f} seconds... (attempt {retry_count}/{max_retries})")
                time.sleep(sleep_time)
            else:
                if retry_count < max_retries:
                    print(f"Local API error: {error_str}. Retrying... (attempt {retry_count}/{max_retries})")
                    time.sleep(2)
                else:
                    print(f"All retries failed. Last error: {error_str}")
                    return []
    print("Failed to get local embeddings after maximum retries")
    return []

if __name__ == "__main__":
    texts = ["Why is the sky blue?"]
    embeddings = get_cluster_embeddings_local(texts)
    print(embeddings)