import os
import threading
import time
from datetime import datetime
import random
from http import HTTPStatus
from dashscope import Generation
from dotenv import load_dotenv

from dotenv import load_dotenv
from openai import AzureOpenAI
from logic_folder.数据库表格 import Statistic, AbnormalCases
import json
import dashscope
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

azure_openai_endpoint = "https://gwgs-openai.openai.azure.com/"
azure_openai_key = "0d28635b8c2a41d9be84331e36a83453"
azure_api_version = "2024-12-01-preview"
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

azure_client = AzureOpenAI(
    api_version=azure_api_version,
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
)
# 配置数据库连接
def create_db_engine(echo=False):
    engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}', echo=echo)
    return engine


def new_create_db_session():
    # 创建数据库引擎
    engine = create_db_engine(False)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session, engine
# ai调用工具

def make_request_with_retry(client, message, case_request=False, ai="qwen", model="qwen-max"):
    retry = 0
    while retry < 20:
        result_container = {}
        if case_request:
            make_api_call(client, message, result_container, True, ai, model)
        else:
            make_api_call(client, message, result_container, False, ai, model)
        if "error" in result_container:
            if "Request timed out" in str(result_container["error"]):
                retry += 1
                print(f"请求超时，正在重试第{retry}...")
                continue
            if "rate_limit_reached_error" in str(result_container["error"]):
                retry += 1
                e = result_container["error"]
                print(f"请求频率过高:{e}，正在重试第{retry}...")
                time.sleep(10)
                continue
            if "high risk" in str(result_container["error"]):
                print(f"敏感信息，切换openai，重试第{retry}次")
                continue

            raise result_container["error"]

        # Add error handling for missing input_money field
        if "input_money" not in result_container:
            result_container["input_money"] = 0
        return {'result': result_container["result"], 'input_money': result_container["input_money"]}

    raise Exception("请求超时，已经重试20次，仍然失败，网络可能出现问题")


def call_qwen_api(client, message, result_container, json_mode, model):
    # print(f"call_qwen_api: type of message = {type(message)}")
    if not isinstance(message, list):
        # print("call_qwen_api: message is not a list, attempting to load if string.")
        try:
            # 确保 message 是字符串类型才能调用 json.loads
            if isinstance(message, str):
                message = json.loads(message)
                # print(f"call_qwen_api: type of message after loads = {type(message)}")
            else:
                # 如果不是字符串，也不能加载，记录错误或按原样传递，取决于期望的行为
                print(f"call_qwen_api: message is not a string, cannot apply json.loads. Type is {type(message)}")
                # 根据您的逻辑，这里可能应该直接报错或有其他处理
        except Exception as e:
            # print(f"call_qwen_api: failed to load message string: {e}")
            result_container["error"] = ValueError("Message is not a list and could not be parsed from JSON string")
            return result_container

    if not isinstance(message, list) or not all(isinstance(item, dict) for item in message):
        # print(f"call_qwen_api: Message must be a list of dicts, but got {type(message)} with content: {message}")
        result_container["error"] = TypeError(f"Message must be a list of dicts, but got {type(message)}")
        return result_container

    if json_mode:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            temperature=0.01,
            messages=message,
            timeout=240
        )
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=0.01,
            messages=message,
            timeout=240
        )
    usage = response.usage

    result_container["usage"] = usage

    result_container[
        'input_money'] = usage.prompt_tokens / 1000 * 0.0024 + usage.completion_tokens / 1000 * 0.0096

    result_container["result"] = response.choices[0].message.content
    return result_container
def call_azure_client(message, result_container, model_deployment_name="gpt-4o-mini"):
    """
    Calls the Azure OpenAI API.

    Args:
        message (str): The message to send to the API, expected to be a JSON string.
        result_container (dict): A dictionary to store the results.
        model_deployment_name (str): The deployment name of the model on Azure.
    """
    try:
        response = azure_client.chat.completions.create(
            messages=json.loads(message), # Assuming message is a JSON string of messages
            temperature=0.1,
            top_p=0.1,
            model=model_deployment_name  # Use the deployment name
        )
        # Assuming the response structure is similar to OpenAI's
        # and that usage information might not be directly available or needed for this setup.
        # If usage is available and needed, it should be extracted here.
        # For now, we'll focus on getting the content.
        result_container["result"] = response.choices[0].message.content
        # If cost calculation is needed, it should be implemented based on Azure's pricing.
        # result_container['input_money'] = calculate_azure_cost(...)
    except Exception as e:
        result_container["error"] = e
    return result_container


def make_api_call(client, message, result_container, json_mode, ai='qwen', model='qwen-max-latest'):
    try:
        if ai == "qwen":
            call_qwen_api(client, message, result_container, json_mode, model)
        elif ai == "azure":
            call_azure_client(message, result_container, model)
        else:
            raise ValueError("Unsupported AI API")
    except Exception as e:
        result_container["error"] = e
    return result_container


#

def record_money(total_usage, db, is_report=False, mission_type=None, is_embedding=None,is_extra=False):
    now = datetime.now()
    time_stamp = now.strftime('%Y-%m-%d')

    retry = 0
    while retry < 5:
        try:

            ###更新stat
            total_stat = db.query(Statistic).filter_by(date="全量").with_for_update().first()
            if not total_stat:
                total_stat = Statistic(date="全量", report_cost=0, version="标准", indicator_num=0,
                                       indicator_cost=0,
                                       report_num=0, summary_type=0, summary_type_cost=0, probe_type=0,
                                       probe_type_cost=0, logic_type=0, logic_type_cost=0, calculation_type=0,
                                       calculation_type_cost=0, embedding_cost=0)

            if is_embedding:
                total_stat.embedding_cost += total_usage
            elif is_extra:
                total_stat.report_cost += total_usage
            else:
                if is_report:
                    total_stat.report_num += 1
                    total_stat.report_cost += total_usage
                else:
                    total_stat.indicator_num += 1
                    total_stat.indicator_cost += total_usage

            db.add(total_stat)

            today_stat = db.query(Statistic).filter_by(date=time_stamp).with_for_update().first()
            if not today_stat:
                today_stat = Statistic(date=time_stamp, report_cost=0, version="标准", indicator_num=0,
                                       indicator_cost=0,
                                       report_num=0, summary_type=0, summary_type_cost=0, probe_type=0,
                                       probe_type_cost=0, logic_type=0, logic_type_cost=0, calculation_type=0,
                                       calculation_type_cost=0, embedding_cost=0)

            if is_embedding:
                today_stat.embedding_cost += total_usage
            elif is_extra:
                today_stat.report_cost += total_usage
            else:
                if is_report:
                    today_stat.report_num += 1
                    today_stat.report_cost += total_usage

                else:
                    today_stat.indicator_num += 1
                    today_stat.indicator_cost += total_usage

            if mission_type:
                if mission_type == "总结型":
                    total_stat.summary_type += 1
                    total_stat.summary_type_cost += total_usage
                    today_stat.summary_type += 1
                    today_stat.summary_type_cost += total_usage
                elif mission_type == "探针型":
                    total_stat.probe_type += 1
                    total_stat.probe_type_cost += total_usage
                    today_stat.probe_type += 1
                    today_stat.probe_type_cost += total_usage
                elif mission_type == "逻辑型":
                    total_stat.logic_type += 1
                    total_stat.logic_type_cost += total_usage
                    today_stat.logic_type += 1
                    today_stat.logic_type_cost += total_usage
                elif mission_type == "计算型":
                    total_stat.calculation_type += 1
                    total_stat.calculation_type_cost += total_usage
                    today_stat.calculation_type += 1
                    today_stat.calculation_type_cost += total_usage

            db.add(today_stat)
            db.commit()
            break
        except Exception as e:
            db.rollback()
            # 等随机1-3秒
            time.sleep(random.randint(1, 2))
            retry += 1

    if retry == 5:
        print(f"算钱的时候有bug")

def safe_get_input_money(completion):
    """
    Safely extracts input_money from completion object with error handling.
    
    Args:
        completion: The completion result from API call
        
    Returns:
        float: The input_money value, or 0 if not found
    """
    try:
        return completion.get('input_money', 0)
    except Exception as e:
        print(f"获取input_money时出错: {e}")
        return 0
