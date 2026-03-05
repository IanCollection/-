import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
global_logger = logging.getLogger('global')

from dotenv import load_dotenv
from sqlalchemy import func
from sqlalchemy.exc import OperationalError

from logic_folder.数据库表格 import DataRecord

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


def data_recorder(data,indicator_collection = False):
    db, engine = create_db_session()
    try:
        question_type = str(data['题目类型'])
        question = data['question']
        answer = str(data['answer'])
        necessary_infos = data['necessary_infos']
        if indicator_collection:
            indicator = str(data['indicator'])
        else:
            indicator = None
        new_data_record = DataRecord(question_type=question_type, question=question, answer=answer, indicator=indicator,
                                     necessary_infos=necessary_infos)
        db.add(new_data_record)
        db.commit()

    except Exception as e:
        print(f"样本记录发生错误: {e}")
    finally:
        db.close()
        engine.dispose()


class Logger(object):
    def __init__(self, file, type_name="测试样本.json", dir_path="."):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = f"{file}_{type_name}"
        self.file_path = os.path.join(dir_path, file_name)

        # 如果文件不存在，初始化一个空的 JSON 数组
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)  # 写入空数组

        # 确保文件在 1GB 限制内
        while self.get_dir_size(dir_path) > 1 * (10 ** 9):
            self.delete_oldest_file(dir_path)

    @staticmethod
    def get_dir_size(path="."):
        total = 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += Logger.get_dir_size(entry.path)
        return total

    @staticmethod
    def delete_oldest_file(path="."):
        files = [
            (f, os.path.getmtime(os.path.join(path, f)))
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        if not files:
            return
        oldest_file = min(files, key=lambda x: x[1])[0]
        os.remove(os.path.join(path, oldest_file))

    def __enter__(self):
        return self  # 在 'with' 语句中返回当前实例

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 这里无需关闭特定资源
        pass

    def write(self, data):
        if not isinstance(data, dict):
            raise ValueError("Data must be一个dictionary.")

        data["timestamp"] = datetime.now().isoformat()  # 添加时间戳

        # 读取现有的 JSON 数组
        with open(self.file_path, "r") as f:
            existing_data = json.load(f)

        # 将新的数据追加到数组中
        existing_data.append(data)

        # 写回 JSON 文件
        with open(self.file_path, "w") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

    def flush(self):
        pass


def split_long_strings(strings, max_length=1000):
    result = []
    for s in strings:
        # 如果字符串的长度小于或等于最大长度，直接添加到结果中
        if len(s) <= max_length:
            result.append(s)
        else:
            # 需要拆分字符串
            while len(s) > max_length:
                # 找到中间位置
                mid = len(s) // 2
                # 取中间最大不超过最大长度的位置
                split_point = mid

                # 继续往前找到空格或适当的分割点
                while split_point > 0 and s[split_point] not in (' ', '\n'):
                    split_point -= 1

                # 如果没有合适的空格，直接在中间分割
                if split_point == 0:
                    split_point = mid

                # 将左侧部分加入结果，继续处理右侧部分
                result.append(s[:split_point])
                s = s[split_point:].lstrip()  # 去掉左侧多余空格

            # 最后加入剩余的部分
            if s:
                result.append(s)

    return result


def auto_retry(method, *args, retry_times=10):
    if retry_times == 0:
        return False
    try:
        return method(*args)
    except Exception as e:
        print(f"Attempt {retry_times} failed: {e}，马上开始重试")
        return auto_retry(method, *args, retry_times=retry_times - 1)


def change_file_names(root_dir):
    #
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.json'):
                underscore_index = filename.find('_')
                if underscore_index != -1:
                    new_file_name = filename[underscore_index + 1:]
                old_file_path = os.path.join(dirpath, filename)
                # 创建与新文件名相同的文件夹
                new_dir_path = root_dir
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                # 构造新文件的完整路径
                new_file_path = os.path.join(root_dir, new_file_name)
                shutil.move(old_file_path, new_file_path)
                print(f"Renamed '{old_file_path}' to '{new_file_path}'")

    return 0


def safe_get_input_money(completion_dict, default=0):
    """Safely extract input_money from API completion response, with error handling.
    
    Args:
        completion_dict: The API response dictionary
        default: Default value to return if input_money is missing
        
    Returns:
        The input_money value or default if not available
    """
    try:
        if isinstance(completion_dict, dict) and 'input_money' in completion_dict:
            return completion_dict['input_money']
        return default
    except Exception as e:
        global_logger.warning(f"Error getting input_money: {e}")
        return default
