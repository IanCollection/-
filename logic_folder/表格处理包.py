import copy
import gc
import json
import os
import pickle
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
import numpy as np
from dotenv import load_dotenv
from sqlalchemy.exc import OperationalError
import time

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker

from 模块工具.openai相关工具 import get_cluster_embeddings
from logic_folder.数据库表格 import Section, Chart, Header, KeyIndex, Description, Vector, Sentences, TableValue, Word
from 模块工具.智能体仓库 import header_check_bot, same_row_checker_bot, title_seek_bot, \
    title_makeup_bot, page_check_bot, title_level_reset_bot, strange_title_check_bot, table_title_expand_bot, \
    mother_company_check_bot, special_title_check_bot, same_row_checker_bot_v2
from 模块工具.通用工具 import auto_retry, split_long_strings

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')


# 配置数据库连接
def create_db_engine(echo=False):
    engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}', echo=echo,
                           isolation_level="READ COMMITTED", pool_pre_ping=True)
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


def page_text_content(page_content):
    page_text = []
    for item in page_content:
        if item["type"] == "plain":
            for each_line in item["lines"]:
                page_text.append(each_line["text"])
    return page_text


def find_redundant_pdf_head(list_of_first5_pages):
    # First, extract the text content from each page
    pure_text_list = [page_text_content(page) for page in list_of_first5_pages]

    # Assume all pages have at least one line of text
    if not pure_text_list or not all(pure_text_list):
        return []

    # Find the shortest page to avoid index errors
    min_length = min(len(page) for page in pure_text_list)

    # Initialize an empty list to store redundant text
    redundant_text = []

    # Iterate through each line index up to the shortest page length
    for i in range(min_length):
        # Check if this line is the same across all pages
        if all(page[i] == pure_text_list[0][i] for page in pure_text_list):
            redundant_text.append(pure_text_list[0][i])
        else:
            # Stop at the first inconsistency
            break

    return redundant_text


def check_cell_repeat(row):
    count_dict = {}
    for index, cell in enumerate(row):
        if index == 0:
            continue
        if cell in count_dict:
            count_dict[cell] += 1
        else:
            count_dict[cell] = 1
    # 检查是否每个cell>2
    repeat = True
    for key in count_dict:
        if count_dict[key] < 2:
            repeat = False
            break
    return repeat


def check_if_more_headers_exist(values_list):
    more_header_idx = None
    for idx, row in enumerate(values_list):
        if check_cell_repeat(row):
            more_header_idx = idx
            break


def check_abnormal_multi_headers(header_list, values_list):
    if len(header_list) == 1 and (
            (header_list[0][0] == " " or header_list[0][0] == "") or check_cell_repeat(header_list[0])):
        new_header_idx = -1
        for idx, row in enumerate(values_list):
            if row[0] != " " and row[0] != "" and row[0] != header_list[0][0]:
                new_header_idx = idx
                break
        return new_header_idx
    else:
        return -1


def create_tables_from_data(data):
    def process_table_data(table_data, start_row, end_row, set_title=None, previous_header=None, skip_head=False):
        # Create an empty DataFrame with the determined dimensions

        max_col = max([cell['end_col'] for cell in table_data]) + 1
        df = pd.DataFrame(index=range(start_row, end_row + 1), columns=range(max_col))
        title = None
        # Populate the DataFrame using the cell data
        for cell in table_data:
            if cell["text"] == np.nan:
                cell["text"] = " "
            cell_content = cell['text'].replace('\n', '')
            # cell_content = cell['text']
            if cell['start_row'] == start_row and cell['end_col'] - cell['start_col'] == max_col - 1:
                title = cell_content
            for row in range(cell['start_row'], cell['end_row'] + 1):
                for col in range(cell['start_col'], cell['end_col'] + 1):
                    df.iat[row - start_row, col] = cell_content

        # Determine the header rows based on row expansion from the first row after the title
        header_row_index = 0 if title is None else df[df.iloc[:, 0] == title].index[0] + 1
        if skip_head:
            header_row_index = 1

        header_rows = [cell['end_row'] for cell in table_data if cell['start_row'] == header_row_index]
        max_header_row = max(header_rows, default=header_row_index)

        # Extract the full header from the main DataFrame
        if skip_head:
            header_df = df.iloc[header_row_index - 1:max_header_row]
        else:
            header_df = df.iloc[header_row_index:max_header_row + 1]
        header_df = header_df.fillna(' ')

        header = header_df.values.tolist()
        preset_header = False

        def contains_digit_or_empty(element):
            return any(char.isdigit() for char in element) or element == ''

        valid_header = all(not contains_digit_or_empty(element) for sublist in header for element in sublist)
        if previous_header and valid_header and header_check_bot(header, previous_header)[1]:
            preset_header = True

        if preset_header or not previous_header:
            # Extracting the values from the DataFrame
            if skip_head:
                values_df = df.iloc[max_header_row:]
            else:
                values_df = df.iloc[max_header_row + 1:]

        else:
            header = previous_header
            values_df = df
        values_df = values_df.dropna(how='all')
        values = values_df.values.tolist()
        cols = [val[0] for val in values if val]
        ## 空index 向下填充
        for i in range(len(cols)):
            if (cols[i] == " " or cols[i] == "") and i != 0:
                cols[i] = cols[i - 1]

        pre_header = copy.deepcopy(header)
        pre_cols = copy.deepcopy(cols)
        pre_values = copy.deepcopy(values)

        # 加一个多层级表头检查，如果某一个row，他除第一个element外的每一个element的count都是大于2的，依旧是多层级表头
        try:
            more_header_check_result = check_abnormal_multi_headers(header, values)
            if more_header_check_result != -1:
                # 更新表头，key_index和新body
                header += values[:more_header_check_result + 1]
                cols = cols[more_header_check_result + 1:]
                values = values[more_header_check_result + 1:]
        except Exception as e:
            print(f"多层级表头检查出现bug:{e}")
            print("恢复到检查前状态")
            header = pre_header
            cols = pre_cols
            values = pre_values

        # Constructing the desired dictionary structure
        if set_title:
            title = set_title
        else:
            title = None
        table = {
            "title": title,
            "unit": "NA",
            "potential_table_titles": "NA",
            "header": header,
            "key_index": cols,
            "values": values,
        }

        return table

    def is_homogeneous_or_empty(row_content):
        # 初始化一个用于比较的非空白字符串
        if len(row_content) == 1:
            return True
        comparison_string = row_content[0]
        for string in row_content[1:]:
            if string != comparison_string:
                return False
        return True

    def fix_ocr_first_row_error(first_row_content, max_column, data):
        # 用于检测ocr识别错误的第一行，如果有的话，将其修正
        detected_cell_num = first_row_content[0].count(" ") + 1
        if detected_cell_num > 1 and max_column % detected_cell_num == 0:
            # 有错误
            original_cell = data[0]
            each_correct_text = original_cell["text"].split(" ")
            ##计算每一格的长度
            each_length = max_column // detected_cell_num
            count = 0
            new_data = []
            start_index = 0
            for i in range(detected_cell_num):
                ## copy original cell
                new_cell = copy.deepcopy(original_cell)
                new_cell["start_col"] = start_index
                new_cell["end_col"] = start_index + each_length - 1
                new_cell["text"] = each_correct_text[count]
                start_index = start_index + each_length
                count += 1
                new_data.append(new_cell)

            data_fix = new_data + data[1:]
            return data_fix
        else:
            return data

    try:
        tables = []  # List to store all tables
        current_table_data = []  # Temporary storage for the current table's data
        start_row_of_current_table = 0  # Start row of the current table
        max_col = max([cell['end_col'] for cell in data]) + 1  # Define max_col based on the data
        max_row = max([cell['end_row'] for cell in data]) + 1  # Define max_row based on the data
        previous_header = None
        # Iterate over each row and decide if it's a new table start or part of the current table
        next_title = None
        new_table = None
        skip_head = False

        ## OCR识别错误处理
        ##first_row
        first_row_content = [cell['text'] for cell in data if cell['start_row'] <= 0 <= cell['end_row']]
        data = fix_ocr_first_row_error(first_row_content, max_col, data)
        for row in range(max_row):

            row_content = [cell['text'] for cell in data if cell['start_row'] <= row <= cell['end_row']]
            if is_homogeneous_or_empty(row_content):
                if row == 0 or row == 1:
                    next_title = row_content[0]
                    if row == 0:
                        start_row_of_current_table = row + 1
                        skip_head = True
                    continue
                # If the row is homogeneous or empty and not the first row, consider it as a new table
                if current_table_data:  # If there's any data collected for the current table, process it
                    if next_title:
                        new_table = process_table_data(current_table_data, start_row_of_current_table, row - 1,
                                                       set_title=next_title, previous_header=previous_header,
                                                       skip_head=skip_head)
                    else:
                        new_table = process_table_data(current_table_data, start_row_of_current_table, row - 1,
                                                       previous_header=previous_header, skip_head=skip_head)
                    next_title = row_content[0]
                    if new_table:
                        tables.append(new_table)
                        previous_header = new_table["header"]

                    # Reset the data for the new table
                    current_table_data = []
                pre_title = next_title
                start_row_of_current_table = row + 1  # The new table starts after this row
                skip_head = False
            else:
                # Add the cells of this row to the current table data
                current_table_data.extend([cell for cell in data if cell['start_row'] <= row <= cell['end_row']])

        # Process the last table if there is any data left
        if current_table_data:
            if next_title:
                tables.append(
                    process_table_data(current_table_data, start_row_of_current_table, max_row - 1,
                                       set_title=next_title,
                                       previous_header=previous_header, skip_head=skip_head))
            else:
                tables.append(process_table_data(current_table_data, start_row_of_current_table, max_row - 1,
                                                 previous_header=previous_header, skip_head=skip_head))

        # 新加一个method，如果有任何重复的column直接删掉
        # tables_to_return = []
        # for table in tables:
        #     table = table_purifier(table)
        #     tables_to_return.append(table)
        return tables
    except Exception as e:
        return []


def redundant_head_checker(redundant_head, page, last_block):
    for each_line in page[0]["lines"]:
        if each_line["text"] not in redundant_head:
            return False, []
    if "table" not in last_block.keys():
        return False, []
    else:
        if len(last_block["table"]) >= 1:
            return True, last_block["table"][-1]
        else:
            return False, []


def same_column_num_check(pre_table, current_table):
    ##检查column数量是不是相同
    if len(pre_table["header"][0]) != len(current_table["header"][0]):
        return False
    return True


def break_table_combination(pre_table, current_table, table_title):
    # 能进来这个method的，一定就是要拼接的
    related_column = False
    same_header = False
    merged_cell_case = False
    total_usage = 0
    attention_case = False

    try:
        # 用来修复ocr扫描错误的
        if pre_table["header"] == current_table["header"]:
            same_header = True
        if len(pre_table["key_index"]) == 0:
            pre_table["key_index"] = [" "]
        if len(pre_table["values"]) == 0 and len(pre_table["header"]) > 0:
            pre_table["values"] = [[" "] * len(pre_table["header"][0])]
        if len(pre_table["values"]) > 0 and len(pre_table["values"][0]) == 0 and len(pre_table["header"]) > 0:
            pre_table["values"] = [[" "] * len(pre_table["header"][0])]

        if len(current_table["key_index"]) == 0:
            current_table["key_index"] = [" "]
        if len(current_table["values"]) == 0 and len(current_table["header"]) > 0:
            current_table["values"] = [[" "] * len(current_table["header"][0])]
        if len(current_table["values"]) > 0 and len(current_table["values"][0]) == 0 and len(
                current_table["header"]) > 0:
            current_table["values"] = [[" "] * len(current_table["header"][0])]

        # 检查是否是否是同一row但是暴力拆分
        if pre_table["values"]:
            if len(pre_table["values"]) == 1 and all(element.strip() == '' for element in pre_table["values"][0]):
                answer = True
                pre_table["values"] = []
                pre_table['key_index'] = []

            else:
                # total_usage, answer, merged_cell_case, modify_index = same_row_checker_bot(pre_table, current_table,table_title,same_header)
                total_usage, answer, merged_cell_case, modify_index, attention_case = same_row_checker_bot_v2(pre_table,
                                                                                                              current_table,
                                                                                                              table_title,
                                                                                                              same_header)

        else:
            answer = True

        if not answer:
            ##检查是否存在关联
            related_column = True
        ##表格拼接
        if same_header and not related_column:
            pre_table["values"].extend(current_table["values"])
            pre_table["key_index"].extend(current_table["key_index"])
            return pre_table, None, total_usage, attention_case
        elif related_column:
            # 断表，但是保留表头，并且下一行就是断表
            if not same_header:
                # 这时候的header其实是values的一部分，要把他放回去
                for i in range(len(current_table["header"])):
                    current_table["values"].insert(i, current_table["header"][i])
                    current_table["key_index"].insert(i, current_table["header"][i][0])
            sticky_range = 0
            sticky_value = ""
            pre_key_index = None
            for each_key_index in current_table["key_index"]:
                if pre_key_index == None:
                    pre_key_index = each_key_index
                if pre_key_index == each_key_index:
                    sticky_range += 1
                else:
                    break
            if not merged_cell_case:
                # 他们是全部暴力拆分的,需要直接融合
                for i in range(len(current_table["values"][0])):
                    if isinstance(current_table["values"][0][i], str):
                        if pre_table["values"][-1][i] == current_table["values"][0][i]:
                            new_value = pre_table["values"][-1][i]
                        else:
                            new_value = pre_table["values"][-1][i] + current_table["values"][0][i]
                        pre_table["values"][-1][i] = new_value
                        if i == 0:
                            sticky_value = new_value
                if isinstance(current_table["values"][0][0], str):
                    if pre_table["key_index"][-1] != current_table["values"][0][0]:
                        pre_table["key_index"][-1] += current_table["values"][0][0]

                for i in range(sticky_range):
                    current_table["values"][i][0] = sticky_value
                    current_table["key_index"][i] = sticky_value
                pre_table["values"].extend(current_table["values"][1:])
                pre_table["key_index"].extend(current_table["key_index"][1:])
            else:
                # 所谓的merged_cell_case，就是存在合并单元格,并不是全部暴力拆分。modify_index就是合并单元格，他们需要被合并，剩下的就不需要合并

                # 将pre_table的合并单元格还原
                for i in modify_index:
                    pre_table["values"][-1][i] += current_table["values"][0][i]
                new_value = pre_table["key_index"][-1] + current_table["values"][0][0]
                pre_table["key_index"][-1] = new_value
                sticky_value = new_value

                # 将current_table的合并单元格还原
                for i in modify_index:
                    current_table["values"][0][i] = pre_table["values"][-1][i]

                # 将current_table还原后的合并单元格根据原断表继续扩展至其他格子(if applicable)
                for i in range(sticky_range):
                    current_table["key_index"][i] = sticky_value
                    current_table["values"][i][0] = sticky_value

                pre_table["values"].extend(current_table["values"])
                pre_table["key_index"].extend(current_table["key_index"])

            return pre_table, None, total_usage, attention_case

        else:

            for j in range(len(current_table["header"])):
                if isinstance(current_table["header"][j][0], str):
                    pre_table["values"].append(current_table["header"][j])
                    pre_table["key_index"].append(current_table["header"][j][0])
            if len(current_table["values"]) > 0:
                if len(current_table["values"]) == 1 and not all(
                        element.strip() == '' for element in current_table["values"][0]):
                    pre_table["values"].extend(current_table["values"])
                elif len(current_table["values"]) > 1:
                    pre_table["values"].extend(current_table["values"])
            if len(current_table["key_index"]) > 0:
                if len(current_table["key_index"]) == 1 and not current_table["key_index"][0].strip() == '':
                    pre_table["key_index"].extend(current_table["key_index"])
                elif len(current_table["key_index"]) > 1:
                    pre_table["key_index"].extend(current_table["key_index"])

            return pre_table, None, total_usage, attention_case
    except Exception as e:
        print("table合并出现bug")
        print(e)
        print(pre_table)
        print(current_table)
    return pre_table, current_table, 0, attention_case


def generate_descriptions_with_prefix(parsed_dict):
    header = parsed_dict['header']
    deep = len(header)
    descriptions = []
    # 使用字典来存储描述和其对应的locat列表
    description_dict = {}

    # Check if header is multi-level or single-level
    if isinstance(header[0], list):  # Multi-level header
        # Prefix from header[0][0]
        prefix = header[0][0]

        # Iterate over each column starting from the second column (index 1)
        for j in range(1, len(header[0])):
            for key_index in range(len(parsed_dict['key_index'])):
                description_parts = []
                for i in range(deep):
                    # If the header is not the same as the previous level and is not the same as the key_index value, append it to the description
                    if i == 0 or (
                            header[i][j] != header[i - 1][j] and header[i][j] != parsed_dict['key_index'][key_index]):
                        description_parts.append(header[i][j])
                description = f"{prefix}: {parsed_dict['key_index'][key_index]}的{'的'.join(description_parts)}"
                # 如果描述已存在，将locat添加到现有列表中
                if description in description_dict:
                    description_dict[description].append([key_index, j])
                else:
                    description_dict[description] = [[key_index, j]]
    else:  # Single-level header
        # Prefix from header[0]
        prefix = header[0]

        # Iterate over each column
        for j in range(1, len(header)):
            for key_idx in range(len(parsed_dict['key_index'])):
                # If the header is not the same as the key_index value, append it to the description
                if header[j] != parsed_dict['key_index'][key_idx]:
                    description = f"{prefix}: {parsed_dict['key_index'][key_idx]}的{header[j]}"
                    # 如果描述已存在，将locat添加到现有列表中
                    if description in description_dict:
                        description_dict[description].append([key_idx, j])
                    else:
                        description_dict[description] = [[key_idx, j]]

    # 将description_dict转换为descriptions列表，包括去重后的描述和对应的locat列表
    for description, locats in description_dict.items():
        descriptions.append({
            "description": description,
            "locat": locats
        })

    return descriptions


def clear_html_tag(text):
    markup = text

    soup = BeautifulSoup(markup, features="lxml")
    return soup.get_text()


def table_title_and_unit(text_list, header, first_line, key_index, current_title, current_section_name,
                         current_sentence):
    unit = "NA"
    table_title = "NA"
    sub_retry = 0
    og_candidate_len = len(text_list)
    text_list.insert(0, current_section_name + current_title)
    text_list.append(current_sentence)
    usage = 0
    # 将"text'里面的无效string去掉
    pure_text_list = []
    for each in text_list:
        if each != "" and each != " " and each != "\n" and each.isdigit() == False:
            pure_text_list.append(each)
    text_list = pure_text_list

    if len(text_list) > 1:
        while sub_retry < 10:
            try:
                first_usage, bot_answer = title_seek_bot(header, first_line, text_list)
                usage += first_usage
                table_title, unit = str(eval(bot_answer)["title"]), str(eval(bot_answer)["unit"])

                if table_title == "NONE":
                    if '单位' in current_sentence and '(涉及账户)' in table_title:
                        if "万" in current_sentence:
                            unit = "万元"
                        elif "千" in current_sentence:
                            unit = "千元"
                        else:
                            unit = "元"
                    else:
                        if current_title:
                            table_title = current_title + "表"
                        elif current_sentence:
                            table_title = current_sentence + "表"
                        else:
                            title_make_usage, table_title = title_makeup_bot(header, key_index)
                            usage += title_make_usage

                break
            except Exception as e:
                sub_retry += 1
                continue
    elif len(text_list) == 1:
        table_title = text_list[0]
    else:
        if current_title:
            table_title = current_title + "表"
        elif current_sentence:
            table_title = current_sentence + "表"
        else:
            title_make_usage, table_title = title_makeup_bot(header, key_index)
            usage += title_make_usage

    return table_title, unit, text_list, usage


def page_text_last_line(page_content):
    # 假设每页至少有一行文本
    if page_content and "lines" in page_content[-1] and len(page_content[-1]["lines"]) > 0:
        last_line = page_content[-1]["lines"][-1]["text"]
        back_count = -2
        while last_line == "":
            try:
                last_line = page_content[-1]["lines"][back_count]["text"]
                back_count -= 1
            except:
                return ""

        return last_line
    return ""


def remove_digits(text):
    # 使用正则表达式去除文本中的所有数字
    return re.sub(r'\d+', '', text)


def find_redundant_tail(list_of_pages):
    try:
        # 首先，提取每一页的最后一行文本并去除数字
        last_lines = [remove_digits(page_text_last_line(page)) for page in list_of_pages]

        # 如果没有页面或页面中没有文本，返回空列表
        if not last_lines or not all(last_lines):
            return []

        # 找出共同的尾部文本
        common_tail = last_lines[0]
        for line in last_lines[1:]:
            # 对比每一个尾部文本，逐步减少common_tail的长度，直至找到共有的部分
            for i in range(min(len(common_tail), len(line)), 0, -1):
                if common_tail[:i] == line[:i]:
                    common_tail = common_tail[:i]
                    break
            else:
                # 如果没有共有部分，返回空字符串
                return ""

        return common_tail
    except Exception as e:
        return ""


def is_text_match_redundant_tail(text, redundant_tail):
    # 首先，去除文本中的数字
    processed_text = remove_digits(text)
    # 然后，直接比较处理后的文本与redundant_tail
    return processed_text == redundant_tail


def find_first_number(text):
    match = re.search(r'\d+', text)
    if match:
        return match.group()
    else:
        return None


def get_full_page_content(page):
    new_page_text = ""
    for each_section in page:
        if each_section["type"] == "plain":
            texts = each_section["lines"]
            for text_item in texts:
                if 'text' in text_item:
                    new_page_text += text_item['text']
        elif each_section["type"] == "table_with_line":
            new_page_text += "这里有个表格（略过）"

        new_page_text += "\n"
    return new_page_text


def check_pattern(pattern, pre_page, current_page):
    total_usage = 0
    if not pattern:
        # 如果空pattern的情况下，如果出现purify后的纯数字组合，抽出倒数两行召唤出bot来确定，这段文字是不是在说页码的，加上前后两夜page的content辅助判定

        # 严格来说，是找到了出现页码了，但并不是出现了pattern
        original_current_last_line = page_text_last_line(current_page)
        if original_current_last_line:
            current_last_line = find_first_number(original_current_last_line)
            if current_last_line and current_last_line.isdigit():
                whole_page = get_full_page_content(current_page)
                usage, first_check = page_check_bot(whole_page, original_current_last_line)
                total_usage += usage
                if first_check:
                    return True

    else:
        original_pre_last_line = page_text_last_line(pre_page)
        original_current_last_line = page_text_last_line(current_page)

        if original_pre_last_line and original_current_last_line:

            pre_last_line = find_first_number(original_pre_last_line)
            current_last_line = find_first_number(original_current_last_line)
            # 情况1 如果上下页都是纯数字，并且顺序，就是在走pattern
            if original_pre_last_line.isdigit() and original_current_last_line.isdigit() and pre_last_line and current_last_line and pre_last_line.isdigit() and current_last_line.isdigit() and int(
                    pre_last_line) + 1 == int(current_last_line):
                return True
            # 情况2 如果前一页和后一页去除数值后，有共同的非空字符，说明他们在走pattern

            pre_last_line = remove_digits(original_pre_last_line)
            current_last_line = remove_digits(original_current_last_line)
            if pre_last_line and current_last_line and pre_last_line == current_last_line:
                return True

        if pattern:
            print("发生转换")
            print(f"前一页:{original_pre_last_line}，后一页:{original_current_last_line}")

            # if original_current_last_line == " " or original_current_last_line == "":

    return False


# def ESG_pdf_material_process(pure_ocr_result, report_name, merged=0):
#     road_map = {}
#     current_title = ""
#     current_content = []
#     current_sentence = ""
#     pre_pattern = False
#     new_pattern = False
#     redundant_tail = []
#     if len(pure_ocr_result) > 20:
#         check_start = int(len(pure_ocr_result) * 0.45)
#         check_end = int(len(pure_ocr_result) * 0.65)
#         redundant_head = find_redundant_pdf_head(pure_ocr_result[check_start:check_end])
#         if merged != 1:
#             redundant_tail = find_redundant_tail(pure_ocr_result[check_start:check_end])
#     else:
#         if len(pure_ocr_result) == 1:
#             redundant_head = []
#         else:
#             redundant_head = find_redundant_pdf_head(pure_ocr_result)
#         if merged != 1:
#             redundant_tail = find_redundant_tail(pure_ocr_result)
#
#     pre_table = []
#     current_section_name = ""
#     pre_table_section = ""
#     last_table_section = False
#     expand_table_check = False
#     total_pages = len(pure_ocr_result)
#     current_title_level = {}
#     current_section_level = 1
#     title_need_to_find = False
#     potential_title_stack = []
#     level_title_record = {}
#     repeat_name_record = {}
#     virtual_title_index = 0
#     page_num = 0
#     pre_section_page = 0
#     pre_page = None
#     total_usage = 0
#     legacy_title = None
#
#     # 中信建投定制
#     title_lock = False
#
#     for page_num, page in enumerate(pure_ocr_result):
#
#         print(f"\r{report_name} Progress: {page_num + 1}/{total_pages} pages processed.", end='')
#
#         if len(road_map) > 0 and len(page) > 0:
#             expand_table_check, pre_table = redundant_head_checker(redundant_head, page,
#                                                                    road_map[list(road_map.keys())[-1]])
#
#         if pre_page and merged == 1:
#             new_pattern = check_pattern(pre_pattern, pre_page, page)
#
#             if pre_pattern and not new_pattern:
#                 # 说明pattern消失了，怀疑是转报告了
#                 level_title_record = {}
#                 current_title_level = {}
#                 if current_title:
#                     legacy_title = current_title
#
#         for section_index, section in enumerate(page):
#             # if "type" not in section.keys():
#
#             if "type" not in section.keys() or section["type"] == "plain":
#                 ##文字段落
#                 for line_index, each_line in enumerate(section["lines"]):
#                     text = each_line["text"]
#
#                     if text in redundant_head:
#                         continue
#
#                     if new_pattern and line_index == len(section["lines"]) - 1 and section_index == len(page) - 1:
#                         # 如果是页码，直接跳过
#                         continue
#
#                     if merged != 1 and is_text_match_redundant_tail(text, redundant_tail):
#                         continue
#
#                     if len(potential_title_stack) > 5:
#                         potential_title_stack.pop(0)
#                     potential_title_stack.append(text)
#
#                     if "。" in text:
#                         while "。" in text:
#                             index = text.index("。")
#                             if index < len(text):
#                                 current_sentence += text[:index + 1]
#                             else:
#                                 current_sentence += text
#
#                             current_content.append(current_sentence)
#                             current_sentence = ""
#                             if index < len(text) - 2:
#                                 text = text[index + 1:]
#                             else:
#                                 text = ""
#                         if text != "":
#                             current_sentence += text
#                     else:
#
#                         current_sentence += text
#
#
#             else:
#                 ##表格段落
#                 ##先处理table
#                 current_table_list = create_tables_from_data(section["table_cells"])
#
#                 if len(current_table_list) == 0 or title_lock:
#                     long_text = ""
#                     for each_cell in section["table_cells"]:
#                         text = each_cell["text"]
#                         if len(long_text) == 0:
#                             long_text += text
#                         else:
#                             long_text += "，" + text
#                     current_content.append(long_text)
#                     continue
#
#                 ##找到title/单位 ##bot启用
#                 if len(current_table_list) > 0:
#                     header = current_table_list[0]["header"]
#
#                     if len(current_table_list[0]["values"]) > 0:
#                         first_line = current_table_list[0]["values"][0]
#                     else:
#                         first_line = ""
#
#                     if len(current_table_list[0]["key_index"]) > 0:
#                         key_index = current_table_list[0]["key_index"]
#                     else:
#                         key_index = ""
#                 else:
#                     continue
#
#                 og_header = []
#                 new_header = []
#                 consistent_title = ""
#                 consistent_unit = ""
#                 consistent_potential_table_titles = ""
#
#                 if len(potential_title_stack) > 0 or current_section_name or current_title or current_sentence:
#                     table_title, unit, potential_table_titles, find_title_usage = table_title_and_unit(
#                         potential_title_stack.copy(), header, key_index, first_line, current_title,
#                         current_section_name, current_sentence)
#                     total_usage += find_title_usage
#
#                 else:
#                     table_title, unit, potential_table_titles = "NA", "NA", "NA"
#
#                 for each_table in current_table_list:
#                     if og_header and each_table["header"] == og_header:
#                         each_table["header"] = new_header
#                         table_title = consistent_title
#                         unit = consistent_unit
#                         potential_table_titles = consistent_potential_table_titles
#                     if each_table["title"] is None:
#                         each_table["title"] = table_title
#                     elif table_title != "NA":
#                         each_table["title"] = table_title + ":" + each_table["title"]
#                     each_table["unit"] = unit
#                     each_table["potential_table_titles"] = potential_table_titles
#
#                     if expand_table_check and same_column_num_check(pre_table, each_table):
#                         if not pre_table:
#                             pre_table = each_table
#                         else:
#
#                             current_table = each_table
#                             og_header = current_table["header"]
#                             pre_table, legacy_table, combine_table_usage, attention_case = break_table_combination(
#                                 pre_table, current_table, table_title)
#
#                             if attention_case:
#                                 table_title += "<跨页合并次级置信度>"
#
#                             if not legacy_table:
#                                 new_header = pre_table["header"]
#                                 consistent_title = pre_table["title"]
#                                 consistent_unit = pre_table["unit"]
#                                 consistent_potential_table_titles = pre_table["potential_table_titles"]
#                                 pre_key = list(road_map.keys())[-1]
#                                 road_map[pre_key]["table"][-1] = pre_table
#                                 expand_table_check = False
#                             else:
#                                 pre_key = list(road_map.keys())[-1]
#                                 road_map[pre_key]["table"].append(pre_table)
#                                 expand_table_check = False
#
#                     else:
#                         expand_table_check = False
#
#                         current_title_purify = table_title
#                         if current_sentence != "":
#                             current_content.append(current_sentence)
#                         if current_title_purify in road_map.keys():
#                             if current_title_purify not in repeat_name_record.keys():
#                                 repeat_name_record[current_title_purify] = 1
#                                 current_title_purify += f"(1)"
#                             else:
#                                 repeat_name_record[current_title_purify] += 1
#                                 current_title_purify += f"({repeat_name_record[current_title_purify]})"
#                         road_map[current_title_purify] = {}
#                         road_map[current_title_purify]["table"] = [each_table]
#                         road_map[current_title_purify]["content"] = split_long_strings(current_content)
#                         road_map[current_title_purify]["level"] = current_section_level
#                         road_map[current_title_purify]["page"] = page_num
#
#                         pre_table_section = current_title_purify
#                     last_table_section = True
#
#                     current_content = []
#                     potential_title_stack = []
#                     current_sentence = ""
#
#         if current_sentence.strip() != "" or current_content:
#
#             key_2_store = str(virtual_title_index)
#             virtual_title_index += 1
#             if current_sentence != "":
#                 current_content.append(current_sentence)
#             if key_2_store not in road_map.keys():
#                 road_map[key_2_store] = {}
#                 road_map[key_2_store]["content"] = split_long_strings(current_content)
#                 road_map[key_2_store]["table"] = []
#                 road_map[key_2_store]["level"] = current_section_level
#                 road_map[key_2_store]["page"] = page_num
#             else:
#                 # 检查是否真的是上一个
#
#                 if current_title not in level_title_record.values():
#                     # 代表只是重复名，不是同一个，要检查是不是母公司的东西
#                     actual_mother_company, penetration_usage = check_if_mother_company(
#                         level_title_record, key_2_store,
#                         current_section_level)
#                     total_usage += penetration_usage
#                     if actual_mother_company:
#                         key_2_store += "(母公司内容)"
#                         road_map[key_2_store] = {}
#                         road_map[key_2_store]["content"] = split_long_strings(current_content)
#                         road_map[key_2_store]["table"] = []
#                         road_map[key_2_store]["level"] = current_section_level
#                         road_map[key_2_store]["page"] = page_num
#                     else:
#                         # 留在原位不要乱跑
#                         if key_2_store not in repeat_name_record.keys():
#                             repeat_name_record[key_2_store] = 1
#                             key_2_store += f"(1)"
#                         else:
#
#                             repeat_name_record[key_2_store] += 1
#                             key_2_store += f"({repeat_name_record[key_2_store]})"
#                         road_map[key_2_store] = {}
#                         road_map[key_2_store]["content"] = split_long_strings(current_content)
#                         road_map[key_2_store]["table"] = []
#                         road_map[key_2_store]["level"] = current_section_level
#                         road_map[key_2_store]["page"] = page_num
#
#                 else:
#
#                     road_map[key_2_store]["content"] += split_long_strings(current_content)
#
#             current_content = []
#             current_sentence = ""
#         pre_pattern = new_pattern
#         pre_page = page
#
#     if current_section_name and (not last_table_section or title_lock):
#         if current_section_name:
#             if not current_title:
#                 current_title = current_sentence
#                 current_sentence = ""
#             if current_sentence != "":
#                 current_content.append(current_sentence)
#
#             key_2_store = current_section_name + " " + str(current_title)
#             if key_2_store not in road_map.keys():
#                 road_map[key_2_store] = {}
#                 road_map[key_2_store]["content"] = split_long_strings(current_content)
#                 road_map[key_2_store]["table"] = []
#                 road_map[key_2_store]["level"] = current_section_level
#                 road_map[key_2_store]["page"] = page_num + 1
#             else:
#                 # 检查是否真的是上一个
#                 if key_2_store not in level_title_record.values():
#                     # 代表只是重复名，不是同一个，要检查是不是母公司的东西
#                     actual_mother_company, penetration_usage = check_if_mother_company(level_title_record, key_2_store,
#                                                                                        current_section_level)
#                     total_usage += penetration_usage
#                     if actual_mother_company:
#                         key_2_store += "(母公司内容)"
#                         road_map[key_2_store] = {}
#                         road_map[key_2_store]["content"] = split_long_strings(current_content)
#                         road_map[key_2_store]["table"] = []
#                         road_map[key_2_store]["level"] = current_section_level
#                         road_map[key_2_store]["page"] = pre_section_page
#                 else:
#                     road_map[key_2_store]["content"] += split_long_strings(current_content)
#     elif last_table_section and pre_table_section:
#         if current_sentence != "":
#             current_content.append(current_sentence)
#         road_map[pre_table_section]["content"] += split_long_strings(current_content)
#     elif current_content:
#         key_2_store = '全部内容'
#         road_map[key_2_store] = {}
#         road_map[key_2_store]["content"] = split_long_strings(current_content)
#         road_map[key_2_store]["table"] = []
#         road_map[key_2_store]["level"] = current_section_level
#         road_map[key_2_store]["page"] = page_num + 1
#
#     return road_map, total_usage


def pdf_material_process(pure_ocr_result, report_name, merged=0):
    try:
        road_map = {}
        current_title = ""
        current_content = []
        current_sentence = ""
        pre_pattern = False
        new_pattern = False
        redundant_tail = []
        if len(pure_ocr_result) > 20:
            check_start = int(len(pure_ocr_result) * 0.45)
            check_end = int(len(pure_ocr_result) * 0.65)
            redundant_head = find_redundant_pdf_head(pure_ocr_result[check_start:check_end])
            if merged != 1:
                redundant_tail = find_redundant_tail(pure_ocr_result[check_start:check_end])
        else:
            if len(pure_ocr_result) == 1:
                redundant_head = []
            else:
                redundant_head = find_redundant_pdf_head(pure_ocr_result)
            if merged != 1:
                redundant_tail = find_redundant_tail(pure_ocr_result)

        pre_table = []
        current_section_name = ""
        pre_table_section = ""
        last_table_section = False
        expand_table_check = False
        total_pages = len(pure_ocr_result)
        current_title_level = {}
        current_section_level = -1
        title_need_to_find = False
        potential_title_stack = []
        level_title_record = {}

        repeat_name_record = {}

        page_num = 0
        pre_section_page = 0
        pre_page = None
        total_usage = 0
        legacy_title = None

        # 中信建投定制
        title_lock = False

        for page_num, page in enumerate(pure_ocr_result):

            # Extract all text from the current page
            # More efficient text concatenation using list join
            page_text_parts = []
            for section in page:
                if "type" not in section.keys() or section["type"] == "plain":
                    page_text_parts.extend(each_line["text"] for each_line in section["lines"])
                elif section["type"] == "table_with_line":
                    page_text_parts.extend(cell["text"] for cell in section["table_cells"])
            page_full_text = "".join(page_text_parts)

            # If the concatenated text does not contain "独立董事", skip this page

            #目前跑独董报告，所以不需要这个
            if not any(term in page_full_text for term in ["独立董事"]):
                continue

            print(f"\r{report_name} Progress: {page_num + 1}/{total_pages} pages processed.", end='')

            if len(road_map) > 0 and len(page) > 0:
                expand_table_check, pre_table = redundant_head_checker(redundant_head, page,
                                                                       road_map[list(road_map.keys())[-1]])

            if pre_page and merged == 1:
                new_pattern = check_pattern(pre_pattern, pre_page, page)

                if pre_pattern and not new_pattern:
                    # 说明pattern消失了，怀疑是转报告了
                    level_title_record = {}
                    current_title_level = {}
                    if current_title:
                        legacy_title = current_title

            for section_index, section in enumerate(page):
                # if "type" not in section.keys():

                if "type" not in section.keys() or section["type"] == "plain":
                    ##文字段落
                    for line_index, each_line in enumerate(section["lines"]):
                        text = each_line["text"]

                        if '募集资金使用情况对照表' in text and line_index <= 5:
                            to_check_text = text
                            if current_sentence:
                                to_check_text = current_sentence + text
                            usage, check_special_title_result = special_title_check_bot(to_check_text)
                            if check_special_title_result:
                                title_lock = True
                                section_page = page_num
                                if current_section_name:
                                    if not current_title:
                                        current_title = current_sentence
                                        current_sentence = ""
                                    if current_sentence != "":
                                        current_content.append(current_sentence)

                                    if current_section_name not in str(current_title):
                                        key_2_store = current_section_name + " " + str(current_title)
                                    else:
                                        key_2_store = str(current_title)
                                    if key_2_store not in road_map.keys():
                                        road_map[key_2_store] = {}
                                        road_map[key_2_store]["content"] = split_long_strings(current_content)
                                        road_map[key_2_store]["table"] = []
                                        road_map[key_2_store]["level"] = current_section_level
                                        road_map[key_2_store]["page"] = pre_section_page
                                    else:
                                        # 检查是否真的是上一个
                                        if current_title not in level_title_record.values():
                                            # 代表只是重复名，不是同一个，要检查是不是母公司的东西
                                            actual_mother_company, penetration_usage = check_if_mother_company(
                                                level_title_record, key_2_store,
                                                current_section_level)
                                            total_usage += penetration_usage
                                            if actual_mother_company:
                                                key_2_store += "(母公司内容)"
                                                road_map[key_2_store] = {}
                                                road_map[key_2_store]["content"] = split_long_strings(current_content)
                                                road_map[key_2_store]["table"] = []
                                                road_map[key_2_store]["level"] = current_section_level
                                                road_map[key_2_store]["page"] = pre_section_page
                                            else:
                                                # 留在原位不要乱跑
                                                if key_2_store not in repeat_name_record.keys():
                                                    repeat_name_record[key_2_store] = 1
                                                    key_2_store += f"(1)"
                                                else:

                                                    repeat_name_record[key_2_store] += 1
                                                    key_2_store += f"({repeat_name_record[key_2_store]})"
                                                road_map[key_2_store] = {}
                                                road_map[key_2_store]["content"] = split_long_strings(current_content)
                                                road_map[key_2_store]["table"] = []
                                                road_map[key_2_store]["level"] = current_section_level
                                                road_map[key_2_store]["page"] = pre_section_page
                                        else:
                                            road_map[key_2_store]["content"] += split_long_strings(current_content)
                                    # 维护层级记录
                                    if legacy_title == current_title:
                                        legacy_title = None
                                    else:
                                        level_title_record[current_section_level] = current_title
                                        # 还要清空新层级后面的记录
                                        for key in list(level_title_record.keys()):
                                            if int(key) > current_section_level:
                                                del level_title_record[key]
                                elif not current_section_name and current_sentence != "" and len(road_map) == 0:
                                    current_title = "前序"
                                    if str(current_title) not in road_map.keys():
                                        road_map[str(current_title)] = {}
                                    current_content.append(current_sentence)
                                    road_map[str(current_title)]["content"] = split_long_strings(current_content)
                                    road_map[str(current_title)]["level"] = 1
                                    road_map[str(current_title)]["page"] = pre_section_page
                                    # 维护层级记录
                                    level_title_record[current_section_level] = current_title

                                    for key in list(level_title_record.keys()):
                                        if int(key) > current_section_level:
                                            del level_title_record[key]
                                current_content = []
                                pre_section_page = section_page
                                current_section_name = "募集资金使用情况对照表"
                                current_title = "募集资金使用情况对照表"
                                title_need_to_find = False
                                current_sentence = ""
                                potential_title_stack = []
                                continue

                        if "isOnRightEdge" in each_line.keys():
                            if_on_edge = each_line["isOnRightEdge"]
                        else:
                            if_on_edge = "0"

                        if text in redundant_head:
                            continue

                        if new_pattern and line_index == len(section["lines"]) - 1 and section_index == len(page) - 1:
                            # 如果是页码，直接跳过
                            continue

                        if merged != 1 and is_text_match_redundant_tail(text, redundant_tail):
                            continue

                        if title_lock:
                            if len(text) > 0:
                                if "。" in text:
                                    while "。" in text:
                                        index = text.index("。")
                                        if index < len(text):
                                            current_sentence += text[:index + 1]
                                        else:
                                            current_sentence += text

                                        current_content.append(current_sentence)
                                        current_sentence = ""
                                        if index < len(text) - 2:
                                            text = text[index + 1:]
                                        else:
                                            text = ""
                                    if text != "":
                                        current_sentence += text
                                elif text[-1] == ":":
                                    while ":" in text:
                                        index = text.index(":")

                                        if index < len(text):
                                            current_sentence += text[:index + 1]
                                        else:
                                            current_sentence += text
                                        current_content.append(current_sentence)
                                        current_sentence = ""
                                        if index < len(text) - 2:
                                            text = text[index + 1:]
                                        else:
                                            text = ""
                                    if text != "":
                                        current_sentence += text
                                elif text[-1] == "：":
                                    while "：" in text:
                                        index = text.index("：")
                                        if index < len(text):
                                            current_sentence += text[:index + 1]
                                        else:
                                            current_sentence += text

                                        current_content.append(current_sentence)
                                        current_sentence = ""
                                        if index < len(text) - 2:
                                            text = text[index + 1:]
                                        else:
                                            text = ""
                                    if text != "":
                                        current_sentence += text
                                else:
                                    current_sentence += text
                                continue

                        level, section_name, title, temp_current_title_level, temp_level_title_record, sentence_remain = formula_title_check(
                            text,
                            current_title_level, level_title_record,
                            if_on_edge, current_content, current_sentence)
                        level_title_record = temp_level_title_record
                        ##这里判断大转折
                        if level > -1:

                            section_page = page_num
                            if current_section_level == -1:
                                current_section_level = 1
                            if current_section_name:
                                if not current_title:
                                    current_title = current_sentence
                                    current_sentence = ""
                                if current_sentence != "":
                                    current_content.append(current_sentence)

                                if current_section_name not in str(current_title):
                                    key_2_store = current_section_name + " " + str(current_title)
                                else:
                                    key_2_store = str(current_title)
                                if key_2_store not in road_map.keys():
                                    road_map[key_2_store] = {}
                                    road_map[key_2_store]["content"] = split_long_strings(current_content)
                                    road_map[key_2_store]["table"] = []
                                    road_map[key_2_store]["level"] = current_section_level
                                    road_map[key_2_store]["page"] = pre_section_page
                                else:
                                    # 检查是否真的是上一个

                                    if current_title not in level_title_record.values():
                                        # 代表只是重复名，不是同一个，要检查是不是母公司的东西
                                        actual_mother_company, penetration_usage = check_if_mother_company(
                                            level_title_record, key_2_store,
                                            current_section_level)
                                        total_usage += penetration_usage
                                        if actual_mother_company:
                                            key_2_store += "(母公司内容)"
                                            road_map[key_2_store] = {}
                                            road_map[key_2_store]["content"] = split_long_strings(current_content)
                                            road_map[key_2_store]["table"] = []
                                            road_map[key_2_store]["level"] = current_section_level
                                            road_map[key_2_store]["page"] = pre_section_page
                                        else:
                                            # 即使不是，也不要让他位置改变
                                            if key_2_store not in repeat_name_record.keys():
                                                repeat_name_record[key_2_store] = 1
                                                key_2_store += f"(1)"
                                            else:
                                                repeat_name_record[key_2_store] += 1
                                                key_2_store += f"({repeat_name_record[key_2_store]})"
                                            road_map[key_2_store] = {}
                                            road_map[key_2_store]["content"] = split_long_strings(current_content)
                                            road_map[key_2_store]["table"] = []
                                            road_map[key_2_store]["level"] = current_section_level
                                            road_map[key_2_store]["page"] = pre_section_page
                                    else:
                                        road_map[key_2_store]["content"] += split_long_strings(current_content)

                                # 维护层级记录
                                if legacy_title == current_title:
                                    legacy_title = None
                                else:
                                    level_title_record[current_section_level] = current_title
                                    # 还要清空新层级后面的记录
                                    for key in list(level_title_record.keys()):
                                        if int(key) > current_section_level:
                                            del level_title_record[key]



                            elif not current_section_name and current_sentence != "" and len(road_map) == 0:
                                current_title = "前序"

                                if str(current_title) not in road_map.keys():
                                    road_map[str(current_title)] = {}

                                current_content.append(current_sentence)
                                road_map[str(current_title)]["content"] = split_long_strings(current_content)
                                road_map[str(current_title)]["level"] = 1
                                road_map[str(current_title)]["page"] = pre_section_page
                                # 维护层级记录
                                level_title_record[current_section_level] = current_title

                                for key in list(level_title_record.keys()):
                                    if int(key) > current_section_level:
                                        del level_title_record[key]

                            current_content = []
                            pre_section_page = section_page
                            current_section_name = section_name
                            if len(title) > 0:
                                current_title = title
                                title_need_to_find = False
                            else:
                                current_title = ""
                                title_need_to_find = True

                            if len(sentence_remain) > 0:
                                current_sentence = sentence_remain
                            else:
                                current_sentence = ""
                            potential_title_stack = []
                            current_title_level = temp_current_title_level
                            current_section_level = level

                        else:
                            ##检查句子是否完整，找"。"
                            if title_need_to_find and len(text) > 0:
                                first = True
                                if "。" in text:
                                    while "。" in text:
                                        index = text.index("。")
                                        if index < len(text):
                                            current_sentence += text[:index + 1]
                                        else:
                                            current_sentence += text
                                        if first:
                                            current_title = current_sentence
                                            first = False
                                            title_need_to_find = False
                                        else:
                                            current_content.append(current_sentence)
                                        current_sentence = ""
                                        if index < len(text) - 2:
                                            text = text[index + 1:]
                                        else:
                                            text = ""
                                    if text != "":
                                        current_sentence += text
                                elif text[-1] == ":":
                                    while ":" in text:
                                        index = text.index(":")

                                        if index < len(text):
                                            current_sentence += text[:index + 1]
                                        else:
                                            current_sentence += text
                                        if first:
                                            current_title = current_sentence
                                            first = False
                                            title_need_to_find = False
                                        else:
                                            current_content.append(current_sentence)
                                        current_sentence = ""
                                        if index < len(text) - 2:
                                            text = text[index + 1:]
                                        else:
                                            text = ""
                                    if text != "":
                                        current_sentence += text
                                elif text[-1] == "：":
                                    while "：" in text:
                                        index = text.index("：")
                                        if index < len(text):
                                            current_sentence += text[:index + 1]
                                        else:
                                            current_sentence += text
                                        if first:
                                            current_title = current_sentence
                                            first = False
                                            title_need_to_find = False
                                        else:
                                            current_content.append(current_sentence)
                                        current_sentence = ""
                                        if index < len(text) - 2:
                                            text = text[index + 1:]
                                        else:
                                            text = ""
                                    if text != "":
                                        current_sentence += text
                                else:
                                    current_sentence += text
                                continue

                            def texts_are_identical(text1, text2):
                                cleaned_text1 = ''.join(
                                    char for char in text1 if not char.isdigit() and not char.isspace())
                                cleaned_text2 = ''.join(
                                    char for char in text2 if not char.isdigit() and not char.isspace())

                                return cleaned_text1 == cleaned_text2

                            last_text = potential_title_stack[-1] if len(potential_title_stack) > 0 else ""
                            if not texts_are_identical(text, last_text):
                                if len(potential_title_stack) > 5:
                                    potential_title_stack.pop(0)
                                potential_title_stack.append(text)
                            if "。" in text:
                                while "。" in text:
                                    index = text.index("。")
                                    if index < len(text):
                                        current_sentence += text[:index + 1]
                                    else:
                                        current_sentence += text

                                    current_content.append(current_sentence)
                                    current_sentence = ""
                                    if index < len(text) - 2:
                                        text = text[index + 1:]
                                    else:
                                        text = ""
                                if text != "":
                                    current_sentence += text
                            else:

                                current_sentence += text


                else:
                    ##表格段落
                    ##先处理table
                    title_need_to_find = False
                    current_table_list = create_tables_from_data(section["table_cells"])

                    if len(current_table_list) == 0 or title_lock:
                        long_text = ""
                        for each_cell in section["table_cells"]:
                            text = each_cell["text"]
                            if len(long_text) == 0:
                                long_text += text
                            else:
                                long_text += "，" + text
                        current_content.append(long_text)
                        continue

                    ##找到title/单位 ##bot启用
                    if len(current_table_list) > 0:
                        header = current_table_list[0]["header"]

                        if len(current_table_list[0]["values"]) > 0:
                            first_line = current_table_list[0]["values"][0]
                        else:
                            first_line = ""

                        if len(current_table_list[0]["key_index"]) > 0:
                            key_index = current_table_list[0]["key_index"]
                        else:
                            key_index = ""
                    else:
                        continue

                    og_header = []
                    new_header = []
                    consistent_title = ""
                    consistent_unit = ""
                    consistent_potential_table_titles = ""

                    if len(potential_title_stack) > 0 or current_section_name or current_title or current_sentence:
                        table_title, unit, potential_table_titles, find_title_usage = table_title_and_unit(
                            potential_title_stack.copy(), header,
                            key_index,
                            first_line,
                            current_title,
                            current_section_name, current_sentence)
                        total_usage += find_title_usage
                        table_title = str(table_title)
                        unit = str(unit)
                    else:
                        table_title, unit, potential_table_titles = "NA", "NA", "NA"

                    # 这里要加一个回溯检查是否母公司表格的method
                    actual_mother_company, penetration_usage = check_if_mother_company(level_title_record, table_title,
                                                                                       current_section_level,
                                                                                       current_title)
                    total_usage += penetration_usage
                    if actual_mother_company:
                        table_title += "(母公司表格)"

                    for each_table in current_table_list:
                        if og_header and each_table["header"] == og_header:
                            each_table["header"] = new_header
                            table_title = str(consistent_title)
                            unit = consistent_unit
                            potential_table_titles = consistent_potential_table_titles
                        if each_table["title"] is None:
                            each_table["title"] = table_title
                        elif table_title != "NA":
                            each_table["title"] = str(table_title) + ":" + each_table["title"]
                        each_table["unit"] = unit
                        each_table["potential_table_titles"] = potential_table_titles

                        if expand_table_check and same_column_num_check(pre_table, each_table):
                            if not pre_table:
                                pre_table = each_table
                            else:

                                current_table = each_table
                                og_header = current_table["header"]
                                pre_table, legacy_table, combine_table_usage, attention_case = break_table_combination(
                                    pre_table, current_table, table_title)

                                if attention_case:
                                    table_title += "<跨页合并次级置信度>"

                                if not legacy_table:
                                    new_header = pre_table["header"]
                                    consistent_title = pre_table["title"]
                                    consistent_unit = pre_table["unit"]
                                    consistent_potential_table_titles = pre_table["potential_table_titles"]
                                    pre_key = list(road_map.keys())[-1]
                                    road_map[pre_key]["table"][-1] = pre_table
                                    expand_table_check = False
                                else:
                                    pre_key = list(road_map.keys())[-1]
                                    road_map[pre_key]["table"].append(pre_table)
                                    expand_table_check = False

                        else:
                            expand_table_check = False

                            if current_section_level == -1:
                                current_section_level = 1

                            if current_section_name and current_title:

                                if current_section_name not in str(current_title):
                                    key_2_store = current_section_name + " " + str(current_title)
                                else:
                                    key_2_store = str(current_title)

                                if current_sentence != "":
                                    current_content.append(current_sentence)
                                if key_2_store not in road_map.keys():
                                    road_map[key_2_store] = {}
                                    road_map[key_2_store]["table"] = [each_table]
                                    road_map[key_2_store]["content"] = split_long_strings(current_content)
                                    road_map[key_2_store]["level"] = current_section_level
                                    road_map[key_2_store]["page"] = pre_section_page
                                else:
                                    ###这里要append
                                    # 检查是否真的是上一个

                                    if current_title not in level_title_record.values():
                                        # 代表只是重复名，不是同一个，要检查是不是母公司的东西
                                        actual_mother_company, penetration_usage = check_if_mother_company(
                                            level_title_record, key_2_store,
                                            current_section_level,
                                            current_title)
                                        total_usage += penetration_usage
                                        if actual_mother_company:
                                            key_2_store += "(母公司内容)"
                                            road_map[key_2_store] = {}
                                            road_map[key_2_store]["content"] = split_long_strings(current_content)
                                            road_map[key_2_store]["table"] = [each_table]
                                            road_map[key_2_store]["level"] = current_section_level
                                            road_map[key_2_store]["page"] = pre_section_page
                                        else:
                                            # 即使不是，也不要让他位置改变
                                            if key_2_store not in repeat_name_record.keys():
                                                repeat_name_record[key_2_store] = 1
                                                key_2_store += f"(1)"
                                            else:
                                                repeat_name_record[key_2_store] += 1
                                                key_2_store += f"({repeat_name_record[key_2_store]})"
                                            road_map[key_2_store] = {}
                                            road_map[key_2_store]["content"] = split_long_strings(current_content)
                                            road_map[key_2_store]["table"] = [each_table]
                                            road_map[key_2_store]["level"] = current_section_level
                                            road_map[key_2_store]["page"] = pre_section_page
                                    else:
                                        road_map[key_2_store]["table"].append(each_table)

                                if legacy_title == current_title:
                                    legacy_title = None
                                else:
                                    level_title_record[current_section_level] = current_title
                                    # 还要清空新层级后面的记录
                                    for key in list(level_title_record.keys()):
                                        if int(key) > current_section_level:
                                            del level_title_record[key]

                                pre_table_section = key_2_store
                            else:

                                current_title_purify = table_title
                                if current_sentence != "":
                                    current_content.append(current_sentence)
                                if current_title_purify in road_map.keys():
                                    if current_title_purify not in repeat_name_record.keys():
                                        repeat_name_record[current_title_purify] = 1
                                        current_title_purify += f"(1)"
                                    else:
                                        repeat_name_record[current_title_purify] += 1
                                        current_title_purify += f"({repeat_name_record[current_title_purify]})"
                                road_map[current_title_purify] = {}
                                road_map[current_title_purify]["table"] = [each_table]
                                road_map[current_title_purify]["content"] = split_long_strings(current_content)
                                road_map[current_title_purify]["level"] = current_section_level
                                road_map[current_title_purify]["page"] = page_num

                                pre_table_section = current_title_purify
                            last_table_section = True

                            current_content = []
                            potential_title_stack = []
                            current_sentence = ""
            pre_pattern = new_pattern
            pre_page = page
        if current_section_name and (not last_table_section or title_lock):
            if current_section_name:
                if not current_title:
                    current_title = current_sentence
                    current_sentence = ""
                if current_sentence != "":
                    current_content.append(current_sentence)

                key_2_store = current_section_name + " " + str(current_title)
                if key_2_store not in road_map.keys():
                    road_map[key_2_store] = {}
                    road_map[key_2_store]["content"] = split_long_strings(current_content)
                    road_map[key_2_store]["table"] = []
                    road_map[key_2_store]["level"] = current_section_level
                    road_map[key_2_store]["page"] = page_num + 1
                else:
                    # 检查是否真的是上一个
                    if key_2_store not in level_title_record.values():
                        # 代表只是重复名，不是同一个，要检查是不是母公司的东西
                        actual_mother_company, penetration_usage = check_if_mother_company(level_title_record, key_2_store,
                                                                                           current_section_level,
                                                                                           current_title)
                        total_usage += penetration_usage
                        if actual_mother_company:
                            key_2_store += "(母公司内容)"
                            road_map[key_2_store] = {}
                            road_map[key_2_store]["content"] = split_long_strings(current_content)
                            road_map[key_2_store]["table"] = []
                            road_map[key_2_store]["level"] = current_section_level
                            road_map[key_2_store]["page"] = page_num
                        else:
                            road_map[key_2_store]["content"] += split_long_strings(current_content)
        elif last_table_section and pre_table_section:
            if current_sentence != "":
                current_content.append(current_sentence)
            road_map[pre_table_section]["content"] += split_long_strings(current_content)
        elif current_content:
            key_2_store = '全部内容'
            road_map[key_2_store] = {}
            road_map[key_2_store]["content"] = split_long_strings(current_content)
            road_map[key_2_store]["table"] = []
            road_map[key_2_store]["level"] = current_section_level
            road_map[key_2_store]["page"] = page_num + 1

        return road_map, total_usage
    except Exception as e:
        print(e)
        print('debug')
        raise e


def check_if_mother_company(level_title_record, title, current_section_level, current_title=None):
    parent_info = ""

    for index, key in enumerate(sorted(level_title_record.keys())):
        if int(key) >= int(current_section_level):
            if current_title and current_title != title:
                parent_info += current_title + "\n" + "\t" * (index - 1)
            break
        parent_info += level_title_record[key] + "\n" + "\t" * (index - 1)
    if "母" not in parent_info:
        return False, 0

    usage, check_result = mother_company_check_bot(parent_info, title)

    if check_result == "N":
        return False, usage
    else:
        return True, usage


def formula_title_check(text, current_title_level, level_title_record, if_on_edge, current_content, current_sentence):
    starter_list = [
        '问题 1.',
        '问题 1',
        '问题1',
        '问题1.',
        '问题一',
        '问题一.',
        '第一章',
        '第一节',
        '第一条',
        '(一)',
        '1.',
        '(1)',
        '1、',
        '1点',
        '第1章',
        '第1节',
        '一、',
        '一．',
        '（一）',
        '（１）',
        '1．',
        '1）',
        '1)'
    ]
    title_re_dict = {
        '问题 1.': '^问题\\s\\d+\\..*',
        '问题 1': '^问题\\s\\d+.*',
        '问题1': '^问题\\d+.*',
        '问题1.': '^问题\\d+\\..*',
        '问题一': '^问题[一二三四五六七八九十]+.*',
        '问题一.': '^问题[一二三四五六七八九十]+\\..*',
        '第几章': '^第[一二三四五六七八九十]+章.*',
        '第几节': '^第[一二三四五六七八九十]+节.*',
        '第几条': '^第[一二三四五六七八九十]+条.*',
        '(一)': '^\\([一二三四五六七八九十]+\\).*',
        '1.': '^\\d+\\..*',
        '(1)': '^\\(\\d+\\).*',
        '1、': '^\\d+、.*',  # 阿拉伯数字加顿号格式
        '第1点': '^第\\d+点.*',  # 阿拉伯数字加汉字"点"
        '第1章': '^第\\d+章.*',  # 英文序数词的章
        '第1节': '^第\\d+节.*',  # 英文序数词的节
        '一、': '^[一二三四五六七八九十]+、.*',  # 汉字开头加顿号的条目
        '一．': '^[一二三四五六七八九十]+．.*',  # 汉字加全角点号
        '（一）': '^\\（[一二三四五六七八九十]+\\）.*',  # 全角括号包裹汉字
        '（１）': '^\\（[０-９]+\\）.*',  # 全角括号包裹全角数字
        '1．': '^\\d+．.*',  # 阿拉伯数字加全角点号
        '1）': '^\\d+）.*',  # 阿拉伯数字加半角右括号
        '1)': '^\\d+\\).*',  # 阿拉伯数字加半角右括号
    }
    split_dict = {
        '问题 1.': '问题\\s\\d+\\.',
        '问题 1': '问题\\s\\d+',
        '问题1': '问题\\d+',
        '问题1.': '问题\\d+\\.',
        '问题一': '问题[一二三四五六七八九十]+',
        '问题一.': '问题[一二三四五六七八九十]+\\.',
        '第几章': '第[一二三四五六七八九十]+章',
        '第几节': '第[一二三四五六七八九十]+节',
        '第几条': '第[一二三四五六七八九十]+条',
        '(一)': '\\([一二三四五六七八九十]+\\)',
        '1.': '\\d+\\.',
        '(1)': '\\(\\d+\\)',
        '1、': '\\d+、',
        '一、': '[一二三四五六七八九十]+、',
        '第1点': '第\\d+点',
        '第1章': '第\\d+章',
        '第1节': '第\\d+节',
        '一．': '[一二三四五六七八九十]+．',
        '（一）': '\\（[一二三四五六七八九十]+\\）',
        '（１）': '\\（[０-９]+\\）',
        '1．': '\\d+．',
        '1）': '\\d+）',
        '1)': '\\d+\\)'}

    forbidden_list = ['1．', '1.']
    title = ""
    section_name = ""
    sentence_remain = text
    level = -1
    # iterate through the dict
    for key, value in title_re_dict.items():
        # 如果找到匹配的表达式
        if re.search(value, text):
            # 使用正则表达式分割文本
            rule = split_dict[key]
            parts = re.split(rule, text, maxsplit=1)
            if len(parts) > 1:
                if len(parts[1].strip(" ")) > 1:
                    title = parts[1].strip(" ")
            section_name = re.search(rule, text).group()

            if current_sentence:
                last_sentence = current_sentence
            elif len(current_content) > 0:
                last_sentence = current_content[-1]
            else:
                last_sentence = ''

            if is_numeric_or_dot(section_name):
                sentence_test = strange_title_check_bot(last_sentence, text)
                if not sentence_test:
                    return -1, "", "", current_title_level, level_title_record, ""

            title_rep = key

            if title_rep:
                if title_rep in current_title_level.keys():
                    if section_name in starter_list and current_title_level[title_rep] == 1:

                        # 检查是不是遇到拼接型文档了，重塑title_level的储存
                        reset_level = title_level_reset_bot(text)
                        if reset_level:
                            current_title_level = {}
                            level_title_record = {}
                            level = len(current_title_level.keys()) + 1
                            current_title_level[title_rep] = level
                        else:
                            return -1, "", "", current_title_level, level_title_record, ""
                    else:
                        level = current_title_level[title_rep]
                elif title_rep in forbidden_list:
                    if section_name in starter_list:
                        level = len(current_title_level.keys()) + 1
                        current_title_level[title_rep] = level
                    else:
                        return -1, "", "", current_title_level, level_title_record, ""
                else:
                    level = len(current_title_level.keys()) + 1
                    current_title_level[title_rep] = level

            if title:
                if title == '':
                    sentence_remain = title
                    title = ""
                else:
                    original_title = title
                    if "。" in title:
                        title = title.split("。")[0] + "。"
                        sentence_remain = '。'.join(original_title.split("。")[1:])
                    elif title[-1] == ":":
                        title = title.split(":")[0] + ":"
                        sentence_remain = ':'.join(original_title.split(":")[1:])
                    elif title[-1] == "：":
                        title = title.split("：")[0] + "："
                        sentence_remain = "：".join(original_title.split("：")[1:])

                    else:
                        # title_complete = level_check_bot(title)
                        if if_on_edge == "0":
                            title_complete = True
                        else:
                            title_complete = False
                        if title_complete:
                            sentence_remain = ""
                        else:
                            sentence_remain = title
                            title = ""

            break

    return level, section_name, title, current_title_level, level_title_record, sentence_remain


def is_numeric_or_dot(input_string):
    # 首先，使用re.sub去除字符串中的所有空格
    processed_string = re.sub(r'\s+', '', input_string)

    # 使用正则表达式检查处理后的字符串是否只包含数字和点号
    # ^ 表示字符串开始，[0-9.]+ 表示一个或多个数字或点号，$ 表示字符串结束
    if re.match(r'^[0-9.]+$', processed_string):
        return True
    else:
        return False


def is_number_and_dot(string):
    return re.fullmatch(r'[0-9.,-]+', string) is not None


def road_map_to_db_v6(road_map, company_id, report_id, level=1, master_section_id_list=[]):
    total_length = len(road_map)
    total_sections_id_list = []
    og_master_section_id_list = master_section_id_list.copy()

    complete = 0

    for section in road_map:
        total_sections_id_list.append(single_section_to_db_v6(section, company_id, report_id, level,
                                                              og_master_section_id_list))
        complete += 1
        print('\r' + '\t' * (
                level - 1) + f"level{level}入库进度: {complete}/{total_length} sections processed.")
    return total_sections_id_list


def road_map_to_db_v7(road_map, company_id, report_id, report_name, level=1, master_section_id_list=[],
                      parent_expand_instruction=False, parent_section_title=None):
    total_length = len(road_map)
    total_sections_id_list = []
    og_master_section_id_list = master_section_id_list.copy()

    complete = 0

    for section in road_map:
        total_sections_id_list.append(single_section_to_db_v7(section, company_id, report_id, report_name, level,
                                                              og_master_section_id_list, parent_expand_instruction,
                                                              parent_section_title))
        complete += 1

    return total_sections_id_list


def single_section_to_db_v6(section, company_id, report_id, level, master_section_id_list):
    db, engine = create_db_session()
    # Initialize id_to_return to a sensible default or handle cases where it might not be set.
    id_to_return = None
    try:
        og_master_section_id_list = master_section_id_list.copy()

        master_section_id_list = og_master_section_id_list.copy()
        page = section["page"]
        section_title = section["section_title"]
        if section_title == "":
            section_title = "NA"

        new_section = Section(title=section_title, company_id=company_id, report_id=report_id, section_level=level,
                              page=page)
        db.add(new_section)
        db.commit()
        master_section_id_list.append(str(new_section.id))
        retry = 0
        compete = False

        while retry < 10:
            try:

                vector = db.query(Vector).filter(Vector.text == section_title).first()
                if not Vector:  # This condition seems to check the class Vector, not the instance. Should likely be 'if not vector:'
                    print(section_title)
                    raise Exception("可能是数据有问题")

                if vector.link == 1 or compete:
                    # 已有，建立一个新的
                    master_vector_to_link = Vector(type="sentence", vector=vector.vector, company_id=company_id,
                                                   report_id=report_id, section_id=str(master_section_id_list),
                                                   level=level,
                                                   link=1, belongs_to_table=1)
                else:
                    vector.link = 1
                    vector.text = ''
                    vector.type = "sentence"
                    vector.belongs_to_table = 1
                    vector.company_id = company_id
                    vector.report_id = report_id
                    vector.section_id = str(master_section_id_list)
                    vector.level = level
                    master_vector_to_link = vector

                ##新内容
                new_sentence = Sentences(content=section_title, section=new_section, company_id=company_id,
                                         report_id=report_id, vector=[master_vector_to_link], is_title="1")
                # print('\r\t' * (level - 1) + f"正在入库{new_sentence}", end="")
                db.add_all([new_sentence, master_vector_to_link])
                db.commit()

                ##word
                words_collection = re.split(r"[,;、]", section_title)

                for word in words_collection:
                    compete2 = False
                    retry2 = 0
                    while retry2 < 10:
                        try:
                            if word == "":
                                break
                            vector = db.query(Vector).filter(Vector.text == word).first()
                            if not vector:  # Corrected from 'if not Vector:'
                                print(word)
                                raise Exception("可能是数据有问题")

                            if vector.link == 1 or compete2:
                                vector_to_link = Vector(type="word", vector=vector.vector, company_id=company_id,
                                                        report_id=report_id, section_id=str(master_section_id_list),
                                                        level=level,
                                                        link=1)
                            else:
                                vector.link = 1
                                vector.text = ''
                                vector.type = "word"
                                vector.company_id = company_id
                                vector.report_id = report_id
                                vector.section_id = str(master_section_id_list)
                                vector.level = level
                                vector_to_link = vector

                            ##新词
                            new_word = Word(content=word, sentence=new_sentence, vector=[vector_to_link])
                            # print('\r\t' * (level - 1) + f"正在入库{new_word}", end="")
                            db.add_all([new_word, vector_to_link])
                            db.commit()

                            break
                        except Exception as error:
                            print(error)
                            print(word)
                            db.rollback()
                            compete2 = True
                            # wait_time = random.uniform(0, 3)
                            # time.sleep(wait_time)
                            retry2 += 1
                            continue

                break
            except Exception as error:
                compete = True
                print(error)
                print(section)
                db.rollback()
                # Changed to check the string content of the error
                if "可能是数据有问题" in str(error):
                    # Consider re-raising a more specific error or handling appropriately
                    raise Exception("可能是数据有问题 (v6 title processing)") from error
                # wait_time = random.uniform(0, 3)
                # time.sleep(wait_time)
                retry += 1
                continue

        full_sentence = []
        for each_key in section.keys():
            if "table" == each_key:
                for table in section["table"]:
                    # new table object
                    ##check if title already exists
                    title = table["title"]

                    new_table = Chart(title=title,
                                      unit=table["unit"],
                                      potential_table_titles=str(table["potential_table_titles"]),
                                      full_header=str(table["header"]),
                                      full_key_index=str(table["key_index"]),
                                      full_value=str(table["values"]),
                                      section=new_section,
                                      page=page
                                      )
                    db.add(new_table)
                    db.commit()
                    if table["title"] != "NA":
                        ##向量化
                        compete = False
                        retry = 0
                        while retry < 10:
                            try:
                                clear_title = title
                                if clear_title == "":
                                    break
                                vector = db.query(Vector).filter(Vector.text == clear_title).first()
                                if not vector:  # Corrected
                                    print(clear_title)
                                    raise Exception("可能是数据有问题")

                                if vector.link == 1 or compete:
                                    vector_to_link = Vector(type="table", vector=vector.vector, company_id=company_id,
                                                            table=new_table,
                                                            report_id=report_id, link=1, level=level, is_table_title=1,
                                                            belongs_to_table=1)
                                else:
                                    vector.text = ''
                                    vector.link = 1
                                    vector.type = "table"
                                    vector.company_id = company_id
                                    vector.is_table_title = 1
                                    vector.belongs_to_table = 1
                                    vector.report_id = report_id
                                    vector.table = new_table
                                    vector.level = level
                                    vector_to_link = vector

                                db.add(vector_to_link)
                                db.commit()

                                break
                            except Exception as error:
                                # if "InvalidRequestError" in str(error):
                                #     break
                                compete = True
                                print(error)
                                print(clear_html_tag(table["title"]).replace("、", ",").replace(" ", ":"))
                                db.rollback()
                                if "可能是数据有问题" in str(error):  # Changed
                                    raise Exception("可能是数据有问题 (v6 table title processing)") from error
                                # wait_time = random.uniform(0, 3)
                                # time.sleep(wait_time)
                                retry += 1
                                continue

                    # new header object
                    for header_row in table["header"]:  # Assuming table["header"] is a list of rows
                        for each_basic_header in header_row:  # Iterating through cells in a header row
                            ##向量化
                            compete = False
                            retry = 0
                            while retry < 10:
                                try:
                                    if each_basic_header == "":
                                        break
                                    vector = db.query(Vector).filter(Vector.text == each_basic_header).first()
                                    if not vector:  # Corrected
                                        print(each_basic_header)
                                        raise Exception("可能是数据有问题")

                                    if vector.link == 1 or compete:

                                        vector_to_link = Vector(type="header", vector=vector.vector,
                                                                company_id=company_id,
                                                                report_id=report_id, link=1, level=level,
                                                                belongs_to_table=1)

                                    else:
                                        vector.link = 1
                                        vector.type = "header"
                                        vector.belongs_to_table = 1
                                        vector.text = ''
                                        vector.company_id = company_id
                                        vector.report_id = report_id
                                        vector.level = level
                                        vector_to_link = vector

                                    ##新单元表头
                                    new_basic_header = Header(content=each_basic_header, chart=new_table,
                                                              vector=[vector_to_link])
                                    # print('\r\t' * (level - 1) + f"正在入库{new_basic_header}", end="")
                                    db.add_all([new_basic_header, vector_to_link])
                                    db.commit()
                                    break
                                except Exception as error:
                                    compete = True
                                    print(error)
                                    print(each_basic_header)
                                    db.rollback()
                                    if "可能是数据有问题" in str(error):  # Changed
                                        raise Exception("可能是数据有问题 (v6 header processing)") from error
                                    # wait_time = random.uniform(0, 3)
                                    # time.sleep(wait_time)
                                    retry += 1
                                    continue

                    # new key_index object
                    for index, key_index_val in enumerate(
                            table["key_index"]):  # Renamed key_index to key_index_val to avoid conflict
                        ##向量化
                        compete = False
                        retry = 0
                        while retry < 10:
                            try:
                                if key_index_val == "":  # Using renamed variable
                                    break
                                vector = db.query(Vector).filter(
                                    Vector.text == key_index_val).first()  # Using renamed variable
                                if not vector:  # Corrected
                                    print(key_index_val)  # Using renamed variable
                                    raise Exception("可能是数据有问题")

                                if vector.link == 1 or compete:
                                    vector_to_link = Vector(type="key_index", vector=vector.vector,
                                                            company_id=company_id,
                                                            report_id=report_id, link=1, level=level,
                                                            belongs_to_table=1)

                                else:
                                    vector.text = ''
                                    vector.link = 1
                                    vector.type = "key_index"
                                    vector.company_id = company_id
                                    vector.belongs_to_table = 1
                                    vector.report_id = report_id
                                    vector.level = level
                                    vector_to_link = vector

                                ##新index
                                new_key_index = KeyIndex(name=key_index_val, chart=new_table, vector=[
                                    vector_to_link])  # Using renamed variable, Changed 'table' to 'chart'
                                # print('\r\t' * (level - 1) + f"正在入库{new_key_index}", end="")
                                db.add_all([new_key_index, vector_to_link])
                                db.commit()
                                ##value 和 index 之间的关系
                                if index < len(table["values"]):  # Check if index is valid for table["values"]
                                    for value_item in table["values"][index]:  # Renamed value to value_item
                                        retry2 = 0
                                        compete2 = False
                                        while retry2 < 10:
                                            try:

                                                if value_item == "" or value_item == " " or is_number_and_dot(
                                                        value_item):  # Using renamed variable
                                                    break
                                                vector = db.query(Vector).filter(
                                                    Vector.text == value_item).first()  # Using renamed variable
                                                if not vector:  # Corrected
                                                    print(value_item)  # Using renamed variable
                                                    raise Exception("可能是数据有问题")

                                                if vector.link == 1 or compete2:

                                                    vector_to_link_val = Vector(type="value", vector=vector.vector,
                                                                                # Renamed to avoid conflict
                                                                                company_id=company_id,
                                                                                report_id=report_id, link=1,
                                                                                level=level,
                                                                                belongs_to_table=1)

                                                else:
                                                    vector.text = ''
                                                    vector.link = 1
                                                    vector.type = "value"
                                                    vector.company_id = company_id
                                                    vector.belongs_to_table = 1
                                                    vector.report_id = report_id
                                                    vector.level = level
                                                    vector_to_link_val = vector  # Renamed

                                                ##新value
                                                new_value = TableValue(value=value_item, vector=[vector_to_link_val],
                                                                       # Using renamed variables
                                                                       key_index=new_key_index)
                                                # print('\r\t' * (level - 1) + f"正在入库{new_value}", end="")
                                                db.add_all([new_value, vector_to_link_val])  # Using renamed variable
                                                db.commit()
                                                break
                                            except Exception as error:
                                                compete2 = True
                                                db.rollback()
                                                if "可能是数据有问题" in str(error):  # Changed
                                                    raise Exception("可能是数据有问题 (v6 value processing)") from error
                                                # wait_time = random.uniform(0, 3)
                                                # time.sleep(wait_time)
                                                print(error)
                                                print(value_item)  # Using renamed variable
                                                retry2 += 1
                                                continue
                                else:  # Handle case where index might be out of bounds for table["values"]
                                    print(f"Warning: Key_index index {index} out of bounds for table values.")

                                break
                            except Exception as error:
                                compete = True
                                db.rollback()
                                if "可能是数据有问题" in str(error):  # Changed
                                    raise Exception("可能是数据有问题 (v6 key_index processing)") from error
                                # wait_time = random.uniform(0, 3)
                                # time.sleep(wait_time)
                                print(error)
                                print(key_index_val)  # Using renamed variable
                                retry += 1
                                continue

                    # new description object
                    for description in generate_descriptions_with_prefix(table):
                        # 向量化
                        retry = 0
                        compete = False
                        while retry < 10:
                            try:
                                if description["description"] == "":
                                    break
                                vector = db.query(Vector).filter(Vector.text == description["description"]).first()

                                if not vector:  # Corrected
                                    print(description["description"])
                                    raise Exception("可能是数据有问题")

                                if vector.link == 1 or compete:

                                    vector_to_link_desc = Vector(type="description", vector=vector.vector,
                                                                 company_id=company_id,  # Renamed
                                                                 report_id=report_id, link=1, level=level,
                                                                 belongs_to_table=1)

                                else:
                                    vector.text = ''
                                    vector.link = 1
                                    vector.type = "description"
                                    vector.belongs_to_table = 1
                                    vector.company_id = company_id
                                    vector.report_id = report_id
                                    vector.level = level
                                    vector_to_link_desc = vector  # Renamed

                                ##新描述
                                new_description = Description(content=description["description"],
                                                              location=str(description["locat"]),
                                                              chart=new_table,  # Changed 'table' to 'chart'
                                                              vector=[vector_to_link_desc])  # Renamed
                                # print('\r\t' * (level - 1) + f"正在入库{new_description}", end="")
                                db.add_all([new_description, vector_to_link_desc])  # Renamed
                                db.commit()
                                break
                            except Exception as error:
                                compete = True
                                print(error)
                                print(description["description"])
                                db.rollback()
                                if "可能是数据有问题" in str(error):  # Changed
                                    raise Exception("可能是数据有问题 (v6 description processing)") from error
                                # wait_time = random.uniform(0, 3)
                                # time.sleep(wait_time)
                                retry += 1
                                continue

                    db.add(
                        new_table)  # This seems redundant as new_table was added before. Consider removing if already added and committed.
                    db.commit()  # Same as above.
            elif "content" == each_key:

                for content_item in section["content"]:  # Renamed content to content_item
                    if content_item == "table" or content_item == "":  # Using renamed variable
                        continue
                    full_sentence.append(content_item)  # Using renamed variable
                    ##向量化
                    retry = 0
                    compete = False
                    while retry < 10:
                        try:

                            vector = db.query(Vector).filter(
                                Vector.text == content_item).first()  # Using renamed variable
                            if not vector:  # Corrected
                                print(content_item)  # Using renamed variable
                                raise Exception("可能是数据有问题")

                            if vector.link == 1 or compete:

                                vector_to_link_content = Vector(type="sentence", vector=vector.vector,
                                                                company_id=company_id,  # Renamed
                                                                report_id=report_id,
                                                                section_id=str(master_section_id_list),
                                                                level=level, link=1)

                            else:
                                vector.text = ''
                                vector.link = 1
                                vector.type = "sentence"
                                vector.company_id = company_id
                                vector.report_id = report_id
                                vector.section_id = str(master_section_id_list)
                                vector.level = level
                                vector_to_link_content = vector  # Renamed

                            ##新内容
                            new_sentence_content = Sentences(content=content_item, section=new_section,
                                                             # Renamed, using content_item
                                                             report_id=report_id, vector=[vector_to_link_content],
                                                             is_title="0",  # Renamed
                                                             page=page)
                            # print('\r\t' * (level - 1) + f"正在入库{new_sentence_content}", end="") # Renamed
                            db.add_all([new_sentence_content, vector_to_link_content])  # Renamed
                            db.commit()
                            ##word
                            words_collection_content = re.split(r"[,;、]", content_item)  # Renamed, using content_item

                            for word_item in words_collection_content:  # Renamed word to word_item
                                retry2 = 0
                                compete2 = False
                                while retry2 < 10:
                                    try:
                                        if word_item == "":  # Using renamed variable
                                            break

                                        vector = db.query(Vector).filter(
                                            Vector.text == word_item).first()  # Using renamed variable

                                        if not vector:  # Corrected
                                            print(word_item)  # Using renamed variable
                                            raise Exception("可能是数据有问题")

                                        if vector.link == 1 or compete2:

                                            vector_to_link_word = Vector(type="word", vector=vector.vector,  # Renamed
                                                                         company_id=company_id,
                                                                         report_id=report_id,
                                                                         section_id=str(master_section_id_list), link=1,
                                                                         level=level)

                                        else:
                                            vector.text = ''
                                            vector.link = 1
                                            vector.type = "word"
                                            vector.company_id = company_id
                                            vector.report_id = report_id
                                            vector.level = level
                                            vector.section_id = str(master_section_id_list)
                                            vector_to_link_word = vector  # Renamed
                                        ##新词
                                        new_word_item = Word(content=word_item, sentence=new_sentence_content,
                                                             vector=[vector_to_link_word])  # Renamed
                                        # print('\r\t' * (level - 1) + f"正在入库{new_word_item}", end="") # Renamed
                                        db.add_all([new_word_item, vector_to_link_word])  # Renamed
                                        db.commit()
                                        break

                                    except Exception as error:
                                        db.rollback()
                                        # wait_time = random.uniform(0, 3)
                                        # time.sleep(wait_time)
                                        print(error)
                                        if "可能是数据有问题" in str(error):  # Changed
                                            raise Exception("可能是数据有问题 (v6 content word processing)") from error
                                        compete2 = True
                                        retry2 += 1
                                        continue

                            break
                        except Exception as error:
                            compete = True
                            db.rollback()
                            # wait_time = random.uniform(0, 3)
                            # time.sleep(wait_time)
                            print(error)
                            print(content_item)  # Using renamed variable
                            if "可能是数据有问题" in str(error):  # Changed
                                raise Exception("可能是数据有问题 (v6 content processing)") from error
                            retry += 1
                            continue
            elif "children" == each_key:
                if len(section["children"]) > 0:
                    master_section_id_list_copy = master_section_id_list.copy()

                    children_sections_id_list = road_map_to_db_v6(section["children"], company_id,
                                                                  report_id,
                                                                  level=level + 1,
                                                                  master_section_id_list=master_section_id_list_copy,
                                                                  )
                    true_children_sections_list = []
                    for each_id in children_sections_id_list:
                        kid = db.query(Section).filter(Section.id == each_id).first()
                        if not kid:
                            print("-" * 20)
                            print(f"大问题：id_list是{children_sections_id_list}")
                            print(f"大问题：id是{each_id}")
                            print("-" * 20)
                        true_children_sections_list.append(kid)
                    new_section.sub_sections = true_children_sections_list
                else:
                    new_section.sub_sections = []

        print("\r" + " " * 20, end='')
        id_to_return = new_section.id
        db.add(new_section)  # This might be redundant if new_section is already persisted and its ID is set.
        db.commit()
        # db.close() # Moved to finally
        # engine.dispose() # Moved to finally
        return id_to_return
    except Exception as error:  # Catch any exception from the try block
        print(f"Error in single_section_to_db_v6: {error}")
        # Optionally re-raise the error or handle it
        # raise # Re-raises the caught exception
        return None  # Or a value indicating failure if preferred over raising
    finally:
        if 'db' in locals() and db is not None:
            db.close()
        if 'engine' in locals() and engine is not None:
            engine.dispose()


def commit_with_retry(db):
    for attempt in range(3):
        try:
            db.commit()
            break
        except OperationalError as e:
            print(e)
            db.rollback()
            time.sleep(1)  # 等待一段时间后重试


def single_section_to_db_v7(section, company_id, report_id, report_name, level, master_section_id_list,
                            parent_expand_instruction=False, parent_section_title=None):
    db, engine = None, None  # Initialize to None
    return_value = None  # Initialize return value
    try:
        db, engine = create_db_session()
        og_master_section_id_list = master_section_id_list.copy()

        master_section_id_list = og_master_section_id_list.copy()
        page = section["page"]
        section_title = str(section["section_title"])
        if section_title == "":
            section_title = "NA"

        table_title_expand = False
        # 检查一下，只有table，其余都是空的
        if len(section["children"]) > 0 and len(section["content"]) == 0 and len(section["table"]) == 0:
            table_title_expand = True

        new_section = Section(title=section_title, report_id=report_id, section_level=level,
                              page=page)
        db.add(new_section)
        commit_with_retry(db)

        master_section_id_list.append(str(new_section.id))

        master_vector_to_link = Vector(type="sentence", company_id=company_id, text=str(section_title),
                                       report_id=report_id, section_id=str(master_section_id_list),
                                       level=level,
                                       link=1, belongs_to_table=1)

        ##标题sentence
        new_sentence = Sentences(content=section_title, section=new_section,
                                 report_id=report_id, vector=[master_vector_to_link], is_title="1")
        # print('\r\t' * (level - 1) + f"正在入库{new_sentence}", end="")
        db.add_all([new_sentence, master_vector_to_link])
        commit_with_retry(db)

        ##标题word
        words_collection = re.split(r"[,;、]", section_title)

        for word in words_collection:

            if word == "":
                continue

            vector_to_link = Vector(type="word", text=str(word),
                                    report_id=report_id, section_id=str(master_section_id_list),
                                    level=level,
                                    link=1)

            ##新词
            new_word = Word(content=str(word), sentence=new_sentence, vector=[vector_to_link])
            # print('\r\t' * (level - 1) + f"正在入库{new_word}", end="")
            db.add_all([new_word, vector_to_link])
            commit_with_retry(db)

        full_sentence = []
        for each_key in section.keys():
            if "table" == each_key:
                for table in section["table"]:
                    # new table object
                    title = table["title"]

                    new_table = Chart(title=title,
                                      unit=str(table["unit"]),
                                      potential_table_titles=str(table["potential_table_titles"]),
                                      full_header=str(table["header"]),
                                      full_key_index=str(table["key_index"]),
                                      full_value=str(table["values"]),
                                      page=page,
                                      file_name=report_name
                                      )
                    db.add(new_table)
                    commit_with_retry(db)
                    if title != "NA":
                        ##向量化
                        vector_to_link = Vector(type="chart", company_id=company_id, text=str(title),
                                                chart=new_table,
                                                report_id=report_id, link=1, level=level, is_table_title=1,
                                                belongs_to_table=1)

                        db.add(vector_to_link)
                        commit_with_retry(db)

                    # new header object
                    for header in table["header"]:
                        for index, each_basic_header in enumerate(header):
                            ##向量化

                            if each_basic_header == "":
                                continue

                            vector_to_link = Vector(type="header", text=str(each_basic_header), company_id=company_id,
                                                    report_id=report_id, link=1, level=level,
                                                    belongs_to_table=1)

                            ##新单元表头
                            new_basic_header = Header(content=str(each_basic_header), chart=new_table,
                                                      vector=[vector_to_link])
                            # print('\r\t' * (level - 1) + f"正在入库{new_basic_header}", end="")
                            db.add_all([new_basic_header, vector_to_link])
                            commit_with_retry(db)

                    # new key_index object
                    for index, key_index in enumerate(table["key_index"]):
                        ##向量化

                        if key_index == "":
                            continue
                        vector_to_link = Vector(type="key_index", company_id=company_id,
                                                text=str(key_index),
                                                report_id=report_id, link=1, level=level,
                                                belongs_to_table=1)

                        ##新index
                        new_key_index = KeyIndex(name=str(key_index), chart=new_table, vector=[vector_to_link])
                        # print('\r\t' * (level - 1) + f"正在入库{new_key_index}", end="")
                        db.add_all([new_key_index, vector_to_link])
                        commit_with_retry(db)
                        ##value 和 index 之间的关系
                        if index < len(table["values"]):
                            for value in table["values"][index]:

                                if value == "" or value == " " or is_number_and_dot(value):
                                    continue

                                vector_to_link = Vector(type="value", text=str(value),
                                                        company_id=company_id,
                                                        report_id=report_id, link=1, level=level,
                                                        belongs_to_table=1)

                                ##新value
                                new_value = TableValue(value=str(value), vector=[vector_to_link],
                                                       key_index=new_key_index)
                                # print('\r\t' * (level - 1) + f"正在入库{new_value}", end="")
                                db.add_all([new_value, vector_to_link])
                                commit_with_retry(db)
                        else:
                            print("debug")
                    # new description object
                    for description in generate_descriptions_with_prefix(table):
                        # 向量化

                        if description["description"] == "":
                            continue

                        vector_to_link = Vector(type="description", company_id=company_id,
                                                text=str(description["description"]),
                                                report_id=report_id, link=1, level=level,
                                                belongs_to_table=1)

                        ##新描述
                        new_description = Description(content=str(description["description"]),
                                                      location=str(description["locat"]),
                                                      chart=new_table,
                                                      vector=[vector_to_link])
                        # print('\r\t' * (level - 1) + f"正在入库{new_description}", end="")
                        db.add_all([new_description, vector_to_link])
                        commit_with_retry(db)

                    db.add(new_table)
                    commit_with_retry(db)
            elif "content" == each_key:

                for content in section["content"]:
                    if content == "table" or content == "":
                        continue
                    full_sentence.append(content)
                    ##向量化

                    vector_to_link = Vector(type="sentence", company_id=company_id, text=str(content),
                                            report_id=report_id, section_id=str(master_section_id_list),
                                            level=level, link=1)

                    ##新内容
                    new_sentence = Sentences(content=str(content), section=new_section,
                                             report_id=report_id, vector=[vector_to_link], is_title="0",
                                             page=page)
                    # print('\r\t' * (level - 1) + f"正在入库{new_sentence}", end="")
                    db.add_all([new_sentence, vector_to_link])
                    commit_with_retry(db)
                    ##word
                    words_collection = re.split(r"[,;、]", str(content))

                    for word in words_collection:

                        if word == "":
                            continue

                        vector_to_link = Vector(type="word", text=str(word),
                                                company_id=company_id,
                                                report_id=report_id,
                                                section_id=str(master_section_id_list), link=1,
                                                level=level)

                        ##新词
                        new_word = Word(content=str(word), sentence=new_sentence, vector=[vector_to_link])
                        # print('\r\t' * (level - 1) + f"正在入库{new_word}", end="")
                        db.add_all([new_word, vector_to_link])
                        commit_with_retry(db)

            elif "children" == each_key:
                if table_title_expand:
                    sub_parent_expand_instruction = True
                    sub_parent_section_title = section_title
                else:
                    sub_parent_expand_instruction = False
                    sub_parent_section_title = None

                if len(section["children"]) > 0:
                    master_section_id_list_copy = master_section_id_list.copy()

                    children_sections_id_list = road_map_to_db_v7(section["children"], company_id,
                                                                 report_id, report_name=report_name,
                                                                 level=level + 1,
                                                                 master_section_id_list=master_section_id_list_copy,
                                                                 parent_expand_instruction=sub_parent_expand_instruction,
                                                                 parent_section_title=sub_parent_section_title
                                                                 )
                    true_children_sections_list = []
                    for each_id in children_sections_id_list:
                        kid = db.query(Section).filter(Section.id == each_id).first()
                        if not kid:
                            print("-" * 20)
                            print(f"大问题：id_list是{children_sections_id_list}")
                            print(f"大问题：id是{each_id}")
                            print("-" * 20)
                        true_children_sections_list.append(kid)
                    new_section.sub_sections = true_children_sections_list
                else:
                    new_section.sub_sections = []

        # print("\r" + " " * 20, end='')
        id_to_return = new_section.id
        db.add(new_section)
        commit_with_retry(db)
        db.close()
        engine.dispose()
        return id_to_return
    except Exception as error:
        print(error)
        raise Exception("single_section_to_db_v7出问题")
    finally:
        if db is not None:  # Check if db was assigned
            db.close()
        if engine is not None:  # Check if engine was assigned
            engine.dispose()


def road_map_text_v6_flat(road_map, db, report_id, report_name, level=1, master_section_id_list=[],
                          parent_expand_instruction=False, parent_section_title=None):
    road_map_index = 0
    try:
        for section in road_map:

            section_title = section["section_title"]

            if pd.isna(section_title) or section_title == "":
                section_title = "NA"
            # new temp text
            new_temp_text = Vector(text=section_title)
            db.add(new_temp_text)
            db.commit()

            table_title_expand = False
            # 检查一下，只有table，其余都是空的
            if len(section["children"]) > 0 and len(section["content"]) == 0 and len(section["table"]) == 0:
                table_title_expand = True

            ##word
            words_collection = re.split(r"[,;、]", section_title)

            for word in words_collection:
                if pd.isna(word) or word == "":
                    continue
                new_temp_text = Vector(text=word)
                db.add(new_temp_text)
                db.commit()

            for each_key in section.keys():
                if "table" == each_key and len(section["table"]) > 0:
                    for table in section["table"]:
                        # new table object
                        ##check if title already exists
                        title = table["title"]
                        if pd.isna(title):
                            title = "NA"
                        clear_title = clear_html_tag(title).replace("、", ",")

                        if clear_title != "":
                            if parent_expand_instruction == True and parent_section_title and clear_title not in parent_section_title.replace(
                                    " ", ""):

                                retry = 0
                                while retry < 10:
                                    try:
                                        if table_title_expand_bot(parent_section_title, clear_title)[1]:
                                            true_table_title = f"{parent_section_title}:{clear_title}"

                                            clear_title = true_table_title
                                        break
                                    except Exception as error:
                                        print(error)
                                        print(f"标题扩充bot出问题，马上开始重试")
                                        retry += 1
                                        continue

                            table["title"] = clear_title
                            new_temp_text = Vector(text=clear_title)
                            db.add(new_temp_text)
                            db.commit()

                        # new header object
                        for header in table["header"]:
                            for index, each_basic_header in enumerate(header):
                                ##向量化
                                if pd.isna(each_basic_header) or each_basic_header == "":
                                    continue
                                new_temp_text = Vector(text=each_basic_header)
                                db.add(new_temp_text)
                                db.commit()

                        # new key_index object
                        for index, key_index in enumerate(table["key_index"]):
                            ##向量化

                            if pd.isna(key_index) or key_index == "":
                                continue
                            new_temp_text = Vector(text=key_index)
                            db.add(new_temp_text)
                            db.commit()

                            ##value
                            for value in table["values"][index]:

                                if pd.isna(value) or value == "" or value == " " or is_number_and_dot(value):
                                    continue
                                new_temp_text = Vector(text=value)
                                db.add(new_temp_text)
                                db.commit()

                        # new description object
                        for description in generate_descriptions_with_prefix(table):

                            if pd.isna(description["description"]) or description["description"] == "":
                                continue
                            new_temp_text = Vector(text=description["description"])
                            db.add(new_temp_text)
                            db.commit()


                elif "content" == each_key:

                    for content in section["content"]:
                        if content == "table" or content == "" or pd.isna(content):
                            continue
                        new_temp_text = Vector(text=content)
                        db.add(new_temp_text)
                        db.commit()

                        ##word
                        words_collection = re.split(r"[,;、]", content)

                        for word in words_collection:
                            if word == "" or pd.isna(word):
                                continue
                            new_temp_text = Vector(text=word)
                            db.add(new_temp_text)
                            db.commit()


                elif "children" == each_key:
                    if table_title_expand:
                        sub_parent_expand_instruction = True
                        sub_parent_section_title = section_title
                    else:
                        sub_parent_expand_instruction = False
                        sub_parent_section_title = None
                    master_section_id_list_copy = master_section_id_list.copy()
                    road_map_index = road_map_text_v6_flat(section["children"], db, report_id,
                                                           report_name,
                                                           level=level + 1,
                                                           master_section_id_list=master_section_id_list_copy,
                                                           parent_expand_instruction=sub_parent_expand_instruction,
                                                           parent_section_title=sub_parent_section_title
                                                           )

    except Exception as error:
        print(error)
        db.rollback()
        raise Exception("road_map_text_v6_flat出问题")
    return road_map_index


def single_embedding_groups_retrieve(current_offset, batch_size=1999):
    """
    为一批记录生成嵌入向量的函数
    优化：减少数据库连接开关次数
    """
    # 在函数开始时创建数据库连接
    db, engine = create_db_session()

    try:
        batch = (db.query(Vector).order_by(Vector.id)
                 .limit(batch_size)
                 .offset(current_offset)
                 .all())
        ready_to_embedding_list = []
        db_object_list = []
        for each_temp_record in batch:
            try:
                if each_temp_record.vector:
                    continue
                text = each_temp_record.text.replace("\n", " ").replace(" ", "")
                if text == "" or text == " ":
                    each_temp_record.have_embedding = 1
                    continue
                ready_to_embedding_list.append(text)
                db_object_list.append(each_temp_record)
            except Exception as e:
                continue

        content_list = ready_to_embedding_list
        num_processed_this_batch = len(content_list)  # Store the number of items to process

        # print(f"current_content_list: {content_list}")

        if num_processed_this_batch == 0:
            return True, 0  # Success, 0 items processed
        try:
            # 增加chunk_size，减少API调用次数
            chunk_size = 25  # 原来是10，增加到25

            # 只提交一次数据库事务，而不是每个chunk都提交
            for i in range(0, len(content_list), chunk_size):
                content_chunk = content_list[i:i + chunk_size]
                db_object_chunk = db_object_list[i:i + chunk_size]

                if not content_chunk:  # Skip if the chunk is empty
                    continue

                retrieved_embeddings = get_cluster_embeddings(content_chunk)
                # print(f"retrieved_embeddings for chunk: {retrieved_embeddings}") # Optional: for debugging
                for index, embedding_vector in enumerate(retrieved_embeddings):
                    db_object_chunk[index].vector = pickle.dumps(embedding_vector)
                    db_object_chunk[index].text = None
                    db_object_chunk[index].have_embedding = 1

            # 批量提交所有更改，减少数据库交互次数
            db.commit()

            return True, num_processed_this_batch  # Success, return count of processed items
        except Exception as e:
            print(f"群组embedding 入库发生问题: {e}")
            db.rollback()
            time.sleep(5)

            return False, str(e), 0  # Failure, error message, 0 items processed
    finally:
        # 确保数据库连接总是关闭
        db.close()
        engine.dispose()
        del batch
        # 确保只在 retrieved_embeddings 已赋值时才删除
        if 'retrieved_embeddings' in locals():
            del retrieved_embeddings
        del db_object_list
        del content_list
        gc.collect()


def get_road_map_embedding_trunks():
    try:
        db, engine = create_db_session()
        # 假设我们知道数据的总量，或者可以通过查询获得
        first_empty_vector_record = db.query(Vector).filter(Vector.have_embedding.is_(None)).order_by(Vector.id).first()

        if first_empty_vector_record:
            # 获取该记录的vector.id
            target_vector_id = first_empty_vector_record.id

            # 查询所有vector.id小于该记录的vector.id的记录数量来确定current_offset
            current_offset = db.query(func.count(Vector.id)).filter(Vector.id < target_vector_id).scalar()
        else:
            # 如果没有找到符合条件的记录，可以根据需要处理这种情况
            return True, []

        total_records = db.query(func.count(Vector.id)).scalar()

        embedding_trunks = []
        # 每一个trunks的长度都是1999，然后要包含start_id
        while current_offset < total_records:
            embedding_trunks.append({'trunk_size': 1999, 'start_id': current_offset})
            current_offset += 1999
        data = embedding_trunks

        return True, data
    except Exception as e:
        print(f"split_road_map_embedding_trunks 出现问题: {e}")
        fail_message = str(e)
        return False, fail_message


def pre_road_map_embedding_convert():
    db, engine = create_db_session()
    # 假设我们知道数据的总量，或者可以通过查询获得
    first_empty_vector_record = db.query(Vector).filter(Vector.have_embedding.is_(None)).order_by(Vector.id).first()

    if first_empty_vector_record:
        # 获取该记录的vector.id
        target_vector_id = first_empty_vector_record.id

        # 查询所有vector.id小于该记录的vector.id的记录数量来确定current_offset
        current_offset = db.query(func.count(Vector.id)).filter(Vector.id < target_vector_id).scalar()
    else:
        # 如果没有找到符合条件的记录，可以根据需要处理这种情况
        return True, []

    total_records = db.query(func.count(Vector.id)).scalar()
    success_num = 0

    # 要定时重启，内存占用太多了
    print('开始提取向量')

    # 调整进程池大小，根据CPU核心数自动调整（但设置上限以避免过多进程）
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    workers = min(cpu_count * 2, 16)  # 使用CPU核心数的2倍，但最多32个进程

    # 优化批处理参数
    max_concurrent_task_submissions = 50  # 保持不变或根据实际情况调整
    batch_size = 100  # 与之前修改匹配

    print(f"使用进程池大小: {workers}, 最大并发任务: {max_concurrent_task_submissions}, 批处理大小: {batch_size}")

    # Create ProcessPoolExecutor and tqdm instances once, outside the main loop
    with ProcessPoolExecutor(max_workers=workers) as executor:
        with tqdm(total=total_records, initial=success_num, desc="处理embedding", unit="条") as pbar:

            while current_offset < total_records:
                # This print can be removed if tqdm provides enough info, or kept for checkpoints
                print(
                    f"总计需要处理: {total_records} 条记录，当前已处理: {success_num} 条，剩余: {total_records - current_offset} 条")

                futures = []
                # Determine how many tasks to submit in this "submission batch"
                tasks_to_submit_in_this_round = 0

                submission_batch_start_offset = current_offset

                # 提交任务批次
                while (submission_batch_start_offset < total_records and
                       tasks_to_submit_in_this_round < max_concurrent_task_submissions):
                    # 将批处理大小从40增加到100
                    future = executor.submit(auto_retry, single_embedding_groups_retrieve,
                                             submission_batch_start_offset, batch_size)
                    futures.append(future)
                    submission_batch_start_offset += batch_size  # 相应地调整偏移量
                    tasks_to_submit_in_this_round += 1

                if not futures:  # Should not happen if current_offset < total_records initially
                    break

                # 使用as_completed更高效地处理结果
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        status = result[0]

                        if status:
                            processed_in_batch = result[1]  # 成功处理的记录数
                            success_num += processed_in_batch
                            pbar.update(processed_in_batch)
                        else:
                            error_msg = result[1]
                            # 确保tqdm.write可用或适当处理错误日志
                            pbar.write(f"embedding 入库发生问题: {error_msg}。此批次0条成功。")
                    except Exception as e:
                        pbar.write(f"处理批次时发生严重错误: {e}")

                # 更新当前偏移量
                current_offset = submission_batch_start_offset

    # 释放资源
    db.commit()
    db.close()
    engine.dispose()
    print("所有记录已经处理完")
    return True


def road_map_to_structure_v2(road_map):
    level_one_sections = []
    stack = []
    pre_level = None
    for key, value in road_map.items():
        current_level = value["level"]
        if len(stack) == 0:
            stack.append({key: value})
            continue
        else:
            pre_level = stack[-1][list(stack[-1].keys())[0]]["level"]

        if current_level > pre_level:
            stack.append({key: value})
            continue

        while current_level <= pre_level:
            if pre_level == 1:
                level_one_sections.append(stack.pop())
                break

            kid = stack.pop()
            kid_key = list(kid.keys())[0]
            master = stack.pop()
            master_key = list(master.keys())[0]
            if "content" in kid[kid_key].keys():
                kid_content = kid[kid_key]["content"]
            else:
                kid_content = []
            if "table" in kid[kid_key].keys():
                kid_table = kid[kid_key]["table"]
            else:
                kid_table = None
            if "section" in kid[kid_key].keys():
                kid_section = kid[kid_key]["section"]
            else:
                kid_section = None

            if kid_table or kid_section or len(kid_content) > 0:
                kid_true_type = "section"
            else:
                kid_true_type = "sentence"

            master_content = master[master_key]["content"]

            # 找出上级
            if len(master_content) > 0 and "modified" not in master[
                master_key].keys() and current_level >= pre_level and len(master_content[-1]) > 0 and (
                    master_content[-1][-1] == ":" or master_content[-1][-1] == "："):
                true_master = {
                    master[master_key]["content"].pop(): {"content": [], "section": [], "table": [], "modified": "add",
                                                          "level": pre_level - 0.5, "page": master[master_key]["page"]}}

                new_master_section = {}
                if kid_true_type == "section":
                    kid[list(kid.keys())[0]]["type"] = "section"
                    if "section" in true_master[list(true_master.keys())[0]].keys():
                        # check_children_content_have_section
                        if len(kid_content) > 0 and "modified" not in kid[
                            kid_key].keys() and len(kid_content[-1]) > 0 and (
                                kid_content[-1][-1] == ":" or kid_content[-1][-1] == "："):
                            new_master_section = {
                                kid[kid_key]["content"].pop(): {"content": [], "section": [], "table": [],
                                                                "modified": "add",
                                                                "level": pre_level - 0.25,
                                                                "page": master[master_key]["page"]}}

                        true_master[list(true_master.keys())[0]]["section"].append(kid)
                    else:
                        if len(kid_content) > 0 and "modified" not in kid[
                            kid_key].keys() and len(kid_content[-1]) > 0 and (
                                kid_content[-1][-1] == ":" or kid_content[-1][-1] == "："):
                            new_master_section = {
                                kid[kid_key]["content"].pop(): {"content": [], "section": [], "table": [],
                                                                "modified": "add",
                                                                "level": pre_level - 0.25,
                                                                "page": master[master_key]["page"]}}

                        true_master[list(true_master.keys())[0]]["section"] = [kid]
                else:
                    kid[list(kid.keys())[0]]["type"] = "plain"
                    if "section" in true_master[list(true_master.keys())[0]].keys():
                        if len(kid_content) > 0 and "modified" not in kid[
                            kid_key].keys() and len(kid_content[-1]) > 0 and (
                                kid_content[-1][-1] == ":" or kid_content[-1][-1] == "："):
                            new_master_section = {
                                kid[kid_key]["content"].pop(): {"content": [], "section": [], "table": [],
                                                                "modified": "add",
                                                                "level": pre_level - 0.25,
                                                                "page": master[master_key]["page"]}}
                        true_master[list(true_master.keys())[0]]["section"].append(kid)
                    else:
                        if len(kid_content) > 0 and "modified" not in kid[
                            kid_key].keys() and len(kid_content[-1]) > 0 and (
                                kid_content[-1][-1] == ":" or kid_content[-1][-1] == "："):
                            new_master_section = {
                                kid[kid_key]["content"].pop(): {"content": [], "section": [], "table": [],
                                                                "modified": "add",
                                                                "level": pre_level - 0.25,
                                                                "page": master[master_key]["page"]}}
                        true_master[list(true_master.keys())[0]]["section"] = [kid]
                stack.append(master)
                stack.append(true_master)
                if len(new_master_section) > 0:
                    stack.append(new_master_section)
            else:
                true_master = master
                new_master_section = {}
                if kid_true_type == "section":
                    kid[list(kid.keys())[0]]["type"] = "section"
                    if "section" in true_master[list(true_master.keys())[0]].keys():
                        if len(kid_content) > 0 and "modified" not in kid[
                            kid_key].keys() and len(kid_content[-1]) > 0 and (
                                kid_content[-1][-1] == ":" or kid_content[-1][-1] == "："):
                            new_master_section = {
                                kid[kid_key]["content"].pop(): {"content": [], "section": [], "table": [],
                                                                "modified": "add",
                                                                "level": pre_level - 0.25,
                                                                "page": master[master_key]["page"]}}
                        true_master[list(true_master.keys())[0]]["section"].append(kid)
                    else:
                        if len(kid_content) > 0 and "modified" not in kid[
                            kid_key].keys() and len(kid_content[-1]) > 0 and (
                                kid_content[-1][-1] == ":" or kid_content[-1][-1] == "："):
                            new_master_section = {
                                kid[kid_key]["content"].pop(): {"content": [], "section": [], "table": [],
                                                                "modified": "add",
                                                                "level": pre_level - 0.25,
                                                                "page": master[master_key]["page"]}}
                        true_master[list(true_master.keys())[0]]["section"] = [kid]
                else:
                    kid[list(kid.keys())[0]]["type"] = "plain"
                    if "section" in true_master[list(true_master.keys())[0]].keys():
                        if len(kid_content) > 0 and "modified" not in kid[
                            kid_key].keys() and len(kid_content[-1]) > 0 and (
                                kid_content[-1][-1] == ":" or kid_content[-1][-1] == "："):
                            new_master_section = {
                                kid[kid_key]["content"].pop(): {"content": [], "section": [], "table": [],
                                                                "modified": "add",
                                                                "level": pre_level - 0.25,
                                                                "page": master[master_key]["page"]}}
                        true_master[list(true_master.keys())[0]]["section"].append(kid)
                    else:
                        if len(kid_content) > 0 and "modified" not in kid[
                            kid_key].keys() and len(kid_content[-1]) > 0 and (
                                kid_content[-1][-1] == ":" or kid_content[-1][-1] == "："):
                            new_master_section = {
                                kid[kid_key]["content"].pop(): {"content": [], "section": [], "table": [],
                                                                "modified": "add",
                                                                "level": pre_level - 0.25,
                                                                "page": master[master_key]["page"]}}
                        true_master[list(true_master.keys())[0]]["section"] = [kid]
                stack.append(true_master)
                if len(new_master_section) > 0:
                    stack.append(new_master_section)
            pre_level = stack[-1][list(stack[-1].keys())[0]]["level"]

        stack.append({key: value})

    # 清空stack
    while len(stack) > 0:
        if len(stack) == 1:
            level_one_sections.append(stack.pop())
            break

        kid = stack.pop()
        kid_key = list(kid.keys())[0]
        master = stack.pop()
        master_key = list(master.keys())[0]
        kid_content = kid[kid_key]["content"]

        if "table" in kid[kid_key].keys():
            kid_table = kid[kid_key]["table"]
        else:
            kid_table = None
        if "section" in kid[kid_key].keys():
            kid_section = kid[kid_key]["section"]
        else:
            kid_section = None

        if kid_table or kid_section or len(kid_content) > 0:
            kid_true_type = "section"
        else:
            kid_true_type = "sentence"

        master_content = master[master_key]["content"]

        # 找出上级
        if len(master_content) > 0 and "modified" not in master[master_key].keys() and len(master_content[-1]) > 0 and (
                master_content[-1][-1] == ":" or master_content[-1][-1] == "："):
            true_master = {
                master[master_key]["content"].pop(): {"content": [], "section": [], "table": [], "modified": "add",
                                                      "level": pre_level, "page": master[master_key]["page"]}}
            if kid_true_type == "section":
                kid[list(kid.keys())[0]]["type"] = "section"
                if "section" in true_master[list(true_master.keys())[0]].keys():

                    true_master[list(true_master.keys())[0]]["section"].append(kid)
                else:

                    true_master[list(true_master.keys())[0]]["section"] = [kid]
            else:
                kid[list(kid.keys())[0]]["type"] = "plain"
                if "section" in true_master[list(true_master.keys())[0]].keys():

                    true_master[list(true_master.keys())[0]]["section"].append(kid)
                else:

                    true_master[list(true_master.keys())[0]]["section"] = [kid]
            stack.append(master)
            stack.append(true_master)
        else:
            true_master = master
            if kid_true_type == "section":
                kid[list(kid.keys())[0]]["type"] = "section"
                if "section" in true_master[list(true_master.keys())[0]].keys():

                    true_master[list(true_master.keys())[0]]["section"].append(kid)
                else:

                    true_master[list(true_master.keys())[0]]["section"] = [kid]
            else:
                kid[list(kid.keys())[0]]["type"] = "plain"
                if "section" in true_master[list(true_master.keys())[0]].keys():

                    true_master[list(true_master.keys())[0]]["section"].append(kid)
                else:

                    true_master[list(true_master.keys())[0]]["section"] = [kid]
            stack.append(true_master)
    return_dict = {}
    for each_dict in level_one_sections:
        return_dict.update(each_dict)
    return return_dict


def judge_is_on_edge(df):
    """
        is_on_left_edge  是不是左顶格
        is_on_right_edge 是不是右顶格
        '1'表示是, '0'表示否
    """
    total_len = len(df)
    df = pd.DataFrame(df)

    def get_width(data):

        half_number = int(len(data) / 2)
        quart_number = int(len(data) / 4)
        left = []
        right = []
        for n in range(quart_number, half_number):
            for item in data.loc[n, 0]['lines']:
                left.append(item['position'][0])
                right.append(item['position'][2])

        left.sort()
        right.sort()
        top_l = sum(left[-10:]) / 10
        top_r = sum(right[-10:]) / 10

        return top_l, top_r

    left_X, right_X = get_width(df)
    for n in range(len(df)):
        tmp = df.loc[n, 0]['lines']
        for m in range(len(tmp)):
            position = tmp[m]['position']
            left_x = position[0]
            right_x = position[2]

            if right_X - right_x < 25:
                tmp[m]['isOnRightEdge'] = '1'
            else:
                tmp[m]['isOnRightEdge'] = '0'
            if left_x - left_X < 25:
                tmp[m]['isOnLeftEdge'] = '1'
            else:
                tmp[m]['isOnLeftEdge'] = '0'

    # df = df.to_dict(orient='list')
    final_result = []
    for key, row in df.iterrows():
        # 遍历每一行的所有列
        list_to_append = []
        for item in row:
            # 只有当元素不是None且不为空时，才添加到结果中
            if item not in [None, '', []]:  # 这里''和[]用于捕获空字符串和空列表
                list_to_append.append(item)
        final_result.append(list_to_append)
    # df = df[list(df.keys())[0]]
    # for item in df:
    #     final_result.append([item])
    return final_result


def judge_is_on_edge_v2(df):
    total_len = len(df)
    df = pd.DataFrame(df)
    try:
        def get_width(data):
            half_number = int(len(data) / 2)
            quart_number = int(len(data) / 4)
            left = []
            right = []
            for n in range(quart_number, half_number):
                try:
                    for item in data.loc[n, 0]['lines']:
                        left.append(item['position'][0])
                        right.append(item['position'][2])
                except Exception as e:
                    continue

            left.sort()
            right.sort()
            top_l = sum(left[-10:]) / 10
            top_r = sum(right[-10:]) / 10

            return top_l, top_r

        left_X, right_X = get_width(df)
        for n in range(len(df)):
            try:
                tmp = df.loc[n, 0]['lines']
                for m in range(len(tmp)):
                    position = tmp[m]['position']
                    left_x = position[0]
                    right_x = position[2]

                    if right_X - right_x < 25:
                        tmp[m]['isOnRightEdge'] = '1'
                    else:
                        tmp[m]['isOnRightEdge'] = '0'
                    if left_x - left_X < 25:
                        tmp[m]['isOnLeftEdge'] = '1'
                    else:
                        tmp[m]['isOnLeftEdge'] = '0'
            except Exception as e:
                continue
    except Exception as e:
        pass
        # 将处理后的数据重新组装成最终结果
    final_result = []
    for key, row in df.iterrows():
        list_to_append = []
        for item in row:
            if item not in [None, '', []]:  # 这里''和[]用于捕获空字符串和空列表
                list_to_append.append(item)
        final_result.append(list_to_append)

    return final_result


def standard_json_switch_v2(og_dict, starter=1):
    def sub_standard_json_switch(og_dict, starter=1):
        standard_json_to_return = []
        new_dict = {}
        skip_list = ["content", "page", "level", "table", "type"]
        children = []
        if isinstance(og_dict, dict):
            for key, value in og_dict.items():
                if key in skip_list:
                    new_dict[key] = value
                elif key == "section":
                    children.append(sub_standard_json_switch(value, 0))
                elif starter == 1:
                    temp_dict = {}
                    temp_dict["section_title"] = key
                    temp_dict.update(sub_standard_json_switch(value, 0))
                    standard_json_to_return.append(temp_dict)
                else:
                    new_dict["section_title"] = key

        else:
            for each_dict in og_dict:
                for key, value in each_dict.items():
                    sub_new_dict = {}
                    sub_children = []
                    sub_new_dict["section_title"] = key
                    for sub_key, sub_value in value.items():
                        if sub_key in skip_list:
                            sub_new_dict[sub_key] = sub_value
                            continue
                        elif sub_key == "section":
                            sub_children = sub_standard_json_switch(sub_value, 0)

                    sub_new_dict["children"] = sub_children
                    children.append(sub_new_dict)

            return children

        new_dict["children"] = children

        if standard_json_to_return:
            return standard_json_to_return
        else:
            return new_dict

    if starter == 1:
        standard_list_of_dict = [{key: value} for key, value in og_dict.items()]
    else:
        standard_list_of_dict = og_dict
    standard_json_to_return = []
    skip_list = ["content", "page", "level", "table", "type"]
    # skip_list = ["content", "table", "type"]

    for each_section_dict in standard_list_of_dict:
        new_dict = {}
        children = []
        for title, structure in each_section_dict.items():
            # 只会循环一次，因为each_section_dict只有一个元素
            new_dict["section_title"] = title
            for key, value in structure.items():
                if key in skip_list:
                    new_dict[key] = value
                elif key == "section":
                    list_of_children = value
                    children = (sub_standard_json_switch(list_of_children, 0))

        new_dict["children"] = children
        standard_json_to_return.append(new_dict)

    return standard_json_to_return

# if __name__ == '__main__':
# db, engine = create_db_session()
# db.query(Vector).update({Vector.text: None}, synchronize_session=False)
# db.commit()
# db.close()
# engine.dispose()
# first_empty_vector_record = db.query(Vector).filter(Vector.vector == None).order_by(Vector.id).first()
#
# if first_empty_vector_record:
#     # 获取该记录的vector.id
#     target_vector_id = first_empty_vector_record.id
#
#     # 查询所有vector.id小于该记录的vector.id的记录数量来确定current_offset
#     current_offset = db.query(func.count(Vector.id)).filter(Vector.id < target_vector_id).scalar()
#
#     print(current_offset)
# db.close()
# engine.dispose()
