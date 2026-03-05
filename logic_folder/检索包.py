import copy
import json
import os
import pickle
import logging

import numpy as np
from openai import OpenAI

from scipy.spatial.distance import cdist
from dotenv import load_dotenv
from sqlalchemy import and_, create_engine, or_
from sqlalchemy.orm import undefer, sessionmaker

from 模块工具.openai相关工具 import count_gpt_tokens, get_cluster_embeddings
from 模块工具.智能体仓库 import find_table_target_bot, find_section_target_bot

load_dotenv()
from logic_folder.数据库表格 import Company, Section, Chart, Vector, Sentences, Reports, Indicators, Missions, \
    FailReports, IndicatorsResults
from sqlalchemy.orm.collections import InstrumentedList

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

# 创建logger对象
logger = logging.getLogger(__name__)

# 配置数据库连接
def create_db_engine(echo=False):
    engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}', echo=echo)
    return engine


def create_db_session():
    # 使用已经创建的引擎来创建新的会话
    engine = create_db_engine(False)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session, engine


def cosine_similarity(embedding1, embedding2):
    return 1 - cdist(embedding1, embedding2, metric='cosine')


def search_relevant(query_embeddings, corpus, top_k=999, min_similarity=0.5):
    # 首先，需要从Vector实例中提取所有向量并反序列化

    corpus_embeddings = np.vstack([pickle.loads(entry.vector) for entry in corpus if entry.vector])

    # 将查询嵌入转换为NumPy数组，并增加一个维度以适应余弦相似度函数
    query_embedding_np = np.array(query_embeddings)
    # query_embedding_2d = query_embedding_np[np.newaxis, :]
    # 计算查询嵌入与语料库嵌入之间的余弦相似度
    similarity_scores = cosine_similarity(query_embedding_np, corpus_embeddings)

    # 计算每个查询的最高相似度分数的索引
    sorted_indices = np.argsort(-similarity_scores, axis=1)

    # 初始化一个用于存储结果的列表
    all_top_indices_scores = []

    # 遍历每个查询的相似度分数
    for query_idx in range(similarity_scores.shape[0]):
        # 获取当前查询的所有相似度分数的排序索引
        query_sorted_indices = sorted_indices[query_idx, :]

        # 从已排序的索引中选择前 top_k 个，如果 top_k 比总数多，则选择所有的索引
        top_k_indices = query_sorted_indices[:min(top_k, query_sorted_indices.size)]

        # 选择高于最小相似度阈值的索引和分数
        top_indices_scores = [(corpus[index], similarity_scores[query_idx, index]) for index in top_k_indices if
                              similarity_scores[query_idx, index] >= min_similarity]

        # 将结果添加到列表中
        all_top_indices_scores += top_indices_scores

    # 返回所有查询的最相关结果
    return all_top_indices_scores


def from_header_to_column(header, table):
    table = table
    whole_header = eval(table.full_header)
    # find header start_i, start_j, end_j
    header_i = 0
    header_j = 0
    header_end_j = 0
    # iterate through whole_header list of list
    for i in range(len(whole_header)):
        for j in range(len(whole_header[i])):
            if whole_header[i][j] == header:
                header_i = i
                header_j = j
                break
    # find header end_j
    for j in range(header_j, len(whole_header[header_i])):
        if whole_header[header_i][j] != header:
            header_end_j = j - 1
            break
    # extract header
    header_to_return = []
    for i in range(len(whole_header)):
        if len(header_to_return) < i + 1:
            header_to_return.append([])
        header_to_return[i].append(whole_header[i][0])
    for i in range(len(whole_header)):
        for j in range(header_j, header_end_j + 1):
            to_add = whole_header[i][j]
            if to_add not in header_to_return:
                header_to_return[i].append(to_add)

    ## zero column and exact column
    whole_value = eval(table.full_value)
    value_to_return = []
    for i in range(len(whole_value)):
        if len(value_to_return) < i + 1:
            value_to_return.append([])
        value_to_return[i].append(whole_value[i][0])
    for i in range(len(whole_value)):
        for j in range(header_j, header_end_j + 1):
            value_to_return[i].append(whole_value[i][j])

    return header_to_return, value_to_return


def from_key_index_to_column(key_index, table):
    table = table
    header_to_return = eval(table.full_header)
    first_index = eval(table.full_key_index).index(key_index)
    key_index_list = eval(table.full_key_index)
    last_index = len(key_index_list) - key_index_list[::-1].index(key_index) - 1
    value_to_return = eval(table.full_value)[first_index:last_index + 1]
    return header_to_return, value_to_return


def from_description_to_column(description, table):
    table = table
    locations = eval(description.location)
    header = eval(table.full_header)
    value = eval(table.full_value)
    header_to_return = []
    value_to_return = []
    for i in range(len(header)):
        if len(header_to_return) < i + 1:
            header_to_return.append([])
        header_to_return[i].append(header[i][0])

    value_to_return.append(value[locations[0][0]][0])
    for i in range(len(header)):
        for j in range(len(locations)):
            header_to_return[i].append(header[i][locations[j][1]])
    for j in range(len(locations)):
        value_to_return.append(value[locations[0][0]][locations[j][1]])
    return header_to_return, value_to_return


def parse_search_result(search_result, db, report_id, mode=0, long_context_mode=True):
    original_structure = []
    must_have_content = ["合计", "总计", "总数", "合计数", "总额", "合计额"]
    unique_table_id = {"table": [], "key_index": {}, "header": [], "description": [], "section": [], "sentence": []}
    for entry in search_result:
        item = entry[0]
        score = entry[1]
        if item.type == "sentence":
            sentence = item.sentence

            if long_context_mode:
                context_sentences = ""
                og_section = sentence.section
                all_og_sentences = og_section.sentences.all()
                target_index = None
                for i, sentence_obj in enumerate(all_og_sentences):
                    if sentence_obj.id == sentence.id:
                        target_index = i
                        break

                if target_index is not None and target_index - 1 >= 0:
                    context_sentences += all_og_sentences[target_index - 1].content
                context_sentences += sentence.content
                if target_index is not None and target_index + 1 < len(all_og_sentences):
                    context_sentences += all_og_sentences[target_index + 1].content
                pack = {"sentence": context_sentences, "score": score, "vector": item, "og_sentence": sentence}
                if sentence.id not in unique_table_id["sentence"]:
                    unique_table_id["sentence"].append(sentence.id)
                    original_structure.append(pack)
            else:
                pack = {"sentence": sentence.content, "score": score, "vector": item, "og_sentence": sentence}
                if sentence.id not in unique_table_id["sentence"]:
                    unique_table_id["sentence"].append(sentence.id)
                    original_structure.append(pack)

        elif item.type == "word":
            sentence = item.word.sentence

            if long_context_mode:
                context_sentences = ""
                og_section = sentence.section
                all_og_sentences = og_section.sentences.all()
                target_index = None
                for i, sentence_obj in enumerate(all_og_sentences):
                    if sentence_obj.id == sentence.id:
                        target_index = i
                        break

                if target_index is not None and target_index - 1 >= 0:
                    context_sentences += all_og_sentences[target_index - 1].content
                context_sentences += sentence.content
                if target_index is not None and target_index + 1 < len(all_og_sentences):
                    context_sentences += all_og_sentences[target_index + 1].content
                pack = {"sentence": context_sentences, "score": score, "vector": sentence.vector,
                        "og_sentence": sentence}
                if sentence.id not in unique_table_id["sentence"]:
                    unique_table_id["sentence"].append(sentence.id)
                    original_structure.append(pack)
            else:
                pack = {"sentence": sentence.content, "score": score, "vector": sentence.vector,
                        "og_sentence": sentence}
                if sentence.id not in unique_table_id["sentence"]:
                    unique_table_id["sentence"].append(sentence.id)
                    original_structure.append(pack)

        elif item.type == "chart":
            chart = item.chart

            if ":" in chart.title:
                true_section_name = chart.title.split(":")[1]
            else:
                true_section_name = chart.title

            if "," in true_section_name:
                true_section_name = true_section_name.split(",")[1]

            location = db.query(Section).filter(
                and_(Section.title.like(f'%{true_section_name}%'), Section.report_id == report_id)).first()
            if location:
                all_sentences = location.sentences
                remaining_content = [sentence.content for sentence in all_sentences]
            else:
                remaining_content = []

            pack = {"title": chart.title, "unit": chart.unit, "header": chart.full_header, "content": chart.full_value,
                    "score": score, "type": "table", "table_id": chart.id, "page": chart.page,
                    "remaining_content": remaining_content}

            original_structure.append(pack)
        elif item.type == "header":

            chart = item.header.chart

            header_to_return, value_to_return = from_header_to_column(item.header.content, chart)
            if mode == 0:
                pack = {"title": chart.title, "unit": chart.unit, "header": header_to_return,
                        "content": value_to_return,
                        "score": score, "type": "header", "table_id": chart.id, "page": chart.page}
            else:
                pack = {"title": chart.title, "unit": chart.unit, "header": chart.full_header,
                        "content": chart.full_value,
                        "score": score, "type": "table", "table_id": chart.id, "page": chart.page}
            original_structure.append(pack)
        elif item.type == "key_index":

            chart = item.key_index.chart

            for each_index in eval(chart.full_key_index):
                if each_index in must_have_content:

                    header_to_return, value_to_return = from_key_index_to_column(each_index, chart)
                    pack = {"title": chart.title, "unit": chart.unit, "header": header_to_return,
                            "content": value_to_return,
                            "score": score, "type": "key_index", "table_id": chart.id, "page": chart.page}
                    if chart.id not in unique_table_id["key_index"].keys():
                        unique_table_id["key_index"][chart.id] = [item.key_index.name]
                        original_structure.append(pack)
                    elif item.key_index.name not in unique_table_id["key_index"][chart.id]:
                        unique_table_id["key_index"][chart.id].append(item.key_index.name)
                        original_structure.append(pack)
            header_to_return, value_to_return = from_key_index_to_column(item.key_index.name, chart)
            if mode == 0:
                pack = {"title": chart.title, "unit": chart.unit, "header": header_to_return,
                        "content": value_to_return,
                        "score": score, "type": "key_index", "table_id": chart.id, "page": chart.page}
            else:
                pack = {"title": chart.title, "unit": chart.unit, "header": chart.full_header,
                        "content": chart.full_value,
                        "score": score, "type": "table", "table_id": chart.id, "page": chart.page}
            if chart.id not in unique_table_id["key_index"].keys():
                unique_table_id["key_index"][chart.id] = [item.key_index.name]
                original_structure.append(pack)
            elif item.key_index.name not in unique_table_id["key_index"][chart.id]:
                unique_table_id["key_index"][chart.id].append(item.key_index.name)
                original_structure.append(pack)
        elif item.type == "value":
            key_index = item.table_value.key_index

            chart = key_index.chart

            for each_index in eval(chart.full_key_index):
                if each_index in must_have_content:
                    header_to_return, value_to_return = from_key_index_to_column(each_index, chart)
                    pack = {"title": chart.title, "unit": chart.unit, "header": header_to_return,
                            "content": value_to_return,
                            "score": score, "type": "key_index", "table_id": chart.id, "page": chart.page}
                    if chart.id not in unique_table_id["key_index"].keys():
                        unique_table_id["key_index"][chart.id] = [key_index.name]
                        original_structure.append(pack)
                    elif key_index.name not in unique_table_id["key_index"][chart.id]:
                        unique_table_id["key_index"][chart.id].append(key_index.name)
                        original_structure.append(pack)
            header_to_return, value_to_return = from_key_index_to_column(key_index.name, chart)
            if mode == 0:
                pack = {"title": chart.title, "unit": chart.unit, "header": header_to_return,
                        "content": value_to_return,
                        "score": score, "type": "key_index", "table_id": chart.id, "page": chart.page}
            else:
                pack = {"title": chart.title, "unit": chart.unit, "header": chart.full_header,
                        "content": chart.full_value,
                        "score": score, "type": "table", "table_id": chart.id, "page": chart.page}
            if chart.id not in unique_table_id["key_index"].keys():
                unique_table_id["key_index"][chart.id] = [key_index.name]
                original_structure.append(pack)
            elif key_index.name not in unique_table_id["key_index"][chart.id]:
                unique_table_id["key_index"][chart.id].append(key_index.name)
                original_structure.append(pack)

        elif item.type == "description":

            chart = item.description.chart

            for each_index in eval(chart.full_key_index):
                if each_index in must_have_content:
                    header_to_return, value_to_return = from_key_index_to_column(each_index, chart)
                    pack = {"title": chart.title, "unit": chart.unit, "header": header_to_return,
                            "content": value_to_return,
                            "score": score, "type": "key_index", "table_id": chart.id, "page": chart.page}
                    original_structure.append(pack)
            header_to_return, value_to_return = from_description_to_column(item.description, chart)
            if mode == 0:
                pack = {"title": chart.title, "unit": chart.unit, "header": header_to_return,
                        "content": value_to_return,
                        "score": score, "type": "description", "table_id": chart.id, "page": chart.page}
            else:
                pack = {"title": chart.title, "unit": chart.unit, "header": chart.full_header,
                        "content": chart.full_value,
                        "score": score, "type": "table", "table_id": chart.id, "page": chart.page}
            original_structure.append(pack)
        # elif item.type == "section":
        #
        #     section = item.section
        #     tables = item.section.tables
        #     tables_to_return = []
        #     if tables:
        #         for table in tables:
        #             pack = {"title": table.title, "unit": table.unit, "header": table.full_header,
        #                     "content": table.full_value,"score": score, "type": "table", "table_id": table.id, "origin": "section"}
        #             tables_to_return.append(pack)
        #
        #     full_sentence = section.full_sentence
        #     if full_sentence:
        #         sentence_collection = full_sentence.split("。")
        #         for sentence in sentence_collection:
        #             if sentence:
        #                 pack = {"sentence": sentence + "。", "score": score}
        #                 original_structure.append(pack)

    return original_structure


def process_and_merge_json_data(data, db):
    def contains_tableID(item):
        return 'table_id' in item and item['table_id']

    grouped_data_with_title = {}
    data_without_title = []

    # 分类处理包含标题和不包含标题的字典
    for item in data:
        if contains_tableID(item):
            table_id = item['table_id']
            if table_id not in grouped_data_with_title:
                grouped_data_with_title[table_id] = []
            grouped_data_with_title[table_id].append(item)
        else:
            data_without_title.append(item)

    # 对包含标题的数据进行处理
    processed_data = []
    for title, group in grouped_data_with_title.items():
        title = group[0]['title']
        table_id = group[0]['table_id']
        unit = group[0]['unit']
        processed_header, processed_rows = construct_and_clean_table_correctly_v3(group, db, table_id)
        if processed_header:
            processed_data.append({
                'title': title,
                'unit': unit,
                'header': processed_header,
                'rows': processed_rows
            })

    # 将不包含标题的数据合并到处理过的数据中
    final_data = processed_data + data_without_title

    return final_data


def construct_and_clean_table_correctly_v3(grouped_data, db, table_id):
    # Function to check if the header is multi-layered
    table_name = grouped_data[0]['title']

    def is_multi_layered(header):
        return len(header) > 1

    def is_list_of_lists(content):

        return isinstance(content[0], list)

    table = db.query(Chart).get(table_id)
    original_header = eval(table.full_header)
    original_rows = eval(table.full_value)
    # Prepare the new header based on the original header
    new_header = original_header[-1]
    new_rows = [["" for _ in new_header] for _ in original_rows]
    exist_whole_table = False
    for record in grouped_data:

        if record['type'] == 'table':
            exist_whole_table = True

        contents = record['content']
        if not contents:
            continue
        if not is_list_of_lists(contents):
            contents = [contents]
        record_header = record['header'][-1]

        # Find the corresponding row in the original rows

        def check_same_row(row1, row2):
            for each_cell in row2:
                if each_cell not in row1:
                    return False
            return True

        for content in contents:
            for i, original_row in enumerate(original_rows):
                if check_same_row(original_row, content):  # Match by the first column (科目)
                    temp_header = copy.deepcopy(new_header)
                    # Find the corresponding columns in the original header
                    for j, cell in enumerate(content):
                        if j < len(record_header) and record_header[j] in temp_header:
                            col_index = temp_header.index(record_header[j])
                            new_rows[i][col_index] = cell  # Update the cell in the new row
                            temp_header[col_index] = ""

    # Remove empty rows

    new_rows = [row for row in new_rows if any(cell != "" for cell in row)]
    # Create a list of tuples (column_index, column_data)
    indexed_columns = list(enumerate(zip(*new_rows)))

    # Filter out empty columns and keep track of their original indices
    filtered_columns_with_indices = [(index, col) for index, col in indexed_columns if any(cell != "" for cell in col)]

    # Check if there are any columns left after filtering
    if filtered_columns_with_indices:
        # Extract the filtered columns and their original indices separately
        filtered_column_indices, filtered_columns = zip(*filtered_columns_with_indices)

        # Reconstruct the rows with the filtered columns
        new_rows = list(zip(*filtered_columns))

        # Adjust the header based on the filtered column indices
        new_header = [new_header[i] for i in filtered_column_indices]
    else:
        # If no columns are left, set rows and header to empty
        new_rows = []
        new_header = []
    if len(new_header) == 1:
        return None, None
    if exist_whole_table:
        new_header = original_header[-1]
        new_rows = original_rows
    if is_multi_layered(original_header):

        header_index = []
        for i, header in enumerate(new_header):
            location = original_header[-1].index(header)
            while location in header_index:
                location = original_header[-1].index(header, location + 1)
            header_index.append(location)

        ##construct new header from the top level
        tempt_new_header = []

        def check_same(text1, text2):
            if "/" in text1:
                text1 = text1.split("/")[-1]
            if text1 == text2:
                return True
            else:
                return False

        for i, header in enumerate(original_header):
            for index, j in enumerate(header_index):
                if i == 0:
                    tempt_new_header.append(header[j])
                else:
                    if check_same(tempt_new_header[index], header[j]):
                        continue
                    else:
                        tempt_new_header[index] = tempt_new_header[index] + "/" + header[j]

        new_header = tempt_new_header
    return new_header, new_rows


def construct_and_clean_table_correctly(grouped_data, db, table_id):
    # Function to check if the header is multi-layered
    def is_multi_layered(header):
        return len(header) > 1

    table = db.query(Chart).get(table_id)
    original_header = eval(table.full_header)
    original_rows = eval(table.full_value)
    if table_id == 159:
        print(grouped_data)
    # Prepare the new header based on the original header
    new_header = original_header if is_multi_layered(original_header) else original_header[0]
    new_rows = [['' for _ in new_header] for _ in original_rows]
    for record in grouped_data:
        if record['type'] == 'table':
            return original_header, original_rows

        content = record['content']
        record_header = record['header'][0]

        # Find the corresponding row in the original rows
        for i, original_row in enumerate(original_rows):
            if original_row[0] == content[0]:  # Match by the first column (科目)
                # Find the corresponding columns in the original header
                for j, cell in enumerate(content):
                    if j < len(record_header) and record_header[j] in new_header:
                        col_index = new_header.index(record_header[j])
                        new_rows[i][col_index] = cell  # Update the cell in the new row

    # Remove empty rows
    new_rows = [row for row in new_rows if any(cell != '' for cell in row)]

    # Remove empty columns if the header is not multi-layered
    if not is_multi_layered(original_header):
        columns = list(zip(*new_rows))
        columns = [col for col in columns if any(cell != '' for cell in col)]
        new_rows = list(zip(*columns))
        new_header = [new_header[i] for i, col in enumerate(columns) if any(cell != '' for cell in col)]

    return new_header, new_rows


def split_dict_with_two_keys(reference_dict, max_tokens, tokenizer):
    """
    Splits the '参考表格' list in the given dictionary into multiple dictionaries,
    keeping the '参考句子' in the first dictionary. Each dictionary's '参考表格'
    portion will have a total GPT token count less than or equal to max_tokens.

    Args:
    reference_dict (dict): The original dictionary to be split.
    max_tokens (int): Maximum allowed GPT tokens per group in '参考表格'.
    tokenizer: The tokenizer used for counting GPT tokens.

    Returns:
    list: A list of dictionaries, each containing a portion of the '参考表格' list.
    """
    subdicts = []
    current_group = []
    current_token_count = 0

    # Count the tokens for '参考句子' and include it in the first subdict
    reference_sentences_token_count = count_gpt_tokens(reference_dict["参考句子"], tokenizer)
    first_group = True

    for element in reference_dict["参考表格"]:
        element_token_count = count_gpt_tokens(element, tokenizer)

        # Check if adding the element would exceed the max token count
        if current_token_count + element_token_count <= max_tokens - (
                reference_sentences_token_count if first_group else 0):
            current_group.append(element)
            current_token_count += element_token_count
        else:
            new_subdict = reference_dict.copy()
            new_subdict["参考表格"] = current_group
            if first_group:
                # Include '参考句子' only in the first group
                first_group = False
            else:
                # Exclude '参考句子' from subsequent groups
                del new_subdict["参考句子"]
            subdicts.append(new_subdict)

            # Start a new group with the current element
            current_group = [element]
            current_token_count = element_token_count

    # Add the last group as a new subdict if it's not empty
    if current_group:
        new_subdict = reference_dict.copy()
        new_subdict["参考表格"] = current_group
        if not first_group:
            del new_subdict["参考句子"]
        subdicts.append(new_subdict)

    return subdicts


def classify_items(input_list):
    categorized_data = {"参考表格": [], "参考句子": []}

    for item in input_list:
        if 'title' in item:
            # 从表格中移除'score'键（如果存在）
            item.pop('score', None)
            categorized_data["参考表格"].append(item)
        else:
            # 从句子中移除'score'键
            sentence = item.get('sentence', '')
            if sentence not in categorized_data["参考句子"]:
                categorized_data["参考句子"].append(sentence)

    return categorized_data


def arrange_section(db, company_id):
    company = db.query(Company).get(company_id)
    sections = company.sections
    section_name_dict = {}
    for section in sections:
        section_name_dict[section.title] = section


def back_to_og_structure(data, db, report_id):
    def find_children_and_content(section, table_id_record_list=[]):
        exist_sentence_id = []
        exist_section_id = []
        dict_to_return = {f"{section.title}": []}
        actual_sentence = False

        content = []
        for index, each_content in enumerate(section.sentences):
            if each_content.content == section.title:
                continue
            content.append(each_content.content)
            exist_sentence_id.append(each_content.id)

        if section.charts.count() > 0:
            original_structure = []
            for table in section.charts:
                if ":" in table.title:
                    true_section_name = table.title.split(":")[1]
                else:
                    true_section_name = table.title

                if "," in true_section_name:
                    true_section_name = true_section_name.split(",")[1]

                location = db.query(Section).filter(
                    and_(Section.title.like(f'%{true_section_name}%'), Section.report_id == report_id)).first()
                if location:
                    all_sentences = location.sentences
                    remaining_content = [sentence.content for sentence in all_sentences]
                else:
                    remaining_content = []

                if len(remaining_content) > 0:
                    pack = {"title": table.title, "unit": table.unit, "header": table.full_header,
                            "content": table.full_value, "remaining_content": remaining_content}
                else:
                    pack = {"title": table.title, "unit": table.unit, "header": table.full_header,
                            "content": table.full_value}
                table_id_record_list.append(table.id)

                original_structure.append(pack)
            content.append(original_structure)

        children_list = section.sub_sections.all()
        while len(children_list) > 0:
            children = children_list.pop(0)
            exist_section_id.append(children.id)
            result_dict_str, sub_exist_sentence_id, sub_exist_section_id, sub_table_id_record_list, actual_sentence = find_children_and_content(
                children,
                table_id_record_list)
            if len(result_dict_str) > 0:
                dict_to_return[list(dict_to_return.keys())[0]].append(result_dict_str)

            exist_sentence_id_set = set(exist_sentence_id)
            exist_section_id_set = set(exist_section_id)
            table_id_record_list_set = set(table_id_record_list)

            exist_sentence_id_set.update(sub_exist_sentence_id)
            exist_section_id_set.update(sub_exist_section_id)
            table_id_record_list_set.update(sub_table_id_record_list)

            exist_sentence_id = list(exist_sentence_id_set)
            exist_section_id = list(exist_section_id_set)
            table_id_record_list = list(table_id_record_list_set)

            # exist_sentence_id += sub_exist_sentence_id
            # exist_section_id += sub_exist_section_id
            # table_id_record_list += sub_table_id_record_list
        if len(content) > 0:
            dict_to_return[list(dict_to_return.keys())[0]].append(content)

        if len(dict_to_return[section.title]) == 0:
            actual_sentence = True
            dict_to_return = section.title
        return dict_to_return, exist_sentence_id, exist_section_id, table_id_record_list, actual_sentence

    section_level_dict = {}
    normal_sentence_id_list = []
    total_representation = []
    table_id_record_list = []
    reference_tracker = {}
    cite_index = 1
    # 将all_vectors通过level分类成 None的，level1的，level2的，出现一个新的level就记录
    for each in data:
        if type(each["vector"]) == InstrumentedList:
            true_vector = each["vector"][0]
        else:
            true_vector = each["vector"]

        if true_vector.sentence.is_title == '0':
            normal_sentence_id_list.append({"content": each["sentence"], "object": each["og_sentence"]})
            continue
        level = int(true_vector.level)
        if level not in section_level_dict.keys():
            section_level_dict[level] = []
        if each["og_sentence"].section not in section_level_dict[level]:
            section_level_dict[level].append(each["og_sentence"].section)

    level_index = sorted(section_level_dict.keys())
    cite_index = 1
    for level in level_index:
        sections = section_level_dict[level]
        for section in sections:

            result_dict_str, sub_exist_sentence_id, sub_exist_section_id, table_id_record_list, actual_sentence = find_children_and_content(
                section)

            total_representation.append({"core": result_dict_str, "page": section.page})
            reference_tracker[f"参考{cite_index}"] = {"id": section.id, "type": "section"}
            cite_index += 1
            # 删掉出现过的句子
            normal_sentence_id_list = [x for x in normal_sentence_id_list if
                                       x["object"].id not in sub_exist_sentence_id]
            # 删掉出现过的章节
            for del_level in range(level + 1, len(level_index) + 1):
                if del_level not in section_level_dict.keys():
                    continue
                section_level_dict[del_level] = [x for x in section_level_dict[del_level] if
                                                 x.id not in sub_exist_section_id]

    # 将剩下的句子加入
    for sentence in normal_sentence_id_list:
        total_representation.append({"core": sentence["content"], "page": sentence["object"].page})
        reference_tracker[f"参考{cite_index}"] = {"id": sentence["object"].id, "type": "sentence"}
        cite_index += 1

    return total_representation, table_id_record_list, reference_tracker, cite_index


def chat_bot_process_and_merge_json_data(data, db, report_id):
    def contains_tableID(item):
        return 'table_id' in item and item['table_id']

    grouped_data_with_title = {}
    data_without_title = []

    # 分类处理包含标题和不包含标题的字典
    for item in data:
        if contains_tableID(item):
            table_id = item['table_id']
            if table_id not in grouped_data_with_title:
                grouped_data_with_title[table_id] = []
            grouped_data_with_title[table_id].append(item)
        else:
            data_without_title.append(item)

    structure_result, table_id_record_list, reference_tracker, cite_index = back_to_og_structure(data_without_title, db,
                                                                                                 report_id)
    new_grouped_data_with_title = {}
    # 将grouped_data_with_title里面的table_id_record_list去重
    for table_id in grouped_data_with_title.keys():
        if table_id not in table_id_record_list:
            new_grouped_data_with_title[table_id] = grouped_data_with_title[table_id]

    # 对table做处理
    processed_data = []
    for title, group in new_grouped_data_with_title.items():
        title = group[0]['title']
        table_id = group[0]['table_id']
        unit = group[0]['unit']
        processed_header, processed_rows = construct_and_clean_table_correctly_v3(group, db, table_id)
        if processed_header:
            if "remaining_content" in group[0].keys() and len(group[0]["remaining_content"]) > 0:
                remaining_content = group[0]["remaining_content"]
                structure_result.append({"core": {
                    'title': title,
                    'unit': unit,
                    'header': processed_header,
                    'rows': processed_rows,
                    'remaining_content': remaining_content
                }, "page": group[0]["page"]})
            else:
                structure_result.append({"core": {
                    'title': title,
                    'unit': unit,
                    'header': processed_header,
                    'rows': processed_rows
                }, "page": group[0]["page"]})
        reference_tracker[f"参考{cite_index}"] = {"id": table_id, "type": "table"}
        cite_index += 1

    return structure_result, reference_tracker


def find_table_source_only(sentence_vectors, db, company_id, report_id, resource_keywords, min_similarity=0.58):
    vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
        and_(Vector.report_id == report_id, Vector.is_table_title == 1)).all()

    if len(vectors) > 0:
        total_search_result = search_relevant(sentence_vectors, vectors, min_similarity=min_similarity)
    else:
        total_search_result = []
    original_structure = parse_search_result(total_search_result, db, report_id)
    # 整理candidate_table_list
    target_table_list = []
    for each_dict in original_structure:
        if each_dict["title"] not in target_table_list:
            target_table_list.append(each_dict["title"])
    if len(original_structure) == 0:
        del vectors
        return original_structure, {}, 0
    usage, final_table_candidate_list = find_table_target_bot(target_table_list, resource_keywords)
    final_table = []
    for each_title in final_table_candidate_list:
        for each_dict in original_structure:
            if each_dict["title"] == each_title:
                final_table.append(each_dict)

    del vectors
    original_structure, reference_tracker = chat_bot_process_and_merge_json_data(final_table, db, report_id)
    # reference = classify_items(original_structure)

    return original_structure, reference_tracker, usage


def find_section_source_only(sentence_vectors, db, company_id, report_id, section_keywords, min_similarity=0.58):
    usage = 0
    total_search_result = []
    # 找出section里面带有"基本信息的section" 硬匹配
    target_sections = []
    for section_keyword in eval(section_keywords.replace("，", ",")):
        sub_target_sections = db.query(Section).filter(
            and_(Section.report_id == report_id, Section.title.like(f'%{section_keyword}%'))).all()
        target_sections += sub_target_sections
    for target_section in target_sections:
        title_sentence = target_section.sentences.filter(Sentences.is_title == "1").first()
        if title_sentence and len(title_sentence.vector) > 0:
            total_search_result.append([title_sentence.vector[0], 0.99])

    # 如果硬匹配为空，则进行软匹配
    if len(total_search_result) == 0:
        target_sections = db.query(Section).filter(Section.report_id == report_id).all()
        all_sections = db.query(Section).filter(Section.report_id == report_id).all()
        all_section_titles_vectors = []
        for section in all_sections:
            title_sentence = section.sentences.filter(Sentences.is_title == "1").first()
            if title_sentence and len(title_sentence.vector) > 0:
                all_section_titles_vectors.append(title_sentence.vector[0])

        if len(all_section_titles_vectors) > 0:
            total_search_result = search_relevant(sentence_vectors, all_section_titles_vectors,
                                                  min_similarity=min_similarity)
        else:
            total_search_result = []

        original_structure = parse_search_result(total_search_result, db, report_id)
        # 整理target_section_list
        target_section_list = []
        for each_dict in original_structure:
            if each_dict["sentence"] not in target_section_list:
                target_section_list.append(each_dict["sentence"])
        if len(original_structure) == 0:
            del all_section_titles_vectors
            return original_structure, {}, 0
        usage, final_section_candidate_list = find_section_target_bot(target_section_list, section_keywords)
        final_section = []
        for each_title in final_section_candidate_list:
            for each_dict in original_structure:
                if each_dict["sentence"] == each_title:
                    final_section.append(each_dict)

        del all_section_titles_vectors
        original_structure, reference_tracker = chat_bot_process_and_merge_json_data(final_section, db, report_id)
    original_structure = parse_search_result(total_search_result, db, report_id)
    structure_to_process = []
    for each_element in original_structure:
        if 'vector' in each_element.keys():
            if each_element['vector'] and each_element['vector'] != []:
                structure_to_process.append(each_element)
    original_structure, reference_tracker = chat_bot_process_and_merge_json_data(structure_to_process, db, report_id)
    # reference = classify_items(original_structure)

    return original_structure, reference_tracker, usage


def find_target(sentence_vectors, db, ref_text_origin, company_id, report_id, min_similarity, vectors=None,
                table_only=0, without_table=0, too_detail=0, info_vector_tracker=None):
    total_search_result = []
    if info_vector_tracker:
        for vector_id, score in info_vector_tracker:
            sub_vector = db.query(Vector).filter(Vector.id == vector_id).first()
            if sub_vector:
                total_search_result.append([sub_vector, score])
    else:
        if vectors is None:

            if table_only == 1:
                # vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
                #     and_(Vector.report_id == report_id, Vector.belongs_to_table == 1, Vector.type.isnot(None))).all()
                vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
                    and_(Vector.report_id == report_id, Vector.belongs_to_table == 1)).all()
            elif without_table == 1:
                # vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
                #     and_(Vector.report_id == report_id, Vector.belongs_to_table.is_(None), Vector.type.isnot(None))).all()
                vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
                    and_(Vector.report_id == report_id, Vector.belongs_to_table.is_(None))).all()
            else:
                # vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
                #     and_(Vector.report_id == report_id, Vector.type.isnot(None))).all()
                vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
                    Vector.report_id == report_id).all()

        if len(vectors) > 0 and len(sentence_vectors) > 0:
            total_search_result = search_relevant(sentence_vectors, vectors, min_similarity=min_similarity)
        else:
            total_search_result = []
        # 找section
        specific_section = db.query(Section).filter(
            and_(Section.report_id == report_id, Section.title == "前序")).first()
        if specific_section:
            title_sentence = specific_section.sentences.filter(Sentences.is_title == "1").first()
            if title_sentence and len(title_sentence.vector) > 0:
                total_search_result.append([title_sentence.vector[0], 0.99])

    total_search_result += ref_text_origin
    original_structure = parse_search_result(total_search_result, db, report_id)

    del vectors
    structure_to_process = []
    for each_element in original_structure:
        if 'vector' in each_element.keys():
            if each_element['vector'] and each_element['vector'] != []:
                structure_to_process.append(each_element)
    original_structure, reference_tracker = chat_bot_process_and_merge_json_data(structure_to_process, db, report_id)
    # reference = classify_items(original_structure)
    return original_structure, reference_tracker


def dynamic_info_matching(dynamic_keywords, db, ref_text_origin, company_id, report_id, min_similarity, vectors=None,table_only=0, without_table=0):
    total_search_result = []

    vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
        Vector.report_id == report_id).all()
    dynamic_keywords = str(dynamic_keywords).replace("，", ",").split(",")
    dynamic_keywords_vectors = get_cluster_embeddings(dynamic_keywords)
    if len(vectors) > 0 and len(dynamic_keywords_vectors) > 0:
        total_search_result = search_relevant(dynamic_keywords_vectors, vectors, min_similarity=min_similarity)
    else:
        total_search_result = []
    original_structure = parse_search_result(total_search_result, db, report_id)
    # 反推整个段落
    re_parse_list = []
    for each_item in original_structure:
        if "og_sentence" in each_item.keys():
            sentence = each_item["og_sentence"]
            if sentence:
                section = sentence.section
                if section:
                    title_sentence = section.sentences.filter(Sentences.is_title == "1").first()
                    if title_sentence  and len(title_sentence.vector) > 0:
                        re_parse_list.append([title_sentence.vector[0], 0.99])
    final_structure = parse_search_result(re_parse_list, db, report_id)
    original_structure += final_structure

    del vectors
    structure_to_process = []
    for each_element in original_structure:
        if 'vector' in each_element.keys():
            if each_element['vector'] and each_element['vector'] != []:
                structure_to_process.append(each_element)
    original_structure, reference_tracker = chat_bot_process_and_merge_json_data(structure_to_process, db, report_id)
    # reference = classify_items(original_structure)
    return original_structure, reference_tracker


def find_text(input, db, company_id, report_id):
    # total_search_result = []
    search_terms = [key for key, value in input.items() if value == 2]
    structured_results = []
    if len(search_terms) > 0:
        or_conditions = or_(*[Sentences.content.like(f'%{term}%') for term in search_terms])

        results = db.query(Sentences).filter(
            and_(Sentences.report_id == report_id, or_conditions)).all()
        # Structure to hold the results and their counts

        # Analyze each fetched row for term count and unique term matches
        for sentence in results:
            term_counts = {term: sentence.content.count(term) for term in search_terms}
            unique_term_count = sum(1 for count in term_counts.values() if count > 0)
            total_term_count = sum(term_counts.values())
            structured_results.append({
                'content': sentence.content,
                'unique_term_count': unique_term_count,
                'total_term_count': total_term_count,
                'term_counts': term_counts,
                'vector': sentence.vector
            })

        # Sort the results first by the number of unique terms matched, then by the total term count
        structured_results.sort(key=lambda x: (-x['unique_term_count'], -x['total_term_count']))

        if len(structured_results) > 5:
            del results
            to_return = []
            for each_target in structured_results[:5]:
                if each_target["vector"] and len(each_target["vector"]) > 0:
                    to_return.append([each_target["vector"][0], 0.99])

            return to_return
        else:
            search_terms_new = [key for key, value in input.items() if value == 1]
            if len(search_terms_new) > 0:
                or_conditions_new = or_(*[Sentences.content.like(f'%{term}%') for term in search_terms_new])
                results_new = db.query(Sentences).filter(
                    and_(Sentences.report_id == report_id,
                         or_conditions_new)).all()

                structured_results_new = []

                # Analyze each fetched row for term count and unique term matches
                for sentence in results_new:
                    term_counts = {term: sentence.content.count(term) for term in search_terms_new}
                    unique_term_count = sum(1 for count in term_counts.values() if count > 0)
                    total_term_count = sum(term_counts.values())
                    structured_results_new.append({
                        'content': sentence.content,
                        'unique_term_count': unique_term_count,
                        'total_term_count': total_term_count,
                        'term_counts': term_counts,
                        "vector": sentence.vector
                    })

                # Sort the results first by the number of unique terms matched, then by the total term count
                structured_results_new.sort(key=lambda x: (-x['unique_term_count'], -x['total_term_count']))
                for i in range(5 - len(structured_results)):
                    if len(structured_results_new) - 1 >= i:
                        structured_results.append(structured_results_new[i])
                del results
                del results_new
                to_return = []
                for each_target in structured_results:
                    if each_target["vector"] and len(each_target["vector"]) > 0:
                        to_return.append([each_target["vector"][0], 0.99])
                return to_return
            else:
                del results

                return []
    else:
        search_terms_new = [key for key, value in input.items() if value == 1]
        if len(search_terms_new) > 0:
            or_conditions_new = or_(*[Sentences.content.like(f'%{term}%') for term in search_terms_new])
            results_new = db.query(Sentences).filter(
                and_(Sentences.report_id == report_id, or_conditions_new)).all()

            structured_results_new = []

            # Analyze each fetched row for term count and unique term matches
            for sentence in results_new:
                term_counts = {term: sentence.content.count(term) for term in search_terms_new}
                unique_term_count = sum(1 for count in term_counts.values() if count > 0)
                total_term_count = sum(term_counts.values())
                structured_results_new.append({
                    'content': sentence.content,
                    'unique_term_count': unique_term_count,
                    'total_term_count': total_term_count,
                    'term_counts': term_counts,
                    'vector': sentence.vector
                })

            # Sort the results first by the number of unique terms matched, then by the total term count
            structured_results_new.sort(key=lambda x: (-x['unique_term_count'], -x['total_term_count']))
            for i in range(5 - len(structured_results)):
                if len(structured_results_new) - 1 >= i:
                    structured_results.append(structured_results_new[i])
            del results_new
            to_return = []
            for each_target in structured_results:
                if each_target["vector"] and len(each_target["vector"]) > 0:
                    to_return.append([each_target["vector"][0], 0.99])
            return to_return
        else:
            return []


def get_reference(db, company_id, search_parameters):
    weight_keywords = search_parameters["weight_keywords"]
    resource_keywords = search_parameters["resource_keywords"]
    resource_keywords_vectors = search_parameters["resource_keywords_vectors"]
    similarity_keywords_vectors = search_parameters["similarity_keywords_vectors"]
    table_only = search_parameters["table_only"]
    without_table = search_parameters["without_table"]
    min_similarity = search_parameters["min_similarity"]
    report_id = search_parameters["report_id"]
    info_vector_tracker = search_parameters["info_vector_tracker"]
    section_keywords = search_parameters["section_keywords"]
    section_keywords_vectors = search_parameters["section_keywords_vectors"]
    dynamic_keywords = str(search_parameters['dynamic_keywords'])
    info_vector_tracker = search_parameters["info_vector_tracker"]
    usage = 0
    table_original_structure = {}
    section_original_structure = {}
    dynamic_keywords_original_structure = {}
    if min_similarity:
        min_similarity = float(min_similarity)
    else:
        min_similarity = 0.6

    # 专门搜索table的算法
    if resource_keywords and len(eval(resource_keywords.replace("，", ","))) > 0:
        table_original_structure, table_reference_tracker, find_table_usage = find_table_source_only(
            resource_keywords_vectors, db,
            company_id, report_id,
            resource_keywords)
        usage += find_table_usage
    if section_keywords and len(eval(section_keywords.replace("，", ","))) > 0:
        section_original_structure, section_reference_tracker, find_section_usage = find_section_source_only(
            section_keywords_vectors, db,
            company_id, report_id, section_keywords)
        usage += find_section_usage

    if dynamic_keywords and len(dynamic_keywords.replace("，", ",").split(",")) > 0:
        dynamic_keywords_original_structure, dynamic_keywords_reference_tracker = dynamic_info_matching(
            dynamic_keywords, db, None, company_id, report_id, min_similarity, vectors=None, table_only=0,
            without_table=0)

    ref_text_origin = []
    weight_keywords_dict = str_to_json(weight_keywords)
    if weight_keywords_dict and type(weight_keywords_dict) == dict:
        ref_text_origin = find_text(weight_keywords_dict, db, company_id, report_id)

    original_structure, reference_tracker = find_target(similarity_keywords_vectors, db, ref_text_origin, company_id,
                                                        report_id, table_only=table_only, without_table=without_table,
                                                        min_similarity=min_similarity,
                                                        info_vector_tracker=info_vector_tracker)
    # 重新排序
    sorted_original_structure = sorted(original_structure, key=lambda x: x['page'])
    final_structure = {}
    text_index_record = 0
    for index, each_reference in enumerate(sorted_original_structure):
        final_structure[f"参考{index}"] = each_reference["core"]
        text_index_record = index + 1
    # 重新排序
    sorted_table_original_structure = sorted(table_original_structure, key=lambda x: x['page'])
    final_table_structure = []
    for index, each_reference in enumerate(sorted_table_original_structure):
        final_table_structure.append({f"参考{index}": each_reference["core"]})
    # 重新排序
    sorted_section_original_structure = sorted(section_original_structure, key=lambda x: x['page'])

    for index, each_reference in enumerate(sorted_section_original_structure):
        final_structure[f"参考{text_index_record}"] = each_reference["core"]
        text_index_record += 1

    # 重新排序
    sorted_dynamic_keywords_original_structure = sorted(dynamic_keywords_original_structure, key=lambda x: x['page'])

    for index, each_reference in enumerate(sorted_dynamic_keywords_original_structure):
        final_structure[f"参考{text_index_record}"] = each_reference["core"]
        text_index_record += 1

    return final_structure, final_table_structure, usage


def str_to_json(str_data):
    import json

    # Replace the single quotes with double quotes for valid JSON format
    str_data = str_data.replace("'", '"')

    # Convert the string to a json object
    try:
        json_data = json.loads(str_data)
        if len(json_data) == 1 and type(json_data[list(json_data.keys())[0]]) == dict:
            return json_data[list(json_data.keys())[0]]
        return json_data
    except json.JSONDecodeError as e:
        return None


def group_searching_within_report(report_task_dict):
    db, engine = create_db_session()
    try:
        report_name = report_task_dict["report_name"]
        indicators = report_task_dict["indicators"]
        level = report_task_dict["level"]

        report = db.query(Reports).filter(Reports.report_name == report_name).first()
        if not report or report.in_db == 0:
            return False,[]

        task_list = []

        report_id = report.id
        all_vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
            Vector.report_id == report_id).all()

        all_vectors_without_table = db.query(Vector).options(undefer(Vector.vector)).filter(
            and_(Vector.report_id == report_id, Vector.belongs_to_table.is_(None))).all()

        all_vectors_table_only = db.query(Vector).options(undefer(Vector.vector)).filter(
            and_(Vector.report_id == report_id, Vector.belongs_to_table == 1)).all()

        def limited_find_target(sentence_vectors, min_similarity=0.55, vectors=None):

            if len(vectors) > 0:
                total_search_result = search_relevant(sentence_vectors, vectors, min_similarity=min_similarity)
            else:
                total_search_result = []

            vector_tracker = []
            for each_vector, score in total_search_result:
                vector_tracker.append((each_vector.id, score))
            return vector_tracker

        for indicator_code in indicators:
            try:
                indicator = db.query(Indicators).filter(Indicators.code == indicator_code).first()
                similarity_keywords_vectors = []
                for each_vector in indicator.similarity_keywords_vectors:
                    similarity_keywords_vectors.append(pickle.loads(each_vector.vector))
                table_only = indicator.table_only
                without_table = indicator.without_table
                min_similarity = indicator.min_similarity
                if table_only == 1:
                    vector = all_vectors_table_only
                elif without_table == 1:
                    vector = all_vectors_without_table
                else:
                    vector = all_vectors
                if not min_similarity:
                    min_similarity = 0.55
                vector_tracker = limited_find_target(similarity_keywords_vectors, min_similarity, vectors=vector)
                task_list.append({"report_name": report_name, "indicator_code": indicator_code,
                                  "info_vector_tracker": vector_tracker,
                                  "company_code": report_task_dict["company_code"], "report_id": report_id,
                                  "year": report_task_dict["year"], "level": level})
            except Exception as e:
                print(e)
                continue
        return True,task_list
    except Exception as e:
        print(e)
        return False, []
    finally:
        db.close()
        engine.dispose()

def group_searching_within_report_v2(report_task_dict):
    db, engine = create_db_session()
    try:
        report_name = report_task_dict["report_name"]
        indicators = report_task_dict["indicators"]
        person = report_task_dict["person"]
        company_code = report_task_dict["company_code"]
        year = str(report_task_dict["year"])

       #不添加已经做过了的
        results = db.query(IndicatorsResults).filter(
            and_(IndicatorsResults.company_code == company_code,
                 IndicatorsResults.person == person,
                 IndicatorsResults.year == year,
                 IndicatorsResults.code.in_(indicators))
        ).all()


        results_dict = {(r.code, r.company_code, r.person, r.year): r for r in results}

        unprocessed_indicators = []
        for indicator_code in indicators:
            result = results_dict.get((indicator_code, company_code, person, year))
            if result and result.value != "信息不足" and result.assumption != 1:
                continue
            unprocessed_indicators.append(indicator_code)

        if len(unprocessed_indicators) == 0:
            return True, []

        report = db.query(Reports).filter(Reports.report_name == report_name).first()
        if not report or report.in_db == 0:
            fail_message = f"{report_name},error:没有reports"
            new_fail_report = FailReports(message=fail_message, step=3)
            db.add(new_fail_report)
            db.commit()
            return False, fail_message

        task_list = []

        report_id = report.id
        all_vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
            Vector.report_id == report_id).all()

        all_vectors_without_table = db.query(Vector).options(undefer(Vector.vector)).filter(
            and_(Vector.report_id == report_id, Vector.belongs_to_table.is_(None))).all()

        all_vectors_table_only = db.query(Vector).options(undefer(Vector.vector)).filter(
            and_(Vector.report_id == report_id, Vector.belongs_to_table == 1)).all()

        def limited_find_target(sentence_vectors, min_similarity=0.55, vectors=None):

            if len(vectors) > 0:
                total_search_result = search_relevant(sentence_vectors, vectors, min_similarity=min_similarity)
            else:
                total_search_result = []

            vector_tracker = []
            for each_vector, score in total_search_result:
                vector_tracker.append((each_vector.id, score))
            return vector_tracker

        for indicator_code in unprocessed_indicators:
            try:

                indicator = db.query(Indicators).filter(Indicators.code == indicator_code).first()

                similarity_keywords_vectors = []
                for each_vector in indicator.indicator_vectors:
                    if each_vector.is_similarity_keywords_vector == 1:
                        similarity_keywords_vectors.append(pickle.loads(each_vector.vector))
                table_only = indicator.table_only
                without_table = indicator.without_table
                min_similarity = indicator.min_similarity
                if table_only == 1:
                    vector = all_vectors_table_only
                elif without_table == 1:
                    vector = all_vectors_without_table
                else:
                    vector = all_vectors
                if not min_similarity:
                    min_similarity = 0.6
                vector_tracker = limited_find_target(similarity_keywords_vectors, min_similarity, vectors=vector)


                #现在是写进数据库
                mission_json = json.dumps({"report_name": report_name, "indicator_code": indicator_code,
                                  "info_vector_tracker": vector_tracker,
                                  "company_code": report_task_dict["company_code"], "report_id": report_id,
                                  "year": report_task_dict["year"],'pre_condition_indicator':indicator.pre_condition_indicator})

                new_mission = Missions( mission_json = mission_json,report_name =report_name,execute_level = indicator.execute_level,execute_section = indicator.execute_section,pre_condition_indicator = indicator.pre_condition_indicator,person = person)
                db.add(new_mission)
                db.commit()
            except Exception as e:
                print(e)
                continue
        return True,task_list
    except Exception as e:
        print(e)
        return False, f"{report_name},error:{e}"
    finally:
        db.close()
        engine.dispose()


def group_searching_within_report_v3_year_reports(report_task_dict):
    db, engine = create_db_session()
    try:
        report_name = report_task_dict["report_name"]
        indicators = report_task_dict["indicators"]
        person = report_task_dict["person"]
        company_code = report_task_dict["company_code"]
        year = str(report_task_dict["year"])

       #不添加已经做过了的
        results = db.query(IndicatorsResults).filter(
            and_(IndicatorsResults.company_code == company_code,
                 IndicatorsResults.person == person,
                 IndicatorsResults.year == year,
                 IndicatorsResults.code.in_(indicators))
        ).all()


        results_dict = {(r.code, r.company_code, r.person, r.year): r for r in results}

        unprocessed_indicators = []
        for indicator_code in indicators:
            result = results_dict.get((indicator_code, company_code, person, year))
            if result and result.value != "信息不足" and result.assumption != 1:
                continue
            unprocessed_indicators.append(indicator_code)

        if len(unprocessed_indicators) == 0:
            return True, []

        report = db.query(Reports).filter(Reports.report_name == report_name).first()
        if not report or report.in_db == 0:
            fail_message = f"{report_name},error:没有reports"
            new_fail_report = FailReports(message=fail_message, step=3)
            db.add(new_fail_report)
            db.commit()
            return False, fail_message

        task_list = []

        report_id = report.id
        all_vectors = db.query(Vector).options(undefer(Vector.vector)).filter(
            Vector.report_id == report_id).all()

        all_vectors_without_table = db.query(Vector).options(undefer(Vector.vector)).filter(
            and_(Vector.report_id == report_id, Vector.belongs_to_table.is_(None))).all()

        all_vectors_table_only = db.query(Vector).options(undefer(Vector.vector)).filter(
            and_(Vector.report_id == report_id, Vector.belongs_to_table == 1)).all()

        def limited_find_target(sentence_vectors, min_similarity=0.55, vectors=None):

            if len(vectors) > 0:
                total_search_result = search_relevant(sentence_vectors, vectors, min_similarity=min_similarity)
            else:
                total_search_result = []

            vector_tracker = []
            for each_vector, score in total_search_result:
                vector_tracker.append((each_vector.id, score))
            return vector_tracker

        for indicator_code in unprocessed_indicators:
            try:

                indicator = db.query(Indicators).filter(Indicators.code == indicator_code).first()

                similarity_keywords_vectors = []
                for each_vector in indicator.indicator_vectors:
                    if each_vector.is_similarity_keywords_vector == 1:
                        similarity_keywords_vectors.append(pickle.loads(each_vector.vector))
                table_only = indicator.table_only
                without_table = indicator.without_table
                min_similarity = indicator.min_similarity
                if table_only == 1:
                    vector = all_vectors_table_only
                elif without_table == 1:
                    vector = all_vectors_without_table
                else:
                    vector = all_vectors
                if not min_similarity:
                    min_similarity = 0.6
                vector_tracker = limited_find_target(similarity_keywords_vectors, min_similarity, vectors=vector)


                #现在是写进数据库
                mission_json = json.dumps({"report_name": report_name, "indicator_code": indicator_code,
                                  "info_vector_tracker": vector_tracker,
                                  "company_code": report_task_dict["company_code"], "report_id": report_id,
                                  "year": report_task_dict["year"],'pre_condition_indicator':indicator.pre_condition_indicator})

                new_mission = Missions( mission_json = mission_json,report_name =report_name,execute_level = indicator.execute_level,execute_section = indicator.execute_section,pre_condition_indicator = indicator.pre_condition_indicator,person = person)
                db.add(new_mission)
                db.commit()
            except Exception as e:
                print(e)
                continue
        return True,task_list
    except Exception as e:
        print(e)
        return False, f"{report_name},error:{e}"
    finally:
        db.close()
        engine.dispose()

def limited_find_target(sentence_vectors, min_similarity=0.55, vectors=None):

            if len(vectors) > 0:
                total_search_result = search_relevant(sentence_vectors, vectors, min_similarity=min_similarity)
            else:
                total_search_result = []

            vector_tracker = []
            for each_vector, score in total_search_result:
                vector_tracker.append((each_vector.id, score))
            return vector_tracker

def group_searching_within_report_v4_year_reports(report_task_dict, db_session=None):
    close_session = False
    if db_session is None:
        db, engine = create_db_session()
        close_session = True
    else:
        db = db_session
    
    try:
        # 提取基本信息
        report_name = report_task_dict["report_name"]
        indicators = report_task_dict["indicators"]
        person = report_task_dict["person"]
        company_code = report_task_dict["company_code"]
        year = str(report_task_dict["year"])
        
        # 批量查询已处理的指标
        results_dict = {
            (r.code, r.company_code, r.person, r.year): r 
            for r in db.query(IndicatorsResults).filter(
                and_(IndicatorsResults.company_code == company_code,
                     IndicatorsResults.person == person,
                     IndicatorsResults.year == year,
                     IndicatorsResults.code.in_(indicators))
            ).all()
        }
        
        # 筛选未处理的指标
        unprocessed_indicators = [
            code for code in indicators 
            if not (results_dict.get((code, company_code, person, year)) and 
                   results_dict.get((code, company_code, person, year)).value != "信息不足" and 
                   results_dict.get((code, company_code, person, year)).assumption != 1)
        ]
        
        if not unprocessed_indicators:
            return True, []
            
        # 检查报告是否存在
        report = db.query(Reports).filter(Reports.report_name == report_name).first()
        if not report or report.in_db == 0:
            fail_message = f"{report_name},error:没有reports"
            new_fail_report = FailReports(message=fail_message, step=3)
            db.add(new_fail_report)
            db.commit()
            return False, fail_message
            
        report_id = report.id
        
        # 预加载所有需要的指标信息
        indicators_data = {
            ind.code: ind for ind in 
            db.query(Indicators).filter(Indicators.code.in_(unprocessed_indicators)).all()
        }
        
        # 预加载所有指标的向量
        indicator_vectors = {}
        for ind_code, indicator in indicators_data.items():
            vectors = []
            for vector in indicator.indicator_vectors:
                if vector.is_similarity_keywords_vector == 1:
                    vectors.append(pickle.loads(vector.vector))
            indicator_vectors[ind_code] = vectors
        
        # 分析指标需求，决定加载哪些向量
        need_table_only = any(ind.table_only == 1 for ind in indicators_data.values() if ind)
        need_without_table = any(ind.without_table == 1 for ind in indicators_data.values() if ind)
        
        # 根据需求有选择地加载向量
        if need_table_only and need_without_table:
            all_vectors = db.query(Vector).options(undefer(Vector.vector)).filter(Vector.report_id == report_id).all()
            all_vectors_table_only = [v for v in all_vectors if v.belongs_to_table == 1]
            all_vectors_without_table = [v for v in all_vectors if v.belongs_to_table is None]
        elif need_table_only:
            all_vectors_table_only = db.query(Vector).options(undefer(Vector.vector)).filter(
                and_(Vector.report_id == report_id, Vector.belongs_to_table == 1)).all()
            all_vectors = all_vectors_table_only
            all_vectors_without_table = []
        elif need_without_table:
            all_vectors_without_table = db.query(Vector).options(undefer(Vector.vector)).filter(
                and_(Vector.report_id == report_id, Vector.belongs_to_table.is_(None))).all()
            all_vectors = all_vectors_without_table
            all_vectors_table_only = []
        else:
            all_vectors = db.query(Vector).options(undefer(Vector.vector)).filter(Vector.report_id == report_id).all()
            all_vectors_table_only = [v for v in all_vectors if v.belongs_to_table == 1]
            all_vectors_without_table = [v for v in all_vectors if v.belongs_to_table is None]
        
        # 准备批量添加的任务
        missions_to_add = []
        
        # 处理每个未处理的指标
        for indicator_code in unprocessed_indicators:
            try:
                indicator = indicators_data.get(indicator_code)
                if not indicator:
                    continue
                    
                similarity_keywords_vectors = indicator_vectors.get(indicator_code, [])
                
                # 选择合适的向量集
                if indicator.table_only == 1:
                    vector = all_vectors_table_only
                elif indicator.without_table == 1:
                    vector = all_vectors_without_table
                else:
                    vector = all_vectors
                    
                min_similarity = indicator.min_similarity or 0.6
                
                # 计算相似度
                vector_tracker = limited_find_target(similarity_keywords_vectors, min_similarity, vectors=vector)
                
                # 创建任务
                mission_json = json.dumps({
                    "report_name": report_name, 
                    "indicator_code": indicator_code,
                    "info_vector_tracker": vector_tracker,
                    "company_code": company_code, 
                    "report_id": report_id,
                    "year": year,
                    'pre_condition_indicator': indicator.pre_condition_indicator
                })
                
                new_mission = Missions(
                    mission_json=mission_json,
                    report_name=report_name,
                    execute_level=indicator.execute_level,
                    execute_section=indicator.execute_section,
                    pre_condition_indicator=indicator.pre_condition_indicator,
                    person=person
                )
                
                missions_to_add.append(new_mission)
                
            except Exception as e:
                logger.debug(f"处理指标 {indicator_code} 时出错: {e}")
                continue
                
        # 批量添加并提交
        if missions_to_add:
            db.add_all(missions_to_add)
            db.commit()
            
        return True, []
        
    except Exception as e:
        logger.error(f"处理报告 {report_task_dict.get('report_name')} 时出错: {e}")
        return False, f"{report_task_dict.get('report_name')},error:{e}"
        
    finally:
        if close_session:
            db.close()
            engine.dispose()