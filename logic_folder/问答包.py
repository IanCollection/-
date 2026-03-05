import json
import os

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

from 模块工具.openai相关工具 import count_gpt_tokens
from logic_folder.检索包 import get_reference
from 模块工具.智能体仓库 import interior_data_collection_messages, indicator_collection_seek_final_answer, qwen_interior_data_collection_top_asking
from ian_evolution.client_manager import qwen_client
load_dotenv()


def split_dict_by_dynamic_keys(reference_dict, max_tokens, tokenizer):
    subdicts = []
    current_group = {}
    current_token_count = 0
    index = 0

    while True:
        key = f"参考{index}"
        if key in reference_dict:
            value = reference_dict[key]
            value_token_count = count_gpt_tokens(str(value), tokenizer)

            if current_token_count + value_token_count <= max_tokens:
                current_group[key] = value
                current_token_count += value_token_count
            else:
                if current_group:
                    subdicts.append(current_group)
                current_group = {key: value}
                current_token_count = value_token_count
            index += 1
        else:
            break

    # Add the last group if it's not empty
    if current_group:
        subdicts.append(current_group)

    return subdicts


def asking(db, sub_indicators_info, ask_parameters, search_parameters, sub_mode=1):
    company_id = ask_parameters["company_id"]
    indicator_type = ask_parameters["indicator_type"]
    code = ask_parameters['indicator_code']


    reference, table_reference, find_table_usage = get_reference(db, company_id, search_parameters)
    db.close()
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    reference_length = count_gpt_tokens(str(reference), tokenizer)
    split_references = split_dict_by_dynamic_keys(reference, 7000, tokenizer)
    og_answer_collection = []
    cost = 0
    cost += find_table_usage
    used_tool = False
    found_answer = False
    question_line = ''
    message_list = []

    # print(f'code:{code}')
    if code == '14.1':
        # print("条件成立")
        for reference_trunk in [table_reference, split_references]:
            for sub_reference in reference_trunk:

                if ask_parameters["mission_type"] == "计算型":
                    message, question_line = interior_data_collection_messages(sub_mode, sub_reference, sub_indicators_info,
                                                                               ask_parameters)

                    temp_record, temp_used_tool, found_answer, temp_cost = qwen_interior_data_collection_top_asking(
                        qwen_client, message, indicator_type, used_tool, sub_indicators_info=None, ai='qwen')
                else:
                    message, question_line = interior_data_collection_messages(sub_mode, sub_reference, sub_indicators_info,
                                                                               ask_parameters, ai="qwen")
                    temp_record, temp_used_tool, found_answer, temp_cost = qwen_interior_data_collection_top_asking(
                        qwen_client, message, indicator_type, used_tool, sub_indicators_info=None)

                og_answer_collection.append(temp_record)
                message_list.append(message)
                cost += temp_cost
    elif code!='14.1':
    # if indicator code ！= 14.1
        for reference_trunk in [table_reference, split_references]:
            for sub_reference in reference_trunk:

                if ask_parameters["mission_type"] == "计算型":
                    message, question_line = interior_data_collection_messages(sub_mode, sub_reference, sub_indicators_info,
                                                                               ask_parameters)

                    temp_record, temp_used_tool, found_answer, temp_cost = qwen_interior_data_collection_top_asking(
                        qwen_client, message, indicator_type, used_tool, sub_indicators_info=None, ai = 'qwen')
                else:
                    message, question_line = interior_data_collection_messages(sub_mode, sub_reference, sub_indicators_info,
                                                                               ask_parameters, ai="qwen")
                    temp_record, temp_used_tool, found_answer, temp_cost = qwen_interior_data_collection_top_asking(
                        qwen_client, message, indicator_type, used_tool, sub_indicators_info=None)

                og_answer_collection.append(temp_record)
                message_list.append(message)
                cost += temp_cost

                if found_answer:
                    break
            if found_answer:
                break



    all_answer_text = []
    all_answer_to_return = {}
    for index, each_answer in enumerate(og_answer_collection):
        all_answer_text.append(each_answer['answer'])
        all_answer_to_return[f"回复{index + 1}"] = each_answer['answer']
    if indicator_type == "单选" or indicator_type == "多选":
        options = ask_parameters["option"]
        result, assumption, temp_cost = indicator_collection_seek_final_answer(all_answer_text, indicator_type,
                                                                               used_tool,
                                                                               question_line,options=options)
    else:
        result, assumption, temp_cost = indicator_collection_seek_final_answer(all_answer_text, indicator_type, used_tool,
                                                                           question_line)

    cost += temp_cost
    all_answer_to_return = json.dumps(all_answer_to_return, ensure_ascii=False, indent=4)
    reference = json.dumps(reference, ensure_ascii=False, indent=4)
    table_reference = json.dumps(table_reference, ensure_ascii=False, indent=4)

    return {'all_answer_to_return': all_answer_to_return, 'result': result, 'reference_length': reference_length,
            'table_reference': table_reference, 'used_tool': used_tool, 'reference': reference, 'cost': cost,
            'assumption': assumption}
