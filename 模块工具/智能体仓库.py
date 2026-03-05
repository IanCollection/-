import copy
import json
import os
import time
from datetime import datetime

import numpy as np
import zhconv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ian_evolution.client_manager import qwen_client
from 模块工具.API调用工具 import make_request_with_retry, record_money, safe_get_input_money
from openai import OpenAI
from dotenv import load_dotenv

from 模块工具.通用工具 import Logger, data_recorder, split_long_strings, auto_retry
from ian_evolution.client_manager import siliconflow_client

load_dotenv()


DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')


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
def check_cell_complete(i, pre_table, current_table, table_title, head, check_cell, ai="azure", model="gpt-4o-mini"):
    def find_cell_example(column_index, pre_table, current_table):
        pre_table_cell_example = ""
        for each_row in pre_table["values"][:-1]:
            if each_row[column_index].strip() != "":
                pre_table_cell_example = each_row[column_index]
                break
        current_table_table_cell_example = ""
        for each_row in current_table["values"][1:]:
            if each_row[column_index].strip() != "":
                current_table_table_cell_example = each_row[column_index]
                break
        return pre_table_cell_example, current_table_table_cell_example

    pre_table_cell_example, current_table_table_cell_example = find_cell_example(i, pre_table, current_table)
    message = [{"role": "user",
                "content": f"text来自于一个表格,text对应的column是{head},table title是{table_title},我还会提供给你两个来自同一个column的两个example cell; example_cell_1:{pre_table_cell_example},example_cell_2:{current_table_table_cell_example}。请你结合标题还有提供的example_cell帮我判断text的结尾是否有非正常截断。{check_cell}\n如果是，只需要回答:Y 如果不是，只需要回答: N。你无需回答思考过程以及答案解释，而是直接给出答案。"}]

    message = json.dumps(message, ensure_ascii=False)


    client = qwen_client

    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result'].upper()

    data = {
        "题目类型": "判断是否是断表-分单元格检查",
        "question": message,
        "answer": answer,
        "necessary_infos": json.dumps({
            "sentence": check_cell,
            "head": head,
            "table_title": table_title
        })

    }
    data_recorder(data)

    temp_usage = completion['input_money']
    check_result = False
    if "Y" in answer:
        check_result = True

    return check_result, temp_usage


def special_title_check_bot(text, ai="qwen", model="qwen-max-latest"):
    # 准备消息1
    message = [{"role": "system",
                "content": f"我会给你一个句子，你需要帮我判断这里提到的'募集资金使用情况对照表'更可能是情况(1)一个表的标题，预示着下文应该就是表的本身。还是情况(2)他只是被引用提到的内容。"},
               {"role": "user",
                "content": f"我给你举个例子,当他是情况(1)一个标题的时候，句子中不会有太多的其余文字。text只要是'募集资金使用情况对照表' 或者'附件: 募集资金使用情况对照表',那他一定是个标题。需要注意的是'2020 年公开增发 A股股票募集资金使用情况对照表'等类似有限定的'募集资金使用情况对照表'也算是的。 当他是情况(2),text可能就会是'公司 2022 年度募集资金的实际使用情况请详见附表 1：2021 年向特定对象发行股票募集资金使用情况对照表。',这种情况,他就是被提到而已"},
               {"role": "user",
                "content": f"如果是情况(1)你就只回复'Y',如果是情况(2),你就只回复'N'"},
               {"role": "user",
                "content": f"以下是需要判断的句子{text}"}]

    # REMOVE: message = json.dumps(message, ensure_ascii=False)

    client = qwen_client
    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result'].upper()
    usage = completion['input_money']
    result = False
    if 'Y' in answer:
        result = True

    return usage, result

def check_combine_cell(head, above_row, lower_row, table_title, check_index, ai="qwen", model="qwen-max-latest"):
    message = [{"role": "user",
                "content": f"row1:{above_row} 和 row2:{lower_row} row，他们来源一同一个表格的上下两个row，已知对应的column header是{head},来自于表格{table_title}。请你帮我判断在row_index:{check_index},他是不是一个合并单元格。"
                           f"\n如果是(row1[{check_index}] 和 row2[{check_index}] 其实是一个合并单元格 )只需要回答:Y 如果不是(row1[{check_index}] 和 row2[{check_index}] 是两个独立的单元格)，只需要回答: N。你无需回答思考过程以及答案解释，而是直接给出答案。"}]

    message = json.dumps(message, ensure_ascii=False)

    client = qwen_client
    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result'].upper()
    usage = completion['input_money']

    data = {
        "题目类型": "判断是否是合并单元格",
        "question": message,
        "answer": answer,
        "necessary_infos": json.dumps({
            "above_row": above_row,
            "lower_row": lower_row,
            "table_title": table_title,
            "header": head})}
    data_recorder(data)
    if "Y" in answer:
        answer = True
    else:
        answer = False
    return usage, answer


def same_row_checker_bot_v2(pre_table, current_table, table_title, same_header, ai="azure", model="gpt-4o-mini"):
    if not same_header:
        first_row = current_table["header"][0]
    else:
        first_row = current_table["values"][0]
    complete = True
    total_usage = 0
    edge_case = False
    attention_case = False
    modify_index = []
    if len(pre_table["values"]) == 0 or '合计' in first_row:
        return total_usage, complete, edge_case, modify_index, attention_case
    last_row = pre_table["values"][-1]

    # 总体检查

    usage, first_check = two_row_checker_bot(pre_table["header"], last_row, first_row, table_title, ai=ai,
                                             model=model)
    total_usage += usage

    def is_number(string):
        try:
            # 去除字符串中的逗号
            string = string.replace(",", "").replace("，", "")
            # 尝试将字符串转换为浮点数
            float(string)
            return True
        except ValueError:
            return False

    # 检测出来是需要处理的case
    if "N" in first_check:
        complete1 = False
        complete2 = True
        at_least_one_complete_pairs = False
        # 分格子检查
        for i in range(len(first_row)):

            if i == 0 and (last_row[i].strip() == "" or first_row[i].strip() == ""):
                complete2 = False
                modify_index.append(i)
                continue
            if isinstance(first_row[i], float) or isinstance(last_row[i], float) or is_number(last_row[i]) or is_number(
                    first_row[i]):
                # 排列组合,如果上下都是数字，那at_least_one_complete_pairs = True
                if isinstance(first_row[i], float) and isinstance(last_row[i], float):
                    at_least_one_complete_pairs = True
                if isinstance(first_row[i], float) and is_number(last_row[i]):
                    at_least_one_complete_pairs = True
                if is_number(first_row[i]) and isinstance(last_row[i], float):
                    at_least_one_complete_pairs = True
                if is_number(first_row[i]) and is_number(last_row[i]):
                    at_least_one_complete_pairs = True
                continue
            if first_row[i].strip() != "" and last_row[i].strip() != "":
                check_cell_first = f"text:\"{first_row[i]}\""
                head = pre_table["header"][-1][i]
                head = f"\"{head}\""
            else:
                continue

            # # 检查下表的格子
            check_answer1, check_usage = check_cell_complete(i, pre_table, current_table, table_title, head,
                                                             check_cell_first, ai=ai, model=model)
            total_usage += check_usage

            # 检查上表的格子
            check_cell_last = f"text:\"{last_row[i]}\""
            check_answer2, check_usage = check_cell_complete(i, pre_table, current_table, table_title, head,
                                                             check_cell_last, ai=ai, model=model)
            total_usage += check_usage

            # given两个完整row,检查某个index是不是合并单元格
            final_check, temp_usage = check_combine_cell(head, last_row, first_row, table_title, i, ai="azure", model="gpt-4o-mini")
            temp_usage += temp_usage
            if final_check:
                modify_index.append(i)
                complete2 = False
            else:
                at_least_one_complete_pairs = True
            if (not check_answer1 and not check_answer2) and final_check:
                attention_case = True

        if complete1 == complete2:
            complete = False

        if not complete and at_least_one_complete_pairs:
            edge_case = True
    return total_usage, complete, edge_case, modify_index, attention_case


# 判断是否是表头
def header_check_bot(header, previous_header, ai="azure", model="gpt-4o-mini"):
    message = [{"role": "user",
                "content": f"rowA是一个表头请你帮我检查这个rowB更可能是他下属的内容，还是另外一个表头。rowA:{previous_header}。rowB:{header}。如果是表头，只需要回答:Y 如果是内容，只需要回答: N 。你无需回答思考过程以及答案解释，而是直接给出答案。"}]
    message = json.dumps(message, ensure_ascii=False)
    # print(message)

    client = qwen_client
    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result'].upper()
    # print(answer)
    if "Y" in answer:
        answer = True
    else:
        answer = False

    data = {
        "题目类型": "判断是否是表头",
        "question": message,
        "answer": answer,
        "necessary_infos": json.dumps({
            "header": header,
            "previous_header": previous_header
        })

    }
    data_recorder(data)
    
    # Add error handling for input_money field
    try:
        usage = completion.get('input_money', 0)
    except Exception as e:
        print(f"获取input_money时出错: {e}")
        usage = 0

    return usage, answer


# 判断是否是断表
def same_row_checker_bot(pre_table, current_table, table_title, same_header, ai="qwen", model="qwen-max-latest"):
    if not same_header:
        first_row = current_table["header"][0]
    else:
        first_row = current_table["values"][0]
    complete = True
    total_usage = 0
    edge_case = False
    modify_index = []
    if len(pre_table["values"]) == 0:
        return total_usage, complete, edge_case, modify_index
    last_row = pre_table["values"][-1]

    # 总体检查

    usage, first_check = two_row_checker_bot(pre_table["header"], last_row, first_row, table_title, ai=ai,
                                             model=model)
    total_usage += usage
    if "N" in first_check:
        complete1 = False
        complete2 = True
        # 分格子检查
        for i in range(len(first_row)):
            if isinstance(first_row[i], float) and isinstance(last_row[i], float) and np.isnan(first_row[i]) and np.isnan(last_row[i]):
                continue
            if first_row[i] != "" and last_row[i] != "":
                sentence = f"text:\"{last_row[i]}\""
                head = pre_table["header"][-1][i]
                head = f"\"{head}\""
            else:
                continue

            message = [{"role": "user",
                        "content": f"text来自于一个表格,text对应的column是{head},table title是{table_title}。请你结合标题帮我判断text的结尾是否有非正常截断。{sentence}\n如果是，只需要回答:Y 如果不是，只需要回答: N。你无需回答思考过程以及答案解释，而是直接给出答案。"}]

            message = json.dumps(message, ensure_ascii=False)


            client = qwen_client

            completion = make_request_with_retry(client, message, ai=ai, model=model)
            answer = completion['result'].upper()

            data = {
                "题目类型": "判断是否是断表-分单元格检查",
                "question": message,
                "answer": answer,
                "necessary_infos": json.dumps({
                    "sentence": sentence,
                    "head": head,
                    "table_title": table_title
                })

            }
            data_recorder(data)

            temp_usage = completion['input_money']
            total_usage += temp_usage

            if "N" in answer:
                if 0 in modify_index:
                    edge_case = True
            else:
                re_check_result = separate_line_bot(head, table_title, last_row[i], first_row[i], ai=ai,
                                                    model=model)
                total_usage = re_check_result[0]
                if "N" in re_check_result[1]:
                    complete2 = False
                    modify_index.append(i)
                    if 0 not in modify_index:
                        break
        if complete1 == complete2:
            complete = False
    return total_usage, complete, edge_case, modify_index

# 判断一个row是否被暴力切割
def two_row_checker_bot(head, above_row, lower_row, table_title, ai="qwen", model="qwen-plus-latest"):
    message = [{"role": "system",
                "content": f"我会提供给你两个row,row1和row2.他们是上下相邻的两个row.row1和row2可能是独立的两行，也有可能row1和row2其实是被截断的同一行或者存在合并单元格,但是因为断表导致看起来是两行。我还会提供他们对应的column_header和表格title。请你结合header帮我判断row1和row2的完整性,row1和row2即是不是独立的两行。"
                           "\n如果是(row1和row2是独立的两行)只需要回答:Y 如果不是(row1和row2其实是被截断的同一行h或者存在合并单元格)，只需要回答: N。"},
               {"role": "user",
                "content": f"例子1,表格title:'截至2023年12月31日,募集资金的存储情况列示如下:',column_header:'[['开户银行', '账号', '初始金额', '募集资金余额', '备注']]',row1:'['广发银行惠州仲恺科技园支行', '9550880207359101248', '161,266,067.92', '0', '已销户']',row2:'['中国工商银行惠州仲恺高新区支行', '2008022029200406025', '33,922,611.32', '0', '已销户']'"},
               {"role": "assistant",
                "content": f"Y"},
               {"role": "user",
                "content": f"例子2, 表格title:'成本分析表:分行业情况',column_header:'[['分行业', '成本构成项目', '本期金额', '本期占总成本比例(%)', '上年同期金额', '上年同期占总成本比例(%)', '本期金额较上年同期变动比例(%)', '情况说明']]',row1:'['集成电', '设备及材料安装劳务', '309,487,462.83', '34.60', '87,164,478.42', '12.91', '255.06', '主要系集成']',row2:'['路', '设备及材料安装劳务', '54,612,718.49', '6.10', '11,577,441.80', '1.71', '371.72', '电路业务增加所致。']'"},
               {"role": "assistant",
                "content": f"N"},
               {"role": "user",
                "content": f"例子3, 表格title:'成本分析表:分行业情况',column_header:'[['销售模式', '营业收入', '营业成本', '毛利率(%)', '营业收入比上年增减(%)', '营业成本比上年增减(%)', '毛利率比上年增减(%)']]',row1:'['直接销售', '1,230,165,184.91', '894,596,111.10', '27.28', '31.43', '32.50', '减少0.59个']',row2:'['', '', '', '', '', '', '百分点']'"},
               {"role": "assistant",
                "content": f"N"},
               {"role": "user",
                "content": f"请分析以下问题,表格title:{table_title},column_header:{head},row1:{above_row},row2:{lower_row}"}
               ]

    message = json.dumps(message, ensure_ascii=False)

    client = qwen_client

    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result'].upper()
    usage = completion['input_money']

    data = {
        "题目类型": "判断是否是断表-整行检查",
        "question": message,
        "answer": answer,
        "necessary_infos": json.dumps({
            "above_row": above_row,
            "lower_row": lower_row,
            "table_title": table_title,
            "header": head})}
    data_recorder(data)
    return usage, answer


# 检查是否同一cell被暴力切割
def separate_line_bot(head, table_title, above_cell, lower_cell,ai="azure", model="gpt-4o-mini"):
    # 准备消息
    if table_title != 'NA':
        message = [{"role": "system",
                    "content": f"我会给你两个text, textA 和 textB,我还会提供一个context, 你需要帮我判断,在这个context下,如果textA+textB合并成一个textC 单纯从textC语义逻辑性的角度考虑考虑,是否是一个合理的项。如果合理的话只需要回答:Y。如果是不合理,只需要回答: N"
                               f"例子1:<textA:<'直接人工'>,textB:<'制造费用'>,context:<'成本构成项目'>。 textC = '直接人工制造费用', 新组成的textC逻辑混乱,语义上也说不通, 所以不合理。>。 例子2:<textA:<'制造费'>,textB:<'用'>,context:<'成本构成项目'> 可以观察到,textC = '制造费用',他是一个语义完整切合乎逻辑的词,所以合理。 "},

                   {"role": "user",
                    "content": f"textA:<{above_cell}>,textB:<{lower_cell}>,context:<{table_title}:{head}>"}
                   ]
    else:
        message = [{"role": "system",
                    "content": f"我会给你两个text, textA 和 textB,我还会提供一个context, 你需要帮我判断,在这个context下,如果textA+textB合并成一个textC 单纯从textC语义逻辑性的角度考虑考虑,是否是一个合理的项。如果合理的话只需要回答:Y。如果是不合理,只需要回答: N"
                               f"例子1:<textA:<'直接人工'>,textB:<'制造费用'>,context:<'成本构成项目'>。 textC = '直接人工制造费用', 新组成的textC逻辑混乱,语义上也说不通, 所以不合理。>。 例子2:<textA:<'制造费'>,textB:<'用'>,context:<'成本构成项目'> 可以观察到,textC = '制造费用',他是一个语义完整切合乎逻辑的词,所以合理。 "},

                   {"role": "user",
                    "content": f"textA:<{above_cell}>,textB:<{lower_cell}>,context:<{head}>"}
                   ]

    message = json.dumps(message, ensure_ascii=False)

    client = qwen_client

    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result'].upper()
    usage = completion['input_money']
    data = {
        "题目类型": "检查是否同一cell被暴力切割",
        "question": message,
        "answer": answer,
        "necessary_infos": json.dumps({
            "above_cell": above_cell,
            "lower_cell": lower_cell,
            "table_title": table_title,
            "header": head})}

    data_recorder(data)
    return usage, answer


# 找表格标题的
def title_seek_bot(header, first_line, text, ai="azure", model="gpt-4o-mini"):
    if len(text) > 3:
        text = text[-3:]
    leading_text = "{\"title\":"
    if first_line:

        message = [
            {"role": "system",
             "content": "我会给你一个表头,表格的第一行和一个文本list,文本list里面是这个表前3行的文字,文本list的排序是根据该文本离表格的距离，由远到近,也就是说文本list[-1]就是这个表理论上的上一行。请根据表头和第一行，帮我从文本list中寻找这个表的标题和数据计量单位(如果有多个符合，优先选择距离最近的)如果某个选项带有'表'这个字',优先选择。请注意!你只能从文本list里面去选择文本作为标题的答案或者数据计量单位！请按照如下json格式提供答案: {\"title\": 找到的标题, \"unit\": 找到的计量单位}。title一定可以找到,但是unit不一定。如果未找到unit，请返回: {\"title\": 找到的标题, \"unit\": \"NONE\"}。你无需回答思考过程以及答案解释，而是直接给出答案。"},
            {"role": "user",
             "content": "以下是例子1。表头:'[['户名', '开户行名称', '银行账号', '募集资金专户余额']]',第一行:'['江苏南大光电材料股份有限公司', '宁波南大光电材料有限公司', '江苏南大光电材料股份有限公司', '江苏南大光电材料股份有限公司', '南大光电(淄博)有限公司', '南大光电(淄博)有限公司', '合计']'}, 文本list:'['截至2023年12月31日止,公司2021年向特定对象发行股票募集资金存放', '专项账户的余额如下:', '单位:人民币元']'"},
            {"role": "assistant",
             "content": "{'title': 公司2021年向特定对象发行股票募集资金存放专项账户的余额, 'unit': '人民币元'}"},
            {'role': 'user',
             'content': f"以下是你需要判断的。表头:{header},第一行:{first_line}, 文本list:{text}"},
            {"role": "assistant", "content": leading_text, "partial": True},
        ]
    else:

        message = [
            {"role": "system",
             "content": "我会给你一个表头和一个文本list,文本list里面是这个表前3行的文字,文本list的排序是根据该文本离表格的距离，由远到近,也就是说文本list[-1]就是这个表理论上的上一行。请根据表头和第一行，帮我从文本list中寻找这个表的标题和数据计量单位(如果有多个符合，优先选择距离最近的)如果某个选项带有'表'这个字',优先选择。请注意!你只能从文本list里面去选择文本作为标题的答案或者数据计量单位！请按照如下json格式提供答案: {\"title\": 找到的标题, \"unit\": 找到的计量单位}。title一定可以找到,但是unit不一定。如果未找到unit，请返回: {\"title\": 找到的标题, \"unit\": \"NONE\"}。你无需回答思考过程以及答案解释，而是直接给出答案。"},
            {"role": "user",
             "content": "以下是例子1。表头:'[['户名', '开户行名称', '银行账号', '募集资金专户余额']]', 文本list:'['截至2023年12月31日止,公司2021年向特定对象发行股票募集资金存放', '专项账户的余额如下:', '单位:人民币元']'"},
            {"role": "assistant",
             "content": "{'title': 公司2021年向特定对象发行股票募集资金存放专项账户的余额, 'unit': '人民币元'}"},
            {'role': 'user',
             'content': f"以下是你需要判断的。表头:{header}, 文本list:{text}"},
            {"role": "assistant", "content": leading_text, "partial": True},
        ]

    message = json.dumps(message, ensure_ascii=False)

    client = qwen_client
    completion = make_request_with_retry(client, message, True, ai=ai, model=model)
    if ai == 'moonshot':
        answer = leading_text + completion['result']
    else:
        answer = completion['result']

    table_title, unit = eval(answer)["title"], eval(answer)["unit"]
    if first_line:
        if "专户" in first_line or "账户" in str(header) or "专户" in str(text):
            table_title += "(涉及账户)"
    elif "账户" in str(header) or "专户" in str(text):
        table_title += "(涉及账户)"

    answer = json.dumps({"title": table_title, "unit": unit})
    usage = completion['input_money']
    if first_line:
        data = {
            "题目类型": "找表格标题",
            "question": message,
            "answer": answer,
            "necessary_infos": json.dumps({
                "header": header,
                "first_line": first_line,
                "text": text
            })

        }
    else:
        data = {
            "题目类型": "找表格标题",
            "question": message,
            "answer": answer,
            "necessary_infos": json.dumps({
                "header": header,
                "text": text
            })

        }
    data_recorder(data)
    return usage, answer


# 造一个表格标题的
def title_makeup_bot(header, key_index, ai="qwen", model="qwen-plus-latest"):
    message = [{"role": "system",
                "content": f"我会给你一个表格的表头，然后还有他们的第一个column即index，你需要帮我给这个表取合适的中文表名。你的回答中，只需要回答这个表名即可。"},
               {"role": "user",
                "content": f"以下是问题。表头:{header}。第一个column:{key_index}。你无需回答思考过程以及答案解释，而是直接给出答案。"}
               ]
    message = json.dumps(message, ensure_ascii=False)
    if ai =='qwen':
        client = qwen_client
    else:
        client = None
    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result']
    usage = completion['input_money']
    data = {
        "题目类型": "造一个表格标题",
        "question": message,
        "answer": answer,
        "necessary_infos": json.dumps({
            "header": header,
            "key_index": key_index
        })

    }
    data_recorder(data)
    return usage, answer

def num_extract_bot(question, pre_answer,  ai="qwen", model="qwen-max-latest"):
    message = [{"role": "system",
                "content": f"我会给你一个问题，一个回答文本，你帮我从回答文本中提取问题所需要的纯数字的结果。我给你举一个例子: 问题: \"请问你的年龄是多少?\" 回答:\"我今年25岁\"。这个问题的答案就是25。你只需要回复'25'"},
               {"role": "user",
                "content": f"以下是问题:{question}。回答文本:{pre_answer}。你无需回答思考过程以及答案解释，而是直接给出答案。"}
               ]
    message = json.dumps(message, ensure_ascii=False)
    if ai =='qwen':
        client = qwen_client
    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result']
    usage = completion['input_money']
    return usage, answer


# 检查时候否是页码的
def page_check_bot(whole_page, text, ai="qwen", model="qwen-plus-latest"):
    # 准备消息
    message = [{"role": "user",
                "content": f"我会给你一页的全部内容的文本和最后一行text,你需要帮我判断text是不是页尾的页码标注。"
                           f"我给你举个例子，本页内容:'balabalaba', text:'审计报告 第3页'。很明显text就是页码标注。"
                           f"我再给你一个例子，本页内容:'balabalaba',text:'3/235'。这个text也是页码标注。"
                           f"如果你觉得text大概率是用来表示页码的，只需要回答: Y。如果不是，只需要回答: N。你无需回答思考过程以及答案解释，而是直接给出答案。"
                           f"以下是问题。本页内容:{whole_page}。text:{text}"
                }]

    message = json.dumps(message, ensure_ascii=False)

    if ai =="qwen":
        client = qwen_client
    else:
        client = None
    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result'].upper()
    true_answer = False
    if "Y" in answer:
        true_answer = True
    usage = completion['input_money']

    data = {
        "题目类型": "检查是否是页码",
        "question": message,
        "answer": answer,
        "necessary_infos": json.dumps({
            "whole_page": whole_page,
            "text": text
        })

    }
    data_recorder(data)
    return usage, true_answer


# 判断是否拼接报告的
def title_level_reset_bot(text,  ai="qwen", model="qwen-max-latest"):
    message = [
        {"role": "system",
         "content": "我会给你一段文字，你需要帮我判断这段文字更像是一个标题，还是正常的一句话。我给你举个例子，\"一、基本情况介绍\"，他就更像是一个标题。如果你觉得更像是一个标题，就只需要回答: Y。如果你觉得更像是一个句子，只需要回答: N。你无需回答思考过程以及答案解释，而是直接给出答案。最后用json格式返回结果。"},
        {"role": "user",
         "content": f"文字:{text}"}]
    message = json.dumps(message, ensure_ascii=False)
    # print(message)
    retry = 0
    total_retry = 10

    client = qwen_client
    while retry < total_retry:
        try:
            completion = make_request_with_retry(client, message, True, ai=ai, model=model)
            answer = completion['result'].upper()

            answer_to_return = False
            if 'Y' in answer:
                answer_to_return = True
            data = {
                "题目类型": "判断是否拼接报告",
                "question": message,
                "answer": answer,
                "necessary_infos": json.dumps({
                    "text": text
                })
            }
            data_recorder(data)
            return answer_to_return
            # usage = completion.usage.total_tokens
        except Exception as e:
            print(e)
            retry += 1
    return False


# 判断是否标题异常识别的
def strange_title_check_bot(pre_sentence, text,  ai="qwen", model="qwen-plus-latest"):
    message = [
        {"role": "system",
         "content": "我会给你两个text，textA是textB的上一行，你需要帮我判断textB是不是一个合理的标题。例子1，textA"
                    ":\"公司经营并无异常\"，textB"
                    ":\"2.公司人力介绍\"，他就是一个标题。例子2，textA"
                    ":\"公司营业收入192,133,12\"，textB"
                    ":\"0. 12 元。\"，他就不是一个合理的标题，而是上一句话的顺延。"
                    "例子3，textA:\"具体业务有:\"，textB"
                    ":\"(1)钢铁业务\"，他就是一个合理的标题。"
                    "如果你觉得textB更像是一个合理标题，就只需要回答: Y,如果你觉得textB不是一个合理的标题，就只需要回答: N。你无需回答思考过程以及答案解释，而是直接给出答案。最后用json格式返回结果。"},
        {"role": "user",
         "content": f"textA:{pre_sentence}。textB:{text}"}]
    message = json.dumps(message, ensure_ascii=False)

    client = qwen_client
    # print(message)
    retry = 0
    total_retry = 10
    while retry < total_retry:
        try:
            completion = make_request_with_retry(client, message, True, ai=ai, model=model)
            answer = completion['result'].upper()
            answer_to_return = False
            if 'Y' in answer:
                answer_to_return = True

            data = {
                "题目类型": "判断是否标题异常识别",
                "question": message,
                "answer": answer,
                "necessary_infos": json.dumps({
                    "pre_sentence": pre_sentence,
                    "text": text
                })

            }
            data_recorder(data)
            return answer_to_return
            # usage = completion.usage.total_tokens
        except Exception as e:
            print(e)
            retry += 1
    return False


# 表格层级穿透的
def table_title_expand_bot(upper_title, table_tile,  ai="qwen", model="qwen-plus-latest"):
    message = [{"role": "system",
                "content": f"我会给你一个上一级标题,和一个表格的名字，"
                           f"你需要帮我判断这个上一级标题是否有对表格的限定信息，言外之意就是如果没有上级标题，我能不能依旧通过表格的名字获得全部的信息。"
                           f"我给你举一个例子，上一级标题:按信用风险特征组合计提坏账准备的应收账款，表格名字:账龄组合。 "
                           f"在这个例子里面，上一级标题是包含有对表格的限定信息的。 因为你只看表格名字:账龄组合，你并不知道他是关于谁的账龄组合，你只有结合上一级标题，才能获得全部信息。所以答案是 是的"},
               {"role": "user",
                "content": f"如果你觉得上一级标题包含对表格的限定信息，只需要回答: Y。如果没有包含，只需要回答: N。你无需回答思考过程以及答案解释，而是直接给出答案。"},
               {"role": "user",
                "content": f"上一级标题:{upper_title}，表格名字:{table_tile}"}
               ]
    message = json.dumps(message, ensure_ascii=False)

    client = qwen_client
    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result'].upper()
    actual_answer = False
    if "Y" in answer:
        actual_answer = True

    usage = completion['input_money']
    db, engine = new_create_db_session()
    try:
        record_money(usage, db, is_report=False, mission_type=None, is_embedding=False, is_extra=True)
    except Exception as e:
        print(e)
    finally:
        db.close()
        engine.dispose()


    return usage, actual_answer


# 母公司表格穿透的
def mother_company_check_bot(parent_info, title, ai="qwen", model="qwen-plus"):
    message = [{"role": "system",
                "content": f"我会给你一个table的标题，还有这个table的文章母层级标题，他是从大到小的包含关系；你需要根据母标题帮我判断，这个table是否是母公司的表格。parent_info: {parent_info},table_title:{title}。我给你举个例子，如果母标题出现' 母公司财务报表细分项'等，很明显是要对母公司的信息作为展开的，table就很有可能是母公司的细分表格"},
               {"role": "user",
                "content": f"如果你觉得这个table是母公司相关的，只需要回答: Y。如果不是，只需要回答: N。你无需回答思考过程以及答案解释，而是直接给出答案。"}
               ]
    message = json.dumps(message, ensure_ascii=False)

    if ai == 'qwen':
        client = qwen_client
    else:
        client = None
    completion = make_request_with_retry(client, message, ai=ai, model=model)
    answer = completion['result'].upper()
    usage = completion['input_money']

    data = {
        "题目类型": "母公司表格穿透",
        "question": message,
        "answer": answer,
        "necessary_infos": json.dumps({
            "parent_info": parent_info,
            "title": title
        })

    }
    data_recorder(data)
    return usage, answer


# 生成similarity keyword的
def get_similarity_words_bot(indicator, ai="qwen", model="qwen-max"):
    try_times = 0
    total_retry_times = 10
    client = qwen_client
    sys_prompt = "请基于给定的指标生成适合用于知识库检索该指标信息的相似词，最后的输出以JSON格式返回1到3个合适的关键词结果"
    example = "指标:'水资源消耗强度（单位产值）'" + "\n" + "最后你输出的形式为JSON:'{'similarity_words':'[水资源消耗]'}'" + "\n" + "指标:'是否披露能源使用管理'" + "\n" + "最后你输出为JSON形式:'{'similarity_words':'['多元能源结构','能源管理','绿色能源']'}'" + "\n" + "指标:'企业节电量'" + "\n" + "最后你输出的JSON:'{'similarity_words':'['节点','节约用电']}'" + "\n" + "指标:'化学需氧量排放量'" + "\n" + "最后你输出的JSON:'{'similarity_words':'[化学需氧量排放]'}'" + "\n"
    message = [
        {"role": "system",
         "content": sys_prompt + "\n" + "example:" + "\n" + example},
        {"role": "user", "content": f"给定的指标为{indicator}"}

    ]
    message = json.dumps(message, ensure_ascii=False)
    while try_times < total_retry_times:
        try:

            completion = make_request_with_retry(client, message, True, ai=ai, model=model)
            result = completion['result']
            to_return_list = json.loads(result)["similarity_words"]
            to_return_list.append(indicator)

            data = {
                "题目类型": "生成similarity keyword",
                "question": message,
                "answer": result,
                "necessary_infos": json.dumps({
                    "indicator": indicator
                })

            }
            data_recorder(data)
            return to_return_list

        except Exception as e:
            # print(f"Error - {e}")
            try_times += 1
    return indicator


# 找正文开始部分的
def get_mainbody_v2(ocr_result, ai="moonshot", model="moonshot-v1-8k"):


    def pure_ocr_result(ocr_json):
        pure_ocr_result = []
        for _, page in ocr_json.items():
            try:
                true_page_content = page["result"]["tables"]
                pure_ocr_result.append(true_page_content)

            except Exception as error:
                print(error)
                print("ocr bug")
                continue
        return pure_ocr_result

    ocr_result = pure_ocr_result(ocr_result)
    return ocr_result
    found_menu = False
    for index, page in enumerate(ocr_result[:10]):
        retry = 0
        total_retry = 5
        while retry < total_retry:
            try:
                if page == []:
                    break
                page_content = page[0]
                if page_content["type"] == "plain":
                    page_texts = page_content["lines"]
                    new_page_text = ""
                    for items in page_texts:
                        if 'text' in items:
                            new_page_text += items['text']
                else:
                    break
                sys_prompt = '我将会给你一篇报告某一页的内容，你帮我识别一下他是否是目录或者是目录的一部分。如果是的话，你直接给我回答 "Y" ,如果不是的话，你直接给我回答 "N"。你无需回答思考过程以及答案解释，而是直接给出答案。'
                messages = [{"role": "system", "content": f"{sys_prompt}"}]
                messages.extend(
                    [{"role": "user", "content": f"内容:{new_page_text}"}])
                message = json.dumps(messages, ensure_ascii=False)
                # 创建完整请求
                client = qwen_client
                completion = make_request_with_retry(client, message, False, ai=ai, model=model)
                result = completion['result'].upper()
                usage = completion['input_money']
                db, engine = new_create_db_session()
                try:
                    record_money(usage, db, is_report=False, mission_type=None, is_embedding=False, is_extra=True)
                except Exception as e:
                    print(e)
                finally:
                    db.close()
                    engine.dispose()
                data = {
                    "题目类型": "找正文开始部分",
                    "question": message,
                    "answer": result,
                    "necessary_infos": json.dumps({
                        "page": new_page_text
                    })

                }
                data_recorder(data)
                if "Y" in result:
                    found_menu = True
                    break
                elif "N" in result and found_menu:
                    return ocr_result[index:]
                elif "N" in result:
                    break
                else:
                    retry += 1
            except Exception as e:
                retry += 1
                print(f"找mainbody的时候有问题，Error - {e}")
                print(f"page:{index}")
                print("马上重试")
                continue

    return ocr_result


# 找子指标的
def check_sub_indicators(description, year, ai="qwen", model="qwen-max"):

    client = qwen_client

    message = [{"role": "system", "content": "你是一个金融和数学专家"},
               {"role": "user",
                "content": "你会被告知一段文字,你需要从文字描述中帮我找到所需的指标、单位（如有）和年份(如有),并用json格式返回。我下面给你一个例子"},
               {"role": "user",
                "content": "问题:利率 =（两年年平均）(营业收入 -营业成本)/营业收入。年份:2023。"},
               {"role": "assistant",
                "content": "{'indicators':[{'name':'营业收入','unit':'无','year':'2023}',{'name':'营业成本',"
                           "'unit':'无','year':'2023},{'name':'营业收入','unit':'无','year':'2022},"
                           "{'name':'营业成本','unit':'无','year':'2022}"},
               {"role": "user",
                "content": "下面是你要解析的文字:{description}。年份:{year}"}
               ]
    for idx, each_dict in enumerate(message):
        content = each_dict["content"]
        new_content = content.replace("{description}", str(description)).replace("{year}", str(year))
        message[idx]["content"] = new_content
    message = json.dumps(message, ensure_ascii=False)
    completion = make_request_with_retry(client, message, True, ai=ai, model=model)
    answer = completion['result']
    usage = completion['input_money']
    return usage, answer


# 找目标section的
def find_section_target_bot(table_name_list, resource_keywords, ai="qwen", model="qwen-max-latest"):
    total_usage = 0


    client = qwen_client
    message = [{"role": "system",
                "content": f"我会给你我想找到的target_section的名字list, 然后我还会给你一个现有的candidate_section的名字list。"
                           f"你需要帮从candidate_section_list里面找到最有可能是我想要的一个或多个section,然后将他们的名字以JSON的格式返回给我。"
                           f"我给你举个例子，target_section_list:['独立董事的基本情况 ']。candidate_section_list:['一、独立董事的基本情况','二、年度履职概况 ','（一）关联交易情况']。"
                           "你就应该以JSON格式返回,答案请使用简体中文，{\"answer\":[\"一、独立董事的基本情况\"]}。你无需回答思考过程以及答案解释，而是直接给出答案。"},
               {"role": "user",
                "content": f"以下是问题。target_section_list:{resource_keywords},candidate_section_list:{table_name_list}"}
               ]
    # message = json.dumps(message, ensure_ascii=False)
    retry = 0
    while retry < 10:
        try:
            completion = make_request_with_retry(client, message, True, ai=ai, model=model)
            answer = completion['result']
            usage = completion['input_money']
            total_usage += usage
            dict_answer = eval(zhconv.convert(answer, 'zh-cn'))
            final_answer = dict_answer["answer"]
            if type(final_answer) == list:
                return total_usage, final_answer
            else:
                retry += 1
                continue
        except Exception as e:
            print(e)
            retry += 1
            continue
    return total_usage, table_name_list


# 找到目标table的
def find_table_target_bot(table_name_list, resource_keywords, ai="qwen", model="qwen-max-latest"):
    total_usage = 0
    client = qwen_client
    message = [{"role": "system",
                "content": f"我会给你我想找到的target_table的名字list, 然后我还会给你一个现有的candidate_table的名字list。"
                           f"你需要帮从candidate_table_list里面找到最有可能是我想要的一个或多个table,然后将他们的名字以JSON的格式返回给我。"
                           f"我给你举个例子，target_table_list:['营业收入']。candidate_table_list:['营业收入,营业成本','资产负债表','应收帐款']。"
                           "你就应该以JSON格式返回,答案请使用简体中文，{\"answer\":[\"营业收入,营业成本\"]}。你无需回答思考过程以及答案解释，而是直接给出答案。"},
               {"role": "user",
                "content": f"以下是问题。target_table_list:{resource_keywords},candidate_table_list:{table_name_list}"}
               ]
    message = json.dumps(message, ensure_ascii=False)
    retry = 0
    while retry < 10:
        try:
            completion = make_request_with_retry(client, message, True, ai=ai, model=model)
            answer = completion['result']
            usage = completion['input_money']
            total_usage += usage
            dict_answer = eval(zhconv.convert(answer, 'zh-cn'))
            final_answer = dict_answer["answer"]
            if type(final_answer) == list:

                data = {
                    "题目类型": "找到目标table",
                    "question": message,
                    "answer": final_answer,
                    "necessary_infos": json.dumps({
                        "table_name_list": table_name_list,
                        "resource_keywords": resource_keywords
                    })

                }

                data_recorder(data)
                return total_usage, final_answer
            else:
                retry += 1
                continue
        except Exception as e:
            print(e)
            retry += 1
            continue
    return total_usage, table_name_list


# 内评数据收集的连环判断

def interior_data_collection_messages(sub_mode, sub_reference, sub_indicators_info, ask_parameters, ai="qwen"):
    indicator_name = ask_parameters["name"]
    year = ask_parameters["year"]
    equation = ask_parameters["equation"]
    explain = ask_parameters["explain"]
    equality_question = ask_parameters["equality_question"]
    option = ask_parameters["option"]
    company_name = ask_parameters["company_name"]
    allow_creation = ask_parameters["allow_creation"]
    indicator_type = ask_parameters["indicator_type"]
    mission_type = ask_parameters["mission_type"]
    positive_example = ask_parameters["positive_example"]
    positive_example_reason = ask_parameters["positive_example_reason"]
    negative_example = ask_parameters["negative_example"]
    negative_example_reason = ask_parameters["negative_example_reason"]
    necessary_points = ask_parameters["necessary_points"]
    missing_fill = ask_parameters.get('missing_fill','')
    if "dynamic_keywords" in ask_parameters:
        dynamic_keywords = ask_parameters["dynamic_keywords"]
    else:
        dynamic_keywords = None
    question_line = ""
    base_question_line = ""
    question = indicator_name
    note = "无"
    if equality_question is not None:
        note = f",注意在这个问题中，你可以认为{equality_question}和{indicator_name}是等价的"
    if indicator_type == "数值":
        if not sub_indicators_info or sub_mode == 0:
            if equation is None:
                question_line = f"请帮我回答关于{dynamic_keywords}的问题:{question},问题解释:{explain},备注信息:{note}。注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{dynamic_keywords}。"
            else:
                question_line = f"请帮我回答关于{dynamic_keywords}的问题:{question},问题解释:{explain},备注信息:{note}。注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{dynamic_keywords}"
            base_question_line = question_line
            if positive_example:
                question_line += f"为了能让你明白判断的尺度我为你提供了正例。正例:{positive_example},原因:{positive_example_reason}。"
            if negative_example:
                question_line += f"为了能让你明白判断的尺度我为你提供了反例:{negative_example},原因:{negative_example_reason}。"

            if necessary_points:
                strictness = f"(注意!以下内容是必须要有的:{necessary_points}。)"
            else:
                strictness = ""

            message = [
                {"role": "user", "content": f"{sub_reference}"},
                {"role": "user",
                 "content": f"你是一个金融分析师，你需要基于我给你提供的参考材料回答问题，但是并不是所有参考信息都是有用信息。注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{dynamic_keywords}。"
                            f"回答的时候需要说出根据哪一个或多个参考得到的答案(你只需要reference参考+序号，不用reference里面的内容),比如'根据参考4,人口总数是 123,321。"
                            f"{question_line}"
                            f"{strictness}"
                            f"如果参考材料完全没有所需信息，则回复'信息不足'。"}
            ]

        if mission_type == "计算型":
            message[-1]["content"] = message[-1]["content"] + "当涉及数学计算时，you can think step by step，你必须展示计算公式和相关数据，编写并且运行代码来回答问题。"


    elif "单选" in indicator_type or "多选" in indicator_type:
        if "单选" in indicator_type:
            question_line = f"请帮我回答关于{dynamic_keywords}的问题:{question},问题解释:{explain},你需要从option中选出一个最佳选项,option:{option},注意要保证你的答案和选项的用词一致性。注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{dynamic_keywords}"
        else:
            question_line = f"请帮我回答关于{dynamic_keywords}的问题:{question},问题解释:{explain},你需要从option中选出一个或多个最佳选项,option:{option},注意要保证你的答案和选项的用词一致性。注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{dynamic_keywords}"
        base_question_line = question_line
        if positive_example:
            question_line += f"为了能让你明白判断的尺度我为你提供了正例。正例:{positive_example},原因:{positive_example_reason}。"
        if negative_example:
            question_line += f"为了能让你明白判断的尺度我为你提供了反例:{negative_example},原因:{negative_example_reason}。"

        strictness = f"(需要注意的是,材料往往并不能满足所有的问题需求,但是如果是有多种情况下的一种得到满足(除非特别说明需要严格满足要求)，都可以视为1。但是你需要说出你的判断理由。"
        if necessary_points:
            strictness += f"但是注意!以下内容是必须要有的:{necessary_points}。)"
        else:
            strictness += ")"

        message = [
            {"role": "user", "content": f"{sub_reference}"},
            {"role": "user",
             "content": f"你是一个金融分析师，你需要基于我给你提供的参考材料做出判定。注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{dynamic_keywords},但是并不是所有参考信息都是有用信息。回答的时候需要展示原文reference(在你的回答中引用原文原句)。"
                        f"{question_line}"
                        f"{strictness}"
                        f"如果参考材料完全没有所需信息，则回复'信息不足'。"
             }

        ]
    elif indicator_type == "文本":
        question_line = f"请帮我回答关于{dynamic_keywords}的问题:{question},问题解释:{explain},备注信息:{note}。注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{dynamic_keywords}。注意!回答需要中文"
        base_question_line = question_line
        if positive_example:
            question_line += f"为了能让你明白判断的尺度我为你提供了正例。正例:{positive_example},原因:{positive_example_reason}。"
        if negative_example:
            question_line += f"为了能让你明白判断的尺度我为你提供了反例:{negative_example},原因:{negative_example_reason}。"

        if necessary_points:
            strictness = f"(注意!以下内容是必须要有的:{necessary_points}。)"
        else:
            strictness = ""

        message = [
            {"role": "user", "content": f"{sub_reference}"},
            {"role": "user",
             "content": f"你是一个金融分析师，你需要基于我给你提供的参考材料回答问题，但是并不是所有参考信息都是有用信息。注意,如果材料中没有提及这个材料是关于谁的，可以默认是关于{dynamic_keywords}。"
                        f"回答的时候需要说出根据哪一个或多个参考得到的答案(你只需要reference参考+序号，不用reference里面的内容),比如'根据参考4,人口总数是 123,321。"
                        f"{question_line}"
                        f"{strictness}"
                        f"如果参考材料完全没有所需信息，则只回复'信息不足'并且不要做多余的解释"}
        ]

    else:
        raise ValueError("指标类型错误")

    if ai != "openai":
        message = [{"role": "system",
                    "content": f"你是一个金融分析师，你需要基于我给你提供的参考材料回答问题，但是并不是所有参考信息都是有用信息。如果没有明确的对象信息，可以从{dynamic_keywords}的角度进行相关信息获取。"
                               f"并且回答问题的对象是{company_name}的信息，不要回答其他无关公司的信息。"
                               f"回答的时候需要说出根据哪一个或多个参考得到的答案(你只需要reference参考+序号，不用reference里面的内容),比如'根据参考4,人口总数是 123,321。"
                               f"{question_line}"
                               f"如果参考材料完全没有所需信息，则回复'信息不足'并且不要做任何多余的解释和假设。"}] + message
    return message, base_question_line


import ast
import json
import copy
import time


def indicator_collection_seek_final_answer(answer, question_type, used_tool, question_line, ai="qwen", model="qwen-max-latest", options=None):
    attempts = 0
    max_attempts = 5  # Consider making this a parameter or a module-level constant
    exam_text = ""
    no_info = True
    current_cost = 0 # Renamed from cost for clarity within the loop
    
    # Counter for numbering answers that actually contain information
    answers_with_info_counter = 0
    leading_text_for_moonshot = ""  # Specific to moonshot's potential output quirks
    with_answer_text = ""
    
    parsed_options_list = None
    if options:
        if isinstance(options, list):
            parsed_options_list = options
        elif isinstance(options, str) and options.strip():
            try:
                # Try parsing as JSON string first (e.g., '["A", "B"]')
                loaded_options = json.loads(options)
                if isinstance(loaded_options, list):
                    parsed_options_list = loaded_options
                else:
                    # If not a JSON list, fall back to ast.literal_eval for Python list string (e.g., "['A', 'B']")
                    # This is safer than eval()
                    parsed_eval_options = ast.literal_eval(options)
                    if isinstance(parsed_eval_options, list):
                        parsed_options_list = parsed_eval_options
                    else:
                        print(f"Warning: Options string '{options}' parsed by ast.literal_eval was not a list.")
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                # If JSON and ast.literal_eval fail, try ast.literal_eval directly (handles simple Python list strings)
                if not parsed_options_list: # if json.loads failed or produced non-list
                    try:
                        parsed_eval_options = ast.literal_eval(options)
                        if isinstance(parsed_eval_options, list):
                            parsed_options_list = parsed_eval_options
                        else:
                             print(f"Warning: Options string '{options}' parsed by ast.literal_eval was not a list (second attempt).")
                    except (ValueError, SyntaxError) as e_ast:
                         print(f"Warning: Could not parse options string '{options}' into a list. JSON error: {e}, AST error: {e_ast}")
            except Exception as e_gen: # Catch any other unexpected error during options parsing
                print(f"Warning: Unexpected error parsing options string '{options}': {e_gen}")
        # If options is not a list or a parsable string, parsed_options_list remains None

    for index, each_answer_item in enumerate(answer):
        if "信息不足" not in each_answer_item:
            no_info = False
            with_answer_text += f"参考{answers_with_info_counter + 1}:{each_answer_item}。\n"
            answers_with_info_counter += 1
        exam_text += f"参考{index + 1}:{each_answer_item}。\n"
    
    if no_info:
        # If all provided answers indicate "信息不足", no need to call LLM for summarization
        return "信息不足", 0, 0 # answer, message_placeholder, cost

    while attempts < max_attempts:
        try:
            system_prompt_base = (
                "我会提供给你一个或多个关于一个问题的答案和答案的参考依据，"
                "请你将这些答案整合，并使用中文提供最终的答案。"
                "请注意，如果任意一个参考依据提供了答案，优先选择。"
                "你无需回答思考过程以及答案解释，而是直接给出答案。"
                f"对于options不为空的情况下，如果答案里有option的答案，尤其是答案里有“是”，那么就直接回答‘是’"
                f"这是一个问题答案的option：{options}"
            )
            json_format_instruction = ""
            additional_instructions = ""

            if question_type in ("单选", "多选"):
                options_constraint_str = ""
                if parsed_options_list:
                    options_constraint_str = f"可选项为：{json.dumps(parsed_options_list, ensure_ascii=False)}。你必须从这些选项里选择答案，不能选择其他内容。"
                
                additional_instructions = f"{options_constraint_str} "
                if question_type == "单选":
                    additional_instructions += "你必须从给定的选项中选择一个最合适的答案。"
                else:  # 多选
                    additional_instructions += "你可以从给定的选项中选择一个或多个最合适的答案。"
                additional_instructions += "如果原答案不完全在选项中，请选择最接近的选项。"
                # Instruct LLM to pick from options, even if source materials are sparse.
                json_format_instruction = f"请按照如下JSON格式提供答案:{{\"answer\": \"选项中的答案\"}}。即使原始材料信息不足，也请尝试从选项中选择一个或多个（对于多选）答案。"

            elif question_type == "数值":
                additional_instructions = ("你只需要回答数值+计量单位，无需添加解释或者限定文字。"
                                         "注意！如果有计量单位的话,必须保留单位。")
                json_format_instruction = f"请按照如下JSON格式提供答案:{{\"answer\": \"你的答案\"}}。如果原答案没有提供任何信息，则应回答:{json.dumps({'answer':'信息不足'}, ensure_ascii=False)}。"

            elif question_type == "文本":
                additional_instructions = "你只需要回答最终答案，无需添加解释或者限定文字。"
                json_format_instruction = f"请按照如下JSON格式提供答案:{{\"answer\": \"你的答案\"}}。如果原答案没有提供任何信息，则应回答:{json.dumps({'answer':'信息不足'}, ensure_ascii=False)}。"
            
            else:  # Default type, includes assumption checking
                additional_instructions = (
                    "你还需要帮我判断原答案是否做了假设。我给你两个作了假设的例子。"
                    "例子1:'假设已使用的银行授信完全体现在短期和长期借款上。'"
                    "例子2:'我将使用这些参考资料来估计盘锦水务集团有限公司的已使用银行授信' "
                    "如果做了假设，对应的value是1，如果没有做假设，对应的value是0。"
                )
                no_info_assumption_response = {"answer": "-0", "assumption": "0"}
                json_format_instruction = f"请按照如下JSON格式提供答案:{{\"answer\": \"你的答案\", \"assumption\": \"判断结果\"}}。如果原答案没有提供任何信息，则应回答:{json.dumps(no_info_assumption_response, ensure_ascii=False)}。"

            system_content = f"{system_prompt_base} {additional_instructions} {json_format_instruction}".strip()
            
            current_message_payload = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"原问题:{question_line}，原答案:{with_answer_text}。"},
            ]

            client_to_use = qwen_client

            message_json_str = json.dumps(current_message_payload, ensure_ascii=False)

            try:
                completion_result = make_request_with_retry(client_to_use, message_json_str, True, ai=ai, model=model)
            except Exception as e_api:
                if "considered high risk" in str(e_api): # Specific error check from original code
                    raise e_api 
                # For other API errors, let the main loop's exception handler deal with retries
                raise # Re-raise to be caught by the outer try-except of this attempt

            raw_llm_output_str = completion_result.get('result')
            current_cost = completion_result.get('input_money', 0)

            if raw_llm_output_str is None:
                print(f"总结员Attempt {attempts + 1} failed: LLM response missing 'result' key. Response: {completion_result}")
                attempts += 1
                time.sleep(1)
                continue

            parsed_llm_response = None
            try:
                parsed_llm_response = json.loads(raw_llm_output_str)
            except json.JSONDecodeError:
                try:
                    parsed_llm_response = ast.literal_eval(raw_llm_output_str)
                except (ValueError, SyntaxError):
                    if ai == 'moonshot' and leading_text_for_moonshot:
                        # Try to fix potentially partial JSON from Moonshot
                        potential_full_string = leading_text_for_moonshot + raw_llm_output_str
                        try:
                            parsed_llm_response = json.loads(potential_full_string)
                        except json.JSONDecodeError:
                            try:
                                parsed_llm_response = ast.literal_eval(potential_full_string)
                            except (ValueError, SyntaxError):
                                pass # Parsing failed even with Moonshot-specific handling
                    
                    if parsed_llm_response is None: # If all parsing attempts failed
                        print(f"总结员Attempt {attempts + 1} failed: Could not parse LLM output string: '{raw_llm_output_str}'")
                        attempts += 1
                        time.sleep(1)
                        continue
            
            if not isinstance(parsed_llm_response, dict):
                print(f"总结员Attempt {attempts + 1} failed: Parsed LLM output is not a dictionary: {parsed_llm_response}")
                attempts += 1
                time.sleep(1)
                continue

            if "error" in parsed_llm_response:
                print(f"LLM returned an explicit error: {parsed_llm_response}")
                print(f"Original inputs to LLM (exam_text):\n{exam_text}")
                return False, parsed_llm_response.get("error", "LLM returned an error structure."), current_cost # Adhering to original error return style

            if "answer" not in parsed_llm_response:
                print(f"总结员Attempt {attempts + 1} failed: LLM response dictionary missing 'answer' key: {parsed_llm_response}")
                attempts += 1
                time.sleep(1)
                continue

            final_answer_value = parsed_llm_response["answer"]

            if question_type in ("单选", "多选"):
                if parsed_options_list:  # Validate only if options were provided
                    is_valid_option = False
                    if question_type == "多选":
                        if isinstance(final_answer_value, list):
                            is_valid_option = all(opt_val in parsed_options_list for opt_val in final_answer_value)
                        else: # Expected a list for multi-choice answer
                            print(f"Warning: For multi-choice '{question_line}', expected a list answer, got {type(final_answer_value)}: {final_answer_value}")
                            is_valid_option = False 
                    else:  # 单选
                        is_valid_option = final_answer_value in parsed_options_list
                    
                    if not is_valid_option:
                        print(f"总结员Attempt {attempts + 1} failed: Answer '{final_answer_value}' is not in the valid options {parsed_options_list} for question type '{question_type}'. Question: {question_line}")
                        attempts += 1
                        time.sleep(1)
                        continue
                
                return final_answer_value, 0, current_cost 
            
            else: # For "数值", "文本", or default type (with assumption)
                return final_answer_value, 0, current_cost

        except Exception as e:
            attempts += 1
            print(f"总结员Attempt {attempts} failed with exception: {e} (Question: {question_line}). Retrying...")
            if attempts >= max_attempts:
                print(f"总结员Attempt {attempts} (Max attempts reached) failed: {e} (Question: {question_line}).")
                raise e # Propagate the error if max attempts are reached
            time.sleep(1) # Wait before retrying

    # This part should ideally be unreachable if max_attempts always leads to an exception.
    # However, as a fallback, indicate failure.
    print(f"总结员 failed after {max_attempts} attempts for question: {question_line}")
    return "信息不足", f"总结失败，达到最大尝试次数 ({max_attempts})", current_cost


def qwen_interior_data_collection_top_asking(client, message, indicator_type, used_tool, sub_indicators_info=None,ai = 'qwen'):
    temp_record = {}
    retry = 0
    found_answer = False
    # code = "" # 'code' variable was initialized but not used in the original function, so omitting.
    cost = 0
    input_usage = 0
    output_usage = 0
    temp_record["message"] = message
    while retry < 5:
        try:
            response = qwen_client.chat.completions.create(
                model="qwen-max-latest",  # Changed to Qwen model
                temperature=0.01,
                messages=message,
                timeout=240
            )

            this_answer = response.choices[0].message.content
            # Qwen API might return usage differently or not at all in the same format.
            # Assuming response.usage exists and has prompt_tokens and completion_tokens for now.
            # If not, this part will need adjustment based on actual Qwen API response structure.
            if hasattr(response, 'usage') and response.usage is not None:
                input_usage += response.usage.prompt_tokens
                output_usage += response.usage.completion_tokens
            else: # Fallback or logging if usage info is not available as expected
                print("Warning: Usage information not found in Qwen response.")


            temp_record["message"] = message
            temp_record["answer"] = this_answer

            if "信息不足" not in this_answer:
                found_answer = True
            break
        except Exception as e:
            print(e)
            if "timed out" in str(e).lower() or "timeout" in str(e).lower(): # Made timeout check case-insensitive
                retry += 1
                if retry >=3:
                    raise Exception(f"Qwen API timed out after {retry} retries: {e}")
                print(f"Qwen API call timed out, retrying ({retry}/3)...")
                time.sleep(1) # Adding a small delay before retrying on timeout
                continue

            # If not a timeout error, try one more time with the same model, similar to original logic
            try:
                print("子指标收集失败，尝试使用ds模型进行重试: deepseek-v3") # Updated print message
                response = qwen_client.chat.completions.create(
                    model="deepseek-v3",  # Changed to Qwen model
                    temperature=0.01,
                    messages=message,
                    timeout=240
                )

                this_answer = response.choices[0].message.content
                if hasattr(response, 'usage') and response.usage is not None:
                    input_usage = response.usage.prompt_tokens # Resetting usage for this attempt
                    output_usage = response.usage.completion_tokens
                else:
                    print("Warning: Usage information not found in Qwen response on retry.")
                    input_usage = 0 # Ensure these are defined
                    output_usage = 0

                temp_record["message"] = message
                temp_record["answer"] = this_answer

                if "信息不足" not in this_answer:
                    found_answer = True
                break
            except Exception as error:
                # If the retry also fails, raise the error
                print("Ds兜底失败，尝试使用minimax")
                response = siliconflow_client.chat.completions.create(
                    model="MiniMaxAI/MiniMax-M1-80k",
                    temperature=0.01,
                    messages=message,
                    timeout=240
                )
                this_answer = response.choices[0].message.content
                if hasattr(response, 'usage') and response.usage is not None:
                    input_usage = response.usage.prompt_tokens # Resetting usage for this attempt
                    output_usage = response.usage.completion_tokens
                else:
                    print("Warning: Usage information not found in Qwen response on retry.")
                    input_usage = 0 # Ensure these are defined
                    output_usage = 0

                temp_record["message"] = message
                temp_record["answer"] = this_answer

                if "信息不足" not in this_answer:
                    found_answer = True
                break
    
    cost += input_usage / 1000 * 0.0024 + output_usage / 1000 * 0.0096
    # record_money(cost) # This function call was not in the original provided snippet, so omitting.
    return temp_record, used_tool, found_answer, cost


