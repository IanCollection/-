import json

import pandas as pd

from logic_folder.表格处理包 import create_tables_from_data, same_column_num_check


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


def simplified_break_table_combination(pre_table, current_table):
    same_header = False

    if pre_table["header"] == current_table["header"]:
        same_header = True

    ##表格拼接
    if same_header:
        pre_table["values"].extend(current_table["values"])
        pre_table["key_index"].extend(current_table["key_index"])
        return pre_table
    else:
        for j in range(len(current_table["header"])):
            if isinstance(current_table["header"][j][0], str):
                pre_table["values"].append(current_table["header"][j])
                pre_table["key_index"].append(current_table["header"][j][0])
        pre_table["values"].extend(current_table["values"])
        pre_table["key_index"].extend(current_table["key_index"])

        return pre_table


def simplified_material_process(pure_ocr_result):
    current_table_list = []
    current_section = ""
    pre_table_section = ""
    last_table_section = False
    expand_table_check = False
    total_pages = len(pure_ocr_result)
    final_table_list = []
    pre_table = None

    for page_num, page in enumerate(pure_ocr_result):

        print(f"\rProgress: {page_num + 1}/{total_pages} pages processed.", end='')
        
        for section_index, section in enumerate(page):
            try:
                # if "type" not in section.keys():
                if "type" not in section.keys() or section["type"] == "plain":
                    ##文字段落
                    for line_index, each_line in enumerate(section["lines"]):
                        text = each_line["text"]

                        if current_section != "" and len(current_table_list) > 0:
                            final_table_list.append(current_table_list[0])
                            current_table_list = []
                        current_section = text

                else:
                    ##表格段落
                    ##先处理table
                    current_holding_table_list = create_tables_from_data(section["table_cells"])

                    if len(current_holding_table_list) > 0:
                        header = current_holding_table_list[0]["header"]
                    else:
                        continue
                    og_header = []
                    new_header = []
                    consistent_title = ""
                    consistent_unit = ""

                    table_title, unit = "NA", "NA"
                    if len(current_table_list) > 0:
                        pre_table = current_table_list[0]
                    else:
                        pre_table = None
                    for each_table in current_holding_table_list:
                        if og_header and each_table["header"] == og_header:
                            each_table["header"] = new_header
                            table_title = consistent_title
                            unit = consistent_unit
                        if each_table["title"] is None:
                            each_table["title"] = table_title
                        elif table_title != "NA":
                            each_table["title"] = table_title + ":" + each_table["title"]
                        each_table["unit"] = unit

                        if pre_table and same_column_num_check(pre_table, each_table):
                            if not pre_table:
                                pre_table = each_table
                            else:

                                current_table = each_table
                                og_header = current_table["header"]
                                pre_table = simplified_break_table_combination(pre_table, current_table)
                                new_header = pre_table["header"]
                                current_table_list[-1] = pre_table

                        else:

                            current_table_list.append(each_table)
            except Exception as e:
                print(e)
                current_table_list = []
                continue
    if current_section != "":
        final_table_list.append(current_table_list[0])

    # 处理每个表格

    df = pd.DataFrame(final_table_list)

    process_dict_list = []

    for index, row in df.iterrows():
        # 先找到指标code
        code = None
        indicator_type = None
        back_up_info = None
        if len(row["values"]) < 4:
            continue
        code_material = row["values"][0]
        if len(code_material) == 2 and '基础层 code' == code_material[0]:
            code = code_material[1]
        indicator_type_material = row["values"][2]
        if len(indicator_type_material) == 2 and '补录需求' == indicator_type_material[0]:
            indicator_type = indicator_type_material[1]
            if "文字" in indicator_type:
                indicator_type = "文字"
            else:
                indicator_type = "数字"
        back_up_info_material = row["values"][-1]
        if len(back_up_info_material) == 2 and '备注' == back_up_info_material[0]:
            back_up_info = back_up_info_material[1]

        if code and indicator_type and back_up_info:
            process_dict_list.append({"code": code, "indicator_type": indicator_type, "back_up_info": back_up_info})

    df = pd.DataFrame(process_dict_list)
    return df


def knowledge_graph_output(file, company_name="随意"):
    file_path = '/Users/zinozhang/Documents/Deloitte/Task21-内评数据/指标描述结构化提取/数据准备报告定稿_2.pdfp_随意.json'
    with open(file_path, "r") as f:
        ocr_json = json.load(f)
    true_ocr_result = pure_ocr_result(ocr_json)

    df = simplified_material_process(true_ocr_result)
    df.to_excel("数据准备报告定稿_提纯版.xlsx")
    return None
