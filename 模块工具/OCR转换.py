import re
import time
import base64
from datetime import datetime

import pandas as pd
import requests
import PyPDF2
import json
import os
import shutil

__all__ = [
    'get_handle_ocr_result'
]


# %%
def requests_ocr(entity_name, file, max_retries=3):
    url = 'https://ibondtest.deloitte.com.cn/read_pdf'  # 接口链接
    pdf_file = open(file, 'rb').read()
    data4_0 = {
        "ocrEntityName": entity_name,
        'ocrPdfData': base64.b64encode(pdf_file).decode(),
    }
    for _ in range(max_retries):
        try:
            response = requests.post(url, data=data4_0)
            # response.raise_for_status()
            json_data = response.json()
            json_file = json_data.get('json')
            if json_file:
                return json_file
            else:
                raise ValueError('No JSON data in response')
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"Error: {e}")
            time.sleep(0.1)

    raise ConnectionError(f'Failed after {max_retries} retries')


def process_full_pdf(
        pdf_path: str,
        pdf_name: str,
        json_name: str
):
    now = datetime.now()
    formatted_now = now.strftime('%Y%m%d%H%M%S%f')
    tmp_pdf_path = f"../原始文件/{formatted_now}{pdf_name}temp_pdf"
    if not os.path.exists(tmp_pdf_path):
        os.makedirs(tmp_pdf_path)

    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    all_pages_json = {}
    total_pages = len(pdf_reader.pages)

    print("OCR开始处理:")
    for page_num in range(total_pages):
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_num])

        temp_pdf_path = tmp_pdf_path + f'/page_{page_num}.pdf'
        with open(temp_pdf_path, 'wb') as temp_pdf:
            pdf_writer.write(temp_pdf)

        try:
            name = pdf_name.split(".")[0] + str(page_num) + formatted_now
            page_json = requests_ocr(pdf_name.split(".")[0] + str(page_num) + formatted_now, temp_pdf_path)
            data = json.loads(page_json)
            print(f"\r当前处理:{pdf_name}, OCR进度: {page_num + 1}/{total_pages} pages processed.", end='')
            all_pages_json[page_num] = data['0']

        except ConnectionError as ce:
            print(f"Connection error: {ce}")

        os.remove(temp_pdf_path)

    directory_to_remove = os.path.join(tmp_pdf_path)
    shutil.rmtree(directory_to_remove)
    pdf_file.close()

    # with open(pdf_path + json_name, 'w', encoding='utf-8') as json_file:
    #     json.dump(all_pages_json, json_file, ensure_ascii=False, indent=4)

    return pd.DataFrame(all_pages_json)


def create_parse_json():
    return ParseJSON()


def get_json_results(file_path, company_name, file_type):
    HJ = HandleJson(file_path=file_path)
    ret = HJ.get_results(company=company_name, file_type=file_type)
    return ret


class ParseJSON:
    def __init__(self) -> None:
        pass

    @classmethod
    def find_rows_with_small_distance_x(cls, excel_data, row_index):
        try:
            target_x_value = excel_data.loc[row_index, 'rt'][0]
            target_y_value = excel_data.loc[row_index, 'rt'][1]

            # Initialize variables to find the closest row meeting the criteria
            min_idx_diff = float('inf')
            closest_row_index = None
            # Iterate over the DataFrame to find the row meeting the specified criteria
            for idx, row in excel_data.iterrows():

                if idx != row_index:  # Ensure the row is after the given row
                    x_diff = abs(row['lt'][0] - target_x_value)
                    y_diff = abs(row['rt'][1] - target_y_value)
                    size = min(excel_data.at[idx, 'size'], excel_data.at[row_index, 'size'])
                    # Check if the row meets all the criteria
                    if x_diff < size and y_diff < size:

                        idx_diff = idx - row_index
                        if idx_diff < min_idx_diff:
                            min_idx_diff = idx_diff
                            closest_row_index = idx

            # Return the closest row if found
            if closest_row_index is not None:
                return excel_data.loc[[closest_row_index]]
            else:
                return None  # Return an empty DataFrame if no row meets the criteria

        except IndexError:
            return "Row index out of range."
        except KeyError:
            return "Incorrect column name."
        except Exception as e:
            return str(e)

    @classmethod
    def find_rows_with_same_lt_x(cls, data, row_index):

        try:
            target_x_value = data.loc[row_index, 'lt'][0]
            target_y_value = data.loc[row_index, 'lt'][1]
            target_text = data.loc[row_index, 'text']
            # Initialize variables to find the closest row meeting the criteria
            min_idx_diff = float('inf')
            closest_row_index = None

            # Iterate over the DataFrame to find the row meeting the specified criteria
            for idx, row in data.iterrows():
                if idx > row_index:  # Ensure the row is after the given row
                    x_diff = abs(row['lt'][0] - target_x_value)
                    y_diff = abs(row['lb'][1] - target_y_value)
                    text = data.loc[idx, 'text']
                    contains_number = bool(re.search(r'\d', target_text))
                    if contains_number is True:
                        size = max(data.at[idx, 'size'], data.at[row_index, 'size'])
                    else:
                        size = min(data.at[idx, 'size'], data.at[row_index, 'size'])
                    # Check if the row meets all the criteria
                    if x_diff < size and y_diff < 1.5 * size:
                        idx_diff = idx - row_index
                        if idx_diff < min_idx_diff:
                            min_idx_diff = idx_diff
                            closest_row_index = idx

            # Return the closest row if found
            if closest_row_index is not None:
                return data.loc[[closest_row_index]]
            else:
                return None  # Return an empty DataFrame if no row meets the criteria

        except IndexError:
            return "Row index out of range."
        except KeyError:
            return "Incorrect column name."
        except Exception as e:
            return str(e)

    @classmethod
    def update_positions(cls, data, reference_row_index, mer_index):
        # Update position for each corner
        data.at[reference_row_index, 'lt'] = (
            min(data.at[reference_row_index, 'lt'][0], data.at[mer_index, 'lt'][0]),
            max(data.at[reference_row_index, 'lt'][1], data.at[mer_index, 'lt'][1])
        )

        data.at[reference_row_index, 'lb'] = (
            min(data.at[reference_row_index, 'lb'][0], data.at[mer_index, 'lb'][0]),
            min(data.at[reference_row_index, 'lb'][1], data.at[mer_index, 'lb'][1])
        )

        data.at[reference_row_index, 'rt'] = (
            max(data.at[reference_row_index, 'rt'][0], data.at[mer_index, 'rt'][0]),
            max(data.at[reference_row_index, 'rt'][1], data.at[mer_index, 'rt'][1])
        )

        data.at[reference_row_index, 'rb'] = (
            max(data.at[reference_row_index, 'rb'][0], data.at[mer_index, 'rb'][0]),
            min(data.at[reference_row_index, 'rb'][1], data.at[mer_index, 'rb'][1])
        )

    @classmethod
    def filter_title(cls, data, row_index):
        if any(char.isdigit() for char in data.at[row_index, "text"]):
            data.at[row_index, "is_title"] = 0
        return True

    @classmethod
    def clean_excel_data(cls, data, row_index):
        text = data.at[row_index, 'text']
        if len(text) == 1 or text.isdigit() or (len(text) == 1 and text.isalpha()):
            data.drop(row_index, inplace=True)
            data.reset_index(drop=True)
        return True

    @classmethod
    def wide_merge(cls, data, row_index):
        try:
            target_left_value = data.loc[row_index, 'lt'][0]
            target_right_value = data.loc[row_index, 'rt'][0]
            target_bottom_value = data.loc[row_index, 'lb'][1]
            target_top_value = data.loc[row_index, 'lt'][1]

            min_lr_diff = float('inf')
            closest_row_index = None

            for idx, row in data.iterrows():
                if idx != row_index:
                    #                 left_diff = abs(row['lt'][0] - target_left_value)
                    #                 right_diff = abs(row['rt'][0] - target_right_value)
                    #                 bottom_diff = abs(row['lb'][1] - target_bottom_value)
                    #                 top_diff = abs(row['lt'][1] - target_top_value)
                    bp_diff = abs(row['lb'][1] - target_top_value)
                    size = max(data.at[idx, 'size'], data.at[row_index, 'size'])
                    if bp_diff < size:
                        lr_diff = abs(target_right_value - row['lt'][0])
                        if lr_diff < min_lr_diff:
                            min_lr_diff = lr_diff
                            if min_lr_diff < 3 * size:
                                closest_row_index = idx

            # Return the closest row if found
            if closest_row_index is not None:
                return data.loc[[closest_row_index]]
            else:
                return None  # Return an empty DataFrame if no row meets the criteria

        except IndexError:
            return "Row index out of range."
        except KeyError:
            return "Incorrect column name."
        except Exception as e:
            return str(e)

    @classmethod
    def link_title(cls, excel_data, row_index):
        try:

            target_left_value = excel_data.loc[row_index, 'lt'][0]
            target_right_value = excel_data.loc[row_index, 'rt'][0]
            target_bottom_value = excel_data.loc[row_index, 'lb'][1]
            target_top_value = excel_data.loc[row_index, 'lt'][1]

            target_title = ""

            center_position = (excel_data.loc[row_index, 'lt'][0] + excel_data.loc[row_index, 'rt'][0]) / 2

            min_distance = float('inf')

            min_distance_idx = None

            dis_target_title = ""

            for idx, row in excel_data.iterrows():
                if row["is_title"] == 1:
                    left_diff = abs(row['lt'][0] - target_left_value)
                    right_diff = abs(row['rt'][0] - target_right_value)
                    bottom_diff = abs(row['lb'][1] - target_bottom_value)
                    top_diff = abs(row['lt'][1] - target_top_value)
                    other_center_position = (row['lt'][0] + row['rt'][0]) / 2
                    center_diff = abs(center_position - other_center_position)

                    distance = abs(center_position - other_center_position)

                    size = max(excel_data.at[idx, 'size'], excel_data.at[row_index, 'size'])

                    if left_diff < size or right_diff < size or bottom_diff < size or top_diff < size or center_diff < size:
                        #                     if target_title is not None:
                        #                         target_title = target_title + ";" + str(idx)
                        #                     else:
                        if target_title == "":
                            target_title = target_title + str(idx)
                        else:
                            target_title = target_title + ";" + str(idx)

                    elif distance < min_distance:
                        min_distance = distance
                        min_distance_idx = idx

            dis_target_title = dis_target_title + str(min_distance_idx)
            excel_data.at[row_index, "target_title"] = target_title
            excel_data.at[row_index, "dis_target_title"] = dis_target_title
            return True
        except IndexError:
            return "Row index out of range."
        except KeyError:
            return "Incorrect column name."
        except Exception as e:
            return str(e)

    @classmethod
    def process_single_page(cls, df):
        # 假设 l_df 是您的 DataFrame
        # 假设 find_rows_with_small_distance_x 和 find_rows_with_same_lt_x 是您已定义的函数
        # 初始化 'to_delete' 和 'is_title' 列
        df['to_delete'] = False
        df['is_title'] = 1

        # 第一个循环
        for index, _ in df.iterrows():
            if not df.at[index, 'to_delete']:
                result = ParseJSON.find_rows_with_small_distance_x(df, index)
                if isinstance(result, pd.DataFrame) and not result.empty:
                    result_index = result.index[0]
                    df.at[index, 'text'] += result.iloc[0]['text']
                    ParseJSON.update_positions(df, index, result_index)
                    df.at[result.index[0], 'to_delete'] = True
                    df.at[index, 'is_title'] = 0  # 因为参与了合并
                    df.at[result.index[0], 'is_title'] = 0  # 被标记为删除

        # 删除标记为删除的行，并重置索引
        df = df[df['to_delete'] != True].drop(['to_delete'], axis=1).reset_index(drop=True)

        df['to_delete'] = False
        # 第二个循环
        index = 0
        while index < len(df):
            merge_indices = []
            current_index = index
            while True:
                result = ParseJSON.find_rows_with_same_lt_x(df, current_index)

                merge_indices.append(current_index)
                if result is None:
                    break
                # print("================")
                # print(result)
                # print(type(result.index))
                current_index = result.index[0]

            if merge_indices:
                combined_text = ' '.join(df.at[idx, 'text'] for idx in merge_indices)
                df.at[index, 'text'] = combined_text
                for idx in merge_indices:
                    ParseJSON.update_positions(df, index, idx)
                    df.at[idx, 'to_delete'] = True if idx != index else df.at[idx, 'to_delete']
                    if len(merge_indices) > 1:
                        df.at[idx, 'is_title'] = 0
            index += 1

        # 删除标记为删除的行，并重置索引
        df = df[df['to_delete'] != True].drop(['to_delete'], axis=1).reset_index(drop=True)

        for index, _ in df.iterrows():
            ParseJSON.clean_excel_data(df, index)

        df['to_delete'] = False
        # df['is_dirty'] = df.apply(lambda row: (row["lt"][1] - row["lb"][1]) < 2 * row["size"], axis=1)
        return df

    def get_results(self, df):
        return ParseJSON.process_single_page(df)


class HandleJson:
    PJ = create_parse_json()

    def __init__(
            self, file_path,
    ) -> None:
        self.file_paths = file_path
        # 先读取路径，之后可以处理任意多的公司（每一个pdf）

    @classmethod
    def extract_text_info(
            cls,
            line: dict
    ) -> dict:

        pos = line['char_positions']
        char_num = len(pos)
        lb = (pos[0][0], pos[0][1])
        rb = (pos[char_num - 1][2], pos[char_num - 1][3])
        rt = (pos[char_num - 1][4], pos[char_num - 1][5])
        lt = (pos[0][6], pos[0][7])
        size = lt[1] - lb[1]
        return {"text": line['text'], "size": size, "lb": lb, "rb": rb, "rt": rt, "lt": lt}

    def _handle_company(self, company, file_type):

        df = process_full_pdf(self.file_paths, company + '.pdf', 'p_' + company + '.json')
        if file_type == '年报':
            return df
        elif file_type == 'ESG报告':
            return HandleJson.parse_and_warp_single_company(df=df)
        else:
            raise ValueError(f'Error file type {file_type}')

    def get_results(self, company, file_type):
        data = self._handle_company(company=company, file_type=file_type)
        print(f"OCR扫描完成, 报告名称:{company}")
        return data

    @classmethod
    def parse_and_warp_single_company(
            cls,
            df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        主要的逻辑是一拆分一边解析
        """
        pdf_page_number = len(df.columns)
        pdf_results = pd.DataFrame(index=['code', 'duration', 'message', 'result', 'version'])
        a4_width = 1191

        for num in range(pdf_page_number):
            # 读取单页
            original_page = df.loc["result", num]
            width = original_page['width']
            height = original_page['height']
            page_results = {"angle": 0, "width": width, "height": height}
            tables = []
            for page_content in original_page['tables']:  # list
                left_page = []
                right_page = []
                content_type = page_content['type']
                lines = []
                tmp_tables = []
                if content_type == 'plain':
                    # pass
                    for line in page_content['lines']:
                        if line['position'][0] < a4_width and line['text'] != '':
                            try:
                                left_page.append(HandleJson.extract_text_info(line))
                            except:
                                print(line)
                                continue
                        elif a4_width <= line['position'][0] < a4_width * 2 and line['text'] != '':
                            right_page.append(HandleJson.extract_text_info(line))
                        else:
                            # raise ValueError("Unexpected content_type")
                            continue

                if content_type in ['table_with_line', 'table_without_line']:
                    tmp_tables.append(page_content)

                    # cells = page_content['table_cells']
                    # for cell in cells:
                    #     for line in cell['lines']:
                    #         if line['position'][0] < a4_width:
                    #             try:
                    #                 left_page.append(HandleJson.extract_text_info(line))
                    #             except:
                    #                 print(line)
                    #                 continue
                    #         elif a4_width <= line['position'][0] < a4_width * 2:
                    #             right_page.append(HandleJson.extract_text_info(line))
                    #         else:
                    #             # raise ValueError("Unexpected content_type")
                    #             continue

                left_page = pd.DataFrame(left_page)
                right_page = pd.DataFrame(right_page)

                left_page = HandleJson.PJ.get_results(left_page)
                right_page = HandleJson.PJ.get_results(right_page)

                for row_num in left_page.index:
                    lb = left_page.loc[row_num, "lb"]
                    rb = left_page.loc[row_num, 'rb']
                    rt = left_page.loc[row_num, "rt"]
                    lt = left_page.loc[row_num, 'lt']
                    # is_dirty = left_page.loc[row_num, 'is_dirty']
                    lines.append({"text": left_page.loc[row_num, "text"],
                                  "position": list(lb + rb + rt + lt)})

                for row_num in right_page.index:
                    lb = right_page.loc[row_num, "lb"]
                    rb = right_page.loc[row_num, 'rb']
                    rt = right_page.loc[row_num, "rt"]
                    lt = right_page.loc[row_num, 'lt']
                    # is_dirty = right_page.loc[row_num, 'is_dirty']
                    lines.append({"text": right_page.loc[row_num, "text"],
                                  "position": list(lb + rb + rt + lt)})
                if content_type not in ['table_with_line', 'table_without_line']:
                    tables.append({"lines": lines,
                                   "position": -1,
                                   "type": "plain"})

                tables += tmp_tables

                page_results['tables'] = tables

            pdf_results[num] = [200, 805, "success", page_results, "v2.0.0"]

        return pdf_results


def get_handle_ocr_result(file_path, company_name, file_type):
    """
    file_type: 年报，ESG报告

    """
    return get_json_results(file_path=file_path, company_name=company_name, file_type=file_type)


if __name__ == "__main__":
    get_handle_ocr_result('', '', '', '年报')
