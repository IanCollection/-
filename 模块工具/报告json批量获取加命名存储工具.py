import base64
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import requests

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pymysql

from logic_folder.数据库表格 import Company, Reports, Indicators, EntityEvd

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


def find_company_details(company_name, company_df):
    """
    公司code找公司名

    """
    company_code = company_df[company_df['code'] == company_name]['company'].values
    company_type = company_df[company_df['code'] == company_name]['industry'].values

    if company_name.size > 0:
        return company_code[0], company_type[0]  # 返回找到的第一个code
    else:
        return None  # 如果没有找到，则返回'Unknown'


def get_one_report_json(report_collection):
    """
    :param report_data: 一条报告信息
    :return:
    """
    retry = 0
    while retry < 1:
        try:
            report_data = report_collection['report_data']
            url = 'https://ibondtest.deloitte.com.cn/readPdfGetJson'  # 外网接口链接

            res = requests.get(url, params=report_data,timeout= 1800)
            res_data = res.json()

            return {'report_json': res_data['json_data'], 'report_name': report_data['fileName'][:-4],
                    'entity_code': report_collection['entity_code'], 'entity_name': report_collection['entity_name'],
                    'entity_type': report_collection['entity_type'], 'fCode': report_data['fCode']}
        except Exception as e:
            print(f"获取报告json失败，文件名：{report_collection['report_data']['fileName']}")
            print(e)
            retry += 1

    return None


def get_report_list(entity_info):
    """
    :param entity_name: 目标主体的名称
    :param entity_year: 获取报告的年份
    :return: 返回该主体对应的年份的报告列表
    """
    entity_name = entity_info['entity_name']
    entity_year = entity_info['entity_year']
    retry = 0
    while retry < 5:
        try:

            url = 'https://ibondtest.deloitte.com.cn/readPdfGetJson'  # 外网接口链接
            data = {
                'entity_name': entity_name,
                'entity_year': entity_year,
            }
            res = requests.post(url, data=data)
            res_data = res.json()
            # print(res.text)
            return {'report_data': res_data['report_data'], 'entity_name': entity_name,
                    'entity_code': entity_info['entity_code'], 'entity_type': entity_info['entity_type']}
        except Exception as e:
            print(f"获取附属报告清单失败，公司名：{entity_name}")

            print(e)
            retry += 1
            time.sleep(5)


def get_report_json(file):
    def sort_by_report_name(all_reports):
        priority_report = []
        normal_report = []
        worst_report = []
        annual_report_pattern = r'(年度报告|年度审计报告|财务报告)'
        for report in all_reports:
            if re.search(annual_report_pattern, report) and '债券' not in report:
                priority_report.append(report)
            elif '募集说明书' in report:
                worst_report.append(report)
            else:
                normal_report.append(report)
        return {1: priority_report, 0: normal_report, -1: worst_report}

    base_dir = 'uploads'
    file = os.path.join(base_dir, file)
    company_df = pd.read_excel(file, sheet_name=0)  # 公司主体维度标：公司code与公司名字、类别等公司信息的表
    # 确保公司名列的内容是字符串类型，以便进行匹配
    company_df['entity_name'] = company_df['entity_name'].astype(str)
    company_df['entity_code'] = company_df['entity_code'].astype(str)
    company_df['industry'] = company_df['industry'].astype(str)
    company_df["year"] = company_df["year"].astype(str)

    # 用时间年月日时间来命名文件夹
    folder_name = "OCR报告文件夹"
    new_folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    to_request_list = []
    entity_report_dict = {}
    try:
        with ProcessPoolExecutor() as executor:
            futures = []

            for index, row in company_df.iterrows():
                entity_code = row['entity_code']
                entity_name = row['entity_name']
                entity_type = row['industry']
                entity_year = row['year']
                entity_info = {'entity_code': entity_code, 'entity_name': entity_name, 'entity_year': entity_year,
                               'entity_type': entity_type}
                futures.append(executor.submit(get_report_list, entity_info))
            for future in as_completed(futures):
                try:
                    result_collection = future.result()
                    report_data = result_collection['report_data']
                    entity_name = result_collection['entity_name']
                    entity_code = result_collection['entity_code']
                    entity_type = result_collection['entity_type']
                    exist_fCode_list = []
                    for each_report in report_data:
                        if each_report['fCode'] in exist_fCode_list or each_report['isDeal'] == 2:
                            continue
                        to_request_list.append(
                            {'entity_code': entity_code, 'entity_name': entity_name, 'entity_type': entity_type,
                             'report_data': each_report})
                        exist_fCode_list.append(each_report['fCode'])

                except Exception as e:
                    print(f"公司：{entity_name}  年份：{entity_year}   获取文件列表失败")

                    print(e)

        total_folders = []
        with ProcessPoolExecutor() as executor:
            futures = []
            task_count = 0
            total_task = len(to_request_list)
            for each_report in to_request_list:
                entity_code = each_report['entity_code']
                entity_name = each_report['entity_name']
                entity_type = each_report['entity_type']
                report_name = each_report['report_data']['fileName'][:-4]
                report_name = report_name.replace('-', '_')

                folder_name = entity_year + '-' + entity_name + '-' + entity_type + '-' + report_name + '-' + '标准' + '-' + entity_code
                new_dir_path = os.path.join(new_folder_path, folder_name)

                if os.path.exists(new_dir_path):
                    print(f"文件已存在：{new_dir_path}")
                    task_count += 1
                    if entity_code not in entity_report_dict.keys():
                        entity_report_dict[entity_code] = [folder_name]
                    else:
                        entity_report_dict[entity_code].append(folder_name)
                    print(f"\r已完成{task_count}/{total_task}", end='')
                    total_folders.append(folder_name)
                    continue
                total_folders.append(folder_name)
                futures.append(executor.submit(get_one_report_json, each_report))

            for future in as_completed(futures):

                try:
                    result_collection = future.result()
                    if not result_collection:
                        continue
                    entity_code = result_collection['entity_code']
                    entity_name = result_collection['entity_name']
                    entity_type = result_collection['entity_type']
                    json_data = result_collection['report_json']
                    report_name = result_collection['report_name']
                    report_name = report_name.replace('-', '_')

                    folder_name = entity_year + '-' + entity_name + '-' + entity_type + '-' + report_name + '-' + '标准' + '-' + entity_code
                    new_dir_path = os.path.join(new_folder_path, folder_name)
                    if entity_code not in entity_report_dict.keys():
                        entity_report_dict[entity_code] = [folder_name]
                    else:
                        entity_report_dict[entity_code].append(folder_name)
                    if not os.path.exists(new_dir_path):
                        os.makedirs(new_dir_path)
                    total_folders.append(folder_name)
                    file_name = entity_year + '-' + entity_name + '-' + entity_type + '-' + report_name + '-' + '标准' + '-' + entity_code + '.json'
                    file_path = os.path.join(new_dir_path, file_name)
                    with open(file_path, 'w') as f:
                        f.write(json_data)
                    task_count += 1
                    print(f"\r已完成{task_count}/{total_task}", end='')
                except Exception as e:
                    print(e)

        report_indicator_list = []

        db, engine = new_create_db_session()

        for index, row in company_df.iterrows():
            company_code = row['entity_code']
            if company_code not in entity_report_dict.keys():
                continue
            all_reports = entity_report_dict[company_code]
            all_reports = sort_by_report_name(all_reports)
            all_indicators = db.query(EntityEvd).filter(EntityEvd.entity_code == company_code).all()
            for level,level_report_list in all_reports.items():
                for old_report_name in level_report_list:
                    year = None
                    report_name = "知识图谱_" + old_report_name
                    sub_report_indicator_list = {"report_name": report_name, "indicators": [], "company_code": company_code,"level":level}
                    for indicator in all_indicators:
                        indicator_code = indicator.evd_code
                        year = row['year']
                        sub_report_indicator_list["indicators"].append(indicator_code)
                    sub_report_indicator_list["year"] = str(year)
                    report_indicator_list.append(sub_report_indicator_list)
        data = {"report_folder_list": total_folders, "indicator_collection_list": report_indicator_list}

        db.close()
        engine.dispose()
        return True, data
    except Exception as e:
        print(e)
        return False, str(e)


if __name__ == '__main__':
    info_collection = {'entity_code': '123', 'entity_name': '中国人寿财产保险股份有限公司', 'entity_year': '2023',
                       'entity_type': '123'}
    a = get_report_list(info_collection)
    print("haha")


    # report_data = {'fileCode': 'F1712148744365', 'fileName': '中国国际航空股份有限公司2024年度第二期超短期融资券募集说明书.PDF', 'url': '/102/2023/3017/F1712148744365.PDF', 'fCode': '102024002', 'type': '募集说明书', 'fileYear': '2023', 'publish_time': '2024-03-26', 'isDeal': 0}
    # report_collection = {'entity_code': '12345', 'entity_name': '中国国际航空股份有限公司', 'entity_type': '交通运输',
    #                                         'report_data': report_data}
    # a = get_one_report_json(report_collection)
    # print("jaja")
