import argparse
import json
import os
import re
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

# Disable verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

from logic_folder.数据库表格 import Base, Reports, Company, Vector, EntityEvd, Indicators, Sentences, FailReports, \
    Missions, IndicatorsResults
from logic_folder.检索包 import group_searching_within_report_v2
from logic_folder.表格处理包 import pre_road_map_embedding_convert
from operation_folder.指标批量收集 import single_data_collection
from operation_folder.指标表录入 import indicators_sheet_to_db
from operation_folder.文件录入 import report_to_db
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
import pymysql

from 模块工具.报告json批量获取加命名存储工具 import get_report_json


pymysql.install_as_MySQLdb()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')


# 配置数据库连接
def create_db_engine(echo=False):
    engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}', echo=echo,isolation_level="READ COMMITTED", pool_pre_ping=True)
    return engine


def new_create_db_session():
    # 创建数据库引擎
    engine = create_db_engine(False)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session, engine


##创建数据库和表格
def create_db_and_tables():
    engine = create_db_engine(True)
    Base.metadata.create_all(engine)
    return True


if __name__ == "__main__":

    # 创建库
    create_db_and_tables()
    # ------------------------------------------------------------------------------------------------------------------------------------

    # 录入指标表
    # indicator_file = '指标表/数据参数准备（述职报告）_v1.xlsx'

    jianchen_df = pd.read_excel('/Users/linxuanxuan/Desktop/zsx_独立董事指标跑批/uploads/指标表/Copy of 年报和独立董事报告use.xlsx')

    tenure_df = pd.read_excel('uploads/指标表/任职周期判断结果_符合条件0620.xlsx')

    # 指标表
    # indicator_file = '指标表/中上协独董评价_述职报告指标0616_v3（周期）.xlsx'
    # indicator_file = '指标表/Copy of 中上协独董评价_重点优化指标 (5)(1).xlsx'

    indicator_file = '指标表/Copy of 中上协独董评价_述职报告指标0620.xlsx'

    # # indicator_file = '/Users/zinozhang/Desktop/v7.1入库表.xlsx'
    year = 2024
    # indicators_sheet_to_db(indicator_file, year)

    # ------------------------------------------------------------------------------------------------------------------------------------

    start_time = datetime.now()
    # 报告入库
    db, engine = new_create_db_session()


    root_folder = "0616独立董事述职报告json"

    # 独董信息excel(从上一个步骤的代码运行后直接用）
    # df_file = 'uploads/指标表/test_indicator_sample0602_50独董.xlsx'

    df_file = "uploads/指标表/Copy 0612_DTT筛选独董评价名单_v6_v2.xlsx"

    #读成df
    df = pd.read_excel(df_file)
    
    # 只取前10个进行测试



    df = df.head(1000)

       # 根据tenure_df筛选df
    # 条件：tenure_df 的 "是否为24年全年在任" = 1，并且 company_code 和 person_name/person 匹配
    df['company_code'] = df['company_code'].astype(str)
    tenure_df['company_code'] = tenure_df['company_code'].astype(str)

    active_in_2024_df = tenure_df[tenure_df['是否为24年全年在任'] == 1]
    df = pd.merge(
        df,
        active_in_2024_df[['company_code', 'person']],
        left_on=['company_code', 'person_name'],
        right_on=['company_code', 'person'],
        how='inner'
    )
    df.drop(columns=['person'], inplace=True)


    # 遍历DataFrame中的每一行，获取公司代码和独立董事姓名
    total_tasks = len(df)
    task_done = 0
    
    # Add a new column to df to store the actual report name for DB lookup
    df['db_report_name'] = pd.NA
    
    processed_files_info = [] # To store info for rows where files were found

    with ProcessPoolExecutor(max_workers=16) as executor:

        futures = []
        for index, row in df.iterrows():
            actual_company_code_val = row['company_code']
            actual_person_name_val = row['person_name']

            found_file = None
            actual_filename_no_ext = None
            for filename in os.listdir(root_folder):
                 if filename.startswith(f"{actual_company_code_val}_") and actual_person_name_val in filename and filename.endswith('.json'):
                     found_file = os.path.join(root_folder, filename)
                     actual_filename_no_ext = filename[:-5] # Remove .json
                     break

            if found_file:
                # Store the actual report name for later use
                # df.loc[index, 'db_report_name'] = actual_filename_no_ext # This might not work well with iterrows, better to update later or collect and merge
                processed_files_info.append({'index': index, 'db_report_name': actual_filename_no_ext})
                # futures.append(executor.submit(report_to_db, found_file, actual_person_name_val, actual_company_code_val))

    # Update DataFrame with the db_report_name for processed files
    for info in processed_files_info:
        df.loc[info['index'], 'db_report_name'] = info['db_report_name']


    # # ------------------------------------------------------------------------------------------------------------------------------------


    # # 报告向量化
    pre_road_map_embedding_convert()


    #跑group_searching_within_report
    # 清空整个mission库
    db.query(Missions).filter(Missions.done == 1).delete()
    db.commit()
    all_indicators = db.query(Indicators).all()
    all_indicator_code_list = [indicator.code for indicator in all_indicators]

    total_target_reports = []
    # df = pd.read_excel(df_file) # REMOVE THIS LINE: df is already loaded and modified with db_report_name
    total_tasks = len(df)
    task_done = 0

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = []
        for index, row in df.iterrows():
            # Use the stored db_report_name
            actual_report_name_for_task = row.get('db_report_name')
            if pd.isna(actual_report_name_for_task):
                # print(f"Skipping group searching for row {index} due to missing report name (original file likely not found).")
                continue

            report_task_dict = {'company_code':row['company_code'],'year':2024,'report_name': actual_report_name_for_task,'indicators':all_indicator_code_list,'person':row['person_name']}
            futures.append(executor.submit(group_searching_within_report_v2,report_task_dict))
        for future in as_completed(futures):
            result,message  = future.result()
            if result:
                task_done += 1
                print(f"分配任务完成{task_done}/{total_tasks}")
            else:
                print(message)




    #------------------------------------------------------------------------------------------------------------------------------------


    #跑dynamic数据收集

    db, engine = new_create_db_session()

    task_done = 0
    # #跑execute_level == 1的
    print("Starting execute_level == 1 tasks processing...")
    with ProcessPoolExecutor(max_workers=30) as executor:
        futures = []
        
        # Pre-fetch all L1 missions to avoid querying in a loop
        valid_pairs_l1 = df.dropna(subset=['db_report_name', 'person_name'])
        report_names_for_query_l1 = valid_pairs_l1['db_report_name'].unique().tolist()

        missions_by_report_person_l1 = {}
        if report_names_for_query_l1:
            all_l1_missions = db.query(Missions).filter(
                Missions.report_name.in_(report_names_for_query_l1),
                Missions.execute_level.is_(None),
                Missions.done.is_(None)
            ).all()

            # Group missions by (report_name, person) for efficient lookup
            for m in all_l1_missions:
                key = (m.report_name, m.person)
                if key not in missions_by_report_person_l1:
                    missions_by_report_person_l1[key] = []
                missions_by_report_person_l1[key].append(m)
        
        # Iterate through the dataframe to find matching missions and submit them
        missions_to_mark_failed_l1 = []
        for index, row in valid_pairs_l1.iterrows():
            report_name = row['db_report_name']
            person_name = row['person_name']
            
            missions_to_run = missions_by_report_person_l1.get((report_name, person_name), [])
            
            for mission in missions_to_run:
                try:
                    indicator_info_dict = json.loads(mission.mission_json)
                    futures.append(executor.submit(single_data_collection, indicator_info_dict, force_execute=0, dynamic_question_word=person_name, mission_id=mission.id))
                except json.JSONDecodeError as e:
                    print(f"Error decoding mission_json for mission {mission.id} (L1): {e}")
                    mission.done = 2 # Mark as failed
                    missions_to_mark_failed_l1.append(mission)
        
        if missions_to_mark_failed_l1:
            db.commit()


        print(f"Collected {len(futures)} execute_level == 1 tasks. Processing...")
        failed_reports_l1 = []
        for future in as_completed(futures):
            result,message = future.result()
            if result:
                task_done += 1
                print(f"任务完成{task_done}")
            else:
                failed_reports_l1.append(FailReports(message = message,step = 3))

        if failed_reports_l1:
            db.add_all(failed_reports_l1)
            db.commit()

    # #跑execute_level == 2的
    print("Starting execute_level == 2 tasks processing...")
    db.close()
    db, engine = new_create_db_session()


    with ProcessPoolExecutor(max_workers=20) as executor: # Ensure this executor is for L2 tasks
        futures = [] # Initialize futures list for L2 tasks
        
        valid_report_person_pairs = df.dropna(subset=['db_report_name', 'person_name'])
        report_names_for_query = valid_report_person_pairs['db_report_name'].unique().tolist()

        if report_names_for_query:
            all_l2_missions = db.query(Missions).filter(
                Missions.report_name.in_(report_names_for_query),
                Missions.execute_level == 2,
                Missions.done.is_(None)
            ).all()

            all_pre_indicator_codes = {m.pre_condition_indicator for m in all_l2_missions if m.pre_condition_indicator}
            
            all_key_indicators = {}
            if all_pre_indicator_codes:
                indicator_results = db.query(IndicatorsResults).filter(
                    IndicatorsResults.code.in_(all_pre_indicator_codes),
                    IndicatorsResults.company_code.in_(valid_report_person_pairs['company_code'].unique().tolist())
                ).all()
                for res in indicator_results:
                    all_key_indicators[(res.company_code, res.code, res.person)] = res

            missions_by_report_person = {}
            for m in all_l2_missions:
                key = (m.report_name, m.person)
                if key not in missions_by_report_person:
                    missions_by_report_person[key] = []
                missions_by_report_person[key].append(m)

            for index, row in valid_report_person_pairs.iterrows():
                company_code = row['company_code']
                dynamic_question_word = row['person_name']
                report_name_base_for_dynamic = row['db_report_name']

                missions_for_l2_check = missions_by_report_person.get((report_name_base_for_dynamic, dynamic_question_word), [])

                if not missions_for_l2_check:
                    continue

                missions_by_pre_indicator = {}
                for m in missions_for_l2_check:
                    key = m.pre_condition_indicator if m.pre_condition_indicator is not None else "_NO_PRECONDITION_"
                    if key not in missions_by_pre_indicator:
                        missions_by_pre_indicator[key] = []
                    missions_by_pre_indicator[key].append(m)

                for pre_indicator_code, related_missions in missions_by_pre_indicator.items():
                    if pre_indicator_code == "_NO_PRECONDITION_" or not pre_indicator_code:
                        print(f"Warning: L2 Missions found with no pre_condition_indicator for report {report_name_base_for_dynamic}, person {dynamic_question_word}. Marking them done.")
                        for m_to_mark_done in related_missions:
                            m_to_mark_done.done = 1
                        continue

                    key_indicator_obj = all_key_indicators.get((str(company_code), pre_indicator_code, dynamic_question_word))
                    
                    precondition_met_for_section1 = False
                    if key_indicator_obj and \
                       key_indicator_obj.value is not None and \
                       str(key_indicator_obj.value).strip() != "" and \
                       str(key_indicator_obj.value).strip().lower() != "不涉及":
                            precondition_met_for_section1 = True

                    for mission_to_process in related_missions:
                        if precondition_met_for_section1:
                            try:
                                indicator_info_dict = json.loads(mission_to_process.mission_json)
                                futures.append(
                                    executor.submit(single_data_collection, indicator_info_dict, force_execute=0,
                                                   dynamic_question_word=dynamic_question_word, mission_id=mission_to_process.id)
                                )
                            except json.JSONDecodeError as e:
                                print(f"Error decoding mission_json for mission {mission_to_process.id}: {e}")
                                mission_to_process.done = 2 # Mark as done if JSON is invalid
                        else:
                            mission_to_process.done = 1
                    
                    db.commit()

        print(f"Collected {len(futures)} execute_level == 2 tasks. Processing...")
        failed_reports_l2 = []
        for future in as_completed(futures):
            result, message = future.result()
            if result:
                task_done += 1
                print(f"任务完成{task_done}")
            else:
                failed_reports_l2.append(FailReports(message = message,step = 3))

        if failed_reports_l2:
            db.add_all(failed_reports_l2)
            db.commit()

    end_time = datetime.now()
    print(f"总共用时：{end_time-start_time}")
    # print(f"第二步总共用时：{end_time-step1_end_time}")

    # ------------------------------------------------------------------------------------------------------------------------------------




    # Read all data from IndicatorsResults table
    indicators_results = db.query(IndicatorsResults).all()
    
    # Get column names from the table
    column_names = [column.name for column in IndicatorsResults.__table__.columns]
    
    # Convert to pandas DataFrame using table's column names
    results_df = pd.DataFrame([{
        column: getattr(result, column) for column in column_names
    } for result in indicators_results])

    # Merge with jianchen_df to get company short names
    if not results_df.empty and not jianchen_df.empty:
        # Convert both merge columns to string type to avoid type mismatch
        results_df['company_code'] = results_df['company_code'].astype(str)
        jianchen_df['code'] = jianchen_df['code'].astype(str)
        
        results_df = results_df.merge(
            jianchen_df[['code', 'name', '公司名称']],
            left_on='company_code',
            right_on='code',
            how='left'
        )
        # Drop the redundant code column after merge if it exists
        if 'code' in results_df.columns:
            results_df.drop('code', axis=1, inplace=True)
    else:
        print("Warning: results_df或jianchen_df为空，跳过公司信息合并")



    # Save to Excel file with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"indicators_results_{DB_NAME}_{timestamp}.xlsx"
    results_df.to_excel(output_filename, index=False)
    print(f"Saved indicators results to {output_filename}")