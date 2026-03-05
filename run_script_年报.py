import os
import logging
logging.basicConfig(level=logging.WARNING)
# --- 开始：脚本早期设置环境变量 DB_NAME ---
# 确保此脚本运行时，所有模块（特别是自定义的如下面的logic_folder中的模块）
# 如果在导入时通过 os.getenv('DB_NAME') 获取数据库名，则能获取到正确的目标数据库名。
TARGET_DB_FOR_THIS_SCRIPT = "zsx_年报指标跑批_0617"
os.environ['DB_NAME'] = TARGET_DB_FOR_THIS_SCRIPT
# --- 结束：脚本早期设置环境变量 DB_NAME ---

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from logic_folder.数据库表格 import Base, Reports, Company, Vector, EntityEvd, Indicators, Sentences, FailReports, \
    Missions, IndicatorsResults
from logic_folder.检索包 import group_searching_within_report, group_searching_within_report_v2
from logic_folder.表格处理包 import pre_road_map_embedding_convert, single_embedding_groups_retrieve
from operation_folder.指标批量收集 import single_data_collection
from operation_folder.指标表录入 import indicators_sheet_to_db
from operation_folder.文件录入 import report_to_db
from sqlalchemy import create_engine, and_, text
from sqlalchemy.orm import sessionmaker
import pymysql

from 模块工具.报告json批量获取加命名存储工具 import get_report_json


pymysql.install_as_MySQLdb()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME') # 此处 DB_NAME 会从已修改的 os.environ 中获取，应为 "zsx_年报"

# 配置数据库连接
def create_db_engine(echo=False):
    # 此函数将使用上面设置的全局 DB_NAME
    engine = create_engine(f'mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}', echo=echo,isolation_level="READ COMMITTED")
    return engine


def new_create_db_session():
    # 创建数据库引擎
    engine = create_db_engine(False)
    Session = sessionmaker(bind=engine)
    session = Session()

    return session, engine


##创建数据库和表格
def create_db_and_tables():
    # 此函数也将通过 create_db_engine 使用正确的 DB_NAME
    engine = create_db_engine(True)
    Base.metadata.create_all(engine)
    return True


if __name__ == "__main__":

    # 创建库表 (将使用已正确设置的 DB_NAME: "zsx_年报")
    create_db_and_tables()
    # ------------------------------------------------------------------------------------------------------------------------------------
    jianchen_df = pd.read_excel('/Users/linxuanxuan/Desktop/zsx_独立董事指标跑批/uploads/指标表/Copy of 年报和独立董事报告use.xlsx')
    indicator_file = '指标表/中上协独董评价_年报指标0617.xlsx'
    year = 2024
    indicators_sheet_to_db(indicator_file, year)
    start_time = datetime.now()
    # 报告入库
    # new_create_db_session 内部创建session时，应使用新的DB_NAME
    db, engine = new_create_db_session()
    # total_files = [] # No longer need this list
    root_folder = "测试"

    # df_file = 'uploads/指标表/independent_directors_info.xlsx'
    df_file = 'uploads/指标表/test_indicator_sample0602_50独董.xlsx'

    #读成df
    df = pd.read_excel(df_file)
    # df = df.head(15)
    # 遍历DataFrame中的每一行，获取公司代码和独立董事姓名
    total_tasks = len(df)
    task_done = 0
    df['db_report_name'] = pd.NA
    processed_files_info = [] # To store info for rows where files were found

    with ProcessPoolExecutor(max_workers=12) as executor:

        futures = []
        for index, row in df.iterrows():
            actual_company_code_val = row['company_code']
            actual_person_name_val = row['person_name']

            found_file = None
            actual_filename_no_ext = None

            for filename in os.listdir(root_folder):
                 if filename.startswith(f"{actual_company_code_val}_") and filename.endswith('.json'):
                     found_file = os.path.join(root_folder, filename)
                     actual_filename_no_ext = filename[:-5] # Remove .json
                     break


            if found_file:
                processed_files_info.append({'index': index, 'db_report_name': actual_filename_no_ext})
                futures.append(executor.submit(report_to_db, found_file,actual_person_name_val, actual_company_code_val))

        for future in as_completed(futures):
           result,message = future.result()
           if result:
               task_done += 1
               print(f"入库任务完成{task_done}/{total_tasks}")
           else:
               new_fail_report = FailReports(message = message,step = 1)
               db.add(new_fail_report)
               db.commit()

    # Update DataFrame with the db_report_name for processed files
    for info in processed_files_info:
        df.loc[info['index'], 'db_report_name'] = info['db_report_name']


    # ------------------------------------------------------------------------------------------------------------------------------------

    # 报告向量化
    # pre_road_map_embedding_convert 内部创建session时，应使用新的DB_NAME
    pre_road_map_embedding_convert()
    #
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
                print(f"Skipping group searching for row {index} due to missing report name (original file likely not found).")
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


    db, engine = new_create_db_session() # 应使用新的DB_NAME

    task_done = 0
    # #跑execute_level == 1的
    with ProcessPoolExecutor(max_workers=18) as executor:

        futures = []
        for index, row in df.iterrows():
            # Use the stored db_report_name
            report_name = row.get('db_report_name')
            if pd.isna(report_name):
                # print(f"Skipping dynamic data collection for row {index} due to missing report name (original file likely not found).")
                continue
                
            dynamic_question_word = row['person_name']
            all_execute_level_1_missions = db.query(Missions).filter(and_(Missions.report_name == report_name,Missions.execute_level.is_(None),Missions.done.is_(None),Missions.person == dynamic_question_word)).all()
            for mission in all_execute_level_1_missions:
                indicator_info_dict = json.loads(mission.mission_json)
                futures.append(executor.submit(single_data_collection,indicator_info_dict,force_execute=0,dynamic_question_word = dynamic_question_word,mission_id = mission.id))

        for future in as_completed(futures):
            result,message = future.result()
            if result:
                task_done += 1
                print(f"任务完成{task_done}")
            else:
                new_fail_report = FailReports(message = message,step = 3)
                db.add(new_fail_report)
                db.commit()


    end_time = datetime.now()
    print(f"总共用时：{end_time-start_time}")
    # print(f"第二步总共用时：{end_time-step1_end_time}")

    # ------------------------------------------------------------------------------------------------------------------------------------


    print("------------------------------------------------------------------------------------------------------------------------------------")
    print("正在从数据库导出indicators_result到Excel...")

    try:
        # Query all IndicatorsResult records
        # Assuming IndicatorsResult model is available in the scope
        all_results = db.query(IndicatorsResults).all()

        if not all_results:
            print("数据库中indicators_result表没有数据。跳过导出。")
        else:
            # Convert query results to a list of dictionaries, including all fields
            results_list = []

            # Use SQLAlchemy inspect to get all mapped columns
            # Assuming 'IndicatorsResult' is a SQLAlchemy declarative model class
            from sqlalchemy import inspect # Assuming sqlalchemy is available for import

            mapper = inspect(IndicatorsResults)
            # Get all column names mapped by SQLAlchemy
            column_names = [c.key for c in mapper.columns]

            for res in all_results:
                row_dict = {}
                for col_name in column_names:
                    # Access the value for each column from the model instance
                    # getattr is a safe way to access attributes dynamically
                    row_dict[col_name] = getattr(res, col_name)
                results_list.append(row_dict)

            # Create DataFrame
            df_results = pd.DataFrame(results_list)
            # Merge with jianchen_df to get company short names
            if not df_results.empty and not jianchen_df.empty:
                # Convert both merge columns to string type to avoid type mismatch
                df_results['company_code'] = df_results['company_code'].astype(str)
                jianchen_df['code'] = jianchen_df['code'].astype(str)

                results_df = df_results.merge(
                    jianchen_df[['code', 'name', '公司名称']],
                    left_on='company_code',
                    right_on='code',
                    how='left'
                )
                # Drop the redundant entity_code column after merge
                results_df.drop('code_x', axis=1, inplace=True)

            # Define output path
            # Assuming os and pandas (pd) are imported elsewhere in the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"indicators_年报_{DB_NAME}_{timestamp}.xlsx"
            excel_path = os.path.join(script_dir, excel_filename)

            # Save to Excel file with current timestamp

            results_df.to_excel(excel_path, index=False)
            print(f"成功将indicators_result导出到: {excel_path}")

    except Exception as e:
        print(f"导出indicators_result到Excel时发生错误: {e}")



